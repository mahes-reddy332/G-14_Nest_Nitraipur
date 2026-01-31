"""
Data Ingestion Layer for the Neural Clinical Data Mesh
Handles loading, parsing, and standardizing clinical trial data files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class ClinicalDataIngester:
    """
    Ingests and standardizes clinical trial data from multiple Excel files
    """
    
    def __init__(self, base_path: Path):
        try:
            import openpyxl  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "openpyxl is required to read .xlsx source files. "
                "Install it with: pip install openpyxl"
            ) from exc
        self.base_path = Path(base_path)
        self.studies: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.file_patterns = {
            'cpid_metrics': ['CPID_EDC_Metrics', 'CPID'],
            'compiled_edrr': ['Compiled_EDRR', 'EDRR'],
            'sae_dashboard': ['eSAE', 'SAE Dashboard', 'SAE_Dashboard'],
            'meddra_coding': ['MedDRA', 'GlobalCodingReport_MedDRA'],
            'whodra_coding': ['WHODD', 'WHODRA', 'GlobalCodingReport_WHODD'],
            'inactivated_forms': ['Inactivated'],
            'missing_lab': ['Missing_Lab'],
            'missing_pages': ['Missing_Pages', 'Missing Pages'],
            'visit_tracker': ['Visit Projection', 'Visit_Projection']
        }
        
    def discover_studies(self) -> List[str]:
        """Discover all study folders in the base path"""
        study_folders = []
        for folder in self.base_path.iterdir():
            if folder.is_dir() and 'CPID' in folder.name:
                # Extract study number
                match = re.search(r'Study\s*(\d+)', folder.name, re.IGNORECASE)
                if match:
                    study_id = f"Study_{match.group(1)}"
                    study_folders.append((study_id, folder))
        
        # Sort by study number
        study_folders.sort(key=lambda x: int(re.search(r'\d+', x[0]).group()))
        return study_folders
    
    def _find_file(self, folder: Path, patterns: List[str]) -> Optional[Path]:
        """Find a file matching any of the given patterns"""
        for file in folder.glob("*.xlsx"):
            for pattern in patterns:
                if pattern.lower() in file.name.lower():
                    return file
        return None
    
    def _standardize_columns(self, df: pd.DataFrame, column_map: Dict[str, List[str]]) -> pd.DataFrame:
        """Standardize column names using a mapping dictionary
        
        Ensures no duplicate column names are created by only mapping one source 
        column to each standard name.
        """
        new_columns = {}
        assigned_standards = set()
        
        for col in df.columns:
            # Skip if we already mapped this column (though unlikely with unique check)
            if col in new_columns:
                continue
                
            # Try to match against standards
            col_clean = str(col).strip().replace('\n', ' ')
            matched = False
            
            for standard_name, possible_names in column_map.items():
                if standard_name in assigned_standards:
                    continue # Already found a source for this standard column
                    
                for possible in possible_names:
                    if not isinstance(possible, str): continue
                    
                    # specific check for exact match or word boundary
                    col_lower = col_clean.lower()
                    possible_lower = possible.lower()
                    
                    # Exact match
                    if col_lower == possible_lower:
                        new_columns[col] = standard_name
                        assigned_standards.add(standard_name)
                        matched = True
                        break
                        
                    # Containment with boundary check
                    if possible_lower in col_lower:
                        idx = col_lower.find(possible_lower)
                        if idx >= 0:
                            # Boundry checks
                            before = col_lower[idx-1] if idx > 0 else ' '
                            after = col_lower[idx+len(possible_lower)] if idx+len(possible_lower) < len(col_lower) else ' '
                            
                            is_valid_before = before in [' ', '\t', '-', '_']
                            is_valid_after = after in [' ', '\t', '-', '_']
                            
                            if idx == 0: is_valid_before = True
                            if idx + len(possible_lower) == len(col_lower): is_valid_after = True
                            
                            if is_valid_before and is_valid_after:
                                new_columns[col] = standard_name
                                assigned_standards.add(standard_name)
                                matched = True
                                break
                if matched: break
        
        # Rename matched columns
        df = df.rename(columns=new_columns)
        
        # Remove duplicates
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def _safe_numeric(self, value: Any) -> float:
        """Safely convert a value to numeric"""
        try:
            if value is None:
                return 0.0
            if isinstance(value, (pd.Series, list, dict)):
                return 0.0
            if pd.isna(value):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def load_cpid_metrics(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and parse CPID EDC Metrics file"""
        try:
            # Read raw first to detect header structure
            df_raw = pd.read_excel(file_path, sheet_name=0, header=None, nrows=10)
            
            # Find the row containing 'Subject ID' - that's our header
            header_row = 0
            for idx in range(min(10, len(df_raw))):
                row_values = df_raw.iloc[idx].tolist()
                row_str = ' '.join(str(v).lower() for v in row_values if pd.notna(v) and not isinstance(v, (list, dict)))
                if 'subject id' in row_str:
                    header_row = idx
                    break
            
            # Find the row with detailed column names (# Uncoded Terms, # Missing Visits, etc.)
            # This is usually 1-2 rows below the first header
            # CPID files have a multi-level header structure:
            #   Row 0: Primary headers (Project Name, Subject ID, Input files, CPMD, SSM)
            #   Row 1: Secondary headers (Missing Visits, Missing Page, # Uncoded Terms, etc.)
            #   Row 2: Tertiary headers (# Pages Entered, # Forms Verified, etc.)
            detail_rows = []
            for idx in range(header_row + 1, min(header_row + 4, len(df_raw))):
                row_values = df_raw.iloc[idx].tolist()
                row_str = ' '.join(str(v).lower() for v in row_values if pd.notna(v) and not isinstance(v, (list, dict)))
                # Check for various indicators of header rows
                has_detail_indicators = any(x in row_str for x in ['uncoded', 'missing page', 'open quer', 'verified', 'pages entered', '# dm quer', '# total', 'expected visit'])
                if has_detail_indicators:
                    detail_rows.append(idx)
            
            # Read with header row
            df = pd.read_excel(file_path, sheet_name=0, header=header_row)
            
            # If we found detail rows, map columns from those rows
            column_remap = {}
            for detail_row in detail_rows:
                detail_row_data = df_raw.iloc[detail_row]
                for col_idx, col_name in enumerate(df.columns):
                    if col_idx < len(detail_row_data):
                        detail_val = str(detail_row_data.iloc[col_idx]).strip()
                        if pd.notna(detail_row_data.iloc[col_idx]) and detail_val and detail_val != 'nan':
                            detail_lower = detail_val.lower()
                            # Only remap if not already remapped
                            if col_name in column_remap:
                                continue
                            if 'uncoded' in detail_lower and 'term' in detail_lower:
                                column_remap[col_name] = 'uncoded_terms'
                            elif '# coded terms' in detail_lower or 'coded terms' in detail_lower:
                                column_remap[col_name] = 'coded_terms'
                            elif 'missing page' in detail_lower:
                                column_remap[col_name] = 'missing_pages'
                            elif 'missing visit' in detail_lower:
                                column_remap[col_name] = 'missing_visits'
                            elif 'open' in detail_lower and ('quer' in detail_lower or 'issue' in detail_lower) and 'lnr' in detail_lower:
                                column_remap[col_name] = 'open_queries'
                            elif detail_lower.strip() == '#total queries' or detail_lower.strip() == '# total queries':
                                # Exact match for the total queries column (not CRFs with queries)
                                column_remap[col_name] = 'total_queries'
                            elif 'expected visit' in detail_lower:
                                column_remap[col_name] = 'expected_visits'
                            elif 'pages entered' in detail_lower:
                                column_remap[col_name] = 'pages_entered'
                            elif 'non-conformant' in detail_lower or 'nonconformant' in detail_lower:
                                column_remap[col_name] = 'non_conformant'
                            elif 'esae' in detail_lower and 'dm' in detail_lower:
                                column_remap[col_name] = 'esae_review'
                            elif 'pd' in detail_lower and 'confirmed' in detail_lower:
                                column_remap[col_name] = 'protocol_deviations'
                            elif 'forms verified' in detail_lower or '# forms verified' in detail_lower:
                                column_remap[col_name] = 'forms_verified'
                            elif 'crf' in detail_lower and 'frozen' in detail_lower:
                                column_remap[col_name] = 'frozen_pages'
                            elif 'crf' in detail_lower and 'locked' in detail_lower:
                                column_remap[col_name] = 'locked_pages'
                            elif 'edrr' in detail_lower or ('reconciliation' in detail_lower and 'open' in detail_lower):
                                column_remap[col_name] = 'reconciliation_issues'
                            elif '% clean' in detail_lower or 'clean %' in detail_lower or 'clean entered' in detail_lower:
                                column_remap[col_name] = 'verification_pct'
                            elif 'dm quer' in detail_lower:
                                column_remap[col_name] = 'dm_queries'
                            elif 'require verification' in detail_lower or 'sdv' in detail_lower:
                                column_remap[col_name] = 'crfs_require_verification'
            
            if column_remap:
                df = df.rename(columns=column_remap)
            
            # Skip rows that are part of the header
            # Use the last detail row found to calculate where data starts
            last_detail_row = detail_rows[-1] if detail_rows else header_row
            data_start_row = last_detail_row - header_row + 1
            # Find first actual data row (where Subject ID looks like a real value)
            for idx in range(min(len(df), 10)):  # Only check first 10 rows
                try:
                    if 'Subject ID' in df.columns:
                        subject_val = str(df.loc[idx, 'Subject ID'])
                    else:
                        subject_val = str(df.iloc[idx, 4]) if len(df.columns) > 4 else ''
                    
                    if subject_val.startswith('Subject ') and any(c.isdigit() for c in subject_val):
                        data_start_row = idx
                        break
                except:
                    continue
            
            df = df.iloc[data_start_row:].reset_index(drop=True)
            
            # Standardize column names
            column_map = {
                'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'SUBJECTID', 'Subject'],
                'site_id': ['Site ID', 'SiteID', 'Site_ID', 'SITEID', 'Site', 'Site Number'],
                'country': ['Country', 'COUNTRY'],
                'region': ['Region', 'REGION'],
                'status': ['Subject Status', 'Status', 'SUBJECT_STATUS', 'Subject_Status', 'Subject Status (Source: PRIMARY Form)'],
                'missing_visits': ['Missing Visits', '# Missing Visits', 'MissingVisits', 'Missing Visit'],
                'missing_pages': ['Missing Page', '# Missing Pages', 'MissingPages', 'Missing Pages'],
                'open_queries': ['# Open Queries', 'Open Queries', 'OpenQueries', 'Open Query'],
                'total_queries': ['# Total Queries', 'Total Queries', 'TotalQueries', '#Total Queries'],
                'uncoded_terms': ['# Uncoded Terms', 'Uncoded Terms', 'UncodedTerms', 'Uncoded'],
                'coded_terms': ['# Coded terms', 'Coded Terms', 'CodedTerms', 'Coded'],
                'verification_pct': ['Data Verification %', 'Verification %', 'VerificationPct', 'DV %', 'Data Verification', '% Clean Entered CRF'],
                'forms_verified': ['# Forms Verified', 'Forms Verified', 'FormsVerified'],
                'expected_visits': ['# Expected Visits', 'Expected Visits', 'ExpectedVisits', '# Expected Visits (Rave EDC : BO4)'],
                'pages_entered': ['# Pages Entered', 'Pages Entered', 'PagesEntered', 'Pages'],
                'non_conformant': ['# Pages with Non-Conformant data', 'Non-Conformant', 'NonConformant', 'Non Conformant'],
                'esae_review': ['# eSAE dashboard review for DM', 'eSAE Review', 'eSAEReview', 'eSAE'],
                'reconciliation_issues': ['# Reconciliation Issues', 'Recon Issues', 'ReconIssues', 'Reconciliation', '# Open Issues reported for 3rd party reconciliation in EDRR'],
                'broken_signatures': ['Broken Signatures', 'BrokenSignatures', 'Broken Sig'],
                'protocol_deviations': ['# PDs Confirmed', 'PDs Confirmed', 'ProtocolDeviations', 'PD', 'Protocol Deviation'],
                'crf_overdue': ['CRFs overdue for signs', 'CRF Overdue', 'CRFOverdue', 'Overdue', 'CRFs overdue for signs within 45 days of Data entry'],
                'crf_overdue_90': ['CRFs overdue for signs beyond 90 days', 'CRF Overdue 90', 'CRFOverdue90', 'CRFs overdue for signs beyond 90 days of Data entry'],
                'locked_pages': ['# Locked Pages', 'Locked Pages', 'LockedPages', 'Locked', '# CRFs Locked'],
                'frozen_pages': ['# Frozen Pages', 'Frozen Pages', 'FrozenPages', 'Frozen', '# CRFs Frozen'],
                'answered_queries': ['# Answered Queries', 'Answered Queries', 'AnsweredQueries', 'Answered']
            }
            
            df = self._standardize_columns(df, column_map)
            
            # Convert numeric columns
            numeric_cols = ['missing_visits', 'missing_pages', 'open_queries', 'total_queries',
                          'uncoded_terms', 'coded_terms', 'verification_pct', 'forms_verified',
                          'expected_visits', 'pages_entered', 'non_conformant', 'reconciliation_issues',
                          'protocol_deviations', 'crf_overdue', 'locked_pages', 'frozen_pages',
                          'answered_queries', 'esae_review', 'broken_signatures']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].apply(self._safe_numeric)
            
            # Remove duplicate columns before derived metrics
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Helper to safely get column sum
            def safe_col_sum(df, col):
                if col not in df.columns:
                    return 0
                try:
                    return df[col].sum()
                except:
                    return 0
            
            # Compute derived metrics if primary columns are empty
            # If open_queries is all zeros but total_queries has values, use that as proxy
            if 'open_queries' in df.columns and 'total_queries' in df.columns:
                open_sum = safe_col_sum(df, 'open_queries')
                total_sum = safe_col_sum(df, 'total_queries')
                if open_sum == 0 and total_sum > 0:
                    df['open_queries'] = df['total_queries'].copy()
            
            # If verification_pct is all zeros but we have forms_verified, compute it
            if 'forms_verified' in df.columns and 'pages_entered' in df.columns:
                verif_sum = safe_col_sum(df, 'verification_pct') if 'verification_pct' in df.columns else 0
                forms_sum = safe_col_sum(df, 'forms_verified')
                if verif_sum == 0 and forms_sum > 0:
                    def compute_verif_pct(row):
                        try:
                            pages = float(row.get('pages_entered', 0) if isinstance(row, dict) else row['pages_entered'])
                            forms = float(row.get('forms_verified', 0) if isinstance(row, dict) else row['forms_verified'])
                            return (forms / pages * 100) if pages > 0 else 100.0
                        except:
                            return 100.0
                    df['verification_pct'] = df.apply(compute_verif_pct, axis=1)
            
            # If missing_pages is all zeros but non_conformant has values, use that
            if 'missing_pages' in df.columns and 'non_conformant' in df.columns:
                missing_sum = safe_col_sum(df, 'missing_pages')
                nonconf_sum = safe_col_sum(df, 'non_conformant')
                if missing_sum == 0 and nonconf_sum > 0:
                    df['missing_pages'] = df['non_conformant'].copy()
            
            logger.info(f"Loaded CPID metrics: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CPID metrics from {file_path}: {e}")
            return None
    
    def load_visit_tracker(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and parse Visit Projection Tracker file"""
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
            column_map = {
                'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'Subject'],
                'site_id': ['Site ID', 'SiteID', 'Site_ID', 'Site', 'Site Number'],
                'visit_name': ['Visit Name', 'VisitName', 'Visit', 'Folder Name', 'Visit/Folder'],
                'projected_date': ['Projected Date', 'ProjectedDate', 'Projected_Date', 'Expected Date'],
                'days_outstanding': ['# Days Outstanding', 'Days Outstanding', 'DaysOutstanding', 'Days Out']
            }
            
            df = self._standardize_columns(df, column_map)
            
            # Convert days_outstanding to numeric
            if 'days_outstanding' in df.columns:
                df['days_outstanding'] = df['days_outstanding'].apply(self._safe_numeric)
            
            logger.info(f"Loaded Visit Tracker: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Visit Tracker from {file_path}: {e}")
            return None
    
    def load_sae_dashboard(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and parse SAE Dashboard file"""
        try:
            # Try to read multiple sheets
            excel_file = pd.ExcelFile(file_path)
            dfs = []
            
            for sheet in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                if len(df) > 0:
                    df['source_sheet'] = sheet
                    dfs.append(df)
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = pd.read_excel(file_path, sheet_name=0)
            
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            
            column_map = {
                'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'Patient ID', 'Subject'],
                'site_id': ['Site ID', 'SiteID', 'Site_ID', 'Site', 'Site Number'],
                'discrepancy_id': ['Discrepancy ID', 'DiscrepancyID', 'Discrepancy_ID', 'ID'],
                'review_status': ['Review Status', 'ReviewStatus', 'Review_Status', 'DM Review Status'],
                'action_status': ['Action Status', 'ActionStatus', 'Action_Status', 'Action'],
                'event_date': ['Event Date', 'EventDate', 'Event_Date', 'SAE Date'],
                'sae_type': ['SAE Type', 'SAEType', 'Type', 'Event Type'],
                'sae_term': ['SAE Term', 'Term', 'Preferred Term', 'Event']
            }
            
            df = self._standardize_columns(df, column_map)
            logger.info(f"Loaded SAE Dashboard: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading SAE Dashboard from {file_path}: {e}")
            return None
    
    def load_coding_report(self, file_path: Path, coding_type: str) -> Optional[pd.DataFrame]:
        """Load and parse GlobalCodingReport (MedDRA or WHODRA)"""
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
            # Skip header row if it exists (common in GlobalCodingReport files)
            if len(df) > 0 and str(df.iloc[0, 0]).lower() in ['meddra coding report', 'whodd coding report', 'whodra coding report']:
                # First row is a title, use second row or find header
                df = pd.read_excel(file_path, sheet_name=0, skiprows=1)
            
            column_map = {
                'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'Subject', 'Patient'],
                'site_id': ['Site ID', 'SiteID', 'Site_ID', 'Site'],
                'verbatim_term': ['Verbatim Term', 'VerbatimTerm', 'Verbatim', 'Term', 'Verbatim Text'],
                'coding_status': ['Coding Status', 'CodingStatus', 'Status', 'Code Status'],
                'coded_term': ['Coded Term', 'CodedTerm', 'LLT', 'PT', 'Preferred Term'],
                'context': ['Context', 'Form', 'Source', 'Source Form', 'Form OID'],
                'dictionary': ['Dictionary', 'Dict', 'Coding Dictionary'],
                'require_coding': ['Require Coding', 'RequireCoding', 'Needs Coding']
            }
            
            df = self._standardize_columns(df, column_map)
            df['coding_type'] = coding_type  # MedDRA or WHODRA
            
            # Filter for uncoded terms if coding_status exists
            # Status values like "Coded Term" mean already coded, we want "Not Coded", "Uncoded", "Pending"
            
            logger.info(f"Loaded {coding_type} Coding Report: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Coding Report from {file_path}: {e}")
            return None
    
    def load_missing_pages(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and parse Missing Pages Report"""
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            
            column_map = {
                'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'Subject'],
                'site_id': ['Site ID', 'SiteID', 'Site_ID', 'Site'],
                'form_name': ['Form Name', 'FormName', 'Form', 'Page Name', 'CRF'],
                'visit_name': ['Visit Name', 'VisitName', 'Visit', 'Folder'],
                'reason': ['Reason', 'Missing Reason', 'Status']
            }
            
            df = self._standardize_columns(df, column_map)
            logger.info(f"Loaded Missing Pages Report: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Missing Pages Report from {file_path}: {e}")
            return None
    
    def load_compiled_edrr(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and parse Compiled EDRR file"""
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
            column_map = {
                'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'Subject'],
                'site_id': ['Site ID', 'SiteID', 'Site_ID', 'Site'],
                'total_issues': ['Total Open issue Count', 'Total Issues', 'Open Issues', 'Issue Count'],
                'critical_issues': ['Critical Issues', 'Critical', 'High Priority'],
                'issue_type': ['Issue Type', 'Type', 'Category']
            }
            
            df = self._standardize_columns(df, column_map)
            
            if 'total_issues' in df.columns:
                df['total_issues'] = df['total_issues'].apply(self._safe_numeric)
            
            logger.info(f"Loaded Compiled EDRR: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Compiled EDRR from {file_path}: {e}")
            return None
    
    def load_inactivated_forms(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and parse Inactivated Forms and Folders Report"""
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
            column_map = {
                'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'Subject'],
                'site_id': ['Site ID', 'SiteID', 'Site_ID', 'Site'],
                'form_name': ['Form Name', 'FormName', 'Form', 'Page Name'],
                'folder_name': ['Folder Name', 'FolderName', 'Folder', 'Visit'],
                'audit_action': ['Audit Action', 'AuditAction', 'Action', 'Reason'],
                'inactivation_date': ['Inactivation Date', 'Date', 'Action Date'],
                'user': ['User', 'Modified By', 'User Name']
            }
            
            df = self._standardize_columns(df, column_map)
            logger.info(f"Loaded Inactivated Forms: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Inactivated Forms from {file_path}: {e}")
            return None
    
    def load_missing_lab(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and parse Missing Lab Name and Ranges file"""
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
            column_map = {
                'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'Subject'],
                'site_id': ['Site ID', 'SiteID', 'Site_ID', 'Site'],
                'lab_name': ['Lab Name', 'LabName', 'Laboratory'],
                'test_name': ['Test Name', 'TestName', 'Test', 'Parameter'],
                'action_for_site': ['Action for Site', 'Action', 'Required Action'],
                'status': ['Status', 'Current Status']
            }
            
            df = self._standardize_columns(df, column_map)
            logger.info(f"Loaded Missing Lab Report: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Missing Lab Report from {file_path}: {e}")
            return None
    
    def ingest_study(self, study_id: str, folder: Path) -> Dict[str, pd.DataFrame]:
        """Ingest all data files for a single study"""
        study_data = {}
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Ingesting {study_id} from {folder.name}")
        logger.info(f"{'='*50}")
        
        # Load CPID Metrics
        file_path = self._find_file(folder, self.file_patterns['cpid_metrics'])
        if file_path:
            study_data['cpid_metrics'] = self.load_cpid_metrics(file_path)
        
        # Load Visit Tracker
        file_path = self._find_file(folder, self.file_patterns['visit_tracker'])
        if file_path:
            study_data['visit_tracker'] = self.load_visit_tracker(file_path)
        
        # Load SAE Dashboard
        file_path = self._find_file(folder, self.file_patterns['sae_dashboard'])
        if file_path:
            study_data['sae_dashboard'] = self.load_sae_dashboard(file_path)
        
        # Load MedDRA Coding
        file_path = self._find_file(folder, self.file_patterns['meddra_coding'])
        if file_path:
            study_data['meddra_coding'] = self.load_coding_report(file_path, 'MedDRA')
        
        # Load WHODRA Coding
        file_path = self._find_file(folder, self.file_patterns['whodra_coding'])
        if file_path:
            study_data['whodra_coding'] = self.load_coding_report(file_path, 'WHODRA')
        
        # Load Missing Pages
        file_path = self._find_file(folder, self.file_patterns['missing_pages'])
        if file_path:
            study_data['missing_pages'] = self.load_missing_pages(file_path)
        
        # Load Compiled EDRR
        file_path = self._find_file(folder, self.file_patterns['compiled_edrr'])
        if file_path:
            study_data['compiled_edrr'] = self.load_compiled_edrr(file_path)
        
        # Load Inactivated Forms
        file_path = self._find_file(folder, self.file_patterns['inactivated_forms'])
        if file_path:
            study_data['inactivated_forms'] = self.load_inactivated_forms(file_path)
        
        # Load Missing Lab
        file_path = self._find_file(folder, self.file_patterns['missing_lab'])
        if file_path:
            study_data['missing_lab'] = self.load_missing_lab(file_path)
        
        self.studies[study_id] = study_data
        
        # Print summary
        loaded_files = [k for k, v in study_data.items() if v is not None]
        logger.info(f"Successfully loaded {len(loaded_files)} files for {study_id}")
        
        return study_data
    
    def ingest_all_studies(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Ingest data from all discovered studies"""
        study_folders = self.discover_studies()
        
        logger.info(f"\nDiscovered {len(study_folders)} studies")
        
        for study_id, folder in study_folders:
            self.ingest_study(study_id, folder)
        
        return self.studies
    
    def get_study_summary(self) -> pd.DataFrame:
        """Generate a summary of all ingested studies"""
        summaries = []
        
        for study_id, data in self.studies.items():
            summary = {'study_id': study_id}
            
            if 'cpid_metrics' in data and data['cpid_metrics'] is not None:
                cpid = data['cpid_metrics']
                summary['total_subjects'] = len(cpid)
                summary['total_sites'] = cpid['site_id'].nunique() if 'site_id' in cpid.columns else 0
                
                if 'open_queries' in cpid.columns:
                    summary['total_open_queries'] = int(cpid['open_queries'].sum())
                if 'missing_visits' in cpid.columns:
                    summary['total_missing_visits'] = int(cpid['missing_visits'].sum())
                if 'uncoded_terms' in cpid.columns:
                    summary['total_uncoded_terms'] = int(cpid['uncoded_terms'].sum())
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def load_all_clinical_data(base_path: str) -> Tuple[ClinicalDataIngester, Dict]:
    """
    Convenience function to load all clinical data
    
    Args:
        base_path: Path to the QC Anonymized Study Files folder
        
    Returns:
        Tuple of (ingester instance, studies dictionary)
    """
    ingester = ClinicalDataIngester(Path(base_path))
    studies = ingester.ingest_all_studies()
    return ingester, studies


if __name__ == "__main__":
    # Example usage
    BASE_PATH = r"d:\6932c39b908b6_detailed_problem_statements_and_datasets\Data for problem Statement 1\NEST 2.0 Data files_Anonymized\Main_Project Track1\QC Anonymized Study Files"
    
    ingester, studies = load_all_clinical_data(BASE_PATH)
    summary = ingester.get_study_summary()
    print("\n" + "="*60)
    print("STUDY SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
