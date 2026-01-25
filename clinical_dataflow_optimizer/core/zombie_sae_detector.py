"""
Zombie SAE Detection Module
===========================

Implements Scenario A: The "Zombie" SAE detection and resolution workflow.

Problem Definition:
- An investigator reports an SAE to pharmacovigilance (entered in Safety DB like Argus)
- Site coordinator forgets to enter corresponding "Adverse Event" form in EDC
- This creates a reconciliation gap that delays database lock

Agentic Solution Pipeline:
1. Data Check: Read SAE Dashboard, identify Discrepancy IDs with Action Status = "Pending"
2. Cross-Reference: Extract Patient ID and Site, query CPID_EDC_Metrics for that Subject
3. Logic Gate: Check "# eSAE dashboard review for DM" column - if 0 or less than SAE count, infer missing entry
4. Verification: Cross-check Global_Missing_Pages_Report for missing "Adverse Event" form
5. Action: Auto-draft query to site requesting data entry or clarification
6. Update: Update Compiled_EDRR "Total Open issue Count" to reflect new risk
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import DEFAULT_AGENT_CONFIG

logger = logging.getLogger(__name__)


class ZombieSAEStatus(Enum):
    """Status of Zombie SAE detection"""
    CONFIRMED = "confirmed"           # Definite zombie SAE - missing EDC entry
    SUSPECTED = "suspected"           # Likely zombie SAE - needs verification
    RESOLVED = "resolved"             # Zombie SAE has been addressed
    FALSE_POSITIVE = "false_positive" # Not actually a zombie SAE
    PENDING_REVIEW = "pending_review" # Under manual review


class ReconciliationOutcome(Enum):
    """Outcome of reconciliation check"""
    CONCORDANT = "concordant"         # Safety DB and EDC are in sync
    DISCREPANT = "discrepant"         # Mismatch found
    EDC_MISSING = "edc_missing"       # Entry in Safety DB but not in EDC
    SAFETY_MISSING = "safety_missing" # Entry in EDC but not in Safety DB
    UNKNOWN = "unknown"               # Cannot determine


@dataclass
class ZombieSAECase:
    """Represents a detected Zombie SAE case"""
    case_id: str
    discrepancy_id: str
    study_id: str
    site_id: str
    patient_id: str
    country: str
    
    # SAE Dashboard Data
    sae_form_name: str
    sae_created_timestamp: datetime
    sae_review_status: str
    sae_action_status: str
    
    # CPID Cross-Reference Data
    cpid_esae_dm_count: int              # Value from "# eSAE dashboard review for DM"
    cpid_esae_safety_count: int          # Value from "# eSAE dashboard review for safety"
    cpid_recon_issues: int               # Value from "# Open Issues reported for 3rd party reconciliation"
    
    # Missing Pages Verification
    missing_ae_forms: List[str] = field(default_factory=list)
    days_ae_missing: int = 0
    
    # Detection Results
    status: ZombieSAEStatus = ZombieSAEStatus.SUSPECTED
    reconciliation_outcome: ReconciliationOutcome = ReconciliationOutcome.UNKNOWN
    confidence_score: float = 0.0
    
    # Auto-Generated Query
    auto_query_text: str = ""
    query_sent: bool = False
    query_sent_timestamp: Optional[datetime] = None
    
    # EDRR Update
    edrr_updated: bool = False
    edrr_new_issue_count: int = 0
    
    # Timestamps
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting"""
        return {
            'case_id': self.case_id,
            'discrepancy_id': self.discrepancy_id,
            'study_id': self.study_id,
            'site_id': self.site_id,
            'patient_id': self.patient_id,
            'country': self.country,
            'sae_form_name': self.sae_form_name,
            'sae_created_timestamp': self.sae_created_timestamp.isoformat() if self.sae_created_timestamp else None,
            'sae_review_status': self.sae_review_status,
            'sae_action_status': self.sae_action_status,
            'cpid_esae_dm_count': self.cpid_esae_dm_count,
            'cpid_esae_safety_count': self.cpid_esae_safety_count,
            'cpid_recon_issues': self.cpid_recon_issues,
            'missing_ae_forms': self.missing_ae_forms,
            'days_ae_missing': self.days_ae_missing,
            'status': self.status.value,
            'reconciliation_outcome': self.reconciliation_outcome.value,
            'confidence_score': round(self.confidence_score, 2),
            'auto_query_text': self.auto_query_text,
            'query_sent': self.query_sent,
            'edrr_updated': self.edrr_updated,
            'edrr_new_issue_count': self.edrr_new_issue_count,
            'detected_at': self.detected_at.isoformat()
        }


@dataclass
class ZombieSAEConfig:
    """Configuration for Zombie SAE detection"""
    # Detection thresholds
    action_status_pending_values: List[str] = field(default_factory=lambda: [
        'Pending', 'pending', 'PENDING', 'Open', 'open', 'OPEN', 
        'Pending for Review', 'Action Required', 'Awaiting Response'
    ])
    
    review_status_pending_values: List[str] = field(default_factory=lambda: [
        'Pending for Review', 'pending for review', 'PENDING FOR REVIEW',
        'Under Review', 'Awaiting Review', 'Not Reviewed'
    ])
    
    # Days threshold before escalation
    days_pending_warning: int = 3
    days_pending_critical: int = 7
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.90
    medium_confidence_threshold: float = 0.75
    
    # AE form patterns to search in Missing Pages
    ae_form_patterns: List[str] = field(default_factory=lambda: [
        'Adverse Event', 'AE', 'adverse event', 'ae form', 
        'Serious Adverse Event', 'SAE', 'Safety Event',
        'adverse_event', 'ae_form', 'sae_form'
    ])
    
    # Auto-query settings
    enable_auto_query: bool = True
    query_requires_approval: bool = False  # Per spec: auto-executable
    
    # EDRR update settings
    enable_edrr_update: bool = True


# Default configuration instance
DEFAULT_ZOMBIE_SAE_CONFIG = ZombieSAEConfig()


class ZombieSAEDetector:
    """
    Detects and resolves "Zombie SAE" cases where SAE exists in Safety Database
    but corresponding AE form is missing in EDC.
    
    This is the core detection engine used by the Reconciliation Agent (Rex).
    
    Pipeline:
    1. Scan SAE Dashboard for pending discrepancies
    2. Cross-reference with CPID_EDC_Metrics
    3. Verify against Global_Missing_Pages_Report
    4. Generate auto-queries
    5. Update EDRR metrics
    """
    
    # Query templates
    ZOMBIE_SAE_QUERY_TEMPLATE = (
        "URGENT: Safety Reconciliation Required\n\n"
        "Safety database indicates an SAE recorded on {sae_date} for Subject {patient_id} at Site {site_id}.\n"
        "However, the corresponding Adverse Event form was not found in EDC.\n\n"
        "Discrepancy Details:\n"
        "- Discrepancy ID: {discrepancy_id}\n"
        "- SAE Form: {form_name}\n"
        "- SAE Dashboard shows pending review\n"
        "- CPID eSAE DM Review count: {esae_dm_count}\n"
        "- Missing AE Form(s): {missing_forms}\n\n"
        "Action Required: Please enter the AE data in EDC or provide clarification.\n"
        "Reference: ICH E6 R2 Section 5.18.4 - Safety Reporting Compliance"
    )
    
    RECON_ALERT_TEMPLATE = (
        "Reconciliation Alert: Subject {patient_id}\n"
        "Safety DB has {sae_count} SAE record(s) but EDC shows {edc_count} review items.\n"
        "Gap of {gap} entries detected. Please investigate and reconcile."
    )
    
    def __init__(self, config: ZombieSAEConfig = None):
        self.config = config or DEFAULT_ZOMBIE_SAE_CONFIG
        self.detected_cases: List[ZombieSAECase] = []
        self._cpid_cache: Dict[str, Dict] = {}
        self._missing_pages_cache: Dict[str, List[Dict]] = {}
        self._edrr_cache: Dict[str, Dict] = {}
    
    def detect(
        self,
        sae_dashboard: pd.DataFrame,
        cpid_data: pd.DataFrame,
        missing_pages: Optional[pd.DataFrame] = None,
        edrr_data: Optional[pd.DataFrame] = None,
        study_id: str = ""
    ) -> List[ZombieSAECase]:
        """
        Main detection pipeline for Zombie SAEs
        
        Steps:
        1. Data Check: Scan SAE Dashboard for Action Status = "Pending"
        2. Cross-Reference: For each pending, query CPID for Subject ID
        3. Logic Gate: Check eSAE dashboard review for DM count
        4. Verification: Cross-check Missing Pages for AE form
        5. Generate cases with auto-queries
        
        Args:
            sae_dashboard: SAE Dashboard data (from eSAE Dashboard_Standard DM_Safety Report)
            cpid_data: CPID EDC Metrics data
            missing_pages: Global Missing Pages Report data
            edrr_data: Compiled EDRR data for issue count updates
            study_id: Study identifier
            
        Returns:
            List of detected ZombieSAECase objects
        """
        self.detected_cases = []
        
        # Build caches for efficient cross-referencing
        self._build_cpid_cache(cpid_data)
        self._build_missing_pages_cache(missing_pages)
        self._build_edrr_cache(edrr_data)
        
        # Step 1: Scan SAE Dashboard for pending discrepancies
        pending_saes = self._scan_sae_dashboard(sae_dashboard)
        
        logger.info(f"[ZombieSAE] Found {len(pending_saes)} pending SAE discrepancies")
        
        # Step 2-5: Process each pending SAE
        for pending_sae in pending_saes:
            case = self._process_pending_sae(pending_sae, study_id)
            if case:
                self.detected_cases.append(case)
        
        # Log summary
        confirmed = sum(1 for c in self.detected_cases if c.status == ZombieSAEStatus.CONFIRMED)
        suspected = sum(1 for c in self.detected_cases if c.status == ZombieSAEStatus.SUSPECTED)
        logger.info(f"[ZombieSAE] Detection complete: {confirmed} confirmed, {suspected} suspected Zombie SAEs")
        
        return self.detected_cases
    
    def _scan_sae_dashboard(self, sae_dashboard: pd.DataFrame) -> List[Dict]:
        """
        Step 1: Data Check
        Scan SAE Dashboard for discrepancies with Action Status = "Pending"
        or Review Status = "Pending for Review"
        """
        if sae_dashboard is None or sae_dashboard.empty:
            return []
        
        # Handle duplicate columns
        if sae_dashboard.columns.duplicated().any():
            sae_dashboard = sae_dashboard.loc[:, ~sae_dashboard.columns.duplicated()]
        
        pending_records = []
        
        # Find column names (handle variations)
        col_mapping = self._find_sae_columns(sae_dashboard)
        
        if not col_mapping.get('patient_id') or not col_mapping.get('site'):
            logger.warning("[ZombieSAE] SAE Dashboard missing required columns (Patient ID, Site)")
            return []
        
        for idx, row in sae_dashboard.iterrows():
            # Check Action Status for "Pending"
            action_status = str(row.get(col_mapping.get('action_status', ''), '')).strip()
            review_status = str(row.get(col_mapping.get('review_status', ''), '')).strip()
            
            is_action_pending = any(p in action_status for p in self.config.action_status_pending_values)
            is_review_pending = any(p in review_status for p in self.config.review_status_pending_values)
            
            # Trigger if either is pending
            if is_action_pending or is_review_pending:
                record = {
                    'discrepancy_id': str(row.get(col_mapping.get('discrepancy_id', ''), idx)),
                    'study_id': str(row.get(col_mapping.get('study_id', ''), '')),
                    'patient_id': str(row.get(col_mapping.get('patient_id', ''), '')),
                    'site_id': str(row.get(col_mapping.get('site', ''), '')),
                    'country': str(row.get(col_mapping.get('country', ''), '')),
                    'form_name': str(row.get(col_mapping.get('form_name', ''), '')),
                    'created_timestamp': row.get(col_mapping.get('created_timestamp', ''), None),
                    'review_status': review_status,
                    'action_status': action_status,
                    'is_action_pending': is_action_pending,
                    'is_review_pending': is_review_pending
                }
                pending_records.append(record)
        
        return pending_records
    
    def _find_sae_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Map standard column names to actual column names in SAE Dashboard"""
        mapping = {}
        
        column_candidates = {
            'discrepancy_id': ['Discrepancy ID', 'DiscrepancyID', 'discrepancy_id', 'ID'],
            'study_id': ['Study ID', 'StudyID', 'study_id', 'Study'],
            'patient_id': ['Patient ID', 'PatientID', 'patient_id', 'Subject ID', 'SubjectID'],
            'site': ['Site', 'Site ID', 'SiteID', 'site_id', 'Site Number'],
            'country': ['Country', 'country', 'COUNTRY'],
            'form_name': ['Form Name', 'FormName', 'form_name', 'Form'],
            'created_timestamp': ['Discrepancy Created Timestamp in Dashboard', 'Created Date', 'CreatedDate', 'Timestamp'],
            'review_status': ['Review Status', 'ReviewStatus', 'review_status'],
            'action_status': ['Action Status', 'ActionStatus', 'action_status', 'Status']
        }
        
        for key, candidates in column_candidates.items():
            for cand in candidates:
                if cand in df.columns:
                    mapping[key] = cand
                    break
        
        return mapping
    
    def _build_cpid_cache(self, cpid_data: Optional[pd.DataFrame]) -> None:
        """Build CPID lookup cache indexed by Subject ID"""
        self._cpid_cache = {}
        
        if cpid_data is None or cpid_data.empty:
            return
        
        # Handle duplicate columns
        if cpid_data.columns.duplicated().any():
            cpid_data = cpid_data.loc[:, ~cpid_data.columns.duplicated()]
        
        # Handle multi-level header CPID files - flatten columns
        # Check if columns are unnamed (indicating multi-level header read incorrectly)
        has_unnamed = any('Unnamed' in str(col) for col in cpid_data.columns[:10])
        
        if has_unnamed:
            # Try to identify columns by position for standard CPID format
            # Standard CPID layout: Region(1), Country(2), Site ID(3), Subject ID(4)
            col_list = list(cpid_data.columns)
            try:
                # Map by position for standard CPID format
                if len(col_list) > 4:
                    subject_col = col_list[4]  # Subject ID is typically column 5 (index 4)
                    site_col = col_list[3]     # Site ID is typically column 4 (index 3)
                    country_col = col_list[2]  # Country is typically column 3 (index 2)
                    
                    # Look for eSAE columns further in the file
                    esae_dm_col = None
                    esae_safety_col = None
                    recon_col = None
                    
                    for i, col in enumerate(col_list):
                        col_lower = str(col).lower()
                        if 'esae' in col_lower and 'dm' in col_lower:
                            esae_dm_col = col
                        elif 'esae' in col_lower and 'safety' in col_lower:
                            esae_safety_col = col
                        elif 'reconcil' in col_lower or ('open' in col_lower and 'issue' in col_lower):
                            recon_col = col
                    
                    # Build cache with positional columns
                    for _, row in cpid_data.iterrows():
                        subject_id = str(row.iloc[4]).strip() if len(row) > 4 else ''
                        if not subject_id or subject_id == 'nan' or pd.isna(subject_id):
                            continue
                        
                        self._cpid_cache[subject_id] = {
                            'subject_id': subject_id,
                            'site_id': str(row.iloc[3]) if len(row) > 3 else '',
                            'esae_dm_count': self._safe_int(row.get(esae_dm_col, 0)) if esae_dm_col else 0,
                            'esae_safety_count': self._safe_int(row.get(esae_safety_col, 0)) if esae_safety_col else 0,
                            'recon_issues': self._safe_int(row.get(recon_col, 0)) if recon_col else 0,
                            'raw_row': {}
                        }
                    
                    logger.info(f"[ZombieSAE] Built CPID cache with {len(self._cpid_cache)} subjects (positional)")
                    return
            except Exception as e:
                logger.warning(f"[ZombieSAE] Positional column mapping failed: {e}")
        
        # Standard column-based lookup
        subject_col = None
        for col in ['Subject ID', 'SubjectID', 'subject_id', 'Patient ID']:
            if col in cpid_data.columns:
                subject_col = col
                break
        
        if subject_col is None:
            # Try to find in flattened multi-level columns
            for col in cpid_data.columns:
                col_str = str(col).lower()
                if 'subject' in col_str and 'id' in col_str:
                    subject_col = col
                    break
        
        if subject_col is None:
            logger.warning("[ZombieSAE] CPID data missing Subject ID column")
            return
        
        # Find eSAE related columns
        esae_dm_col = None
        esae_safety_col = None
        recon_col = None
        site_col = None
        
        for col in cpid_data.columns:
            col_lower = str(col).lower()
            if 'esae' in col_lower and 'dm' in col_lower:
                esae_dm_col = col
            elif 'esae' in col_lower and 'safety' in col_lower:
                esae_safety_col = col
            elif 'reconcil' in col_lower or 'recon' in col_lower:
                recon_col = col
            elif 'site' in col_lower and 'id' in col_lower:
                site_col = col
        
        for _, row in cpid_data.iterrows():
            subject_id = str(row.get(subject_col, '')).strip()
            if not subject_id or subject_id == 'nan':
                continue
            
            self._cpid_cache[subject_id] = {
                'subject_id': subject_id,
                'site_id': str(row.get(site_col, '')) if site_col else '',
                'esae_dm_count': self._safe_int(row.get(esae_dm_col, 0)) if esae_dm_col else 0,
                'esae_safety_count': self._safe_int(row.get(esae_safety_col, 0)) if esae_safety_col else 0,
                'recon_issues': self._safe_int(row.get(recon_col, 0)) if recon_col else 0,
                'raw_row': row.to_dict() if hasattr(row, 'to_dict') else {}
            }
        
        logger.info(f"[ZombieSAE] Built CPID cache with {len(self._cpid_cache)} subjects")
    
    def _build_missing_pages_cache(self, missing_pages: Optional[pd.DataFrame]) -> None:
        """Build Missing Pages lookup cache indexed by Subject Name"""
        self._missing_pages_cache = {}
        
        if missing_pages is None or missing_pages.empty:
            return
        
        # Handle duplicate columns
        if missing_pages.columns.duplicated().any():
            missing_pages = missing_pages.loc[:, ~missing_pages.columns.duplicated()]
        
        # Find relevant columns
        subject_col = None
        for col in ['Subject Name', 'Subject ID', 'subject_id', 'SubjectName']:
            if col in missing_pages.columns:
                subject_col = col
                break
        
        page_col = None
        for col in ['Page Name', 'Form Name', 'page_name', 'PageName']:
            if col in missing_pages.columns:
                page_col = col
                break
        
        days_col = None
        for col in ['# of Days Missing', 'Days Missing', 'days_missing', 'DaysMissing']:
            if col in missing_pages.columns:
                days_col = col
                break
        
        if subject_col is None or page_col is None:
            logger.warning("[ZombieSAE] Missing Pages data missing required columns")
            return
        
        for _, row in missing_pages.iterrows():
            subject_id = str(row.get(subject_col, '')).strip()
            if not subject_id or subject_id == 'nan':
                continue
            
            if subject_id not in self._missing_pages_cache:
                self._missing_pages_cache[subject_id] = []
            
            self._missing_pages_cache[subject_id].append({
                'page_name': str(row.get(page_col, '')),
                'days_missing': self._safe_int(row.get(days_col, 0)) if days_col else 0,
                'visit_name': str(row.get('Visit Name', row.get('visit_name', ''))),
                'raw_row': row.to_dict() if hasattr(row, 'to_dict') else {}
            })
    
    def _build_edrr_cache(self, edrr_data: Optional[pd.DataFrame]) -> None:
        """Build EDRR lookup cache indexed by Subject"""
        self._edrr_cache = {}
        
        if edrr_data is None or edrr_data.empty:
            return
        
        # Handle duplicate columns
        if edrr_data.columns.duplicated().any():
            edrr_data = edrr_data.loc[:, ~edrr_data.columns.duplicated()]
        
        subject_col = None
        for col in ['Subject', 'Subject ID', 'subject_id', 'SubjectID']:
            if col in edrr_data.columns:
                subject_col = col
                break
        
        issue_col = None
        for col in ['Total Open issue Count per subject', 'Total Open issue Count', 
                    'total_issues', 'TotalIssues']:
            if col in edrr_data.columns:
                issue_col = col
                break
        
        if subject_col is None:
            return
        
        for _, row in edrr_data.iterrows():
            subject_id = str(row.get(subject_col, '')).strip()
            if not subject_id or subject_id == 'nan':
                continue
            
            self._edrr_cache[subject_id] = {
                'subject_id': subject_id,
                'total_issues': self._safe_int(row.get(issue_col, 0)) if issue_col else 0,
                'raw_row': row.to_dict() if hasattr(row, 'to_dict') else {}
            }
    
    def _process_pending_sae(self, pending_sae: Dict, study_id: str) -> Optional[ZombieSAECase]:
        """
        Process a single pending SAE through the detection pipeline:
        2. Cross-Reference: Query CPID for Subject ID
        3. Logic Gate: Check eSAE dashboard review for DM count
        4. Verification: Check Missing Pages for AE form
        5. Generate case with auto-query
        """
        patient_id = pending_sae['patient_id']
        site_id = pending_sae['site_id']
        
        if not patient_id:
            return None
        
        # Step 2: Cross-Reference with CPID
        cpid_data = self._cpid_cache.get(patient_id, {})
        esae_dm_count = cpid_data.get('esae_dm_count', 0)
        esae_safety_count = cpid_data.get('esae_safety_count', 0)
        recon_issues = cpid_data.get('recon_issues', 0)
        
        # Step 3: Logic Gate - Check if eSAE DM count indicates missing entry
        # If count is 0 (or less than expected), infer missing EDC entry
        is_potentially_zombie = esae_dm_count == 0 or pending_sae['is_review_pending']
        
        # Step 4: Verification - Check Missing Pages for AE form
        missing_ae_forms = []
        days_ae_missing = 0
        
        # Look for missing AE forms for this subject
        subject_missing_pages = self._missing_pages_cache.get(patient_id, [])
        for page in subject_missing_pages:
            page_name_lower = page['page_name'].lower()
            if any(pattern.lower() in page_name_lower for pattern in self.config.ae_form_patterns):
                missing_ae_forms.append(page['page_name'])
                days_ae_missing = max(days_ae_missing, page['days_missing'])
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            is_potentially_zombie=is_potentially_zombie,
            has_missing_ae_forms=len(missing_ae_forms) > 0,
            esae_dm_count=esae_dm_count,
            is_action_pending=pending_sae['is_action_pending'],
            is_review_pending=pending_sae['is_review_pending'],
            days_ae_missing=days_ae_missing
        )
        
        # Determine status based on confidence and evidence
        if confidence >= self.config.high_confidence_threshold and len(missing_ae_forms) > 0:
            status = ZombieSAEStatus.CONFIRMED
            outcome = ReconciliationOutcome.EDC_MISSING
        elif confidence >= self.config.medium_confidence_threshold:
            status = ZombieSAEStatus.SUSPECTED
            outcome = ReconciliationOutcome.DISCREPANT
        else:
            status = ZombieSAEStatus.PENDING_REVIEW
            outcome = ReconciliationOutcome.UNKNOWN
        
        # Parse timestamp
        created_ts = pending_sae.get('created_timestamp')
        if created_ts and not pd.isna(created_ts):
            try:
                created_ts = pd.to_datetime(created_ts)
            except:
                created_ts = datetime.now()
        else:
            created_ts = datetime.now()
        
        # Create the case
        case = ZombieSAECase(
            case_id=f"ZOMBIE_{pending_sae['discrepancy_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            discrepancy_id=pending_sae['discrepancy_id'],
            study_id=study_id or pending_sae.get('study_id', ''),
            site_id=site_id,
            patient_id=patient_id,
            country=pending_sae.get('country', ''),
            sae_form_name=pending_sae.get('form_name', ''),
            sae_created_timestamp=created_ts,
            sae_review_status=pending_sae.get('review_status', ''),
            sae_action_status=pending_sae.get('action_status', ''),
            cpid_esae_dm_count=esae_dm_count,
            cpid_esae_safety_count=esae_safety_count,
            cpid_recon_issues=recon_issues,
            missing_ae_forms=missing_ae_forms,
            days_ae_missing=days_ae_missing,
            status=status,
            reconciliation_outcome=outcome,
            confidence_score=confidence
        )
        
        # Step 5: Generate auto-query
        if self.config.enable_auto_query and status in [ZombieSAEStatus.CONFIRMED, ZombieSAEStatus.SUSPECTED]:
            case.auto_query_text = self._generate_auto_query(case)
        
        # Step 6: Calculate EDRR update
        if self.config.enable_edrr_update and status == ZombieSAEStatus.CONFIRMED:
            current_edrr = self._edrr_cache.get(patient_id, {}).get('total_issues', 0)
            case.edrr_new_issue_count = current_edrr + 1
            case.edrr_updated = True
        
        return case
    
    def _calculate_confidence(
        self,
        is_potentially_zombie: bool,
        has_missing_ae_forms: bool,
        esae_dm_count: int,
        is_action_pending: bool,
        is_review_pending: bool,
        days_ae_missing: int
    ) -> float:
        """Calculate confidence score for Zombie SAE detection"""
        confidence = 0.0
        
        # Base confidence from action/review status
        if is_action_pending:
            confidence += 0.20
        if is_review_pending:
            confidence += 0.25
        
        # Evidence from CPID
        if is_potentially_zombie:
            confidence += 0.20
        if esae_dm_count == 0:
            confidence += 0.15
        
        # Strong evidence from Missing Pages
        if has_missing_ae_forms:
            confidence += 0.25
            # Boost if AE form has been missing for a while
            if days_ae_missing >= 7:
                confidence += 0.10
            elif days_ae_missing >= 3:
                confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _generate_auto_query(self, case: ZombieSAECase) -> str:
        """Generate auto-query text for site"""
        missing_forms_str = ', '.join(case.missing_ae_forms) if case.missing_ae_forms else 'Unknown AE Form'
        
        query = self.ZOMBIE_SAE_QUERY_TEMPLATE.format(
            sae_date=case.sae_created_timestamp.strftime('%Y-%m-%d') if case.sae_created_timestamp else 'Unknown',
            patient_id=case.patient_id,
            site_id=case.site_id,
            discrepancy_id=case.discrepancy_id,
            form_name=case.sae_form_name,
            esae_dm_count=case.cpid_esae_dm_count,
            missing_forms=missing_forms_str
        )
        
        return query
    
    def _safe_int(self, value) -> int:
        """Safely convert value to integer"""
        if pd.isna(value):
            return 0
        try:
            return int(float(value))
        except:
            return 0
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of detected Zombie SAEs"""
        if not self.detected_cases:
            return {
                'total_cases': 0,
                'confirmed': 0,
                'suspected': 0,
                'pending_review': 0,
                'by_site': {},
                'by_country': {},
                'queries_to_send': 0,
                'edrr_updates_needed': 0
            }
        
        confirmed = [c for c in self.detected_cases if c.status == ZombieSAEStatus.CONFIRMED]
        suspected = [c for c in self.detected_cases if c.status == ZombieSAEStatus.SUSPECTED]
        pending = [c for c in self.detected_cases if c.status == ZombieSAEStatus.PENDING_REVIEW]
        
        # Group by site
        by_site = defaultdict(list)
        for c in self.detected_cases:
            by_site[c.site_id].append(c.to_dict())
        
        # Group by country
        by_country = defaultdict(list)
        for c in self.detected_cases:
            by_country[c.country].append(c.to_dict())
        
        return {
            'total_cases': len(self.detected_cases),
            'confirmed': len(confirmed),
            'suspected': len(suspected),
            'pending_review': len(pending),
            'high_confidence_cases': [c.to_dict() for c in self.detected_cases if c.confidence_score >= 0.90],
            'by_site': dict(by_site),
            'by_country': dict(by_country),
            'queries_to_send': len([c for c in self.detected_cases if c.auto_query_text and not c.query_sent]),
            'edrr_updates_needed': len([c for c in self.detected_cases if c.edrr_updated]),
            'generated_at': datetime.now().isoformat()
        }
    
    def get_auto_queries(self) -> List[Dict]:
        """Get list of auto-generated queries ready to send"""
        queries = []
        for case in self.detected_cases:
            if case.auto_query_text and not case.query_sent:
                queries.append({
                    'case_id': case.case_id,
                    'patient_id': case.patient_id,
                    'site_id': case.site_id,
                    'study_id': case.study_id,
                    'query_text': case.auto_query_text,
                    'priority': 'CRITICAL' if case.status == ZombieSAEStatus.CONFIRMED else 'HIGH',
                    'confidence': case.confidence_score,
                    'requires_approval': self.config.query_requires_approval
                })
        return queries
    
    def get_edrr_updates(self) -> List[Dict]:
        """Get list of EDRR updates to apply"""
        updates = []
        for case in self.detected_cases:
            if case.edrr_updated:
                updates.append({
                    'subject_id': case.patient_id,
                    'study_id': case.study_id,
                    'new_issue_count': case.edrr_new_issue_count,
                    'reason': f'Zombie SAE detected - Discrepancy ID: {case.discrepancy_id}',
                    'case_id': case.case_id
                })
        return updates


# Export classes
__all__ = [
    'ZombieSAEDetector',
    'ZombieSAECase',
    'ZombieSAEConfig',
    'ZombieSAEStatus',
    'ReconciliationOutcome',
    'DEFAULT_ZOMBIE_SAE_CONFIG'
]
