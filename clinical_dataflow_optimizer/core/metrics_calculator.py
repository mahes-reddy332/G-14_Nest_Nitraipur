"""
Clean Patient Status Calculator and Data Quality Index (DQI) Engine
Implements the core metrics calculation logic for the Neural Clinical Data Mesh
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.data_models import (
    DigitalPatientTwin, SiteMetrics, StudyMetrics,
    BlockingItem, RiskMetrics, PatientStatus, RiskLevel
)
from config.settings import (
    DQIWeights, CleanPatientThresholds, 
    DEFAULT_DQI_WEIGHTS, DEFAULT_CLEAN_THRESHOLDS
)

logger = logging.getLogger(__name__)


class CleanPatientCalculator:
    """
    Calculates Clean Patient Status for individual patients
    
    A patient is considered CLEAN if and only if ALL of the following conditions are met:
    1. Missing Visits = 0 AND Days Outstanding < threshold
    2. Missing Pages = 0
    3. Open Queries = 0
    4. Uncoded Terms = 0
    5. Reconciliation Issues = 0
    6. Verification % >= 100%
    7. Safety data is reconciled (EDC matches Safety DB)
    """
    
    def __init__(self, thresholds: CleanPatientThresholds = None):
        self.thresholds = thresholds or DEFAULT_CLEAN_THRESHOLDS
    
    def calculate_clean_status(
        self,
        cpid_row: pd.Series,
        visit_data: Optional[pd.DataFrame] = None,
        sae_data: Optional[pd.DataFrame] = None,
        subject_id: str = None
    ) -> Tuple[bool, float, List[BlockingItem]]:
        """
        Calculate clean patient status for a single patient
        
        Returns:
            Tuple of (is_clean: bool, clean_percentage: float, blocking_items: List)
        """
        blocking_items = []
        checks_passed = 0
        total_checks = 7
        
        # Helper to safely get numeric value
        def safe_get(series, key, default=0):
            try:
                val = series.get(key, default)
                if pd.isna(val):
                    return default
                return float(val)
            except:
                return default
        
        # Check 1: Missing Visits
        missing_visits = safe_get(cpid_row, 'missing_visits', 0)
        if missing_visits == 0:
            checks_passed += 1
        else:
            blocking_items.append(BlockingItem(
                item_type="Missing Visit",
                description=f"{int(missing_visits)} visit(s) not entered in EDC",
                source_file="CPID_EDC_Metrics",
                severity="High" if missing_visits > 2 else "Medium"
            ))
        
        # Check visit tracker for days outstanding
        if visit_data is not None and subject_id:
            subject_visits = visit_data[
                visit_data['subject_id'].astype(str) == str(subject_id)
            ] if 'subject_id' in visit_data.columns else pd.DataFrame()
            
            if len(subject_visits) > 0 and 'days_outstanding' in subject_visits.columns:
                max_days = subject_visits['days_outstanding'].max()
                if max_days > self.thresholds.max_days_outstanding:
                    blocking_items.append(BlockingItem(
                        item_type="Overdue Visit",
                        description=f"Visit overdue by {int(max_days)} days (threshold: {self.thresholds.max_days_outstanding})",
                        source_file="Visit_Projection_Tracker",
                        severity="Critical" if max_days > 60 else "High",
                        days_outstanding=int(max_days)
                    ))
        
        # Check 2: Missing Pages
        missing_pages = safe_get(cpid_row, 'missing_pages', 0)
        if missing_pages == 0:
            checks_passed += 1
        else:
            blocking_items.append(BlockingItem(
                item_type="Missing Page",
                description=f"{int(missing_pages)} CRF page(s) missing",
                source_file="CPID_EDC_Metrics",
                severity="High" if missing_pages > 5 else "Medium"
            ))
        
        # Check 3: Open Queries
        open_queries = safe_get(cpid_row, 'open_queries', 0)
        if open_queries <= self.thresholds.max_open_queries:
            checks_passed += 1
        else:
            blocking_items.append(BlockingItem(
                item_type="Open Query",
                description=f"{int(open_queries)} query(ies) remain open",
                source_file="CPID_EDC_Metrics",
                severity="High" if open_queries > 10 else "Medium"
            ))
        
        # Check 4: Uncoded Terms
        uncoded_terms = safe_get(cpid_row, 'uncoded_terms', 0)
        if uncoded_terms <= self.thresholds.max_uncoded_terms:
            checks_passed += 1
        else:
            blocking_items.append(BlockingItem(
                item_type="Uncoded Term",
                description=f"{int(uncoded_terms)} term(s) require medical coding",
                source_file="CPID_EDC_Metrics / GlobalCodingReport",
                severity="Medium"
            ))
        
        # Check 5: Reconciliation Issues
        recon_issues = safe_get(cpid_row, 'reconciliation_issues', 0)
        if recon_issues <= self.thresholds.max_reconciliation_issues:
            checks_passed += 1
        else:
            blocking_items.append(BlockingItem(
                item_type="Reconciliation Issue",
                description=f"{int(recon_issues)} EDC/Safety reconciliation issue(s)",
                source_file="CPID_EDC_Metrics / Compiled_EDRR",
                severity="Critical"
            ))
        
        # Check 6: Verification Percentage
        verification_pct = safe_get(cpid_row, 'verification_pct', 0)
        if verification_pct >= self.thresholds.min_verification_pct:
            checks_passed += 1
        else:
            blocking_items.append(BlockingItem(
                item_type="Incomplete Verification",
                description=f"Data verification at {verification_pct:.1f}% (required: {self.thresholds.min_verification_pct}%)",
                source_file="CPID_EDC_Metrics",
                severity="Medium"
            ))
        
        # Check 7: Safety Reconciliation (SAE consistency)
        esae_review = safe_get(cpid_row, 'esae_review', 0)
        safety_reconciled = True
        
        if sae_data is not None and subject_id:
            subject_saes = sae_data[
                sae_data['subject_id'].astype(str) == str(subject_id)
            ] if 'subject_id' in sae_data.columns else pd.DataFrame()
            
            if len(subject_saes) > 0:
                # Check for pending reviews
                if 'review_status' in subject_saes.columns:
                    pending = subject_saes[
                        subject_saes['review_status'].str.lower().str.contains('pending', na=False)
                    ]
                    if len(pending) > 0:
                        safety_reconciled = False
                        blocking_items.append(BlockingItem(
                            item_type="Safety Review Pending",
                            description=f"{len(pending)} SAE(s) pending DM review",
                            source_file="SAE_Dashboard",
                            severity="Critical"
                        ))
        
        if safety_reconciled:
            checks_passed += 1
        
        # Calculate overall clean status
        clean_percentage = (checks_passed / total_checks) * 100
        is_clean = len(blocking_items) == 0
        
        return is_clean, clean_percentage, blocking_items


class DataQualityIndexCalculator:
    """
    Calculates the Data Quality Index (DQI) using a weighted penalization model
    
    DQI = 100 - (W_visit * f(M_visit) + W_query * f(M_query) + 
                 W_conform * f(M_conform) + W_safety * f(M_safety))
    
    Where weights align with RBQM principles:
    - Visit Adherence: 20%
    - Query Responsiveness: 20%
    - Conformance: 20%
    - Safety Criticality: 40% (highest weight due to patient safety implications)
    """
    
    def __init__(self, weights: DQIWeights = None):
        self.weights = weights or DEFAULT_DQI_WEIGHTS
    
    def _normalize_metric(self, value: float, max_expected: float = 100) -> float:
        """Normalize a metric to 0-1 scale using sigmoid-like transformation"""
        if max_expected == 0:
            return 0
        # Use logarithmic scaling for better sensitivity at low values
        normalized = min(value / max_expected, 1.0)
        return normalized
    
    def _calculate_visit_penalty(
        self,
        missing_visits: float,
        max_days_outstanding: float,
        total_expected_visits: float
    ) -> float:
        """
        Calculate visit adherence penalty
        Higher penalty for more missing visits and longer outstanding days
        """
        if total_expected_visits == 0:
            return 0
        
        # Missing visit rate
        missing_rate = missing_visits / max(total_expected_visits, 1)
        
        # Days outstanding penalty (exponential for urgency)
        days_penalty = 0
        if max_days_outstanding > 30:
            days_penalty = min((max_days_outstanding - 30) / 60, 1.0)  # Max at 90 days
        
        combined_penalty = (missing_rate * 0.6) + (days_penalty * 0.4)
        return min(combined_penalty * 100, 100)  # Scale to 0-100
    
    def _calculate_query_penalty(
        self,
        open_queries: float,
        total_queries: float,
        crf_overdue: float,
        pages_entered: float
    ) -> float:
        """
        Calculate query responsiveness penalty
        Considers both open query ratio and query density
        """
        if total_queries == 0 and open_queries == 0:
            return 0
        
        # Open query ratio
        open_ratio = open_queries / max(total_queries, 1) if total_queries > 0 else 0
        
        # Query density (queries per page)
        density = total_queries / max(pages_entered, 1) if pages_entered > 0 else 0
        density_penalty = min(density * 10, 1.0)  # 10% queries per page = max penalty
        
        # Overdue penalty
        overdue_penalty = min(crf_overdue / 10, 1.0)  # 10+ overdue = max penalty
        
        combined_penalty = (open_ratio * 0.5) + (density_penalty * 0.3) + (overdue_penalty * 0.2)
        return min(combined_penalty * 100, 100)
    
    def _calculate_conformance_penalty(
        self,
        non_conformant_pages: float,
        pages_entered: float,
        protocol_deviations: float,
        inactivation_count: float = 0
    ) -> float:
        """
        Calculate data conformance penalty
        Non-conformant data and protocol deviations indicate quality issues
        """
        if pages_entered == 0:
            return 0
        
        # Non-conformance rate
        non_conform_rate = non_conformant_pages / max(pages_entered, 1)
        
        # Protocol deviation penalty (exponential for severity)
        pd_penalty = min(protocol_deviations / 5, 1.0)  # 5+ PDs = max penalty
        
        # Manipulation risk from inactivations
        inactivation_penalty = min(inactivation_count / 10, 1.0)
        
        combined_penalty = (non_conform_rate * 0.5) + (pd_penalty * 0.35) + (inactivation_penalty * 0.15)
        return min(combined_penalty * 100, 100)
    
    def _calculate_safety_penalty(
        self,
        reconciliation_issues: float,
        pending_sae_reviews: float,
        missing_lab_issues: float = 0
    ) -> float:
        """
        Calculate safety criticality penalty
        This carries the highest weight due to direct patient safety implications
        """
        # Reconciliation issues are critical
        recon_penalty = min(reconciliation_issues / 3, 1.0)  # 3+ = max penalty
        
        # Pending SAE reviews
        sae_penalty = min(pending_sae_reviews / 2, 1.0)  # 2+ = max penalty
        
        # Missing lab data
        lab_penalty = min(missing_lab_issues / 5, 1.0)
        
        combined_penalty = (recon_penalty * 0.5) + (sae_penalty * 0.35) + (lab_penalty * 0.15)
        return min(combined_penalty * 100, 100)
    
    def calculate_patient_dqi(
        self,
        cpid_row: pd.Series,
        visit_data: Optional[pd.DataFrame] = None,
        sae_data: Optional[pd.DataFrame] = None,
        subject_id: str = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate DQI for a single patient
        
        Returns:
            Tuple of (dqi_score: float, component_scores: Dict)
        """
        # Helper to safely get numeric value
        def safe_get(series, key, default=0):
            try:
                val = series.get(key, default)
                if pd.isna(val):
                    return default
                return float(val)
            except:
                return default
        
        # Extract metrics from CPID row
        missing_visits = safe_get(cpid_row, 'missing_visits', 0)
        missing_pages = safe_get(cpid_row, 'missing_pages', 0)
        open_queries = safe_get(cpid_row, 'open_queries', 0)
        total_queries = safe_get(cpid_row, 'total_queries', 0)
        uncoded_terms = safe_get(cpid_row, 'uncoded_terms', 0)
        pages_entered = safe_get(cpid_row, 'pages_entered', 1)
        non_conformant = safe_get(cpid_row, 'non_conformant', 0)
        recon_issues = safe_get(cpid_row, 'reconciliation_issues', 0)
        protocol_devs = safe_get(cpid_row, 'protocol_deviations', 0)
        crf_overdue = safe_get(cpid_row, 'crf_overdue', 0)
        expected_visits = safe_get(cpid_row, 'expected_visits', 1)
        
        # Get max days outstanding from visit tracker
        max_days = 0
        if visit_data is not None and subject_id:
            subject_visits = visit_data[
                visit_data['subject_id'].astype(str) == str(subject_id)
            ] if 'subject_id' in visit_data.columns else pd.DataFrame()
            
            if len(subject_visits) > 0 and 'days_outstanding' in subject_visits.columns:
                max_days = subject_visits['days_outstanding'].max()
                if pd.isna(max_days):
                    max_days = 0
        
        # Count pending SAE reviews
        pending_sae = 0
        if sae_data is not None and subject_id:
            subject_saes = sae_data[
                sae_data['subject_id'].astype(str) == str(subject_id)
            ] if 'subject_id' in sae_data.columns else pd.DataFrame()
            
            if len(subject_saes) > 0 and 'review_status' in subject_saes.columns:
                pending_sae = len(subject_saes[
                    subject_saes['review_status'].str.lower().str.contains('pending', na=False)
                ])
        
        # Calculate component penalties
        visit_penalty = self._calculate_visit_penalty(
            missing_visits + missing_pages, max_days, expected_visits
        )
        
        query_penalty = self._calculate_query_penalty(
            open_queries, total_queries, crf_overdue, pages_entered
        )
        
        conformance_penalty = self._calculate_conformance_penalty(
            non_conformant, pages_entered, protocol_devs
        )
        
        safety_penalty = self._calculate_safety_penalty(
            recon_issues, pending_sae
        )
        
        # Calculate weighted DQI
        total_penalty = (
            self.weights.visit_adherence * visit_penalty +
            self.weights.query_responsiveness * query_penalty +
            self.weights.conformance * conformance_penalty +
            self.weights.safety_criticality * safety_penalty
        )
        
        dqi = max(0, 100 - total_penalty)
        
        component_scores = {
            'visit_adherence': round(100 - visit_penalty, 1),
            'query_responsiveness': round(100 - query_penalty, 1),
            'conformance': round(100 - conformance_penalty, 1),
            'safety_criticality': round(100 - safety_penalty, 1)
        }
        
        return round(dqi, 1), component_scores
    
    def calculate_site_dqi(self, site_metrics: SiteMetrics) -> Tuple[float, Dict[str, float]]:
        """Calculate aggregated DQI for a site"""
        # Use aggregated metrics from SiteMetrics
        visit_penalty = self._calculate_visit_penalty(
            site_metrics.total_missing_visits,
            0,  # Would need to aggregate from visit tracker
            site_metrics.total_patients * 10  # Estimate expected visits
        )
        
        query_penalty = self._calculate_query_penalty(
            site_metrics.total_open_queries,
            site_metrics.total_queries,
            0,  # CRF overdue
            site_metrics.total_pages_entered
        )
        
        conformance_penalty = self._calculate_conformance_penalty(
            site_metrics.total_non_conformant,
            site_metrics.total_pages_entered,
            site_metrics.total_protocol_deviations
        )
        
        safety_penalty = 0  # Would need SAE aggregation
        
        total_penalty = (
            self.weights.visit_adherence * visit_penalty +
            self.weights.query_responsiveness * query_penalty +
            self.weights.conformance * conformance_penalty +
            self.weights.safety_criticality * safety_penalty
        )
        
        dqi = max(0, 100 - total_penalty)
        
        component_scores = {
            'visit_adherence': round(100 - visit_penalty, 1),
            'query_responsiveness': round(100 - query_penalty, 1),
            'conformance': round(100 - conformance_penalty, 1),
            'safety_criticality': round(100 - safety_penalty, 1)
        }
        
        return round(dqi, 1), component_scores
    
    @staticmethod
    def get_risk_level(dqi: float) -> RiskLevel:
        """Determine risk level from DQI score"""
        if dqi < 50:
            return RiskLevel.CRITICAL
        elif dqi < 75:
            return RiskLevel.HIGH
        elif dqi < 90:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


class PatientTwinBuilder:
    """
    Builds Digital Patient Twins from raw data sources
    Integrates data from all files to create unified patient representations
    """
    
    def __init__(self):
        self.clean_calculator = CleanPatientCalculator()
        self.dqi_calculator = DataQualityIndexCalculator()
    
    def build_patient_twin(
        self,
        subject_id: str,
        cpid_row: pd.Series,
        study_id: str,
        visit_data: Optional[pd.DataFrame] = None,
        sae_data: Optional[pd.DataFrame] = None,
        coding_data: Optional[pd.DataFrame] = None,
        missing_pages_data: Optional[pd.DataFrame] = None
    ) -> DigitalPatientTwin:
        """Build a complete Digital Patient Twin for a subject"""
        
        # Helper to safely get values
        def safe_get(series, key, default=None):
            try:
                val = series.get(key, default)
                if pd.isna(val):
                    return default
                return val
            except:
                return default
        
        def safe_get_num(series, key, default=0):
            val = safe_get(series, key, default)
            try:
                return float(val)
            except:
                return default
        
        # Create base twin
        twin = DigitalPatientTwin(
            subject_id=str(subject_id),
            site_id=str(safe_get(cpid_row, 'site_id', 'Unknown')),
            study_id=study_id,
            country=str(safe_get(cpid_row, 'country', '')),
            region=str(safe_get(cpid_row, 'region', ''))
        )
        
        # Set patient status
        status_str = str(safe_get(cpid_row, 'status', 'Unknown')).lower()
        if 'ongoing' in status_str or 'active' in status_str:
            twin.status = PatientStatus.ONGOING
        elif 'completed' in status_str or 'complete' in status_str:
            twin.status = PatientStatus.COMPLETED
        elif 'discontinued' in status_str or 'withdrawn' in status_str:
            twin.status = PatientStatus.DISCONTINUED
        elif 'screen' in status_str and 'fail' in status_str:
            twin.status = PatientStatus.SCREEN_FAILED
        else:
            twin.status = PatientStatus.UNKNOWN
        
        # Populate core metrics
        twin.missing_visits = int(safe_get_num(cpid_row, 'missing_visits', 0))
        twin.missing_pages = int(safe_get_num(cpid_row, 'missing_pages', 0))
        twin.open_queries = int(safe_get_num(cpid_row, 'open_queries', 0))
        twin.total_queries = int(safe_get_num(cpid_row, 'total_queries', 0))
        twin.uncoded_terms = int(safe_get_num(cpid_row, 'uncoded_terms', 0))
        twin.coded_terms = int(safe_get_num(cpid_row, 'coded_terms', 0))
        twin.verification_pct = safe_get_num(cpid_row, 'verification_pct', 0)
        twin.forms_verified = int(safe_get_num(cpid_row, 'forms_verified', 0))
        twin.expected_visits = int(safe_get_num(cpid_row, 'expected_visits', 0))
        twin.pages_entered = int(safe_get_num(cpid_row, 'pages_entered', 0))
        twin.non_conformant_pages = int(safe_get_num(cpid_row, 'non_conformant', 0))
        twin.reconciliation_issues = int(safe_get_num(cpid_row, 'reconciliation_issues', 0))
        twin.protocol_deviations = int(safe_get_num(cpid_row, 'protocol_deviations', 0))
        
        # Calculate clean patient status
        is_clean, clean_pct, blocking = self.clean_calculator.calculate_clean_status(
            cpid_row, visit_data, sae_data, subject_id
        )
        twin.clean_status = is_clean
        twin.clean_percentage = clean_pct
        twin.blocking_items = blocking
        
        # Calculate DQI
        dqi, components = self.dqi_calculator.calculate_patient_dqi(
            cpid_row, visit_data, sae_data, subject_id
        )
        twin.data_quality_index = dqi
        
        # Build risk metrics
        twin.risk_metrics = RiskMetrics(
            query_aging_index=twin.open_queries / max(twin.total_queries, 1) if twin.total_queries > 0 else 0,
            protocol_deviation_count=twin.protocol_deviations,
            visit_compliance_rate=100 - (twin.missing_visits / max(twin.expected_visits, 1) * 100) if twin.expected_visits > 0 else 100,
            data_density_score=twin.total_queries / max(twin.pages_entered, 1) if twin.pages_entered > 0 else 0
        )
        
        # Add outstanding visits from visit tracker
        if visit_data is not None and 'subject_id' in visit_data.columns:
            subject_visits = visit_data[
                visit_data['subject_id'].astype(str) == str(subject_id)
            ]
            if len(subject_visits) > 0:
                twin.outstanding_visits = subject_visits.to_dict('records')
        
        # Add SAE records
        if sae_data is not None and 'subject_id' in sae_data.columns:
            subject_saes = sae_data[
                sae_data['subject_id'].astype(str) == str(subject_id)
            ]
            if len(subject_saes) > 0:
                twin.sae_records = subject_saes.to_dict('records')
                
                # Check reconciliation status
                if 'review_status' in subject_saes.columns:
                    pending = subject_saes[
                        subject_saes['review_status'].str.lower().str.contains('pending', na=False)
                    ]
                    if len(pending) > 0:
                        twin.safety_reconciliation_status = "Pending Review"
                    else:
                        twin.safety_reconciliation_status = "Reconciled"
        
        # Add uncoded terms from coding data
        if coding_data is not None and 'subject_id' in coding_data.columns:
            try:
                subject_coding = coding_data[
                    coding_data['subject_id'].astype(str) == str(subject_id)
                ]
                # Filter for uncoded terms if coding_status column exists
                if len(subject_coding) > 0 and 'coding_status' in subject_coding.columns:
                    uncoded_mask = subject_coding['coding_status'].astype(str).str.lower().str.contains('uncoded|not coded|pending', na=False)
                    subject_uncoded = subject_coding[uncoded_mask]
                    if len(subject_uncoded) > 0:
                        twin.uncoded_terms_list = subject_uncoded.to_dict('records')
            except Exception:
                pass  # Silently handle any data issues
        
        twin.last_updated = datetime.now()
        
        return twin
    
    def build_all_twins(
        self,
        study_data: Dict[str, pd.DataFrame],
        study_id: str
    ) -> List[DigitalPatientTwin]:
        """Build Digital Patient Twins for all subjects in a study"""
        twins = []
        
        cpid = study_data.get('cpid_metrics')
        if cpid is None or len(cpid) == 0:
            logger.warning(f"No CPID metrics found for {study_id}")
            return twins
        
        visit_data = study_data.get('visit_tracker')
        sae_data = study_data.get('sae_dashboard')
        
        # Combine coding data
        coding_data = None
        meddra = study_data.get('meddra_coding')
        whodra = study_data.get('whodra_coding')
        if meddra is not None or whodra is not None:
            coding_dfs = []
            for df in [meddra, whodra]:
                if df is not None and len(df) > 0:
                    # Handle duplicate columns by keeping first occurrence
                    df = df.loc[:, ~df.columns.duplicated()]
                    coding_dfs.append(df)
            if coding_dfs:
                # Use outer join and handle column mismatches
                try:
                    coding_data = pd.concat(coding_dfs, ignore_index=True, sort=False)
                except Exception:
                    coding_data = coding_dfs[0] if coding_dfs else None
        
        missing_pages = study_data.get('missing_pages')
        
        # Remove duplicate columns from CPID data
        cpid = cpid.loc[:, ~cpid.columns.duplicated()]
        
        # Build twin for each subject
        for _, row in cpid.iterrows():
            subject_id = row.get('subject_id')
            # Handle case where subject_id might be a Series (duplicate column)
            if isinstance(subject_id, pd.Series):
                subject_id = subject_id.iloc[0] if len(subject_id) > 0 else None
            if pd.isna(subject_id):
                continue
            
            twin = self.build_patient_twin(
                subject_id=str(subject_id),
                cpid_row=row,
                study_id=study_id,
                visit_data=visit_data,
                sae_data=sae_data,
                coding_data=coding_data,
                missing_pages_data=missing_pages
            )
            twins.append(twin)
        
        logger.info(f"Built {len(twins)} Digital Patient Twins for {study_id}")
        return twins


class SiteMetricsAggregator:
    """
    Aggregates patient-level metrics to site-level
    """
    
    def __init__(self):
        self.dqi_calculator = DataQualityIndexCalculator()
    
    def aggregate_site_metrics(
        self,
        twins: List[DigitalPatientTwin],
        study_id: str
    ) -> Dict[str, SiteMetrics]:
        """Aggregate patient twins to site-level metrics"""
        sites = {}
        
        for twin in twins:
            site_id = twin.site_id
            
            if site_id not in sites:
                sites[site_id] = SiteMetrics(
                    site_id=site_id,
                    study_id=study_id,
                    country=twin.country,
                    region=twin.region
                )
            
            site = sites[site_id]
            site.total_patients += 1
            
            if twin.clean_status:
                site.clean_patients += 1
            
            if twin.status == PatientStatus.ONGOING:
                site.ongoing_patients += 1
            
            # Aggregate metrics
            site.total_missing_visits += twin.missing_visits
            site.total_missing_pages += twin.missing_pages
            site.total_open_queries += twin.open_queries
            site.total_queries += twin.total_queries
            site.total_uncoded_terms += twin.uncoded_terms
            site.total_protocol_deviations += twin.protocol_deviations
            site.total_non_conformant += twin.non_conformant_pages
            site.total_pages_entered += twin.pages_entered
        
        # Calculate derived metrics for each site
        for site_id, site in sites.items():
            site.calculate_derived_metrics()
            
            # Calculate site DQI
            dqi, _ = self.dqi_calculator.calculate_site_dqi(site)
            site.data_quality_index = dqi
            site.risk_level = DataQualityIndexCalculator.get_risk_level(dqi)
            
            # Determine intervention requirement
            site.requires_intervention = site.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
            
            # Set SSM status based on DQI
            if dqi >= 90:
                site.ssm_status = "Green"
            elif dqi >= 75:
                site.ssm_status = "Yellow"
            else:
                site.ssm_status = "Red"
        
        return sites


class StudyMetricsAggregator:
    """
    Aggregates site-level metrics to study-level
    """
    
    def aggregate_study_metrics(
        self,
        site_metrics: Dict[str, SiteMetrics],
        twins: List[DigitalPatientTwin],
        study_id: str
    ) -> StudyMetrics:
        """Aggregate site metrics to study-level"""
        study = StudyMetrics(
            study_id=study_id,
            study_name=study_id.replace('_', ' ')
        )
        
        study.total_sites = len(site_metrics)
        study.total_patients = len(twins)
        study.clean_patients = sum(1 for t in twins if t.clean_status)
        study.ongoing_patients = sum(1 for t in twins if t.status == PatientStatus.ONGOING)
        
        # Count sites by risk level
        for site in site_metrics.values():
            study.sites_by_risk[site.risk_level.value] += 1
            if site.requires_intervention:
                study.sites_at_risk += 1
            
            if site.country and site.country not in study.countries:
                study.countries.append(site.country)
        
        # Calculate global DQI (weighted average by patient count)
        total_weighted_dqi = 0
        total_weight = 0
        for site in site_metrics.values():
            total_weighted_dqi += site.data_quality_index * site.total_patients
            total_weight += site.total_patients
        
        if total_weight > 0:
            study.global_dqi = round(total_weighted_dqi / total_weight, 1)
        
        # Check interim analysis readiness
        study.calculate_interim_readiness()
        
        return study


class FeatureEnhancedTwinBuilder:
    """
    Enhanced Digital Patient Twin builder that integrates feature engineering
    Creates twins with sophisticated engineered features for AI/ML models

    Features Integrated:
    1. Operational Velocity Index (Query resolution velocity)
    2. Normalized Data Density (Queries per page with percentiles)
    3. Manipulation Risk Score (Based on inactivation patterns)
    """

    def __init__(self):
        self.clean_calculator = CleanPatientCalculator()
        self.dqi_calculator = DataQualityIndexCalculator()

        # Import feature engineering components
        try:
            from core.feature_engineering import (
                SiteFeatureEngineer,
                engineer_study_features,
                OperationalVelocityIndex,
                NormalizedDataDensity,
                ManipulationRiskScore
            )
            self.feature_engineer_available = True
        except ImportError:
            logger.warning("Feature engineering module not available, falling back to basic metrics")
            self.feature_engineer_available = False

    def build_patient_twin_with_features(
        self,
        subject_id: str,
        cpid_row: pd.Series,
        study_id: str,
        site_features: Optional[Dict[str, Any]] = None,
        visit_data: Optional[pd.DataFrame] = None,
        sae_data: Optional[pd.DataFrame] = None,
        coding_data: Optional[pd.DataFrame] = None,
        missing_pages_data: Optional[pd.DataFrame] = None,
        inactivated_forms_data: Optional[pd.DataFrame] = None
    ) -> DigitalPatientTwin:
        """
        Build a Digital Patient Twin with integrated feature engineering

        Args:
            subject_id: Patient identifier
            cpid_row: Patient data from CPID metrics
            study_id: Study identifier
            site_features: Pre-computed site-level features (optional)
            visit_data: Visit tracking data
            sae_data: Safety event data
            coding_data: Medical coding data
            missing_pages_data: Missing pages report
            inactivated_forms_data: Inactivated forms for manipulation risk

        Returns:
            DigitalPatientTwin with engineered features in risk_metrics
        """

        # Start with basic twin building (reuse existing logic)
        basic_twin = self._build_basic_twin(
            subject_id, cpid_row, study_id, visit_data, sae_data,
            coding_data, missing_pages_data
        )

        # Enhance with engineered features if available
        if self.feature_engineer_available and site_features:
            basic_twin.risk_metrics = self._enhance_risk_metrics_with_features(
                basic_twin.risk_metrics, site_features, subject_id, inactivated_forms_data
            )
        else:
            logger.info(f"Using basic risk metrics for patient {subject_id} (feature engineering not available)")

        return basic_twin

    def _build_basic_twin(
        self,
        subject_id: str,
        cpid_row: pd.Series,
        study_id: str,
        visit_data: Optional[pd.DataFrame] = None,
        sae_data: Optional[pd.DataFrame] = None,
        coding_data: Optional[pd.DataFrame] = None,
        missing_pages_data: Optional[pd.DataFrame] = None
    ) -> DigitalPatientTwin:
        """Build basic twin using existing PatientTwinBuilder logic"""

        # Create a temporary PatientTwinBuilder to reuse logic
        temp_builder = PatientTwinBuilder()
        return temp_builder.build_patient_twin(
            subject_id, cpid_row, study_id, visit_data, sae_data,
            coding_data, missing_pages_data
        )

    def _enhance_risk_metrics_with_features(
        self,
        basic_metrics: RiskMetrics,
        site_features: Dict[str, Any],
        subject_id: str,
        inactivated_forms_data: Optional[pd.DataFrame] = None
    ) -> RiskMetrics:
        """
        Enhance basic risk metrics with engineered features

        Args:
            basic_metrics: Basic risk metrics from twin builder
            site_features: Site-level engineered features
            subject_id: Patient identifier for patient-specific features
            inactivated_forms_data: Data for manipulation risk calculation

        Returns:
            Enhanced RiskMetrics with engineered features
        """

        # Feature 1: Operational Velocity Index
        if 'operational_velocity' in site_features:
            velocity_data = site_features['operational_velocity']
            basic_metrics.resolution_velocity = velocity_data.get('resolution_velocity', 0.0)
            basic_metrics.accumulation_velocity = velocity_data.get('accumulation_velocity', 0.0)
            basic_metrics.net_velocity = velocity_data.get('net_velocity', 0.0)
            basic_metrics.is_bottleneck = velocity_data.get('is_bottleneck', False)

        # Feature 2: Normalized Data Density
        if 'data_density' in site_features:
            density_data = site_features['data_density']
            basic_metrics.data_density_score = density_data.get('density_score', 0.0)
            basic_metrics.query_density_normalized = density_data.get('normalized_density', 0.0)
            basic_metrics.query_density_percentile = density_data.get('percentile', 0.0)

        # Feature 3: Manipulation Risk Score
        if 'manipulation_risk' in site_features:
            risk_data = site_features['manipulation_risk']
            basic_metrics.manipulation_risk_score = risk_data.get('risk_level', 'Low')
            basic_metrics.manipulation_risk_value = risk_data.get('risk_score', 0.0)
            basic_metrics.endpoint_risk_score = risk_data.get('endpoint_risk', 0.0)
            basic_metrics.inactivation_rate = risk_data.get('inactivation_rate', 0.0)

        # Calculate composite risk score
        basic_metrics.composite_risk_score = self._calculate_composite_risk_score(basic_metrics)
        basic_metrics.requires_intervention = basic_metrics.composite_risk_score >= 60  # High risk threshold

        logger.info(f"Enhanced risk metrics for patient {subject_id}: composite_score={basic_metrics.composite_risk_score:.1f}")

        return basic_metrics

    def _calculate_composite_risk_score(self, metrics: RiskMetrics) -> float:
        """
        Calculate composite risk score from engineered features
        Weights: Velocity (40%), Density (30%), Manipulation Risk (30%)
        """

        # Velocity component (0-40 points)
        velocity_score = 0.0
        if metrics.is_bottleneck:
            velocity_score = 40.0  # Maximum penalty for bottlenecks
        elif metrics.net_velocity < 0:
            velocity_score = 30.0  # High penalty for negative velocity
        elif metrics.net_velocity < 0.5:
            velocity_score = 20.0  # Medium penalty for low velocity
        else:
            velocity_score = max(0, 40.0 - (metrics.net_velocity * 10))  # Lower score for higher velocity

        # Density component (0-30 points)
        density_score = metrics.query_density_normalized * 30.0

        # Manipulation risk component (0-30 points)
        manipulation_score = metrics.manipulation_risk_value * 0.3  # Scale 0-100 to 0-30

        composite = velocity_score + density_score + manipulation_score

        return min(100.0, max(0.0, composite))  # Clamp to 0-100 range

    def build_all_twins(
        self,
        study_data: Dict[str, pd.DataFrame],
        study_id: str
    ) -> List[DigitalPatientTwin]:
        """
        Build Digital Patient Twins for all subjects in a study with feature engineering

        This method integrates feature engineering into the twin creation process:
        1. First runs feature engineering on site-level data
        2. Then builds individual patient twins with engineered features
        """
        twins = []

        cpid = study_data.get('cpid_metrics')
        if cpid is None or len(cpid) == 0:
            logger.warning(f"No CPID metrics found for {study_id}")
            return twins

        # Get supporting data
        visit_data = study_data.get('visit_tracker')
        sae_data = study_data.get('safety_events')
        coding_data = study_data.get('coding_data')
        missing_pages_data = study_data.get('missing_pages')
        inactivated_forms_data = study_data.get('inactivated_forms')

        # Step 1: Run feature engineering if available
        site_features = {}
        if self.feature_engineer_available:
            try:
                logger.info(f"Running feature engineering for {study_id}")
                site_features = self._run_feature_engineering_for_study(
                    study_data, study_id
                )
                logger.info(f"Feature engineering completed: {len(site_features)} sites processed")
            except Exception as e:
                logger.error(f"Feature engineering failed for {study_id}: {e}")
                logger.info("Falling back to basic twin building")

        # Step 2: Build twins for each patient
        for idx, patient_row in cpid.iterrows():
            try:
                subject_id = str(patient_row.get('subject_id', f'unknown_{idx}'))

                # Get site-specific features for this patient
                site_id = str(patient_row.get('site_id', 'unknown'))
                patient_site_features = site_features.get(site_id, {})

                # Build enhanced twin
                twin = self.build_patient_twin_with_features(
                    subject_id=subject_id,
                    cpid_row=patient_row,
                    study_id=study_id,
                    site_features=patient_site_features,
                    visit_data=visit_data,
                    sae_data=sae_data,
                    coding_data=coding_data,
                    missing_pages_data=missing_pages_data,
                    inactivated_forms_data=inactivated_forms_data
                )

                twins.append(twin)

            except Exception as e:
                logger.error(f"Failed to build twin for patient {idx}: {e}")
                continue

        logger.info(f"Built {len(twins)} enhanced Digital Patient Twins for {study_id}")
        return twins

    def _run_feature_engineering_for_study(
        self,
        study_data: Dict[str, pd.DataFrame],
        study_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run feature engineering for all sites in a study

        Returns:
            Dict mapping site_id to feature data
        """
        from core.feature_engineering import engineer_study_features

        # Run feature engineering
        feature_results = engineer_study_features(
            study_id=study_id,
            cpid_metrics=study_data.get('cpid_metrics'),
            inactivated_forms=study_data.get('inactivated_forms')
        )

        # Organize by site for easy lookup
        site_features = {}
        for site_result in feature_results.get('site_features', []):
            site_id = site_result.get('site_id')
            if site_id:
                site_features[str(site_id)] = site_result.get('features', {})

        return site_features

    def build_twins(
        self,
        cpid_data: pd.DataFrame,
        sae_data: Optional[pd.DataFrame] = None,
        visit_data: Optional[pd.DataFrame] = None,
        coding_data: Optional[pd.DataFrame] = None,
        study_id: str = "unknown"
    ) -> List[DigitalPatientTwin]:
        """
        Compatibility method for web app - builds twins from individual DataFrames
        Wraps build_all_twins with study_data format
        """
        study_data = {
            'cpid_metrics': cpid_data,
            'safety_events': sae_data,
            'visit_tracker': visit_data,
            'coding_data': coding_data
        }

        return self.build_all_twins(study_data, study_id)
