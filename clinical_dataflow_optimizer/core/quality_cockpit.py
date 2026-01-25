r"""
Risk-Based Quality Cockpit - Metrics & Visualization Engine
=============================================================

This module implements the mathematically rigorous metrics required to assess
"Clean Patient Status" and overall "Data Quality" for clinical trials.

Key Components:
1. Clean Patient Status ($S_c$) - Binary metric from boolean logic tree
2. Data Quality Index (DQI) - Weighted penalization model
3. Clean Patient Progress Tracker - Real-time visualization
4. Dynamic Recalculation Engine - Updates on data ingestion

Mathematical Foundation:
------------------------
A patient is considered CLEAN ($S_c = 1$) if and only if:

$S_c = V \land P \land Q \land C \land R \land S \land E$

Where:
- V: Visits condition (missing_visits = 0 AND days_outstanding < threshold)
- P: Pages condition (missing_pages = 0)
- Q: Queries condition (open_queries = 0 AND query_status = "Closed")
- C: Coding condition (uncoded_terms = 0 AND coded_terms > 0)
- R: Reconciliation condition (reconciliation_issues = 0)
- S: Safety condition (EDC ↔ SAE Dashboard consistency)
- E: Verification condition (verification_pct = 100% AND forms_verified = expected_visits)

Clean Progress Percentage:
$P_{clean} = \frac{\sum_{i=1}^{7} w_i \cdot c_i}{\sum_{i=1}^{7} w_i} \times 100$

Where $c_i \in \{0,1\}$ is the condition result and $w_i$ is the weight.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import json
import re

logger = logging.getLogger(__name__)


class CleanCondition(Enum):
    """Clean Patient Status conditions"""
    VISITS = "visits"
    PAGES = "pages"
    QUERIES = "queries"
    CODING = "coding"
    RECONCILIATION = "reconciliation"
    SAFETY = "safety"
    VERIFICATION = "verification"


class BlockerSeverity(Enum):
    """Severity levels for blocking items"""
    CRITICAL = "critical"     # Immediate action required
    HIGH = "high"             # Urgent attention needed
    MEDIUM = "medium"         # Should be addressed soon
    LOW = "low"               # Minor issue
    INFORMATIONAL = "info"    # For awareness


@dataclass
class CleanConditionResult:
    """Result of evaluating a clean condition"""
    condition: CleanCondition
    is_met: bool
    score: float  # 0.0 to 1.0
    weight: float  # Condition weight
    weighted_score: float  # score * weight
    blockers: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    severity: BlockerSeverity = BlockerSeverity.INFORMATIONAL
    source_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'condition': self.condition.value,
            'is_met': self.is_met,
            'score': round(self.score, 4),
            'weight': self.weight,
            'weighted_score': round(self.weighted_score, 4),
            'blockers': self.blockers,
            'metrics': self.metrics,
            'severity': self.severity.value,
            'source_files': self.source_files
        }


@dataclass
class CleanPatientStatus:
    r"""
    Complete Clean Patient Status assessment
    
    The status follows a binary logical conjunction:
    $S_c = V \land P \land Q \land C \land R \land S \land E$
    
    A patient is CLEAN if and only if ALL conditions are met.
    """
    subject_id: str
    site_id: str
    is_clean: bool  # The final binary status ($S_c$)
    clean_percentage: float  # Weighted percentage (0-100)
    conditions: List[CleanConditionResult] = field(default_factory=list)
    total_blockers: int = 0
    primary_blocker: Optional[str] = None
    status_summary: str = ""
    calculated_at: datetime = field(default_factory=datetime.now)
    
    # Metadata
    data_sources_used: List[str] = field(default_factory=list)
    calculation_version: str = "2.0"
    
    def to_dict(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'site_id': self.site_id,
            'is_clean': self.is_clean,
            'clean_percentage': round(self.clean_percentage, 2),
            'conditions': [c.to_dict() for c in self.conditions],
            'total_blockers': self.total_blockers,
            'primary_blocker': self.primary_blocker,
            'status_summary': self.status_summary,
            'calculated_at': self.calculated_at.isoformat(),
            'data_sources_used': self.data_sources_used,
            'calculation_version': self.calculation_version
        }
    
    def get_progress_bar_data(self) -> Dict:
        """
        Generate data for the Clean Patient Progress Bar visualization
        
        Returns dict with:
        - percentage: float (0-100)
        - status_text: str (e.g., "95% Clean - Blocked by 1 Coding Query")
        - color: str (hex color based on percentage)
        - segments: List of segment data for each condition
        """
        # Determine color based on percentage
        if self.clean_percentage >= 100:
            color = "#00CC00"  # Green - Clean
        elif self.clean_percentage >= 90:
            color = "#7FD77F"  # Light green
        elif self.clean_percentage >= 75:
            color = "#FFCC00"  # Yellow
        elif self.clean_percentage >= 50:
            color = "#FF6600"  # Orange
        else:
            color = "#FF0000"  # Red
        
        # Build status text
        if self.is_clean:
            status_text = "100% Clean ✓"
        elif self.total_blockers == 1 and self.primary_blocker:
            status_text = f"{self.clean_percentage:.0f}% Clean - Blocked by {self.primary_blocker}"
        elif self.total_blockers > 0:
            status_text = f"{self.clean_percentage:.0f}% Clean - {self.total_blockers} blocking items"
        else:
            status_text = f"{self.clean_percentage:.0f}% Clean"
        
        # Build segments for detailed visualization
        segments = []
        for condition in self.conditions:
            segments.append({
                'name': condition.condition.value.title(),
                'score': condition.score * 100,
                'weight': condition.weight,
                'is_met': condition.is_met,
                'color': "#00CC00" if condition.is_met else "#FF6600" if condition.score > 0.5 else "#FF0000"
            })
        
        return {
            'percentage': self.clean_percentage,
            'status_text': status_text,
            'color': color,
            'segments': segments,
            'is_clean': self.is_clean,
            'total_blockers': self.total_blockers
        }


@dataclass
class QualityCockpitConfig:
    """Configuration for Quality Cockpit calculations"""
    # Condition weights (must sum to 1.0)
    weight_visits: float = 0.15
    weight_pages: float = 0.15
    weight_queries: float = 0.15
    weight_coding: float = 0.10
    weight_reconciliation: float = 0.15
    weight_safety: float = 0.20
    weight_verification: float = 0.10
    
    # Thresholds
    max_days_outstanding: int = 30
    max_open_queries: int = 0
    max_uncoded_terms: int = 0
    min_verification_pct: float = 100.0
    max_reconciliation_issues: int = 0
    
    # Partial scoring configuration
    enable_partial_scoring: bool = True  # Allow partial credit
    visit_days_grace_period: int = 7     # Grace period before penalty
    query_aging_threshold: int = 14      # Days before query becomes urgent
    
    def validate(self) -> bool:
        """Validate configuration"""
        total_weight = (
            self.weight_visits + self.weight_pages + self.weight_queries +
            self.weight_coding + self.weight_reconciliation +
            self.weight_safety + self.weight_verification
        )
        return abs(total_weight - 1.0) < 0.001


# Default configuration
DEFAULT_COCKPIT_CONFIG = QualityCockpitConfig()


class CleanPatientStatusCalculator:
    r"""
    Calculates Clean Patient Status using mathematically rigorous boolean logic
    
    The Clean Patient Status is a binary metric derived from:
    $S_c = V \land P \land Q \land C \land R \land S \land E$
    
    This calculation is performed dynamically every time new data is ingested.
    """
    
    # Column mappings for standardization
    COLUMN_MAPS = {
        'subject_id': ['Subject ID', 'Subject', 'SubjectID', 'SUBJECTID', 'subject_id'],
        'site_id': ['Site ID', 'Site', 'SiteID', 'SITEID', 'site_id'],
        'missing_visits': ['Missing Visits', '# Missing Visits', 'MissingVisits', 'missing_visits'],
        'missing_pages': ['Missing Page', '# Missing Pages', 'MissingPages', 'missing_pages'],
        'open_queries': ['# Open Queries', 'Open Queries', 'OpenQueries', 'open_queries'],
        'query_status': ['Queries status', 'Query Status', 'QueryStatus', 'query_status'],
        'uncoded_terms': ['# Uncoded Terms', 'Uncoded Terms', 'UncodedTerms', 'uncoded_terms'],
        'coded_terms': ['# Coded terms', 'Coded Terms', 'CodedTerms', 'coded_terms'],
        'reconciliation_issues': ['# Reconciliation Issues', 'Recon Issues', 'ReconIssues', 'reconciliation_issues'],
        'verification_pct': ['Data Verification %', 'Verification %', 'VerificationPct', 'verification_pct'],
        'forms_verified': ['# Forms Verified', 'Forms Verified', 'FormsVerified', 'forms_verified'],
        'expected_visits': ['# Expected Visits', 'Expected Visits', 'ExpectedVisits', 'expected_visits'],
        'esae_review': ['# eSAE dashboard review for DM', 'eSAE Review', 'eSAEReview', 'esae_review'],
        'days_outstanding': ['# Days Outstanding', 'Days Outstanding', 'DaysOutstanding', 'days_outstanding'],
    }
    
    def __init__(self, config: QualityCockpitConfig = None):
        """Initialize calculator with configuration"""
        self.config = config or DEFAULT_COCKPIT_CONFIG
        if not self.config.validate():
            logger.warning("Configuration weights do not sum to 1.0, normalizing...")
        
        self.data_sources: Dict[str, pd.DataFrame] = {}
        self._column_cache: Dict[str, Dict[str, str]] = {}
        logger.info("CleanPatientStatusCalculator initialized")
    
    def load_data(self, data_sources: Dict[str, pd.DataFrame]):
        """Load data sources for calculation"""
        self.data_sources = data_sources
        self._column_cache = {}  # Clear cache
        logger.info(f"Loaded {len(data_sources)} data sources")
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str], df_name: str = "") -> Optional[str]:
        """Find a column from a list of candidates"""
        cache_key = f"{df_name}_{candidates[0]}"
        if cache_key in self._column_cache:
            return self._column_cache.get(cache_key, {}).get('found')
        
        for col in candidates:
            if col in df.columns:
                self._column_cache[cache_key] = {'found': col}
                return col
        
        # Try case-insensitive match
        lower_cols = {c.lower(): c for c in df.columns}
        for col in candidates:
            if col.lower() in lower_cols:
                found = lower_cols[col.lower()]
                self._column_cache[cache_key] = {'found': found}
                return found
        
        return None
    
    def _safe_get(self, row: pd.Series, key: str, default: Any = 0) -> Any:
        """Safely get value from row with default"""
        try:
            val = row.get(key, default)
            if pd.isna(val):
                return default
            return val
        except:
            return default
    
    def _safe_numeric(self, row: pd.Series, key: str, default: float = 0.0) -> float:
        """Safely get numeric value from row"""
        try:
            val = row.get(key, default)
            if pd.isna(val):
                return default
            return float(val)
        except:
            return default
    
    def calculate_status(
        self,
        subject_id: str,
        cpid_row: Optional[pd.Series] = None,
        visit_data: Optional[pd.DataFrame] = None,
        sae_data: Optional[pd.DataFrame] = None,
        missing_pages_data: Optional[pd.DataFrame] = None
    ) -> CleanPatientStatus:
        """
        Calculate Clean Patient Status for a single subject
        
        Args:
            subject_id: Subject identifier
            cpid_row: Row from CPID_EDC_Metrics (if None, will try to find)
            visit_data: Visit Projection Tracker data
            sae_data: SAE Dashboard data
            missing_pages_data: Global_Missing_Pages_Report data
            
        Returns:
            CleanPatientStatus with complete assessment
        """
        start_time = datetime.now()
        data_sources_used = []
        
        # Get CPID row if not provided
        if cpid_row is None:
            cpid_df = self.data_sources.get('cpid')
            if cpid_df is not None:
                subj_col = self._find_column(cpid_df, self.COLUMN_MAPS['subject_id'], 'cpid')
                if subj_col:
                    matches = cpid_df[cpid_df[subj_col].astype(str).str.contains(str(subject_id), na=False)]
                    if len(matches) > 0:
                        cpid_row = matches.iloc[0]
                        data_sources_used.append('CPID_EDC_Metrics')
        else:
            data_sources_used.append('CPID_EDC_Metrics')
        
        # Get supplementary data
        if visit_data is None:
            visit_data = self.data_sources.get('visit_tracker')
        if visit_data is not None and not visit_data.empty:
            data_sources_used.append('Visit_Projection_Tracker')
        
        if sae_data is None:
            sae_data = self.data_sources.get('esae')
            if sae_data is None:
                sae_data = self.data_sources.get('sae')
        if sae_data is not None and not sae_data.empty:
            data_sources_used.append('SAE_Dashboard')
        
        if missing_pages_data is None:
            missing_pages_data = self.data_sources.get('missing_pages')
        if missing_pages_data is not None and not missing_pages_data.empty:
            data_sources_used.append('Global_Missing_Pages_Report')
        
        # Extract site_id
        site_id = "Unknown"
        if cpid_row is not None:
            site_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['site_id'])
            if site_col:
                site_id = str(self._safe_get(cpid_row, site_col, "Unknown"))
        
        # Calculate each condition
        conditions = []
        
        # Condition 1: VISITS
        visit_result = self._evaluate_visits_condition(cpid_row, visit_data, subject_id)
        conditions.append(visit_result)
        
        # Condition 2: PAGES
        pages_result = self._evaluate_pages_condition(cpid_row, missing_pages_data, subject_id)
        conditions.append(pages_result)
        
        # Condition 3: QUERIES
        queries_result = self._evaluate_queries_condition(cpid_row)
        conditions.append(queries_result)
        
        # Condition 4: CODING
        coding_result = self._evaluate_coding_condition(cpid_row)
        conditions.append(coding_result)
        
        # Condition 5: RECONCILIATION
        recon_result = self._evaluate_reconciliation_condition(cpid_row)
        conditions.append(recon_result)
        
        # Condition 6: SAFETY
        safety_result = self._evaluate_safety_condition(cpid_row, sae_data, subject_id)
        conditions.append(safety_result)
        
        # Condition 7: VERIFICATION
        verify_result = self._evaluate_verification_condition(cpid_row)
        conditions.append(verify_result)
        
        # Calculate final status
        # Binary: All conditions must be met
        is_clean = all(c.is_met for c in conditions)
        
        # Calculate weighted percentage
        total_weighted_score = sum(c.weighted_score for c in conditions)
        total_weight = sum(c.weight for c in conditions)
        clean_percentage = (total_weighted_score / total_weight * 100) if total_weight > 0 else 0
        
        # Count blockers and find primary blocker
        all_blockers = []
        for c in conditions:
            all_blockers.extend(c.blockers)
        
        total_blockers = len(all_blockers)
        
        # Primary blocker is the one with highest severity from failed conditions
        primary_blocker = None
        if not is_clean:
            failed_conditions = [c for c in conditions if not c.is_met]
            if failed_conditions:
                # Sort by severity and weight
                severity_order = {
                    BlockerSeverity.CRITICAL: 0,
                    BlockerSeverity.HIGH: 1,
                    BlockerSeverity.MEDIUM: 2,
                    BlockerSeverity.LOW: 3,
                    BlockerSeverity.INFORMATIONAL: 4
                }
                failed_conditions.sort(key=lambda x: (severity_order[x.severity], -x.weight))
                if failed_conditions[0].blockers:
                    primary_blocker = failed_conditions[0].blockers[0]
        
        # Build status summary
        if is_clean:
            status_summary = "Patient is CLEAN - All conditions met"
        else:
            failed = [c.condition.value for c in conditions if not c.is_met]
            status_summary = f"Patient NOT CLEAN - Failed conditions: {', '.join(failed)}"
        
        status = CleanPatientStatus(
            subject_id=subject_id,
            site_id=site_id,
            is_clean=is_clean,
            clean_percentage=clean_percentage,
            conditions=conditions,
            total_blockers=total_blockers,
            primary_blocker=primary_blocker,
            status_summary=status_summary,
            calculated_at=datetime.now(),
            data_sources_used=data_sources_used
        )
        
        logger.debug(f"Calculated status for {subject_id} in {(datetime.now() - start_time).total_seconds():.3f}s")
        return status
    
    def _evaluate_visits_condition(
        self,
        cpid_row: Optional[pd.Series],
        visit_data: Optional[pd.DataFrame],
        subject_id: str
    ) -> CleanConditionResult:
        """
        Evaluate Visits condition:
        CPID_EDC_Metrics["Missing Visits"] == 0 AND
        Visit_Projection_Tracker is below threshold (< 30 days)
        """
        blockers = []
        metrics = {}
        score = 1.0
        is_met = True
        severity = BlockerSeverity.INFORMATIONAL
        source_files = ['CPID_EDC_Metrics']
        
        # Check missing visits from CPID
        missing_visits = 0
        if cpid_row is not None:
            mv_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['missing_visits'])
            if mv_col:
                missing_visits = self._safe_numeric(cpid_row, mv_col, 0)
        
        metrics['missing_visits'] = missing_visits
        
        if missing_visits > 0:
            is_met = False
            score = max(0, 1 - (missing_visits / 5))  # Partial score
            blockers.append(f"{int(missing_visits)} missing visit(s)")
            severity = BlockerSeverity.HIGH if missing_visits > 2 else BlockerSeverity.MEDIUM
        
        # Check visit projection tracker for days outstanding
        max_days_outstanding = 0
        if visit_data is not None and len(visit_data) > 0:
            source_files.append('Visit_Projection_Tracker')
            subj_col = self._find_column(visit_data, self.COLUMN_MAPS['subject_id'], 'visit')
            days_col = self._find_column(visit_data, self.COLUMN_MAPS['days_outstanding'], 'visit')
            
            if subj_col and days_col:
                subject_visits = visit_data[
                    visit_data[subj_col].astype(str).str.contains(str(subject_id), na=False)
                ]
                if len(subject_visits) > 0:
                    max_days_outstanding = pd.to_numeric(
                        subject_visits[days_col], errors='coerce'
                    ).max()
                    if pd.isna(max_days_outstanding):
                        max_days_outstanding = 0
        
        metrics['max_days_outstanding'] = max_days_outstanding
        
        if max_days_outstanding > self.config.max_days_outstanding:
            is_met = False
            days_over = max_days_outstanding - self.config.max_days_outstanding
            days_penalty = min(days_over / 30, 0.5)  # Max 50% penalty for days
            score = max(0, score - days_penalty)
            blockers.append(f"Visit overdue by {int(max_days_outstanding)} days (threshold: {self.config.max_days_outstanding})")
            severity = BlockerSeverity.CRITICAL if max_days_outstanding > 60 else BlockerSeverity.HIGH
        
        return CleanConditionResult(
            condition=CleanCondition.VISITS,
            is_met=is_met,
            score=score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0),
            weight=self.config.weight_visits,
            weighted_score=(score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0)) * self.config.weight_visits,
            blockers=blockers,
            metrics=metrics,
            severity=severity,
            source_files=source_files
        )
    
    def _evaluate_pages_condition(
        self,
        cpid_row: Optional[pd.Series],
        missing_pages_data: Optional[pd.DataFrame],
        subject_id: str
    ) -> CleanConditionResult:
        """
        Evaluate Pages condition:
        CPID_EDC_Metrics["Missing Page"] == 0 AND
        Global_Missing_Pages_Report returns zero rows for this subject
        """
        blockers = []
        metrics = {}
        score = 1.0
        is_met = True
        severity = BlockerSeverity.INFORMATIONAL
        source_files = ['CPID_EDC_Metrics']
        
        # Check missing pages from CPID
        missing_pages = 0
        if cpid_row is not None:
            mp_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['missing_pages'])
            if mp_col:
                missing_pages = self._safe_numeric(cpid_row, mp_col, 0)
        
        metrics['missing_pages_cpid'] = missing_pages
        
        if missing_pages > 0:
            is_met = False
            score = max(0, 1 - (missing_pages / 10))
            blockers.append(f"{int(missing_pages)} missing page(s) in CPID")
            severity = BlockerSeverity.HIGH if missing_pages > 5 else BlockerSeverity.MEDIUM
        
        # Check Global_Missing_Pages_Report
        missing_pages_report = 0
        if missing_pages_data is not None and len(missing_pages_data) > 0:
            source_files.append('Global_Missing_Pages_Report')
            subj_col = self._find_column(missing_pages_data, self.COLUMN_MAPS['subject_id'], 'missing_pages')
            if subj_col:
                subject_missing = missing_pages_data[
                    missing_pages_data[subj_col].astype(str).str.contains(str(subject_id), na=False)
                ]
                missing_pages_report = len(subject_missing)
        
        metrics['missing_pages_report'] = missing_pages_report
        
        if missing_pages_report > 0:
            is_met = False
            score = max(0, score - 0.3)  # Additional penalty
            blockers.append(f"{missing_pages_report} row(s) in Missing Pages Report")
        
        return CleanConditionResult(
            condition=CleanCondition.PAGES,
            is_met=is_met,
            score=score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0),
            weight=self.config.weight_pages,
            weighted_score=(score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0)) * self.config.weight_pages,
            blockers=blockers,
            metrics=metrics,
            severity=severity,
            source_files=source_files
        )
    
    def _evaluate_queries_condition(
        self,
        cpid_row: Optional[pd.Series]
    ) -> CleanConditionResult:
        """
        Evaluate Queries condition:
        CPID_EDC_Metrics["# Open Queries"] == 0 AND
        CPID_EDC_Metrics["Queries status"] == "Closed"
        """
        blockers = []
        metrics = {}
        score = 1.0
        is_met = True
        severity = BlockerSeverity.INFORMATIONAL
        source_files = ['CPID_EDC_Metrics']
        
        open_queries = 0
        query_status = "Unknown"
        
        if cpid_row is not None:
            oq_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['open_queries'])
            if oq_col:
                open_queries = self._safe_numeric(cpid_row, oq_col, 0)
            
            qs_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['query_status'])
            if qs_col:
                query_status = str(self._safe_get(cpid_row, qs_col, "Unknown"))
        
        metrics['open_queries'] = open_queries
        metrics['query_status'] = query_status
        
        if open_queries > self.config.max_open_queries:
            is_met = False
            score = max(0, 1 - (open_queries / 20))
            blockers.append(f"{int(open_queries)} open query(ies)")
            severity = BlockerSeverity.HIGH if open_queries > 10 else BlockerSeverity.MEDIUM
        
        if query_status.lower() not in ['closed', 'completed', 'resolved', 'unknown']:
            if open_queries > 0:  # Only flag if there are queries
                is_met = False
                score = max(0, score - 0.2)
                blockers.append(f"Query status is '{query_status}' (expected: Closed)")
        
        return CleanConditionResult(
            condition=CleanCondition.QUERIES,
            is_met=is_met,
            score=score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0),
            weight=self.config.weight_queries,
            weighted_score=(score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0)) * self.config.weight_queries,
            blockers=blockers,
            metrics=metrics,
            severity=severity,
            source_files=source_files
        )
    
    def _evaluate_coding_condition(
        self,
        cpid_row: Optional[pd.Series]
    ) -> CleanConditionResult:
        """
        Evaluate Coding condition:
        CPID_EDC_Metrics["# Uncoded Terms"] == 0 AND
        CPID_EDC_Metrics["# Coded terms"] > 0 (ensuring coding has occurred)
        """
        blockers = []
        metrics = {}
        score = 1.0
        is_met = True
        severity = BlockerSeverity.INFORMATIONAL
        source_files = ['CPID_EDC_Metrics']
        
        uncoded_terms = 0
        coded_terms = 0
        
        if cpid_row is not None:
            ut_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['uncoded_terms'])
            if ut_col:
                uncoded_terms = self._safe_numeric(cpid_row, ut_col, 0)
            
            ct_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['coded_terms'])
            if ct_col:
                coded_terms = self._safe_numeric(cpid_row, ct_col, 0)
        
        metrics['uncoded_terms'] = uncoded_terms
        metrics['coded_terms'] = coded_terms
        
        if uncoded_terms > self.config.max_uncoded_terms:
            is_met = False
            score = max(0, 1 - (uncoded_terms / 10))
            blockers.append(f"{int(uncoded_terms)} uncoded term(s) requiring medical coding")
            severity = BlockerSeverity.MEDIUM
        
        # Check that coding has actually occurred (if terms exist)
        total_terms = uncoded_terms + coded_terms
        if total_terms > 0 and coded_terms == 0:
            is_met = False
            score = max(0, score - 0.3)
            blockers.append("No terms have been coded (coding not started)")
            severity = BlockerSeverity.HIGH
        
        metrics['coding_completion_pct'] = (coded_terms / total_terms * 100) if total_terms > 0 else 100
        
        return CleanConditionResult(
            condition=CleanCondition.CODING,
            is_met=is_met,
            score=score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0),
            weight=self.config.weight_coding,
            weighted_score=(score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0)) * self.config.weight_coding,
            blockers=blockers,
            metrics=metrics,
            severity=severity,
            source_files=source_files
        )
    
    def _evaluate_reconciliation_condition(
        self,
        cpid_row: Optional[pd.Series]
    ) -> CleanConditionResult:
        """
        Evaluate Reconciliation condition:
        CPID_EDC_Metrics["# Reconciliation Issues"] == 0
        """
        blockers = []
        metrics = {}
        score = 1.0
        is_met = True
        severity = BlockerSeverity.INFORMATIONAL
        source_files = ['CPID_EDC_Metrics']
        
        recon_issues = 0
        
        if cpid_row is not None:
            ri_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['reconciliation_issues'])
            if ri_col:
                recon_issues = self._safe_numeric(cpid_row, ri_col, 0)
        
        metrics['reconciliation_issues'] = recon_issues
        
        if recon_issues > self.config.max_reconciliation_issues:
            is_met = False
            score = max(0, 1 - (recon_issues / 5))
            blockers.append(f"{int(recon_issues)} EDC/Safety reconciliation issue(s)")
            severity = BlockerSeverity.CRITICAL  # Reconciliation is always critical
        
        return CleanConditionResult(
            condition=CleanCondition.RECONCILIATION,
            is_met=is_met,
            score=score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0),
            weight=self.config.weight_reconciliation,
            weighted_score=(score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0)) * self.config.weight_reconciliation,
            blockers=blockers,
            metrics=metrics,
            severity=severity,
            source_files=source_files
        )
    
    def _evaluate_safety_condition(
        self,
        cpid_row: Optional[pd.Series],
        sae_data: Optional[pd.DataFrame],
        subject_id: str
    ) -> CleanConditionResult:
        """
        Evaluate Safety condition:
        CPID_EDC_Metrics["eSAE Review"] MATCHES SAE Dashboard (Consistency check)
        """
        blockers = []
        metrics = {}
        score = 1.0
        is_met = True
        severity = BlockerSeverity.INFORMATIONAL
        source_files = ['CPID_EDC_Metrics']
        
        esae_review_count = 0
        sae_pending_count = 0
        
        if cpid_row is not None:
            er_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['esae_review'])
            if er_col:
                esae_review_count = self._safe_numeric(cpid_row, er_col, 0)
        
        metrics['esae_review_cpid'] = esae_review_count
        
        # Check SAE Dashboard for pending reviews
        if sae_data is not None and len(sae_data) > 0:
            source_files.append('SAE_Dashboard')
            subj_col = self._find_column(sae_data, self.COLUMN_MAPS['subject_id'], 'sae')
            
            if subj_col:
                subject_saes = sae_data[
                    sae_data[subj_col].astype(str).str.contains(str(subject_id), na=False)
                ]
                
                if len(subject_saes) > 0:
                    # Look for pending status in any column
                    for col in subject_saes.columns:
                        if 'status' in col.lower() or 'review' in col.lower():
                            pending = subject_saes[
                                subject_saes[col].astype(str).str.lower().str.contains('pending|open|incomplete', na=False)
                            ]
                            sae_pending_count += len(pending)
        
        metrics['sae_pending_count'] = sae_pending_count
        
        # Check for consistency
        if esae_review_count > 0:
            is_met = False
            score = max(0, 1 - (esae_review_count / 5))
            blockers.append(f"{int(esae_review_count)} SAE(s) pending DM review (from CPID)")
            severity = BlockerSeverity.CRITICAL
        
        if sae_pending_count > 0:
            is_met = False
            score = max(0, score - 0.2)
            blockers.append(f"{sae_pending_count} SAE(s) pending in SAE Dashboard")
            severity = BlockerSeverity.CRITICAL
        
        return CleanConditionResult(
            condition=CleanCondition.SAFETY,
            is_met=is_met,
            score=score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0),
            weight=self.config.weight_safety,
            weighted_score=(score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0)) * self.config.weight_safety,
            blockers=blockers,
            metrics=metrics,
            severity=severity,
            source_files=source_files
        )
    
    def _evaluate_verification_condition(
        self,
        cpid_row: Optional[pd.Series]
    ) -> CleanConditionResult:
        """
        Evaluate Verification condition:
        CPID_EDC_Metrics["Data Verification %"] == 100% AND
        CPID_EDC_Metrics["# Forms Verified"] == CPID_EDC_Metrics["# Expected Visits"]
        """
        blockers = []
        metrics = {}
        score = 1.0
        is_met = True
        severity = BlockerSeverity.INFORMATIONAL
        source_files = ['CPID_EDC_Metrics']
        
        verification_pct = 100.0
        forms_verified = 0
        expected_visits = 0
        
        if cpid_row is not None:
            vp_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['verification_pct'])
            if vp_col:
                verification_pct = self._safe_numeric(cpid_row, vp_col, 100.0)
            
            fv_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['forms_verified'])
            if fv_col:
                forms_verified = self._safe_numeric(cpid_row, fv_col, 0)
            
            ev_col = self._find_column(pd.DataFrame([cpid_row]), self.COLUMN_MAPS['expected_visits'])
            if ev_col:
                expected_visits = self._safe_numeric(cpid_row, ev_col, 0)
        
        metrics['verification_pct'] = verification_pct
        metrics['forms_verified'] = forms_verified
        metrics['expected_visits'] = expected_visits
        
        if verification_pct < self.config.min_verification_pct:
            is_met = False
            score = verification_pct / 100
            blockers.append(f"Data verification at {verification_pct:.1f}% (required: {self.config.min_verification_pct}%)")
            severity = BlockerSeverity.MEDIUM if verification_pct >= 75 else BlockerSeverity.HIGH
        
        if expected_visits > 0 and forms_verified < expected_visits:
            is_met = False
            missing_forms = expected_visits - forms_verified
            score = max(0, score - (missing_forms / expected_visits * 0.5))
            blockers.append(f"{int(forms_verified)}/{int(expected_visits)} forms verified")
            if severity == BlockerSeverity.INFORMATIONAL:
                severity = BlockerSeverity.MEDIUM
        
        return CleanConditionResult(
            condition=CleanCondition.VERIFICATION,
            is_met=is_met,
            score=score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0),
            weight=self.config.weight_verification,
            weighted_score=(score if self.config.enable_partial_scoring else (1.0 if is_met else 0.0)) * self.config.weight_verification,
            blockers=blockers,
            metrics=metrics,
            severity=severity,
            source_files=source_files
        )
    
    def calculate_batch(
        self,
        subject_ids: List[str] = None
    ) -> Dict[str, CleanPatientStatus]:
        """
        Calculate Clean Patient Status for multiple subjects
        
        Args:
            subject_ids: List of subject IDs. If None, calculates for all subjects in CPID.
            
        Returns:
            Dictionary of subject_id -> CleanPatientStatus
        """
        results = {}
        
        cpid_df = self.data_sources.get('cpid')
        if cpid_df is None:
            logger.warning("No CPID data available for batch calculation")
            return results
        
        subj_col = self._find_column(cpid_df, self.COLUMN_MAPS['subject_id'], 'cpid')
        if not subj_col:
            logger.warning("Could not find subject ID column in CPID data")
            return results
        
        if subject_ids is None:
            subject_ids = cpid_df[subj_col].dropna().unique().tolist()
        
        logger.info(f"Calculating Clean Patient Status for {len(subject_ids)} subjects")
        
        for subject_id in subject_ids:
            try:
                # Find subject row
                matches = cpid_df[cpid_df[subj_col].astype(str) == str(subject_id)]
                if len(matches) > 0:
                    cpid_row = matches.iloc[0]
                    status = self.calculate_status(str(subject_id), cpid_row)
                    results[str(subject_id)] = status
            except Exception as e:
                logger.error(f"Error calculating status for {subject_id}: {e}")
        
        return results
    
    def get_study_summary(
        self,
        statuses: Dict[str, CleanPatientStatus] = None
    ) -> Dict[str, Any]:
        """
        Generate study-level summary of Clean Patient Status
        
        Returns summary statistics and distribution
        """
        if statuses is None:
            statuses = self.calculate_batch()
        
        if not statuses:
            return {'error': 'No statuses calculated'}
        
        total = len(statuses)
        clean_count = sum(1 for s in statuses.values() if s.is_clean)
        
        # Calculate distribution buckets
        distribution = {
            '100% Clean': 0,
            '90-99%': 0,
            '75-89%': 0,
            '50-74%': 0,
            '<50%': 0
        }
        
        for status in statuses.values():
            pct = status.clean_percentage
            if pct >= 100:
                distribution['100% Clean'] += 1
            elif pct >= 90:
                distribution['90-99%'] += 1
            elif pct >= 75:
                distribution['75-89%'] += 1
            elif pct >= 50:
                distribution['50-74%'] += 1
            else:
                distribution['<50%'] += 1
        
        # Blocker analysis
        blocker_counts = {}
        for status in statuses.values():
            for condition in status.conditions:
                if not condition.is_met:
                    cond_name = condition.condition.value
                    blocker_counts[cond_name] = blocker_counts.get(cond_name, 0) + 1
        
        # Sort blockers by frequency
        sorted_blockers = sorted(blocker_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_subjects': total,
            'clean_count': clean_count,
            'clean_percentage': (clean_count / total * 100) if total > 0 else 0,
            'distribution': distribution,
            'top_blockers': sorted_blockers[:5],
            'average_clean_pct': np.mean([s.clean_percentage for s in statuses.values()]),
            'median_clean_pct': np.median([s.clean_percentage for s in statuses.values()]),
            'calculated_at': datetime.now().isoformat()
        }


class QualityCockpitVisualizer:
    """
    Visualization components for the Quality Cockpit
    
    Creates interactive visualizations for:
    - Clean Patient Progress Bars
    - Condition breakdown charts
    - Study-level dashboards
    """
    
    COLORS = {
        'clean': '#00CC00',
        'almost_clean': '#7FD77F',
        'warning': '#FFCC00',
        'concern': '#FF6600',
        'critical': '#FF0000',
        'neutral': '#CCCCCC'
    }
    
    def __init__(self):
        self.figures = {}
    
    def create_progress_bar_html(self, status: CleanPatientStatus) -> str:
        """
        Generate HTML for a Clean Patient Progress Bar
        
        Creates a visual progress bar showing:
        - Overall percentage
        - Color-coded status
        - Blocking item indicator
        """
        progress_data = status.get_progress_bar_data()
        pct = progress_data['percentage']
        color = progress_data['color']
        text = progress_data['status_text']
        
        html = f'''
        <div class="patient-card" style="background: var(--bg-tertiary); border-radius: 16px; padding: 20px; margin-bottom: 16px; border: 1px solid var(--border-color); transition: all 0.3s ease;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="width: 40px; height: 40px; background: var(--primary-glow); border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-user" style="color: var(--primary);"></i>
                    </div>
                    <span style="font-weight: 600; color: var(--text-primary);">{status.subject_id}</span>
                </div>
                <span style="color: {color}; font-weight: 600; background: {'rgba(0,204,0,0.1)' if pct >= 100 else 'rgba(255,102,0,0.1)'}; padding: 6px 14px; border-radius: 20px; font-size: 13px;">
                    {text}
                </span>
            </div>
            <div style="background: var(--bg-primary); border-radius: 10px; height: 12px; overflow: hidden; margin-bottom: 12px;">
                <div style="background: linear-gradient(90deg, {color}, {color}dd); height: 100%; width: {min(pct, 100):.1f}%; transition: width 0.5s ease; border-radius: 10px;"></div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        '''
        
        # Add condition indicators
        for segment in progress_data['segments']:
            seg_color = segment['color']
            seg_name = segment['name']
            check = '✓' if segment['is_met'] else '✗'
            bg_color = 'rgba(0,204,0,0.15)' if segment['is_met'] else 'rgba(255,102,0,0.15)'
            html += f'''
                <span style="background: {bg_color}; color: {seg_color}; padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 500; border: 1px solid {seg_color}33;">
                    {check} {seg_name}
                </span>
            '''
        
        html += '''
            </div>
        </div>
        '''
        
        return html
    
    def create_study_dashboard_html(
        self,
        statuses: Dict[str, CleanPatientStatus],
        summary: Dict[str, Any]
    ) -> str:
        """Generate complete HTML dashboard for study"""
        
        clean_rate = summary.get('clean_percentage', 0)
        status_color = '#10b981' if clean_rate >= 90 else '#f59e0b' if clean_rate >= 70 else '#ef4444'
        
        html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Cockpit - Clean Patient Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #2563eb;
            --primary-glow: rgba(37, 99, 235, 0.15);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.15);
            --warning: #f59e0b;
            --warning-glow: rgba(245, 158, 11, 0.15);
            --danger: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.15);
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --gradient-primary: linear-gradient(135deg, #2563eb, #3b82f6);
            --gradient-success: linear-gradient(135deg, #10b981, #34d399);
            --transition-fast: 0.2s ease;
        }}
        
        [data-theme="dark"] {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --border-color: #475569;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }}
        
        .header {{
            background: var(--gradient-primary);
            color: white;
            padding: 32px 40px;
            border-radius: 20px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 10px 40px rgba(37, 99, 235, 0.3);
        }}
        
        .header-title {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        
        .header-logo {{
            width: 56px;
            height: 56px;
            background: rgba(255,255,255,0.2);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }}
        
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
        }}
        
        .header p {{
            opacity: 0.85;
            font-size: 14px;
            margin-top: 4px;
        }}
        
        .theme-toggle {{
            background: rgba(255,255,255,0.2);
            border: none;
            width: 44px;
            height: 44px;
            border-radius: 12px;
            color: white;
            cursor: pointer;
            font-size: 18px;
            transition: all var(--transition-fast);
        }}
        
        .theme-toggle:hover {{
            background: rgba(255,255,255,0.3);
            transform: rotate(20deg);
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        
        .metric-card {{
            background: var(--bg-primary);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        }}
        
        .metric-icon {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            margin-bottom: 16px;
        }}
        
        .metric-icon.blue {{ background: var(--primary-glow); color: var(--primary); }}
        .metric-icon.green {{ background: var(--success-glow); color: var(--success); }}
        .metric-icon.orange {{ background: var(--warning-glow); color: var(--warning); }}
        .metric-icon.red {{ background: var(--danger-glow); color: var(--danger); }}
        
        .metric-value {{
            font-size: 36px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 4px;
        }}
        
        .metric-label {{
            color: var(--text-muted);
            font-size: 14px;
            font-weight: 500;
        }}
        
        .section {{
            background: var(--bg-primary);
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 24px;
            border: 1px solid var(--border-color);
        }}
        
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .section-title i {{
            color: var(--primary);
        }}
        
        .blocker-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }}
        
        .blocker-tag {{
            background: var(--warning-glow);
            border: 1px solid var(--warning);
            color: var(--warning);
            padding: 10px 18px;
            border-radius: 10px;
            font-weight: 500;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all var(--transition-fast);
        }}
        
        .blocker-tag:hover {{
            transform: scale(1.02);
        }}
        
        .search-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 10px 16px;
            width: 280px;
        }}
        
        .search-bar input {{
            border: none;
            background: transparent;
            outline: none;
            width: 100%;
            color: var(--text-primary);
            font-size: 14px;
        }}
        
        .search-bar i {{
            color: var(--text-muted);
        }}
        
        .footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-size: 13px;
        }}
        
        @media (max-width: 768px) {{
            .header {{
                flex-direction: column;
                gap: 16px;
                text-align: center;
            }}
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .search-bar {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <header class="header animate__animated animate__fadeIn">
            <div class="header-title">
                <div class="header-logo">
                    <i class="fas fa-shield-alt"></i>
                </div>
                <div>
                    <h1>Quality Cockpit</h1>
                    <p>Clean Patient Status Dashboard | {datetime.now().strftime('%B %d, %Y')}</p>
                </div>
            </div>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle Dark Mode">
                <i class="fas fa-moon"></i>
            </button>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card animate__animated animate__fadeInUp">
                <div class="metric-icon green"><i class="fas fa-user-check"></i></div>
                <div class="metric-value" style="color: var(--success);">{summary['clean_count']}</div>
                <div class="metric-label">Clean Patients</div>
            </div>
            <div class="metric-card animate__animated animate__fadeInUp" style="animation-delay: 0.1s;">
                <div class="metric-icon blue"><i class="fas fa-users"></i></div>
                <div class="metric-value">{summary['total_subjects']}</div>
                <div class="metric-label">Total Patients</div>
            </div>
            <div class="metric-card animate__animated animate__fadeInUp" style="animation-delay: 0.2s;">
                <div class="metric-icon {'green' if clean_rate >= 90 else 'orange' if clean_rate >= 70 else 'red'}">
                    <i class="fas fa-percentage"></i>
                </div>
                <div class="metric-value" style="color: {status_color};">{clean_rate:.1f}%</div>
                <div class="metric-label">Clean Rate</div>
            </div>
            <div class="metric-card animate__animated animate__fadeInUp" style="animation-delay: 0.3s;">
                <div class="metric-icon orange"><i class="fas fa-chart-line"></i></div>
                <div class="metric-value">{summary['average_clean_pct']:.1f}%</div>
                <div class="metric-label">Avg. Progress</div>
            </div>
        </div>
        
        <div class="section animate__animated animate__fadeIn">
            <div class="section-header">
                <h2 class="section-title"><i class="fas fa-exclamation-triangle"></i> Top Blockers</h2>
                <span style="color: var(--text-muted); font-size: 14px;">{len(summary.get('top_blockers', []))} categories</span>
            </div>
            <div class="blocker-grid">
        '''
        
        for blocker, count in summary.get('top_blockers', []):
            html += f'''
                <div class="blocker-tag">
                    <i class="fas fa-ban"></i>
                    {blocker.title()}: <strong>{count}</strong> patients
                </div>
            '''
        
        html += '''
            </div>
        </div>
        
        <div class="section animate__animated animate__fadeIn">
            <div class="section-header">
                <h2 class="section-title"><i class="fas fa-clipboard-list"></i> Patient Progress</h2>
                <div class="search-bar">
                    <i class="fas fa-search"></i>
                    <input type="text" placeholder="Search patients..." onkeyup="filterPatients(this.value)">
                </div>
            </div>
            <div id="patient-list">
        '''
        
        # Sort by clean percentage descending
        sorted_statuses = sorted(statuses.values(), key=lambda x: x.clean_percentage, reverse=True)
        
        for status in sorted_statuses[:50]:  # Limit to 50 for performance
            html += self.create_progress_bar_html(status)
        
        html += f'''
            </div>
            {f'<p style="text-align: center; color: var(--text-muted); margin-top: 20px;">Showing 50 of {len(statuses)} patients</p>' if len(statuses) > 50 else ''}
        </div>
        
        <footer class="footer">
            <p><i class="fas fa-shield-alt" style="margin-right: 8px;"></i>Quality Cockpit - Clinical Data Intelligence</p>
            <p style="margin-top: 4px;">Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </footer>
    </div>
    
    <script>
        function toggleTheme() {{
            const body = document.body;
            const currentTheme = body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            body.setAttribute('data-theme', newTheme);
            localStorage.setItem('cockpit-theme', newTheme);
            const icon = document.querySelector('.theme-toggle i');
            icon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }}
        
        // Load saved theme
        const savedTheme = localStorage.getItem('cockpit-theme') || 'light';
        if (savedTheme === 'dark') {{
            document.body.setAttribute('data-theme', 'dark');
            document.querySelector('.theme-toggle i').className = 'fas fa-sun';
        }}
        
        function filterPatients(query) {{
            const q = query.toLowerCase();
            document.querySelectorAll('.patient-card').forEach(card => {{
                const text = card.textContent.toLowerCase();
                card.style.display = text.includes(q) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>
        '''
        
        return html
    
    def create_condition_breakdown_chart(
        self,
        statuses: Dict[str, CleanPatientStatus]
    ) -> Dict[str, Any]:
        """
        Create data for condition breakdown visualization
        
        Returns data structure for plotting condition pass/fail rates
        """
        condition_stats = {cond.value: {'passed': 0, 'failed': 0} for cond in CleanCondition}
        
        for status in statuses.values():
            for condition in status.conditions:
                cond_name = condition.condition.value
                if condition.is_met:
                    condition_stats[cond_name]['passed'] += 1
                else:
                    condition_stats[cond_name]['failed'] += 1
        
        return {
            'conditions': list(condition_stats.keys()),
            'passed': [stats['passed'] for stats in condition_stats.values()],
            'failed': [stats['failed'] for stats in condition_stats.values()],
            'pass_rates': [
                stats['passed'] / (stats['passed'] + stats['failed']) * 100
                if (stats['passed'] + stats['failed']) > 0 else 0
                for stats in condition_stats.values()
            ]
        }


# Export convenience function
def calculate_clean_patient_status(
    subject_id: str,
    data_sources: Dict[str, pd.DataFrame],
    config: QualityCockpitConfig = None
) -> CleanPatientStatus:
    """
    Convenience function to calculate Clean Patient Status
    
    Args:
        subject_id: Subject identifier
        data_sources: Dictionary of data source name -> DataFrame
        config: Optional configuration
        
    Returns:
        CleanPatientStatus object
    """
    calculator = CleanPatientStatusCalculator(config)
    calculator.load_data(data_sources)
    return calculator.calculate_status(subject_id)
