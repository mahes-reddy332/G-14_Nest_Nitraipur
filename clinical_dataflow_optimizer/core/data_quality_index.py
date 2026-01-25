r"""
Data Quality Index (DQI) - Weighted Penalization Model
=======================================================

This module implements the TransCelerate RACT-inspired Data Quality Index,
a composite score (0-100) assessing overall reliability of data at Site or Patient level.

Mathematical Foundation:
------------------------
For any given site $i \in S$ (the set of all sites), the DQI is calculated as:

$$DQI_i = 100 - \left( W_{visit} \cdot f(M_{visit}) + W_{query} \cdot f(M_{query}) + 
                        W_{conform} \cdot f(M_{conform}) + W_{safety} \cdot f(M_{safety}) \right)$$

Where:
- $M_{visit}$ (Visit Adherence): Normalized count of Missing Visits and Days Outstanding
  $W_{visit} = 0.2$ (20% impact)
  
- $M_{query}$ (Responsiveness): Function of CRFs overdue and Total Queries
  $W_{query} = 0.2$ (20% impact)
  
- $M_{conform}$ (Conformance): Derived from Non-Conformant Pages and Inactivated Forms
  $W_{conform} = 0.2$ (20% impact)
  
- $M_{safety}$ (Safety Criticality): SAE Dashboard discrepancies and Missing Lab Names
  $W_{safety} = 0.4$ (40% impact) - Highest due to patient safety implications

Interpretation Thresholds:
- DQI > 90 (Green): Site performing well. Low touch monitoring required.
- DQI 75-90 (Yellow): Warning signs. Targeted monitoring by "Lia" Agent.
- DQI < 75 (Red): Critical failure. Immediate onsite audit required.

Visualization Dashboard UX Patterns:
1. Global Map View (Executive Level): Heatmap by DQI Score
2. Site Detail View (CRA Level): Deviation vs Enrollment scatter plot
3. Patient Detail View (DM Level): Visual timeline with Clean Status flags
4. Operational Backlog (Sankey): Query flow from Opened → Answered → Closed
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import json
import math

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DQILevel(Enum):
    """DQI interpretation levels"""
    GREEN = "green"      # DQI > 90: Low touch monitoring
    YELLOW = "yellow"    # DQI 75-90: Targeted intervention
    RED = "red"          # DQI < 75: Critical failure


class MetricCategory(Enum):
    """DQI metric categories"""
    VISIT_ADHERENCE = "visit"
    QUERY_RESPONSIVENESS = "query"
    DATA_CONFORMANCE = "conform"
    SAFETY_CRITICALITY = "safety"


class RiskQuadrant(Enum):
    """Site risk quadrants for scatter plot"""
    LOW_VOLUME_LOW_RISK = "Low Volume / Low Risk"
    LOW_VOLUME_HIGH_RISK = "Low Volume / High Risk"
    HIGH_VOLUME_LOW_RISK = "High Volume / Low Risk"
    HIGH_VOLUME_HIGH_RISK = "High Volume / High Risk"  # Greatest threat


class QueryFlowStage(Enum):
    """Query lifecycle stages for Sankey diagram"""
    OPENED = "Opened"
    ANSWERED = "Answered"
    CLOSED = "Closed"
    OVERDUE = "Overdue"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DQIConfig:
    """Configuration for DQI calculations following TransCelerate RACT methodology"""
    # Weights (must sum to 1.0)
    weight_visit: float = 0.20      # Visit Adherence (20%)
    weight_query: float = 0.20      # Query Responsiveness (20%)
    weight_conform: float = 0.20    # Data Conformance (20%)
    weight_safety: float = 0.40     # Safety Criticality (40%)
    
    # Thresholds for normalization
    max_missing_visits_penalty: int = 10       # Maximum missing visits for full penalty
    max_days_outstanding_penalty: int = 90     # Maximum days for full penalty
    max_queries_penalty: int = 50              # Maximum queries for full penalty
    query_response_days_threshold: int = 14    # Days before query considered overdue
    max_nonconformant_pages_penalty: int = 20  # Maximum non-conformant pages for full penalty
    max_sae_discrepancies_penalty: int = 5     # Maximum SAE discrepancies for full penalty
    
    # DQI Level thresholds
    green_threshold: float = 90.0   # DQI > 90 = Green
    yellow_threshold: float = 75.0  # DQI 75-90 = Yellow, < 75 = Red
    
    # Scatter plot thresholds
    high_enrollment_threshold: int = 10   # Patients above this = high volume
    high_deviation_threshold: float = 5   # Deviations above this = high risk
    
    def validate(self) -> bool:
        """Validate configuration weights sum to 1.0"""
        total = self.weight_visit + self.weight_query + self.weight_conform + self.weight_safety
        return abs(total - 1.0) < 0.001
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Return weights as dictionary"""
        return {
            MetricCategory.VISIT_ADHERENCE.value: self.weight_visit,
            MetricCategory.QUERY_RESPONSIVENESS.value: self.weight_query,
            MetricCategory.DATA_CONFORMANCE.value: self.weight_conform,
            MetricCategory.SAFETY_CRITICALITY.value: self.weight_safety
        }


@dataclass
class MetricPenalty:
    """Individual metric penalty calculation result"""
    category: MetricCategory
    raw_value: float           # Raw metric value
    normalized_penalty: float  # Normalized penalty (0-100)
    weight: float              # Category weight
    weighted_penalty: float    # normalized_penalty * weight
    components: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'category': self.category.value,
            'raw_value': round(self.raw_value, 2),
            'normalized_penalty': round(self.normalized_penalty, 2),
            'weight': self.weight,
            'weighted_penalty': round(self.weighted_penalty, 2),
            'components': self.components,
            'description': self.description
        }


@dataclass
class DQIResult:
    """Complete DQI calculation result for a site or patient"""
    entity_id: str              # Site ID or Patient ID
    entity_type: str            # "site" or "patient"
    dqi_score: float            # Final DQI score (0-100)
    level: DQILevel             # Interpretation level
    total_penalty: float        # Sum of all weighted penalties
    penalties: List[MetricPenalty] = field(default_factory=list)
    recommendation: str = ""
    calculated_at: datetime = field(default_factory=datetime.now)
    data_sources_used: List[str] = field(default_factory=list)
    
    @property
    def level_color(self) -> str:
        """Return color code for DQI level"""
        colors = {
            DQILevel.GREEN: "#00CC00",
            DQILevel.YELLOW: "#FFCC00",
            DQILevel.RED: "#FF0000"
        }
        return colors.get(self.level, "#808080")
    
    @property
    def level_description(self) -> str:
        """Return description for DQI level"""
        descriptions = {
            DQILevel.GREEN: "Site performing well. Low touch monitoring required.",
            DQILevel.YELLOW: "Warning signs detected. Targeted monitoring intervention recommended.",
            DQILevel.RED: "Critical failure. Immediate onsite audit or rescue intervention required."
        }
        return descriptions.get(self.level, "Unknown status")
    
    def to_dict(self) -> Dict:
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'dqi_score': round(self.dqi_score, 2),
            'level': self.level.value,
            'level_color': self.level_color,
            'level_description': self.level_description,
            'total_penalty': round(self.total_penalty, 2),
            'penalties': [p.to_dict() for p in self.penalties],
            'recommendation': self.recommendation,
            'calculated_at': self.calculated_at.isoformat(),
            'data_sources_used': self.data_sources_used
        }


@dataclass
class SiteRiskProfile:
    """Site risk profile for scatter plot visualization"""
    site_id: str
    enrollment_count: int       # Number of enrolled patients
    deviation_count: float      # Number of deviations
    dqi_score: float
    quadrant: RiskQuadrant
    risk_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'enrollment_count': self.enrollment_count,
            'deviation_count': self.deviation_count,
            'dqi_score': round(self.dqi_score, 2),
            'quadrant': self.quadrant.value,
            'risk_factors': self.risk_factors
        }


@dataclass
class QueryFlowData:
    """Query flow data for Sankey diagram"""
    site_id: str
    opened_count: int
    answered_count: int
    closed_count: int
    overdue_count: int
    bottleneck: str  # "site" or "dm" or "none"
    bottleneck_description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'opened': self.opened_count,
            'answered': self.answered_count,
            'closed': self.closed_count,
            'overdue': self.overdue_count,
            'bottleneck': self.bottleneck,
            'bottleneck_description': self.bottleneck_description
        }


@dataclass
class PatientTimelineEvent:
    """Patient timeline event for visual timeline"""
    subject_id: str
    visit_name: str
    visit_date: Optional[datetime]
    status: str  # "completed", "missing", "blocked", "upcoming"
    blockers: List[str] = field(default_factory=list)
    clean_status: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'visit_name': self.visit_name,
            'visit_date': self.visit_date.isoformat() if self.visit_date else None,
            'status': self.status,
            'blockers': self.blockers,
            'clean_status': self.clean_status
        }


# Default configuration
DEFAULT_DQI_CONFIG = DQIConfig()


# =============================================================================
# DQI CALCULATOR
# =============================================================================

class DataQualityIndexCalculator:
    r"""
    Calculates Data Quality Index using Weighted Penalization Model
    
    The DQI formula:
    $$DQI_i = 100 - \left( W_{visit} \cdot f(M_{visit}) + W_{query} \cdot f(M_{query}) + 
                            W_{conform} \cdot f(M_{conform}) + W_{safety} \cdot f(M_{safety}) \right)$$
    """
    
    # Column mappings for standardization
    COLUMN_MAPS = {
        'site_id': ['Site ID', 'Site', 'SiteID', 'SITEID', 'site_id'],
        'subject_id': ['Subject ID', 'Subject', 'SubjectID', 'SUBJECTID', 'subject_id'],
        'missing_visits': ['Missing Visits', '# Missing Visits', 'MissingVisits', 'missing_visits'],
        'days_outstanding': ['# Days Outstanding', 'Days Outstanding', 'DaysOutstanding', 'days_outstanding'],
        'total_queries': ['# Total Queries', 'Total Queries', 'TotalQueries', 'total_queries'],
        'open_queries': ['# Open Queries', 'Open Queries', 'OpenQueries', 'open_queries'],
        'answered_queries': ['# Answered Queries', 'Answered Queries', 'answered_queries'],
        'closed_queries': ['# Closed Queries', 'Closed Queries', 'closed_queries'],
        'overdue_crfs': ['CRFs overdue for signs', '# CRFs Overdue', 'OverdueCRFs', 'overdue_crfs'],
        'time_lag': ['Time lag (Days)', 'Time Lag', 'TimeLag', 'time_lag'],
        'nonconformant_pages': ['# Pages with Non-Conformant data', 'Non-Conformant Pages', 'nonconformant_pages'],
        'inactivated_forms': ['# Inactivated Forms', 'Inactivated Forms', 'inactivated_forms'],
        'sae_discrepancies': ['# Reconciliation Issues', 'SAE Discrepancies', 'sae_discrepancies'],
        'missing_lab_name': ['Missing_Lab_Name', 'Missing Lab Name', 'missing_lab_name'],
        'enrollment_count': ['# Enrolled Patients', 'Enrollment', 'enrollment_count'],
    }
    
    def __init__(self, config: DQIConfig = None):
        """Initialize DQI Calculator with configuration"""
        self.config = config or DEFAULT_DQI_CONFIG
        if not self.config.validate():
            logger.warning("DQI Configuration weights do not sum to 1.0, normalizing...")
        
        self.data_sources: Dict[str, pd.DataFrame] = {}
        self._column_cache: Dict[str, Dict[str, str]] = {}
        logger.info("DataQualityIndexCalculator initialized")
    
    def load_data(self, data_sources: Dict[str, pd.DataFrame]) -> None:
        """Load data sources for DQI calculation"""
        self.data_sources = data_sources
        self._column_cache.clear()
        logger.info(f"Loaded {len(data_sources)} data sources")
    
    def _get_column(self, df: pd.DataFrame, col_type: str, df_name: str = "") -> Optional[str]:
        """Get standardized column name from DataFrame"""
        cache_key = f"{df_name}_{col_type}"
        if cache_key in self._column_cache:
            return self._column_cache[cache_key].get(col_type)
        
        possible_names = self.COLUMN_MAPS.get(col_type, [])
        for name in possible_names:
            if name in df.columns:
                if cache_key not in self._column_cache:
                    self._column_cache[cache_key] = {}
                self._column_cache[cache_key][col_type] = name
                return name
        return None
    
    def _normalize_penalty(self, value: float, max_value: float) -> float:
        """
        Normalize a penalty value to 0-100 scale
        Uses sigmoid-like function for smooth scaling
        """
        if max_value <= 0:
            return 0.0
        
        # Linear normalization with cap at 100
        normalized = min(100.0, (value / max_value) * 100.0)
        return max(0.0, normalized)
    
    def _calculate_visit_adherence_penalty(
        self,
        site_id: str,
        site_data: pd.DataFrame
    ) -> MetricPenalty:
        """
        Calculate Visit Adherence penalty ($M_{visit}$)
        
        Based on:
        - Missing Visits count (severe impact on statistical power)
        - Days Outstanding (time-based urgency)
        """
        missing_visits_col = self._get_column(site_data, 'missing_visits', 'cpid')
        days_outstanding_col = self._get_column(site_data, 'days_outstanding', 'cpid')
        
        total_missing_visits = 0
        total_days_outstanding = 0
        patient_count = len(site_data)
        
        if missing_visits_col and missing_visits_col in site_data.columns:
            total_missing_visits = pd.to_numeric(
                site_data[missing_visits_col], errors='coerce'
            ).fillna(0).sum()
        
        if days_outstanding_col and days_outstanding_col in site_data.columns:
            total_days_outstanding = pd.to_numeric(
                site_data[days_outstanding_col], errors='coerce'
            ).fillna(0).mean()
        
        # Combine penalties: missing visits + normalized days outstanding
        visit_penalty = self._normalize_penalty(
            total_missing_visits, 
            self.config.max_missing_visits_penalty * max(1, patient_count)
        )
        days_penalty = self._normalize_penalty(
            total_days_outstanding,
            self.config.max_days_outstanding_penalty
        )
        
        # Combined penalty (weighted average)
        raw_value = total_missing_visits + (total_days_outstanding / 10)
        normalized_penalty = (visit_penalty * 0.7 + days_penalty * 0.3)
        
        return MetricPenalty(
            category=MetricCategory.VISIT_ADHERENCE,
            raw_value=raw_value,
            normalized_penalty=normalized_penalty,
            weight=self.config.weight_visit,
            weighted_penalty=normalized_penalty * self.config.weight_visit,
            components={
                'total_missing_visits': int(total_missing_visits),
                'avg_days_outstanding': round(total_days_outstanding, 1),
                'patient_count': patient_count,
                'visit_penalty_pct': round(visit_penalty, 2),
                'days_penalty_pct': round(days_penalty, 2)
            },
            description=f"{int(total_missing_visits)} missing visits, {total_days_outstanding:.0f} avg days outstanding"
        )
    
    def _calculate_query_responsiveness_penalty(
        self,
        site_id: str,
        site_data: pd.DataFrame
    ) -> MetricPenalty:
        """
        Calculate Query Responsiveness penalty ($M_{query}$)
        
        Sites with many queries but quick responses are penalized less
        than sites that ignore queries.
        """
        total_queries_col = self._get_column(site_data, 'total_queries', 'cpid')
        open_queries_col = self._get_column(site_data, 'open_queries', 'cpid')
        overdue_col = self._get_column(site_data, 'overdue_crfs', 'cpid')
        time_lag_col = self._get_column(site_data, 'time_lag', 'cpid')
        
        total_queries = 0
        open_queries = 0
        overdue_crfs = 0
        avg_time_lag = 0
        
        if total_queries_col and total_queries_col in site_data.columns:
            total_queries = pd.to_numeric(
                site_data[total_queries_col], errors='coerce'
            ).fillna(0).sum()
        
        if open_queries_col and open_queries_col in site_data.columns:
            open_queries = pd.to_numeric(
                site_data[open_queries_col], errors='coerce'
            ).fillna(0).sum()
        
        if overdue_col and overdue_col in site_data.columns:
            overdue_crfs = pd.to_numeric(
                site_data[overdue_col], errors='coerce'
            ).fillna(0).sum()
        
        if time_lag_col and time_lag_col in site_data.columns:
            avg_time_lag = pd.to_numeric(
                site_data[time_lag_col], errors='coerce'
            ).fillna(0).mean()
        
        # Calculate response ratio (lower is better)
        response_ratio = open_queries / max(1, total_queries) if total_queries > 0 else 0
        
        # Penalty based on open queries ratio and time lag
        query_penalty = self._normalize_penalty(
            open_queries + overdue_crfs,
            self.config.max_queries_penalty
        )
        time_penalty = self._normalize_penalty(
            avg_time_lag,
            self.config.query_response_days_threshold * 2
        )
        
        raw_value = open_queries + overdue_crfs + (avg_time_lag / 7)
        normalized_penalty = (query_penalty * 0.6 + time_penalty * 0.4)
        
        return MetricPenalty(
            category=MetricCategory.QUERY_RESPONSIVENESS,
            raw_value=raw_value,
            normalized_penalty=normalized_penalty,
            weight=self.config.weight_query,
            weighted_penalty=normalized_penalty * self.config.weight_query,
            components={
                'total_queries': int(total_queries),
                'open_queries': int(open_queries),
                'overdue_crfs': int(overdue_crfs),
                'avg_time_lag_days': round(avg_time_lag, 1),
                'response_ratio': round(response_ratio, 2)
            },
            description=f"{int(open_queries)} open queries, {int(overdue_crfs)} overdue CRFs, {avg_time_lag:.0f}d avg lag"
        )
    
    def _calculate_conformance_penalty(
        self,
        site_id: str,
        site_data: pd.DataFrame
    ) -> MetricPenalty:
        """
        Calculate Data Conformance penalty ($M_{conform}$)
        
        High non-conformance suggests poor site training or potential misconduct.
        """
        nonconformant_col = self._get_column(site_data, 'nonconformant_pages', 'cpid')
        inactivated_col = self._get_column(site_data, 'inactivated_forms', 'cpid')
        
        nonconformant_pages = 0
        inactivated_forms = 0
        
        if nonconformant_col and nonconformant_col in site_data.columns:
            nonconformant_pages = pd.to_numeric(
                site_data[nonconformant_col], errors='coerce'
            ).fillna(0).sum()
        
        if inactivated_col and inactivated_col in site_data.columns:
            inactivated_forms = pd.to_numeric(
                site_data[inactivated_col], errors='coerce'
            ).fillna(0).sum()
        
        # Combined conformance penalty
        raw_value = nonconformant_pages + (inactivated_forms * 2)  # Inactivated forms weighted higher
        normalized_penalty = self._normalize_penalty(
            raw_value,
            self.config.max_nonconformant_pages_penalty * len(site_data)
        )
        
        return MetricPenalty(
            category=MetricCategory.DATA_CONFORMANCE,
            raw_value=raw_value,
            normalized_penalty=normalized_penalty,
            weight=self.config.weight_conform,
            weighted_penalty=normalized_penalty * self.config.weight_conform,
            components={
                'nonconformant_pages': int(nonconformant_pages),
                'inactivated_forms': int(inactivated_forms),
                'patient_count': len(site_data)
            },
            description=f"{int(nonconformant_pages)} non-conformant pages, {int(inactivated_forms)} inactivated forms"
        )
    
    def _calculate_safety_criticality_penalty(
        self,
        site_id: str,
        site_data: pd.DataFrame,
        sae_data: Optional[pd.DataFrame] = None
    ) -> MetricPenalty:
        """
        Calculate Safety Criticality penalty ($M_{safety}$)
        
        Highest weight (40%) due to direct patient safety implications.
        Based on SAE Dashboard discrepancies and Missing Lab Names.
        """
        recon_col = self._get_column(site_data, 'sae_discrepancies', 'cpid')
        
        sae_discrepancies = 0
        missing_lab_count = 0
        
        if recon_col and recon_col in site_data.columns:
            sae_discrepancies = pd.to_numeric(
                site_data[recon_col], errors='coerce'
            ).fillna(0).sum()
        
        # Check SAE data if available
        if sae_data is not None and not sae_data.empty:
            site_col = self._get_column(sae_data, 'site_id', 'sae')
            if site_col:
                site_sae = sae_data[sae_data[site_col] == site_id]
                missing_lab_col = self._get_column(site_sae, 'missing_lab_name', 'sae')
                if missing_lab_col and missing_lab_col in site_sae.columns:
                    missing_lab_count = site_sae[missing_lab_col].notna().sum()
        
        # Safety penalty is critical - even small numbers are significant
        raw_value = (sae_discrepancies * 5) + (missing_lab_count * 3)
        normalized_penalty = self._normalize_penalty(
            sae_discrepancies + missing_lab_count,
            self.config.max_sae_discrepancies_penalty
        )
        
        return MetricPenalty(
            category=MetricCategory.SAFETY_CRITICALITY,
            raw_value=raw_value,
            normalized_penalty=normalized_penalty,
            weight=self.config.weight_safety,
            weighted_penalty=normalized_penalty * self.config.weight_safety,
            components={
                'sae_discrepancies': int(sae_discrepancies),
                'missing_lab_names': int(missing_lab_count),
                'severity_multiplier': 'CRITICAL' if sae_discrepancies > 0 else 'Normal'
            },
            description=f"{int(sae_discrepancies)} SAE discrepancies, {int(missing_lab_count)} missing lab names"
        )
    
    def _determine_dqi_level(self, dqi_score: float) -> DQILevel:
        """Determine DQI interpretation level"""
        if dqi_score >= self.config.green_threshold:
            return DQILevel.GREEN
        elif dqi_score >= self.config.yellow_threshold:
            return DQILevel.YELLOW
        else:
            return DQILevel.RED
    
    def _generate_recommendation(self, dqi_result: DQIResult) -> str:
        """Generate actionable recommendation based on DQI result"""
        if dqi_result.level == DQILevel.GREEN:
            return "Continue routine monitoring. Site demonstrates strong data quality practices."
        
        # Find the worst penalty category
        worst_penalty = max(dqi_result.penalties, key=lambda p: p.weighted_penalty)
        
        recommendations = {
            MetricCategory.VISIT_ADHERENCE: (
                f"Address visit adherence issues. {worst_penalty.description}. "
                "Consider scheduling adherence training and implementing visit reminders."
            ),
            MetricCategory.QUERY_RESPONSIVENESS: (
                f"Improve query response times. {worst_penalty.description}. "
                "Escalate to Site Coordinator and implement daily query review process."
            ),
            MetricCategory.DATA_CONFORMANCE: (
                f"Data conformance issues detected. {worst_penalty.description}. "
                "Schedule re-training on EDC entry guidelines and protocol requirements."
            ),
            MetricCategory.SAFETY_CRITICALITY: (
                f"CRITICAL: Safety data issues require immediate attention. {worst_penalty.description}. "
                "Initiate immediate audit and reconciliation with SAE dashboard."
            )
        }
        
        base_rec = recommendations.get(worst_penalty.category, "Review site performance metrics.")
        
        if dqi_result.level == DQILevel.RED:
            return f"URGENT: {base_rec} Immediate onsite audit or rescue intervention required."
        else:
            return f"WARNING: {base_rec} Targeted monitoring by Lia Agent recommended."
    
    def calculate_site_dqi(
        self,
        site_id: str,
        cpid_data: Optional[pd.DataFrame] = None,
        sae_data: Optional[pd.DataFrame] = None
    ) -> DQIResult:
        """
        Calculate DQI for a specific site
        
        Args:
            site_id: Site identifier
            cpid_data: CPID EDC Metrics data (optional, uses loaded data if not provided)
            sae_data: SAE Dashboard data (optional)
            
        Returns:
            DQIResult with complete DQI assessment
        """
        data_sources_used = []
        
        # Get CPID data for site
        if cpid_data is None:
            cpid_data = self.data_sources.get('cpid')
        
        if cpid_data is None or cpid_data.empty:
            logger.warning(f"No CPID data available for site {site_id}")
            return DQIResult(
                entity_id=site_id,
                entity_type="site",
                dqi_score=0.0,
                level=DQILevel.RED,
                total_penalty=100.0,
                recommendation="No data available for DQI calculation."
            )
        
        # Filter for site
        site_col = self._get_column(cpid_data, 'site_id', 'cpid')
        if site_col:
            site_data = cpid_data[cpid_data[site_col] == site_id]
        else:
            # Assume all data is for this site
            site_data = cpid_data
        
        if site_data.empty:
            logger.warning(f"No data found for site {site_id}")
            return DQIResult(
                entity_id=site_id,
                entity_type="site",
                dqi_score=100.0,
                level=DQILevel.GREEN,
                total_penalty=0.0,
                recommendation="No patients enrolled at this site yet."
            )
        
        data_sources_used.append('CPID_EDC_Metrics')
        
        # Get SAE data if available
        if sae_data is None:
            sae_data = self.data_sources.get('esae')
            if sae_data is None:
                sae_data = self.data_sources.get('sae')
        
        if sae_data is not None and not sae_data.empty:
            data_sources_used.append('SAE_Dashboard')
        
        # Calculate all penalties
        penalties = [
            self._calculate_visit_adherence_penalty(site_id, site_data),
            self._calculate_query_responsiveness_penalty(site_id, site_data),
            self._calculate_conformance_penalty(site_id, site_data),
            self._calculate_safety_criticality_penalty(site_id, site_data, sae_data)
        ]
        
        # Calculate total penalty and DQI
        total_penalty = sum(p.weighted_penalty for p in penalties)
        dqi_score = max(0.0, 100.0 - total_penalty)
        level = self._determine_dqi_level(dqi_score)
        
        result = DQIResult(
            entity_id=site_id,
            entity_type="site",
            dqi_score=dqi_score,
            level=level,
            total_penalty=total_penalty,
            penalties=penalties,
            data_sources_used=data_sources_used
        )
        
        result.recommendation = self._generate_recommendation(result)
        
        return result
    
    def calculate_all_sites_dqi(
        self,
        cpid_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, DQIResult]:
        """
        Calculate DQI for all sites in the data
        
        Returns:
            Dictionary mapping site_id -> DQIResult
        """
        if cpid_data is None:
            cpid_data = self.data_sources.get('cpid')
        
        if cpid_data is None or cpid_data.empty:
            logger.warning("No CPID data available")
            return {}
        
        site_col = self._get_column(cpid_data, 'site_id', 'cpid')
        if not site_col:
            logger.warning("Could not find site ID column")
            return {}
        
        sites = cpid_data[site_col].unique()
        logger.info(f"Calculating DQI for {len(sites)} sites")
        
        results = {}
        for site_id in sites:
            results[site_id] = self.calculate_site_dqi(site_id, cpid_data)
        
        return results
    
    def get_study_dqi_summary(
        self,
        site_results: Dict[str, DQIResult] = None
    ) -> Dict[str, Any]:
        """
        Generate study-level DQI summary statistics
        """
        if site_results is None:
            site_results = self.calculate_all_sites_dqi()
        
        if not site_results:
            return {
                'total_sites': 0,
                'avg_dqi': 0.0,
                'green_count': 0,
                'yellow_count': 0,
                'red_count': 0
            }
        
        scores = [r.dqi_score for r in site_results.values()]
        levels = [r.level for r in site_results.values()]
        
        return {
            'total_sites': len(site_results),
            'avg_dqi': round(np.mean(scores), 2),
            'median_dqi': round(np.median(scores), 2),
            'min_dqi': round(min(scores), 2),
            'max_dqi': round(max(scores), 2),
            'std_dqi': round(np.std(scores), 2),
            'green_count': sum(1 for l in levels if l == DQILevel.GREEN),
            'yellow_count': sum(1 for l in levels if l == DQILevel.YELLOW),
            'red_count': sum(1 for l in levels if l == DQILevel.RED),
            'green_pct': round(sum(1 for l in levels if l == DQILevel.GREEN) / len(levels) * 100, 1),
            'yellow_pct': round(sum(1 for l in levels if l == DQILevel.YELLOW) / len(levels) * 100, 1),
            'red_pct': round(sum(1 for l in levels if l == DQILevel.RED) / len(levels) * 100, 1),
            'sites_by_level': {
                'green': [r.entity_id for r in site_results.values() if r.level == DQILevel.GREEN],
                'yellow': [r.entity_id for r in site_results.values() if r.level == DQILevel.YELLOW],
                'red': [r.entity_id for r in site_results.values() if r.level == DQILevel.RED]
            }
        }


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

class DQIVisualizationEngine:
    """
    Visualization engine for DQI Dashboard
    
    Implements "Drill-Down" architecture:
    1. Global Map View (Executive Level)
    2. Site Detail View (CRA Level)
    3. Patient Detail View (DM Level)
    4. Operational Backlog (Sankey)
    """
    
    def __init__(self, config: DQIConfig = None):
        self.config = config or DEFAULT_DQI_CONFIG
    
    def create_site_risk_profile(
        self,
        site_id: str,
        enrollment_count: int,
        deviation_count: float,
        dqi_score: float
    ) -> SiteRiskProfile:
        """
        Create site risk profile for scatter plot
        Determines quadrant based on enrollment and deviations
        """
        high_volume = enrollment_count >= self.config.high_enrollment_threshold
        high_risk = deviation_count >= self.config.high_deviation_threshold
        
        if high_volume and high_risk:
            quadrant = RiskQuadrant.HIGH_VOLUME_HIGH_RISK
            risk_factors = ["High enrollment with high deviation rate - Greatest threat to trial integrity"]
        elif high_volume and not high_risk:
            quadrant = RiskQuadrant.HIGH_VOLUME_LOW_RISK
            risk_factors = ["High enrollment with good data quality"]
        elif not high_volume and high_risk:
            quadrant = RiskQuadrant.LOW_VOLUME_HIGH_RISK
            risk_factors = ["Low enrollment but quality concerns"]
        else:
            quadrant = RiskQuadrant.LOW_VOLUME_LOW_RISK
            risk_factors = ["Low enrollment, good data quality"]
        
        return SiteRiskProfile(
            site_id=site_id,
            enrollment_count=enrollment_count,
            deviation_count=deviation_count,
            dqi_score=dqi_score,
            quadrant=quadrant,
            risk_factors=risk_factors
        )
    
    def calculate_query_flow(
        self,
        site_id: str,
        site_data: pd.DataFrame
    ) -> QueryFlowData:
        """
        Calculate query flow data for Sankey diagram
        Identifies bottleneck: site (not answering) vs DM (not closing)
        """
        # Get query counts
        open_queries = 0
        answered_queries = 0
        closed_queries = 0
        
        open_col = None
        for col in ['# Open Queries', 'Open Queries', 'open_queries']:
            if col in site_data.columns:
                open_col = col
                break
        
        if open_col:
            open_queries = int(pd.to_numeric(site_data[open_col], errors='coerce').fillna(0).sum())
        
        # Estimate answered and closed from total
        total_col = None
        for col in ['# Total Queries', 'Total Queries', 'total_queries']:
            if col in site_data.columns:
                total_col = col
                break
        
        total_queries = 0
        if total_col:
            total_queries = int(pd.to_numeric(site_data[total_col], errors='coerce').fillna(0).sum())
        
        # Estimate flow (in real scenario, would have actual answered/closed counts)
        closed_queries = max(0, total_queries - open_queries)
        answered_queries = int(closed_queries * 0.9)  # Assume 90% of closed were answered
        
        # Identify bottleneck
        if open_queries > total_queries * 0.3:
            bottleneck = "site"
            bottleneck_desc = "Site not responding to queries in timely manner"
        elif answered_queries > closed_queries * 0.5:
            bottleneck = "dm"
            bottleneck_desc = "DM/Medical Monitor backlog in closing answered queries"
        else:
            bottleneck = "none"
            bottleneck_desc = "Query flow is healthy"
        
        # Calculate overdue
        overdue = int(open_queries * 0.2)  # Estimate 20% overdue
        
        return QueryFlowData(
            site_id=site_id,
            opened_count=total_queries,
            answered_count=answered_queries,
            closed_count=closed_queries,
            overdue_count=overdue,
            bottleneck=bottleneck,
            bottleneck_description=bottleneck_desc
        )
    
    def create_global_heatmap_data(
        self,
        site_results: Dict[str, DQIResult]
    ) -> Dict[str, Any]:
        """
        Create data for Global Map View (Executive Level)
        Returns heatmap data with site locations and DQI scores
        """
        heatmap_data = {
            'sites': [],
            'summary': {
                'total_sites': len(site_results),
                'critical_sites': 0,
                'warning_sites': 0,
                'healthy_sites': 0
            }
        }
        
        for site_id, result in site_results.items():
            site_entry = {
                'site_id': site_id,
                'dqi_score': result.dqi_score,
                'level': result.level.value,
                'color': result.level_color,
                'pulsate': result.level == DQILevel.RED,  # Red sites pulsate
                'recommendation': result.recommendation
            }
            heatmap_data['sites'].append(site_entry)
            
            if result.level == DQILevel.RED:
                heatmap_data['summary']['critical_sites'] += 1
            elif result.level == DQILevel.YELLOW:
                heatmap_data['summary']['warning_sites'] += 1
            else:
                heatmap_data['summary']['healthy_sites'] += 1
        
        return heatmap_data
    
    def create_scatter_plot_data(
        self,
        site_results: Dict[str, DQIResult],
        cpid_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Create data for Site Detail View (CRA Level)
        Deviation vs Enrollment scatter plot
        """
        scatter_data = {
            'sites': [],
            'quadrant_counts': {
                RiskQuadrant.HIGH_VOLUME_HIGH_RISK.value: 0,
                RiskQuadrant.HIGH_VOLUME_LOW_RISK.value: 0,
                RiskQuadrant.LOW_VOLUME_HIGH_RISK.value: 0,
                RiskQuadrant.LOW_VOLUME_LOW_RISK.value: 0
            },
            'thresholds': {
                'enrollment': self.config.high_enrollment_threshold,
                'deviation': self.config.high_deviation_threshold
            }
        }
        
        site_col = None
        for col in ['Site ID', 'Site', 'site_id']:
            if col in cpid_data.columns:
                site_col = col
                break
        
        if not site_col:
            return scatter_data
        
        for site_id, result in site_results.items():
            site_data = cpid_data[cpid_data[site_col] == site_id]
            enrollment = len(site_data)
            
            # Calculate deviations (sum of penalties as proxy)
            deviation = result.total_penalty / 10  # Scale down for visualization
            
            profile = self.create_site_risk_profile(
                site_id, enrollment, deviation, result.dqi_score
            )
            
            scatter_data['sites'].append(profile.to_dict())
            scatter_data['quadrant_counts'][profile.quadrant.value] += 1
        
        return scatter_data
    
    def create_patient_timeline_data(
        self,
        subject_id: str,
        patient_data: pd.Series,
        clean_status: Any = None
    ) -> List[PatientTimelineEvent]:
        """
        Create Patient Detail View (DM Level) timeline data
        Visual timeline of patient visits with Clean Status flags
        """
        timeline_events = []
        
        # Get expected visits
        expected_col = None
        for col in ['# Expected Visits', 'Expected Visits', 'expected_visits']:
            if col in patient_data.index:
                expected_col = col
                break
        
        expected_visits = 1
        if expected_col:
            expected_visits = int(pd.to_numeric(patient_data.get(expected_col, 1), errors='coerce') or 1)
        
        # Get missing visits
        missing_col = None
        for col in ['Missing Visits', '# Missing Visits', 'missing_visits']:
            if col in patient_data.index:
                missing_col = col
                break
        
        missing_visits = 0
        if missing_col:
            missing_visits = int(pd.to_numeric(patient_data.get(missing_col, 0), errors='coerce') or 0)
        
        completed_visits = max(0, expected_visits - missing_visits)
        
        # Create timeline events
        for i in range(1, expected_visits + 1):
            visit_name = f"Visit {i}"
            
            if i <= completed_visits:
                status = "completed"
                blockers = []
                clean = True
            elif i <= completed_visits + missing_visits:
                status = "missing"
                blockers = ["Missing visit data"]
                clean = False
            else:
                status = "upcoming"
                blockers = []
                clean = True
            
            # Add specific blockers if clean_status provided
            if clean_status and hasattr(clean_status, 'conditions'):
                for cond in clean_status.conditions:
                    if not cond.is_met and status == "missing":
                        blockers.extend(cond.blockers[:1])  # Add first blocker
            
            timeline_events.append(PatientTimelineEvent(
                subject_id=subject_id,
                visit_name=visit_name,
                visit_date=None,  # Would need actual dates
                status=status,
                blockers=blockers,
                clean_status=clean
            ))
        
        return timeline_events
    
    def create_sankey_diagram_data(
        self,
        site_results: Dict[str, DQIResult],
        cpid_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Create Operational Backlog Sankey diagram data
        Shows query flow: Opened → Answered → Closed
        """
        sankey_data = {
            'nodes': [
                {'id': 'opened', 'label': 'Opened'},
                {'id': 'answered', 'label': 'Answered'},
                {'id': 'closed', 'label': 'Closed'},
                {'id': 'overdue', 'label': 'Overdue'}
            ],
            'links': [],
            'bottleneck_analysis': {
                'site_bottleneck_count': 0,
                'dm_bottleneck_count': 0,
                'healthy_count': 0
            },
            'sites': []
        }
        
        site_col = None
        for col in ['Site ID', 'Site', 'site_id']:
            if col in cpid_data.columns:
                site_col = col
                break
        
        if not site_col:
            return sankey_data
        
        total_opened = 0
        total_answered = 0
        total_closed = 0
        total_overdue = 0
        
        for site_id in site_results.keys():
            site_data = cpid_data[cpid_data[site_col] == site_id]
            flow = self.calculate_query_flow(site_id, site_data)
            
            sankey_data['sites'].append(flow.to_dict())
            
            total_opened += flow.opened_count
            total_answered += flow.answered_count
            total_closed += flow.closed_count
            total_overdue += flow.overdue_count
            
            if flow.bottleneck == 'site':
                sankey_data['bottleneck_analysis']['site_bottleneck_count'] += 1
            elif flow.bottleneck == 'dm':
                sankey_data['bottleneck_analysis']['dm_bottleneck_count'] += 1
            else:
                sankey_data['bottleneck_analysis']['healthy_count'] += 1
        
        # Create links for Sankey
        sankey_data['links'] = [
            {'source': 'opened', 'target': 'answered', 'value': total_answered},
            {'source': 'opened', 'target': 'overdue', 'value': total_overdue},
            {'source': 'answered', 'target': 'closed', 'value': total_closed}
        ]
        
        sankey_data['totals'] = {
            'opened': total_opened,
            'answered': total_answered,
            'closed': total_closed,
            'overdue': total_overdue
        }
        
        return sankey_data
    
    def generate_dqi_dashboard_html(
        self,
        site_results: Dict[str, DQIResult],
        summary: Dict[str, Any],
        scatter_data: Dict[str, Any],
        sankey_data: Dict[str, Any],
        study_name: str = "Clinical Trial"
    ) -> str:
        """
        Generate complete DQI Dashboard HTML
        Implements drill-down architecture with all visualization components
        """
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Index Dashboard - {study_name}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        .dashboard {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            margin-bottom: 20px;
        }}
        .header h1 {{ 
            margin: 0; 
            font-size: 2.5em;
            background: linear-gradient(90deg, #00cc00, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .subtitle {{ color: #888; margin-top: 10px; }}
        
        /* Summary Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s ease;
        }}
        .summary-card:hover {{ transform: translateY(-5px); }}
        .summary-value {{
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .summary-label {{ color: #888; font-size: 0.9em; }}
        
        /* DQI Level Colors */
        .dqi-green {{ color: #00CC00; }}
        .dqi-yellow {{ color: #FFCC00; }}
        .dqi-red {{ color: #FF0000; }}
        
        /* Section Containers */
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255,255,255,0.1);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        /* Site Table */
        .site-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .site-table th, .site-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .site-table th {{
            background: rgba(0,0,0,0.2);
            font-weight: 600;
        }}
        .site-table tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        /* DQI Score Badge */
        .dqi-badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .dqi-badge.green {{ background: rgba(0,204,0,0.2); color: #00CC00; }}
        .dqi-badge.yellow {{ background: rgba(255,204,0,0.2); color: #FFCC00; }}
        .dqi-badge.red {{ background: rgba(255,0,0,0.2); color: #FF0000; animation: pulse 2s infinite; }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        
        /* Scatter Plot Container */
        .scatter-container {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }}
        .scatter-plot {{
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 20px;
            min-height: 300px;
            position: relative;
        }}
        .quadrant-legend {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .quadrant-item {{
            padding: 15px;
            border-radius: 8px;
            background: rgba(0,0,0,0.2);
        }}
        .quadrant-item.high-risk {{ border-left: 4px solid #FF0000; }}
        .quadrant-item.medium-risk {{ border-left: 4px solid #FFCC00; }}
        .quadrant-item.low-risk {{ border-left: 4px solid #00CC00; }}
        
        /* Sankey Diagram */
        .sankey-container {{
            display: flex;
            align-items: center;
            justify-content: space-around;
            padding: 30px 0;
        }}
        .sankey-node {{
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            min-width: 120px;
        }}
        .sankey-node.opened {{ background: rgba(100,100,255,0.3); }}
        .sankey-node.answered {{ background: rgba(255,200,0,0.3); }}
        .sankey-node.closed {{ background: rgba(0,200,0,0.3); }}
        .sankey-node.overdue {{ background: rgba(255,0,0,0.3); }}
        .sankey-arrow {{
            font-size: 2em;
            color: #666;
        }}
        .sankey-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        /* Bottleneck Analysis */
        .bottleneck-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}
        .bottleneck-card {{
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .bottleneck-card.site {{ background: rgba(255,100,100,0.2); border: 1px solid rgba(255,0,0,0.3); }}
        .bottleneck-card.dm {{ background: rgba(255,200,100,0.2); border: 1px solid rgba(255,150,0,0.3); }}
        .bottleneck-card.healthy {{ background: rgba(100,255,100,0.2); border: 1px solid rgba(0,255,0,0.3); }}
        
        /* Penalty Breakdown */
        .penalty-bar {{
            height: 24px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .penalty-fill {{
            height: 100%;
            transition: width 0.5s ease;
        }}
        .penalty-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            color: #aaa;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>📊 Data Quality Index Dashboard</h1>
            <div class="subtitle">{study_name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        
        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-value">{summary['total_sites']}</div>
                <div class="summary-label">Total Sites</div>
            </div>
            <div class="summary-card">
                <div class="summary-value dqi-green">{summary['avg_dqi']:.1f}</div>
                <div class="summary-label">Average DQI Score</div>
            </div>
            <div class="summary-card">
                <div class="summary-value dqi-green">{summary['green_count']}</div>
                <div class="summary-label">Green Sites ({summary['green_pct']:.0f}%)</div>
            </div>
            <div class="summary-card">
                <div class="summary-value dqi-yellow">{summary['yellow_count']}</div>
                <div class="summary-label">Yellow Sites ({summary['yellow_pct']:.0f}%)</div>
            </div>
            <div class="summary-card">
                <div class="summary-value dqi-red">{summary['red_count']}</div>
                <div class="summary-label">Red Sites ({summary['red_pct']:.0f}%)</div>
            </div>
        </div>
        
        <!-- Global Map View (Executive Level) -->
        <div class="section">
            <h2>🗺️ Global Site Overview (Executive Level)</h2>
            <p style="color: #888;">DQI-colored site indicators. Red sites pulsate to draw attention.</p>
            <table class="site-table">
                <thead>
                    <tr>
                        <th>Site ID</th>
                        <th>DQI Score</th>
                        <th>Status</th>
                        <th>Recommendation</th>
                    </tr>
                </thead>
                <tbody>
'''
        
        # Add site rows sorted by DQI (worst first)
        sorted_sites = sorted(site_results.items(), key=lambda x: x[1].dqi_score)
        for site_id, result in sorted_sites:
            badge_class = result.level.value
            html += f'''
                    <tr>
                        <td><strong>{site_id}</strong></td>
                        <td><span class="dqi-badge {badge_class}">{result.dqi_score:.1f}</span></td>
                        <td>{result.level.value.upper()}</td>
                        <td style="font-size: 0.85em; color: #aaa;">{result.recommendation[:100]}...</td>
                    </tr>
'''
        
        html += '''
                </tbody>
            </table>
        </div>
        
        <!-- Site Detail View (CRA Level) - Scatter Plot -->
        <div class="section">
            <h2>📈 Site Risk Quadrants (CRA Level)</h2>
            <p style="color: #888;">Deviation vs Enrollment scatter plot. High Volume/High Risk sites (top-right) represent greatest threat.</p>
            <div class="scatter-container">
                <div class="scatter-plot">
                    <svg viewBox="0 0 400 300" style="width: 100%; height: 300px;">
                        <!-- Axes -->
                        <line x1="50" y1="250" x2="380" y2="250" stroke="#666" stroke-width="2"/>
                        <line x1="50" y1="250" x2="50" y2="20" stroke="#666" stroke-width="2"/>
                        
                        <!-- Axis Labels -->
                        <text x="200" y="290" fill="#888" text-anchor="middle">Enrollment Count →</text>
                        <text x="15" y="140" fill="#888" text-anchor="middle" transform="rotate(-90, 15, 140)">Deviation Score →</text>
                        
                        <!-- Quadrant Lines -->
                        <line x1="200" y1="250" x2="200" y2="20" stroke="#444" stroke-dasharray="5,5"/>
                        <line x1="50" y1="140" x2="380" y2="140" stroke="#444" stroke-dasharray="5,5"/>
                        
                        <!-- Quadrant Labels -->
                        <text x="120" y="200" fill="#00CC00" font-size="10">Low Vol/Low Risk</text>
                        <text x="280" y="200" fill="#FFCC00" font-size="10">High Vol/Low Risk</text>
                        <text x="120" y="80" fill="#FFCC00" font-size="10">Low Vol/High Risk</text>
                        <text x="280" y="80" fill="#FF0000" font-size="10">High Vol/High Risk</text>
'''
        
        # Add scatter points
        for i, site in enumerate(scatter_data.get('sites', [])[:20]):
            x = 50 + min(330, site['enrollment_count'] * 15)
            y = 250 - min(230, site['deviation_count'] * 20)
            color = "#FF0000" if "HIGH_VOLUME_HIGH_RISK" in site['quadrant'] else \
                   "#FFCC00" if "HIGH" in site['quadrant'] else "#00CC00"
            html += f'''
                        <circle cx="{x}" cy="{y}" r="8" fill="{color}" opacity="0.8">
                            <title>{site['site_id']}: {site['dqi_score']:.1f} DQI</title>
                        </circle>
'''
        
        html += f'''
                    </svg>
                </div>
                <div class="quadrant-legend">
                    <div class="quadrant-item high-risk">
                        <strong>🔴 High Volume / High Risk</strong>
                        <div style="font-size: 2em; margin: 10px 0;">{scatter_data['quadrant_counts'].get(RiskQuadrant.HIGH_VOLUME_HIGH_RISK.value, 0)}</div>
                        <div style="color: #888; font-size: 0.85em;">Greatest threat to trial integrity</div>
                    </div>
                    <div class="quadrant-item medium-risk">
                        <strong>🟡 Warning Quadrants</strong>
                        <div style="font-size: 2em; margin: 10px 0;">{scatter_data['quadrant_counts'].get(RiskQuadrant.HIGH_VOLUME_LOW_RISK.value, 0) + scatter_data['quadrant_counts'].get(RiskQuadrant.LOW_VOLUME_HIGH_RISK.value, 0)}</div>
                        <div style="color: #888; font-size: 0.85em;">Requires monitoring attention</div>
                    </div>
                    <div class="quadrant-item low-risk">
                        <strong>🟢 Low Volume / Low Risk</strong>
                        <div style="font-size: 2em; margin: 10px 0;">{scatter_data['quadrant_counts'].get(RiskQuadrant.LOW_VOLUME_LOW_RISK.value, 0)}</div>
                        <div style="color: #888; font-size: 0.85em;">Low touch monitoring</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Operational Backlog (Sankey) -->
        <div class="section">
            <h2>📊 Query Flow Analysis (Operational Backlog)</h2>
            <p style="color: #888;">Sankey diagram showing query lifecycle. Identifies bottlenecks: Site (not answering) vs DM (not closing).</p>
            
            <div class="sankey-container">
                <div class="sankey-node opened">
                    <div>📥 Opened</div>
                    <div class="sankey-value">{sankey_data['totals']['opened']}</div>
                </div>
                <div class="sankey-arrow">→</div>
                <div class="sankey-node answered">
                    <div>✅ Answered</div>
                    <div class="sankey-value">{sankey_data['totals']['answered']}</div>
                </div>
                <div class="sankey-arrow">→</div>
                <div class="sankey-node closed">
                    <div>🔒 Closed</div>
                    <div class="sankey-value">{sankey_data['totals']['closed']}</div>
                </div>
            </div>
            
            <div style="text-align: center; margin: 20px 0;">
                <div class="sankey-node overdue" style="display: inline-block;">
                    <div>⚠️ Overdue</div>
                    <div class="sankey-value">{sankey_data['totals']['overdue']}</div>
                </div>
            </div>
            
            <h3 style="margin-top: 30px;">Bottleneck Analysis</h3>
            <div class="bottleneck-grid">
                <div class="bottleneck-card site">
                    <div style="font-size: 0.9em; color: #ff6666;">Site Bottleneck</div>
                    <div style="font-size: 2.5em; font-weight: bold;">{sankey_data['bottleneck_analysis']['site_bottleneck_count']}</div>
                    <div style="font-size: 0.8em; color: #aaa;">Sites not responding to queries</div>
                </div>
                <div class="bottleneck-card dm">
                    <div style="font-size: 0.9em; color: #ffaa66;">DM Bottleneck</div>
                    <div style="font-size: 2.5em; font-weight: bold;">{sankey_data['bottleneck_analysis']['dm_bottleneck_count']}</div>
                    <div style="font-size: 0.8em; color: #aaa;">DM backlog closing queries</div>
                </div>
                <div class="bottleneck-card healthy">
                    <div style="font-size: 0.9em; color: #66ff66;">Healthy Flow</div>
                    <div style="font-size: 2.5em; font-weight: bold;">{sankey_data['bottleneck_analysis']['healthy_count']}</div>
                    <div style="font-size: 0.8em; color: #aaa;">Sites with good query flow</div>
                </div>
            </div>
        </div>
        
        <!-- DQI Interpretation Guide -->
        <div class="section">
            <h2>📋 DQI Interpretation Guide</h2>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div style="padding: 20px; background: rgba(0,204,0,0.1); border-radius: 12px; border-left: 4px solid #00CC00;">
                    <h3 style="color: #00CC00; margin-top: 0;">🟢 DQI > 90 (Green)</h3>
                    <p>Site performing well. Low touch monitoring required. Continue routine oversight.</p>
                </div>
                <div style="padding: 20px; background: rgba(255,204,0,0.1); border-radius: 12px; border-left: 4px solid #FFCC00;">
                    <h3 style="color: #FFCC00; margin-top: 0;">🟡 DQI 75-90 (Yellow)</h3>
                    <p>Warning signs detected. Targeted monitoring intervention by "Lia" Agent recommended.</p>
                </div>
                <div style="padding: 20px; background: rgba(255,0,0,0.1); border-radius: 12px; border-left: 4px solid #FF0000;">
                    <h3 style="color: #FF0000; margin-top: 0;">🔴 DQI < 75 (Red)</h3>
                    <p>Critical failure. Immediate onsite audit or rescue intervention required.</p>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Data Quality Index Dashboard | TransCelerate RACT Methodology</p>
            <p>Weights: Visit Adherence (20%) | Query Responsiveness (20%) | Data Conformance (20%) | Safety Criticality (40%)</p>
        </div>
    </div>
</body>
</html>
'''
        
        return html


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_site_dqi(
    site_id: str,
    data_sources: Dict[str, pd.DataFrame],
    config: DQIConfig = None
) -> DQIResult:
    """
    Convenience function to calculate DQI for a single site
    """
    calculator = DataQualityIndexCalculator(config)
    calculator.load_data(data_sources)
    return calculator.calculate_site_dqi(site_id)


def calculate_study_dqi(
    data_sources: Dict[str, pd.DataFrame],
    config: DQIConfig = None
) -> Tuple[Dict[str, DQIResult], Dict[str, Any]]:
    """
    Convenience function to calculate DQI for entire study
    
    Returns:
        Tuple of (site_results dict, summary dict)
    """
    calculator = DataQualityIndexCalculator(config)
    calculator.load_data(data_sources)
    
    site_results = calculator.calculate_all_sites_dqi()
    summary = calculator.get_study_dqi_summary(site_results)
    
    return site_results, summary


def generate_dqi_dashboard(
    data_sources: Dict[str, pd.DataFrame],
    study_name: str = "Clinical Trial",
    config: DQIConfig = None
) -> str:
    """
    Convenience function to generate complete DQI Dashboard HTML
    """
    calculator = DataQualityIndexCalculator(config)
    calculator.load_data(data_sources)
    
    site_results = calculator.calculate_all_sites_dqi()
    summary = calculator.get_study_dqi_summary(site_results)
    
    visualizer = DQIVisualizationEngine(config)
    cpid_data = data_sources.get('cpid', pd.DataFrame())
    
    scatter_data = visualizer.create_scatter_plot_data(site_results, cpid_data)
    sankey_data = visualizer.create_sankey_diagram_data(site_results, cpid_data)
    
    return visualizer.generate_dqi_dashboard_html(
        site_results, summary, scatter_data, sankey_data, study_name
    )
