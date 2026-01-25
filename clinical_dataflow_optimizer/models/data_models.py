"""
Data Models for the Neural Clinical Data Mesh
Defines the core data structures for Clinical Trial Dataflow Optimization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class PatientStatus(Enum):
    """Patient enrollment status"""
    ONGOING = "Ongoing"
    COMPLETED = "Completed"
    DISCONTINUED = "Discontinued"
    SCREEN_FAILED = "Screen Failed"
    UNKNOWN = "Unknown"


class QueryStatus(Enum):
    """Query resolution status"""
    OPEN = "Open"
    ANSWERED = "Answered"
    CLOSED = "Closed"
    CANCELLED = "Cancelled"


class CodingStatus(Enum):
    """Medical coding status"""
    UNCODED = "UnCoded"
    CODED = "Coded"
    PENDING_REVIEW = "Pending Review"
    CLARIFICATION_NEEDED = "Clarification Needed"


class RiskLevel(Enum):
    """Risk classification levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class AgentAction(Enum):
    """Actions that can be taken by AI agents"""
    QUERY_GENERATED = "Query Generated"
    ALERT_SENT = "Alert Sent"
    AUTO_CODED = "Auto Coded"
    ESCALATED = "Escalated"
    RECONCILED = "Reconciled"
    FLAGGED_FOR_REVIEW = "Flagged for Review"


@dataclass
class BlockingItem:
    """Represents an item blocking clean patient status"""
    item_type: str
    description: str
    source_file: str
    severity: str
    days_outstanding: Optional[int] = None
    related_query_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'type': self.item_type,
            'description': self.description,
            'source': self.source_file,
            'severity': self.severity,
            'days_outstanding': self.days_outstanding,
            'query_id': self.related_query_id
        }


@dataclass
class RiskMetrics:
    """
    Aggregated risk metrics for a patient or site
    
    Includes three engineered features for AI/ML models:
    1. Operational Velocity Index (resolution_velocity, accumulation_velocity)
    2. Normalized Data Density (data_density_score, query_density_normalized)
    3. Manipulation Risk Score (manipulation_risk_score, manipulation_risk_value)
    """
    # Core metrics
    query_aging_index: float = 0.0
    protocol_deviation_count: int = 0
    safety_signal_count: int = 0
    visit_compliance_rate: float = 100.0
    
    # Feature 1: Operational Velocity Index
    # V_res = Δ(# Closed Queries) / Δt
    resolution_velocity: float = 0.0          # Queries closed per day
    accumulation_velocity: float = 0.0        # Queries opened per day  
    net_velocity: float = 0.0                 # Resolution - Accumulation
    is_bottleneck: bool = False               # True if queries accumulating
    
    # Feature 2: Normalized Data Density
    # D_density = Total Queries / # Pages Entered
    data_density_score: float = 0.0           # Raw queries per page
    query_density_normalized: float = 0.0     # Normalized 0-1 scale
    query_density_percentile: float = 0.0     # Compared to other sites
    
    # Feature 3: Manipulation Risk Score
    # Based on inactivated forms and audit actions
    manipulation_risk_score: str = "Low"      # Risk level label
    manipulation_risk_value: float = 0.0      # Numeric score 0-100
    endpoint_risk_score: float = 0.0          # Risk to primary endpoint data
    inactivation_rate: float = 0.0            # Inactivations per month
    
    # Composite score for RBM engine
    composite_risk_score: float = 0.0         # Weighted combination
    requires_intervention: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'query_aging_index': round(self.query_aging_index, 2),
            'protocol_deviation_count': self.protocol_deviation_count,
            'safety_signal_count': self.safety_signal_count,
            'visit_compliance_rate': round(self.visit_compliance_rate, 2),
            # Feature 1: Velocity Index
            'velocity_index': {
                'resolution_velocity': round(self.resolution_velocity, 3),
                'accumulation_velocity': round(self.accumulation_velocity, 3),
                'net_velocity': round(self.net_velocity, 3),
                'is_bottleneck': self.is_bottleneck
            },
            # Feature 2: Data Density
            'data_density': {
                'density_score': round(self.data_density_score, 4),
                'normalized': round(self.query_density_normalized, 4),
                'percentile': round(self.query_density_percentile, 1)
            },
            # Feature 3: Manipulation Risk
            'manipulation_risk': {
                'level': self.manipulation_risk_score,
                'score': round(self.manipulation_risk_value, 1),
                'endpoint_risk': round(self.endpoint_risk_score, 1),
                'inactivation_rate': round(self.inactivation_rate, 2)
            },
            # Composite for RBM
            'composite_risk_score': round(self.composite_risk_score, 1),
            'requires_intervention': self.requires_intervention
        }


@dataclass
class DigitalPatientTwin:
    """
    The Digital Patient Twin - unified representation of a patient's trial data
    This serves as the single source of truth for both UI and AI agents
    """
    subject_id: str
    site_id: str
    study_id: str
    country: str = ""
    region: str = ""
    status: PatientStatus = PatientStatus.UNKNOWN
    
    # Clean Patient Status
    clean_status: bool = False
    clean_percentage: float = 0.0
    blocking_items: List[BlockingItem] = field(default_factory=list)
    
    # Core Metrics from CPID
    missing_visits: int = 0
    missing_pages: int = 0
    open_queries: int = 0
    total_queries: int = 0
    uncoded_terms: int = 0
    coded_terms: int = 0
    verification_pct: float = 0.0
    forms_verified: int = 0
    expected_visits: int = 0
    pages_entered: int = 0
    non_conformant_pages: int = 0
    reconciliation_issues: int = 0
    protocol_deviations: int = 0
    
    # Derived Metrics
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    data_quality_index: float = 100.0
    
    # Visit Data
    outstanding_visits: List[Dict] = field(default_factory=list)
    
    # Safety Data
    sae_records: List[Dict] = field(default_factory=list)
    safety_reconciliation_status: str = "Not Applicable"
    
    # Coding Data
    uncoded_terms_list: List[Dict] = field(default_factory=list)
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    snapshot_date: Optional[datetime] = None
    
    def calculate_clean_percentage(self) -> float:
        """Calculate how close the patient is to being clean"""
        total_checks = 7  # Number of clean status criteria
        passed_checks = 0
        
        if self.missing_visits == 0:
            passed_checks += 1
        if self.missing_pages == 0:
            passed_checks += 1
        if self.open_queries == 0:
            passed_checks += 1
        if self.uncoded_terms == 0:
            passed_checks += 1
        if self.reconciliation_issues == 0:
            passed_checks += 1
        if self.verification_pct >= 100.0:
            passed_checks += 1
        if len(self.sae_records) == 0 or self.safety_reconciliation_status == "Reconciled":
            passed_checks += 1
            
        self.clean_percentage = (passed_checks / total_checks) * 100
        return self.clean_percentage
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'subject_id': self.subject_id,
            'site_id': self.site_id,
            'study_id': self.study_id,
            'country': self.country,
            'region': self.region,
            'status': self.status.value,
            'clean_status': self.clean_status,
            'clean_percentage': round(self.clean_percentage, 1),
            'blocking_items': [item.to_dict() for item in self.blocking_items],
            'metrics': {
                'missing_visits': self.missing_visits,
                'missing_pages': self.missing_pages,
                'open_queries': self.open_queries,
                'total_queries': self.total_queries,
                'uncoded_terms': self.uncoded_terms,
                'verification_pct': round(self.verification_pct, 1),
                'non_conformant_pages': self.non_conformant_pages,
                'reconciliation_issues': self.reconciliation_issues,
                'protocol_deviations': self.protocol_deviations
            },
            'risk_metrics': self.risk_metrics.to_dict(),
            'data_quality_index': round(self.data_quality_index, 1),
            'outstanding_visits': self.outstanding_visits,
            'sae_count': len(self.sae_records),
            'last_updated': self.last_updated.isoformat()
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SiteMetrics:
    """Aggregated metrics for a clinical site"""
    site_id: str
    study_id: str
    country: str = ""
    region: str = ""
    
    # Patient counts
    total_patients: int = 0
    clean_patients: int = 0
    ongoing_patients: int = 0
    
    # Aggregated metrics
    total_missing_visits: int = 0
    total_missing_pages: int = 0
    total_open_queries: int = 0
    total_queries: int = 0
    total_uncoded_terms: int = 0
    total_protocol_deviations: int = 0
    total_non_conformant: int = 0
    total_pages_entered: int = 0
    
    # Derived metrics
    data_quality_index: float = 100.0
    risk_level: RiskLevel = RiskLevel.LOW
    query_density: float = 0.0  # Queries per page
    clean_patient_rate: float = 0.0
    visit_compliance_rate: float = 100.0
    
    # Velocity metrics (change over time)
    query_resolution_velocity: float = 0.0
    error_accumulation_velocity: float = 0.0
    
    # Site status
    ssm_status: str = "Green"  # Site Status Metric
    requires_intervention: bool = False
    
    def calculate_derived_metrics(self):
        """Calculate derived metrics from raw data"""
        if self.total_pages_entered > 0:
            self.query_density = self.total_queries / self.total_pages_entered
        
        if self.total_patients > 0:
            self.clean_patient_rate = (self.clean_patients / self.total_patients) * 100
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'study_id': self.study_id,
            'country': self.country,
            'region': self.region,
            'total_patients': self.total_patients,
            'clean_patients': self.clean_patients,
            'clean_patient_rate': round(self.clean_patient_rate, 1),
            'data_quality_index': round(self.data_quality_index, 1),
            'risk_level': self.risk_level.value,
            'metrics': {
                'missing_visits': self.total_missing_visits,
                'missing_pages': self.total_missing_pages,
                'open_queries': self.total_open_queries,
                'query_density': round(self.query_density, 4),
                'protocol_deviations': self.total_protocol_deviations,
                'non_conformant': self.total_non_conformant
            },
            'velocity': {
                'query_resolution': round(self.query_resolution_velocity, 2),
                'error_accumulation': round(self.error_accumulation_velocity, 2)
            },
            'ssm_status': self.ssm_status,
            'requires_intervention': self.requires_intervention
        }


@dataclass
class StudyMetrics:
    """Aggregated metrics for an entire study"""
    study_id: str
    study_name: str = ""
    
    # Site counts
    total_sites: int = 0
    sites_at_risk: int = 0
    
    # Patient counts
    total_patients: int = 0
    clean_patients: int = 0
    ongoing_patients: int = 0
    
    # Global metrics
    global_dqi: float = 100.0
    global_clean_rate: float = 0.0
    interim_analysis_ready: bool = False
    
    # Site breakdown
    sites_by_risk: Dict[str, int] = field(default_factory=lambda: {
        'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0
    })
    
    # Country breakdown
    countries: List[str] = field(default_factory=list)
    metrics_by_country: Dict[str, Dict] = field(default_factory=dict)
    
    def calculate_interim_readiness(self, threshold: float = 80.0) -> bool:
        """Determine if study is ready for interim analysis"""
        if self.total_patients > 0:
            self.global_clean_rate = (self.clean_patients / self.total_patients) * 100
        self.interim_analysis_ready = self.global_clean_rate >= threshold
        return self.interim_analysis_ready
    
    def to_dict(self) -> Dict:
        return {
            'study_id': self.study_id,
            'study_name': self.study_name,
            'total_sites': self.total_sites,
            'sites_at_risk': self.sites_at_risk,
            'total_patients': self.total_patients,
            'clean_patients': self.clean_patients,
            'global_dqi': round(self.global_dqi, 1),
            'global_clean_rate': round(self.global_clean_rate, 1),
            'interim_analysis_ready': self.interim_analysis_ready,
            'sites_by_risk': self.sites_by_risk,
            'countries': self.countries
        }


@dataclass
class AgentActionLog:
    """Log entry for an action taken by an AI agent"""
    action_id: str
    agent_name: str  # Rex, Codex, or Lia
    action_type: AgentAction
    subject_id: str
    site_id: str
    study_id: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: Optional[float] = None
    requires_human_review: bool = False
    human_approved: Optional[bool] = None
    source_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'action_id': self.action_id,
            'agent': self.agent_name,
            'action_type': self.action_type.value,
            'subject_id': self.subject_id,
            'site_id': self.site_id,
            'study_id': self.study_id,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence_score,
            'requires_review': self.requires_human_review,
            'approved': self.human_approved
        }
