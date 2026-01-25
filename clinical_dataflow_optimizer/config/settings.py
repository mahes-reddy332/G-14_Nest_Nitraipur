"""
Configuration settings for the Neural Clinical Data Mesh Framework
Clinical Trial Dataflow Optimization System
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, rely on system environment
    pass

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "QC Anonymized Study Files"

# DQI Weights (aligned with RBQM principles)
@dataclass
class DQIWeights:
    """
    Data Quality Index weight configuration
    Based on TransCelerate RACT methodology
    """
    visit_adherence: float = 0.20      # Missing visits impact
    query_responsiveness: float = 0.20  # Query handling efficiency
    conformance: float = 0.20           # Non-conformant data rate
    safety_criticality: float = 0.40    # Safety data integrity (highest weight)

# Clean Patient Status Thresholds
@dataclass
class CleanPatientThresholds:
    """
    Thresholds for determining Clean Patient Status
    Note: These are industry-standard thresholds for interim analysis readiness
    """
    max_days_outstanding: int = 30      # Maximum acceptable days for visit
    max_open_queries: int = 0           # Must be zero for clean status (queries = blocking items)
    max_uncoded_terms: int = 0          # Must be zero for clean status (coding must be complete)
    min_verification_pct: float = 75.0  # Required verification percentage (relaxed from 100%)
    max_reconciliation_issues: int = 0  # Must be zero for clean status (safety reconciliation)
    max_non_conformant: int = 0         # Maximum non-conformant data entries

# Agent Configuration
@dataclass
class AgentConfig:
    """
    Configuration for Agentic AI agents
    """
    # Reconciliation Agent (Rex) thresholds
    sae_pending_days_threshold: int = 7
    
    # Zombie SAE Detection (Scenario A)
    zombie_sae_enabled: bool = True
    zombie_sae_auto_query: bool = True
    zombie_sae_update_edrr: bool = True
    
    # Ambiguous Coding Detection (Scenario B)
    ambiguous_coding_enabled: bool = True
    ambiguous_coding_auto_query: bool = True
    ambiguous_coding_learning: bool = True
    
    # Ghost Visit Detection (Scenario C)
    ghost_visit_enabled: bool = True
    ghost_visit_auto_reminder: bool = True
    ghost_visit_escalation_enabled: bool = True
    
    # Coding Agent (Codex) confidence levels
    coding_auto_apply_threshold: float = 0.95
    coding_propose_threshold: float = 0.80
    
    # Site Liaison Agent (Lia) thresholds
    visit_reminder_days: int = 5
    escalation_days: int = 14
    
    # Alert fatigue prevention
    max_queries_per_site_per_day: int = 10


# =============================================================================
# ZOMBIE SAE DETECTION CONFIGURATION (Scenario A)
# =============================================================================

@dataclass
class ZombieSAEDetectionConfig:
    """
    Configuration for Zombie SAE Detection (Scenario A)
    
    Problem: SAE reported to safety DB but AE form missing in EDC
    
    Pipeline:
    1. Data Check: Scan SAE Dashboard for Action Status = "Pending"
    2. Cross-Reference: Query CPID_EDC_Metrics for Subject ID
    3. Logic Gate: Check "# eSAE dashboard review for DM" column
    4. Verification: Cross-check Global_Missing_Pages_Report for AE form
    5. Action: Auto-draft query to site
    6. Update: Update Compiled_EDRR "Total Open issue Count"
    """
    # Detection triggers
    action_status_pending_values: List[str] = field(default_factory=lambda: [
        'Pending', 'pending', 'PENDING', 'Open', 'open', 'OPEN',
        'Pending for Review', 'Action Required', 'Awaiting Response'
    ])
    
    review_status_pending_values: List[str] = field(default_factory=lambda: [
        'Pending for Review', 'pending for review', 'PENDING FOR REVIEW',
        'Under Review', 'Awaiting Review', 'Not Reviewed'
    ])
    
    # Days thresholds for escalation
    days_pending_warning: int = 3
    days_pending_critical: int = 7
    
    # Confidence thresholds for classification
    high_confidence_threshold: float = 0.90
    medium_confidence_threshold: float = 0.75
    
    # AE form patterns to search in Missing Pages Report
    ae_form_patterns: List[str] = field(default_factory=lambda: [
        'Adverse Event', 'AE', 'adverse event', 'ae form',
        'Serious Adverse Event', 'SAE', 'Safety Event',
        'adverse_event', 'ae_form', 'sae_form'
    ])
    
    # Auto-query settings
    enable_auto_query: bool = True
    query_requires_approval: bool = False  # Per Scenario A spec: auto-executable
    
    # EDRR update settings
    enable_edrr_update: bool = True
    
    # Query template
    query_template: str = (
        "URGENT: Safety Reconciliation Required\n\n"
        "Safety database indicates an SAE recorded on {sae_date} for Subject {patient_id} at Site {site_id}.\n"
        "However, the corresponding Adverse Event form was not found in EDC.\n\n"
        "Discrepancy Details:\n"
        "- Discrepancy ID: {discrepancy_id}\n"
        "- SAE Form: {form_name}\n"
        "- CPID eSAE DM Review count: {esae_dm_count}\n"
        "- Missing AE Form(s): {missing_forms}\n\n"
        "Action Required: Please enter the AE data in EDC or provide clarification.\n"
        "Reference: ICH E6 R2 Section 5.18.4 - Safety Reporting Compliance"
    )


# =============================================================================
# AMBIGUOUS CODING DETECTION CONFIGURATION (Scenario B)
# =============================================================================

@dataclass
class AmbiguousCodingDetectionConfig:
    """
    Configuration for Ambiguous Concomitant Medication Detection (Scenario B)
    
    Problem: Site enters vague terms like "Pain killer" that cannot be coded in WHODRA
    
    Pipeline:
    1. Data Check: Scan GlobalCodingReport_WHODRA for Coding Status = "UnCoded Term"
    2. LLM Query: Assess if term is specific enough for WHODRA coding
    3. Reasoning: Evaluate confidence (High >95%, Medium 80-95%, Low <80%)
    4. Action: Auto-code, propose for approval, or trigger clarification workflow
    5. Learning: Track site clarifications for probability weight updates
    """
    # Confidence thresholds
    high_confidence_threshold: float = 0.95
    medium_confidence_threshold: float = 0.80
    
    # Ambiguous drug class terms (trigger clarification)
    ambiguous_drug_classes: List[str] = field(default_factory=lambda: [
        'pain killer', 'painkiller', 'pain medication', 'pain med',
        'antibiotic', 'antibiotics', 'blood pressure medication',
        'blood thinner', 'diabetes medication', 'heart medication',
        'sleeping pill', 'anxiety medication', 'allergy medication',
        'cough medicine', 'stomach medication', 'steroid',
        'supplement', 'vitamin', 'herbal', 'natural remedy',
        'over the counter', 'otc'
    ])
    
    # Enable learning from site responses
    enable_learning: bool = True
    
    # Query settings
    enable_auto_query: bool = True
    query_requires_approval: bool = False
    
    # Query template for drug class
    query_template_drug_class: str = (
        "Term '{verbatim}' is a drug class, not a specific medication. "
        "Please provide the specific Trade Name or Generic Name "
        "(e.g., for 'pain killer': Paracetamol, Ibuprofen, Aspirin). "
        "[Reference: WHO Drug Dictionary coding requirement]"
    )
    
    # Query template for abbreviation
    query_template_abbreviation: str = (
        "Abbreviation '{verbatim}' could refer to multiple medications: {options}. "
        "Please clarify the exact Trade Name or Generic Name for Subject {subject_id}."
    )


# =============================================================================
# GHOST VISIT DETECTION CONFIGURATION (Scenario C)
# =============================================================================

@dataclass
class GhostVisitDetectionConfig:
    """
    Configuration for Ghost Visit Detection (Scenario C)
    
    Problem: Patient scheduled for visit but no data entered; site hasn't flagged "Missed Visit"
    
    Pipeline:
    1. Data Check: Scan Visit Projection Tracker for Projected Date < Current Date
    2. Calculation: Calculate Days Outstanding
    3. Context Check: Check Inactivated forms for valid reasons (withdrawal, etc.)
    4. Action:
       - Valid inactivation: No query needed
       - Green site (good standing): Standard reminder
       - Red site (non-compliance): Escalate to CRA for phone call
    """
    # Days thresholds
    days_warning_threshold: int = 7       # Days before warning
    days_critical_threshold: int = 14     # Days before critical escalation
    days_sponsor_escalation: int = 21     # Days before sponsor notification
    
    # Site standing thresholds (issues per subject)
    site_red_threshold: int = 5           # Red status threshold
    site_yellow_threshold: int = 3        # Yellow status threshold
    
    # Valid inactivation keywords (no query needed)
    valid_inactivation_keywords: List[str] = field(default_factory=lambda: [
        'withdrew', 'withdrawal', 'discontinued', 'lost to follow',
        'not applicable', 'not done', 'deceased', 'death',
        'protocol deviation', 'screen failure', 'consent withdrawn'
    ])
    
    # Enable risk scoring
    enable_risk_scoring: bool = True
    
    # Data entry deadline buffer
    data_entry_deadline_days: int = 5
    
    # Reminder template for standard reminder
    reminder_template_standard: str = (
        "Reminder: Visit '{visit}' for Subject {subject_id} was projected for "
        "{projected_date}. It is now {days_outstanding} days past the scheduled date. "
        "Please enter the visit data or mark as 'Missed Visit' if applicable."
    )
    
    # Reminder template for CRA escalation
    reminder_template_escalation: str = (
        "ESCALATION REQUIRED: Site {site_id} - Subject {subject_id}\n"
        "Visit '{visit}' is {days_outstanding} days overdue (projected {projected_date}).\n"
        "Site has history of data entry delays. Recommend immediate phone call to Site Coordinator.\n"
        "Reference: ICH E6 R2 Section 4.9.5 - Source document verification"
    )


# File mapping patterns for each study
@dataclass
class FilePatterns:
    """
    File naming patterns for data ingestion
    """
    cpid_metrics: str = "CPID_EDC_Metrics"
    compiled_edrr: str = "Compiled_EDRR"
    sae_dashboard: str = "eSAE"
    meddra_coding: str = "MedDRA"
    whodra_coding: str = "WHODD"
    inactivated_forms: str = "Inactivated"
    missing_lab: str = "Missing_Lab"
    missing_pages: str = "Missing_Pages"
    visit_tracker: str = "Visit Projection"

# Column mappings for standardization
CPID_COLUMN_MAP = {
    'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'SUBJECTID'],
    'site_id': ['Site ID', 'SiteID', 'Site_ID', 'SITEID', 'Site'],
    'country': ['Country', 'COUNTRY'],
    'region': ['Region', 'REGION'],
    'status': ['Subject Status', 'Status', 'SUBJECT_STATUS'],
    'missing_visits': ['Missing Visits', '# Missing Visits', 'MissingVisits'],
    'missing_pages': ['Missing Page', '# Missing Pages', 'MissingPages'],
    'open_queries': ['# Open Queries', 'Open Queries', 'OpenQueries'],
    'total_queries': ['# Total Queries', 'Total Queries', 'TotalQueries'],
    'uncoded_terms': ['# Uncoded Terms', 'Uncoded Terms', 'UncodedTerms'],
    'coded_terms': ['# Coded terms', 'Coded Terms', 'CodedTerms'],
    'verification_pct': ['Data Verification %', 'Verification %', 'VerificationPct'],
    'forms_verified': ['# Forms Verified', 'Forms Verified', 'FormsVerified'],
    'expected_visits': ['# Expected Visits', 'Expected Visits', 'ExpectedVisits'],
    'pages_entered': ['# Pages Entered', 'Pages Entered', 'PagesEntered'],
    'non_conformant': ['# Pages with Non-Conformant data', 'Non-Conformant', 'NonConformant'],
    'esae_review': ['# eSAE dashboard review for DM', 'eSAE Review', 'eSAEReview'],
    'reconciliation_issues': ['# Reconciliation Issues', 'Recon Issues', 'ReconIssues'],
    'broken_signatures': ['Broken Signatures', 'BrokenSignatures'],
    'protocol_deviations': ['# PDs Confirmed', 'PDs Confirmed', 'ProtocolDeviations'],
    'crf_overdue': ['CRFs overdue for signs', 'CRF Overdue', 'CRFOverdue'],
    'crf_overdue_90': ['CRFs overdue for signs beyond 90 days', 'CRF Overdue 90', 'CRFOverdue90'],
    'locked_pages': ['# Locked Pages', 'Locked Pages', 'LockedPages'],
    'frozen_pages': ['# Frozen Pages', 'Frozen Pages', 'FrozenPages']
}

VISIT_TRACKER_COLUMN_MAP = {
    'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID'],
    'site_id': ['Site ID', 'SiteID', 'Site_ID', 'Site'],
    'visit_name': ['Visit Name', 'VisitName', 'Visit'],
    'projected_date': ['Projected Date', 'ProjectedDate', 'Projected_Date'],
    'days_outstanding': ['# Days Outstanding', 'Days Outstanding', 'DaysOutstanding']
}

SAE_COLUMN_MAP = {
    'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID', 'Patient ID'],
    'site_id': ['Site ID', 'SiteID', 'Site_ID', 'Site'],
    'discrepancy_id': ['Discrepancy ID', 'DiscrepancyID', 'Discrepancy_ID'],
    'review_status': ['Review Status', 'ReviewStatus', 'Review_Status'],
    'action_status': ['Action Status', 'ActionStatus', 'Action_Status'],
    'event_date': ['Event Date', 'EventDate', 'Event_Date'],
    'sae_type': ['SAE Type', 'SAEType', 'Type']
}

CODING_COLUMN_MAP = {
    'subject_id': ['Subject ID', 'SubjectID', 'Subject_ID'],
    'verbatim_term': ['Verbatim Term', 'VerbatimTerm', 'Verbatim'],
    'coding_status': ['Coding Status', 'CodingStatus', 'Status'],
    'coded_term': ['Coded Term', 'CodedTerm', 'LLT', 'PT'],
    'context': ['Context', 'Form', 'Source']
}

# Risk Level Definitions
RISK_LEVELS = {
    'critical': {'min_dqi': 0, 'max_dqi': 50, 'color': '#FF0000', 'action': 'immediate_intervention'},
    'high': {'min_dqi': 50, 'max_dqi': 75, 'color': '#FF6600', 'action': 'onsite_audit'},
    'medium': {'min_dqi': 75, 'max_dqi': 90, 'color': '#FFCC00', 'action': 'targeted_monitoring'},
    'low': {'min_dqi': 90, 'max_dqi': 100, 'color': '#00CC00', 'action': 'standard_monitoring'}
}

# Knowledge Graph Configuration
@dataclass
class GraphConfig:
    """
    Configuration for the Neural Clinical Data Mesh Knowledge Graph
    Uses NetworkX as offline graph database (alternative to Neo4j/Amazon Neptune)
    """
    # Graph persistence settings
    enable_persistence: bool = True
    persistence_dir: str = "graph_data"
    
    # Node creation settings
    create_site_nodes: bool = True
    create_visit_nodes: bool = True
    create_sae_nodes: bool = True
    create_coding_nodes: bool = True
    create_edrr_nodes: bool = True
    
    # Query engine settings
    enable_query_caching: bool = True
    max_traversal_depth: int = 3
    
    # Risk scoring weights
    risk_weight_missing_visits: float = 10.0
    risk_weight_open_queries: float = 5.0
    risk_weight_uncoded_terms: float = 5.0
    risk_weight_reconciliation: float = 15.0
    risk_weight_protocol_deviations: float = 10.0
    
    # Multi-hop query thresholds
    attention_threshold_missing_visits: int = 1
    attention_threshold_open_queries: int = 1
    attention_threshold_uncoded_terms: int = 1
    attention_threshold_days_outstanding: int = 30

# Graph Node Type Definitions for Data Mesh
GRAPH_NODE_TYPES = {
    'patient': {
        'description': 'Central anchor node - represents a clinical trial subject',
        'source_files': ['CPID_EDC_Metrics'],
        'key_attributes': ['subject_id', 'site_id', 'country', 'region', 'status']
    },
    'site': {
        'description': 'Clinical trial site - aggregated from patient data',
        'source_files': ['CPID_EDC_Metrics'],
        'key_attributes': ['site_id', 'country', 'region', 'total_patients']
    },
    'event': {
        'description': 'Visit/Event node for temporal progression modeling',
        'source_files': ['Visit Projection Tracker'],
        'key_attributes': ['visit_name', 'projected_date', 'days_outstanding']
    },
    'discrepancy': {
        'description': 'Query/Discrepancy node capturing data friction',
        'source_files': ['CPID_EDC_Metrics'],
        'key_attributes': ['query_id', 'query_type', 'status', 'days_open']
    },
    'sae': {
        'description': 'Serious Adverse Event node for safety data',
        'source_files': ['SAE Dashboard'],
        'key_attributes': ['review_status', 'action_status', 'requires_reconciliation']
    },
    'coding_term': {
        'description': 'Medical coding term node',
        'source_files': ['GlobalCodingReport_MedDRA', 'GlobalCodingReport_WHODRA'],
        'key_attributes': ['verbatim_term', 'coded_term', 'coding_status']
    }
}

# Graph Edge Type Definitions for Data Mesh
GRAPH_EDGE_TYPES = {
    'HAS_VISIT': {
        'description': 'Patient has scheduled/completed visit',
        'source': 'Patient',
        'target': 'Event',
        'properties': ['projected_date', 'days_outstanding']
    },
    'HAS_ADVERSE_EVENT': {
        'description': 'Patient has SAE record',
        'source': 'Patient',
        'target': 'SAE',
        'properties': ['review_status', 'action_status']
    },
    'HAS_CODING_ISSUE': {
        'description': 'Patient has uncoded medical term',
        'source': 'Patient',
        'target': 'CodingTerm',
        'properties': ['verbatim_term', 'coding_status']
    },
    'HAS_QUERY': {
        'description': 'Patient has open query/discrepancy',
        'source': 'Patient',
        'target': 'Discrepancy',
        'properties': ['query_type', 'status']
    },
    'ENROLLED_AT': {
        'description': 'Patient enrolled at site',
        'source': 'Patient',
        'target': 'Site',
        'properties': []
    }
}

# Feature Engineering Configuration for AI/ML Models
@dataclass
class FeatureEngineeringConfig:
    """
    Configuration for feature engineering thresholds
    Used by the AI/ML model training pipeline
    """
    # Velocity Index Configuration
    # V_res = Δ(# Closed Queries) / Δt
    velocity_window_days: int = 7           # Rolling window for velocity calculation
    critical_velocity_threshold: float = -5.0  # Negative = queries accumulating (bottleneck)
    positive_velocity_threshold: float = 2.0   # Positive = queries being resolved
    
    # Data Density Configuration  
    # D_density = Total Queries / # Pages Entered
    high_density_threshold: float = 0.10     # 10% - queries per page is concerning
    critical_density_threshold: float = 0.15  # 15% - queries per page is critical
    low_density_threshold: float = 0.02       # 2% - acceptable threshold
    
    # Manipulation Risk Configuration
    # Based on Inactivated forms/folders and Audit Actions
    primary_endpoint_forms: List[str] = field(default_factory=lambda: [
        'Efficacy', 'Primary', 'Endpoint', 'Tumor', 'Response', 
        'Survival', 'PFS', 'OS', 'ORR', 'DOR', 'Assessment'
    ])
    high_risk_audit_actions: List[str] = field(default_factory=lambda: [
        'Inactivated', 'Deleted', 'Removed', 'Cleared', 'Reset',
        'Unverified', 'Unsigned', 'Unlocked'
    ])
    inactivation_frequency_threshold: int = 5  # Inactivations per month
    
    # Risk score weights for composite calculation
    weight_inactivation_frequency: float = 0.30
    weight_endpoint_data_risk: float = 0.35
    weight_temporal_pattern: float = 0.20
    weight_audit_trail_anomaly: float = 0.15


# =============================================================================
# VISIT ADHERENCE TRACKING CONFIGURATION
# =============================================================================

@dataclass
class VisitAdherenceConfig:
    """
    Configuration for Visit Adherence Agent and Top 10 Offenders tracking
    
    Scientific Question Addressed:
    "Which sites/patients have the most missing visits?"
    """
    # Top Offenders Configuration
    top_offenders_count: int = 10                    # Number of offenders to display
    days_outstanding_critical: int = 60              # Critical threshold
    days_outstanding_high: int = 30                  # High priority threshold
    days_outstanding_medium: int = 14                # Medium priority threshold
    
    # Visit Adherence Score Thresholds
    adherence_green_threshold: float = 95.0          # Green: >95% visits on time
    adherence_yellow_threshold: float = 80.0         # Yellow: 80-95%
    adherence_red_threshold: float = 0.0             # Red: <80%
    
    # Aggregation settings
    aggregate_by_site: bool = True                   # Enable site-level aggregation
    aggregate_by_country: bool = True                # Enable country-level aggregation
    include_patient_details: bool = True             # Include patient-level breakdown
    
    # Alert Configuration
    send_alerts_for_critical: bool = True            # Auto-alert for critical cases
    escalation_after_days: int = 90                  # Escalate to sponsor after N days


# =============================================================================
# DQI HEATMAP AND GEOGRAPHIC ANALYSIS
# =============================================================================

@dataclass
class DQIHeatmapConfig:
    """
    Configuration for DQI Heatmap visualization
    
    Scientific Question Addressed:
    "Where are the highest rates of non-conformant data?"
    """
    # Geographic visualization
    enable_geographic_view: bool = True
    color_scale: str = "RdYlGn"                      # Red-Yellow-Green gradient
    pulsate_critical_sites: bool = True              # Pulsating animation for critical sites
    
    # Non-conformance tracking
    non_conformant_weight: float = 0.25              # Weight in DQI calculation
    non_conformant_critical_threshold: int = 10     # Pages for critical flag
    non_conformant_high_threshold: int = 5          # Pages for high flag
    
    # Re-training intervention triggers
    trigger_retraining_threshold: int = 3            # Consecutive weeks of high non-conformance
    retraining_types: List[str] = field(default_factory=lambda: [
        'EDC Entry Guidelines',
        'Protocol Requirements',
        'Source Data Verification',
        'Query Resolution Process'
    ])


# =============================================================================
# SITE INTERVENTION AND FLAGGING
# =============================================================================

@dataclass
class SiteInterventionConfig:
    """
    Configuration for site intervention flagging
    
    Scientific Question Addressed:
    "Which sites require immediate attention?"
    """
    # DQI-based flagging
    immediate_intervention_dqi: float = 75.0         # DQI < 75 = immediate attention
    high_priority_dqi: float = 85.0                  # DQI < 85 = high priority
    
    # Delta Engine - Velocity of Change
    delta_window_weeks: int = 4                      # Weeks for trend calculation
    negative_velocity_threshold: float = -5.0        # DQI drop per week threshold
    critical_velocity_threshold: float = -10.0       # Severe decline threshold
    
    # Combined flagging (DQI < threshold AND negative velocity)
    combined_flag_enabled: bool = True
    
    # Intervention types by severity
    intervention_actions: Dict[str, List[str]] = field(default_factory=lambda: {
        'critical': ['Immediate onsite audit', 'Study Director notification', 'Enrollment pause consideration'],
        'high': ['Targeted monitoring increase', 'CRA visit scheduling', 'Site rescue plan initiation'],
        'medium': ['Remote monitoring intensification', 'Additional training', 'Query response deadline enforcement'],
        'low': ['Standard monitoring', 'Routine follow-up', 'Best practice reminders']
    })


# =============================================================================
# DELTA ENGINE - TREND AND VELOCITY TRACKING
# =============================================================================

@dataclass
class DeltaEngineConfig:
    """
    Configuration for the Delta Engine - tracks velocity of change in metrics
    
    Enables predictive flagging: DQI < 75 AND high velocity of negative change
    """
    # Calculation windows
    short_term_window_days: int = 7                  # 1 week rolling average
    medium_term_window_days: int = 28               # 4 week rolling average
    long_term_window_days: int = 84                 # 12 week rolling average
    
    # Velocity thresholds (change per week)
    velocity_critical: float = -10.0                 # Critical negative trend
    velocity_warning: float = -5.0                   # Warning negative trend
    velocity_improving: float = 5.0                  # Significant improvement
    
    # Acceleration tracking (rate of velocity change)
    enable_acceleration: bool = True
    acceleration_warning: float = -2.0               # Accelerating decline
    
    # Persistence settings
    store_historical_snapshots: bool = True
    max_snapshots_per_site: int = 52                # Keep 1 year of weekly snapshots
    
    # Alerting
    alert_on_trend_reversal: bool = True            # Alert when positive goes negative
    alert_threshold_weeks: int = 2                   # Consecutive weeks before alert


# =============================================================================
# GLOBAL CLEANLINESS METER - INTERIM ANALYSIS READINESS
# =============================================================================

@dataclass
class GlobalCleanlinessMeterConfig:
    """
    Configuration for Global Cleanliness Meter
    
    Scientific Question Addressed:
    "Is the snapshot clean enough for interim analysis?"
    
    Outputs definitive YES/NO based on statistician-defined thresholds
    """
    # Power thresholds for interim analysis
    clean_patient_threshold_itt: float = 80.0        # ITT population: >80% clean patients
    clean_patient_threshold_pp: float = 85.0         # Per-Protocol: >85% clean patients
    clean_patient_threshold_safety: float = 95.0     # Safety population: >95% clean
    
    # Population definitions
    populations: List[str] = field(default_factory=lambda: [
        'ITT',           # Intent-to-Treat
        'mITT',          # Modified Intent-to-Treat
        'PP',            # Per-Protocol
        'Safety'         # Safety Population
    ])
    
    # Readiness assessment criteria
    require_all_critical_clean: bool = True          # All critical patients must be clean
    require_primary_endpoint_clean: bool = True      # Primary endpoint data must be clean
    allow_minor_issues: bool = True                  # Allow <5% minor issues
    minor_issue_threshold: float = 5.0               # Percentage threshold
    
    # Output configuration
    output_definitive_answer: bool = True            # YES/NO output
    include_confidence_interval: bool = True         # Include CI in output
    confidence_level: float = 0.95                   # 95% confidence
    
    # Snapshot timing
    interim_analysis_dates: List[str] = field(default_factory=list)
    pre_lock_buffer_days: int = 14                   # Days before lock to assess


# =============================================================================
# ROI AND EFFICIENCY METRICS
# =============================================================================

@dataclass
class ROIMetricsConfig:
    """
    Configuration for Return on Investment tracking
    
    Tracks efficiency gains from automation and quality improvements
    """
    # Efficiency Metrics
    baseline_query_time_hours: float = 2.0           # Hours per manual query
    automated_query_time_hours: float = 0.5          # Hours per automated query
    routine_query_automation_target: float = 0.70    # Target: 70% automation
    
    # Quality Metrics
    baseline_dqi_average: float = 75.0               # Baseline DQI before system
    target_dqi_improvement: float = 15.0             # Target improvement points
    
    # Speed Metrics (Time to Database Lock)
    baseline_lock_months: float = 3.0                # Baseline months to lock
    target_lock_months: float = 2.0                  # Target months to lock
    
    # Financial Impact
    monthly_operational_cost: float = 1000000.0      # $1M per month operational
    daily_revenue_opportunity: float = 1000000.0     # $1M per day early entry
    
    # Tracking settings
    track_dm_workload_reduction: bool = True
    track_error_catch_rate: bool = True
    track_rolling_vs_batch_cleaning: bool = True


# =============================================================================
# SCALABILITY AND CROSS-PLATFORM CONFIGURATION
# =============================================================================

@dataclass
class ScalabilityConfig:
    """
    Configuration for cross-therapeutic and cross-platform scalability
    
    Architecture is metadata-driven, not data-driven
    """
    # Therapeutic Area Configurations
    current_therapeutic_area: str = "Oncology"
    therapeutic_area_configs: Dict[str, Dict] = field(default_factory=lambda: {
        'Oncology': {
            'dqi_weights': {'safety': 0.45, 'efficacy': 0.25, 'visit': 0.15, 'query': 0.15},
            'critical_forms': ['Tumor Assessment', 'Response Evaluation', 'Survival', 'AE', 'SAE'],
            'coding_priority': ['MedDRA', 'WHODRA'],
            'safety_signal_weight': 1.2  # 20% boost for safety
        },
        'Cardiology': {
            'dqi_weights': {'safety': 0.40, 'efficacy': 0.30, 'visit': 0.15, 'query': 0.15},
            'critical_forms': ['ECG', 'Cardiac Events', 'Device Data', 'AE', 'SAE'],
            'coding_priority': ['MedDRA', 'Device Codes'],
            'safety_signal_weight': 1.3  # Cardiac safety higher
        },
        'Neurology': {
            'dqi_weights': {'safety': 0.35, 'efficacy': 0.35, 'visit': 0.15, 'query': 0.15},
            'critical_forms': ['Cognitive Assessment', 'MRI', 'Neurological Exam', 'AE', 'SAE'],
            'coding_priority': ['MedDRA', 'WHODRA'],
            'safety_signal_weight': 1.1
        },
        'Rare_Disease': {
            'dqi_weights': {'safety': 0.40, 'efficacy': 0.30, 'visit': 0.20, 'query': 0.10},
            'critical_forms': ['Disease Assessment', 'Biomarker', 'Quality of Life', 'AE', 'SAE'],
            'coding_priority': ['MedDRA', 'Orphan Drug Codes'],
            'safety_signal_weight': 1.0
        }
    })
    
    # Platform Agnostic Settings (Rave, Veeva, Inform, etc.)
    supported_platforms: List[str] = field(default_factory=lambda: [
        'Medidata Rave',
        'Veeva Vault',
        'Oracle Inform',
        'Oracle Siebel CTMS',
        'Custom EDC'
    ])
    
    # Schema mapping for cross-platform compatibility
    schema_version: str = "2.0"
    enable_auto_schema_detection: bool = True
    fallback_column_mapping: bool = True


# =============================================================================
# QUERY GENERATION AND AUTOMATION
# =============================================================================

@dataclass
class QueryAutomationConfig:
    """
    Configuration for automated query generation
    
    Target: 70% automation of routine queries
    """
    # Query types and automation eligibility
    query_types: Dict[str, Dict] = field(default_factory=lambda: {
        'missing_pages': {'automated': True, 'confidence_required': 0.95, 'template_based': True},
        'missing_visits': {'automated': True, 'confidence_required': 0.95, 'template_based': True},
        'coding_clarification': {'automated': True, 'confidence_required': 0.80, 'template_based': True},
        'simple_reconciliation': {'automated': True, 'confidence_required': 0.90, 'template_based': True},
        'data_discrepancy': {'automated': False, 'confidence_required': 0.99, 'template_based': False},
        'protocol_deviation': {'automated': False, 'confidence_required': 0.99, 'template_based': False},
        'safety_signal': {'automated': False, 'confidence_required': 0.99, 'template_based': False}
    })
    
    # Alert fatigue prevention
    max_queries_per_site_per_day: int = 15
    max_queries_per_patient_per_week: int = 5
    consolidate_similar_queries: bool = True
    
    # Human-in-the-loop settings
    require_approval_above_threshold: int = 5        # More than 5 queries need approval
    batch_approval_enabled: bool = True


# =============================================================================
# REGULATORY COMPLIANCE CONFIGURATION (ICH E6 R2/R3 & 21 CFR Part 11)
# =============================================================================

@dataclass
class RegulatoryComplianceConfig:
    """
    Configuration for regulatory compliance features.
    
    Implements:
    - ICH E6(R2): Risk-based quality management
    - ICH E6(R3) Draft: Technology-enabled monitoring
    - 21 CFR Part 11: Electronic records and signatures
    - FDA AI/ML SaMD guidance: Human oversight requirements
    """
    
    # Compliance Standards Enabled
    enable_ich_e6_r2: bool = True
    enable_ich_e6_r3: bool = True
    enable_21_cfr_part_11: bool = True
    enable_gdpr_logging: bool = False          # Optional for EU studies
    
    # Audit Trail Settings (21 CFR Part 11.10(e))
    audit_trail_enabled: bool = True
    audit_retention_years: int = 15            # Clinical trial data retention
    audit_hash_chain_enabled: bool = True      # Tamper-evident logging
    audit_storage_path: str = "audit_logs"
    
    # Human-in-the-Loop Settings (FDA AI/ML SaMD)
    hitl_enabled: bool = True
    hitl_approval_timeout_hours: int = 24
    hitl_escalation_hours: int = 4
    hitl_auto_approve_low_risk: bool = True
    hitl_auto_approve_medium_risk: bool = True
    hitl_require_approval_high_risk: bool = True
    hitl_require_approval_critical: bool = True
    
    # Agent Identification for Audit (distinct from human users)
    agent_identifier_prefix: str = "System-Agent"
    agent_responsible_field: str = "Responsible LF for action"
    agent_audit_marker: str = "System-Agent-01"
    
    # Risk-Based Approach (ICH E6 R2/R3)
    risk_based_monitoring_enabled: bool = True
    dqi_threshold_red: float = 0.70            # Sites below this are "red quadrant"
    dqi_threshold_yellow: float = 0.85         # Sites below this need attention
    targeted_monitoring_enabled: bool = True    # Focus on high-risk sites
    sdv_reduction_for_green_sites: float = 0.50  # 50% SDV reduction for compliant sites
    
    # Critical Action Types (require HITL per FDA guidance)
    critical_action_types: List[str] = field(default_factory=lambda: [
        'CLOSE_QUERY',
        'LOCK_FORM',
        'UNLOCK_FORM',
        'DELETE_RECORD',
        'MODIFY_SAFETY_DATA',
        'RECONCILIATION_OVERRIDE',
        'PROTOCOL_DEVIATION_FLAG',
        'UNBLINDING_ACTION'
    ])
    
    # Non-Critical Actions (auto-executable per protocol)
    non_critical_action_types: List[str] = field(default_factory=lambda: [
        'GENERATE_QUERY_DRAFT',
        'SEND_REMINDER',
        'CALCULATE_METRICS',
        'GENERATE_REPORT',
        'PROPOSE_CODE',
        'FLAG_FOR_REVIEW'
    ])
    
    # Electronic Signature Requirements (21 CFR Part 11.50)
    require_electronic_signature: bool = False  # Can enable for high-risk
    signature_components: List[str] = field(default_factory=lambda: [
        'printed_name',
        'date_time',
        'meaning'  # e.g., "review", "approval", "responsibility"
    ])
    
    # Data Integrity Settings (ALCOA+ principles)
    ensure_attributable: bool = True
    ensure_legible: bool = True
    ensure_contemporaneous: bool = True
    ensure_original: bool = True
    ensure_accurate: bool = True
    ensure_complete: bool = True
    ensure_consistent: bool = True
    ensure_enduring: bool = True
    ensure_available: bool = True


@dataclass
class RiskBasedMonitoringConfig:
    """
    Configuration for ICH E6 R2/R3 Risk-Based Quality Management.
    
    Implements targeted monitoring based on site risk profiles
    rather than 100% SDV across all sites.
    """
    
    # Risk Indicator Weights (for site risk score calculation)
    risk_weights: Dict[str, float] = field(default_factory=lambda: {
        'open_query_rate': 0.20,
        'overdue_visits': 0.15,
        'safety_reporting_delay': 0.25,
        'data_entry_timeliness': 0.15,
        'protocol_deviations': 0.15,
        'training_compliance': 0.10
    })
    
    # Site Risk Categories (based on DQI scatter plot quadrants)
    site_risk_categories: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'GREEN': {
            'dqi_min': 0.85,
            'sdv_rate': 0.25,           # 25% SDV
            'monitoring_frequency': 'Quarterly',
            'auto_query_enabled': True,
            'escalation_delay_days': 14
        },
        'YELLOW': {
            'dqi_min': 0.70,
            'dqi_max': 0.85,
            'sdv_rate': 0.50,           # 50% SDV
            'monitoring_frequency': 'Monthly',
            'auto_query_enabled': True,
            'escalation_delay_days': 7
        },
        'RED': {
            'dqi_max': 0.70,
            'sdv_rate': 1.00,           # 100% SDV
            'monitoring_frequency': 'Weekly',
            'auto_query_enabled': False,  # Require CRA review
            'escalation_delay_days': 3
        }
    })
    
    # Centralized Monitoring Triggers
    centralized_monitoring_enabled: bool = True
    cmo_alerts: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'enrollment_deviation': {'threshold': 0.20, 'action': 'ALERT'},
        'safety_signal': {'threshold': 0.05, 'action': 'ESCALATE'},
        'data_quality_decline': {'threshold': 0.10, 'action': 'ALERT'},
        'query_rate_spike': {'threshold': 2.0, 'action': 'ALERT'},  # 2x normal
        'protocol_deviation_cluster': {'threshold': 3, 'action': 'ESCALATE'}
    })
    
    # Focused Monitoring Patterns
    focused_monitoring_triggers: List[str] = field(default_factory=lambda: [
        'New Site Activation (first 3 months)',
        'Post-Protocol Amendment',
        'Post-CAP Implementation',
        'Safety Event Cluster',
        'Data Quality Trend Decline'
    ])


# Default configuration instances
DEFAULT_DQI_WEIGHTS = DQIWeights()
DEFAULT_CLEAN_THRESHOLDS = CleanPatientThresholds()
DEFAULT_AGENT_CONFIG = AgentConfig()
DEFAULT_FILE_PATTERNS = FilePatterns()
DEFAULT_GRAPH_CONFIG = GraphConfig()
DEFAULT_FEATURE_ENGINEERING_CONFIG = FeatureEngineeringConfig()

# New enhanced configuration instances
DEFAULT_VISIT_ADHERENCE_CONFIG = VisitAdherenceConfig()
DEFAULT_DQI_HEATMAP_CONFIG = DQIHeatmapConfig()
DEFAULT_SITE_INTERVENTION_CONFIG = SiteInterventionConfig()
DEFAULT_DELTA_ENGINE_CONFIG = DeltaEngineConfig()
DEFAULT_GLOBAL_CLEANLINESS_CONFIG = GlobalCleanlinessMeterConfig()
DEFAULT_ROI_METRICS_CONFIG = ROIMetricsConfig()
DEFAULT_SCALABILITY_CONFIG = ScalabilityConfig()
DEFAULT_QUERY_AUTOMATION_CONFIG = QueryAutomationConfig()

# Zombie SAE Detection (Scenario A)
DEFAULT_ZOMBIE_SAE_CONFIG = ZombieSAEDetectionConfig()

# Regulatory Compliance (ICH E6 R2/R3 & 21 CFR Part 11)
DEFAULT_REGULATORY_COMPLIANCE_CONFIG = RegulatoryComplianceConfig()
DEFAULT_RISK_BASED_MONITORING_CONFIG = RiskBasedMonitoringConfig()

# LongCat AI Integration Configuration
@dataclass
class LongCatConfig:
    """
    Configuration for LongCat AI API integration
    """
    api_key: str = field(default_factory=lambda: os.getenv('API_KEY_Longcat', ''))
    base_url: str = "https://api.longcat.chat"
    model: str = "LongCat-Flash-Chat"
    thinking_model: str = "LongCat-Flash-Thinking-2601"
    max_tokens: int = 4096
    temperature: float = 0.7
    enable_thinking: bool = False
    thinking_budget: int = 1024
    timeout: int = 30
    retry_attempts: int = 3
    connect_timeout: int = 10  # Connection timeout in seconds
    read_timeout: int = 45     # Read timeout in seconds
    cache_ttl: int = 1800      # Cache TTL in seconds (30 minutes)
    use_for_agent_reasoning: bool = True
    use_for_narrative_generation: bool = True
    use_for_anomaly_explanation: bool = True

DEFAULT_LONGCAT_CONFIG = LongCatConfig()

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

@dataclass
class FeatureEngineeringConfig:
    """
    Configuration for engineered feature calculation and processing
    """

    # Operational Velocity Index (Feature 1)
    velocity_enabled: bool = True
    velocity_lookback_days: int = 30
    bottleneck_threshold_net_velocity: float = 0.0  # queries/day
    velocity_smoothing_factor: float = 0.1

    # Normalized Data Density (Feature 2)
    density_enabled: bool = True
    density_normalization_method: str = "percentile"  # "percentile" or "zscore"
    density_percentile_threshold: float = 80.0  # High density threshold
    density_min_queries_for_calculation: int = 5

    # Manipulation Risk Score (Feature 3)
    manipulation_enabled: bool = True
    manipulation_risk_levels: Dict[str, float] = field(default_factory=lambda: {
        'Low': 25.0,
        'Medium': 50.0,
        'High': 75.0,
        'Critical': 90.0
    })
    manipulation_inactivation_threshold: int = 3  # per month
    manipulation_endpoint_risk_weight: float = 0.6

    # Composite Risk Score (Feature 4)
    composite_enabled: bool = True
    composite_weights: Dict[str, float] = field(default_factory=lambda: {
        'velocity_bottleneck': 0.3,
        'density_percentile': 0.2,
        'manipulation_risk': 0.4,
        'intervention_flag': 0.1
    })
    composite_intervention_threshold: float = 60.0

    # Performance and Caching
    enable_feature_caching: bool = True
    feature_cache_ttl_seconds: int = 1800  # 30 minutes
    parallel_processing_enabled: bool = True
    max_workers: int = 4

    # Quality Assurance
    feature_validation_enabled: bool = True
    outlier_detection_method: str = "iqr"  # "iqr", "zscore", or "isolation_forest"
    outlier_threshold: float = 3.0

    # Agent Integration
    agent_feature_boost_enabled: bool = True
    agent_velocity_weight: float = 1.5
    agent_density_weight: float = 1.2
    agent_manipulation_weight: float = 2.0
    agent_composite_weight: float = 1.8

DEFAULT_FEATURE_ENGINEERING_CONFIG = FeatureEngineeringConfig()

# =============================================================================
# PRODUCTION READINESS CONFIGURATION
# =============================================================================

@dataclass
class ProductionConfig:
    """
    Production readiness configuration
    """

    # Response Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 1800  # 30 minutes
    cache_max_size_mb: int = 100
    cache_cleanup_interval_seconds: int = 300  # 5 minutes

    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_monitoring_enabled: bool = True

    # Graceful Degradation
    graceful_degradation_enabled: bool = True
    degradation_check_interval_seconds: int = 30
    service_health_check_enabled: bool = True

    # Performance Monitoring
    performance_monitoring_enabled: bool = True
    performance_log_level: str = "INFO"
    performance_metrics_retention_days: int = 30

    # Error Handling
    error_reporting_enabled: bool = True
    error_retry_attempts: int = 3
    error_retry_backoff_seconds: int = 1

DEFAULT_PRODUCTION_CONFIG = ProductionConfig()
