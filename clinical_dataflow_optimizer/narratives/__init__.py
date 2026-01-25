"""
Narratives Module - GenAI-Powered Report Generation
=====================================================

Automated generation of clinical narratives and monitoring reports:

1. Patient Safety Narratives - For Medical Monitors
   - Synthesizes SAE Dashboard, CPID_EDC_Metrics, GlobalCodingReport, Missing_Lab_Name
   - Produces coherent medical narratives for safety review

2. RBM Report Generation - For CRAs
   - Analyzes CPID_EDC_Metrics for compliance risks
   - Drafts CRA Visit Letters and Monitoring Reports

3. Generative Narrative Engine - AI-Powered Dynamic Generation
   - LLM-powered narrative synthesis
   - Regulatory compliance validation
   - Multiple report types (patient safety, site performance, executive summary)

Impact:
- Accelerates medical review by synthesizing scattered data
- Moves CRAs from "detective" to "solver" role
- Reduces administrative burden on clinical teams
"""

from .patient_narrative_generator import (
    PatientNarrativeGenerator,
    PatientSafetyNarrative,
    SubjectProfile,
    AdverseEventSummary,
    MedicationSummary,
    LabIssueSummary,
    NarrativeSeverity
)

from .rbm_report_generator import (
    RBMReportGenerator,
    MonitoringReport,
    CRAVisitLetter,
    SiteRiskProfile,
    ComplianceIssue,
    RiskCategory,
    IssueType
)

from .generative_narrative_engine import (
    GenerativeNarrativeEngine,
    GeneratedNarrative,
    NarrativeContext,
    NarrativeType,
    NarrativeSeverity as GenNarrativeSeverity,
    generate_narrative
)

__all__ = [
    # Patient Narrative Components
    'PatientNarrativeGenerator',
    'PatientSafetyNarrative',
    'SubjectProfile',
    'AdverseEventSummary',
    'MedicationSummary',
    'LabIssueSummary',
    'NarrativeSeverity',
    
    # RBM Report Components
    'RBMReportGenerator',
    'MonitoringReport',
    'CRAVisitLetter',
    'SiteRiskProfile',
    'ComplianceIssue',
    'RiskCategory',
    'IssueType',
    
    # Generative Narrative Engine
    'GenerativeNarrativeEngine',
    'GeneratedNarrative',
    'NarrativeContext',
    'NarrativeType',
    'GenNarrativeSeverity',
    'generate_narrative'
]
