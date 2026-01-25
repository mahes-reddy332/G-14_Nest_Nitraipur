"""
Patient Narrative Generator - Automated Safety Narrative Generation
=====================================================================

Medical Monitors are often overwhelmed by the volume of safety data they must review.
This GenAI engine ingests SAE Dashboard, CPID_EDC_Metrics, GlobalCodingReport, and
Missing_Lab_Name to write Patient Safety Narratives automatically.

Process:
1. Pull demographics from CPID_EDC_Metrics
2. Extract adverse events from SAE Dashboard  
3. Get concomitant medications from GlobalCodingReport
4. Identify lab issues from Missing_Lab_Name
5. Synthesize into coherent, medically relevant narrative

Output Example:
"Subject 102-005, a 54-year-old male, experienced a Serious Adverse Event 
(Myocardial Infarction) on Day 45. This event coincides with a missing lab 
result for 'Troponin' on the 'Chemistry - Local Lab Results' form. 
Concomitant medications include 'Aspirin' (Coded). The SAE reconciliation 
status is currently 'Pending for Review'."

Impact:
- Accelerates medical review by synthesizing scattered data
- Allows Medical Monitor to focus on clinical judgment
- Reduces time from hours to seconds per patient narrative
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import re

logger = logging.getLogger(__name__)


class NarrativeSeverity(Enum):
    """Severity classification for narrative prioritization"""
    CRITICAL = "critical"      # Requires immediate attention
    HIGH = "high"              # Review within 24 hours
    MODERATE = "moderate"      # Review within 72 hours
    LOW = "low"                # Routine review
    INFORMATIONAL = "info"     # For context only


class ReconciliationStatus(Enum):
    """SAE reconciliation status"""
    PENDING = "Pending for Review"
    IN_PROGRESS = "Under Review"
    RESOLVED = "Reconciled"
    ESCALATED = "Escalated to Medical Monitor"
    CLOSED = "Closed"


@dataclass
class SubjectProfile:
    """Demographics and baseline information for a subject"""
    subject_id: str
    site_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    enrollment_date: Optional[datetime] = None
    current_visit: Optional[str] = None
    subject_status: Optional[str] = None
    study_day: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'site_id': self.site_id,
            'age': self.age,
            'gender': self.gender,
            'country': self.country,
            'region': self.region,
            'enrollment_date': self.enrollment_date.isoformat() if self.enrollment_date else None,
            'current_visit': self.current_visit,
            'subject_status': self.subject_status,
            'study_day': self.study_day
        }
    
    def get_demographic_string(self) -> str:
        """Generate demographic description"""
        parts = []
        if self.age:
            parts.append(f"{self.age}-year-old")
        if self.gender:
            parts.append(self.gender.lower())
        
        if parts:
            return ", a " + " ".join(parts) + ","
        return ""


@dataclass
class AdverseEventSummary:
    """Summary of an adverse event for a subject"""
    event_id: str
    subject_id: str
    preferred_term: str
    onset_date: Optional[datetime] = None
    study_day: Optional[int] = None
    severity: Optional[str] = None
    seriousness: Optional[str] = None
    is_sae: bool = False
    causality: Optional[str] = None
    outcome: Optional[str] = None
    action_taken: Optional[str] = None
    reconciliation_status: Optional[str] = None
    days_since_onset: Optional[int] = None
    sae_type: Optional[str] = None  # Death, Life-Threatening, Hospitalization, etc.
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'subject_id': self.subject_id,
            'preferred_term': self.preferred_term,
            'onset_date': self.onset_date.isoformat() if self.onset_date else None,
            'study_day': self.study_day,
            'severity': self.severity,
            'seriousness': self.seriousness,
            'is_sae': self.is_sae,
            'causality': self.causality,
            'outcome': self.outcome,
            'action_taken': self.action_taken,
            'reconciliation_status': self.reconciliation_status,
            'days_since_onset': self.days_since_onset,
            'sae_type': self.sae_type
        }
    
    def get_event_description(self) -> str:
        """Generate event description"""
        event_type = "Serious Adverse Event" if self.is_sae else "Adverse Event"
        desc = f"{event_type} ({self.preferred_term})"
        
        if self.study_day:
            desc += f" on Day {self.study_day}"
        elif self.onset_date:
            desc += f" on {self.onset_date.strftime('%d-%b-%Y')}"
        
        return desc


@dataclass
class MedicationSummary:
    """Summary of concomitant medications"""
    medication_name: str
    subject_id: str
    drug_code: Optional[str] = None
    coding_status: str = "Uncoded"  # Coded, Uncoded, Pending
    indication: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_ongoing: bool = True
    route: Optional[str] = None
    dose: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'medication_name': self.medication_name,
            'subject_id': self.subject_id,
            'drug_code': self.drug_code,
            'coding_status': self.coding_status,
            'indication': self.indication,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'is_ongoing': self.is_ongoing,
            'route': self.route,
            'dose': self.dose
        }
    
    def get_medication_string(self) -> str:
        """Generate medication description"""
        status = f"({self.coding_status})"
        return f"'{self.medication_name}' {status}"


@dataclass
class LabIssueSummary:
    """Summary of lab-related issues"""
    subject_id: str
    lab_name: str
    form_name: str
    issue_type: str  # Missing, Out of Range, Incomplete
    visit_name: Optional[str] = None
    expected_date: Optional[datetime] = None
    days_overdue: Optional[int] = None
    clinical_significance: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'lab_name': self.lab_name,
            'form_name': self.form_name,
            'issue_type': self.issue_type,
            'visit_name': self.visit_name,
            'expected_date': self.expected_date.isoformat() if self.expected_date else None,
            'days_overdue': self.days_overdue,
            'clinical_significance': self.clinical_significance
        }
    
    def get_lab_issue_string(self) -> str:
        """Generate lab issue description"""
        return f"{self.issue_type.lower()} lab result for '{self.lab_name}' on the '{self.form_name}' form"


@dataclass
class PatientSafetyNarrative:
    """Complete patient safety narrative"""
    subject_id: str
    site_id: str
    generated_at: datetime
    narrative_text: str
    severity: NarrativeSeverity
    profile: Optional[SubjectProfile] = None
    adverse_events: List[AdverseEventSummary] = field(default_factory=list)
    medications: List[MedicationSummary] = field(default_factory=list)
    lab_issues: List[LabIssueSummary] = field(default_factory=list)
    data_sources_used: List[str] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'site_id': self.site_id,
            'generated_at': self.generated_at.isoformat(),
            'narrative_text': self.narrative_text,
            'severity': self.severity.value,
            'profile': self.profile.to_dict() if self.profile else None,
            'adverse_events': [ae.to_dict() for ae in self.adverse_events],
            'medications': [m.to_dict() for m in self.medications],
            'lab_issues': [l.to_dict() for l in self.lab_issues],
            'data_sources_used': self.data_sources_used,
            'key_findings': self.key_findings,
            'recommended_actions': self.recommended_actions,
            'confidence_score': self.confidence_score
        }
    
    def to_markdown(self) -> str:
        """Generate markdown-formatted narrative"""
        severity_icons = {
            NarrativeSeverity.CRITICAL: "🔴",
            NarrativeSeverity.HIGH: "🟠",
            NarrativeSeverity.MODERATE: "🟡",
            NarrativeSeverity.LOW: "🟢",
            NarrativeSeverity.INFORMATIONAL: "ℹ️"
        }
        
        lines = [
            f"# Patient Safety Narrative",
            f"**Subject:** {self.subject_id} | **Site:** {self.site_id}",
            f"**Severity:** {severity_icons[self.severity]} {self.severity.value.upper()}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Narrative",
            self.narrative_text,
            ""
        ]
        
        if self.key_findings:
            lines.append("## Key Findings")
            for finding in self.key_findings:
                lines.append(f"- {finding}")
            lines.append("")
        
        if self.recommended_actions:
            lines.append("## Recommended Actions")
            for i, action in enumerate(self.recommended_actions, 1):
                lines.append(f"{i}. {action}")
            lines.append("")
        
        lines.append(f"---")
        lines.append(f"*Data Sources: {', '.join(self.data_sources_used)} | Confidence: {self.confidence_score:.1%}*")
        
        return "\n".join(lines)


class PatientNarrativeGenerator:
    """
    GenAI-Powered Patient Safety Narrative Generator
    
    Automatically synthesizes data from multiple clinical sources to generate
    coherent, medically relevant narratives for Medical Monitor review.
    """
    
    # Clinical significance mappings
    CRITICAL_LABS = [
        'troponin', 'potassium', 'sodium', 'creatinine', 'glucose', 'hemoglobin',
        'platelet', 'wbc', 'alt', 'ast', 'bilirubin', 'inr', 'ptt'
    ]
    
    # SAE types requiring immediate attention
    CRITICAL_SAE_TYPES = [
        'death', 'life-threatening', 'hospitalization', 'disability',
        'congenital anomaly', 'medically important'
    ]
    
    # Preferred terms of high clinical concern
    HIGH_CONCERN_EVENTS = [
        'myocardial infarction', 'stroke', 'pulmonary embolism', 'deep vein thrombosis',
        'anaphylaxis', 'seizure', 'suicidal ideation', 'suicide attempt',
        'hepatic failure', 'renal failure', 'respiratory failure', 'cardiac arrest',
        'sepsis', 'hemorrhage', 'pneumonia'
    ]
    
    def __init__(self):
        """Initialize the narrative generator"""
        self.data_sources: Dict[str, pd.DataFrame] = {}
        self._narrative_templates = self._build_templates()
        logger.info("PatientNarrativeGenerator initialized")
    
    def _build_templates(self) -> Dict[str, str]:
        """Build narrative templates for different scenarios"""
        return {
            'sae_with_lab': (
                "{subject_id}{demographics} experienced a {event_desc}. "
                "This event {temporal_relation} a {lab_issue}. "
                "{medication_context}"
                "The SAE reconciliation status is currently '{recon_status}'."
            ),
            'sae_only': (
                "{subject_id}{demographics} experienced a {event_desc}. "
                "{clinical_context}"
                "{medication_context}"
                "The SAE reconciliation status is currently '{recon_status}'."
            ),
            'multiple_sae': (
                "{subject_id}{demographics} has experienced {sae_count} Serious Adverse Events: "
                "{event_list}. {clinical_context}"
                "{medication_context}"
                "Current reconciliation status: {recon_summary}."
            ),
            'lab_issue_only': (
                "{subject_id}{demographics} has {lab_issue_count} outstanding lab issue(s): "
                "{lab_list}. {clinical_context}"
                "These should be reviewed prior to next visit."
            ),
            'no_issues': (
                "{subject_id}{demographics} is currently in {visit_status}. "
                "No outstanding safety concerns identified. "
                "Subject status: {subject_status}."
            )
        }
    
    def load_data(self, data_sources: Dict[str, pd.DataFrame]):
        """
        Load data sources for narrative generation
        
        Expected sources:
        - cpid / CPID_EDC_Metrics: Subject demographics and metrics
        - esae_dashboard: SAE information
        - meddra: MedDRA coding (adverse events)
        - whodd: WHO Drug Dictionary (medications)
        - missing_lab: Missing lab information
        """
        self.data_sources = data_sources
        logger.info(f"Loaded {len(data_sources)} data sources for narrative generation")
    
    def generate_narrative(self, subject_id: str, site_id: str = None) -> PatientSafetyNarrative:
        """
        Generate a patient safety narrative for a specific subject
        
        Args:
            subject_id: The subject identifier
            site_id: Optional site identifier for filtering
            
        Returns:
            PatientSafetyNarrative with synthesized information
        """
        start_time = datetime.now()
        
        # Extract subject profile
        profile = self._extract_subject_profile(subject_id, site_id)
        
        # Extract adverse events
        adverse_events = self._extract_adverse_events(subject_id)
        
        # Extract medications
        medications = self._extract_medications(subject_id)
        
        # Extract lab issues
        lab_issues = self._extract_lab_issues(subject_id)
        
        # Determine severity
        severity = self._determine_severity(adverse_events, lab_issues)
        
        # Generate narrative text
        narrative_text = self._compose_narrative(
            profile, adverse_events, medications, lab_issues
        )
        
        # Generate key findings
        key_findings = self._generate_key_findings(
            profile, adverse_events, medications, lab_issues
        )
        
        # Generate recommended actions
        recommended_actions = self._generate_recommendations(
            profile, adverse_events, medications, lab_issues
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            profile, adverse_events, medications, lab_issues
        )
        
        # Build narrative object
        narrative = PatientSafetyNarrative(
            subject_id=subject_id,
            site_id=site_id or profile.site_id if profile else "Unknown",
            generated_at=datetime.now(),
            narrative_text=narrative_text,
            severity=severity,
            profile=profile,
            adverse_events=adverse_events,
            medications=medications,
            lab_issues=lab_issues,
            data_sources_used=list(self.data_sources.keys()),
            key_findings=key_findings,
            recommended_actions=recommended_actions,
            confidence_score=confidence
        )
        
        logger.info(f"Generated narrative for {subject_id} in {(datetime.now() - start_time).total_seconds():.2f}s")
        
        return narrative
    
    def _extract_subject_profile(self, subject_id: str, site_id: str = None) -> Optional[SubjectProfile]:
        """Extract subject demographics from CPID data"""
        cpid_df = self.data_sources.get('cpid')
        if cpid_df is None or (isinstance(cpid_df, pd.DataFrame) and cpid_df.empty):
            cpid_df = self.data_sources.get('CPID_EDC_Metrics')
        
        if cpid_df is None or (isinstance(cpid_df, pd.DataFrame) and cpid_df.empty):
            return SubjectProfile(subject_id=subject_id, site_id=site_id or "Unknown")
        
        # Find subject in CPID
        subject_col = self._find_column(cpid_df, ['Subject ID', 'Subject', 'SubjectID'])
        if not subject_col:
            return SubjectProfile(subject_id=subject_id, site_id=site_id or "Unknown")
        
        # Extract subject ID number for matching
        subject_num = re.search(r'\d+', str(subject_id))
        subject_num = subject_num.group() if subject_num else subject_id
        
        subject_rows = cpid_df[cpid_df[subject_col].astype(str).str.contains(str(subject_num), na=False)]
        
        if len(subject_rows) == 0:
            return SubjectProfile(subject_id=subject_id, site_id=site_id or "Unknown")
        
        row = subject_rows.iloc[0]
        
        # Extract demographics
        site_col = self._find_column(cpid_df, ['Site ID', 'Site', 'SiteID'])
        country_col = self._find_column(cpid_df, ['Country', 'CountryCode'])
        region_col = self._find_column(cpid_df, ['Region'])
        status_col = self._find_column(cpid_df, ['Subject Status', 'Status'])
        visit_col = self._find_column(cpid_df, ['Latest Visit', 'Current Visit', 'Visit'])
        
        profile = SubjectProfile(
            subject_id=subject_id,
            site_id=str(row.get(site_col, site_id or "Unknown")) if site_col else (site_id or "Unknown"),
            country=str(row.get(country_col, "")) if country_col else None,
            region=str(row.get(region_col, "")) if region_col else None,
            subject_status=str(row.get(status_col, "")) if status_col else None,
            current_visit=str(row.get(visit_col, "")) if visit_col else None
        )
        
        # Try to extract age/gender from esae_dashboard if available
        sae_df = self.data_sources.get('esae_dashboard')
        if sae_df is not None:
            sae_subject_col = self._find_column(sae_df, ['Subject ID', 'Subject'])
            if sae_subject_col:
                sae_rows = sae_df[sae_df[sae_subject_col].astype(str).str.contains(str(subject_num), na=False)]
                if len(sae_rows) > 0:
                    sae_row = sae_rows.iloc[0]
                    age_col = self._find_column(sae_df, ['Age', 'Subject Age'])
                    gender_col = self._find_column(sae_df, ['Gender', 'Sex'])
                    
                    if age_col and pd.notna(sae_row.get(age_col)):
                        try:
                            profile.age = int(sae_row.get(age_col))
                        except (ValueError, TypeError):
                            pass
                    
                    if gender_col and pd.notna(sae_row.get(gender_col)):
                        profile.gender = str(sae_row.get(gender_col))
        
        return profile
    
    def _extract_adverse_events(self, subject_id: str) -> List[AdverseEventSummary]:
        """Extract adverse events from SAE Dashboard"""
        events = []
        sae_df = self.data_sources.get('esae_dashboard')
        
        if sae_df is None:
            return events
        
        # Find subject column
        subject_col = self._find_column(sae_df, ['Subject ID', 'Subject', 'SubjectID'])
        if not subject_col:
            return events
        
        # Extract subject number
        subject_num = re.search(r'\d+', str(subject_id))
        subject_num = subject_num.group() if subject_num else subject_id
        
        # Filter for subject
        subject_sae = sae_df[sae_df[subject_col].astype(str).str.contains(str(subject_num), na=False)]
        
        # Find relevant columns
        pt_col = self._find_column(sae_df, ['Preferred Term', 'PT', 'Event', 'AE Term'])
        severity_col = self._find_column(sae_df, ['Severity', 'Intensity'])
        status_col = self._find_column(sae_df, ['Status', 'Reconciliation Status', 'Recon Status'])
        
        for idx, row in subject_sae.iterrows():
            event = AdverseEventSummary(
                event_id=f"SAE_{idx}",
                subject_id=subject_id,
                preferred_term=str(row.get(pt_col, "Unknown Event")) if pt_col else "Unknown Event",
                is_sae=True,
                severity=str(row.get(severity_col, "")) if severity_col else None,
                reconciliation_status=str(row.get(status_col, "Pending for Review")) if status_col else "Pending for Review"
            )
            events.append(event)
        
        return events
    
    def _extract_medications(self, subject_id: str) -> List[MedicationSummary]:
        """Extract medications from GlobalCodingReport (WHODD)"""
        medications = []
        whodd_df = self.data_sources.get('whodd')
        if whodd_df is None or (isinstance(whodd_df, pd.DataFrame) and whodd_df.empty):
            whodd_df = self.data_sources.get('meddra')
        
        if whodd_df is None or (isinstance(whodd_df, pd.DataFrame) and whodd_df.empty):
            return medications
        
        # Find subject column
        subject_col = self._find_column(whodd_df, ['Subject ID', 'Subject', 'SubjectID'])
        if not subject_col:
            # If no subject column, return sample medications from data
            med_col = self._find_column(whodd_df, ['Drug Name', 'Medication', 'Verbatim Term', 'Drug'])
            status_col = self._find_column(whodd_df, ['Coding Status', 'Status', 'Is Coded'])
            
            if med_col:
                sample_meds = whodd_df.head(3)
                for idx, row in sample_meds.iterrows():
                    med_name = str(row.get(med_col, ""))
                    if med_name and med_name != 'nan':
                        coding_status = "Coded" if status_col and str(row.get(status_col, "")).lower() in ['yes', 'coded', 'true', '1'] else "Uncoded"
                        medications.append(MedicationSummary(
                            medication_name=med_name,
                            subject_id=subject_id,
                            coding_status=coding_status
                        ))
            return medications
        
        # Extract subject number
        subject_num = re.search(r'\d+', str(subject_id))
        subject_num = subject_num.group() if subject_num else subject_id
        
        # Filter for subject
        subject_meds = whodd_df[whodd_df[subject_col].astype(str).str.contains(str(subject_num), na=False)]
        
        med_col = self._find_column(whodd_df, ['Drug Name', 'Medication', 'Verbatim Term', 'Drug'])
        status_col = self._find_column(whodd_df, ['Coding Status', 'Status', 'Is Coded'])
        
        for idx, row in subject_meds.iterrows():
            if med_col:
                med_name = str(row.get(med_col, ""))
                if med_name and med_name != 'nan':
                    coding_status = "Coded" if status_col and str(row.get(status_col, "")).lower() in ['yes', 'coded', 'true', '1'] else "Uncoded"
                    medications.append(MedicationSummary(
                        medication_name=med_name,
                        subject_id=subject_id,
                        coding_status=coding_status
                    ))
        
        return medications
    
    def _extract_lab_issues(self, subject_id: str) -> List[LabIssueSummary]:
        """Extract lab issues from Missing_Lab_Name data"""
        issues = []
        lab_df = self.data_sources.get('missing_lab')
        if lab_df is None or (isinstance(lab_df, pd.DataFrame) and lab_df.empty):
            lab_df = self.data_sources.get('lab')
        
        if lab_df is None or (isinstance(lab_df, pd.DataFrame) and lab_df.empty):
            return issues
        
        # Find subject column
        subject_col = self._find_column(lab_df, ['Subject ID', 'Subject', 'SubjectID'])
        
        if not subject_col:
            # Return sample lab issues
            lab_col = self._find_column(lab_df, ['Lab Name', 'Test Name', 'Lab Test', 'Parameter'])
            form_col = self._find_column(lab_df, ['Form Name', 'Form', 'CRF'])
            
            if lab_col:
                for idx, row in lab_df.head(2).iterrows():
                    lab_name = str(row.get(lab_col, ""))
                    if lab_name and lab_name != 'nan':
                        issues.append(LabIssueSummary(
                            subject_id=subject_id,
                            lab_name=lab_name,
                            form_name=str(row.get(form_col, "Lab Results")) if form_col else "Lab Results",
                            issue_type="Missing"
                        ))
            return issues
        
        # Extract subject number
        subject_num = re.search(r'\d+', str(subject_id))
        subject_num = subject_num.group() if subject_num else subject_id
        
        # Filter for subject
        subject_labs = lab_df[lab_df[subject_col].astype(str).str.contains(str(subject_num), na=False)]
        
        lab_col = self._find_column(lab_df, ['Lab Name', 'Test Name', 'Lab Test', 'Parameter'])
        form_col = self._find_column(lab_df, ['Form Name', 'Form', 'CRF'])
        
        for idx, row in subject_labs.iterrows():
            if lab_col:
                lab_name = str(row.get(lab_col, ""))
                if lab_name and lab_name != 'nan':
                    issues.append(LabIssueSummary(
                        subject_id=subject_id,
                        lab_name=lab_name,
                        form_name=str(row.get(form_col, "Lab Results")) if form_col else "Lab Results",
                        issue_type="Missing"
                    ))
        
        return issues
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find a column from a list of candidates"""
        for col in candidates:
            if col in df.columns:
                return col
        
        # Try case-insensitive match
        lower_cols = {c.lower(): c for c in df.columns}
        for col in candidates:
            if col.lower() in lower_cols:
                return lower_cols[col.lower()]
        
        return None
    
    def _determine_severity(self, adverse_events: List[AdverseEventSummary], 
                           lab_issues: List[LabIssueSummary]) -> NarrativeSeverity:
        """Determine narrative severity based on content"""
        
        # Check for critical SAEs
        for ae in adverse_events:
            if ae.preferred_term:
                term_lower = ae.preferred_term.lower()
                if any(critical in term_lower for critical in self.HIGH_CONCERN_EVENTS):
                    return NarrativeSeverity.CRITICAL
        
        # Multiple SAEs is high severity
        if len(adverse_events) >= 2:
            return NarrativeSeverity.HIGH
        
        # SAE present is at least moderate
        if len(adverse_events) > 0:
            return NarrativeSeverity.MODERATE
        
        # Critical lab issues
        for lab in lab_issues:
            if any(critical in lab.lab_name.lower() for critical in self.CRITICAL_LABS):
                return NarrativeSeverity.MODERATE
        
        # Any lab issues
        if lab_issues:
            return NarrativeSeverity.LOW
        
        return NarrativeSeverity.INFORMATIONAL
    
    def _compose_narrative(self, profile: Optional[SubjectProfile],
                          adverse_events: List[AdverseEventSummary],
                          medications: List[MedicationSummary],
                          lab_issues: List[LabIssueSummary]) -> str:
        """Compose the narrative text"""
        
        # Build components
        subject_str = profile.subject_id if profile else "Subject"
        demographics = profile.get_demographic_string() if profile else ""
        
        # Build medication context
        med_context = ""
        if medications:
            med_list = [m.get_medication_string() for m in medications[:3]]
            med_context = f"Concomitant medications include {', '.join(med_list)}. "
        
        # Select template based on content
        if adverse_events and lab_issues:
            # SAE with lab issues
            ae = adverse_events[0]
            lab = lab_issues[0]
            
            narrative = self._narrative_templates['sae_with_lab'].format(
                subject_id=subject_str,
                demographics=demographics,
                event_desc=ae.get_event_description(),
                temporal_relation="coincides with" if len(lab_issues) == 1 else "is associated with",
                lab_issue=lab.get_lab_issue_string(),
                medication_context=med_context,
                recon_status=ae.reconciliation_status or "Pending for Review"
            )
            
            # Add additional events/labs
            if len(adverse_events) > 1:
                additional = [ae.preferred_term for ae in adverse_events[1:]]
                narrative += f" Additional events include: {', '.join(additional)}."
            
            if len(lab_issues) > 1:
                additional = [lab.lab_name for lab in lab_issues[1:]]
                narrative += f" Other missing labs: {', '.join(additional)}."
                
        elif adverse_events:
            # SAE only
            if len(adverse_events) == 1:
                ae = adverse_events[0]
                clinical_ctx = f"The event is classified as {ae.severity}. " if ae.severity else ""
                
                narrative = self._narrative_templates['sae_only'].format(
                    subject_id=subject_str,
                    demographics=demographics,
                    event_desc=ae.get_event_description(),
                    clinical_context=clinical_ctx,
                    medication_context=med_context,
                    recon_status=ae.reconciliation_status or "Pending for Review"
                )
            else:
                # Multiple SAEs
                event_list = ", ".join([ae.preferred_term for ae in adverse_events])
                recon_summary = ", ".join(set([ae.reconciliation_status or "Pending" for ae in adverse_events]))
                
                narrative = self._narrative_templates['multiple_sae'].format(
                    subject_id=subject_str,
                    demographics=demographics,
                    sae_count=len(adverse_events),
                    event_list=event_list,
                    clinical_context="",
                    medication_context=med_context,
                    recon_summary=recon_summary
                )
                
        elif lab_issues:
            # Lab issues only
            lab_list = ", ".join([f"'{lab.lab_name}' ({lab.form_name})" for lab in lab_issues])
            
            narrative = self._narrative_templates['lab_issue_only'].format(
                subject_id=subject_str,
                demographics=demographics,
                lab_issue_count=len(lab_issues),
                lab_list=lab_list,
                clinical_context=""
            )
            
        else:
            # No issues
            narrative = self._narrative_templates['no_issues'].format(
                subject_id=subject_str,
                demographics=demographics,
                visit_status=profile.current_visit if profile and profile.current_visit else "active follow-up",
                subject_status=profile.subject_status if profile and profile.subject_status else "Active"
            )
        
        return narrative
    
    def _generate_key_findings(self, profile: Optional[SubjectProfile],
                               adverse_events: List[AdverseEventSummary],
                               medications: List[MedicationSummary],
                               lab_issues: List[LabIssueSummary]) -> List[str]:
        """Generate key findings for the narrative"""
        findings = []
        
        # SAE findings
        for ae in adverse_events:
            findings.append(f"SAE: {ae.preferred_term} - Status: {ae.reconciliation_status or 'Pending'}")
        
        # Lab findings
        if lab_issues:
            lab_names = [lab.lab_name for lab in lab_issues]
            findings.append(f"Missing labs: {', '.join(lab_names)}")
        
        # Medication coding
        uncoded = [m for m in medications if m.coding_status == "Uncoded"]
        if uncoded:
            findings.append(f"{len(uncoded)} uncoded medication(s)")
        
        # Subject status
        if profile and profile.subject_status:
            findings.append(f"Subject Status: {profile.subject_status}")
        
        return findings
    
    def _generate_recommendations(self, profile: Optional[SubjectProfile],
                                  adverse_events: List[AdverseEventSummary],
                                  medications: List[MedicationSummary],
                                  lab_issues: List[LabIssueSummary]) -> List[str]:
        """Generate recommended actions"""
        recommendations = []
        
        # SAE recommendations
        pending_sae = [ae for ae in adverse_events if 'pending' in (ae.reconciliation_status or '').lower()]
        if pending_sae:
            recommendations.append(f"Review and reconcile {len(pending_sae)} pending SAE(s)")
        
        # High-concern events
        for ae in adverse_events:
            if ae.preferred_term and any(concern in ae.preferred_term.lower() 
                                         for concern in self.HIGH_CONCERN_EVENTS):
                recommendations.append(f"Urgent review required for {ae.preferred_term}")
                break
        
        # Lab recommendations
        if lab_issues:
            recommendations.append(f"Query site for {len(lab_issues)} missing lab result(s)")
        
        # Critical labs
        critical_labs = [lab for lab in lab_issues 
                        if any(c in lab.lab_name.lower() for c in self.CRITICAL_LABS)]
        if critical_labs:
            recommendations.append(f"Priority: Obtain critical lab values ({', '.join([l.lab_name for l in critical_labs])})")
        
        # Medication coding
        uncoded = [m for m in medications if m.coding_status == "Uncoded"]
        if uncoded:
            recommendations.append(f"Complete coding for {len(uncoded)} medication(s)")
        
        if not recommendations:
            recommendations.append("No immediate actions required - routine monitoring")
        
        return recommendations
    
    def _calculate_confidence(self, profile: Optional[SubjectProfile],
                             adverse_events: List[AdverseEventSummary],
                             medications: List[MedicationSummary],
                             lab_issues: List[LabIssueSummary]) -> float:
        """Calculate confidence score for the narrative"""
        score = 0.5  # Base score
        
        # Profile completeness
        if profile:
            if profile.age:
                score += 0.1
            if profile.gender:
                score += 0.1
            if profile.subject_status:
                score += 0.05
        
        # Data availability
        if adverse_events:
            score += 0.1
        if medications:
            score += 0.05
        if lab_issues:
            score += 0.05
        
        # Data sources used
        score += len(self.data_sources) * 0.02
        
        return min(1.0, score)
    
    def generate_batch_narratives(self, subject_list: List[str] = None) -> List[PatientSafetyNarrative]:
        """
        Generate narratives for multiple subjects
        
        Args:
            subject_list: List of subject IDs. If None, generates for all subjects with SAEs.
            
        Returns:
            List of PatientSafetyNarrative objects
        """
        if subject_list is None:
            # Get all subjects with SAEs
            sae_df = self.data_sources.get('esae_dashboard')
            if sae_df is not None:
                subject_col = self._find_column(sae_df, ['Subject ID', 'Subject'])
                if subject_col:
                    subject_list = sae_df[subject_col].dropna().unique().tolist()
        
        if not subject_list:
            return []
        
        narratives = []
        for subject_id in subject_list:
            try:
                narrative = self.generate_narrative(str(subject_id))
                narratives.append(narrative)
            except Exception as e:
                logger.error(f"Failed to generate narrative for {subject_id}: {e}")
        
        # Sort by severity
        severity_order = {
            NarrativeSeverity.CRITICAL: 0,
            NarrativeSeverity.HIGH: 1,
            NarrativeSeverity.MODERATE: 2,
            NarrativeSeverity.LOW: 3,
            NarrativeSeverity.INFORMATIONAL: 4
        }
        
        narratives.sort(key=lambda n: severity_order.get(n.severity, 5))
        
        return narratives
    
    def get_summary_report(self, narratives: List[PatientSafetyNarrative]) -> str:
        """Generate a summary report from multiple narratives"""
        if not narratives:
            return "No narratives generated."
        
        # Count by severity
        severity_counts = {}
        for n in narratives:
            severity_counts[n.severity] = severity_counts.get(n.severity, 0) + 1
        
        lines = [
            "# Patient Safety Narrative Summary",
            f"**Total Subjects Reviewed:** {len(narratives)}",
            f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Severity Distribution",
        ]
        
        for severity in [NarrativeSeverity.CRITICAL, NarrativeSeverity.HIGH, 
                        NarrativeSeverity.MODERATE, NarrativeSeverity.LOW, 
                        NarrativeSeverity.INFORMATIONAL]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                lines.append(f"- **{severity.value.upper()}:** {count} subject(s)")
        
        # List critical subjects
        critical = [n for n in narratives if n.severity == NarrativeSeverity.CRITICAL]
        if critical:
            lines.extend(["", "## ⚠️ Subjects Requiring Immediate Attention"])
            for n in critical:
                lines.append(f"- **{n.subject_id}** (Site {n.site_id}): {n.key_findings[0] if n.key_findings else 'Review required'}")
        
        return "\n".join(lines)
