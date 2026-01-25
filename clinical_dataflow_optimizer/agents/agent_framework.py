"""
Agentic AI Framework for Clinical Trial Data Management
Implements the Autonomous Data Steward multi-agent system

Agents:
- Rex (Reconciliation Agent): Ensures concordance between Clinical and Safety databases
- Codex (Coding Agent): Automates medical coding with human-in-the-loop
- Lia (Site Liaison Agent): Proactive site management and visit compliance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
from abc import ABC, abstractmethod

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.data_models import (
    DigitalPatientTwin, SiteMetrics, AgentActionLog, AgentAction,
    BlockingItem, RiskLevel, RiskMetrics
)
from config.settings import AgentConfig, DEFAULT_AGENT_CONFIG
from core.longcat_integration import longcat_client

logger = logging.getLogger(__name__)


class ActionPriority(Enum):
    """Priority levels for agent actions"""
    CRITICAL = 1  # Immediate action required
    HIGH = 2      # Same day action
    MEDIUM = 3    # Within 48 hours
    LOW = 4       # Routine/batch processing


class ActionType(Enum):
    """Types of actions agents can take"""
    QUERY_TO_SITE = "Query to Site"
    QUERY_TO_DM = "Query to Data Manager"
    QUERY_TO_SAFETY = "Query to Safety Team"
    ALERT_CRA = "Alert CRA"
    ALERT_MEDICAL_MONITOR = "Alert Medical Monitor"
    AUTO_CODE = "Auto-Code Term"
    PROPOSE_CODE = "Propose Code for Review"
    REQUEST_CLARIFICATION = "Request Clarification"
    SEND_REMINDER = "Send Reminder"
    ESCALATE = "Escalate Issue"
    FLAG_FOR_AUDIT = "Flag for Audit"


@dataclass
class AgentRecommendation:
    """Represents a recommendation from an agent"""
    recommendation_id: str
    agent_name: str
    action_type: ActionType
    priority: ActionPriority
    subject_id: str
    site_id: str
    study_id: str
    title: str
    description: str
    rationale: str
    confidence_score: float
    requires_human_approval: bool
    auto_executable: bool
    source_data: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.recommendation_id,
            'agent': self.agent_name,
            'action_type': self.action_type.value,
            'priority': self.priority.name,
            'subject_id': self.subject_id,
            'site_id': self.site_id,
            'study_id': self.study_id,
            'title': self.title,
            'description': self.description,
            'rationale': self.rationale,
            'confidence': round(self.confidence_score, 2),
            'requires_approval': self.requires_human_approval,
            'auto_executable': self.auto_executable,
            'created_at': self.created_at.isoformat()
        }


class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, name: str, config: AgentConfig = None):
        self.name = name
        self.config = config or DEFAULT_AGENT_CONFIG
        self.recommendations: List[AgentRecommendation] = []
        self.action_log: List[AgentActionLog] = []
    
    @abstractmethod
    def analyze(self, *args, **kwargs) -> List[AgentRecommendation]:
        """Analyze data and generate recommendations"""
        pass
    
    def _generate_id(self) -> str:
        """Generate unique ID for recommendations"""
        return f"{self.name}_{uuid.uuid4().hex[:8]}"
    
    def _create_recommendation(
        self,
        action_type: ActionType,
        priority: ActionPriority,
        subject_id: str,
        site_id: str,
        study_id: str,
        title: str,
        description: str,
        rationale: str,
        confidence: float = 0.9,
        requires_approval: bool = True,
        auto_executable: bool = False,
        source_data: Dict = None
    ) -> AgentRecommendation:
        """Create a new recommendation"""
        return AgentRecommendation(
            recommendation_id=self._generate_id(),
            agent_name=self.name,
            action_type=action_type,
            priority=priority,
            subject_id=subject_id,
            site_id=site_id,
            study_id=study_id,
            title=title,
            description=description,
            rationale=rationale,
            confidence_score=confidence,
            requires_human_approval=requires_approval,
            auto_executable=auto_executable,
            source_data=source_data or {}
        )
    
    def log_action(
        self,
        action_type: AgentAction,
        subject_id: str,
        site_id: str,
        study_id: str,
        description: str,
        confidence: float = None,
        requires_review: bool = False
    ) -> AgentActionLog:
        """Log an action taken by the agent"""
        log = AgentActionLog(
            action_id=self._generate_id(),
            agent_name=self.name,
            action_type=action_type,
            subject_id=subject_id,
            site_id=site_id,
            study_id=study_id,
            description=description,
            confidence_score=confidence,
            requires_human_review=requires_review
        )
        self.action_log.append(log)
        return log

    def enhance_reasoning_with_longcat(self, context: str, task: str, data: Dict) -> str:
        """
        Use LongCat AI to enhance agent reasoning and decision-making
        """
        from config.settings import DEFAULT_LONGCAT_CONFIG
        if not DEFAULT_LONGCAT_CONFIG.use_for_agent_reasoning:
            return "LongCat reasoning disabled in configuration"

        try:
            return longcat_client.generate_agent_reasoning(context, task, data)
        except Exception as e:
            logger.warning(f"LongCat reasoning failed: {e}")
            return f"Standard reasoning: {task}"


class ReconciliationAgent(BaseAgent):
    """
    Rex - The Reconciliation Agent
    
    Primary Objective: Ensure concordance between Clinical and Safety databases
    
    Triggers:
    1. Discrepancies between CPID_EDC_Metrics (# eSAE dashboard review for DM) and SAE Dashboard (Review Status)
    2. SAE records with "Pending for Review" status > threshold days (default 7)
    3. "Zombie SAEs" - events in safety DB but missing corresponding AE form in EDC
    
    Actions:
    - Draft queries to Safety Team
    - Flag Medical Monitor for urgent cases
    - Auto-generate site queries for Zombie SAEs
    - Cross-reference EDRR for subjects with high "Total Open Issue Count"
    
    Enhanced with ZombieSAEDetector for comprehensive Scenario A handling:
    - Full pipeline: SAE Dashboard → CPID Cross-Reference → Missing Pages Verification
    - Auto-query generation with ICH E6 R2 compliance references
    - EDRR update tracking for risk metric propagation
    """
    
    # Query templates for auto-generated messages
    ZOMBIE_SAE_QUERY = (
        "SAE recorded in Safety DB on {sae_date} but AE form missing in EDC. "
        "Please reconcile. Subject: {subject_id}, Site: {site_id}."
    )
    
    PENDING_SAE_QUERY = (
        "SAE has been pending review for {days_pending} days (Threshold: {threshold} days). "
        "Subject: {subject_id}, SAE Event: {event_type}. Immediate action required."
    )
    
    RECON_DISCREPANCY_QUERY = (
        "Discrepancy detected between EDC and Safety database. "
        "CPID shows {cpid_count} eSAE review items, SAE Dashboard shows {sae_pending} pending. "
        "Please investigate and reconcile."
    )
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("Rex", config)
        self._sae_aging_cache: Dict[str, List[Dict]] = {}
        self._edrr_cache: Dict[str, Dict] = {}
        self._zombie_sae_detector = None  # Lazy initialized
        self._zombie_sae_cases: List = []
    
    def analyze(
        self,
        twins: List[DigitalPatientTwin],
        sae_data: Optional[pd.DataFrame] = None,
        edrr_data: Optional[pd.DataFrame] = None,
        missing_pages: Optional[pd.DataFrame] = None,
        cpid_data: Optional[pd.DataFrame] = None,
        study_id: str = ""
    ) -> List[AgentRecommendation]:
        """
        Analyze safety reconciliation status and generate recommendations
        
        Analysis Pipeline:
        1. Check CPID reconciliation issues (EDC vs Safety DB concordance)
        2. Scan SAE Dashboard for records pending > threshold days
        3. Cross-reference EDRR for high-issue subjects
        4. Detect Zombie SAEs via ZombieSAEDetector (Scenario A)
        5. Generate prioritized action items
        
        Enhanced Zombie SAE Detection (Scenario A):
        - Scans SAE Dashboard for Action Status = "Pending"
        - Cross-references with CPID_EDC_Metrics (# eSAE dashboard review for DM)
        - Verifies against Global_Missing_Pages_Report for missing AE forms
        - Auto-generates site queries
        - Updates EDRR Total Open Issue Count
        """
        self.recommendations = []
        self._zombie_sae_cases = []
        
        # Build EDRR lookup cache for cross-referencing
        self._build_edrr_cache(edrr_data)
        
        # Analyze SAE aging if SAE data available
        if sae_data is not None:
            self._analyze_sae_aging(sae_data, twins, study_id)
        
        # *** ENHANCED: Zombie SAE Detection (Scenario A) ***
        # Full pipeline: SAE Dashboard → CPID → Missing Pages → Auto-Query → EDRR Update
        if sae_data is not None and cpid_data is not None:
            self._detect_zombie_saes(
                sae_dashboard=sae_data,
                cpid_data=cpid_data,
                missing_pages=missing_pages,
                edrr_data=edrr_data,
                study_id=study_id
            )
        
        for twin in twins:
            # Check 1: Reconciliation Issues from CPID
            # Trigger: Discrepancy between # eSAE dashboard review for DM and SAE Dashboard
            if twin.reconciliation_issues > 0:
                # Cross-reference with EDRR for severity assessment
                edrr_issues = self._edrr_cache.get(twin.subject_id, {}).get('total_issues', 0)
                severity_boost = edrr_issues >= 5  # High EDRR issue count = higher priority
                
                description = self.RECON_DISCREPANCY_QUERY.format(
                    cpid_count=twin.reconciliation_issues,
                    sae_pending=len(twin.sae_records) if twin.sae_records else 0
                )
                
                # Generate enhanced rationale using LongCat AI
                enhanced_rationale = self.enhance_reasoning_with_longcat(
                    context="Safety reconciliation analysis for clinical trial",
                    task="Analyze reconciliation discrepancy and determine appropriate action",
                    data={
                        'subject_id': twin.subject_id,
                        'reconciliation_issues': twin.reconciliation_issues,
                        'edrr_issues': edrr_issues,
                        'sae_records': len(twin.sae_records) if twin.sae_records else 0,
                        'severity_boost': severity_boost
                    }
                )
                
                # Combine original rationale with AI enhancement
                base_rationale = f"Reconciliation issues indicate potential missing SAE reports. " + \
                               (f"Subject also has {edrr_issues} open EDRR issues." if severity_boost else 
                                "Must be resolved for regulatory compliance.")
                
                final_rationale = base_rationale
                if not enhanced_rationale.startswith("Standard reasoning:"):
                    final_rationale += f"\n\nAI Enhanced Analysis: {enhanced_rationale}"
                
                rec = self._create_recommendation(
                    action_type=ActionType.QUERY_TO_SAFETY,
                    priority=ActionPriority.CRITICAL if severity_boost else ActionPriority.HIGH,
                    subject_id=twin.subject_id,
                    site_id=twin.site_id,
                    study_id=study_id,
                    title=f"Safety Reconciliation Required" + (" [HIGH EDRR]" if severity_boost else ""),
                    description=description,
                    rationale=final_rationale,
                    confidence=0.95,
                    requires_approval=False,
                    auto_executable=True,
                    source_data={
                        'reconciliation_issues': twin.reconciliation_issues,
                        'edrr_issues': edrr_issues,
                        'source': 'CPID_EDC_Metrics + Compiled_EDRR'
                    }
                )
                self.recommendations.append(rec)
            
            # Check 2: SAE records with pending review - Now uses cached aging analysis
            aged_saes = self._sae_aging_cache.get(twin.subject_id, [])
            if aged_saes:
                for aged_sae in aged_saes:
                    days_pending = aged_sae.get('days_pending', 0)
                    event_type = aged_sae.get('event_type', 'Unknown SAE')
                    
                    # Determine priority based on days pending
                    if days_pending >= 14:  # Critical: > 14 days
                        priority = ActionPriority.CRITICAL
                        action = ActionType.ALERT_MEDICAL_MONITOR
                    elif days_pending >= 7:  # High: 7-14 days
                        priority = ActionPriority.HIGH
                        action = ActionType.QUERY_TO_SAFETY
                    else:
                        continue  # Skip if under threshold
                    
                    description = self.PENDING_SAE_QUERY.format(
                        days_pending=days_pending,
                        threshold=self.config.sae_pending_days_threshold,
                        subject_id=twin.subject_id,
                        event_type=event_type
                    )
                    
                    # Generate enhanced rationale using LongCat AI
                    enhanced_rationale = self.enhance_reasoning_with_longcat(
                        context="SAE review timeliness analysis for clinical trial safety monitoring",
                        task="Evaluate SAE pending review duration and determine appropriate escalation",
                        data={
                            'subject_id': twin.subject_id,
                            'days_pending': days_pending,
                            'event_type': event_type,
                            'threshold': self.config.sae_pending_days_threshold,
                            'priority_level': priority.name
                        }
                    )
                    
                    # Combine original rationale with AI enhancement
                    base_rationale = f"SAE pending > {self.config.sae_pending_days_threshold} days impacts regulatory reporting timelines (ICH E6 R2)."
                    
                    final_rationale = base_rationale
                    if not enhanced_rationale.startswith("Standard reasoning:"):
                        final_rationale += f"\n\nAI Enhanced Analysis: {enhanced_rationale}"
                    
                    rec = self._create_recommendation(
                        action_type=action,
                        priority=priority,
                        subject_id=twin.subject_id,
                        site_id=twin.site_id,
                        study_id=study_id,
                        title=f"SAE Pending Review: {days_pending} days ({event_type})",
                        description=description,
                        rationale=final_rationale,
                        confidence=0.98,
                        requires_approval=priority == ActionPriority.CRITICAL,
                        auto_executable=priority != ActionPriority.CRITICAL,
                        source_data=aged_sae
                    )
                    self.recommendations.append(rec)
            
            # Check 3: Check for "Zombie SAEs" - safety db entry but missing EDC form
            # This requires cross-referencing SAE data with missing pages
            if twin.sae_records and missing_pages is not None:
                self._check_zombie_saes(twin, missing_pages, study_id)
        
        # Check EDRR data for high-risk subjects
        if edrr_data is not None:
            self._analyze_edrr(edrr_data, twins, study_id)
        
        logger.info(f"Rex generated {len(self.recommendations)} recommendations")
        return self.recommendations
    
    def _build_edrr_cache(self, edrr_data: Optional[pd.DataFrame]) -> None:
        """Build EDRR lookup cache for cross-referencing"""
        self._edrr_cache = {}
        if edrr_data is None:
            return
        
        # Standardize column names
        if edrr_data.columns.duplicated().any():
            edrr_data = edrr_data.loc[:, ~edrr_data.columns.duplicated()]
        
        subject_col = None
        for col in ['subject_id', 'Subject ID', 'SubjectID']:
            if col in edrr_data.columns:
                subject_col = col
                break
        
        if subject_col is None:
            return
        
        for _, row in edrr_data.iterrows():
            subject_id = str(row.get(subject_col, ''))
            if subject_id:
                # Look for total issues column
                total_issues = 0
                for col in ['total_issues', 'Total Open issue Count', 'Total Issues']:
                    if col in row and not pd.isna(row[col]):
                        total_issues = int(row[col])
                        break
                
                self._edrr_cache[subject_id] = {
                    'total_issues': total_issues,
                    'raw_data': row.to_dict() if hasattr(row, 'to_dict') else {}
                }
    
    def _analyze_sae_aging(self, sae_data: pd.DataFrame, twins: List[DigitalPatientTwin], study_id: str) -> None:
        """
        Analyze SAE Dashboard for records with Review Status = 'Pending for Review'
        and pending > threshold days
        
        This implements the core Rex trigger logic:
        - Scan SAE Dashboard
        - Identify records where Review Status is 'Pending for Review'
        - Check if pending > 7 days (configurable threshold)
        """
        self._sae_aging_cache = {}
        
        if sae_data is None or sae_data.empty:
            return
        
        # Handle duplicate columns
        if sae_data.columns.duplicated().any():
            sae_data = sae_data.loc[:, ~sae_data.columns.duplicated()]
        
        # Find relevant columns
        subject_col = None
        for col in ['subject_id', 'Subject ID', 'SubjectID', 'Patient ID']:
            if col in sae_data.columns:
                subject_col = col
                break
        
        review_status_col = None
        for col in ['review_status', 'Review Status', 'ReviewStatus']:
            if col in sae_data.columns:
                review_status_col = col
                break
        
        date_col = None
        for col in ['event_date', 'Event Date', 'EventDate', 'Created Date']:
            if col in sae_data.columns:
                date_col = col
                break
        
        if subject_col is None or review_status_col is None:
            logger.warning("SAE data missing required columns (subject_id, review_status)")
            return
        
        # Filter for pending review status
        pending_mask = sae_data[review_status_col].astype(str).str.lower().str.contains(
            'pending|awaiting|under review', na=False
        )
        pending_saes = sae_data[pending_mask]
        
        # Calculate days pending for each SAE
        today = datetime.now()
        threshold_days = self.config.sae_pending_days_threshold
        
        for _, row in pending_saes.iterrows():
            subject_id = str(row.get(subject_col, ''))
            if not subject_id:
                continue
            
            # Calculate days pending
            days_pending = threshold_days + 1  # Default to over threshold if no date
            if date_col and date_col in row and not pd.isna(row[date_col]):
                try:
                    event_date = pd.to_datetime(row[date_col])
                    days_pending = (today - event_date).days
                except:
                    pass
            
            # Only flag if over threshold
            if days_pending >= threshold_days:
                if subject_id not in self._sae_aging_cache:
                    self._sae_aging_cache[subject_id] = []
                
                # Extract event type/description
                event_type = 'Unknown'
                for col in ['sae_type', 'SAE Type', 'Event Type', 'Description']:
                    if col in row and not pd.isna(row[col]):
                        event_type = str(row[col])[:50]  # Truncate
                        break
                
                self._sae_aging_cache[subject_id].append({
                    'days_pending': days_pending,
                    'review_status': str(row.get(review_status_col, '')),
                    'event_type': event_type,
                    'event_date': str(row.get(date_col, '')) if date_col else None,
                    'subject_id': subject_id,
                    'threshold': threshold_days
                })
        
        logger.info(f"Rex identified {sum(len(v) for v in self._sae_aging_cache.values())} SAEs pending > {threshold_days} days")
    
    def _check_zombie_saes(
        self,
        twin: DigitalPatientTwin,
        missing_pages: pd.DataFrame,
        study_id: str
    ):
        """
        Check for 'Zombie SAEs' - events recorded in safety database but missing
        the corresponding 'Adverse Event' form in the EDC
        
        Verification via Global_Missing_Pages_Report:
        - Look for AE/SAE related form names in missing pages
        - Cross-reference with SAE records from twin
        - Auto-generate query to site if zombie detected
        """
        # Handle duplicate columns in missing_pages
        if missing_pages.columns.duplicated().any():
            missing_pages = missing_pages.loc[:, ~missing_pages.columns.duplicated()]
        
        # Find subject column
        subject_col = None
        for col in ['subject_id', 'Subject ID', 'SubjectID']:
            if col in missing_pages.columns:
                subject_col = col
                break
        
        if subject_col is None:
            return
        
        subject_missing = missing_pages[
            missing_pages[subject_col].astype(str) == str(twin.subject_id)
        ]
        
        if len(subject_missing) == 0:
            return
        
        # Find form name column
        form_col = None
        for col in ['form_name', 'Form Name', 'FormName', 'Page Name']:
            if col in subject_missing.columns:
                form_col = col
                break
        
        if form_col is None:
            return
        
        # Look for AE/SAE related forms that are missing
        ae_patterns = ['adverse', 'ae ', ' ae', 'sae', 'safety event', 'serious adverse']
        ae_mask = subject_missing[form_col].astype(str).str.lower().apply(
            lambda x: any(p in x for p in ae_patterns)
        )
        ae_forms = subject_missing[ae_mask]
        
        if len(ae_forms) > 0 and twin.sae_records and len(twin.sae_records) > 0:
            # Found potential Zombie SAE!
            # Get the earliest SAE date for the query
            sae_date = "Unknown Date"
            for sae in twin.sae_records:
                if sae.get('event_date'):
                    sae_date = str(sae.get('event_date'))[:10]
                    break
            
            # Auto-generate the query message
            auto_query = self.ZOMBIE_SAE_QUERY.format(
                sae_date=sae_date,
                subject_id=twin.subject_id,
                site_id=twin.site_id
            )
            
            missing_form_names = ae_forms[form_col].tolist()[:5]
            
            # Generate enhanced rationale using LongCat AI
            enhanced_rationale = self.enhance_reasoning_with_longcat(
                context="Zombie SAE detection and regulatory compliance analysis",
                task="Analyze missing AE form for reported SAE and assess regulatory implications",
                data={
                    'subject_id': twin.subject_id,
                    'sae_count': len(twin.sae_records),
                    'missing_forms': missing_form_names,
                    'sae_date': sae_date,
                    'regulatory_context': 'ICH E6 R2 5.18.4 - Safety Reporting'
                }
            )
            
            # Combine original rationale with AI enhancement
            base_rationale = f"SAE exists in Safety DB ({len(twin.sae_records)} records) but corresponding " + \
                           f"AE form(s) missing in EDC: {', '.join(missing_form_names)}. " + \
                           "This creates regulatory and data integrity risks (ICH E6 R2 5.18.4)."
            
            final_rationale = base_rationale
            if not enhanced_rationale.startswith("Standard reasoning:"):
                final_rationale += f"\n\nAI Enhanced Analysis: {enhanced_rationale}"
            
            rec = self._create_recommendation(
                action_type=ActionType.QUERY_TO_SITE,
                priority=ActionPriority.CRITICAL,
                subject_id=twin.subject_id,
                site_id=twin.site_id,
                study_id=study_id,
                title="[ZOMBIE SAE] EDC Form Missing",
                description=auto_query,
                rationale=final_rationale,
                confidence=0.92,
                requires_approval=False,  # Auto-executable per spec
                auto_executable=True,
                source_data={
                    'sae_count': len(twin.sae_records),
                    'missing_forms': missing_form_names,
                    'sae_date': sae_date,
                    'auto_generated_query': auto_query,
                    'verification_source': 'Global_Missing_Pages_Report'
                }
            )
            self.recommendations.append(rec)
    
    def _detect_zombie_saes(
        self,
        sae_dashboard: pd.DataFrame,
        cpid_data: pd.DataFrame,
        missing_pages: Optional[pd.DataFrame],
        edrr_data: Optional[pd.DataFrame],
        study_id: str
    ) -> None:
        """
        Enhanced Zombie SAE Detection - Implements Scenario A
        
        Full detection pipeline:
        1. Data Check: Scan SAE Dashboard for Action Status = "Pending"
        2. Cross-Reference: Query CPID_EDC_Metrics for each Subject ID
        3. Logic Gate: Check "# eSAE dashboard review for DM" column
        4. Verification: Cross-check Global_Missing_Pages_Report for AE form
        5. Action: Auto-generate site queries
        6. Update: Track EDRR issue count updates
        
        This method uses the ZombieSAEDetector for comprehensive analysis
        and converts detected cases into agent recommendations.
        """
        try:
            from core.zombie_sae_detector import ZombieSAEDetector, ZombieSAEStatus
        except ImportError:
            try:
                from clinical_dataflow_optimizer.core.zombie_sae_detector import ZombieSAEDetector, ZombieSAEStatus
            except ImportError:
                logger.warning("[Rex] ZombieSAEDetector not available - skipping enhanced detection")
                return
        
        # Initialize detector if needed
        if self._zombie_sae_detector is None:
            self._zombie_sae_detector = ZombieSAEDetector()
        
        # Run detection pipeline
        zombie_cases = self._zombie_sae_detector.detect(
            sae_dashboard=sae_dashboard,
            cpid_data=cpid_data,
            missing_pages=missing_pages,
            edrr_data=edrr_data,
            study_id=study_id
        )
        
        self._zombie_sae_cases = zombie_cases
        
        # Convert detected cases to agent recommendations
        for case in zombie_cases:
            if case.status in [ZombieSAEStatus.CONFIRMED, ZombieSAEStatus.SUSPECTED]:
                # Determine priority based on status and confidence
                if case.status == ZombieSAEStatus.CONFIRMED and case.confidence_score >= 0.90:
                    priority = ActionPriority.CRITICAL
                elif case.confidence_score >= 0.75:
                    priority = ActionPriority.HIGH
                else:
                    priority = ActionPriority.MEDIUM
                
                # Create recommendation with auto-query
                rec = self._create_recommendation(
                    action_type=ActionType.QUERY_TO_SITE,
                    priority=priority,
                    subject_id=case.patient_id,
                    site_id=case.site_id,
                    study_id=study_id,
                    title=f"[ZOMBIE SAE] {case.status.value.upper()} - Discrepancy {case.discrepancy_id}",
                    description=case.auto_query_text if case.auto_query_text else (
                        f"SAE in Safety DB but missing AE form in EDC for Subject {case.patient_id}. "
                        f"Please enter data or clarify."
                    ),
                    rationale=(
                        f"Zombie SAE detected via comprehensive pipeline:\n"
                        f"- SAE Dashboard: {case.sae_review_status}\n"
                        f"- CPID eSAE DM Count: {case.cpid_esae_dm_count}\n"
                        f"- Missing AE Forms: {', '.join(case.missing_ae_forms) if case.missing_ae_forms else 'None found'}\n"
                        f"- Confidence: {case.confidence_score:.0%}\n"
                        f"ICH E6 R2 5.18.4 compliance required."
                    ),
                    confidence=case.confidence_score,
                    requires_approval=False,  # Auto-executable per Scenario A spec
                    auto_executable=True,
                    source_data={
                        'case_id': case.case_id,
                        'discrepancy_id': case.discrepancy_id,
                        'cpid_esae_dm_count': case.cpid_esae_dm_count,
                        'cpid_esae_safety_count': case.cpid_esae_safety_count,
                        'cpid_recon_issues': case.cpid_recon_issues,
                        'missing_ae_forms': case.missing_ae_forms,
                        'days_ae_missing': case.days_ae_missing,
                        'edrr_new_issue_count': case.edrr_new_issue_count,
                        'detection_pipeline': 'SAE_Dashboard → CPID → Missing_Pages → Auto_Query → EDRR_Update',
                        'source': 'ZombieSAEDetector'
                    }
                )
                self.recommendations.append(rec)
                
                # Log action for audit trail
                self.log_action(
                    action_type=AgentAction.GENERATE_QUERY,
                    subject_id=case.patient_id,
                    site_id=case.site_id,
                    study_id=study_id,
                    description=f"Zombie SAE auto-query generated for Discrepancy {case.discrepancy_id}",
                    confidence=case.confidence_score,
                    requires_review=False
                )
        
        # Log summary
        summary = self._zombie_sae_detector.get_summary_report()
        logger.info(
            f"[Rex] Zombie SAE Detection: {summary['confirmed']} confirmed, "
            f"{summary['suspected']} suspected, {summary['queries_to_send']} queries ready"
        )
    
    def get_zombie_sae_summary(self) -> Dict:
        """Get summary of detected Zombie SAE cases"""
        if self._zombie_sae_detector:
            return self._zombie_sae_detector.get_summary_report()
        return {'total_cases': 0, 'message': 'Zombie SAE detection not run'}
    
    def get_zombie_sae_auto_queries(self) -> List[Dict]:
        """Get auto-generated queries for Zombie SAE cases"""
        if self._zombie_sae_detector:
            return self._zombie_sae_detector.get_auto_queries()
        return []
    
    def get_zombie_sae_edrr_updates(self) -> List[Dict]:
        """Get EDRR updates required for Zombie SAE cases"""
        if self._zombie_sae_detector:
            return self._zombie_sae_detector.get_edrr_updates()
        return []
    
    def _analyze_edrr(
        self,
        edrr_data: pd.DataFrame,
        twins: List[DigitalPatientTwin],
        study_id: str
    ):
        """Analyze EDRR data for high-risk subjects"""
        if 'total_issues' not in edrr_data.columns:
            return
        
        # Find subjects with high issue counts
        twin_subjects = {t.subject_id: t for t in twins}
        
        for _, row in edrr_data.iterrows():
            subject_id = str(row.get('subject_id', ''))
            total_issues = row.get('total_issues', 0)
            
            if pd.isna(total_issues):
                total_issues = 0
            
            if total_issues >= 5 and subject_id in twin_subjects:
                twin = twin_subjects[subject_id]
                rec = self._create_recommendation(
                    action_type=ActionType.FLAG_FOR_AUDIT,
                    priority=ActionPriority.HIGH,
                    subject_id=subject_id,
                    site_id=twin.site_id,
                    study_id=study_id,
                    title=f"High EDRR Issue Count ({int(total_issues)} issues)",
                    description=f"Subject {subject_id} has {int(total_issues)} open issues in EDRR. Consider targeted review.",
                    rationale="High issue counts may indicate systematic data quality problems requiring intervention.",
                    confidence=0.90,
                    source_data={'total_issues': int(total_issues)}
                )
                self.recommendations.append(rec)


class CodingAgent(BaseAgent):
    """
    Codex - The Coding Agent
    
    Primary Objective: Automate the mapping of verbatim terms to standardized dictionaries
    (MedDRA for Adverse Events/Medical History, WHO Drug for Concomitant Medications)
    
    Triggers:
    1. # Uncoded Terms > 0 in CPID_EDC_Metrics
    2. Coding Status = "UnCoded Term" in GlobalCodingReport_MedDRA or GlobalCodingReport_WHODRA
    
    Action Logic:
    - Ingests Verbatim Term + Context (Form OID determines AE, MH, CM, etc.)
    - Queries simulated LLM dictionary matching for best LLT/PT or Trade Name
    
    Human-in-the-Loop Protocol:
    - High Confidence (>95%): Auto-apply code, update Coding Status
    - Medium Confidence (80-95%): Propose code for single-click dashboard approval
    - Low Confidence (<80%): Raise query to site for clarification
    """
    
    # Query templates for site clarification
    AMBIGUOUS_TERM_QUERY = (
        "Term '{verbatim}' in {context} form is ambiguous. "
        "Please provide specific {term_type} to enable accurate coding."
    )
    
    ILLEGIBLE_TERM_QUERY = (
        "Term '{verbatim}' appears illegible or contains non-standard abbreviations. "
        "Please clarify the intended term for Subject {subject_id}."
    )
    
    UNKNOWN_TERM_QUERY = (
        "Term '{verbatim}' not found in {dictionary} dictionary. "
        "Please verify spelling or provide the standard Trade Name/Generic Name."
    )
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("Codex", config)
        
        # Extended MedDRA term mappings (LLT -> PT)
        # Simulates LLM fine-tuned on MedDRA dictionary
        self.meddra_mappings = {
            # Common adverse events
            'headache': {'llt': 'Headache', 'pt': 'Headache', 'soc': 'Nervous system disorders', 'confidence': 0.99},
            'severe headache': {'llt': 'Headache severe', 'pt': 'Headache', 'soc': 'Nervous system disorders', 'confidence': 0.97},
            'migraine': {'llt': 'Migraine', 'pt': 'Migraine', 'soc': 'Nervous system disorders', 'confidence': 0.98},
            'severe migraine': {'llt': 'Migraine aggravated', 'pt': 'Migraine', 'soc': 'Nervous system disorders', 'confidence': 0.95},
            'nausea': {'llt': 'Nausea', 'pt': 'Nausea', 'soc': 'Gastrointestinal disorders', 'confidence': 0.99},
            'vomiting': {'llt': 'Vomiting', 'pt': 'Vomiting', 'soc': 'Gastrointestinal disorders', 'confidence': 0.99},
            'fatigue': {'llt': 'Fatigue', 'pt': 'Fatigue', 'soc': 'General disorders', 'confidence': 0.98},
            'tiredness': {'llt': 'Fatigue', 'pt': 'Fatigue', 'soc': 'General disorders', 'confidence': 0.96},
            'dizziness': {'llt': 'Dizziness', 'pt': 'Dizziness', 'soc': 'Nervous system disorders', 'confidence': 0.98},
            'vertigo': {'llt': 'Vertigo', 'pt': 'Vertigo', 'soc': 'Ear disorders', 'confidence': 0.97},
            'rash': {'llt': 'Rash', 'pt': 'Rash', 'soc': 'Skin disorders', 'confidence': 0.96},
            'skin rash': {'llt': 'Rash', 'pt': 'Rash', 'soc': 'Skin disorders', 'confidence': 0.97},
            'insomnia': {'llt': 'Insomnia', 'pt': 'Insomnia', 'soc': 'Psychiatric disorders', 'confidence': 0.98},
            'anxiety': {'llt': 'Anxiety', 'pt': 'Anxiety', 'soc': 'Psychiatric disorders', 'confidence': 0.97},
            'depression': {'llt': 'Depression', 'pt': 'Depression', 'soc': 'Psychiatric disorders', 'confidence': 0.96},
            'cough': {'llt': 'Cough', 'pt': 'Cough', 'soc': 'Respiratory disorders', 'confidence': 0.98},
            'fever': {'llt': 'Pyrexia', 'pt': 'Pyrexia', 'soc': 'General disorders', 'confidence': 0.97},
            'pyrexia': {'llt': 'Pyrexia', 'pt': 'Pyrexia', 'soc': 'General disorders', 'confidence': 0.99},
            'hypertension': {'llt': 'Hypertension', 'pt': 'Hypertension', 'soc': 'Vascular disorders', 'confidence': 0.98},
            'diabetes': {'llt': 'Diabetes mellitus', 'pt': 'Diabetes mellitus', 'soc': 'Metabolism disorders', 'confidence': 0.95},
            'back pain': {'llt': 'Back pain', 'pt': 'Back pain', 'soc': 'Musculoskeletal disorders', 'confidence': 0.97},
            'joint pain': {'llt': 'Arthralgia', 'pt': 'Arthralgia', 'soc': 'Musculoskeletal disorders', 'confidence': 0.94},
            'arthralgia': {'llt': 'Arthralgia', 'pt': 'Arthralgia', 'soc': 'Musculoskeletal disorders', 'confidence': 0.99},
        }
        
        # Extended WHO Drug mappings (Trade Name -> ATC)
        # Simulates LLM fine-tuned on WHO Drug dictionary
        self.whodrug_mappings = {
            # Common medications
            'aspirin': {'trade_name': 'Aspirin', 'generic': 'Acetylsalicylic acid', 'atc': 'B01AC06', 'confidence': 0.99},
            'ibuprofen': {'trade_name': 'Advil', 'generic': 'Ibuprofen', 'atc': 'M01AE01', 'confidence': 0.99},
            'advil': {'trade_name': 'Advil', 'generic': 'Ibuprofen', 'atc': 'M01AE01', 'confidence': 0.98},
            'paracetamol': {'trade_name': 'Tylenol', 'generic': 'Paracetamol', 'atc': 'N02BE01', 'confidence': 0.99},
            'acetaminophen': {'trade_name': 'Tylenol', 'generic': 'Paracetamol', 'atc': 'N02BE01', 'confidence': 0.98},
            'tylenol': {'trade_name': 'Tylenol', 'generic': 'Paracetamol', 'atc': 'N02BE01', 'confidence': 0.99},
            'metformin': {'trade_name': 'Glucophage', 'generic': 'Metformin', 'atc': 'A10BA02', 'confidence': 0.98},
            'atorvastatin': {'trade_name': 'Lipitor', 'generic': 'Atorvastatin', 'atc': 'C10AA05', 'confidence': 0.98},
            'lipitor': {'trade_name': 'Lipitor', 'generic': 'Atorvastatin', 'atc': 'C10AA05', 'confidence': 0.99},
            'omeprazole': {'trade_name': 'Prilosec', 'generic': 'Omeprazole', 'atc': 'A02BC01', 'confidence': 0.98},
            'lisinopril': {'trade_name': 'Zestril', 'generic': 'Lisinopril', 'atc': 'C09AA03', 'confidence': 0.97},
            'amlodipine': {'trade_name': 'Norvasc', 'generic': 'Amlodipine', 'atc': 'C08CA01', 'confidence': 0.98},
            'losartan': {'trade_name': 'Cozaar', 'generic': 'Losartan', 'atc': 'C09CA01', 'confidence': 0.97},
            'gabapentin': {'trade_name': 'Neurontin', 'generic': 'Gabapentin', 'atc': 'N03AX12', 'confidence': 0.98},
            'prednisone': {'trade_name': 'Deltasone', 'generic': 'Prednisone', 'atc': 'H02AB07', 'confidence': 0.97},
            'multivitamin': {'trade_name': 'Centrum', 'generic': 'Multivitamins', 'atc': 'A11AA03', 'confidence': 0.92},
            'vitamin d': {'trade_name': 'Vitamin D3', 'generic': 'Colecalciferol', 'atc': 'A11CC05', 'confidence': 0.94},
        }
        
        # Ambiguous terms requiring clarification (low confidence)
        self.ambiguous_patterns = [
            'pain', 'ache', 'discomfort', 'unwell', 'sick', 'other', 'unknown',
            'medication', 'drug', 'tablet', 'pill', 'capsule', 'medicine',
            'supplement', 'vitamin', 'herbal', 'traditional', 'home remedy',
            'n/a', 'none', 'nothing', 'see above', 'as above', 'same'
        ]
        
        # Patterns suggesting illegible or non-standard text
        self.illegible_patterns = [
            r'^[a-z]{1,2}$',  # Single or two letters
            r'^\d+$',         # Only numbers
            r'^[^a-zA-Z]+$',  # No letters at all
            r'.*\?\?.*',      # Contains ??
            r'.*illegible.*',
            r'.*unclear.*',
            r'.*unreadable.*'
        ]
        
        # Context mapping (Form OID -> Context type)
        self.context_map = {
            'AEG': 'Adverse Event',
            'AE': 'Adverse Event',
            'MHG': 'Medical History',
            'MH': 'Medical History',
            'CMG': 'Concomitant Medication',
            'CM': 'Concomitant Medication',
            'PRG': 'Prior Medication',
            'PR': 'Prior Medication'
        }
        
        # Ambiguous Coding Detector (Scenario B) - lazy initialized
        self._ambiguous_detector = None
        self._ambiguous_terms: List = []
    
    def analyze(
        self,
        twins: List[DigitalPatientTwin],
        meddra_data: Optional[pd.DataFrame] = None,
        whodra_data: Optional[pd.DataFrame] = None,
        cpid_data: Optional[pd.DataFrame] = None,
        study_id: str = ""
    ) -> List[AgentRecommendation]:
        """
        Analyze uncoded terms and generate coding recommendations
        
        Analysis Pipeline:
        1. Check CPID for patients with # Uncoded Terms > 0
        2. Scan GlobalCodingReport_MedDRA for Coding Status = 'UnCoded Term'
        3. Scan GlobalCodingReport_WHODRA for Coding Status = 'UnCoded Term'
        4. For each uncoded term, determine confidence and action
        
        Enhanced (Scenario B - Ambiguous Coding):
        5. Detect ambiguous terms that are drug classes (e.g., "Pain killer")
        6. Use LLM-simulated assessment for term specificity
        7. Auto-generate clarification queries for low confidence terms
        8. Track clarifications for learning
        """
        self.recommendations = []
        self._ambiguous_terms = []
        self._coding_stats = {
            'auto_coded': 0,
            'proposed': 0,
            'clarification_needed': 0,
            'total_uncoded': 0
        }
        
        # Process MedDRA uncoded terms directly from report
        if meddra_data is not None:
            self._process_coding_report(meddra_data, 'MedDRA', twins, study_id)
        
        # Process WHODRA uncoded terms directly from report  
        if whodra_data is not None:
            self._process_coding_report(whodra_data, 'WHODRA', twins, study_id)
            
            # *** ENHANCED: Ambiguous Coding Detection (Scenario B) ***
            # Full pipeline: WHODRA scan → LLM assessment → clarification workflow
            if self.config.ambiguous_coding_enabled:
                self._detect_ambiguous_coding(whodra_data, cpid_data, study_id)
        
        # Also check twins for any additional uncoded terms from CPID
        self._process_twin_uncoded_terms(twins, study_id)
        
        logger.info(f"Codex generated {len(self.recommendations)} recommendations")
        logger.info(f"Codex stats: {self._coding_stats}")
        return self.recommendations
    
    def _detect_ambiguous_coding(
        self,
        whodra_data: pd.DataFrame,
        cpid_data: Optional[pd.DataFrame],
        study_id: str
    ) -> None:
        """
        Enhanced Ambiguous Coding Detection - Implements Scenario B
        
        Full detection pipeline:
        1. Data Check: Scan GlobalCodingReport_WHODRA for Coding Status = "UnCoded Term"
        2. LLM Query: Assess if verbatim term is specific enough for WHODRA coding
        3. Reasoning: Classify confidence (High >95%, Medium 80-95%, Low <80%)
        4. Action: Auto-code, propose for approval, or trigger clarification workflow
        5. Learning: Track clarifications for probability weight updates
        """
        try:
            from core.ambiguous_coding_detector import (
                AmbiguousCodingDetector, CodingConfidence, AmbiguityLevel
            )
        except ImportError:
            try:
                from clinical_dataflow_optimizer.core.ambiguous_coding_detector import (
                    AmbiguousCodingDetector, CodingConfidence, AmbiguityLevel
                )
            except ImportError:
                logger.warning("[Codex] AmbiguousCodingDetector not available - skipping Scenario B")
                return
        
        # Initialize detector if needed
        if self._ambiguous_detector is None:
            self._ambiguous_detector = AmbiguousCodingDetector()
        
        # Run detection pipeline
        ambiguous_terms = self._ambiguous_detector.detect(
            whodra_data=whodra_data,
            cpid_data=cpid_data,
            study_id=study_id
        )
        
        self._ambiguous_terms = ambiguous_terms
        
        # Convert detected terms to agent recommendations
        for term in ambiguous_terms:
            if term.confidence == CodingConfidence.HIGH:
                # High confidence: Auto-apply code
                if term.suggested_matches:
                    suggested = term.suggested_matches[0]
                    # Generate enhanced rationale using LongCat AI
                    enhanced_rationale = self.enhance_reasoning_with_longcat(
                        context="Medical coding decision analysis for clinical trial data",
                        task="Evaluate auto-coding recommendation for adverse event term",
                        data={
                            'verbatim_term': term.verbatim_term,
                            'suggested_code': suggested.get('generic', 'Unknown'),
                            'confidence_score': term.confidence_score,
                            'llm_assessment': term.llm_assessment,
                            'dictionary': 'WHODRA',
                            'threshold': 0.95
                        }
                    )
                    
                    # Combine original rationale with AI enhancement
                    base_rationale = f"Confidence {term.confidence_score*100:.0f}% exceeds auto-apply threshold (95%)."
                    
                    final_rationale = base_rationale
                    if not enhanced_rationale.startswith("Standard reasoning:"):
                        final_rationale += f"\n\nAI Enhanced Analysis: {enhanced_rationale}"
                    
                    rec = self._create_recommendation(
                        action_type=ActionType.AUTO_CODE,
                        priority=ActionPriority.LOW,
                        subject_id=term.subject_id,
                        site_id=term.site_id,
                        study_id=study_id,
                        title=f"[CODEX] Auto-Code: {term.verbatim_term[:25]} -> {suggested.get('generic', 'Unknown')}",
                        description=(
                            f"High confidence match. Auto-applying code:\n"
                            f"Verbatim: '{term.verbatim_term}'\n"
                            f"Coded: '{suggested.get('generic', 'Unknown')}' (WHODRA)\n"
                            f"ATC: {suggested.get('atc', 'N/A')}"
                        ),
                        rationale=final_rationale,
                        confidence=term.confidence_score,
                        requires_approval=False,
                        auto_executable=True,
                        source_data={
                            'verbatim': term.verbatim_term,
                            'suggested_code': suggested,
                            'form_oid': term.form_oid,
                            'logline': term.logline,
                            'llm_assessment': term.llm_assessment,
                            'source': 'AmbiguousCodingDetector'
                        }
                    )
                    self.recommendations.append(rec)
                    self._coding_stats['auto_coded'] += 1
                    
            elif term.confidence == CodingConfidence.MEDIUM:
                # Medium confidence: Propose for single-click approval
                if term.suggested_matches:
                    suggested = term.suggested_matches[0]
                    rec = self._create_recommendation(
                        action_type=ActionType.PROPOSE_CODE,
                        priority=ActionPriority.MEDIUM,
                        subject_id=term.subject_id,
                        site_id=term.site_id,
                        study_id=study_id,
                        title=f"[CODEX] Proposed: {term.verbatim_term[:25]} -> {suggested.get('generic', 'Unknown')}",
                        description=(
                            f"Medium confidence match. Requires approval:\n"
                            f"Verbatim: '{term.verbatim_term}'\n"
                            f"Proposed Code: '{suggested.get('generic', 'Unknown')}'\n"
                            f"[APPROVE] [REJECT] [MODIFY]"
                        ),
                        rationale=f"Confidence {term.confidence_score*100:.0f}% is medium (80-95%). Human verification recommended.",
                        confidence=term.confidence_score,
                        requires_approval=True,
                        auto_executable=False,
                        source_data={
                            'verbatim': term.verbatim_term,
                            'proposed_code': suggested,
                            'form_oid': term.form_oid,
                            'logline': term.logline,
                            'llm_assessment': term.llm_assessment,
                            'source': 'AmbiguousCodingDetector'
                        }
                    )
                    self.recommendations.append(rec)
                    self._coding_stats['proposed'] += 1
                    
            else:
                # Low confidence: Request clarification
                rec = self._create_recommendation(
                    action_type=ActionType.REQUEST_CLARIFICATION,
                    priority=ActionPriority.MEDIUM,
                    subject_id=term.subject_id,
                    site_id=term.site_id,
                    study_id=study_id,
                    title=f"[CODEX] Ambiguous Term: {term.verbatim_term[:30]}",
                    description=term.auto_query if term.auto_query else (
                        f"Term '{term.verbatim_term}' is too vague for WHODRA coding. "
                        f"Please provide specific Trade Name or Generic Name."
                    ),
                    rationale=(
                        f"LLM Assessment: {term.llm_assessment}\n"
                        f"Confidence: {term.confidence_score*100:.0f}% (below 80% threshold)\n"
                        f"Reason: {term.reason.name}\n"
                        f"WHO Drug Dictionary requires specific medication names."
                    ),
                    confidence=term.confidence_score,
                    requires_approval=False,
                    auto_executable=True,  # Auto-send clarification query
                    source_data={
                        'verbatim': term.verbatim_term,
                        'form_oid': term.form_oid,
                        'field_oid': term.field_oid,
                        'logline': term.logline,
                        'ambiguity_level': term.ambiguity_level.name,
                        'reason': term.reason.name,
                        'suggested_matches': term.suggested_matches,
                        'auto_query': term.auto_query,
                        'source': 'AmbiguousCodingDetector'
                    }
                )
                self.recommendations.append(rec)
                self._coding_stats['clarification_needed'] += 1
        
        # Log summary
        summary = self._ambiguous_detector.get_summary_report()
        logger.info(
            f"[Codex] Ambiguous Coding Detection: {summary['breakdown']['high_confidence_auto_code']} auto-code, "
            f"{summary['breakdown']['medium_confidence_proposed']} proposed, "
            f"{summary['breakdown']['low_confidence_clarification']} clarifications"
        )
    
    def get_ambiguous_coding_summary(self) -> Dict:
        """Get summary of ambiguous coding detection"""
        if self._ambiguous_detector:
            return self._ambiguous_detector.get_summary_report()
        return {'total_uncoded': 0, 'message': 'Ambiguous coding detection not run'}
    
    def get_clarification_queries(self) -> List[Dict]:
        """Get auto-generated clarification queries for ambiguous terms"""
        if self._ambiguous_detector:
            return self._ambiguous_detector.get_clarification_queries()
        return []
    
    def get_proposed_codes(self) -> List[Dict]:
        """Get proposed codes requiring single-click approval"""
        if self._ambiguous_detector:
            return self._ambiguous_detector.get_proposed_codes()
        return []
    
    def learn_from_clarification(
        self,
        verbatim: str,
        resolved_term: str,
        generic_name: str,
        trade_name: str = "",
        atc_code: str = ""
    ):
        """
        Learn from site clarification response (Scenario B Step 5)
        
        When site responds with clarification (e.g., "Pain killer" -> "Advil"),
        update learning cache for future reference.
        """
        if self._ambiguous_detector and self.config.ambiguous_coding_learning:
            self._ambiguous_detector.learn_from_resolution(
                verbatim=verbatim,
                resolved_term=resolved_term,
                generic_name=generic_name,
                trade_name=trade_name,
                atc_code=atc_code
            )
            logger.info(f"[Codex] Learned: '{verbatim}' -> '{resolved_term}' ({generic_name})")
    
    def _process_coding_report(
        self,
        coding_data: pd.DataFrame,
        dictionary_type: str,
        twins: List[DigitalPatientTwin],
        study_id: str
    ):
        """Process GlobalCodingReport for uncoded terms"""
        # Handle duplicate columns
        if coding_data.columns.duplicated().any():
            coding_data = coding_data.loc[:, ~coding_data.columns.duplicated()]
        
        # Find the coding status column
        status_col = None
        for col in ['Coding Status', 'coding_status', 'CodingStatus']:
            if col in coding_data.columns:
                status_col = col
                break
        
        if status_col is None:
            logger.warning(f"No Coding Status column found in {dictionary_type} data")
            return
        
        # Find subject column
        subject_col = None
        for col in ['Subject', 'subject_id', 'Subject ID']:
            if col in coding_data.columns:
                subject_col = col
                break
        
        # Find form OID for context
        form_col = None
        for col in ['Form OID', 'form_oid', 'FormOID']:
            if col in coding_data.columns:
                form_col = col
                break
        
        # Find field OID for term type
        field_col = None
        for col in ['Field OID', 'field_oid', 'FieldOID']:
            if col in coding_data.columns:
                field_col = col
                break
        
        # Build twin lookup
        twin_map = {t.subject_id: t for t in twins}
        
        # Filter for uncoded terms: Coding Status = 'UnCoded Term'
        uncoded_mask = coding_data[status_col].astype(str).str.lower().str.contains(
            'uncoded|un-coded|not coded|pending', na=False
        )
        uncoded_terms = coding_data[uncoded_mask]
        
        self._coding_stats['total_uncoded'] += len(uncoded_terms)
        
        for _, row in uncoded_terms.iterrows():
            subject_id = str(row.get(subject_col, '')) if subject_col else ''
            twin = twin_map.get(subject_id)
            
            if not twin:
                # Try to find matching twin
                for sid, t in twin_map.items():
                    if subject_id in sid or sid in subject_id:
                        twin = t
                        break
            
            if not twin:
                continue
            
            # Extract context from Form OID
            form_oid = str(row.get(form_col, '')) if form_col else ''
            context = self._get_context_from_form(form_oid)
            
            # Get field type (AETERM, CMTRT, etc.)
            field_oid = str(row.get(field_col, '')) if field_col else ''
            
            # Since we don't have the actual verbatim term in this data,
            # we'll create a placeholder indicating the uncoded item
            term_placeholder = f"Uncoded {field_oid} in {form_oid}"
            
            # Analyze and create recommendation
            rec = self._analyze_term_with_context(
                verbatim=term_placeholder,
                context=context,
                dictionary_type=dictionary_type,
                twin=twin,
                study_id=study_id,
                form_oid=form_oid,
                field_oid=field_oid,
                logline=row.get('Logline', '')
            )
            
            if rec:
                self.recommendations.append(rec)
    
    def _get_context_from_form(self, form_oid: str) -> str:
        """Map Form OID to human-readable context"""
        for prefix, context in self.context_map.items():
            if form_oid.upper().startswith(prefix):
                return context
        return 'Clinical Form'
    
    def _process_twin_uncoded_terms(self, twins: List[DigitalPatientTwin], study_id: str):
        """Process uncoded terms from Digital Patient Twins (from CPID # Uncoded Terms)"""
        for twin in twins:
            if twin.uncoded_terms > 0 and twin.uncoded_terms_list:
                for term_info in twin.uncoded_terms_list[:5]:  # Limit to 5 per patient
                    verbatim = term_info.get('verbatim_term', '')
                    context = term_info.get('context', 'Unknown')
                    dictionary_type = 'MedDRA' if 'AE' in context or 'MH' in context else 'WHODRA'
                    
                    rec = self._analyze_term_with_context(
                        verbatim=verbatim,
                        context=context,
                        dictionary_type=dictionary_type,
                        twin=twin,
                        study_id=study_id
                    )
                    if rec:
                        self.recommendations.append(rec)
    
    def _analyze_term_with_context(
        self,
        verbatim: str,
        context: str,
        dictionary_type: str,
        twin: DigitalPatientTwin,
        study_id: str,
        form_oid: str = "",
        field_oid: str = "",
        logline: Any = None
    ) -> Optional[AgentRecommendation]:
        """
        Analyze a term with full context and determine appropriate action
        
        Decision Logic (Human-in-the-Loop Protocol):
        - High Confidence (>95%): AUTO_CODE - Apply automatically
        - Medium Confidence (80-95%): PROPOSE_CODE - Single-click approval
        - Low Confidence (<80%): REQUEST_CLARIFICATION - Query to site
        """
        if not verbatim or pd.isna(verbatim):
            return None
        
        verbatim_clean = str(verbatim).lower().strip()
        
        # Check for illegible/non-standard patterns first
        import re
        for pattern in self.illegible_patterns:
            if re.match(pattern, verbatim_clean, re.IGNORECASE):
                self._coding_stats['clarification_needed'] += 1
                return self._create_recommendation(
                    action_type=ActionType.REQUEST_CLARIFICATION,
                    priority=ActionPriority.MEDIUM,
                    subject_id=twin.subject_id,
                    site_id=twin.site_id,
                    study_id=study_id,
                    title=f"[CODEX] Illegible Term: {verbatim[:30]}",
                    description=self.ILLEGIBLE_TERM_QUERY.format(
                        verbatim=verbatim,
                        subject_id=twin.subject_id
                    ),
                    rationale="Term appears illegible or contains non-standard abbreviations. "
                             "Site clarification required per ICH E6 R2 guidelines.",
                    confidence=0.2,
                    requires_approval=False,
                    auto_executable=True,
                    source_data={
                        'verbatim': verbatim,
                        'context': context,
                        'dictionary': dictionary_type,
                        'reason': 'illegible_term',
                        'form_oid': form_oid,
                        'field_oid': field_oid
                    }
                )
        
        # Check for ambiguous terms
        if any(amb in verbatim_clean for amb in self.ambiguous_patterns):
            self._coding_stats['clarification_needed'] += 1
            term_type = "Trade Name or Generic Name" if dictionary_type == 'WHODRA' else "specific medical term (LLT/PT)"
            return self._create_recommendation(
                action_type=ActionType.REQUEST_CLARIFICATION,
                priority=ActionPriority.MEDIUM,
                subject_id=twin.subject_id,
                site_id=twin.site_id,
                study_id=study_id,
                title=f"[CODEX] Ambiguous Term: {verbatim[:30]}",
                description=self.AMBIGUOUS_TERM_QUERY.format(
                    verbatim=verbatim,
                    context=context,
                    term_type=term_type
                ),
                rationale=f"Term '{verbatim}' is too generic for reliable {dictionary_type} coding. "
                         "Specific terminology required to ensure accurate medical coding.",
                confidence=0.4,
                requires_approval=False,
                auto_executable=True,
                source_data={
                    'verbatim': verbatim,
                    'context': context,
                    'dictionary': dictionary_type,
                    'reason': 'ambiguous_term',
                    'form_oid': form_oid
                }
            )
        
        # Try to match against dictionary
        if dictionary_type == 'MedDRA':
            match = self._match_meddra_term(verbatim_clean)
        else:
            match = self._match_whodrug_term(verbatim_clean)
        
        if match:
            confidence = match.get('confidence', 0.5)
            
            if confidence >= self.config.coding_auto_apply_threshold:
                # HIGH CONFIDENCE (>95%): Auto-apply code
                self._coding_stats['auto_coded'] += 1
                coded_term = match.get('pt', match.get('generic', match.get('trade_name', '')))
                return self._create_recommendation(
                    action_type=ActionType.AUTO_CODE,
                    priority=ActionPriority.LOW,
                    subject_id=twin.subject_id,
                    site_id=twin.site_id,
                    study_id=study_id,
                    title=f"[CODEX] Auto-Code: {verbatim[:25]} -> {coded_term}",
                    description=f"High confidence match. Auto-applying code:\n"
                               f"Verbatim: '{verbatim}'\n"
                               f"Coded: '{coded_term}' ({dictionary_type})\n"
                               f"Context: {context}",
                    rationale=f"Confidence {confidence*100:.0f}% exceeds auto-apply threshold (95%). "
                             f"Term directly maps to {dictionary_type} dictionary entry.",
                    confidence=confidence,
                    requires_approval=False,
                    auto_executable=True,
                    source_data={
                        'verbatim': verbatim,
                        'coded_term': coded_term,
                        'dictionary': dictionary_type,
                        'confidence': confidence,
                        'context': context,
                        'match_details': match,
                        'action': 'AUTO_APPLIED'
                    }
                )
            
            elif confidence >= self.config.coding_propose_threshold:
                # MEDIUM CONFIDENCE (80-95%): Propose for single-click approval
                self._coding_stats['proposed'] += 1
                coded_term = match.get('pt', match.get('generic', match.get('trade_name', '')))
                return self._create_recommendation(
                    action_type=ActionType.PROPOSE_CODE,
                    priority=ActionPriority.MEDIUM,
                    subject_id=twin.subject_id,
                    site_id=twin.site_id,
                    study_id=study_id,
                    title=f"[CODEX] Proposed: {verbatim[:25]} -> {coded_term}",
                    description=f"Medium confidence match. Requires single-click approval:\n"
                               f"Verbatim: '{verbatim}'\n"
                               f"Proposed Code: '{coded_term}' ({dictionary_type})\n"
                               f"Context: {context}\n"
                               f"[APPROVE] [REJECT] [MODIFY]",
                    rationale=f"Confidence {confidence*100:.0f}% is medium (80-95%). "
                             f"Human verification recommended before applying code.",
                    confidence=confidence,
                    requires_approval=True,
                    auto_executable=False,
                    source_data={
                        'verbatim': verbatim,
                        'proposed_code': coded_term,
                        'dictionary': dictionary_type,
                        'confidence': confidence,
                        'context': context,
                        'match_details': match,
                        'action': 'PENDING_APPROVAL'
                    }
                )
        
        # No match found or low confidence - request clarification
        self._coding_stats['clarification_needed'] += 1
        return self._create_recommendation(
            action_type=ActionType.REQUEST_CLARIFICATION,
            priority=ActionPriority.MEDIUM,
            subject_id=twin.subject_id,
            site_id=twin.site_id,
            study_id=study_id,
            title=f"[CODEX] Unknown Term: {verbatim[:30]}",
            description=self.UNKNOWN_TERM_QUERY.format(
                verbatim=verbatim,
                dictionary=dictionary_type
            ),
            rationale=f"Term '{verbatim}' not found in {dictionary_type} dictionary. "
                     "Site verification needed to ensure accurate coding.",
            confidence=0.3,
            requires_approval=False,
            auto_executable=True,
            source_data={
                'verbatim': verbatim,
                'context': context,
                'dictionary': dictionary_type,
                'reason': 'unknown_term',
                'form_oid': form_oid,
                'action': 'QUERY_SENT'
            }
        )
    
    def _match_meddra_term(self, verbatim: str) -> Optional[Dict]:
        """Match verbatim term against MedDRA dictionary (LLM simulation)"""
        # Direct match
        if verbatim in self.meddra_mappings:
            return self.meddra_mappings[verbatim]
        
        # Fuzzy matching simulation
        for term, mapping in self.meddra_mappings.items():
            if term in verbatim or verbatim in term:
                # Partial match - reduce confidence
                reduced_conf = mapping['confidence'] * 0.85
                return {**mapping, 'confidence': reduced_conf}
        
        return None
    
    def _match_whodrug_term(self, verbatim: str) -> Optional[Dict]:
        """Match verbatim term against WHO Drug dictionary (LLM simulation)"""
        # Direct match
        if verbatim in self.whodrug_mappings:
            return self.whodrug_mappings[verbatim]
        
        # Fuzzy matching simulation
        for term, mapping in self.whodrug_mappings.items():
            if term in verbatim or verbatim in term:
                # Partial match - reduce confidence
                reduced_conf = mapping['confidence'] * 0.85
                return {**mapping, 'confidence': reduced_conf}
        
        return None
    
    def get_coding_statistics(self) -> Dict:
        """Get statistics about coding recommendations"""
        return self._coding_stats


class SiteLiaisonAgent(BaseAgent):
    """
    Lia - The Site Liaison Agent
    
    Primary Objective: Proactive site management and visit compliance
    
    Triggers:
    1. Visit Projection Tracker: # Days Outstanding > threshold (default 5 days)
    2. Missing_Lab_Name_and_Missing_Ranges: "Action for Site" field populated
    3. Accumulated issues requiring aggregated communication
    
    Action Logic:
    - Drafts personalized, context-aware emails/notifications to Site Coordinator
    - Includes specific details: Subject ID, Visit name, days overdue, lab issues
    
    Context Awareness:
    - Checks CPID_EDC_Metrics for SSM (Site Status Metric) and open query count
    - If site is "Overburdened" (Red status) or has high query volume:
      - Softens tone in communications
      - Aggregates multiple queries into "Weekly Digest"
      - Prevents site burnout and alert fatigue
    """
    
    # Email/Notification Templates
    VISIT_REMINDER_TEMPLATE = (
        "Dear Site Coordinator,\n\n"
        "Subject {subject_id} was projected for {visit_name} on {projected_date}. "
        "It is now {days_outstanding} days overdue.\n\n"
        "{additional_issues}"
        "Please update the EDC at your earliest convenience.\n\n"
        "Thank you for your continued support.\n"
        "Clinical Data Management Team"
    )
    
    VISIT_ESCALATION_TEMPLATE = (
        "URGENT: Visit Data Entry Required\n\n"
        "Dear Site Coordinator,\n\n"
        "Subject {subject_id}, {visit_name} is now {days_outstanding} days overdue, "
        "exceeding our {threshold}-day threshold.\n\n"
        "This delay may impact study timelines and data quality assessments. "
        "Please prioritize data entry or contact the study team if assistance is needed.\n\n"
        "This matter has been escalated to the CRA for follow-up.\n\n"
        "Clinical Data Management Team"
    )
    
    SITE_DIGEST_TEMPLATE = (
        "Weekly Site Summary: {site_id}\n"
        "{'='*50}\n\n"
        "Dear Site Coordinator,\n\n"
        "This is your consolidated weekly summary to help manage outstanding items efficiently.\n\n"
        "SUMMARY:\n"
        "- Open Queries: {open_queries}\n"
        "- Missing Visits: {missing_visits}\n"
        "- Missing Pages: {missing_pages}\n"
        "- Data Quality Index: {dqi_score:.1f}%\n\n"
        "OUTSTANDING SUBJECTS:\n"
        "{subject_list}\n\n"
        "We understand your workload and have consolidated these items to minimize disruption. "
        "Please address items by priority when possible.\n\n"
        "Thank you for your partnership.\n"
        "Clinical Data Management Team"
    )
    
    SOFT_REMINDER_TEMPLATE = (
        "Dear Site Coordinator,\n\n"
        "We hope this message finds you well. We understand you have been managing "
        "a high volume of activities, and we appreciate your dedication.\n\n"
        "When time permits, the following items would benefit from your attention:\n"
        "{items_list}\n\n"
        "Please don't hesitate to reach out if you need support or have questions.\n\n"
        "With appreciation,\n"
        "Clinical Data Management Team"
    )
    
    # Site burden thresholds
    HIGH_QUERY_THRESHOLD = 15  # Queries per site
    OVERBURDENED_STATUS = ["Red", "Critical", "At Risk"]
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("Lia", config)
        self._site_burden_cache: Dict[str, Dict] = {}
        self._site_pending_items: Dict[str, List[Dict]] = {}
        self._communication_stats = {
            'standard_reminders': 0,
            'escalations': 0,
            'soft_reminders': 0,
            'weekly_digests': 0,
            'ghost_visits_detected': 0,
            'cra_escalations': 0
        }
        
        # Ghost Visit Detector (Scenario C) - lazy initialized
        self._ghost_visit_detector = None
        self._ghost_visits: List = []
    
    def analyze(
        self,
        twins: List[DigitalPatientTwin],
        site_metrics: Dict[str, SiteMetrics],
        visit_data: Optional[pd.DataFrame] = None,
        missing_lab_data: Optional[pd.DataFrame] = None,
        inactivated_forms_data: Optional[pd.DataFrame] = None,
        cpid_data: Optional[pd.DataFrame] = None,
        study_id: str = ""
    ) -> List[AgentRecommendation]:
        """
        Analyze visit compliance and generate site communications
        
        Analysis Pipeline:
        1. Assess site burden (SSM status, open query count)
        2. Process Visit Projection Tracker for overdue visits
        3. Process Missing Lab data for site actions
        4. Determine communication strategy per site
        5. Generate appropriate notifications (standard/soft/digest)
        
        Enhanced (Scenario C - Ghost Visit Detection):
        6. Detect ghost visits (scheduled but no data entered)
        7. Check inactivated forms for valid reasons
        8. Escalate based on site standing (Green=reminder, Red=CRA call)
        """
        self.recommendations = []
        self._site_pending_items = {}
        self._ghost_visits = []
        
        # Step 1: Build site burden assessment
        self._assess_site_burden(site_metrics, twins)
        
        # Step 2: Process Visit Projection Tracker
        if visit_data is not None:
            self._process_visit_tracker(visit_data, twins, site_metrics, study_id)
            
            # *** ENHANCED: Ghost Visit Detection (Scenario C) ***
            # Full pipeline: Visit Tracker → Inactivation Check → Site Standing → Action
            if self.config.ghost_visit_enabled:
                self._detect_ghost_visits(
                    visit_data=visit_data,
                    inactivated_forms_data=inactivated_forms_data,
                    cpid_data=cpid_data,
                    study_id=study_id
                )
        
        # Step 3: Process outstanding visits from twins
        for twin in twins:
            if twin.outstanding_visits:
                self._analyze_outstanding_visits(twin, site_metrics, study_id)
            
            if twin.missing_visits > 0:
                self._track_missing_visits(twin)
        
        # Step 4: Process missing lab data
        if missing_lab_data is not None:
            self._analyze_missing_labs(missing_lab_data, twins, site_metrics, study_id)
        
        # Step 5: Generate aggregated communications for overburdened sites
        self._generate_smart_communications(site_metrics, twins, study_id)
        
        logger.info(f"Lia generated {len(self.recommendations)} recommendations")
        logger.info(f"Lia communication stats: {self._communication_stats}")
        return self.recommendations
    
    def _detect_ghost_visits(
        self,
        visit_data: pd.DataFrame,
        inactivated_forms_data: Optional[pd.DataFrame],
        cpid_data: Optional[pd.DataFrame],
        study_id: str
    ) -> None:
        """
        Enhanced Ghost Visit Detection - Implements Scenario C
        
        Full detection pipeline:
        1. Data Check: Scan Visit Projection Tracker for Projected Date < Current Date
        2. Calculation: Calculate Days Outstanding
        3. Context Check: Check Inactivated forms for valid reasons
        4. Action:
           - Valid inactivation: No query needed
           - Green site (good standing): Send standard reminder
           - Red site (non-compliance history): Escalate to CRA for phone call
        """
        try:
            from core.ghost_visit_detector import (
                GhostVisitDetector, VisitStatus, SiteStanding, EscalationLevel
            )
        except ImportError:
            try:
                from clinical_dataflow_optimizer.core.ghost_visit_detector import (
                    GhostVisitDetector, VisitStatus, SiteStanding, EscalationLevel
                )
            except ImportError:
                logger.warning("[Lia] GhostVisitDetector not available - skipping Scenario C")
                return
        
        # Initialize detector if needed
        if self._ghost_visit_detector is None:
            self._ghost_visit_detector = GhostVisitDetector()
        
        # Run detection pipeline
        ghost_visits = self._ghost_visit_detector.detect(
            visit_tracker_data=visit_data,
            inactivated_forms_data=inactivated_forms_data,
            cpid_data=cpid_data,
            study_id=study_id
        )
        
        self._ghost_visits = ghost_visits
        
        # Convert detected ghost visits to agent recommendations
        for ghost in ghost_visits:
            if ghost.visit_status == VisitStatus.VALIDLY_INACTIVATED:
                # No action needed - visit was properly inactivated
                continue
            
            if ghost.visit_status == VisitStatus.GHOST_CONFIRMED:
                self._communication_stats['ghost_visits_detected'] += 1
                
                # Determine action based on site standing and escalation level
                if ghost.escalation_level in [EscalationLevel.CRA_ESCALATION, 
                                               EscalationLevel.SPONSOR_ESCALATION]:
                    # CRA Escalation - site has non-compliance history
                    self._communication_stats['cra_escalations'] += 1
                    
                    rec = self._create_recommendation(
                        action_type=ActionType.ALERT_CRA,
                        priority=ActionPriority.HIGH if ghost.escalation_level == EscalationLevel.CRA_ESCALATION 
                                 else ActionPriority.CRITICAL,
                        subject_id=ghost.subject_id,
                        site_id=ghost.site_id,
                        study_id=study_id,
                        title=f"[GHOST VISIT] CRA Escalation: {ghost.visit_name} ({ghost.days_outstanding} days)",
                        description=ghost.auto_reminder,
                        rationale=(
                            f"Ghost visit detected with CRA escalation required:\n"
                            f"- Site Standing: {ghost.site_standing.name} (non-compliance history)\n"
                            f"- Days Outstanding: {ghost.days_outstanding}\n"
                            f"- Risk Score: {ghost.risk_score:.0%}\n"
                            f"- No valid inactivation found\n"
                            f"Recommend direct phone call to Site Coordinator."
                        ),
                        confidence=0.95,
                        requires_approval=True,  # CRA escalation needs review
                        auto_executable=False,
                        source_data={
                            'visit_name': ghost.visit_name,
                            'projected_date': ghost.projected_date.strftime('%Y-%m-%d'),
                            'days_outstanding': ghost.days_outstanding,
                            'site_standing': ghost.site_standing.name,
                            'escalation_level': ghost.escalation_level.name,
                            'risk_score': ghost.risk_score,
                            'country': ghost.country,
                            'inactivation_check': ghost.inactivation_check,
                            'data_entry_deadline': ghost.data_entry_deadline.strftime('%Y-%m-%d') if ghost.data_entry_deadline else None,
                            'source': 'GhostVisitDetector'
                        }
                    )
                    self.recommendations.append(rec)
                    
                else:
                    # Standard or Enhanced Reminder - site in good/moderate standing
                    priority = ActionPriority.MEDIUM if ghost.escalation_level == EscalationLevel.STANDARD_REMINDER \
                              else ActionPriority.HIGH
                    
                    rec = self._create_recommendation(
                        action_type=ActionType.SEND_REMINDER,
                        priority=priority,
                        subject_id=ghost.subject_id,
                        site_id=ghost.site_id,
                        study_id=study_id,
                        title=f"[GHOST VISIT] Reminder: {ghost.visit_name} ({ghost.days_outstanding} days)",
                        description=ghost.auto_reminder,
                        rationale=(
                            f"Ghost visit detected - visit scheduled but no data entered:\n"
                            f"- Site Standing: {ghost.site_standing.name}\n"
                            f"- Days Outstanding: {ghost.days_outstanding}\n"
                            f"- Risk Score: {ghost.risk_score:.0%}\n"
                            f"- No valid inactivation found\n"
                            f"Standard reminder appropriate for site in good standing."
                        ),
                        confidence=0.92,
                        requires_approval=False,
                        auto_executable=self.config.ghost_visit_auto_reminder,
                        source_data={
                            'visit_name': ghost.visit_name,
                            'projected_date': ghost.projected_date.strftime('%Y-%m-%d'),
                            'days_outstanding': ghost.days_outstanding,
                            'site_standing': ghost.site_standing.name,
                            'escalation_level': ghost.escalation_level.name,
                            'risk_score': ghost.risk_score,
                            'country': ghost.country,
                            'source': 'GhostVisitDetector'
                        }
                    )
                    self.recommendations.append(rec)
        
        # Log summary
        summary = self._ghost_visit_detector.get_summary_report()
        logger.info(
            f"[Lia] Ghost Visit Detection: {summary['breakdown']['ghost_confirmed']} confirmed, "
            f"{summary['breakdown']['validly_inactivated']} validly inactivated, "
            f"{summary.get('requires_cra_action', 0)} CRA escalations"
        )
    
    def get_ghost_visit_summary(self) -> Dict:
        """Get summary of ghost visit detection"""
        if self._ghost_visit_detector:
            return self._ghost_visit_detector.get_summary_report()
        return {'total_analyzed': 0, 'message': 'Ghost visit detection not run'}
    
    def get_cra_escalations(self) -> List[Dict]:
        """Get visits requiring CRA escalation (phone call)"""
        if self._ghost_visit_detector:
            return self._ghost_visit_detector.get_cra_escalations()
        return []
    
    def get_standard_reminders(self) -> List[Dict]:
        """Get visits requiring standard reminders"""
        if self._ghost_visit_detector:
            return self._ghost_visit_detector.get_standard_reminders()
        return []
    
    def get_valid_inactivations(self) -> List[Dict]:
        """Get visits that were validly inactivated (no action needed)"""
        if self._ghost_visit_detector:
            return self._ghost_visit_detector.get_valid_inactivations()
        return []
    
    def get_site_standing_summary(self) -> Dict[str, Dict]:
        """Get summary of site standings from ghost visit analysis"""
        if self._ghost_visit_detector:
            return self._ghost_visit_detector.get_site_standing_summary()
        return {}
    
    def _assess_site_burden(
        self,
        site_metrics: Dict[str, SiteMetrics],
        twins: List[DigitalPatientTwin]
    ):
        """Assess burden level for each site based on SSM and query count"""
        self._site_burden_cache = {}
        
        # Aggregate patient issues by site
        site_issues = {}
        for twin in twins:
            site_id = twin.site_id
            if site_id not in site_issues:
                site_issues[site_id] = {
                    'patients': 0,
                    'open_queries': 0,
                    'missing_visits': 0,
                    'missing_pages': 0
                }
            site_issues[site_id]['patients'] += 1
            site_issues[site_id]['open_queries'] += twin.open_queries
            site_issues[site_id]['missing_visits'] += twin.missing_visits
            site_issues[site_id]['missing_pages'] += twin.missing_pages
        
        for site_id in set(list(site_metrics.keys()) + list(site_issues.keys())):
            metrics = site_metrics.get(site_id)
            issues = site_issues.get(site_id, {})
            
            ssm_status = metrics.ssm_status if metrics else 'Unknown'
            total_queries = issues.get('open_queries', 0)
            if metrics:
                total_queries = max(total_queries, metrics.total_open_queries)
            
            # Determine if site is overburdened
            is_overburdened = (
                ssm_status in self.OVERBURDENED_STATUS or
                total_queries >= self.HIGH_QUERY_THRESHOLD
            )
            
            self._site_burden_cache[site_id] = {
                'ssm_status': ssm_status,
                'total_queries': total_queries,
                'is_overburdened': is_overburdened,
                'patient_count': issues.get('patients', 0),
                'missing_visits': issues.get('missing_visits', 0),
                'communication_strategy': 'digest' if is_overburdened else 'standard'
            }
    
    def _process_visit_tracker(
        self,
        visit_data: pd.DataFrame,
        twins: List[DigitalPatientTwin],
        site_metrics: Dict[str, SiteMetrics],
        study_id: str
    ):
        """
        Process Visit Projection Tracker for visits with # Days Outstanding > threshold
        
        Columns expected: Country, Site, Subject, Visit, Projected Date, # Days Outstanding
        """
        # Handle duplicate columns
        if visit_data.columns.duplicated().any():
            visit_data = visit_data.loc[:, ~visit_data.columns.duplicated()]
        
        # Find column names
        subject_col = None
        for col in ['Subject', 'subject_id', 'Subject ID']:
            if col in visit_data.columns:
                subject_col = col
                break
        
        site_col = None
        for col in ['Site', 'site_id', 'Site ID']:
            if col in visit_data.columns:
                site_col = col
                break
        
        visit_col = None
        for col in ['Visit', 'visit_name', 'Visit Name']:
            if col in visit_data.columns:
                visit_col = col
                break
        
        date_col = None
        for col in ['Projected Date', 'projected_date', 'ProjectedDate']:
            if col in visit_data.columns:
                date_col = col
                break
        
        days_col = None
        for col in ['# Days Outstanding', 'Days Outstanding', 'days_outstanding']:
            if col in visit_data.columns:
                days_col = col
                break
        
        if not all([subject_col, days_col]):
            logger.warning("Visit tracker missing required columns")
            return
        
        # Build twin lookup
        twin_map = {t.subject_id: t for t in twins}
        
        # Process each overdue visit
        threshold_days = self.config.visit_reminder_days
        
        for _, row in visit_data.iterrows():
            days_out = row.get(days_col, 0)
            if pd.isna(days_out):
                continue
            days_out = int(days_out)
            
            if days_out < threshold_days:
                continue
            
            subject_id = str(row.get(subject_col, ''))
            twin = twin_map.get(subject_id)
            
            if not twin:
                # Try fuzzy match
                for sid, t in twin_map.items():
                    if subject_id in sid or sid in subject_id:
                        twin = t
                        break
            
            if not twin:
                continue
            
            site_id = str(row.get(site_col, twin.site_id)) if site_col else twin.site_id
            visit_name = str(row.get(visit_col, 'Unknown Visit')) if visit_col else 'Unknown Visit'
            projected_date = str(row.get(date_col, '')) if date_col else ''
            
            # Check site burden to determine communication strategy
            site_burden = self._site_burden_cache.get(site_id, {})
            is_overburdened = site_burden.get('is_overburdened', False)
            
            # Track pending item for potential aggregation
            if site_id not in self._site_pending_items:
                self._site_pending_items[site_id] = []
            
            self._site_pending_items[site_id].append({
                'type': 'overdue_visit',
                'subject_id': subject_id,
                'visit_name': visit_name,
                'days_outstanding': days_out,
                'projected_date': projected_date
            })
            
            # If site is NOT overburdened, create individual recommendation
            if not is_overburdened:
                self._create_visit_recommendation(
                    twin, site_id, visit_name, days_out, projected_date, 
                    site_metrics, study_id, site_burden
                )
            # Overburdened sites get aggregated digest later
    
    def _create_visit_recommendation(
        self,
        twin: DigitalPatientTwin,
        site_id: str,
        visit_name: str,
        days_outstanding: int,
        projected_date: str,
        site_metrics: Dict[str, SiteMetrics],
        study_id: str,
        site_burden: Dict = None
    ):
        """Create individual visit recommendation with personalized message"""
        
        # Determine if escalation is needed
        is_escalation = days_outstanding >= self.config.escalation_days
        
        if is_escalation:
            # Generate escalation message
            message = self.VISIT_ESCALATION_TEMPLATE.format(
                subject_id=twin.subject_id,
                visit_name=visit_name,
                days_outstanding=days_outstanding,
                threshold=self.config.escalation_days
            )
            
            # Generate enhanced rationale using LongCat AI
            enhanced_rationale = self.enhance_reasoning_with_longcat(
                context="Clinical trial visit compliance and escalation analysis",
                task="Evaluate overdue visit and determine appropriate escalation strategy",
                data={
                    'subject_id': twin.subject_id,
                    'visit_name': visit_name,
                    'days_outstanding': days_outstanding,
                    'escalation_threshold': self.config.escalation_days,
                    'is_escalation': True,
                    'site_burden': site_burden
                }
            )
            
            # Combine original rationale with AI enhancement
            base_rationale = f"Visit exceeds {self.config.escalation_days}-day threshold. CRA intervention required."
            
            final_rationale = base_rationale
            if not enhanced_rationale.startswith("Standard reasoning:"):
                final_rationale += f"\n\nAI Enhanced Analysis: {enhanced_rationale}"
            
            rec = self._create_recommendation(
                action_type=ActionType.ALERT_CRA,
                priority=ActionPriority.HIGH,
                subject_id=twin.subject_id,
                site_id=site_id,
                study_id=study_id,
                title=f"ESCALATION: {visit_name} ({days_outstanding} days overdue)",
                description=message,
                rationale=final_rationale,
                confidence=0.98,
                requires_approval=True,
                source_data={
                    'visit': visit_name,
                    'days_outstanding': days_outstanding,
                    'projected_date': projected_date,
                    'escalation': True
                }
            )
            self.recommendations.append(rec)
            self._communication_stats['escalations'] += 1
        else:
            # Generate standard personalized reminder
            additional_issues = ""
            
            # Check if there are lab issues for this subject
            if site_id in self._site_pending_items:
                lab_issues = [
                    item for item in self._site_pending_items[site_id]
                    if item.get('type') == 'missing_lab' and item.get('subject_id') == twin.subject_id
                ]
                if lab_issues:
                    lab_info = lab_issues[0]
                    additional_issues = f"Additionally, the {lab_info.get('test_name', 'Lab Name')} is missing for the {lab_info.get('form_name', 'form')}.\n\n"
            
            message = self.VISIT_REMINDER_TEMPLATE.format(
                subject_id=twin.subject_id,
                visit_name=visit_name,
                projected_date=projected_date or 'the scheduled date',
                days_outstanding=days_outstanding,
                additional_issues=additional_issues
            )
            
            # Generate enhanced rationale using LongCat AI
            enhanced_rationale = self.enhance_reasoning_with_longcat(
                context="Clinical trial visit compliance and patient retention analysis",
                task="Evaluate overdue visit and determine appropriate reminder strategy",
                data={
                    'subject_id': twin.subject_id,
                    'visit_name': visit_name,
                    'days_outstanding': days_outstanding,
                    'projected_date': projected_date,
                    'is_escalation': False,
                    'additional_issues': bool(additional_issues.strip()),
                    'site_burden': site_burden
                }
            )
            
            # Combine original rationale with AI enhancement
            base_rationale = "Timely visit data entry ensures data quality and protocol compliance."
            
            final_rationale = base_rationale
            if not enhanced_rationale.startswith("Standard reasoning:"):
                final_rationale += f"\n\nAI Enhanced Analysis: {enhanced_rationale}"
            
            rec = self._create_recommendation(
                action_type=ActionType.SEND_REMINDER,
                priority=ActionPriority.MEDIUM,
                subject_id=twin.subject_id,
                site_id=site_id,
                study_id=study_id,
                title=f"Visit Reminder: {visit_name} ({days_outstanding} days overdue)",
                description=message,
                rationale=final_rationale,
                confidence=0.95,
                requires_approval=False,
                auto_executable=True,
                source_data={
                    'visit': visit_name,
                    'days_outstanding': days_outstanding,
                    'projected_date': projected_date
                }
            )
            self.recommendations.append(rec)
            self._communication_stats['standard_reminders'] += 1
    
    def _analyze_outstanding_visits(
        self,
        twin: DigitalPatientTwin,
        site_metrics: Dict[str, SiteMetrics],
        study_id: str
    ):
        """Analyze visits from twin's outstanding_visits list"""
        site_burden = self._site_burden_cache.get(twin.site_id, {})
        is_overburdened = site_burden.get('is_overburdened', False)
        
        for visit in twin.outstanding_visits:
            days_out = visit.get('days_outstanding', 0)
            if pd.isna(days_out):
                days_out = 0
            
            if days_out < self.config.visit_reminder_days:
                continue
            
            visit_name = visit.get('visit_name', 'Unknown Visit')
            projected_date = visit.get('projected_date', '')
            
            # Track for potential aggregation
            if twin.site_id not in self._site_pending_items:
                self._site_pending_items[twin.site_id] = []
            
            self._site_pending_items[twin.site_id].append({
                'type': 'overdue_visit',
                'subject_id': twin.subject_id,
                'visit_name': visit_name,
                'days_outstanding': int(days_out),
                'projected_date': projected_date
            })
            
            if not is_overburdened:
                self._create_visit_recommendation(
                    twin, twin.site_id, visit_name, int(days_out), 
                    str(projected_date), site_metrics, study_id
                )
    
    def _track_missing_visits(self, twin: DigitalPatientTwin):
        """Track missing visits for aggregation"""
        if twin.site_id not in self._site_pending_items:
            self._site_pending_items[twin.site_id] = []
        
        self._site_pending_items[twin.site_id].append({
            'type': 'missing_visit',
            'subject_id': twin.subject_id,
            'count': twin.missing_visits
        })
    
    def _analyze_missing_labs(
        self,
        missing_lab_data: pd.DataFrame,
        twins: List[DigitalPatientTwin],
        site_metrics: Dict[str, SiteMetrics],
        study_id: str
    ):
        """
        Analyze missing lab data - Missing_Lab_Name_and_Missing_Ranges
        
        Generates queries/reminders when "Action for Site" is populated
        """
        # Find relevant columns
        subject_col = None
        for col in ['Subject', 'subject_id', 'Subject ID']:
            if col in missing_lab_data.columns:
                subject_col = col
                break
        
        if not subject_col:
            logger.warning("Missing lab data has no subject column")
            return
        
        site_col = None
        for col in ['Site', 'site_id', 'Site ID']:
            if col in missing_lab_data.columns:
                site_col = col
                break
        
        action_col = None
        for col in ['Action for Site', 'action_for_site', 'Action']:
            if col in missing_lab_data.columns:
                action_col = col
                break
        
        twin_map = {t.subject_id: t for t in twins}
        
        for _, row in missing_lab_data.iterrows():
            subject_id = str(row.get(subject_col, ''))
            twin = twin_map.get(subject_id)
            
            if not twin:
                for sid, t in twin_map.items():
                    if subject_id in sid or sid in subject_id:
                        twin = t
                        break
            
            if not twin:
                continue
            
            site_id = str(row.get(site_col, twin.site_id)) if site_col else twin.site_id
            lab_name = row.get('Lab Name', row.get('lab_name', 'Unknown Lab'))
            test_name = row.get('Test Name', row.get('test_name', 'Unknown Test'))
            action_needed = row.get(action_col, '') if action_col else ''
            form_name = row.get('Form', row.get('form_name', 'Form'))
            
            # Track for aggregation
            if site_id not in self._site_pending_items:
                self._site_pending_items[site_id] = []
            
            self._site_pending_items[site_id].append({
                'type': 'missing_lab',
                'subject_id': subject_id,
                'lab_name': str(lab_name) if not pd.isna(lab_name) else 'Unknown',
                'test_name': str(test_name) if not pd.isna(test_name) else 'Unknown',
                'form_name': str(form_name) if not pd.isna(form_name) else 'Form',
                'action': str(action_needed) if not pd.isna(action_needed) else ''
            })
            
            # Check if site is overburdened
            site_burden = self._site_burden_cache.get(site_id, {})
            is_overburdened = site_burden.get('is_overburdened', False)
            
            # Only create individual recommendation for non-overburdened sites
            if not is_overburdened and action_needed and not pd.isna(action_needed):
                rec = self._create_recommendation(
                    action_type=ActionType.QUERY_TO_SITE,
                    priority=ActionPriority.MEDIUM,
                    subject_id=subject_id,
                    site_id=site_id,
                    study_id=study_id,
                    title=f"Missing Lab Data: {test_name}",
                    description=(
                        f"Dear Site Coordinator,\n\n"
                        f"Subject {subject_id}: The Lab Name or Reference Range is missing "
                        f"for {test_name} in the {form_name} form.\n\n"
                        f"Action Required: {action_needed}\n\n"
                        f"Please update the EDC or provide the laboratory information.\n"
                        f"Clinical Data Management Team"
                    ),
                    rationale="Missing lab reference ranges may impact safety assessments and data interpretation.",
                    confidence=0.90,
                    requires_approval=False,
                    auto_executable=True,
                    source_data={
                        'lab_name': str(lab_name),
                        'test_name': str(test_name),
                        'action': str(action_needed),
                        'form': str(form_name)
                    }
                )
                self.recommendations.append(rec)
    
    def _generate_smart_communications(
        self,
        site_metrics: Dict[str, SiteMetrics],
        twins: List[DigitalPatientTwin],
        study_id: str
    ):
        """
        Generate smart, aggregated communications for overburdened sites
        
        For sites with SSM="Red" or high query counts:
        - Soften tone
        - Aggregate multiple issues into Weekly Digest
        - Prevent alert fatigue
        """
        for site_id, pending_items in self._site_pending_items.items():
            site_burden = self._site_burden_cache.get(site_id, {})
            
            if not site_burden.get('is_overburdened', False):
                continue  # Already handled individually
            
            if not pending_items:
                continue
            
            metrics = site_metrics.get(site_id)
            
            # Group items by subject
            subjects_data = {}
            for item in pending_items:
                subj = item.get('subject_id', 'Unknown')
                if subj not in subjects_data:
                    subjects_data[subj] = {'visits': [], 'labs': [], 'missing_visits': 0}
                
                if item['type'] == 'overdue_visit':
                    subjects_data[subj]['visits'].append(item)
                elif item['type'] == 'missing_lab':
                    subjects_data[subj]['labs'].append(item)
                elif item['type'] == 'missing_visit':
                    subjects_data[subj]['missing_visits'] = item.get('count', 0)
            
            # Build subject list for digest
            subject_lines = []
            priority_subjects = []
            
            for subj_id, data in subjects_data.items():
                issues = []
                max_days = 0
                
                for visit in data['visits']:
                    days = visit.get('days_outstanding', 0)
                    max_days = max(max_days, days)
                    issues.append(f"{visit.get('visit_name', 'Visit')} ({days} days)")
                
                for lab in data['labs']:
                    issues.append(f"Missing {lab.get('test_name', 'lab data')}")
                
                if data['missing_visits'] > 0:
                    issues.append(f"{data['missing_visits']} missing visit(s)")
                
                if issues:
                    line = f"  • Subject {subj_id}: {', '.join(issues)}"
                    subject_lines.append(line)
                    
                    if max_days >= self.config.escalation_days:
                        priority_subjects.append(subj_id)
            
            if not subject_lines:
                continue
            
            # Determine if we need a soft reminder or digest
            ssm_status = site_burden.get('ssm_status', 'Unknown')
            
            if ssm_status in ['Red', 'Critical']:
                # Use soft reminder tone for very stressed sites
                items_text = "\n".join(subject_lines)
                message = self.SOFT_REMINDER_TEMPLATE.format(items_list=items_text)
                title = f"Friendly Reminder: Site {site_id} ({len(subjects_data)} subjects)"
                self._communication_stats['soft_reminders'] += 1
            else:
                # Use weekly digest format
                message = (
                    f"Weekly Site Summary: {site_id}\n"
                    f"{'='*50}\n\n"
                    f"Dear Site Coordinator,\n\n"
                    f"This is your consolidated weekly summary to help manage outstanding items efficiently.\n\n"
                    f"SUMMARY:\n"
                    f"- Open Queries: {site_burden.get('total_queries', 0)}\n"
                    f"- Subjects with Issues: {len(subjects_data)}\n"
                    f"- SSM Status: {ssm_status}\n"
                    f"- DQI Score: {metrics.data_quality_index:.1f}%\n\n" if metrics else "\n"
                    f"OUTSTANDING SUBJECTS:\n"
                    f"{chr(10).join(subject_lines)}\n\n"
                    f"We understand your workload and have consolidated these items to minimize disruption. "
                    f"Please address items by priority when possible.\n\n"
                    f"Thank you for your partnership.\n"
                    f"Clinical Data Management Team"
                )
                title = f"Weekly Site Digest: {site_id}"
                self._communication_stats['weekly_digests'] += 1
            
            # Determine priority based on escalation needs
            priority = ActionPriority.HIGH if priority_subjects else ActionPriority.MEDIUM
            
            rec = self._create_recommendation(
                action_type=ActionType.SEND_REMINDER,
                priority=priority,
                subject_id="MULTIPLE",
                site_id=site_id,
                study_id=study_id,
                title=title,
                description=message,
                rationale="Aggregated communication reduces alert fatigue while ensuring site awareness. "
                         f"Site has SSM={ssm_status} and {site_burden.get('total_queries', 0)} open queries.",
                confidence=0.88,
                requires_approval=True,  # Aggregated messages should be reviewed
                source_data={
                    'site_burden': site_burden,
                    'subjects_affected': list(subjects_data.keys()),
                    'total_issues': len(pending_items),
                    'priority_subjects': priority_subjects,
                    'communication_type': 'soft_reminder' if ssm_status in ['Red', 'Critical'] else 'weekly_digest'
                }
            )
            self.recommendations.append(rec)
    
    def get_liaison_statistics(self) -> Dict[str, Any]:
        """Get statistics about Lia's communications"""
        return {
            'total_recommendations': len(self.recommendations),
            'communication_stats': self._communication_stats,
            'sites_analyzed': len(self._site_burden_cache),
            'overburdened_sites': sum(
                1 for s in self._site_burden_cache.values() 
                if s.get('is_overburdened', False)
            )
        }


class SupervisorAgent:
    """
    The Supervisor Agent - Orchestrates the Multi-Agent System
    
    Design Pattern: Blackboard Architecture with Supervisor Coordination
    
    The Supervisor constantly monitors the unified "Digital Patient Twin" and:
    1. DELEGATES tasks: "Rex, check safety. Codex, check coding. Lia, check visits."
    2. AGGREGATES and prioritizes results across all agents
    3. OVERRIDES lower-priority actions when critical issues arise
    4. MANAGES signal-to-noise ratio to prevent alert fatigue
    5. ENFORCES SOP compliance (checks Page Action Status before querying)
    
    Key Innovation - "White Space" Reduction:
    - Traditional cycle: Data Entry -> Extract -> Manual Review -> Query (2+ weeks)
    - Agentic cycle: Data Snapshot -> Immediate Analysis -> Auto-Action (near real-time)
    
    SOP Compliance:
    - Checks Page Action Status (Locked, Frozen) before any action
    - Never attempts to query a "Locked" page
    - Respects Queries status columns in CPID_EDC_Metrics
    """
    
    # SOP Status Constants
    PAGE_STATUS_LOCKED = 'Locked'
    PAGE_STATUS_FROZEN = 'Frozen'
    PAGE_STATUS_OPEN = 'Open'
    
    # Priority Override Thresholds
    SAFETY_OVERRIDE_PRIORITY = ActionPriority.CRITICAL
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or DEFAULT_AGENT_CONFIG
        
        # Initialize specialized agents
        self.rex = ReconciliationAgent(config)
        self.codex = CodingAgent(config)
        self.lia = SiteLiaisonAgent(config)
        
        # Recommendation storage
        self.all_recommendations: List[AgentRecommendation] = []
        self.prioritized_recommendations: List[AgentRecommendation] = []
        self.suppressed_recommendations: List[AgentRecommendation] = []
        
        # Blackboard - shared state across agents
        self._blackboard: Dict[str, Any] = {
            'critical_safety_subjects': set(),
            'critical_safety_sites': set(),
            'locked_pages': {},  # subject_id -> list of locked form OIDs
            'frozen_pages': {},  # subject_id -> list of frozen form OIDs
            'page_action_status': {},  # subject_id -> {form_oid: status}
            'active_queries': {},  # subject_id -> count of open queries
            'analysis_timestamp': None,
            'white_space_metrics': {
                'data_snapshot_time': None,
                'analysis_start_time': None,
                'analysis_end_time': None,
                'total_processing_seconds': 0
            }
        }
        
        # Orchestration statistics
        self._orchestration_stats = {
            'total_delegations': 0,
            'safety_overrides': 0,
            'sop_blocks': 0,  # Actions blocked due to Locked/Frozen status
            'signal_noise_filtered': 0,
            'white_space_reduction_hours': 0
        }
    
    def run_analysis(
        self,
        twins: List[DigitalPatientTwin],
        site_metrics: Dict[str, SiteMetrics],
        study_data: Dict[str, pd.DataFrame],
        study_id: str,
        cpid_data: Optional[pd.DataFrame] = None,
        knowledge_graph: Optional[Any] = None,
        query_engine: Optional[Any] = None,
        federated_query_engine: Optional[Any] = None
    ) -> Dict[str, List[AgentRecommendation]]:
        """
        Run full orchestrated analysis with all agents
        
        Orchestration Pipeline:
        1. Initialize blackboard with SOP compliance data
        2. Delegate to Rex (Safety/Reconciliation) - HIGHEST PRIORITY
        3. Delegate to Codex (Coding) - HIGH PRIORITY
        4. Delegate to Lia (Site Liaison) - MEDIUM PRIORITY
        5. Apply safety overrides (suppress routine items if critical issues exist)
        6. Filter by SOP compliance (no actions on Locked pages)
        7. Aggregate and prioritize final recommendations
        """
        from datetime import datetime
        
        # Track white space reduction
        self._blackboard['white_space_metrics']['analysis_start_time'] = datetime.now()
        
        results = {
            'rex': [],
            'codex': [],
            'lia': [],
            'all': [],
            'prioritized': [],
            'suppressed': [],
            'sop_blocked': []
        }
        
        # Step 1: Initialize blackboard with SOP compliance data
        logger.info("Supervisor: Initializing blackboard with SOP compliance data...")
        self._initialize_blackboard(twins, cpid_data)
        
        # Step 2: Delegate to Rex (Safety/Reconciliation) - FIRST
        logger.info("Supervisor: Delegating to Rex (Reconciliation Agent)...")
        self._orchestration_stats['total_delegations'] += 1
        results['rex'] = self.rex.analyze(
            twins=twins,
            sae_data=study_data.get('sae_dashboard'),
            edrr_data=study_data.get('compiled_edrr'),
            missing_pages=study_data.get('missing_pages'),
            study_id=study_id
        )
        
        # Update blackboard with critical safety issues
        self._update_safety_blackboard(results['rex'])
        
        # Step 3: Delegate to Codex (Coding)
        logger.info("Supervisor: Delegating to Codex (Coding Agent)...")
        self._orchestration_stats['total_delegations'] += 1
        results['codex'] = self.codex.analyze(
            twins=twins,
            meddra_data=study_data.get('meddra_coding'),
            whodra_data=study_data.get('whodra_coding'),
            study_id=study_id
        )
        
        # Step 5: Graph-based analysis (if available)
        graph_recommendations = []
        if knowledge_graph and query_engine:
            logger.info("Supervisor: Running graph-based analysis...")
            graph_recommendations = self._run_graph_based_analysis(
                knowledge_graph, query_engine, study_id, federated_query_engine
            )
            results['graph'] = graph_recommendations
        
        # Aggregate all recommendations
        self.all_recommendations = results['rex'] + results['codex'] + results['lia']
        results['all'] = self.all_recommendations
        
        # Step 5: Apply safety overrides
        logger.info("Supervisor: Applying safety override logic...")
        post_override_recs, suppressed = self._apply_safety_overrides(self.all_recommendations)
        self.suppressed_recommendations = suppressed
        results['suppressed'] = suppressed
        
        # Step 6: Filter by SOP compliance
        logger.info("Supervisor: Checking SOP compliance (Locked/Frozen pages)...")
        sop_compliant_recs, sop_blocked = self._filter_by_sop_compliance(post_override_recs)
        results['sop_blocked'] = sop_blocked
        
        # Step 7: Prioritize and limit
        logger.info("Supervisor: Prioritizing and filtering recommendations...")
        self.prioritized_recommendations = self._prioritize_recommendations(sop_compliant_recs)

        # Step 8: Enhance prioritization with engineered features
        logger.info("Supervisor: Enhancing prioritization with engineered features...")
        self.prioritized_recommendations = self._enhance_prioritization_with_features(
            self.prioritized_recommendations, twins
        )

        results['prioritized'] = self.prioritized_recommendations
        
        # Calculate white space metrics
        self._blackboard['white_space_metrics']['analysis_end_time'] = datetime.now()
        processing_time = (
            self._blackboard['white_space_metrics']['analysis_end_time'] - 
            self._blackboard['white_space_metrics']['analysis_start_time']
        ).total_seconds()
        self._blackboard['white_space_metrics']['total_processing_seconds'] = processing_time
        
        # Estimate white space reduction (traditional = 14 days = 336 hours)
        traditional_hours = 336  # 2-week manual cycle
        agentic_hours = processing_time / 3600
        self._orchestration_stats['white_space_reduction_hours'] = traditional_hours - agentic_hours
        
        logger.info(
            f"Supervisor completed analysis: "
            f"{len(self.all_recommendations)} total, "
            f"{len(self.prioritized_recommendations)} prioritized, "
            f"{len(suppressed)} suppressed (safety override), "
            f"{len(sop_blocked)} blocked (SOP compliance)"
        )
        
        return results
    
    def _initialize_blackboard(
        self,
        twins: List[DigitalPatientTwin],
        cpid_data: Optional[pd.DataFrame]
    ):
        """
        Initialize the blackboard with SOP compliance data
        
        Extracts from CPID_EDC_Metrics:
        - Page Action Status (Locked, Frozen)
        - Queries status
        - Open queries count
        """
        self._blackboard['critical_safety_subjects'] = set()
        self._blackboard['critical_safety_sites'] = set()
        self._blackboard['locked_pages'] = {}
        self._blackboard['frozen_pages'] = {}
        self._blackboard['active_queries'] = {}
        self._blackboard['analysis_timestamp'] = datetime.now()
        
        if cpid_data is None:
            logger.warning("No CPID data provided - SOP compliance checks will be limited")
            return
        
        # Process CPID data for each subject
        for _, row in cpid_data.iterrows():
            subject_id = str(row.get('Subject ID', ''))
            if not subject_id or pd.isna(subject_id) or subject_id == 'nan':
                continue
            
            # Extract page status counts (handle NaN values)
            crfs_frozen_val = row.get('# CRFs Frozen', 0)
            crfs_frozen = int(crfs_frozen_val) if not pd.isna(crfs_frozen_val) else 0
            
            crfs_locked_val = row.get('# CRFs Locked', 0)
            crfs_locked = int(crfs_locked_val) if not pd.isna(crfs_locked_val) else 0
            
            # Track subjects with locked/frozen pages
            if crfs_locked > 0:
                self._blackboard['locked_pages'][subject_id] = crfs_locked
            
            if crfs_frozen > 0:
                self._blackboard['frozen_pages'][subject_id] = crfs_frozen
            
            # Track active queries
            open_queries = 0
            for col in row.index:
                if 'Open' in str(col) and 'Queries' in str(col):
                    val = row.get(col, 0)
                    if not pd.isna(val):
                        open_queries += int(val)
            
            self._blackboard['active_queries'][subject_id] = open_queries
        
        logger.info(
            f"Blackboard initialized: "
            f"{len(self._blackboard['locked_pages'])} subjects with locked pages, "
            f"{len(self._blackboard['frozen_pages'])} subjects with frozen pages"
        )
    
    def _update_safety_blackboard(self, rex_recommendations: List[AgentRecommendation]):
        """
        Update blackboard with critical safety issues from Rex
        
        These subjects/sites will trigger override logic for lower-priority agents
        """
        for rec in rex_recommendations:
            # Check for critical safety issues
            is_critical_safety = (
                rec.priority in [ActionPriority.CRITICAL, ActionPriority.HIGH] and
                rec.action_type in [
                    ActionType.QUERY_TO_SAFETY,
                    ActionType.ALERT_MEDICAL_MONITOR,
                    ActionType.ALERT_CRA
                ]
            )
            
            # Check for specific safety keywords
            safety_keywords = ['SAE', 'safety', 'unreported', 'serious', 'death', 'hospitalization']
            has_safety_keyword = any(
                kw.lower() in rec.title.lower() or kw.lower() in rec.description.lower()
                for kw in safety_keywords
            )
            
            if is_critical_safety or has_safety_keyword:
                self._blackboard['critical_safety_subjects'].add(rec.subject_id)
                self._blackboard['critical_safety_sites'].add(rec.site_id)
        
        logger.info(
            f"Safety blackboard updated: "
            f"{len(self._blackboard['critical_safety_subjects'])} critical subjects, "
            f"{len(self._blackboard['critical_safety_sites'])} critical sites"
        )
    
    def _apply_safety_overrides(
        self,
        recommendations: List[AgentRecommendation]
    ) -> Tuple[List[AgentRecommendation], List[AgentRecommendation]]:
        """
        Apply safety override logic
        
        Rule: If Rex finds a serious safety issue (e.g., unreported SAE),
        suppress routine reminders from Lia to ensure site focuses on safety.
        
        Override Logic:
        1. For subjects with critical safety issues:
           - Keep: Safety queries, escalations, CRA alerts
           - Suppress: Routine visit reminders, missing page alerts
        2. For sites with critical safety issues:
           - Suppress: Weekly digests, soft reminders
           - Keep: Individual urgent items
        """
        kept_recommendations = []
        suppressed_recommendations = []
        
        critical_subjects = self._blackboard['critical_safety_subjects']
        critical_sites = self._blackboard['critical_safety_sites']
        
        for rec in recommendations:
            should_suppress = False
            suppression_reason = None
            
            # Check if this subject has critical safety issues
            if rec.subject_id in critical_subjects:
                # Suppress routine items for critical subjects
                if rec.agent_name == "Lia" and rec.action_type in [
                    ActionType.SEND_REMINDER,
                    ActionType.QUERY_TO_SITE
                ]:
                    # Check if it's truly routine (not safety-related)
                    safety_keywords = ['SAE', 'safety', 'serious', 'urgent']
                    is_safety_related = any(
                        kw.lower() in rec.title.lower()
                        for kw in safety_keywords
                    )
                    
                    if not is_safety_related:
                        should_suppress = True
                        suppression_reason = f"Subject {rec.subject_id} has critical safety issue - routine reminder suppressed"
            
            # Check if this site has critical safety issues
            if rec.site_id in critical_sites:
                # Suppress aggregated communications
                if rec.subject_id == "MULTIPLE" or "Digest" in rec.title:
                    should_suppress = True
                    suppression_reason = f"Site {rec.site_id} has critical safety issue - aggregated communication suppressed"
            
            if should_suppress:
                # Mark as suppressed with reason
                rec.source_data['suppressed'] = True
                rec.source_data['suppression_reason'] = suppression_reason
                suppressed_recommendations.append(rec)
                self._orchestration_stats['safety_overrides'] += 1
            else:
                kept_recommendations.append(rec)
        
        return kept_recommendations, suppressed_recommendations
    
    def _filter_by_sop_compliance(
        self,
        recommendations: List[AgentRecommendation]
    ) -> Tuple[List[AgentRecommendation], List[AgentRecommendation]]:
        """
        Filter recommendations by SOP compliance
        
        Rule: Never attempt to query a "Locked" page
        
        Actions blocked for Locked pages:
        - QUERY_TO_SITE
        - REQUEST_CLARIFICATION
        - AUTO_CODE (can't modify locked data)
        
        Actions allowed for Locked pages:
        - QUERY_TO_SAFETY (safety supersedes locks)
        - ALERT_CRA
        - ALERT_MEDICAL_MONITOR
        """
        compliant_recommendations = []
        blocked_recommendations = []
        
        locked_subjects = set(self._blackboard['locked_pages'].keys())
        
        # Actions that should be blocked for locked pages
        blockable_actions = [
            ActionType.QUERY_TO_SITE,
            ActionType.REQUEST_CLARIFICATION,
            ActionType.AUTO_CODE,
            ActionType.PROPOSE_CODE
        ]
        
        # Actions that override locks (safety-critical)
        override_actions = [
            ActionType.QUERY_TO_SAFETY,
            ActionType.ALERT_CRA,
            ActionType.ALERT_MEDICAL_MONITOR
        ]
        
        for rec in recommendations:
            should_block = False
            block_reason = None
            
            # Check if subject has locked pages
            if rec.subject_id in locked_subjects:
                # Check if action should be blocked
                if rec.action_type in blockable_actions:
                    # Safety-related items override locks
                    if rec.action_type not in override_actions:
                        should_block = True
                        block_reason = (
                            f"Subject {rec.subject_id} has {self._blackboard['locked_pages'][rec.subject_id]} "
                            f"locked CRF(s) - action blocked per SOP"
                        )
            
            if should_block:
                rec.source_data['sop_blocked'] = True
                rec.source_data['block_reason'] = block_reason
                blocked_recommendations.append(rec)
                self._orchestration_stats['sop_blocks'] += 1
            else:
                compliant_recommendations.append(rec)
        
        return compliant_recommendations, blocked_recommendations
    
    def _prioritize_recommendations(
        self,
        recommendations: List[AgentRecommendation]
    ) -> List[AgentRecommendation]:
        """
        Prioritize and filter recommendations to manage signal-to-noise ratio
        
        Priority Order:
        1. CRITICAL - Always include (safety issues)
        2. HIGH - Include up to limit per site
        3. MEDIUM - Include if site not saturated
        4. LOW - Include only if room available
        
        Signal-to-Noise Management:
        - Max queries per site per day (configurable)
        - Prefer safety over data quality over routine
        """
        # Separate by priority
        by_priority = {
            ActionPriority.CRITICAL: [],
            ActionPriority.HIGH: [],
            ActionPriority.MEDIUM: [],
            ActionPriority.LOW: []
        }
        
        for rec in recommendations:
            by_priority[rec.priority].append(rec)
        
        # Track queries per site
        site_query_counts = {}
        final_recommendations = []
        
        # Process in priority order
        for priority in [ActionPriority.CRITICAL, ActionPriority.HIGH, 
                        ActionPriority.MEDIUM, ActionPriority.LOW]:
            for rec in by_priority[priority]:
                site_id = rec.site_id
                if site_id not in site_query_counts:
                    site_query_counts[site_id] = 0
                
                # Always include critical
                if priority == ActionPriority.CRITICAL:
                    final_recommendations.append(rec)
                    site_query_counts[site_id] += 1
                # Check limit for others
                elif site_query_counts[site_id] < self.config.max_queries_per_site_per_day:
                    final_recommendations.append(rec)
                    site_query_counts[site_id] += 1
                else:
                    self._orchestration_stats['signal_noise_filtered'] += 1
        
        # Sort by priority, then by timestamp
        final_recommendations.sort(
            key=lambda r: (r.priority.value, r.created_at)
        )
        
        return final_recommendations
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary of orchestrated analysis"""
        return {
            'total_recommendations': len(self.all_recommendations),
            'prioritized_recommendations': len(self.prioritized_recommendations),
            'suppressed_by_safety_override': len(self.suppressed_recommendations),
            'blocked_by_sop': self._orchestration_stats['sop_blocks'],
            'by_agent': {
                'Rex (Reconciliation)': len(self.rex.recommendations),
                'Codex (Coding)': len(self.codex.recommendations),
                'Lia (Site Liaison)': len(self.lia.recommendations)
            },
            'by_priority': {
                'Critical': len([r for r in self.prioritized_recommendations 
                               if r.priority == ActionPriority.CRITICAL]),
                'High': len([r for r in self.prioritized_recommendations 
                           if r.priority == ActionPriority.HIGH]),
                'Medium': len([r for r in self.prioritized_recommendations 
                             if r.priority == ActionPriority.MEDIUM]),
                'Low': len([r for r in self.prioritized_recommendations 
                          if r.priority == ActionPriority.LOW])
            },
            'orchestration_stats': self._orchestration_stats,
            'white_space_metrics': self._blackboard['white_space_metrics'],
            'blackboard_status': {
                'critical_safety_subjects': len(self._blackboard['critical_safety_subjects']),
                'critical_safety_sites': len(self._blackboard['critical_safety_sites']),
                'subjects_with_locked_pages': len(self._blackboard['locked_pages']),
                'subjects_with_frozen_pages': len(self._blackboard['frozen_pages'])
            }
        }
    
    def get_blackboard_state(self) -> Dict:
        """Get current state of the shared blackboard"""
        return {
            'critical_safety_subjects': list(self._blackboard['critical_safety_subjects']),
            'critical_safety_sites': list(self._blackboard['critical_safety_sites']),
            'locked_pages': self._blackboard['locked_pages'],
            'frozen_pages': self._blackboard['frozen_pages'],
            'active_queries': self._blackboard['active_queries'],
            'analysis_timestamp': str(self._blackboard['analysis_timestamp'])
        }
    
    def explain_orchestration_decision(self, recommendation_id: str) -> str:
        """
        Explain why a specific recommendation was made or suppressed
        
        Useful for audit trails and understanding agent decisions
        """
        # Search in all recommendations
        for rec in self.all_recommendations:
            if rec.recommendation_id == recommendation_id:
                explanation = [
                    f"Recommendation: {rec.title}",
                    f"Agent: {rec.agent_name}",
                    f"Priority: {rec.priority.value}",
                    f"Action Type: {rec.action_type.value}",
                    f"Confidence: {rec.confidence_score:.1%}",
                    "",
                    "Orchestration Decision:"
                ]
                
                if rec.source_data.get('suppressed'):
                    explanation.append(f"  STATUS: SUPPRESSED (Safety Override)")
                    explanation.append(f"  Reason: {rec.source_data.get('suppression_reason', 'Unknown')}")
                elif rec.source_data.get('sop_blocked'):
                    explanation.append(f"  STATUS: BLOCKED (SOP Compliance)")
                    explanation.append(f"  Reason: {rec.source_data.get('block_reason', 'Unknown')}")
                elif rec in self.prioritized_recommendations:
                    explanation.append(f"  STATUS: APPROVED")
                    explanation.append(f"  Auto-executable: {rec.auto_executable}")
                    explanation.append(f"  Requires Human Approval: {rec.requires_human_approval}")
                else:
                    explanation.append(f"  STATUS: FILTERED (Signal-to-Noise)")
                    explanation.append(f"  Reason: Site query limit reached")
                
                return "\n".join(explanation)
        
        return f"Recommendation {recommendation_id} not found"
    
    def analyze_cross_study_patterns(self, graph_nodes: List[Any]) -> List[AgentRecommendation]:
        """
        Analyze cross-study patterns from knowledge graph nodes
        
        This method provides insights across multiple studies by analyzing
        patterns in the knowledge graph nodes.
        
        Args:
            graph_nodes: List of nodes from the knowledge graph
            
        Returns:
            List of recommendations based on cross-study analysis
        """
        recommendations = []
        
        if not graph_nodes:
            return recommendations
        
        try:
            # Extract study patterns from nodes
            study_patterns = {}
            patient_patterns = {}
            
            for node in graph_nodes:
                # Extract study information from node data
                if hasattr(node, 'get') and callable(node.get):
                    study_id = node.get('study_id', 'unknown')
                    patient_id = node.get('patient_id', 'unknown')
                    node_type = node.get('type', 'unknown')
                    
                    if study_id not in study_patterns:
                        study_patterns[study_id] = {'nodes': 0, 'types': set()}
                    study_patterns[study_id]['nodes'] += 1
                    study_patterns[study_id]['types'].add(node_type)
                    
                    if patient_id not in patient_patterns:
                        patient_patterns[patient_id] = {'studies': set(), 'types': set()}
                    patient_patterns[patient_id]['studies'].add(study_id)
                    patient_patterns[patient_id]['types'].add(node_type)
            
            # Generate cross-study insights
            multi_study_patients = [
                pid for pid, data in patient_patterns.items() 
                if len(data['studies']) > 1
            ]
            
            if multi_study_patients:
                rec = self._create_recommendation(
                    action_type=ActionType.ALERT_MEDICAL_MONITOR,
                    priority=ActionPriority.HIGH,
                    subject_id="MULTIPLE",
                    site_id="MULTIPLE",
                    study_id="CROSS_STUDY",
                    title=f"Cross-Study Patient Analysis: {len(multi_study_patients)} patients appear in multiple studies",
                    description=f"Analysis of knowledge graph reveals {len(multi_study_patients)} patients participating in multiple clinical studies. This may indicate patient overlap or data integration opportunities.",
                    rationale="Cross-study patient participation can reveal important patterns for patient safety monitoring and data quality assurance.",
                    confidence=0.85,
                    requires_approval=True,
                    source_data={
                        'multi_study_patients': multi_study_patients,
                        'total_studies_analyzed': len(study_patterns),
                        'study_breakdown': study_patterns
                    }
                )
                recommendations.append(rec)
            
            # Analyze study size patterns
            study_sizes = {sid: data['nodes'] for sid, data in study_patterns.items()}
            if study_sizes:
                avg_size = sum(study_sizes.values()) / len(study_sizes)
                large_studies = [sid for sid, size in study_sizes.items() if size > avg_size * 1.5]
                
                if large_studies:
                    rec = self._create_recommendation(
                        action_type=ActionType.ALERT_CRA,
                        priority=ActionPriority.MEDIUM,
                        subject_id="MULTIPLE",
                        site_id="MULTIPLE", 
                        study_id="CROSS_STUDY",
                        title=f"Study Size Analysis: {len(large_studies)} studies significantly larger than average",
                        description=f"Cross-study analysis shows {len(large_studies)} studies with data volume {avg_size:.0f}% above average. These studies may require additional monitoring.",
                        rationale="Identifying unusually large studies helps prioritize monitoring resources and identify potential data quality concerns.",
                        confidence=0.75,
                        requires_approval=False,
                        source_data={
                            'study_sizes': study_sizes,
                            'average_size': avg_size,
                            'large_studies': large_studies
                        }
                    )
                    recommendations.append(rec)
            
            logger.info(f"Cross-study analysis generated {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Error in cross-study pattern analysis: {e}")
        
        return recommendations
    
    def _run_graph_based_analysis(
        self,
        knowledge_graph: Any,
        query_engine: Any,
        study_id: str,
        federated_query_engine: Optional[Any] = None
    ) -> List[AgentRecommendation]:
        """
        Run graph-based analysis to generate recommendations
        
        Uses the knowledge graph to identify patterns that would be
        difficult to find with traditional SQL queries.
        """
        recommendations = []
        
        try:
            # Graph query: Patients needing immediate attention
            attention_patients = query_engine.find_patients_needing_attention()
            if attention_patients:
                rec = AgentRecommendation(
                    recommendation_id=f"graph_attention_{study_id}_{len(attention_patients)}",
                    agent_name="Graph Analytics",
                    action_type=ActionType.ALERT_MEDICAL_MONITOR,
                    priority=ActionPriority.CRITICAL,
                    subject_id="MULTIPLE",
                    site_id="MULTIPLE",
                    study_id=study_id,
                    title=f"Critical Patient Issues Detected via Graph Analysis",
                    description=(
                        f"Graph traversal identified {len(attention_patients)} patients with "
                        "Missing Visit AND Open Safety Query AND Uncoded Term - a complex "
                        "multi-condition pattern impossible to efficiently query in SQL."
                    ),
                    rationale=(
                        "Graph-native query executed in O(n) time where n = patient connections. "
                        "Traditional SQL would require 3-table JOIN with complex WHERE clauses "
                        "and perform O(n*m*k) worst-case operations."
                    ),
                    confidence_score=0.95,
                    requires_human_approval=True,
                    auto_executable=False,
                    source_data={
                        'query_type': 'multi_hop_attention',
                        'patient_count': len(attention_patients),
                        'conditions': ['missing_visits > 0', 'open_queries > 0', 'uncoded_terms > 0'],
                        'logic': 'AND'
                    }
                )
                recommendations.append(rec)
            
            # Graph analytics: Risk profiling
            from graph.graph_analytics import GraphAnalytics
            analytics = GraphAnalytics(knowledge_graph)
            risk_report = analytics.export_risk_report()
            
            if risk_report.get('high_risk_patients', 0) > 0:
                rec = AgentRecommendation(
                    recommendation_id=f"graph_risk_{study_id}_{risk_report.get('high_risk_patients', 0)}",
                    agent_name="Graph Analytics",
                    action_type=ActionType.ALERT_CRA,
                    priority=ActionPriority.HIGH,
                    subject_id="MULTIPLE",
                    site_id="MULTIPLE", 
                    study_id=study_id,
                    title=f"High-Risk Patient Cluster Identified",
                    description=(
                        f"Graph centrality analysis found {risk_report.get('high_risk_patients', 0)} "
                        "high-risk patients based on connection patterns and issue diversity."
                    ),
                    rationale=(
                        "Graph analytics calculated patient risk scores using network centrality "
                        "metrics (degree centrality, betweenness) that reveal hidden risk patterns "
                        "not visible in tabular data."
                    ),
                    confidence_score=0.90,
                    requires_human_approval=True,
                    auto_executable=False,
                    source_data=risk_report
                )
                recommendations.append(rec)
                
            # Cross-study analysis (if federated engine available)
            if federated_query_engine:
                logger.info("Supervisor: Running cross-study federated analysis...")
                cross_study_recs = self._run_cross_study_analysis(federated_query_engine, study_id)
                recommendations.extend(cross_study_recs)
                
        except Exception as e:
            logger.error(f"Error in graph-based analysis: {e}")
        
        return recommendations

    def _run_cross_study_analysis(
        self,
        federated_query_engine: Any,
        current_study_id: str
    ) -> List[AgentRecommendation]:
        """
        Run cross-study federated analysis to identify patterns across multiple studies
        
        Uses federated query engine to analyze patterns that span multiple clinical studies,
        identifying systemic issues, country-specific risks, and study performance comparisons.
        """
        recommendations = []
        
        try:
            # Run cross-study pattern analysis
            pattern_criteria = {
                'min_open_queries': 1,
                'min_uncoded_terms': 1,
                'min_missing_visits': 1,
                'risk_threshold': 5.0,
                'min_issue_types': 2
            }
            
            cross_study_results = federated_query_engine.query_cross_study_patient_patterns(pattern_criteria)
            
            # Recommendation 1: Cross-study country risk patterns
            country_patterns = [p for p in cross_study_results.get('patterns_identified', []) 
                              if p.get('pattern_type') == 'cross_study_country_risk']
            
            if country_patterns:
                for pattern in country_patterns:
                    rec = AgentRecommendation(
                        recommendation_id=f"cross_study_country_risk_{pattern['country']}_{len(pattern['studies_affected'])}",
                        agent_name="Federated Graph Analytics",
                        action_type=ActionType.ALERT_MEDICAL_MONITOR,
                        priority=ActionPriority.HIGH,
                        subject_id="MULTIPLE",
                        site_id="MULTIPLE",
                        study_id="MULTIPLE",
                        title=f"Cross-Study Risk Pattern: {pattern['country']}",
                        description=(
                            f"Country {pattern['country']} shows consistent high-risk patterns "
                            f"across {len(pattern['studies_affected'])} studies: {', '.join(pattern['studies_affected'])}. "
                            f"{pattern['description']}"
                        ),
                        rationale=(
                            "Federated graph analysis across multiple studies identified systemic "
                            "risk patterns that would be invisible in single-study analysis. "
                            "This suggests potential regional training needs or protocol issues."
                        ),
                        confidence_score=0.85,
                        requires_human_approval=True,
                        auto_executable=False,
                        source_data={
                            'pattern_type': 'cross_study_country_risk',
                            'country': pattern['country'],
                            'studies_affected': pattern['studies_affected'],
                            'severity': pattern.get('severity', 'Medium')
                        }
                    )
                    recommendations.append(rec)
            
            # Recommendation 2: Study performance comparison
            performance_patterns = [p for p in cross_study_results.get('patterns_identified', []) 
                                  if 'performance' in p.get('pattern_type', '')]
            
            if performance_patterns:
                for pattern in performance_patterns:
                    priority = ActionPriority.CRITICAL if 'low_performing' in pattern['pattern_type'] else ActionPriority.MEDIUM
                    
                    rec = AgentRecommendation(
                        recommendation_id=f"study_performance_{pattern['pattern_type']}_{len(pattern.get('studies', []))}",
                        agent_name="Federated Graph Analytics",
                        action_type=ActionType.ALERT_MEDICAL_MONITOR,
                        priority=priority,
                        subject_id="MULTIPLE",
                        site_id="MULTIPLE",
                        study_id="MULTIPLE",
                        title=f"Study Performance Analysis: {pattern['pattern_type'].replace('_', ' ').title()}",
                        description=(
                            f"Cross-study analysis identified {pattern['description']}. "
                            f"Affected studies: {', '.join(pattern.get('studies', []))}"
                        ),
                        rationale=(
                            "Federated analytics compared risk patterns across studies to identify "
                            "performance outliers. High-performing studies may have effective practices "
                            "worth replicating; low-performing studies may need additional oversight."
                        ),
                        confidence_score=0.80,
                        requires_human_approval=True,
                        auto_executable=False,
                        source_data={
                            'pattern_type': pattern['pattern_type'],
                            'studies': pattern.get('studies', []),
                            'avg_risk_ratio': pattern.get('avg_risk_ratio')
                        }
                    )
                    recommendations.append(rec)
            
            # Recommendation 3: Systemic multi-study issues
            aggregates = cross_study_results.get('cross_study_aggregates', {})
            total_studies = cross_study_results.get('total_studies', 0)
            
            if total_studies > 1 and aggregates.get('studies_with_issues', 0) == total_studies:
                rec = AgentRecommendation(
                    recommendation_id=f"systemic_multi_study_issues_{total_studies}",
                    agent_name="Federated Graph Analytics",
                    action_type=ActionType.ALERT_MEDICAL_MONITOR,
                    priority=ActionPriority.CRITICAL,
                    subject_id="MULTIPLE",
                    site_id="MULTIPLE",
                    study_id="MULTIPLE",
                    title="Systemic Issues Across All Studies",
                    description=(
                        f"All {total_studies} studies show patient risk patterns requiring attention. "
                        f"Total at-risk patients: {aggregates.get('total_at_risk_patients', 0)}, "
                        f"Total high-risk patients: {aggregates.get('total_high_risk_patients', 0)}. "
                        f"This suggests potential protocol-level or training issues affecting the entire program."
                    ),
                    rationale=(
                        "Federated analysis revealed that every study in the program exhibits "
                        "similar risk patterns, indicating systemic issues rather than study-specific problems. "
                        "This requires program-level intervention and root cause analysis."
                    ),
                    confidence_score=0.95,
                    requires_human_approval=True,
                    auto_executable=False,
                    source_data={
                        'total_studies': total_studies,
                        'studies_with_issues': aggregates.get('studies_with_issues', 0),
                        'aggregates': aggregates
                    }
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error(f"Error in cross-study analysis: {e}")
        
        return recommendations

    def _enhance_prioritization_with_features(
        self,
        recommendations: List[AgentRecommendation],
        twins: List[DigitalPatientTwin]
    ) -> List[AgentRecommendation]:
        """
        Enhance recommendation prioritization using engineered features from DigitalPatientTwin

        Uses three engineered features:
        1. Operational Velocity Index - identifies bottleneck sites
        2. Normalized Data Density - prioritizes based on query patterns
        3. Manipulation Risk Score - flags high-risk data integrity issues

        Args:
            recommendations: List of agent recommendations to enhance
            twins: List of DigitalPatientTwin objects with engineered features

        Returns:
            Enhanced recommendations with feature-based prioritization
        """
        if not recommendations or not twins:
            return recommendations

        # Build twin lookup by subject_id for fast access
        twin_lookup = {twin.subject_id: twin for twin in twins}

        enhanced_recommendations = []

        for rec in recommendations:
            # Skip multi-subject recommendations for now
            if rec.subject_id == "MULTIPLE":
                enhanced_recommendations.append(rec)
                continue

            # Get twin for this subject
            twin = twin_lookup.get(rec.subject_id)
            if not twin:
                enhanced_recommendations.append(rec)
                continue

            # Extract engineered features
            features = twin.risk_metrics

            # Feature 1: Operational Velocity Index
            # Bottleneck sites get higher priority
            velocity_boost = 0
            if features.is_bottleneck:
                velocity_boost = 2  # Major priority boost for bottlenecks
            elif features.net_velocity < 0:
                velocity_boost = 1  # Minor boost for negative velocity

            # Feature 2: Normalized Data Density
            # Sites with high query density relative to peers get attention
            density_boost = 0
            if features.query_density_percentile > 80:
                density_boost = 1  # High density sites need monitoring

            # Feature 3: Manipulation Risk Score
            # High manipulation risk gets highest priority
            risk_boost = 0
            if features.manipulation_risk_score == "Critical":
                risk_boost = 3  # Maximum boost for critical manipulation risk
            elif features.manipulation_risk_score == "High":
                risk_boost = 2
            elif features.manipulation_risk_score == "Elevated":
                risk_boost = 1

            # Feature 4: Composite Risk Score
            # Use engineered composite score for additional prioritization
            composite_boost = 0
            if features.composite_risk_score >= 80:
                composite_boost = 2
            elif features.composite_risk_score >= 60:
                composite_boost = 1

            # Calculate total priority boost
            total_boost = velocity_boost + density_boost + risk_boost + composite_boost

            # Apply priority enhancement
            original_priority_value = rec.priority.value
            enhanced_priority_value = max(1, original_priority_value - total_boost)

            # Map back to ActionPriority enum
            enhanced_priority = ActionPriority(enhanced_priority_value)

            # Update recommendation with enhanced priority and feature context
            enhanced_rec = AgentRecommendation(
                recommendation_id=rec.recommendation_id,
                agent_name=rec.agent_name,
                action_type=rec.action_type,
                priority=enhanced_priority,
                subject_id=rec.subject_id,
                site_id=rec.site_id,
                study_id=rec.study_id,
                title=rec.title,
                description=rec.description,
                rationale=rec.rationale + self._generate_feature_rationale(features),
                confidence_score=min(1.0, rec.confidence_score + (total_boost * 0.05)),  # Slight confidence boost
                requires_human_approval=rec.requires_human_approval,
                auto_executable=rec.auto_executable,
                source_data={
                    **rec.source_data,
                    'engineered_features': {
                        'velocity_boost': velocity_boost,
                        'density_boost': density_boost,
                        'risk_boost': risk_boost,
                        'composite_boost': composite_boost,
                        'total_boost': total_boost,
                        'features_used': {
                            'is_bottleneck': features.is_bottleneck,
                            'query_density_percentile': features.query_density_percentile,
                            'manipulation_risk_level': features.manipulation_risk_score,
                            'composite_risk_score': features.composite_risk_score
                        }
                    }
                },
                created_at=rec.created_at
            )

            enhanced_recommendations.append(enhanced_rec)

        logger.info(f"Enhanced prioritization for {len(enhanced_recommendations)} recommendations using engineered features")
        return enhanced_recommendations

    def _generate_feature_rationale(self, features: RiskMetrics) -> str:
        """
        Generate rationale text based on engineered features

        Args:
            features: RiskMetrics with engineered features

        Returns:
            Additional rationale text for recommendation
        """
        rationale_parts = []

        if features.is_bottleneck:
            rationale_parts.append("Patient is at a bottleneck site with negative query resolution velocity.")

        if features.query_density_percentile > 80:
            rationale_parts.append(f"Site has high query density ({features.query_density_percentile:.1f} percentile) indicating potential data quality issues.")

        if features.manipulation_risk_score in ["Critical", "High"]:
            rationale_parts.append(f"Patient shows {features.manipulation_risk_score.lower()} manipulation risk based on inactivation patterns.")

        if features.composite_risk_score >= 60:
            rationale_parts.append(f"Composite risk score of {features.composite_risk_score:.1f} indicates elevated overall risk.")

        if rationale_parts:
            return " Engineered feature analysis: " + " ".join(rationale_parts)
        else:
            return ""

