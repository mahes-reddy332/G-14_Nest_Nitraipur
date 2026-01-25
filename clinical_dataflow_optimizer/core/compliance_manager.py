"""
Regulatory Compliance Manager

Central integration point for all regulatory compliance features:
- ICH E6 R2/R3: Risk-Based Quality Management
- 21 CFR Part 11: Electronic Records and Signatures
- FDA AI/ML SaMD: Human-in-the-Loop oversight

This module coordinates:
1. Audit Trail logging for all agent actions
2. HITL approval workflows for critical actions
3. Risk-based monitoring based on DQI scores
4. Site categorization (Green/Yellow/Red quadrants)
5. Compliance reporting for regulatory submissions
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import json

# Local imports
from .audit_trail import (
    AuditTrailManager, 
    AuditEntry, 
    ActionType, 
    ActionCategory,
    AgentIdentifier,
    ComplianceStandard,
    get_audit_manager,
    log_agent_action
)
from .hitl_workflow import (
    HITLManager,
    ApprovalRequest,
    ApprovalStatus,
    ActionRiskLevel,
    ApproverRole,
    get_hitl_manager,
    requires_hitl_approval
)


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class SiteRiskCategory(Enum):
    """Site risk categories based on DQI scores (scatter plot quadrants)"""
    GREEN = auto()      # High DQI, low issues - reduced monitoring
    YELLOW = auto()     # Medium DQI - standard monitoring
    RED = auto()        # Low DQI, high issues - intensive monitoring


class MonitoringAction(Enum):
    """Actions to take based on monitoring results"""
    NONE = auto()
    ALERT = auto()
    ESCALATE = auto()
    IMMEDIATE_ACTION = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SiteRiskProfile:
    """Risk profile for a clinical trial site"""
    site_id: str
    site_name: str = ""
    country: str = ""
    
    # DQI-based metrics
    dqi_score: float = 0.0
    risk_category: SiteRiskCategory = SiteRiskCategory.YELLOW
    
    # Component scores
    visit_adherence_score: float = 0.0
    query_responsiveness_score: float = 0.0
    safety_reporting_score: float = 0.0
    data_entry_timeliness_score: float = 0.0
    protocol_compliance_score: float = 0.0
    
    # Monitoring settings
    sdv_rate: float = 0.50
    monitoring_frequency: str = "Monthly"
    auto_query_enabled: bool = True
    escalation_delay_days: int = 7
    
    # Trends
    dqi_trend: str = "STABLE"  # IMPROVING, STABLE, DECLINING
    previous_dqi_scores: List[float] = field(default_factory=list)
    
    # Flags
    requires_cra_intervention: bool = False
    requires_sponsor_escalation: bool = False
    cap_in_progress: bool = False
    
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'site_id': self.site_id,
            'site_name': self.site_name,
            'country': self.country,
            'dqi_score': self.dqi_score,
            'risk_category': self.risk_category.name,
            'component_scores': {
                'visit_adherence': self.visit_adherence_score,
                'query_responsiveness': self.query_responsiveness_score,
                'safety_reporting': self.safety_reporting_score,
                'data_entry_timeliness': self.data_entry_timeliness_score,
                'protocol_compliance': self.protocol_compliance_score
            },
            'monitoring_settings': {
                'sdv_rate': self.sdv_rate,
                'monitoring_frequency': self.monitoring_frequency,
                'auto_query_enabled': self.auto_query_enabled,
                'escalation_delay_days': self.escalation_delay_days
            },
            'dqi_trend': self.dqi_trend,
            'flags': {
                'requires_cra_intervention': self.requires_cra_intervention,
                'requires_sponsor_escalation': self.requires_sponsor_escalation,
                'cap_in_progress': self.cap_in_progress
            },
            'last_updated': self.last_updated.isoformat()
        }


@dataclass 
class ComplianceReport:
    """Regulatory compliance report for a study"""
    study_id: str
    report_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Audit statistics
    total_agent_actions: int = 0
    actions_by_agent: Dict[str, int] = field(default_factory=dict)
    actions_by_type: Dict[str, int] = field(default_factory=dict)
    
    # HITL statistics
    total_hitl_requests: int = 0
    approved_requests: int = 0
    rejected_requests: int = 0
    pending_requests: int = 0
    auto_approved_requests: int = 0
    avg_approval_time_hours: float = 0.0
    
    # Site risk distribution
    green_sites: int = 0
    yellow_sites: int = 0
    red_sites: int = 0
    
    # Risk-based monitoring impact
    sdv_reduction_percentage: float = 0.0
    estimated_hours_saved: float = 0.0
    
    # Compliance status
    audit_chain_valid: bool = True
    compliance_standards: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'study_id': self.study_id,
            'report_date': self.report_date.isoformat(),
            'audit_statistics': {
                'total_agent_actions': self.total_agent_actions,
                'by_agent': self.actions_by_agent,
                'by_type': self.actions_by_type
            },
            'hitl_statistics': {
                'total_requests': self.total_hitl_requests,
                'approved': self.approved_requests,
                'rejected': self.rejected_requests,
                'pending': self.pending_requests,
                'auto_approved': self.auto_approved_requests,
                'avg_approval_time_hours': self.avg_approval_time_hours
            },
            'site_risk_distribution': {
                'green': self.green_sites,
                'yellow': self.yellow_sites,
                'red': self.red_sites
            },
            'risk_based_monitoring_impact': {
                'sdv_reduction_percentage': self.sdv_reduction_percentage,
                'estimated_hours_saved': self.estimated_hours_saved
            },
            'compliance': {
                'audit_chain_valid': self.audit_chain_valid,
                'standards': self.compliance_standards
            }
        }


# =============================================================================
# COMPLIANCE MANAGER
# =============================================================================

class ComplianceManager:
    """
    Central manager for regulatory compliance operations.
    
    Implements:
    - ICH E6(R2): Risk-based quality management
    - ICH E6(R3): Technology-enabled monitoring
    - 21 CFR Part 11: Electronic records requirements
    - FDA AI/ML SaMD: Human oversight of AI recommendations
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        study_id: str = "",
        dqi_threshold_red: float = 0.70,
        dqi_threshold_yellow: float = 0.85
    ):
        """Initialize compliance manager"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.study_id = study_id
        self.dqi_threshold_red = dqi_threshold_red
        self.dqi_threshold_yellow = dqi_threshold_yellow
        
        # Get singleton managers
        self._audit_manager = get_audit_manager()
        self._hitl_manager = get_hitl_manager()
        
        # Site risk profiles cache
        self._site_profiles: Dict[str, SiteRiskProfile] = {}
        
        # Configuration
        self._initialized = True
        
        logger.info(f"Compliance Manager initialized for study {study_id}")
    
    # =========================================================================
    # AUDIT TRAIL INTEGRATION
    # =========================================================================
    
    def log_agent_action(
        self,
        agent: AgentIdentifier,
        action_type: ActionType,
        description: str,
        subject_id: str = "",
        site_id: str = "",
        previous_value: Optional[str] = None,
        new_value: Optional[str] = None,
        reason: str = "",
        requires_approval: bool = False,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[AuditEntry, Optional[ApprovalRequest]]:
        """
        Log an agent action with full compliance tracking.
        
        Args:
            agent: Agent performing the action
            action_type: Type of action
            description: Action description
            subject_id: Subject identifier
            site_id: Site identifier
            previous_value: Value before change
            new_value: Value after change
            reason: Reason for action
            requires_approval: Whether HITL approval is needed
            additional_data: Additional context
            
        Returns:
            Tuple of (AuditEntry, ApprovalRequest if pending)
        """
        # Determine if action requires HITL based on type
        risk_level = self._assess_action_risk(action_type)
        needs_hitl = requires_approval or requires_hitl_approval(risk_level, action_type.name)
        
        # Log to audit trail
        audit_entry = self._audit_manager.log_action(
            agent=agent,
            action_type=action_type,
            description=description,
            study_id=self.study_id,
            subject_id=subject_id,
            site_id=site_id,
            previous_value=previous_value,
            new_value=new_value,
            reason=reason,
            requires_approval=needs_hitl,
            additional_data=additional_data
        )
        
        # Create HITL request if needed
        approval_request = None
        if needs_hitl:
            request, auto_approved = self._hitl_manager.request_approval(
                agent_id=agent.value,
                agent_name=agent.name,
                action_type=action_type.name,
                action_description=description,
                proposed_action=f"{action_type.name}: {description}",
                study_id=self.study_id,
                subject_id=subject_id,
                site_id=site_id,
                current_value=previous_value,
                proposed_value=new_value,
                justification=reason,
                risk_level=risk_level,
                audit_entry_id=audit_entry.entry_id
            )
            
            if not auto_approved:
                approval_request = request
        
        return audit_entry, approval_request
    
    def _assess_action_risk(self, action_type: ActionType) -> ActionRiskLevel:
        """Assess risk level of an action"""
        critical_actions = {
            ActionType.DELETE,
            ActionType.APPROVE,
            ActionType.REJECT
        }
        
        high_risk_actions = {
            ActionType.UPDATE,
            ActionType.CREATE,
            ActionType.ESCALATE
        }
        
        medium_risk_actions = {
            ActionType.PROPOSE,
            ActionType.GENERATE
        }
        
        if action_type in critical_actions:
            return ActionRiskLevel.CRITICAL
        elif action_type in high_risk_actions:
            return ActionRiskLevel.HIGH
        elif action_type in medium_risk_actions:
            return ActionRiskLevel.MEDIUM
        else:
            return ActionRiskLevel.LOW
    
    # =========================================================================
    # RISK-BASED MONITORING
    # =========================================================================
    
    def categorize_site(
        self,
        site_id: str,
        dqi_score: float,
        component_scores: Optional[Dict[str, float]] = None
    ) -> SiteRiskProfile:
        """
        Categorize a site based on DQI score (scatter plot quadrant assignment).
        
        Per ICH E6(R2), sites are categorized for targeted monitoring:
        - GREEN: DQI >= 0.85 - Reduced SDV (25%)
        - YELLOW: 0.70 <= DQI < 0.85 - Standard SDV (50%)
        - RED: DQI < 0.70 - Intensive SDV (100%)
        
        Args:
            site_id: Site identifier
            dqi_score: Overall DQI score (0-1)
            component_scores: Individual component scores
            
        Returns:
            SiteRiskProfile for the site
        """
        # Determine category
        if dqi_score >= self.dqi_threshold_yellow:
            category = SiteRiskCategory.GREEN
            sdv_rate = 0.25
            monitoring_freq = "Quarterly"
            auto_query = True
            escalation_days = 14
        elif dqi_score >= self.dqi_threshold_red:
            category = SiteRiskCategory.YELLOW
            sdv_rate = 0.50
            monitoring_freq = "Monthly"
            auto_query = True
            escalation_days = 7
        else:
            category = SiteRiskCategory.RED
            sdv_rate = 1.00
            monitoring_freq = "Weekly"
            auto_query = False  # Require CRA review
            escalation_days = 3
        
        # Create or update profile
        profile = self._site_profiles.get(site_id, SiteRiskProfile(site_id=site_id))
        
        # Update DQI trend
        if profile.dqi_score > 0:
            profile.previous_dqi_scores.append(profile.dqi_score)
            if len(profile.previous_dqi_scores) > 6:
                profile.previous_dqi_scores.pop(0)
            
            # Calculate trend
            if dqi_score > profile.dqi_score + 0.05:
                profile.dqi_trend = "IMPROVING"
            elif dqi_score < profile.dqi_score - 0.05:
                profile.dqi_trend = "DECLINING"
            else:
                profile.dqi_trend = "STABLE"
        
        # Update scores
        profile.dqi_score = dqi_score
        profile.risk_category = category
        profile.sdv_rate = sdv_rate
        profile.monitoring_frequency = monitoring_freq
        profile.auto_query_enabled = auto_query
        profile.escalation_delay_days = escalation_days
        profile.last_updated = datetime.now(timezone.utc)
        
        # Update component scores if provided
        if component_scores:
            profile.visit_adherence_score = component_scores.get('visit_adherence', 0)
            profile.query_responsiveness_score = component_scores.get('query_responsiveness', 0)
            profile.safety_reporting_score = component_scores.get('safety_reporting', 0)
            profile.data_entry_timeliness_score = component_scores.get('data_entry_timeliness', 0)
            profile.protocol_compliance_score = component_scores.get('protocol_compliance', 0)
        
        # Flag for intervention if RED or declining
        profile.requires_cra_intervention = (
            category == SiteRiskCategory.RED or 
            profile.dqi_trend == "DECLINING"
        )
        
        # Cache profile
        self._site_profiles[site_id] = profile
        
        # Log the categorization
        self.log_agent_action(
            agent=AgentIdentifier.SYSTEM,
            action_type=ActionType.CLASSIFY,
            description=f"Site {site_id} categorized as {category.name} (DQI: {dqi_score:.2f})",
            site_id=site_id,
            additional_data={
                'dqi_score': dqi_score,
                'category': category.name,
                'sdv_rate': sdv_rate,
                'trend': profile.dqi_trend
            }
        )
        
        return profile
    
    def get_site_profile(self, site_id: str) -> Optional[SiteRiskProfile]:
        """Get risk profile for a site"""
        return self._site_profiles.get(site_id)
    
    def get_all_site_profiles(self) -> List[SiteRiskProfile]:
        """Get all site risk profiles"""
        return list(self._site_profiles.values())
    
    def get_sites_by_category(self, category: SiteRiskCategory) -> List[SiteRiskProfile]:
        """Get sites in a specific risk category"""
        return [p for p in self._site_profiles.values() if p.risk_category == category]
    
    # =========================================================================
    # COMPLIANCE REPORTING
    # =========================================================================
    
    def generate_compliance_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ComplianceReport:
        """
        Generate a regulatory compliance report for the study.
        
        This report is suitable for regulatory inspection and demonstrates
        compliance with ICH E6 R2/R3 and 21 CFR Part 11.
        
        Args:
            start_date: Report period start
            end_date: Report period end
            
        Returns:
            ComplianceReport for regulatory submission
        """
        # Get audit statistics
        audit_report = self._audit_manager.generate_compliance_report(
            study_id=self.study_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get HITL statistics
        hitl_stats = self._hitl_manager.get_statistics(study_id=self.study_id)
        
        # Calculate site distribution
        green_sites = len([p for p in self._site_profiles.values() 
                          if p.risk_category == SiteRiskCategory.GREEN])
        yellow_sites = len([p for p in self._site_profiles.values() 
                           if p.risk_category == SiteRiskCategory.YELLOW])
        red_sites = len([p for p in self._site_profiles.values() 
                        if p.risk_category == SiteRiskCategory.RED])
        
        # Calculate SDV reduction impact
        total_sites = green_sites + yellow_sites + red_sites
        if total_sites > 0:
            # Weighted average SDV rate
            avg_sdv = (green_sites * 0.25 + yellow_sites * 0.50 + red_sites * 1.00) / total_sites
            sdv_reduction = (1.0 - avg_sdv) * 100  # Reduction from 100% SDV
            
            # Estimate hours saved (assume 2 hours per subject at 100% SDV)
            hours_per_subject = 2.0
            estimated_hours_saved = (1.0 - avg_sdv) * total_sites * hours_per_subject * 10  # ~10 subjects per site
        else:
            sdv_reduction = 0.0
            estimated_hours_saved = 0.0
        
        # Build report
        report = ComplianceReport(
            study_id=self.study_id,
            total_agent_actions=audit_report['statistics']['total_entries'],
            actions_by_agent=audit_report['statistics']['actions_by_agent'],
            actions_by_type=audit_report['statistics']['actions_by_type'],
            total_hitl_requests=hitl_stats['total_requests'],
            approved_requests=hitl_stats['by_status'].get('APPROVED', 0),
            rejected_requests=hitl_stats['by_status'].get('REJECTED', 0),
            pending_requests=hitl_stats['pending_count'],
            auto_approved_requests=hitl_stats['by_status'].get('AUTO_APPROVED', 0),
            avg_approval_time_hours=hitl_stats['avg_approval_time_hours'],
            green_sites=green_sites,
            yellow_sites=yellow_sites,
            red_sites=red_sites,
            sdv_reduction_percentage=sdv_reduction,
            estimated_hours_saved=estimated_hours_saved,
            audit_chain_valid=audit_report['integrity']['chain_valid'],
            compliance_standards=[
                ComplianceStandard.ICH_E6_R2.value,
                ComplianceStandard.ICH_E6_R3.value,
                ComplianceStandard.CFR_21_PART_11.value
            ]
        )
        
        return report
    
    def export_for_inspection(self, output_path: Path) -> Dict[str, str]:
        """
        Export all compliance documentation for regulatory inspection.
        
        Args:
            output_path: Directory for export files
            
        Returns:
            Dictionary of exported file paths
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        # Export audit trail
        audit_file = self._audit_manager.export_for_inspection(
            study_id=self.study_id,
            output_path=output_path
        )
        exports['audit_trail'] = audit_file
        
        # Export compliance report
        report = self.generate_compliance_report()
        report_file = output_path / f"compliance_report_{self.study_id}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        exports['compliance_report'] = str(report_file)
        
        # Export site risk profiles
        profiles_file = output_path / f"site_risk_profiles_{self.study_id}_{datetime.now().strftime('%Y%m%d')}.json"
        profiles_data = {
            'study_id': self.study_id,
            'export_date': datetime.now(timezone.utc).isoformat(),
            'profiles': [p.to_dict() for p in self._site_profiles.values()]
        }
        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=2)
        exports['site_profiles'] = str(profiles_file)
        
        logger.info(f"Compliance documentation exported to {output_path}")
        return exports
    
    # =========================================================================
    # CPID INTEGRATION
    # =========================================================================
    
    def update_cpid_responsible_field(
        self,
        site_id: str,
        subject_id: str,
        action_description: str,
        agent: AgentIdentifier
    ) -> Dict[str, Any]:
        """
        Update the "Responsible LF for action" field in CPID_EDC_Metrics.
        
        Per the audit trail requirement, agent actions are marked with
        "System-Agent-01" (or specific agent ID) rather than human user ID.
        
        Args:
            site_id: Site identifier
            subject_id: Subject identifier
            action_description: Description of action taken
            agent: Agent that performed the action
            
        Returns:
            Update record for CPID
        """
        update_record = {
            'site_id': site_id,
            'subject_id': subject_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'responsible_lf_for_action': agent.value,
            'action_description': action_description,
            'action_type': 'SYSTEM_AGENT_ACTION',
            'audit_trail_reference': True
        }
        
        # Log the update
        self.log_agent_action(
            agent=agent,
            action_type=ActionType.UPDATE,
            description=f"Updated Responsible LF for action: {action_description}",
            subject_id=subject_id,
            site_id=site_id,
            previous_value="[Human User]",
            new_value=agent.value,
            reason="Automated agent action per protocol"
        )
        
        return update_record


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_compliance_manager(
    study_id: str = "",
    dqi_threshold_red: float = 0.70,
    dqi_threshold_yellow: float = 0.85
) -> ComplianceManager:
    """Get or create the singleton compliance manager"""
    manager = ComplianceManager(study_id, dqi_threshold_red, dqi_threshold_yellow)
    if study_id:
        manager.study_id = study_id
    return manager


def log_compliant_action(
    agent: AgentIdentifier,
    action_type: ActionType,
    description: str,
    study_id: str,
    **kwargs
) -> Tuple[AuditEntry, Optional[ApprovalRequest]]:
    """Convenience function for logging compliant actions"""
    manager = get_compliance_manager(study_id=study_id)
    return manager.log_agent_action(
        agent=agent,
        action_type=action_type,
        description=description,
        **kwargs
    )
