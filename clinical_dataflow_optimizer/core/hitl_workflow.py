"""
Human-in-the-Loop (HITL) Approval Workflow Module

Implements FDA guidance on AI/ML in Software as a Medical Device (SaMD):
- Any "Write" action to the database must be traceable
- Critical actions require human approval before execution
- All decisions are logged for regulatory audit

This module provides:
1. Approval workflow management for critical agent actions
2. Queue management for pending approvals
3. Escalation handling when approvals are delayed
4. Integration with audit trail for complete traceability
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import threading
import queue


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ApprovalStatus(Enum):
    """Status of an approval request"""
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    EXPIRED = auto()
    ESCALATED = auto()
    AUTO_APPROVED = auto()      # For non-critical actions per protocol


class ActionRiskLevel(Enum):
    """Risk level classification for actions"""
    LOW = auto()                # Read-only, informational
    MEDIUM = auto()             # Query generation, reminders
    HIGH = auto()               # Database updates, form modifications
    CRITICAL = auto()           # Safety data, regulatory submissions


class ApproverRole(Enum):
    """Roles that can approve actions"""
    DATA_MANAGER = auto()
    CLINICAL_RESEARCH_ASSOCIATE = auto()
    STUDY_COORDINATOR = auto()
    MEDICAL_MONITOR = auto()
    SAFETY_OFFICER = auto()
    SYSTEM_ADMINISTRATOR = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ApprovalRequest:
    """
    Request for human approval of an agent action.
    
    Per FDA AI/ML SaMD guidance, all write actions must have:
    - Clear description of proposed action
    - Context for decision making
    - Audit trail linkage
    - Time constraints for response
    """
    # Identifiers
    request_id: str = ""
    audit_entry_id: str = ""        # Link to audit trail
    
    # Request details
    agent_id: str = ""
    agent_name: str = ""
    action_type: str = ""
    action_description: str = ""
    
    # Context
    study_id: str = ""
    subject_id: str = ""
    site_id: str = ""
    form_id: str = ""
    
    # Proposed changes
    proposed_action: str = ""
    current_value: Optional[str] = None
    proposed_value: Optional[str] = None
    justification: str = ""
    
    # Risk assessment
    risk_level: ActionRiskLevel = ActionRiskLevel.MEDIUM
    risk_assessment: str = ""
    regulatory_reference: str = ""
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    escalation_at: Optional[datetime] = None
    
    # Status
    status: ApprovalStatus = ApprovalStatus.PENDING
    
    # Approval details
    required_approver_roles: List[ApproverRole] = field(default_factory=list)
    approved_by_id: Optional[str] = None
    approved_by_name: Optional[str] = None
    approved_by_role: Optional[ApproverRole] = None
    approval_timestamp: Optional[datetime] = None
    approval_comments: str = ""
    
    # Callback for action execution
    _action_callback: Optional[str] = None  # Stored as string for serialization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            result['expires_at'] = self.expires_at.isoformat()
        if self.escalation_at:
            result['escalation_at'] = self.escalation_at.isoformat()
        if self.approval_timestamp:
            result['approval_timestamp'] = self.approval_timestamp.isoformat()
        result['risk_level'] = self.risk_level.name
        result['status'] = self.status.name
        result['required_approver_roles'] = [r.name for r in self.required_approver_roles]
        if self.approved_by_role:
            result['approved_by_role'] = self.approved_by_role.name
        return result


@dataclass
class HITLConfig:
    """Configuration for Human-in-the-Loop workflow"""
    # Timing
    default_expiry_hours: int = 24
    escalation_hours: int = 4
    
    # Risk-based HITL requirements
    require_approval_for_high_risk: bool = True
    require_approval_for_critical_risk: bool = True
    auto_approve_low_risk: bool = True
    auto_approve_medium_risk: bool = True
    
    # Escalation
    enable_escalation: bool = True
    escalation_to_role: ApproverRole = ApproverRole.STUDY_COORDINATOR
    
    # Persistence
    storage_path: Path = field(default_factory=lambda: Path("hitl_queue"))
    
    # Actions requiring HITL by default
    hitl_required_actions: List[str] = field(default_factory=lambda: [
        "CLOSE_QUERY",
        "LOCK_FORM",
        "UNLOCK_FORM",
        "DELETE_RECORD",
        "MODIFY_SAFETY_DATA",
        "RECONCILIATION_OVERRIDE"
    ])
    
    # Role permissions for different risk levels
    role_permissions: Dict[str, List[str]] = field(default_factory=lambda: {
        'DATA_MANAGER': ['LOW', 'MEDIUM', 'HIGH'],
        'CLINICAL_RESEARCH_ASSOCIATE': ['LOW', 'MEDIUM'],
        'STUDY_COORDINATOR': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
        'MEDICAL_MONITOR': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
        'SAFETY_OFFICER': ['CRITICAL'],
        'SYSTEM_ADMINISTRATOR': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    })


# =============================================================================
# HITL MANAGER
# =============================================================================

class HITLManager:
    """
    Manages Human-in-the-Loop approval workflows.
    
    Implements FDA AI/ML SaMD guidance requirements:
    1. Human oversight of AI-generated recommendations
    2. Clear audit trail of all decisions
    3. Time-limited approval windows
    4. Escalation for delayed approvals
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[HITLConfig] = None):
        """Initialize HITL manager"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = config or HITLConfig()
        self._pending_requests: Dict[str, ApprovalRequest] = {}
        self._completed_requests: List[ApprovalRequest] = []
        self._callbacks: Dict[str, Callable] = {}
        self._request_counter = 0
        self._initialized = True
        
        # Ensure storage directory
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load pending requests
        self._load_pending_requests()
        
        logger.info(f"HITL Manager initialized. Storage: {self.config.storage_path}")
    
    def _load_pending_requests(self):
        """Load pending requests from storage"""
        try:
            pending_file = self.config.storage_path / "pending_requests.json"
            if pending_file.exists():
                with open(pending_file, 'r') as f:
                    data = json.load(f)
                    for req_dict in data.get('requests', []):
                        req = self._dict_to_request(req_dict)
                        self._pending_requests[req.request_id] = req
        except Exception as e:
            logger.warning(f"Could not load pending requests: {e}")
    
    def _dict_to_request(self, data: Dict[str, Any]) -> ApprovalRequest:
        """Convert dictionary to ApprovalRequest"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        if data.get('escalation_at'):
            data['escalation_at'] = datetime.fromisoformat(data['escalation_at'])
        if data.get('approval_timestamp'):
            data['approval_timestamp'] = datetime.fromisoformat(data['approval_timestamp'])
        data['risk_level'] = ActionRiskLevel[data['risk_level']]
        data['status'] = ApprovalStatus[data['status']]
        data['required_approver_roles'] = [ApproverRole[r] for r in data['required_approver_roles']]
        if data.get('approved_by_role'):
            data['approved_by_role'] = ApproverRole[data['approved_by_role']]
        return ApprovalRequest(**data)
    
    def _persist_pending_requests(self):
        """Save pending requests to storage"""
        try:
            pending_file = self.config.storage_path / "pending_requests.json"
            data = {
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'requests': [r.to_dict() for r in self._pending_requests.values()]
            }
            with open(pending_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist pending requests: {e}")
    
    def request_approval(
        self,
        agent_id: str,
        agent_name: str,
        action_type: str,
        action_description: str,
        proposed_action: str,
        study_id: str,
        subject_id: str = "",
        site_id: str = "",
        form_id: str = "",
        current_value: Optional[str] = None,
        proposed_value: Optional[str] = None,
        justification: str = "",
        risk_level: ActionRiskLevel = ActionRiskLevel.MEDIUM,
        risk_assessment: str = "",
        regulatory_reference: str = "",
        audit_entry_id: str = "",
        on_approved: Optional[Callable] = None,
        on_rejected: Optional[Callable] = None
    ) -> Tuple[ApprovalRequest, bool]:
        """
        Submit an action for human approval.
        
        Args:
            agent_id: ID of the agent requesting approval
            agent_name: Name of the agent
            action_type: Type of action being requested
            action_description: Human-readable description
            proposed_action: The specific action to be taken
            study_id: Study identifier
            subject_id: Subject identifier (if applicable)
            site_id: Site identifier (if applicable)
            form_id: Form identifier (if applicable)
            current_value: Current value before change
            proposed_value: Proposed new value
            justification: Reason for the proposed action
            risk_level: Risk classification
            risk_assessment: Detailed risk analysis
            regulatory_reference: Applicable regulations
            audit_entry_id: Link to audit trail
            on_approved: Callback when approved
            on_rejected: Callback when rejected
            
        Returns:
            Tuple of (ApprovalRequest, auto_approved)
        """
        with self._lock:
            self._request_counter += 1
            request_id = f"HITL-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._request_counter:04d}"
            
            # Check if auto-approval is applicable
            auto_approved = self._check_auto_approval(risk_level, action_type)
            
            # Determine required approver roles based on risk
            required_roles = self._get_required_approvers(risk_level)
            
            # Calculate timing
            now = datetime.now(timezone.utc)
            expiry = now + timedelta(hours=self.config.default_expiry_hours)
            escalation = now + timedelta(hours=self.config.escalation_hours)
            
            # Create request
            request = ApprovalRequest(
                request_id=request_id,
                audit_entry_id=audit_entry_id,
                agent_id=agent_id,
                agent_name=agent_name,
                action_type=action_type,
                action_description=action_description,
                study_id=study_id,
                subject_id=subject_id,
                site_id=site_id,
                form_id=form_id,
                proposed_action=proposed_action,
                current_value=current_value,
                proposed_value=proposed_value,
                justification=justification,
                risk_level=risk_level,
                risk_assessment=risk_assessment,
                regulatory_reference=regulatory_reference,
                expires_at=expiry,
                escalation_at=escalation if self.config.enable_escalation else None,
                required_approver_roles=required_roles,
                status=ApprovalStatus.AUTO_APPROVED if auto_approved else ApprovalStatus.PENDING
            )
            
            # Store callbacks
            if on_approved:
                self._callbacks[f"{request_id}_approved"] = on_approved
            if on_rejected:
                self._callbacks[f"{request_id}_rejected"] = on_rejected
            
            if auto_approved:
                request.approved_by_id = "SYSTEM"
                request.approved_by_name = "Auto-Approval (Low Risk)"
                request.approval_timestamp = now
                request.approval_comments = f"Auto-approved per protocol for {risk_level.name} risk actions"
                self._completed_requests.append(request)
                
                # Execute callback
                if on_approved:
                    try:
                        on_approved(request)
                    except Exception as e:
                        logger.error(f"Error executing approval callback: {e}")
                
                logger.info(f"HITL: Request {request_id} auto-approved (risk level: {risk_level.name})")
            else:
                self._pending_requests[request_id] = request
                self._persist_pending_requests()
                logger.warning(f"HITL: Request {request_id} pending approval (risk level: {risk_level.name})")
            
            return request, auto_approved
    
    def _check_auto_approval(self, risk_level: ActionRiskLevel, action_type: str) -> bool:
        """Check if action qualifies for auto-approval"""
        # Check if action type requires HITL regardless of risk
        if action_type in self.config.hitl_required_actions:
            return False
        
        # Check risk-based auto-approval
        if risk_level == ActionRiskLevel.LOW and self.config.auto_approve_low_risk:
            return True
        if risk_level == ActionRiskLevel.MEDIUM and self.config.auto_approve_medium_risk:
            return True
        
        return False
    
    def _get_required_approvers(self, risk_level: ActionRiskLevel) -> List[ApproverRole]:
        """Get required approver roles based on risk level"""
        roles = []
        risk_name = risk_level.name
        
        for role_name, allowed_risks in self.config.role_permissions.items():
            if risk_name in allowed_risks:
                roles.append(ApproverRole[role_name])
        
        return roles
    
    def approve(
        self,
        request_id: str,
        approver_id: str,
        approver_name: str,
        approver_role: ApproverRole,
        comments: str = ""
    ) -> Tuple[bool, str]:
        """
        Approve a pending request.
        
        Args:
            request_id: The request to approve
            approver_id: ID of the approver
            approver_name: Name of the approver
            approver_role: Role of the approver
            comments: Approval comments
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if request_id not in self._pending_requests:
                return False, f"Request {request_id} not found or already processed"
            
            request = self._pending_requests[request_id]
            
            # Check if approver has permission
            if approver_role not in request.required_approver_roles:
                return False, f"Role {approver_role.name} not authorized for this request"
            
            # Check if expired
            if request.expires_at and datetime.now(timezone.utc) > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                del self._pending_requests[request_id]
                self._completed_requests.append(request)
                return False, "Request has expired"
            
            # Approve
            request.status = ApprovalStatus.APPROVED
            request.approved_by_id = approver_id
            request.approved_by_name = approver_name
            request.approved_by_role = approver_role
            request.approval_timestamp = datetime.now(timezone.utc)
            request.approval_comments = comments
            
            # Move to completed
            del self._pending_requests[request_id]
            self._completed_requests.append(request)
            self._persist_pending_requests()
            
            # Execute callback
            callback_key = f"{request_id}_approved"
            if callback_key in self._callbacks:
                try:
                    self._callbacks[callback_key](request)
                except Exception as e:
                    logger.error(f"Error executing approval callback: {e}")
                del self._callbacks[callback_key]
            
            logger.info(f"HITL: Request {request_id} approved by {approver_name} ({approver_role.name})")
            return True, f"Request {request_id} approved"
    
    def reject(
        self,
        request_id: str,
        rejector_id: str,
        rejector_name: str,
        rejector_role: ApproverRole,
        reason: str
    ) -> Tuple[bool, str]:
        """
        Reject a pending request.
        
        Args:
            request_id: The request to reject
            rejector_id: ID of the rejector
            rejector_name: Name of the rejector
            rejector_role: Role of the rejector
            reason: Reason for rejection (required)
            
        Returns:
            Tuple of (success, message)
        """
        if not reason:
            return False, "Rejection reason is required per FDA guidance"
        
        with self._lock:
            if request_id not in self._pending_requests:
                return False, f"Request {request_id} not found or already processed"
            
            request = self._pending_requests[request_id]
            
            # Check if approver has permission
            if rejector_role not in request.required_approver_roles:
                return False, f"Role {rejector_role.name} not authorized for this request"
            
            # Reject
            request.status = ApprovalStatus.REJECTED
            request.approved_by_id = rejector_id
            request.approved_by_name = rejector_name
            request.approved_by_role = rejector_role
            request.approval_timestamp = datetime.now(timezone.utc)
            request.approval_comments = reason
            
            # Move to completed
            del self._pending_requests[request_id]
            self._completed_requests.append(request)
            self._persist_pending_requests()
            
            # Execute callback
            callback_key = f"{request_id}_rejected"
            if callback_key in self._callbacks:
                try:
                    self._callbacks[callback_key](request)
                except Exception as e:
                    logger.error(f"Error executing rejection callback: {e}")
                del self._callbacks[callback_key]
            
            logger.info(f"HITL: Request {request_id} rejected by {rejector_name}: {reason}")
            return True, f"Request {request_id} rejected"
    
    def check_escalations(self) -> List[ApprovalRequest]:
        """
        Check for requests that need escalation.
        
        Returns:
            List of requests requiring escalation
        """
        escalations = []
        now = datetime.now(timezone.utc)
        
        with self._lock:
            for request in self._pending_requests.values():
                if request.escalation_at and now > request.escalation_at:
                    if request.status == ApprovalStatus.PENDING:
                        request.status = ApprovalStatus.ESCALATED
                        escalations.append(request)
                        logger.warning(
                            f"HITL: Request {request.request_id} escalated "
                            f"(pending since {request.created_at.isoformat()})"
                        )
        
        return escalations
    
    def check_expirations(self) -> List[ApprovalRequest]:
        """
        Check for expired requests.
        
        Returns:
            List of expired requests
        """
        expirations = []
        now = datetime.now(timezone.utc)
        
        with self._lock:
            for request_id in list(self._pending_requests.keys()):
                request = self._pending_requests[request_id]
                if request.expires_at and now > request.expires_at:
                    request.status = ApprovalStatus.EXPIRED
                    expirations.append(request)
                    del self._pending_requests[request_id]
                    self._completed_requests.append(request)
                    logger.warning(f"HITL: Request {request_id} expired")
        
        if expirations:
            self._persist_pending_requests()
        
        return expirations
    
    def get_pending_requests(
        self,
        approver_role: Optional[ApproverRole] = None,
        study_id: Optional[str] = None
    ) -> List[ApprovalRequest]:
        """
        Get pending approval requests.
        
        Args:
            approver_role: Filter by approver role
            study_id: Filter by study
            
        Returns:
            List of pending requests
        """
        requests = list(self._pending_requests.values())
        
        if approver_role:
            requests = [r for r in requests if approver_role in r.required_approver_roles]
        if study_id:
            requests = [r for r in requests if r.study_id == study_id]
        
        return requests
    
    def get_request_by_id(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a specific request by ID"""
        if request_id in self._pending_requests:
            return self._pending_requests[request_id]
        
        for req in self._completed_requests:
            if req.request_id == request_id:
                return req
        
        return None
    
    def get_statistics(self, study_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get HITL workflow statistics.
        
        Args:
            study_id: Filter by study
            
        Returns:
            Statistics dictionary
        """
        all_requests = list(self._pending_requests.values()) + self._completed_requests
        
        if study_id:
            all_requests = [r for r in all_requests if r.study_id == study_id]
        
        status_counts = {}
        risk_counts = {}
        agent_counts = {}
        avg_approval_time = []
        
        for req in all_requests:
            status_counts[req.status.name] = status_counts.get(req.status.name, 0) + 1
            risk_counts[req.risk_level.name] = risk_counts.get(req.risk_level.name, 0) + 1
            agent_counts[req.agent_name] = agent_counts.get(req.agent_name, 0) + 1
            
            if req.status == ApprovalStatus.APPROVED and req.approval_timestamp:
                delta = (req.approval_timestamp - req.created_at).total_seconds() / 3600
                avg_approval_time.append(delta)
        
        return {
            'total_requests': len(all_requests),
            'pending_count': len(self._pending_requests),
            'completed_count': len(self._completed_requests),
            'by_status': status_counts,
            'by_risk_level': risk_counts,
            'by_agent': agent_counts,
            'avg_approval_time_hours': sum(avg_approval_time) / len(avg_approval_time) if avg_approval_time else 0,
            'compliance_note': (
                "All approval actions are logged per FDA AI/ML SaMD guidance "
                "and 21 CFR Part 11 requirements."
            )
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_hitl_manager(config: Optional[HITLConfig] = None) -> HITLManager:
    """Get or create the singleton HITL manager"""
    return HITLManager(config)


def requires_hitl_approval(risk_level: ActionRiskLevel, action_type: str) -> bool:
    """Check if an action requires HITL approval"""
    manager = get_hitl_manager()
    return not manager._check_auto_approval(risk_level, action_type)
