"""
Audit Trail Module for ICH E6 R2/R3 and 21 CFR Part 11 Compliance

This module implements comprehensive audit logging for all Agentic AI actions,
ensuring regulatory compliance with:
- ICH E6(R2) Section 5.5.3: Documentation and Essential Documents
- ICH E6(R3): Risk-based quality management and computerized systems
- 21 CFR Part 11: Electronic Records; Electronic Signatures

Key Features:
1. Complete traceability of all agent actions
2. Tamper-evident logging with cryptographic hashing
3. Distinction between human and system-agent actions
4. Time-stamped entries with UTC timezone
5. Action categorization (Read/Write/Approve/Reject)
"""

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ActionType(Enum):
    """Types of actions that can be audited"""
    # Read actions (non-critical)
    READ = auto()
    QUERY = auto()
    ANALYZE = auto()
    
    # Write actions (require HITL for critical)
    CREATE = auto()
    UPDATE = auto()
    DELETE = auto()
    
    # Workflow actions
    PROPOSE = auto()
    APPROVE = auto()
    REJECT = auto()
    ESCALATE = auto()
    
    # System actions
    DETECT = auto()
    CLASSIFY = auto()
    GENERATE = auto()
    LEARN = auto()


class ActionCategory(Enum):
    """Category of action for regulatory classification"""
    NON_CRITICAL = auto()       # Read-only, no database modification
    CRITICAL_AUTO = auto()       # Write action, auto-executable per protocol
    CRITICAL_HITL = auto()       # Write action, requires Human-in-the-Loop approval
    SYSTEM_INTERNAL = auto()     # Internal system operation


class AgentIdentifier(Enum):
    """Standardized agent identifiers for audit trail"""
    REX = "System-Agent-Rex-01"       # Reconciliation Agent
    CODEX = "System-Agent-Codex-01"   # Coding Agent
    LIA = "System-Agent-Lia-01"       # Site Liaison Agent
    SUPERVISOR = "System-Agent-Supervisor-01"  # Supervisor Agent
    SYSTEM = "System-Agent-Core-01"   # Core system operations


class ComplianceStandard(Enum):
    """Regulatory compliance standards"""
    ICH_E6_R2 = "ICH E6(R2)"
    ICH_E6_R3 = "ICH E6(R3) Draft"
    CFR_21_PART_11 = "21 CFR Part 11"
    GDPR = "GDPR Article 30"
    HIPAA = "HIPAA Security Rule"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AuditEntry:
    """
    Single audit trail entry conforming to 21 CFR Part 11 requirements.
    
    Required fields per 21 CFR 11.10(e):
    - Identity of individual who performed action
    - Date and time of action
    - Description of action
    - Reason for action (if applicable)
    
    Additional fields for clinical trial compliance:
    - Study/Protocol identifier
    - Subject identifier (if applicable)
    - Site identifier (if applicable)
    - Previous and new values for modifications
    """
    # Unique identifiers
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_number: int = 0
    
    # Timestamp (UTC per 21 CFR Part 11)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Actor identification
    actor_id: str = ""                    # Agent ID or Human User ID
    actor_type: str = "SYSTEM_AGENT"      # SYSTEM_AGENT or HUMAN_USER
    actor_name: str = ""                  # Display name
    
    # Action details
    action_type: ActionType = ActionType.READ
    action_category: ActionCategory = ActionCategory.NON_CRITICAL
    action_description: str = ""
    action_reason: str = ""
    
    # Context
    study_id: str = ""
    protocol_id: str = ""
    subject_id: str = ""
    site_id: str = ""
    form_id: str = ""
    field_id: str = ""
    
    # Data changes (for modifications)
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    
    # Compliance references
    compliance_standards: List[str] = field(default_factory=list)
    regulatory_reference: str = ""
    
    # Verification
    requires_approval: bool = False
    approval_status: str = "N/A"          # PENDING, APPROVED, REJECTED, N/A
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    approval_reason: Optional[str] = None
    
    # Integrity
    previous_hash: str = ""
    entry_hash: str = ""
    
    # Metadata
    module_name: str = ""
    function_name: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['action_type'] = self.action_type.name
        result['action_category'] = self.action_category.name
        if self.approval_timestamp:
            result['approval_timestamp'] = self.approval_timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEntry':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['action_type'] = ActionType[data['action_type']]
        data['action_category'] = ActionCategory[data['action_category']]
        if data.get('approval_timestamp'):
            data['approval_timestamp'] = datetime.fromisoformat(data['approval_timestamp'])
        return cls(**data)


@dataclass
class AuditTrailConfig:
    """Configuration for audit trail system"""
    # Storage
    storage_path: Path = field(default_factory=lambda: Path("audit_logs"))
    file_prefix: str = "audit_trail"
    max_entries_per_file: int = 10000
    
    # Retention (per 21 CFR Part 11 and ICH E6)
    retention_years: int = 15           # Clinical trial data retention
    archive_after_days: int = 90
    
    # Integrity
    enable_hash_chain: bool = True      # Cryptographic chain for tamper detection
    hash_algorithm: str = "sha256"
    
    # Real-time logging
    enable_real_time_log: bool = True
    log_level: str = "INFO"
    
    # Compliance settings
    require_reason_for_changes: bool = True
    require_electronic_signature: bool = False  # Can be enabled for high-risk
    
    # HITL settings
    hitl_actions: List[str] = field(default_factory=lambda: [
        "DELETE", "UPDATE", "APPROVE", "REJECT"
    ])


# =============================================================================
# AUDIT TRAIL MANAGER
# =============================================================================

class AuditTrailManager:
    """
    Central manager for audit trail operations.
    
    Implements:
    - 21 CFR Part 11.10(e): Use of secure, computer-generated, time-stamped audit trails
    - 21 CFR Part 11.10(k): Use of appropriate controls over system documentation
    - ICH E6(R2) 5.5.3: Audit trail for electronic trial data
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for consistent audit management"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[AuditTrailConfig] = None):
        """Initialize audit trail manager"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = config or AuditTrailConfig()
        self._entries: List[AuditEntry] = []
        self._sequence_counter = 0
        self._last_hash = "GENESIS"
        self._pending_approvals: Dict[str, AuditEntry] = {}
        self._initialized = True
        
        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing entries if any
        self._load_existing_entries()
        
        logger.info(f"Audit Trail Manager initialized. Storage: {self.config.storage_path}")
    
    def _load_existing_entries(self):
        """Load existing audit entries for hash chain continuity"""
        try:
            files = sorted(self.config.storage_path.glob(f"{self.config.file_prefix}_*.json"))
            if files:
                with open(files[-1], 'r') as f:
                    data = json.load(f)
                    if data.get('entries'):
                        last_entry = data['entries'][-1]
                        self._sequence_counter = last_entry.get('sequence_number', 0)
                        self._last_hash = last_entry.get('entry_hash', 'GENESIS')
        except Exception as e:
            logger.warning(f"Could not load existing audit entries: {e}")
    
    def _calculate_hash(self, entry: AuditEntry) -> str:
        """Calculate cryptographic hash for entry (tamper-evident)"""
        hash_input = json.dumps({
            'sequence': entry.sequence_number,
            'timestamp': entry.timestamp.isoformat(),
            'actor_id': entry.actor_id,
            'action_type': entry.action_type.name,
            'action_description': entry.action_description,
            'previous_value': entry.previous_value,
            'new_value': entry.new_value,
            'previous_hash': entry.previous_hash
        }, sort_keys=True)
        
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def log_action(
        self,
        agent: AgentIdentifier,
        action_type: ActionType,
        description: str,
        study_id: str = "",
        subject_id: str = "",
        site_id: str = "",
        previous_value: Optional[str] = None,
        new_value: Optional[str] = None,
        reason: str = "",
        form_id: str = "",
        field_id: str = "",
        requires_approval: bool = False,
        compliance_refs: Optional[List[ComplianceStandard]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Log an action to the audit trail.
        
        Args:
            agent: The agent performing the action
            action_type: Type of action being performed
            description: Human-readable description
            study_id: Study/Protocol identifier
            subject_id: Subject/Patient identifier
            site_id: Site identifier
            previous_value: Value before modification
            new_value: Value after modification
            reason: Reason for the action
            form_id: Form/CRF identifier
            field_id: Field identifier
            requires_approval: Whether HITL approval is required
            compliance_refs: Applicable compliance standards
            additional_data: Any additional context data
            
        Returns:
            AuditEntry: The created audit entry
        """
        with self._lock:
            self._sequence_counter += 1
            
            # Determine action category
            action_category = self._categorize_action(action_type, requires_approval)
            
            # Build compliance references
            standards = []
            if compliance_refs:
                standards = [std.value for std in compliance_refs]
            else:
                standards = self._default_compliance_refs(action_type)
            
            # Create entry
            entry = AuditEntry(
                sequence_number=self._sequence_counter,
                actor_id=agent.value,
                actor_type="SYSTEM_AGENT",
                actor_name=agent.name,
                action_type=action_type,
                action_category=action_category,
                action_description=description,
                action_reason=reason,
                study_id=study_id,
                subject_id=subject_id,
                site_id=site_id,
                form_id=form_id,
                field_id=field_id,
                previous_value=previous_value,
                new_value=new_value,
                compliance_standards=standards,
                requires_approval=requires_approval,
                approval_status="PENDING" if requires_approval else "N/A",
                previous_hash=self._last_hash,
                additional_data=additional_data or {}
            )
            
            # Calculate hash
            if self.config.enable_hash_chain:
                entry.entry_hash = self._calculate_hash(entry)
                self._last_hash = entry.entry_hash
            
            # Store entry
            self._entries.append(entry)
            
            # Handle HITL if required
            if requires_approval:
                self._pending_approvals[entry.entry_id] = entry
            
            # Persist
            self._persist_entry(entry)
            
            # Log
            if self.config.enable_real_time_log:
                self._log_entry(entry)
            
            return entry
    
    def _categorize_action(
        self, 
        action_type: ActionType, 
        requires_approval: bool
    ) -> ActionCategory:
        """Categorize action for regulatory purposes"""
        read_actions = {ActionType.READ, ActionType.QUERY, ActionType.ANALYZE}
        write_actions = {ActionType.CREATE, ActionType.UPDATE, ActionType.DELETE}
        
        if action_type in read_actions:
            return ActionCategory.NON_CRITICAL
        elif action_type in write_actions:
            if requires_approval:
                return ActionCategory.CRITICAL_HITL
            else:
                return ActionCategory.CRITICAL_AUTO
        else:
            return ActionCategory.SYSTEM_INTERNAL
    
    def _default_compliance_refs(self, action_type: ActionType) -> List[str]:
        """Get default compliance references based on action type"""
        refs = [ComplianceStandard.ICH_E6_R2.value]
        
        if action_type in {ActionType.UPDATE, ActionType.DELETE, ActionType.CREATE}:
            refs.append(ComplianceStandard.CFR_21_PART_11.value)
        
        return refs
    
    def _persist_entry(self, entry: AuditEntry):
        """Persist entry to storage"""
        try:
            date_str = entry.timestamp.strftime('%Y%m%d')
            filename = f"{self.config.file_prefix}_{date_str}.json"
            filepath = self.config.storage_path / filename
            
            # Load existing or create new
            data = {'entries': [], 'metadata': {}}
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
            
            # Add entry
            data['entries'].append(entry.to_dict())
            data['metadata'] = {
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'entry_count': len(data['entries']),
                'compliance_standards': [
                    ComplianceStandard.ICH_E6_R2.value,
                    ComplianceStandard.CFR_21_PART_11.value
                ]
            }
            
            # Write atomically
            temp_path = filepath.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(filepath)
            
        except Exception as e:
            logger.error(f"Failed to persist audit entry: {e}")
    
    def _log_entry(self, entry: AuditEntry):
        """Log entry to application logger"""
        log_msg = (
            f"AUDIT: [{entry.actor_id}] {entry.action_type.name} - "
            f"{entry.action_description}"
        )
        if entry.study_id:
            log_msg += f" | Study: {entry.study_id}"
        if entry.subject_id:
            log_msg += f" | Subject: {entry.subject_id}"
        if entry.site_id:
            log_msg += f" | Site: {entry.site_id}"
        
        if entry.action_category == ActionCategory.CRITICAL_HITL:
            logger.warning(log_msg + " [REQUIRES APPROVAL]")
        else:
            logger.info(log_msg)
    
    def approve_action(
        self,
        entry_id: str,
        approver_id: str,
        approver_name: str,
        reason: str = ""
    ) -> Tuple[bool, str]:
        """
        Approve a pending HITL action.
        
        Args:
            entry_id: The audit entry ID to approve
            approver_id: ID of the human approver
            approver_name: Name of the approver
            reason: Reason for approval
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if entry_id not in self._pending_approvals:
                return False, f"Entry {entry_id} not found in pending approvals"
            
            entry = self._pending_approvals[entry_id]
            entry.approval_status = "APPROVED"
            entry.approved_by = f"{approver_id} ({approver_name})"
            entry.approval_timestamp = datetime.now(timezone.utc)
            entry.approval_reason = reason
            
            # Log the approval
            self.log_action(
                agent=AgentIdentifier.SYSTEM,
                action_type=ActionType.APPROVE,
                description=f"Action approved by {approver_name}",
                study_id=entry.study_id,
                subject_id=entry.subject_id,
                site_id=entry.site_id,
                reason=reason,
                additional_data={'approved_entry_id': entry_id}
            )
            
            del self._pending_approvals[entry_id]
            return True, f"Entry {entry_id} approved"
    
    def reject_action(
        self,
        entry_id: str,
        rejector_id: str,
        rejector_name: str,
        reason: str
    ) -> Tuple[bool, str]:
        """
        Reject a pending HITL action.
        
        Args:
            entry_id: The audit entry ID to reject
            rejector_id: ID of the human rejector
            rejector_name: Name of the rejector
            reason: Reason for rejection (required)
            
        Returns:
            Tuple of (success, message)
        """
        if not reason:
            return False, "Reason is required for rejection per 21 CFR Part 11"
        
        with self._lock:
            if entry_id not in self._pending_approvals:
                return False, f"Entry {entry_id} not found in pending approvals"
            
            entry = self._pending_approvals[entry_id]
            entry.approval_status = "REJECTED"
            entry.approved_by = f"{rejector_id} ({rejector_name})"
            entry.approval_timestamp = datetime.now(timezone.utc)
            entry.approval_reason = reason
            
            # Log the rejection
            self.log_action(
                agent=AgentIdentifier.SYSTEM,
                action_type=ActionType.REJECT,
                description=f"Action rejected by {rejector_name}: {reason}",
                study_id=entry.study_id,
                subject_id=entry.subject_id,
                site_id=entry.site_id,
                reason=reason,
                additional_data={'rejected_entry_id': entry_id}
            )
            
            del self._pending_approvals[entry_id]
            return True, f"Entry {entry_id} rejected"
    
    def get_pending_approvals(self) -> List[AuditEntry]:
        """Get all entries pending HITL approval"""
        return list(self._pending_approvals.values())
    
    def get_entries(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        action_type: Optional[ActionType] = None,
        study_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        site_id: Optional[str] = None
    ) -> List[AuditEntry]:
        """
        Query audit entries with filters.
        
        Args:
            start_date: Filter entries after this date
            end_date: Filter entries before this date
            agent_id: Filter by agent identifier
            action_type: Filter by action type
            study_id: Filter by study
            subject_id: Filter by subject
            site_id: Filter by site
            
        Returns:
            List of matching audit entries
        """
        results = self._entries.copy()
        
        if start_date:
            results = [e for e in results if e.timestamp >= start_date]
        if end_date:
            results = [e for e in results if e.timestamp <= end_date]
        if agent_id:
            results = [e for e in results if e.actor_id == agent_id]
        if action_type:
            results = [e for e in results if e.action_type == action_type]
        if study_id:
            results = [e for e in results if e.study_id == study_id]
        if subject_id:
            results = [e for e in results if e.subject_id == subject_id]
        if site_id:
            results = [e for e in results if e.site_id == site_id]
        
        return results
    
    def verify_chain_integrity(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify the integrity of the audit trail hash chain.
        
        Per 21 CFR Part 11.10(e): Audit trails must be tamper-evident.
        
        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []
        previous_hash = "GENESIS"
        
        for entry in self._entries:
            # Check hash chain
            if entry.previous_hash != previous_hash:
                violations.append({
                    'entry_id': entry.entry_id,
                    'sequence': entry.sequence_number,
                    'issue': 'Previous hash mismatch',
                    'expected': previous_hash,
                    'found': entry.previous_hash
                })
            
            # Verify entry hash
            if self.config.enable_hash_chain:
                calculated_hash = self._calculate_hash(entry)
                if entry.entry_hash != calculated_hash:
                    violations.append({
                        'entry_id': entry.entry_id,
                        'sequence': entry.sequence_number,
                        'issue': 'Entry hash mismatch - possible tampering',
                        'expected': calculated_hash,
                        'found': entry.entry_hash
                    })
            
            previous_hash = entry.entry_hash
        
        return len(violations) == 0, violations
    
    def generate_compliance_report(
        self,
        study_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a compliance report for regulatory submission.
        
        Args:
            study_id: Study to report on
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report dictionary
        """
        entries = self.get_entries(
            start_date=start_date,
            end_date=end_date,
            study_id=study_id
        )
        
        # Calculate statistics
        action_counts = {}
        agent_counts = {}
        approval_counts = {'APPROVED': 0, 'REJECTED': 0, 'PENDING': 0, 'N/A': 0}
        
        for entry in entries:
            action_counts[entry.action_type.name] = \
                action_counts.get(entry.action_type.name, 0) + 1
            agent_counts[entry.actor_id] = \
                agent_counts.get(entry.actor_id, 0) + 1
            approval_counts[entry.approval_status] = \
                approval_counts.get(entry.approval_status, 0) + 1
        
        # Verify integrity
        is_valid, violations = self.verify_chain_integrity()
        
        return {
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'study_id': study_id,
            'date_range': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            },
            'statistics': {
                'total_entries': len(entries),
                'actions_by_type': action_counts,
                'actions_by_agent': agent_counts,
                'approval_status': approval_counts
            },
            'integrity': {
                'chain_valid': is_valid,
                'violations_count': len(violations),
                'violations': violations[:10]  # Limit for report
            },
            'compliance_standards': [
                ComplianceStandard.ICH_E6_R2.value,
                ComplianceStandard.ICH_E6_R3.value,
                ComplianceStandard.CFR_21_PART_11.value
            ],
            'certification': (
                "This audit trail report is generated in compliance with "
                "21 CFR Part 11.10(e) and ICH E6(R2) Section 5.5.3. "
                "All entries are time-stamped, attributed, and tamper-evident."
            )
        }
    
    def export_for_inspection(
        self,
        study_id: str,
        output_path: Path
    ) -> str:
        """
        Export audit trail for regulatory inspection.
        
        Args:
            study_id: Study to export
            output_path: Path for export file
            
        Returns:
            Path to exported file
        """
        entries = self.get_entries(study_id=study_id)
        
        export_data = {
            'export_metadata': {
                'generated': datetime.now(timezone.utc).isoformat(),
                'study_id': study_id,
                'entry_count': len(entries),
                'compliance': [
                    ComplianceStandard.ICH_E6_R2.value,
                    ComplianceStandard.CFR_21_PART_11.value
                ]
            },
            'entries': [e.to_dict() for e in entries]
        }
        
        output_file = output_path / f"audit_trail_export_{study_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Audit trail exported to {output_file}")
        return str(output_file)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_audit_manager(config: Optional[AuditTrailConfig] = None) -> AuditTrailManager:
    """Get or create the singleton audit trail manager"""
    return AuditTrailManager(config)


def log_agent_action(
    agent: AgentIdentifier,
    action_type: ActionType,
    description: str,
    **kwargs
) -> AuditEntry:
    """Convenience function to log an agent action"""
    manager = get_audit_manager()
    return manager.log_action(
        agent=agent,
        action_type=action_type,
        description=description,
        **kwargs
    )
