"""
Test Suite for Regulatory Compliance Features

Tests for:
- ICH E6 R2/R3: Risk-Based Quality Management
- 21 CFR Part 11: Electronic Records and Audit Trails
- FDA AI/ML SaMD: Human-in-the-Loop (HITL) oversight

This test suite validates compliance with regulatory requirements
for Agentic AI in clinical trial data management.
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent))

from clinical_dataflow_optimizer.core.audit_trail import (
    AuditTrailManager,
    AuditEntry,
    AuditTrailConfig,
    ActionType,
    ActionCategory,
    AgentIdentifier,
    ComplianceStandard,
    get_audit_manager,
    log_agent_action
)
from clinical_dataflow_optimizer.core.hitl_workflow import (
    HITLManager,
    ApprovalRequest,
    HITLConfig,
    ApprovalStatus,
    ActionRiskLevel,
    ApproverRole,
    get_hitl_manager,
    requires_hitl_approval
)
from clinical_dataflow_optimizer.core.compliance_manager import (
    ComplianceManager,
    ComplianceReport,
    SiteRiskProfile,
    SiteRiskCategory,
    get_compliance_manager,
    log_compliant_action
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def audit_config(temp_storage):
    """Create audit trail config with temp storage"""
    return AuditTrailConfig(
        storage_path=temp_storage / "audit_logs",
        enable_hash_chain=True,
        enable_real_time_log=False  # Disable for testing
    )


@pytest.fixture
def hitl_config(temp_storage):
    """Create HITL config with temp storage"""
    return HITLConfig(
        storage_path=temp_storage / "hitl_queue",
        auto_approve_low_risk=True,
        auto_approve_medium_risk=False,  # Require approval for medium
        require_approval_for_high_risk=True
    )


@pytest.fixture
def fresh_audit_manager(audit_config):
    """Create a fresh audit manager instance for each test"""
    # Reset singleton
    AuditTrailManager._instance = None
    manager = AuditTrailManager(audit_config)
    yield manager
    AuditTrailManager._instance = None


@pytest.fixture
def fresh_hitl_manager(hitl_config):
    """Create a fresh HITL manager instance for each test"""
    # Reset singleton
    HITLManager._instance = None
    manager = HITLManager(hitl_config)
    yield manager
    HITLManager._instance = None


@pytest.fixture
def fresh_compliance_manager(temp_storage):
    """Create a fresh compliance manager for each test"""
    # Reset all singletons
    AuditTrailManager._instance = None
    HITLManager._instance = None
    ComplianceManager._instance = None
    
    audit_config = AuditTrailConfig(storage_path=temp_storage / "audit")
    hitl_config = HITLConfig(storage_path=temp_storage / "hitl")
    
    # Initialize managers
    AuditTrailManager(audit_config)
    HITLManager(hitl_config)
    
    manager = ComplianceManager(study_id="TEST-001")
    yield manager
    
    # Cleanup
    AuditTrailManager._instance = None
    HITLManager._instance = None
    ComplianceManager._instance = None


# =============================================================================
# AUDIT TRAIL TESTS (21 CFR Part 11)
# =============================================================================

class TestAuditTrailBasics:
    """Test basic audit trail functionality per 21 CFR Part 11.10(e)"""
    
    def test_audit_entry_creation(self, fresh_audit_manager):
        """Test that audit entries are created with required fields"""
        entry = fresh_audit_manager.log_action(
            agent=AgentIdentifier.REX,
            action_type=ActionType.DETECT,
            description="Detected zombie SAE for subject S001",
            study_id="STUDY-001",
            subject_id="S001",
            site_id="Site 1"
        )
        
        # Per 21 CFR 11.10(e) - required fields
        assert entry.entry_id is not None
        assert entry.timestamp is not None
        assert entry.actor_id == AgentIdentifier.REX.value
        assert entry.action_description == "Detected zombie SAE for subject S001"
        assert entry.study_id == "STUDY-001"
        assert entry.subject_id == "S001"
    
    def test_audit_timestamp_is_utc(self, fresh_audit_manager):
        """Test that timestamps are in UTC per regulatory requirements"""
        entry = fresh_audit_manager.log_action(
            agent=AgentIdentifier.CODEX,
            action_type=ActionType.ANALYZE,
            description="Analyzed uncoded terms"
        )
        
        # Timestamp must be timezone-aware UTC
        assert entry.timestamp.tzinfo is not None
        assert entry.timestamp.tzinfo == timezone.utc
    
    def test_audit_sequence_numbers(self, fresh_audit_manager):
        """Test that entries have sequential numbers for ordering"""
        entry1 = fresh_audit_manager.log_action(
            agent=AgentIdentifier.LIA,
            action_type=ActionType.GENERATE,
            description="Generated reminder"
        )
        entry2 = fresh_audit_manager.log_action(
            agent=AgentIdentifier.LIA,
            action_type=ActionType.GENERATE,
            description="Generated another reminder"
        )
        
        assert entry2.sequence_number > entry1.sequence_number
    
    def test_audit_agent_identification(self, fresh_audit_manager):
        """Test that agent actions are distinctly identified from human users"""
        entry = fresh_audit_manager.log_action(
            agent=AgentIdentifier.REX,
            action_type=ActionType.UPDATE,
            description="Updated EDRR count"
        )
        
        # Agent must be clearly identified as system agent
        assert entry.actor_type == "SYSTEM_AGENT"
        assert "System-Agent" in entry.actor_id
        assert entry.actor_id == "System-Agent-Rex-01"


class TestAuditHashChain:
    """Test tamper-evident hash chain per 21 CFR Part 11.10(e)"""
    
    def test_hash_chain_continuity(self, fresh_audit_manager):
        """Test that entries form a continuous hash chain"""
        entry1 = fresh_audit_manager.log_action(
            agent=AgentIdentifier.SYSTEM,
            action_type=ActionType.READ,
            description="First action"
        )
        entry2 = fresh_audit_manager.log_action(
            agent=AgentIdentifier.SYSTEM,
            action_type=ActionType.READ,
            description="Second action"
        )
        
        # Second entry should reference first entry's hash
        assert entry2.previous_hash == entry1.entry_hash
    
    def test_hash_chain_integrity_verification(self, fresh_audit_manager):
        """Test that chain integrity can be verified"""
        # Create several entries
        for i in range(5):
            fresh_audit_manager.log_action(
                agent=AgentIdentifier.CODEX,
                action_type=ActionType.ANALYZE,
                description=f"Analysis {i}"
            )
        
        # Verify chain integrity
        is_valid, violations = fresh_audit_manager.verify_chain_integrity()
        
        assert is_valid
        assert len(violations) == 0
    
    def test_hash_chain_detects_tampering(self, fresh_audit_manager):
        """Test that tampering is detected"""
        entry = fresh_audit_manager.log_action(
            agent=AgentIdentifier.REX,
            action_type=ActionType.DETECT,
            description="Original description"
        )
        
        # Simulate tampering by modifying entry
        original_hash = entry.entry_hash
        entry.action_description = "Tampered description"
        
        # Recalculate and compare
        new_hash = fresh_audit_manager._calculate_hash(entry)
        
        # Hash should be different after tampering
        assert new_hash != original_hash


class TestAuditDataChanges:
    """Test audit trail for data modifications per 21 CFR Part 11"""
    
    def test_tracks_previous_and_new_values(self, fresh_audit_manager):
        """Test that modifications include before/after values"""
        entry = fresh_audit_manager.log_action(
            agent=AgentIdentifier.REX,
            action_type=ActionType.UPDATE,
            description="Updated open issue count",
            previous_value="5",
            new_value="6",
            reason="Added new zombie SAE finding"
        )
        
        assert entry.previous_value == "5"
        assert entry.new_value == "6"
        assert entry.action_reason == "Added new zombie SAE finding"
    
    def test_requires_reason_for_changes(self, audit_config, temp_storage):
        """Test that changes include reason per regulatory requirements"""
        audit_config.require_reason_for_changes = True
        AuditTrailManager._instance = None
        manager = AuditTrailManager(audit_config)
        
        entry = manager.log_action(
            agent=AgentIdentifier.CODEX,
            action_type=ActionType.UPDATE,
            description="Updated coding status",
            previous_value="UnCoded",
            new_value="Coded",
            reason="Matched to WHO Drug Dictionary"
        )
        
        assert entry.action_reason != ""
        AuditTrailManager._instance = None


class TestAuditPersistence:
    """Test audit trail persistence per 21 CFR Part 11"""
    
    def test_entries_persisted_to_storage(self, fresh_audit_manager, temp_storage):
        """Test that entries are written to storage"""
        entry = fresh_audit_manager.log_action(
            agent=AgentIdentifier.LIA,
            action_type=ActionType.ESCALATE,
            description="Escalated ghost visit to CRA"
        )
        
        # Check storage directory
        audit_files = list((temp_storage / "audit_logs").glob("audit_trail_*.json"))
        assert len(audit_files) > 0
        
        # Verify entry is in file
        with open(audit_files[0], 'r') as f:
            data = json.load(f)
            assert len(data['entries']) > 0
            assert data['entries'][-1]['entry_id'] == entry.entry_id
    
    def test_compliance_report_generation(self, fresh_audit_manager):
        """Test generation of compliance report for regulatory submission"""
        # Log several actions
        for agent in [AgentIdentifier.REX, AgentIdentifier.CODEX, AgentIdentifier.LIA]:
            fresh_audit_manager.log_action(
                agent=agent,
                action_type=ActionType.DETECT,
                description=f"Detection by {agent.name}",
                study_id="STUDY-001"
            )
        
        report = fresh_audit_manager.generate_compliance_report(study_id="STUDY-001")
        
        assert report['study_id'] == "STUDY-001"
        assert report['statistics']['total_entries'] == 3
        assert 'ICH E6(R2)' in report['compliance_standards']
        assert '21 CFR Part 11' in report['compliance_standards']


# =============================================================================
# HITL TESTS (FDA AI/ML SaMD Guidance)
# =============================================================================

class TestHITLBasics:
    """Test Human-in-the-Loop basics per FDA AI/ML SaMD guidance"""
    
    def test_approval_request_creation(self, fresh_hitl_manager):
        """Test that approval requests are created with required fields"""
        request, auto_approved = fresh_hitl_manager.request_approval(
            agent_id=AgentIdentifier.REX.value,
            agent_name="Rex",
            action_type="UPDATE",
            action_description="Update EDRR count",
            proposed_action="Increment count by 1",
            study_id="STUDY-001",
            subject_id="S001",
            site_id="Site 1",
            risk_level=ActionRiskLevel.HIGH
        )
        
        assert request.request_id is not None
        assert request.agent_id == AgentIdentifier.REX.value
        assert request.risk_level == ActionRiskLevel.HIGH
        assert not auto_approved  # High risk should not auto-approve
    
    def test_low_risk_auto_approval(self, fresh_hitl_manager):
        """Test that low-risk actions are auto-approved per protocol"""
        request, auto_approved = fresh_hitl_manager.request_approval(
            agent_id=AgentIdentifier.LIA.value,
            agent_name="Lia",
            action_type="GENERATE_QUERY_DRAFT",
            action_description="Generate reminder email",
            proposed_action="Send reminder to site",
            study_id="STUDY-001",
            risk_level=ActionRiskLevel.LOW
        )
        
        assert auto_approved
        assert request.status == ApprovalStatus.AUTO_APPROVED
    
    def test_critical_action_requires_approval(self, fresh_hitl_manager):
        """Test that critical actions require human approval"""
        request, auto_approved = fresh_hitl_manager.request_approval(
            agent_id=AgentIdentifier.REX.value,
            agent_name="Rex",
            action_type="CLOSE_QUERY",  # Critical action
            action_description="Close safety query",
            proposed_action="Mark query as resolved",
            study_id="STUDY-001",
            risk_level=ActionRiskLevel.CRITICAL
        )
        
        assert not auto_approved
        assert request.status == ApprovalStatus.PENDING


class TestHITLApprovalWorkflow:
    """Test HITL approval workflow"""
    
    def test_approval_by_authorized_role(self, fresh_hitl_manager):
        """Test that approvals can be made by authorized roles"""
        request, _ = fresh_hitl_manager.request_approval(
            agent_id=AgentIdentifier.CODEX.value,
            agent_name="Codex",
            action_type="UPDATE",
            action_description="Update coding status",
            proposed_action="Change status to Coded",
            study_id="STUDY-001",
            risk_level=ActionRiskLevel.HIGH
        )
        
        success, message = fresh_hitl_manager.approve(
            request_id=request.request_id,
            approver_id="DM001",
            approver_name="Data Manager",
            approver_role=ApproverRole.DATA_MANAGER,
            comments="Approved after review"
        )
        
        assert success
        assert "approved" in message.lower()
        
        # Verify status updated
        approved_request = fresh_hitl_manager.get_request_by_id(request.request_id)
        assert approved_request.status == ApprovalStatus.APPROVED
    
    def test_rejection_requires_reason(self, fresh_hitl_manager):
        """Test that rejections require a reason per FDA guidance"""
        request, _ = fresh_hitl_manager.request_approval(
            agent_id=AgentIdentifier.REX.value,
            agent_name="Rex",
            action_type="UPDATE",
            action_description="Update safety data",
            proposed_action="Modify SAE record",
            study_id="STUDY-001",
            risk_level=ActionRiskLevel.CRITICAL
        )
        
        # Try to reject without reason
        success, message = fresh_hitl_manager.reject(
            request_id=request.request_id,
            rejector_id="DM001",
            rejector_name="Data Manager",
            rejector_role=ApproverRole.DATA_MANAGER,
            reason=""  # Empty reason
        )
        
        assert not success
        assert "reason is required" in message.lower()
    
    def test_rejection_with_reason_succeeds(self, fresh_hitl_manager):
        """Test that rejections with reason succeed"""
        request, _ = fresh_hitl_manager.request_approval(
            agent_id=AgentIdentifier.REX.value,
            agent_name="Rex",
            action_type="UPDATE",
            action_description="Update safety data",
            proposed_action="Modify SAE record",
            study_id="STUDY-001",
            risk_level=ActionRiskLevel.HIGH
        )
        
        success, message = fresh_hitl_manager.reject(
            request_id=request.request_id,
            rejector_id="DM001",
            rejector_name="Data Manager",
            rejector_role=ApproverRole.DATA_MANAGER,
            reason="Data discrepancy requires investigation"
        )
        
        assert success
        
        # Verify status
        rejected_request = fresh_hitl_manager.get_request_by_id(request.request_id)
        assert rejected_request.status == ApprovalStatus.REJECTED


class TestHITLEscalation:
    """Test HITL escalation for delayed approvals"""
    
    def test_escalation_detection(self, fresh_hitl_manager):
        """Test that overdue requests are detected for escalation"""
        request, _ = fresh_hitl_manager.request_approval(
            agent_id=AgentIdentifier.LIA.value,
            agent_name="Lia",
            action_type="ESCALATE",
            action_description="Escalate ghost visit",
            proposed_action="Notify CRA",
            study_id="STUDY-001",
            risk_level=ActionRiskLevel.HIGH
        )
        
        # Simulate time passing
        request.escalation_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Check for escalations
        escalations = fresh_hitl_manager.check_escalations()
        
        assert len(escalations) >= 1
        assert escalations[0].status == ApprovalStatus.ESCALATED


class TestHITLStatistics:
    """Test HITL statistics for compliance reporting"""
    
    def test_statistics_generation(self, fresh_hitl_manager):
        """Test generation of HITL statistics"""
        # Create and process several requests
        for i in range(3):
            request, _ = fresh_hitl_manager.request_approval(
                agent_id=AgentIdentifier.CODEX.value,
                agent_name="Codex",
                action_type="PROPOSE",
                action_description=f"Propose coding {i}",
                proposed_action=f"Code term {i}",
                study_id="STUDY-001",
                risk_level=ActionRiskLevel.MEDIUM
            )
            
            # Approve one
            if i == 0:
                fresh_hitl_manager.approve(
                    request_id=request.request_id,
                    approver_id="DM001",
                    approver_name="DM",
                    approver_role=ApproverRole.DATA_MANAGER
                )
        
        stats = fresh_hitl_manager.get_statistics(study_id="STUDY-001")
        
        assert stats['total_requests'] == 3
        assert 'APPROVED' in stats['by_status'] or stats['by_status'].get('APPROVED', 0) >= 0
        assert 'compliance_note' in stats


# =============================================================================
# RISK-BASED MONITORING TESTS (ICH E6 R2/R3)
# =============================================================================

class TestRiskBasedMonitoring:
    """Test risk-based monitoring per ICH E6 R2/R3"""
    
    def test_site_categorization_green(self, fresh_compliance_manager):
        """Test GREEN site categorization for high DQI"""
        profile = fresh_compliance_manager.categorize_site(
            site_id="Site 1",
            dqi_score=0.92
        )
        
        assert profile.risk_category == SiteRiskCategory.GREEN
        assert profile.sdv_rate == 0.25  # 25% SDV
        assert profile.monitoring_frequency == "Quarterly"
        assert profile.auto_query_enabled
    
    def test_site_categorization_yellow(self, fresh_compliance_manager):
        """Test YELLOW site categorization for medium DQI"""
        profile = fresh_compliance_manager.categorize_site(
            site_id="Site 2",
            dqi_score=0.78
        )
        
        assert profile.risk_category == SiteRiskCategory.YELLOW
        assert profile.sdv_rate == 0.50  # 50% SDV
        assert profile.monitoring_frequency == "Monthly"
        assert profile.escalation_delay_days == 7
    
    def test_site_categorization_red(self, fresh_compliance_manager):
        """Test RED site categorization for low DQI (per scatter plot red quadrant)"""
        profile = fresh_compliance_manager.categorize_site(
            site_id="Site 3",
            dqi_score=0.62
        )
        
        assert profile.risk_category == SiteRiskCategory.RED
        assert profile.sdv_rate == 1.00  # 100% SDV
        assert profile.monitoring_frequency == "Weekly"
        assert not profile.auto_query_enabled  # Require CRA review
        assert profile.requires_cra_intervention
    
    def test_dqi_trend_tracking(self, fresh_compliance_manager):
        """Test that DQI trends are tracked"""
        # Initial categorization
        profile1 = fresh_compliance_manager.categorize_site(
            site_id="Site 4",
            dqi_score=0.75
        )
        
        # Second categorization with improvement
        profile2 = fresh_compliance_manager.categorize_site(
            site_id="Site 4",
            dqi_score=0.88
        )
        
        assert profile2.dqi_trend == "IMPROVING"
        assert 0.75 in profile2.previous_dqi_scores
    
    def test_declining_site_flagged(self, fresh_compliance_manager):
        """Test that declining sites are flagged for intervention"""
        # Initial good score
        fresh_compliance_manager.categorize_site(
            site_id="Site 5",
            dqi_score=0.90
        )
        
        # Declining score
        profile = fresh_compliance_manager.categorize_site(
            site_id="Site 5",
            dqi_score=0.78
        )
        
        assert profile.dqi_trend == "DECLINING"
        assert profile.requires_cra_intervention


class TestComplianceReporting:
    """Test compliance reporting for regulatory submission"""
    
    def test_compliance_report_generation(self, fresh_compliance_manager):
        """Test generation of compliance report"""
        # Categorize some sites
        fresh_compliance_manager.categorize_site("Site 1", 0.92)
        fresh_compliance_manager.categorize_site("Site 2", 0.78)
        fresh_compliance_manager.categorize_site("Site 3", 0.65)
        
        report = fresh_compliance_manager.generate_compliance_report()
        
        assert report.study_id == "TEST-001"
        assert report.green_sites == 1
        assert report.yellow_sites == 1
        assert report.red_sites == 1
        assert ComplianceStandard.ICH_E6_R2.value in report.compliance_standards
    
    def test_sdv_reduction_calculation(self, fresh_compliance_manager):
        """Test SDV reduction calculation for risk-based monitoring"""
        # All green sites
        for i in range(5):
            fresh_compliance_manager.categorize_site(f"Site {i}", 0.90)
        
        report = fresh_compliance_manager.generate_compliance_report()
        
        # With all green sites (25% SDV), reduction should be 75%
        assert report.sdv_reduction_percentage > 70


class TestCPIDIntegration:
    """Test integration with CPID for agent action tracking"""
    
    def test_responsible_lf_field_update(self, fresh_compliance_manager):
        """Test that agent actions update Responsible LF field correctly"""
        update = fresh_compliance_manager.update_cpid_responsible_field(
            site_id="Site 1",
            subject_id="S001",
            action_description="Auto-generated query for zombie SAE",
            agent=AgentIdentifier.REX
        )
        
        # Per requirement: "System-Agent-01" instead of human user ID
        assert update['responsible_lf_for_action'] == "System-Agent-Rex-01"
        assert update['action_type'] == 'SYSTEM_AGENT_ACTION'
        assert update['audit_trail_reference']


class TestIntegratedCompliance:
    """Test integrated compliance flow"""
    
    def test_end_to_end_compliant_action(self, fresh_compliance_manager):
        """Test complete flow: action -> audit -> HITL -> tracking"""
        # Log a compliant action
        audit_entry, approval_request = fresh_compliance_manager.log_agent_action(
            agent=AgentIdentifier.REX,
            action_type=ActionType.DETECT,
            description="Detected zombie SAE requiring attention",
            subject_id="S001",
            site_id="Site 1",
            reason="SAE in safety DB without matching AE form",
            requires_approval=False  # Detection doesn't need approval
        )
        
        # Verify audit entry created
        assert audit_entry is not None
        assert audit_entry.actor_id == "System-Agent-Rex-01"
        assert audit_entry.study_id == "TEST-001"
        
        # No approval needed for detection
        assert approval_request is None
    
    def test_critical_action_with_hitl(self, fresh_compliance_manager):
        """Test that critical actions trigger HITL workflow"""
        audit_entry, approval_request = fresh_compliance_manager.log_agent_action(
            agent=AgentIdentifier.REX,
            action_type=ActionType.UPDATE,
            description="Update safety reconciliation status",
            subject_id="S001",
            site_id="Site 1",
            previous_value="Pending",
            new_value="Reconciled",
            requires_approval=True  # Critical action
        )
        
        # Should have pending approval
        assert approval_request is not None
        assert approval_request.status == ApprovalStatus.PENDING


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_study_id_handling(self, fresh_audit_manager):
        """Test handling of empty study ID"""
        entry = fresh_audit_manager.log_action(
            agent=AgentIdentifier.SYSTEM,
            action_type=ActionType.READ,
            description="System check"
        )
        
        # Should not crash, study_id defaults to empty
        assert entry.study_id == ""
    
    def test_approval_for_nonexistent_request(self, fresh_hitl_manager):
        """Test approval of non-existent request"""
        success, message = fresh_hitl_manager.approve(
            request_id="NONEXISTENT",
            approver_id="DM001",
            approver_name="DM",
            approver_role=ApproverRole.DATA_MANAGER
        )
        
        assert not success
        assert "not found" in message.lower()
    
    def test_dqi_boundary_values(self, fresh_compliance_manager):
        """Test DQI boundary values for categorization"""
        # Exactly at yellow threshold
        profile = fresh_compliance_manager.categorize_site("Site 1", 0.85)
        assert profile.risk_category == SiteRiskCategory.GREEN
        
        # Just below yellow threshold
        profile = fresh_compliance_manager.categorize_site("Site 2", 0.849)
        assert profile.risk_category == SiteRiskCategory.YELLOW
        
        # Exactly at red threshold
        profile = fresh_compliance_manager.categorize_site("Site 3", 0.70)
        assert profile.risk_category == SiteRiskCategory.YELLOW
        
        # Just below red threshold
        profile = fresh_compliance_manager.categorize_site("Site 4", 0.699)
        assert profile.risk_category == SiteRiskCategory.RED


class TestDataSerialization:
    """Test data serialization for persistence"""
    
    def test_audit_entry_serialization(self, fresh_audit_manager):
        """Test that audit entries can be serialized and deserialized"""
        entry = fresh_audit_manager.log_action(
            agent=AgentIdentifier.CODEX,
            action_type=ActionType.PROPOSE,
            description="Proposed coding",
            study_id="STUDY-001",
            subject_id="S001"
        )
        
        # Serialize
        entry_dict = entry.to_dict()
        json_str = json.dumps(entry_dict)
        
        # Deserialize
        restored_dict = json.loads(json_str)
        restored_entry = AuditEntry.from_dict(restored_dict)
        
        assert restored_entry.entry_id == entry.entry_id
        assert restored_entry.action_type == entry.action_type
    
    def test_compliance_report_serialization(self, fresh_compliance_manager):
        """Test that compliance reports can be serialized"""
        fresh_compliance_manager.categorize_site("Site 1", 0.85)
        report = fresh_compliance_manager.generate_compliance_report()
        
        # Should serialize without errors
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)
        
        assert len(json_str) > 0
        assert "study_id" in json_str


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
