
import pandas as pd
from analytics.kri_engine import calculate_site_kris, calculate_subject_kris
from api.services.data_service import ClinicalDataService

from core.audit.audit_service import AuditService, AuditEventType
import uuid

class RiskScoringService:
    """
    Service to orchestrate Risk Scoring using derived features.
    """
    
    def __init__(self, data_service: ClinicalDataService, audit_service: AuditService):
        self.data_service = data_service
        self.audit_service = audit_service
        
    async def get_site_risk_profile(self, study_id: str, user_id: str = "system") -> list[dict]:
        """
        Generates risk profiles for all sites in a study.
        """
        # Audit Start
        await self.audit_service.log_event(
            event_type=AuditEventType.RISK_SCORING,
            user_id=user_id,
            action="GENERATE_SITE_RISK",
            details={"study_id": study_id}
        )

        # 1. Fetch raw features (In MVP, we might calculate them on fly or fetch from dbt view)
        # For simulation, we mock the feature retrieval based on real data
        # In a real impl, this would query `fct_query_aging` and `fct_missing_visits`
        
        # Simulating feature dataframe for demonstration
        # This connects the dot between "Data Service" and "Analytics Engine"
        data = await self.data_service.get_missing_lab_data(study_id)
        # ... logic to transform raw data to features ...
        
        # Mock features for now to prove the KRI engine integration
        mock_features = pd.DataFrame([
            {'study_id': study_id, 'site_id': 'Site_101', 'open_query_count': 60, 'query_aging_index': 20, 'missing_visit_ratio': 0.0},
            {'study_id': study_id, 'site_id': 'Site_102', 'open_query_count': 10, 'query_aging_index': 5, 'missing_visit_ratio': 0.0},
        ])
        
        # 2. Apply KRI Logic
        scored_df = calculate_site_kris(mock_features)
        
        return scored_df.to_dict(orient='records')

    async def get_subject_readiness(self, study_id: str) -> list[dict]:
        """
        Generates readiness scores for subjects (e.g. for Database Lock).
        """
        # Mock features
        mock_features = pd.DataFrame([
             {'study_id': study_id, 'subject_id': 'SUBJ-001', 'missing_visit_ratio': 0.15},
             {'study_id': study_id, 'subject_id': 'SUBJ-002', 'missing_visit_ratio': 0.02},
        ])
        
        scored_df = calculate_subject_kris(mock_features)
        
        return scored_df.to_dict(orient='records')
