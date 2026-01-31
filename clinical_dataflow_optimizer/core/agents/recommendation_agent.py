
from typing import List, Dict, Any
from enum import Enum
import uuid
from datetime import datetime

class AgentRole(Enum):
    MONITOR = "monitor"
    RECOMMENDER = "recommender"
    APPROVER = "approver"

class RecommendationAgent:
    """
    Agent responsible for suggesting actions based on risk signals.
    Follows 'Human-in-the-loop' pattern.
    """
    
    def __init__(self, risk_service):
        self.risk_service = risk_service
        
    async def analyze_and_recommend(self, study_id: str) -> List[Dict[str, Any]]:
        """
        Analyzes study risk profile and generates recommendations.
        """
        # 1. Get Risk Profile from Risk Service
        site_risks = await self.risk_service.get_site_risk_profile(study_id)
        
        recommendations = []
        
        # 2. Heuristic Logic (Agent Brain)
        for site in site_risks:
            if site['risk_level'] == 'High':
                rec = {
                    "id": str(uuid.uuid4()),
                    "type": "site_intervention",
                    "target_site": site['site_id'],
                    "reason": f"High risk score ({site['site_risk_score']}) due to query aging.",
                    "suggested_action": "Schedule monitoring visit or contact site coordinator.",
                    "status": "pending_approval", # Human-in-the-loop
                    "created_at": datetime.now().isoformat()
                }
                recommendations.append(rec)
                
            elif site['risk_level'] == 'Medium' and site['open_query_count'] > 30:
                 rec = {
                    "id": str(uuid.uuid4()),
                    "type": "remote_check",
                    "target_site": site['site_id'],
                    "reason": "Accumulating query backlog.",
                    "suggested_action": "Email follow-up regarding open queries.",
                    "status": "pending_approval",
                    "created_at": datetime.now().isoformat()
                }
                 recommendations.append(rec)

        return recommendations
