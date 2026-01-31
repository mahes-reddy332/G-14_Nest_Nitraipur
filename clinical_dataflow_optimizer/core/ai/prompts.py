
from typing import Dict, Optional
from pydantic import BaseModel
import datetime

class PromptTemplate(BaseModel):
    id: str
    version: str
    template: str
    description: str
    last_approved: datetime.date
    approved_by: str

class PromptRegistry:
    """
    Centralized registry for approved AI prompts.
    Ensures only version-controlled, compliant prompts are used.
    """
    
    SITE_RISK_SUMMARY_V1 = PromptTemplate(
        id="SITE_RISK_SUMMARY",
        version="1.0.0",
        template="""
        You are a Clinical Data Scientist assistant. Analyze the following site performance metrics and produce a concise summary.
        
        Context:
        - Study ID: {study_id}
        - Site ID: {site_id}
        - Risk Level: {risk_level}
        
        Metrics:
        {metrics_json}
        
        Instructions:
        1. Identify the primary driver of the risk score.
        2. Suggest one specific mitigation action.
        3. Do NOT make medical diagnosis claims.
        4. Cite specific metrics in your explanation.
        """,
        description="Generates a summary of site risks based on KRI metrics.",
        last_approved=datetime.date(2023, 10, 27),
        approved_by="Dr. Sarah Chen (Head of Data Management)"
    )

    _registry: Dict[str, PromptTemplate] = {
        "SITE_RISK_SUMMARY": SITE_RISK_SUMMARY_V1
    }

    @classmethod
    def get_template(cls, prompt_id: str) -> Optional[PromptTemplate]:
        return cls._registry.get(prompt_id)
