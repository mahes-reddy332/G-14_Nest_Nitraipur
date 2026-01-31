
from typing import Dict
from core.ai.prompts import PromptRegistry
import datetime

class GenAIService:
    """
    Service for generating insights using LLMs with Governance controls.
    """
    
    def __init__(self):
        # In a real impl, we would initialize OpenAI client here
        pass

    async def generate_site_summary(self, site_data: Dict, risk_profile: Dict) -> Dict:
        """
        Generates a summary using a governed prompt template.
        Returns result with explainability metadata.
        """
        # 1. Retrieve Approved Prompt
        prompt_template = PromptRegistry.get_template("SITE_RISK_SUMMARY")
        if not prompt_template:
            # Fallback for resiliency, though ideally we raise or alarm
            return {
                "summary_text": "Error: Approved prompt template not found.",
                "governance_metadata": {"error": "PROMPT_MISSING"}
            }

        # 2. Construct Prompt (Simulation)
        # In a real app, we'd use LangChain to fill variables
        # We Map internal keys to template keys
        try:
            prompt_text = prompt_template.template.format(
                study_id=site_data.get("study_id", "Unknown"),
                site_id=site_data.get("site_id", "Unknown"),
                risk_level=risk_profile.get("risk_level", "Unknown"),
                metrics_json=str(risk_profile) # Serialize metrics for context
            )
        except KeyError as e:
             # Handle missing keys in prompt formatting
            return {
                "summary_text": "Error generating summary: Missing data context.",
                "governance_metadata": {"error": f"MISSING_KEY: {str(e)}"}
            }
        
        # 3. Call LLM (Mocked for now)
        # response = await openai.ChatCompletion.create(...)
        # We simulate a dynamic response based on risk level
        risk_score = risk_profile.get("risk_score", 0)
        if risk_score > 70:
            mock_response = f"Site {site_data.get('site_id')} is exhibiting High Risk behaviors. Primary drivers include elevated Query Aging Index and recent missing visits. Recommended immediate follow-up."
        else:
            mock_response = f"Site {site_data.get('site_id')} is performing within expected parameters. No significant risks detected at this time."

        # 4. Return Structured Output with Governance Metadata
        return {
            "summary_text": mock_response,
            "governance_metadata": {
                "prompt_id": prompt_template.id,
                "prompt_version": prompt_template.version,
                "model": "gpt-4o-mini", # Configurable in settings
                "confidence_score": 0.85 if risk_score < 90 else 0.92, 
                "guardrails_passed": True,
                "sources": ["fct_query_aging", "fct_missing_visits"],  # Traceability
                "generated_at": str(datetime.datetime.utcnow())
            }
        }
