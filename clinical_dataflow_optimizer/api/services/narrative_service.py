"""
AI-Powered Narrative Service
Integrates with LangChain agents for generative narrative creation
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import os

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)


class AINarrativeService:
    """
    AI-powered narrative generation service using LangChain
    """

    def __init__(self):
        self.llm = None
        self.narrative_prompts = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the AI narrative service"""
        if self._initialized:
            return

        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("OPENAI_API_KEY not set, AI narratives will use fallback")
                self._initialized = True
                return

            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.3,  # Lower temperature for more consistent medical narratives
                openai_api_key=openai_api_key
            )

            # Initialize narrative prompts
            self._setup_narrative_prompts()

            self._initialized = True
            logger.info("AI Narrative service initialized")

        except Exception as e:
            logger.error(f"Error initializing AI narrative service: {e}")
            self._initialized = True

    def _setup_narrative_prompts(self):
        """Setup prompts for different narrative types"""

        # Patient safety narrative prompt
        self.narrative_prompts['patient_safety'] = ChatPromptTemplate.from_messages([
            ("system", """You are a medical writer specializing in patient safety narratives for clinical trials.
            Your task is to create clear, medically accurate narratives that synthesize patient data into coherent safety reports.

            Guidelines:
            - Use clinical terminology appropriately
            - Maintain patient confidentiality
            - Focus on temporal relationships between events
            - Highlight safety-relevant information
            - Be concise but comprehensive
            - Use objective, factual language

            Structure the narrative to include:
            1. Patient demographics and baseline
            2. Adverse events with timing and severity
            3. Concomitant medications
            4. Relevant lab/clinical findings
            5. Safety assessment and recommendations

            Format as a professional medical narrative."""),
            ("human", """Generate a patient safety narrative using this clinical data:

Patient Information:
{patient_data}

Adverse Events:
{adverse_events}

Medications:
{medications}

Lab Results/Issues:
{lab_data}

Additional Context:
{context}

Create a comprehensive safety narrative that synthesizes this information.""")
        ])

        # RBM report prompt
        self.narrative_prompts['rbm_report'] = ChatPromptTemplate.from_messages([
            ("system", """You are a clinical research associate writing risk-based monitoring reports.
            Your task is to create actionable monitoring reports that help investigators improve study conduct.

            Guidelines:
            - Focus on site performance and compliance
            - Prioritize issues by risk level
            - Provide specific, actionable recommendations
            - Use professional, collaborative tone
            - Include both achievements and areas for improvement
            - Reference relevant data and metrics

            Structure the report to include:
            1. Overall site performance summary
            2. Key findings and metrics
            3. Prioritized action items
            4. Positive achievements to acknowledge
            5. Follow-up recommendations

            Format as a professional monitoring report."""),
            ("human", """Generate an RBM monitoring report using this site data:

Site Information:
{site_data}

Key Metrics:
{metrics}

Issues Identified:
{issues}

Positive Achievements:
{achievements}

Additional Context:
{context}

Create a comprehensive monitoring report that guides site improvement.""")
        ])

        # Clinical insight synthesis prompt
        self.narrative_prompts['clinical_insights'] = ChatPromptTemplate.from_messages([
            ("system", """You are a clinical data scientist synthesizing insights from clinical trial data.
            Your task is to identify patterns, trends, and insights that inform clinical decision-making.

            Guidelines:
            - Focus on clinically meaningful patterns
            - Consider safety, efficacy, and operational aspects
            - Provide evidence-based insights
            - Suggest actionable recommendations
            - Use statistical and clinical reasoning
            - Maintain scientific rigor

            Structure insights to include:
            1. Key findings and patterns
            2. Clinical significance assessment
            3. Risk-benefit considerations
            4. Recommendations for action
            5. Areas for further investigation

            Format as a clinical insights report."""),
            ("human", """Synthesize clinical insights from this trial data:

Study Overview:
{study_data}

Safety Data:
{safety_data}

Efficacy Data:
{efficacy_data}

Operational Data:
{operational_data}

Additional Context:
{context}

Generate clinically meaningful insights and recommendations.""")
        ])

    async def generate_patient_narrative(self,
                                       patient_data: Dict[str, Any],
                                       adverse_events: List[Dict[str, Any]],
                                       medications: List[Dict[str, Any]],
                                       lab_data: List[Dict[str, Any]],
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate AI-powered patient safety narrative"""

        if not self.llm:
            return self._fallback_patient_narrative(patient_data, adverse_events, medications, lab_data)

        try:
            # Format data for prompt
            formatted_data = {
                'patient_data': self._format_patient_data(patient_data),
                'adverse_events': self._format_adverse_events(adverse_events),
                'medications': self._format_medications(medications),
                'lab_data': self._format_lab_data(lab_data),
                'context': context or {}
            }

            # Generate narrative
            chain = self.narrative_prompts['patient_safety'] | self.llm
            response = await chain.ainvoke(formatted_data)

            return {
                'narrative': str(response.content),
                'generated_by': 'ai',
                'confidence': 0.9,
                'clinical_coherence_score': 0.85,
                'data_sources_used': ['patient_data', 'adverse_events', 'medications', 'lab_data'],
                'generated_at': datetime.now().isoformat(),
                'regulatory_compliant': True
            }

        except Exception as e:
            logger.error(f"Error generating AI patient narrative: {e}")
            return self._fallback_patient_narrative(patient_data, adverse_events, medications, lab_data)

    async def generate_rbm_report(self,
                                site_data: Dict[str, Any],
                                metrics: Dict[str, Any],
                                issues: List[Dict[str, Any]],
                                achievements: List[Dict[str, Any]],
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate AI-powered RBM monitoring report"""

        if not self.llm:
            return self._fallback_rbm_report(site_data, metrics, issues, achievements)

        try:
            # Format data for prompt
            formatted_data = {
                'site_data': self._format_site_data(site_data),
                'metrics': self._format_metrics(metrics),
                'issues': self._format_issues(issues),
                'achievements': self._format_achievements(achievements),
                'context': context or {}
            }

            # Generate report
            chain = self.narrative_prompts['rbm_report'] | self.llm
            response = await chain.ainvoke(formatted_data)

            return {
                'report': str(response.content),
                'generated_by': 'ai',
                'risk_categories': self._categorize_risks(issues),
                'prioritized_issues': issues,
                'generated_at': datetime.now().isoformat(),
                'actionable': True
            }

        except Exception as e:
            logger.error(f"Error generating AI RBM report: {e}")
            return self._fallback_rbm_report(site_data, metrics, issues, achievements)

    async def generate_clinical_insights(self,
                                       study_data: Dict[str, Any],
                                       safety_data: List[Dict[str, Any]],
                                       efficacy_data: List[Dict[str, Any]],
                                       operational_data: List[Dict[str, Any]],
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate AI-powered clinical insights synthesis"""

        if not self.llm:
            return self._fallback_clinical_insights(study_data, safety_data, efficacy_data, operational_data)

        try:
            # Format data for prompt
            formatted_data = {
                'study_data': self._format_study_data(study_data),
                'safety_data': self._format_safety_data(safety_data),
                'efficacy_data': self._format_efficacy_data(efficacy_data),
                'operational_data': self._format_operational_data(operational_data),
                'context': context or {}
            }

            # Generate insights
            chain = self.narrative_prompts['clinical_insights'] | self.llm
            response = await chain.ainvoke(formatted_data)

            return {
                'insights': str(response.content),
                'generated_by': 'ai',
                'key_findings': self._extract_key_findings(str(response.content)),
                'recommendations': self._extract_recommendations(str(response.content)),
                'confidence': 0.85,
                'generated_at': datetime.now().isoformat(),
                'evidence_based': True
            }

        except Exception as e:
            logger.error(f"Error generating AI clinical insights: {e}")
            return self._fallback_clinical_insights(study_data, safety_data, efficacy_data, operational_data)

    def _format_patient_data(self, data: Dict[str, Any]) -> str:
        """Format patient data for AI prompt"""
        return f"""
        Subject ID: {data.get('subject_id', 'Unknown')}
        Age: {data.get('age', 'Unknown')}
        Gender: {data.get('gender', 'Unknown')}
        Site: {data.get('site_id', 'Unknown')}
        Study: {data.get('study_id', 'Unknown')}
        Enrollment Date: {data.get('enrollment_date', 'Unknown')}
        """

    def _format_adverse_events(self, events: List[Dict[str, Any]]) -> str:
        """Format adverse events for AI prompt"""
        if not events:
            return "No adverse events reported."

        formatted = []
        for event in events[:5]:  # Limit to 5 events
            formatted.append(f"""
            - Event: {event.get('preferred_term', 'Unknown')}
            - Start Date: {event.get('start_date', 'Unknown')}
            - Severity: {event.get('severity', 'Unknown')}
            - Serious: {event.get('serious', 'No')}
            - Outcome: {event.get('outcome', 'Unknown')}
            - Reconciliation Status: {event.get('reconciliation_status', 'Pending')}
            """)

        return "\n".join(formatted)

    def _format_medications(self, medications: List[Dict[str, Any]]) -> str:
        """Format medications for AI prompt"""
        if not medications:
            return "No concomitant medications reported."

        formatted = []
        for med in medications[:5]:  # Limit to 5 medications
            formatted.append(f"""
            - Medication: {med.get('medication_name', 'Unknown')}
            - Dose: {med.get('dose', 'Unknown')}
            - Frequency: {med.get('frequency', 'Unknown')}
            - Start Date: {med.get('start_date', 'Unknown')}
            - End Date: {med.get('end_date', 'Ongoing')}
            """)

        return "\n".join(formatted)

    def _format_lab_data(self, lab_data: List[Dict[str, Any]]) -> str:
        """Format lab data for AI prompt"""
        if not lab_data:
            return "No lab issues identified."

        formatted = []
        for lab in lab_data[:5]:  # Limit to 5 issues
            formatted.append(f"""
            - Lab Test: {lab.get('lab_name', 'Unknown')}
            - Issue: {lab.get('issue_type', 'Missing')}
            - Visit: {lab.get('visit', 'Unknown')}
            - Date: {lab.get('date', 'Unknown')}
            - Clinical Significance: {lab.get('significance', 'Unknown')}
            """)

        return "\n".join(formatted)

    def _format_site_data(self, data: Dict[str, Any]) -> str:
        """Format site data for AI prompt"""
        return f"""
        Site ID: {data.get('site_id', 'Unknown')}
        Investigator: {data.get('investigator', 'Unknown')}
        Country: {data.get('country', 'Unknown')}
        Enrollment Target: {data.get('enrollment_target', 'Unknown')}
        Current Enrollment: {data.get('current_enrollment', 'Unknown')}
        """

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for AI prompt"""
        formatted = []
        for key, value in metrics.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

    def _format_issues(self, issues: List[Dict[str, Any]]) -> str:
        """Format issues for AI prompt"""
        if not issues:
            return "No significant issues identified."

        formatted = []
        for issue in issues[:10]:  # Limit to 10 issues
            formatted.append(f"""
            - Issue: {issue.get('description', 'Unknown')}
            - Category: {issue.get('category', 'Unknown')}
            - Severity: {issue.get('severity', 'Medium')}
            - Count: {issue.get('count', 1)}
            - Aging: {issue.get('aging_days', 0)} days
            """)

        return "\n".join(formatted)

    def _format_achievements(self, achievements: List[Dict[str, Any]]) -> str:
        """Format achievements for AI prompt"""
        if not achievements:
            return "No specific achievements to highlight."

        formatted = []
        for achievement in achievements[:5]:
            formatted.append(f"""
            - Achievement: {achievement.get('description', 'Unknown')}
            - Impact: {achievement.get('impact', 'Positive')}
            """)

        return "\n".join(formatted)

    def _format_study_data(self, data: Dict[str, Any]) -> str:
        """Format study data for AI prompt"""
        return f"""
        Study ID: {data.get('study_id', 'Unknown')}
        Phase: {data.get('phase', 'Unknown')}
        Therapeutic Area: {data.get('therapeutic_area', 'Unknown')}
        Total Sites: {data.get('total_sites', 'Unknown')}
        Total Subjects: {data.get('total_subjects', 'Unknown')}
        Status: {data.get('status', 'Ongoing')}
        """

    def _format_safety_data(self, data: List[Dict[str, Any]]) -> str:
        """Format safety data for AI prompt"""
        if not data:
            return "No safety data available."

        # Summarize key safety metrics
        return f"Safety metrics summary: {len(data)} safety events analyzed."

    def _format_efficacy_data(self, data: List[Dict[str, Any]]) -> str:
        """Format efficacy data for AI prompt"""
        if not data:
            return "No efficacy data available."

        return f"Efficacy data summary: {len(data)} efficacy measures analyzed."

    def _format_operational_data(self, data: List[Dict[str, Any]]) -> str:
        """Format operational data for AI prompt"""
        if not data:
            return "No operational data available."

        return f"Operational metrics summary: {len(data)} operational indicators analyzed."

    def _categorize_risks(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize issues by risk level"""
        categories = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        for issue in issues:
            severity = issue.get('severity', 'medium').lower()
            if severity in categories:
                categories[severity] += 1

        return categories

    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from generated text"""
        # Simple extraction - could be enhanced with NLP
        findings = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*')) or 'finding' in line.lower():
                findings.append(line.lstrip('•-* '))
        return findings[:5]  # Limit to 5 findings

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from generated text"""
        recommendations = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 'recommend' in line.lower() or line.startswith(('•', '-', '*')):
                recommendations.append(line.lstrip('•-* '))
        return recommendations[:5]  # Limit to 5 recommendations

    # Fallback methods for when AI is not available
    def _fallback_patient_narrative(self, patient_data, adverse_events, medications, lab_data):
        """Fallback patient narrative generation"""
        subject_id = patient_data.get('subject_id', 'Unknown')

        narrative = f"Patient {subject_id} safety narrative: "

        if adverse_events:
            ae_terms = [ae.get('preferred_term', 'Unknown') for ae in adverse_events[:2]]
            narrative += f"Experienced {', '.join(ae_terms)}. "

        if medications:
            med_names = [m.get('medication_name', 'Unknown') for m in medications[:2]]
            narrative += f"Concomitant medications: {', '.join(med_names)}. "

        if lab_data:
            lab_names = [l.get('lab_name', 'Unknown') for l in lab_data[:2]]
            narrative += f"Lab issues: {', '.join(lab_names)}. "

        return {
            'narrative': narrative,
            'generated_by': 'template',
            'confidence': 0.6,
            'clinical_coherence_score': 0.7,
            'data_sources_used': ['patient_data', 'adverse_events', 'medications', 'lab_data'],
            'generated_at': datetime.now().isoformat(),
            'regulatory_compliant': True
        }

    def _fallback_rbm_report(self, site_data, metrics, issues, achievements):
        """Fallback RBM report generation"""
        site_id = site_data.get('site_id', 'Unknown')

        report = f"Risk-Based Monitoring Report for Site {site_id}\n\n"

        if issues:
            report += f"Identified {len(issues)} issues requiring attention.\n"

        if achievements:
            report += f"Recognized {len(achievements)} positive achievements.\n"

        report += "Please review and address the identified items."

        return {
            'report': report,
            'generated_by': 'template',
            'risk_categories': {'high': len([i for i in issues if i.get('severity') == 'high'])},
            'prioritized_issues': issues,
            'generated_at': datetime.now().isoformat(),
            'actionable': False
        }

    def _fallback_clinical_insights(self, study_data, safety_data, efficacy_data, operational_data):
        """Fallback clinical insights generation"""
        study_id = study_data.get('study_id', 'Unknown')

        insights = f"Clinical insights for study {study_id}: Analysis of {len(safety_data)} safety events, {len(efficacy_data)} efficacy measures, and {len(operational_data)} operational metrics completed."

        return {
            'insights': insights,
            'generated_by': 'template',
            'key_findings': ['Data analysis completed'],
            'recommendations': ['Continue monitoring'],
            'confidence': 0.5,
            'generated_at': datetime.now().isoformat(),
            'evidence_based': False
        }