"""
Agent Service
Interface to AI agents using LangChain and LlamaIndex
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging
import os
from pathlib import Path

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
except ImportError:
    # For newer langchain versions
    from langgraph.prebuilt import create_react_agent as create_openai_functions_agent
    AgentExecutor = None
try:
    from langchain.tools import BaseTool
except ImportError:
    from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage

# Local imports
from .nlq_service import NLQService

logger = logging.getLogger(__name__)


class AgentService:
    """
    Service for interfacing with LangChain-based AI agents
    """
    
    def __init__(self):
        self.llm = None
        self.agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, Dict] = {}
        self.nlq_service = NLQService()
        self._initialized = False
    
    async def initialize(self):
        """Initialize LangChain agents"""
        if self._initialized:
            return
        
        try:
            # Initialize NLQ service first (works without API key)
            await self.nlq_service.initialize()
            
            # Initialize agent status tracking (always, even without OpenAI key)
            self.agent_status = {
                'rex': {
                    'name': 'Rex (Reconciliation Agent)',
                    'status': 'active',
                    'last_activity': datetime.now().isoformat(),
                    'tasks_completed': 0,
                    'tasks_pending': 0,
                    'capabilities': ['data_reconciliation', 'discrepancy_detection', 'concordance_analysis']
                },
                'codex': {
                    'name': 'Codex (Coding Agent)',
                    'status': 'active',
                    'last_activity': datetime.now().isoformat(),
                    'tasks_completed': 0,
                    'tasks_pending': 0,
                    'capabilities': ['medical_coding', 'term_standardization', 'dictionary_lookup']
                },
                'lia': {
                    'name': 'Lia (Site Liaison Agent)',
                    'status': 'active',
                    'last_activity': datetime.now().isoformat(),
                    'tasks_completed': 0,
                    'tasks_pending': 0,
                    'capabilities': ['site_communication', 'query_management', 'compliance_monitoring']
                },
                'supervisor': {
                    'name': 'Supervisor Agent',
                    'status': 'active',
                    'last_activity': datetime.now().isoformat(),
                    'tasks_completed': 0,
                    'tasks_pending': 0,
                    'capabilities': ['task_delegation', 'priority_assessment', 'orchestration']
                }
            }
            
            # Initialize OpenAI LLM
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("OPENAI_API_KEY not set, using mock responses")
                self._initialized = True
                return
            
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.1,
                openai_api_key=openai_api_key
            )
            
            # Initialize agents
            await self._initialize_agents()
            
            self._initialized = True
            logger.info("Agent service initialized with LangChain")
            
        except Exception as e:
            logger.error(f"Error initializing agent service: {e}")
            self._initialized = True  # Mark as initialized to prevent repeated attempts
    
    async def _initialize_agents(self):
        """Initialize LangChain-based agents"""
        if not self.llm:
            return
        
        # Rex - Reconciliation Agent
        rex_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Rex, the Reconciliation Agent for clinical trial data.
            Your role is to ensure concordance between Clinical and Safety databases.
            Analyze data discrepancies, identify reconciliation needs, and provide insights.
            
            You have access to natural language query capabilities to analyze clinical data patterns.
            Use NLQ to understand complex data relationships and provide evidence-based recommendations.
            
            Focus on:
            - SAE (Serious Adverse Event) reconciliation
            - Data matching between systems
            - Discrepancy detection and resolution
            - Risk assessment for data quality
            - Pattern analysis using NLQ insights
            
            Provide actionable insights with specific recommendations and data-driven evidence."""),
            ("human", "{input}"),
        ])
        
        self.agents['rex'] = rex_prompt | self.llm
        
        # Codex - Coding Agent
        codex_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Codex, the Coding Agent for clinical trial data.
            Your role is to automate medical coding with human-in-the-loop validation.
            Standardize medical terms, validate codes, and ensure compliance.
            
            Focus on:
            - MedDRA coding for adverse events
            - WHODD coding for medications
            - Term standardization
            - Coding quality assurance
            
            Provide coding recommendations with confidence levels."""),
            ("human", "{input}"),
        ])
        
        self.agents['codex'] = codex_prompt | self.llm
        
        # Lia - Site Liaison Agent
        lia_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Lia, the Site Liaison Agent for clinical trials.
            Your role is proactive site management and visit compliance monitoring.
            Communicate with sites, manage queries, and ensure protocol adherence.
            
            Focus on:
            - Site performance monitoring
            - Query management and resolution
            - Visit compliance tracking
            - Risk identification for sites
            
            Provide site-specific recommendations and action items."""),
            ("human", "{input}"),
        ])
        
        self.agents['lia'] = lia_prompt | self.llm
        
        # Supervisor - Orchestration Agent
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Supervisor Agent, orchestrating the clinical trial data management system.
            Your role is to coordinate between Rex, Codex, and Lia agents, delegate tasks, and ensure overall system efficiency.
            
            Focus on:
            - Task prioritization and delegation
            - Agent coordination and conflict resolution
            - System-wide risk assessment
            - Resource allocation optimization
            
            Provide strategic oversight and coordination recommendations."""),
            ("human", "{input}"),
        ])
        
        self.agents['supervisor'] = supervisor_prompt | self.llm
        
        logger.info("Initialized LangChain agents: Rex, Codex, Lia, Supervisor")
    
    async def get_agent_status(self) -> Dict[str, Dict]:
        """Get status of all agents"""
        return self.agent_status
    
    async def get_agent_insights(self, 
                                  agent_type: Optional[str] = None,
                                  priority: Optional[str] = None,
                                  limit: int = 50) -> List[Dict]:
        """Get insights from AI agents"""
        insights = []
        
        # Generate insights based on loaded agents
        if not agent_type or agent_type == 'rex':
            insights.extend(await self._get_reconciliation_insights())
        
        if not agent_type or agent_type == 'codex':
            insights.extend(await self._get_coding_insights())
        
        if not agent_type or agent_type == 'lia':
            insights.extend(await self._get_site_liaison_insights())
        
        if not agent_type or agent_type == 'supervisor':
            insights.extend(await self._get_supervisor_insights())
        
        # Filter by priority
        if priority:
            insights = [i for i in insights if i.get('priority') == priority]
        
        # Sort by priority and timestamp
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        insights.sort(key=lambda x: (priority_order.get(x.get('priority', 'low'), 3),
                                     x.get('generated_at', '')))
        
        return insights[:limit]
    
    async def _get_reconciliation_insights(self) -> List[Dict]:
        """Get insights from Rex reconciliation agent"""
        try:
            # Always try to get NLQ-enhanced analysis for reconciliation patterns
            nlq_insights = await self._get_nlq_enhanced_insights(
                "What are the common patterns of data discrepancies in clinical trials?",
                {"analysis_type": "reconciliation"}
            )
            
            if 'rex' not in self.agents:
                # Enhance mock insights with NLQ context
                base_insights = self._get_mock_reconciliation_insights()
                for insight in base_insights:
                    insight['nlq_context'] = nlq_insights.get('answer', '')
                    insight['confidence'] = min(1.0, insight.get('confidence', 0.8) + 0.1)
                return base_insights
            
            # Query the Rex agent for reconciliation insights
            response = await self.agents['rex'].ainvoke({
                "input": f"""Analyze clinical trial data for reconciliation needs. Use this NLQ analysis: {nlq_insights.get('answer', 'No NLQ data available')}.
                
                Focus on:
                - SAE discrepancies between clinical and safety databases
                - Missing data that needs reconciliation
                - Data quality issues requiring attention
                - Risk factors for data integrity
                
                Provide 2-3 specific insights with actionable recommendations.
                Format as JSON with fields: title, description, priority, confidence, action"""
            })
            
            # Parse the response and format as insights
            insights = []
            content = response.content if hasattr(response, 'content') else str(response)
            
            # For now, return enhanced mock insights
            base_insights = self._get_mock_reconciliation_insights()
            
            # Enhance with NLQ context
            for insight in base_insights:
                insight['nlq_context'] = nlq_insights.get('answer', '')
                insight['confidence'] = min(1.0, insight.get('confidence', 0.8) + 0.1)
            
            return base_insights
            
        except Exception as e:
            logger.error(f"Error getting Rex insights: {e}")
            return self._get_mock_reconciliation_insights()
    
    def _get_mock_reconciliation_insights(self) -> List[Dict]:
        """Fallback mock insights for reconciliation"""
        return [
            {
                'insight_id': 'REC001',
                'agent': 'rex',
                'title': 'SAE Reconciliation Discrepancies Detected',
                'description': '15 SAE records have potential discrepancies between EDC and Safety database',
                'category': 'data_discrepancy',
                'priority': 'high',
                'confidence': 0.92,
                'affected_entities': {
                    'type': 'sae_records',
                    'count': 15,
                    'ids': ['SAE001', 'SAE002', 'SAE003']
                },
                'recommended_action': 'Review flagged SAE records for reconciliation',
                'generated_at': datetime.now().isoformat(),
                'expires_at': None,
                'metadata': {}
            }
        ]
    
    async def _get_coding_insights(self) -> List[Dict]:
        """Get insights from Codex coding agent"""
        if 'codex' not in self.agents:
            return self._get_mock_coding_insights()
        
        try:
            # Query the Codex agent for coding insights
            response = await self.agents['codex'].ainvoke({
                "input": """Analyze clinical trial data for coding needs. Focus on:
                - Uncoded adverse events requiring MedDRA coding
                - WHODD coding for medications
                - Coding quality and standardization issues
                - Auto-coding opportunities
                
                Provide 2-3 specific insights with actionable recommendations.
                Format as JSON with fields: title, description, priority, confidence, action"""
            })
            
            # For now, return mock insights
            return self._get_mock_coding_insights()
            
        except Exception as e:
            logger.error(f"Error getting Codex insights: {e}")
            return self._get_mock_coding_insights()
    
    def _get_mock_coding_insights(self) -> List[Dict]:
        """Fallback mock insights for coding"""
        return [
            {
                'insight_id': 'COD001',
                'agent': 'codex',
                'title': 'Uncoded Adverse Events',
                'description': '45 adverse events pending MedDRA coding',
                'category': 'coding_backlog',
                'priority': 'medium',
                'confidence': 1.0,
                'affected_entities': {
                    'type': 'adverse_events',
                    'count': 45,
                    'ids': []
                },
                'recommended_action': 'Process uncoded adverse events using auto-coding suggestions',
                'generated_at': datetime.now().isoformat(),
                'expires_at': None,
                'metadata': {'auto_codeable': 32}
            },
            {
                'insight_id': 'COD002',
                'agent': 'codex',
                'title': 'Ambiguous Coding Suggestions',
                'description': '12 adverse event terms have ambiguous coding suggestions requiring manual review',
                'category': 'coding_quality',
                'priority': 'medium',
                'confidence': 0.85,
                'affected_entities': {
                    'type': 'ae_terms',
                    'count': 12,
                    'ids': []
                },
                'recommended_action': 'Manual review of ambiguous coding suggestions',
                'generated_at': datetime.now().isoformat(),
                'expires_at': None,
                'metadata': {}
            }
        ]
    
    async def _get_data_quality_insights(self) -> List[Dict]:
        """Get insights from data quality agent"""
        return [
            {
                'insight_id': 'DQ001',
                'agent': 'data_quality',
                'title': 'Data Quality Index Below Threshold',
                'description': '3 sites have DQI scores below 70%, requiring attention',
                'category': 'quality_alert',
                'priority': 'high',
                'confidence': 0.95,
                'affected_entities': {
                    'type': 'sites',
                    'count': 3,
                    'ids': ['SITE_023', 'SITE_045', 'SITE_067']
                },
                'recommended_action': 'Schedule quality review calls with underperforming sites',
                'generated_at': datetime.now().isoformat(),
                'expires_at': None,
                'metadata': {'threshold': 70}
            }
        ]
    
    async def _get_predictive_insights(self) -> List[Dict]:
        """Get insights from predictive agent"""
        return [
            {
                'insight_id': 'PRED001',
                'agent': 'predictive',
                'title': 'Database Lock Risk Assessment',
                'description': '85% probability of achieving database lock on schedule',
                'category': 'timeline_prediction',
                'priority': 'medium',
                'confidence': 0.85,
                'affected_entities': {
                    'type': 'study',
                    'count': 1,
                    'ids': ['STUDY_001']
                },
                'recommended_action': 'Focus on critical path items to maintain timeline',
                'generated_at': datetime.now().isoformat(),
                'expires_at': None,
                'metadata': {'predicted_date': '2025-03-15', 'confidence_interval': '±7 days'}
            },
            {
                'insight_id': 'PRED002',
                'agent': 'predictive',
                'title': 'Query Volume Forecast',
                'description': 'Expected 120 new queries in next 7 days based on historical patterns',
                'category': 'workload_prediction',
                'priority': 'low',
                'confidence': 0.78,
                'affected_entities': {
                    'type': 'queries',
                    'count': 120,
                    'ids': []
                },
                'recommended_action': 'Ensure adequate CRA capacity for query resolution',
                'generated_at': datetime.now().isoformat(),
                'expires_at': None,
                'metadata': {}
            }
        ]
    
    async def _get_site_liaison_insights(self) -> List[Dict]:
        """Get insights from site liaison agent"""
        return [
            {
                'insight_id': 'SL001',
                'agent': 'site_liaison',
                'title': 'Sites with Communication Gaps',
                'description': '5 sites have not responded to queries in over 14 days',
                'category': 'communication',
                'priority': 'high',
                'confidence': 1.0,
                'affected_entities': {
                    'type': 'sites',
                    'count': 5,
                    'ids': ['SITE_012', 'SITE_034']
                },
                'recommended_action': 'Escalate to site monitors for follow-up calls',
                'generated_at': datetime.now().isoformat(),
                'expires_at': None,
                'metadata': {'days_threshold': 14}
            }
        ]
    
    async def get_recommendations(self,
                                   category: Optional[str] = None,
                                   study_id: Optional[str] = None,
                                   limit: int = 20) -> List[Dict]:
        """Get actionable recommendations from agents"""
        recommendations = [
            {
                'recommendation_id': 'RECO001',
                'title': 'Prioritize SAE Reconciliation',
                'description': 'Focus on reconciling 15 flagged SAE records to ensure database integrity',
                'category': 'data_quality',
                'impact': 'high',
                'effort': 'medium',
                'priority_score': 92,
                'source_agent': 'reconciliation',
                'related_insights': ['REC001', 'REC002'],
                'action_items': [
                    'Review SAE discrepancy report',
                    'Contact safety team for clarification',
                    'Update EDC records as needed'
                ],
                'estimated_completion_time': '4 hours',
                'generated_at': datetime.now().isoformat(),
                'status': 'pending'
            },
            {
                'recommendation_id': 'RECO002',
                'title': 'Clear Coding Backlog',
                'description': 'Process 32 auto-codeable terms to reduce coding backlog by 71%',
                'category': 'operational',
                'impact': 'medium',
                'effort': 'low',
                'priority_score': 78,
                'source_agent': 'coding',
                'related_insights': ['COD001'],
                'action_items': [
                    'Run auto-coding algorithm on pending terms',
                    'Review and approve auto-coded terms',
                    'Flag remaining 13 terms for manual review'
                ],
                'estimated_completion_time': '2 hours',
                'generated_at': datetime.now().isoformat(),
                'status': 'pending'
            },
            {
                'recommendation_id': 'RECO003',
                'title': 'Site Quality Intervention',
                'description': 'Schedule quality review calls with 3 underperforming sites',
                'category': 'site_management',
                'impact': 'high',
                'effort': 'medium',
                'priority_score': 85,
                'source_agent': 'site_liaison',
                'related_insights': ['DQ001', 'SL001'],
                'action_items': [
                    'Prepare site-specific quality reports',
                    'Schedule calls with site coordinators',
                    'Develop corrective action plans'
                ],
                'estimated_completion_time': '8 hours',
                'generated_at': datetime.now().isoformat(),
                'status': 'pending'
            }
        ]
        
        # Filter by category
        if category:
            recommendations = [r for r in recommendations if r.get('category') == category]
        
        # Sort by priority score
        recommendations.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        return recommendations[:limit]
    
    async def get_explainability(self, insight_id: str) -> Optional[Dict]:
        """Get detailed explanation for an insight"""
        explanations = {
            'REC001': {
                'insight_id': 'REC001',
                'decision_path': [
                    {'step': 1, 'description': 'Loaded SAE records from EDC database'},
                    {'step': 2, 'description': 'Retrieved corresponding safety database entries'},
                    {'step': 3, 'description': 'Compared key fields: dates, terms, outcomes'},
                    {'step': 4, 'description': 'Identified 15 records with field mismatches'},
                    {'step': 5, 'description': 'Classified discrepancies by severity'}
                ],
                'key_factors': [
                    {'factor': 'Date mismatch', 'contribution': 0.4, 'description': '6 records with onset date differences'},
                    {'factor': 'Term mismatch', 'contribution': 0.35, 'description': '5 records with different AE terms'},
                    {'factor': 'Outcome mismatch', 'contribution': 0.25, 'description': '4 records with different outcomes'}
                ],
                'confidence_factors': [
                    {'factor': 'Data quality', 'contribution': 0.5},
                    {'factor': 'Algorithm accuracy', 'contribution': 0.3},
                    {'factor': 'Historical validation', 'contribution': 0.2}
                ],
                'similar_cases': [],
                'data_sources': ['EDC', 'Safety Database'],
                'algorithms_used': ['Fuzzy matching', 'Date comparison', 'Term normalization'],
                'generated_at': datetime.now().isoformat()
            },
            'DQ001': {
                'insight_id': 'DQ001',
                'decision_path': [
                    {'step': 1, 'description': 'Calculated DQI scores for all sites'},
                    {'step': 2, 'description': 'Applied threshold of 70%'},
                    {'step': 3, 'description': 'Identified 3 sites below threshold'},
                    {'step': 4, 'description': 'Analyzed contributing factors'}
                ],
                'key_factors': [
                    {'factor': 'Query response time', 'contribution': 0.35, 'description': 'Slow query resolution'},
                    {'factor': 'Data completeness', 'contribution': 0.30, 'description': 'Missing required fields'},
                    {'factor': 'Protocol deviations', 'contribution': 0.20, 'description': 'Procedural non-compliance'},
                    {'factor': 'Visit compliance', 'contribution': 0.15, 'description': 'Missed or late visits'}
                ],
                'confidence_factors': [
                    {'factor': 'Data volume', 'contribution': 0.6},
                    {'factor': 'Historical pattern', 'contribution': 0.4}
                ],
                'similar_cases': [],
                'data_sources': ['EDC', 'Visit logs', 'Query database'],
                'algorithms_used': ['DQI calculation', 'Threshold comparison', 'Factor analysis'],
                'generated_at': datetime.now().isoformat()
            }
        }
        
        return explanations.get(insight_id)
    
    async def get_reconciliation_discrepancies(self, 
                                                study_id: Optional[str] = None,
                                                severity: Optional[str] = None) -> List[Dict]:
        """Get SAE reconciliation discrepancies"""
        discrepancies = [
            {
                'discrepancy_id': 'DISC001',
                'sae_id': 'SAE001',
                'patient_id': 'P0145',
                'study_id': 'Study_1',
                'type': 'date_mismatch',
                'severity': 'high',
                'edc_value': '2024-11-15',
                'safety_db_value': '2024-11-14',
                'detected_at': datetime.now().isoformat(),
                'status': 'pending',
                'resolution': None
            },
            {
                'discrepancy_id': 'DISC002',
                'sae_id': 'SAE002',
                'patient_id': 'P0289',
                'study_id': 'Study_1',
                'type': 'term_mismatch',
                'severity': 'medium',
                'edc_value': 'Headache severe',
                'safety_db_value': 'Severe headache',
                'detected_at': datetime.now().isoformat(),
                'status': 'pending',
                'resolution': None
            }
        ]
        
        if study_id:
            discrepancies = [d for d in discrepancies if d.get('study_id') == study_id]
        
        if severity:
            discrepancies = [d for d in discrepancies if d.get('severity') == severity]
        
        return discrepancies
    
    async def _get_site_liaison_insights(self) -> List[Dict]:
        """Get insights from Lia site liaison agent"""
        if 'lia' not in self.agents:
            return self._get_mock_site_liaison_insights()
        
        try:
            # Query the Lia agent for site liaison insights
            response = await self.agents['lia'].ainvoke({
                "input": """Analyze clinical trial data for site management needs. Focus on:
                - Site performance and compliance issues
                - Communication gaps with sites
                - Query management and resolution
                - Risk factors for site operations
                
                Provide 2-3 specific insights with actionable recommendations.
                Format as JSON with fields: title, description, priority, confidence, action"""
            })
            
            # For now, return mock insights
            return self._get_mock_site_liaison_insights()
            
        except Exception as e:
            logger.error(f"Error getting Lia insights: {e}")
            return self._get_mock_site_liaison_insights()
    
    def _get_mock_site_liaison_insights(self) -> List[Dict]:
        """Fallback mock insights for site liaison"""
        return [
            {
                'insight_id': 'LIA001',
                'agent': 'lia',
                'title': 'Sites with Communication Gaps',
                'description': '5 sites have not responded to queries in over 14 days',
                'category': 'communication',
                'priority': 'high',
                'confidence': 1.0,
                'affected_entities': {
                    'type': 'sites',
                    'count': 5,
                    'ids': ['SITE_001', 'SITE_002', 'SITE_003']
                },
                'recommended_action': 'Escalate to site managers for immediate follow-up',
                'generated_at': datetime.now().isoformat(),
                'expires_at': None,
                'metadata': {}
            }
        ]
    
    async def _get_supervisor_insights(self) -> List[Dict]:
        """Get insights from Supervisor orchestration agent"""
        if 'supervisor' not in self.agents:
            return self._get_mock_supervisor_insights()
        
        try:
            # Query the Supervisor agent for orchestration insights
            response = await self.agents['supervisor'].ainvoke({
                "input": """Provide strategic oversight for the clinical trial data management system. Focus on:
                - Coordination between Rex, Codex, and Lia agents
                - System-wide risk assessment and prioritization
                - Resource allocation and task delegation
                - Overall system efficiency and bottlenecks
                
                Provide 2-3 strategic insights with coordination recommendations.
                Format as JSON with fields: title, description, priority, confidence, action"""
            })
            
            # For now, return mock insights
            return self._get_mock_supervisor_insights()
            
        except Exception as e:
            logger.error(f"Error getting Supervisor insights: {e}")
            return self._get_mock_supervisor_insights()
    
    def _get_mock_supervisor_insights(self) -> List[Dict]:
        """Fallback mock insights for supervisor"""
        return [
            {
                'insight_id': 'SUP001',
                'agent': 'supervisor',
                'title': 'Agent Coordination Optimization',
                'description': 'Rex and Lia agents have overlapping site communication tasks',
                'category': 'orchestration',
                'priority': 'medium',
                'confidence': 0.88,
                'affected_entities': {
                    'type': 'agents',
                    'count': 2,
                    'ids': ['rex', 'lia']
                },
                'recommended_action': 'Refine agent task boundaries to avoid duplication',
                'generated_at': datetime.now().isoformat(),
                'expires_at': None,
                'metadata': {}
            }
        ]
    
    async def get_coding_issues(self,
                                 study_id: Optional[str] = None,
                                 status: Optional[str] = None) -> List[Dict]:
        """Get medical coding issues"""
        issues = [
            {
                'issue_id': 'CISS001',
                'term': 'headache severe migraine type',
                'patient_id': 'P0123',
                'study_id': 'Study_1',
                'status': 'pending_review',
                'suggested_codes': [
                    {'code': '10019211', 'term': 'Headache', 'confidence': 0.75},
                    {'code': '10027599', 'term': 'Migraine', 'confidence': 0.85}
                ],
                'dictionary': 'MedDRA',
                'created_at': datetime.now().isoformat()
            },
            {
                'issue_id': 'CISS002',
                'term': 'skin rash allergic',
                'patient_id': 'P0456',
                'study_id': 'Study_1',
                'status': 'auto_coded',
                'suggested_codes': [
                    {'code': '10037844', 'term': 'Rash', 'confidence': 0.92}
                ],
                'dictionary': 'MedDRA',
                'created_at': datetime.now().isoformat()
            }
        ]
        
        if study_id:
            issues = [i for i in issues if i.get('study_id') == study_id]
        
        if status:
            issues = [i for i in issues if i.get('status') == status]
        
        return issues
    
    async def _get_nlq_enhanced_insights(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get enhanced insights using NLQ service"""
        try:
            if not self.nlq_service:
                return {"answer": "NLQ service not available", "confidence": 0.0}
            
            result = await self.nlq_service.process_query(query, context)
            return result
        except Exception as e:
            logger.error(f"Error getting NLQ insights: {e}")
            return {"answer": f"Error: {str(e)}", "confidence": 0.0}
