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
from uuid import uuid4

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# LongCat AI integration
from core.longcat_integration import longcat_client, LongCatClient
from config.settings import DEFAULT_LONGCAT_CONFIG
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


class AgentServiceUnavailableError(Exception):
    """Raised when AI agent services are unavailable"""


class AgentService:
    """
    Service for interfacing with LangChain-based AI agents
    Uses LongCat AI as primary LLM provider for reasoning
    """
    
    def __init__(self):
        self.llm = None
        self._llm_available = False
        self.longcat_client = None  # LongCat AI client for enhanced reasoning
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
            
            # Initialize LongCat LLM (primary) or fallback to OpenAI
            longcat_api_key = os.getenv("API_KEY_Longcat", "")
            if longcat_api_key:
                # Use LongCat AI via OpenAI-compatible interface
                self.llm = ChatOpenAI(
                    model=DEFAULT_LONGCAT_CONFIG.model,
                    temperature=0.1,
                    openai_api_key=longcat_api_key,
                    openai_api_base="https://api.longcat.chat/v1"
                )
                self.longcat_client = longcat_client
                self._llm_available = True
                logger.info("Agent service initialized with LongCat AI LLM")
            else:
                # Fallback to OpenAI if available
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    logger.warning("Neither API_KEY_Longcat nor OPENAI_API_KEY set, AI insights unavailable")
                    self._initialized = True
                    self._llm_available = False
                    return
                
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.1,
                    openai_api_key=openai_api_key
                )
                self.longcat_client = None
                self._llm_available = True
                logger.info("Agent service initialized with OpenAI (fallback)")
            
            # Initialize agents
            await self._initialize_agents()
            self._llm_available = True
            
            self._initialized = True
            logger.info("Agent service initialized with LangChain")
            
        except Exception as e:
            logger.error(f"Error initializing agent service: {e}")
            self._initialized = True  # Mark as initialized to prevent repeated attempts
            self._llm_available = False
    
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

    def _ensure_llm_available(self):
        if not self._llm_available:
            raise AgentServiceUnavailableError("AI agent services are unavailable: missing LLM configuration")

    def _build_insight(
        self,
        agent: str,
        title: str,
        description: str,
        category: str,
        priority: str,
        confidence: float,
        affected_entities: Optional[Dict[str, Any]] = None,
        recommended_action: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "insight_id": f"INS-{uuid4().hex[:8]}",
            "agent": agent,
            "title": title,
            "description": description,
            "category": category,
            "priority": priority,
            "confidence": max(0.0, min(1.0, confidence)),
            "affected_entities": affected_entities or {},
            "recommended_action": recommended_action,
            "generated_at": datetime.now().isoformat(),
            "expires_at": None,
            "metadata": metadata or {},
        }

    async def _get_rule_based_insights(self, agent_type: Optional[str] = None) -> List[Dict]:
        from api.config import get_initialized_data_service

        data_service = await get_initialized_data_service()
        summary = await data_service.get_dashboard_summary(None)
        coding = await data_service.get_coding_aggregate(None)
        sae = await data_service.get_sae_aggregate(None)
        agg = await data_service.get_cpid_aggregate(None)
        sites = await data_service.get_sites({}, "dqi_score", "asc")

        insights: List[Dict[str, Any]] = []
        total_patients = summary.get("total_patients", 0) or 0
        overall_dqi = float(summary.get("overall_dqi", 0) or 0)
        open_queries = int(summary.get("open_queries", 0) or 0)
        pending_saes = int(summary.get("pending_saes", 0) or 0)
        uncoded_terms = int(summary.get("uncoded_terms", 0) or 0)

        normalized_agent = agent_type

        # Reconciliation insights
        if not normalized_agent or normalized_agent in {"reconciliation", "rex"}:
            if pending_saes > 0:
                priority = "critical" if pending_saes >= 5 else "high"
                insights.append(self._build_insight(
                    agent="reconciliation",
                    title="Pending SAE reconciliation",
                    description=f"{pending_saes} SAE records require reconciliation review.",
                    category="risk",
                    priority=priority,
                    confidence=0.9 if pending_saes >= 5 else 0.75,
                    affected_entities={"study_id": "all"},
                    recommended_action="Review pending SAEs and reconcile safety/clinical records.",
                    metadata={"pending_saes": pending_saes}
                ))

        # Coding insights
        if not normalized_agent or normalized_agent in {"coding", "codex"}:
            if uncoded_terms > 0:
                ratio = uncoded_terms / max(1, total_patients)
                priority = "high" if ratio >= 0.3 or uncoded_terms >= 50 else "medium"
                insights.append(self._build_insight(
                    agent="coding",
                    title="Uncoded terms backlog",
                    description=f"{uncoded_terms} medical terms remain uncoded across studies.",
                    category="compliance",
                    priority=priority,
                    confidence=0.8 if priority == "high" else 0.65,
                    affected_entities={"study_id": "all"},
                    recommended_action="Prioritize MedDRA/WHODrug coding for pending terms.",
                    metadata={"uncoded_terms": uncoded_terms, "pending_terms": coding.get("pending_terms", 0)}
                ))

        # Site liaison insights
        if not normalized_agent or normalized_agent in {"site_liaison", "lia"}:
            for site in sites[:2]:
                site_id = site.get("site_id") or site.get("site_name")
                if not site_id:
                    continue
                site_dqi = float(site.get("dqi_score", 0) or 0)
                site_queries = int(site.get("open_queries", 0) or 0)
                if site_dqi < 75 or site_queries > 20:
                    priority = "high" if site_dqi < 60 or site_queries > 50 else "medium"
                    insights.append(self._build_insight(
                        agent="site_liaison",
                        title="Site performance risk",
                        description=f"Site {site_id} has DQI {site_dqi:.1f} and {site_queries} open queries.",
                        category="operational",
                        priority=priority,
                        confidence=0.7 if priority == "medium" else 0.85,
                        affected_entities={"site_id": site_id, "study_id": site.get("study_id")},
                        recommended_action="Coordinate with site to resolve queries and improve data capture.",
                        metadata={"dqi_score": site_dqi, "open_queries": site_queries}
                    ))

        # Supervisor insights
        if not normalized_agent or normalized_agent in {"supervisor", "supervisor"}:
            if overall_dqi < 80 or open_queries > max(30, int(total_patients * 0.3)):
                priority = "high" if overall_dqi < 70 or open_queries > max(60, int(total_patients * 0.5)) else "medium"
                insights.append(self._build_insight(
                    agent="supervisor",
                    title="Overall data quality risk",
                    description=f"Overall DQI is {overall_dqi:.1f} with {open_queries} open queries.",
                    category="quality",
                    priority=priority,
                    confidence=0.75 if priority == "medium" else 0.88,
                    affected_entities={"study_id": "all"},
                    recommended_action="Escalate data quality remediation and query resolution focus.",
                    metadata={
                        "overall_dqi": overall_dqi,
                        "open_queries": open_queries,
                        "missing_visits": agg.get("missing_visits", 0),
                        "missing_pages": agg.get("missing_pages", 0),
                    }
                ))

        return insights
    
    async def get_agent_insights(self, 
                                  agent_type: Optional[str] = None,
                                  priority: Optional[str] = None,
                                  limit: int = 50) -> List[Dict]:
        """Get insights from AI agents"""
        insights: List[Dict[str, Any]] = []
        normalized_agent_type = agent_type
        if agent_type in {"reconciliation", "coding", "site_liaison"}:
            normalized_agent_type = agent_type
        elif agent_type in {"rex", "codex", "lia"}:
            normalized_agent_type = {
                "rex": "reconciliation",
                "codex": "coding",
                "lia": "site_liaison",
            }.get(agent_type, agent_type)

        # Generate insights based on loaded agents
        if not normalized_agent_type or normalized_agent_type == 'reconciliation':
            insights.extend(await self._get_reconciliation_insights())
        
        if not normalized_agent_type or normalized_agent_type == 'coding':
            insights.extend(await self._get_coding_insights())
        
        if not normalized_agent_type or normalized_agent_type == 'site_liaison':
            insights.extend(await self._get_site_liaison_insights())
        
        if not normalized_agent_type or normalized_agent_type == 'supervisor':
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
        return await self._get_rule_based_insights("reconciliation")
    
    
    async def _get_coding_insights(self) -> List[Dict]:
        """Get insights from Codex coding agent"""
        return await self._get_rule_based_insights("coding")
    
    
    async def _get_data_quality_insights(self) -> List[Dict]:
        """Get insights from data quality agent"""
        raise AgentServiceUnavailableError("Data quality agent not implemented")
    
    async def _get_predictive_insights(self) -> List[Dict]:
        """Get insights from predictive agent"""
        raise AgentServiceUnavailableError("Predictive agent not implemented")
    
    async def _get_site_liaison_insights(self) -> List[Dict]:
        """Get insights from site liaison agent"""
        return await self._get_rule_based_insights("site_liaison")

    async def _get_supervisor_insights(self) -> List[Dict]:
        """Get insights from supervisor agent"""
        return await self._get_rule_based_insights("supervisor")
    
    async def get_recommendations(self,
                                   category: Optional[str] = None,
                                   study_id: Optional[str] = None,
                                   limit: int = 20) -> List[Dict]:
        """Get actionable recommendations from agents"""
        self._ensure_llm_available()
        raise AgentServiceUnavailableError("Recommendations require live agent pipelines")
    
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
        raise AgentServiceUnavailableError("Reconciliation discrepancies require live data pipelines")
    
    
    async def get_coding_issues(self,
                                 study_id: Optional[str] = None,
                                 status: Optional[str] = None) -> List[Dict]:
        """Get medical coding issues"""
        raise AgentServiceUnavailableError("Coding issues require live coding pipeline data")
    
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
