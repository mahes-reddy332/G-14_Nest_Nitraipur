"""
Conversational Engine - Main Orchestration Layer for NLQ
=========================================================

The ConversationalEngine is the primary interface for natural language querying.
It orchestrates the query parser, executor, and insight generator to provide
a seamless conversational experience.

Features:
- Session management with conversation history
- Context-aware responses
- Query refinement and clarification
- Multi-turn conversations
- RAG-powered knowledge retrieval
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import pandas as pd
import json

from .query_parser import QueryParser, ParsedQuery, QueryIntent, MetricType, EntityType
from .query_executor import QueryExecutor, QueryResult
from .insight_generator import InsightGenerator, InsightContext, ConversationalResponse, Insight

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    timestamp: datetime
    user_query: str
    parsed_query: Optional[ParsedQuery]
    response: ConversationalResponse
    feedback: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'user_query': self.user_query,
            'parsed_query': self.parsed_query.to_dict() if self.parsed_query else None,
            'response_summary': self.response.answer[:200] if self.response else None,
            'feedback': self.feedback
        }


@dataclass
class ConversationSession:
    """Manages a conversation session with history and context"""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    history: List[ConversationTurn] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    active_filters: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn to history"""
        self.history.append(turn)
        
        # Update context with entities mentioned
        if turn.parsed_query:
            for ef in turn.parsed_query.entity_filters:
                self.active_filters[ef.entity_type.name] = ef.values
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get summary of recent conversation for context"""
        recent = self.history[-n:] if len(self.history) >= n else self.history
        summaries = []
        for turn in recent:
            summaries.append(f"Q: {turn.user_query}")
            if turn.response:
                summaries.append(f"A: {turn.response.answer[:100]}...")
        return "\n".join(summaries)
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'turns': len(self.history),
            'active_filters': self.active_filters
        }


class ConversationalEngine:
    """
    Main interface for Natural Language Querying of Clinical Data
    
    Provides a conversational experience for data exploration with:
    - Natural language understanding
    - Query execution against multiple data sources
    - Insight generation with RAG
    - Multi-turn conversation support
    - Session management
    
    Example:
        >>> engine = ConversationalEngine()
        >>> engine.load_data({'cpid': cpid_df, 'sae': sae_df})
        >>> response = engine.ask("Show me sites where missing visits is trending up")
        >>> print(response.to_markdown())
    """
    
    # Clarification prompts for ambiguous queries
    CLARIFICATION_PROMPTS = {
        'no_metric': "I'd be happy to help! Could you specify which metric you're interested in? For example: missing visits, open queries, SAE count, or data quality index.",
        'no_entity': "Which level would you like me to analyze? Sites, subjects, countries, or visits?",
        'ambiguous_time': "What time period should I analyze? Last 3 snapshots, this month, or a specific date range?",
        'low_confidence': "I'm not fully certain I understood your question. Could you rephrase or provide more details?"
    }
    
    # Quick command shortcuts
    QUICK_COMMANDS = {
        'help': "I can help you analyze clinical trial data. Try asking:\n- 'Show me sites with high missing visits'\n- 'What's the trend in open queries?'\n- 'Find anomalies in SAE data'\n- 'Compare site performance'",
        'metrics': "Available metrics:\n- Missing Visits\n- Open Queries\n- Uncoded Terms\n- SAE Count\n- Data Quality Index\n- Frozen/Locked CRFs",
        'entities': "I can analyze data by:\n- Sites\n- Subjects/Patients\n- Countries\n- Visits/Cycles\n- Forms/CRFs"
    }
    
    def __init__(self, data_sources: Dict[str, pd.DataFrame] = None, graph=None):
        """
        Initialize the Conversational Engine
        
        Args:
            data_sources: Dictionary of DataFrames keyed by source name
            graph: Optional NetworkX graph for graph-based queries
        """
        self.parser = QueryParser()
        self.executor = QueryExecutor(data_sources=data_sources, graph=graph)
        self.generator = InsightGenerator()
        
        self.data_sources = data_sources or {}
        self.graph = graph
        
        # Session management
        self.sessions: Dict[str, ConversationSession] = {}
        self.current_session: Optional[ConversationSession] = None
        
        # Knowledge base for RAG
        self._knowledge_base = self._build_knowledge_base()
        
        logger.info("ConversationalEngine initialized")
    
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Build knowledge base for retrieval-augmented generation"""
        return {
            'clinical_terminology': {
                'SAE': 'Serious Adverse Event - an adverse event that results in death, is life-threatening, requires hospitalization, or results in persistent disability',
                'CRF': 'Case Report Form - a document used to collect patient data during a clinical trial',
                'DQI': 'Data Quality Index - a composite score measuring data completeness and accuracy',
                'SSM': 'Site Status Metric - indicator of overall site performance',
                'EDRR': 'Expedited Drug Regulatory Reporting - urgent safety reports to regulatory authorities'
            },
            'best_practices': [
                'Review high-risk sites weekly',
                'Address queries older than 14 days',
                'Escalate SAEs immediately',
                'Monitor trends over at least 3 snapshots',
                'Prioritize sites with SSM < 70%'
            ],
            'thresholds': {
                'missing_visits': {'warning': 5, 'critical': 10},
                'open_queries': {'warning': 20, 'critical': 50},
                'sae_aging_days': {'warning': 14, 'critical': 30}
            }
        }
    
    def load_data(self, data_sources: Dict[str, pd.DataFrame], graph=None):
        """
        Load or update data sources
        
        Args:
            data_sources: Dictionary of DataFrames
            graph: Optional NetworkX graph
        """
        self.data_sources = data_sources
        for name, df in data_sources.items():
            self.executor.set_data_source(name, df)
            logger.info(f"Loaded data source '{name}' with {len(df)} rows")
        
        if graph:
            self.graph = graph
            self.executor.set_graph(graph)
    
    def start_session(self, session_id: str = None) -> str:
        """
        Start a new conversation session
        
        Args:
            session_id: Optional session ID. Auto-generated if not provided.
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = ConversationSession(session_id=session_id)
        self.sessions[session_id] = self.current_session
        
        logger.info(f"Started session: {session_id}")
        return session_id
    
    def ask(self, query: str, session_id: str = None) -> ConversationalResponse:
        """
        Process a natural language query and return a conversational response
        
        Args:
            query: Natural language query from user
            session_id: Optional session ID for multi-turn conversations
            
        Returns:
            ConversationalResponse with answer, insights, and suggestions
        """
        start_time = datetime.now()
        
        # Ensure we have a session
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        elif self.current_session:
            session = self.current_session
        else:
            self.start_session()
            session = self.current_session
        
        # Check for quick commands
        if query.lower().strip() in self.QUICK_COMMANDS:
            response = ConversationalResponse(
                query=query,
                understanding="Quick command recognized",
                answer=self.QUICK_COMMANDS[query.lower().strip()],
                confidence=1.0
            )
            return response
        
        # Parse the query
        parsed_query = self.parser.parse(query)
        
        # Apply session context (carry over filters from previous turns)
        parsed_query = self._apply_session_context(parsed_query, session)
        
        # Check if clarification is needed
        if self._needs_clarification(parsed_query):
            response = self._generate_clarification_response(query, parsed_query)
        elif parsed_query.intent == QueryIntent.NARRATIVE_GENERATION:
            # Handle narrative generation
            response = self._generate_narrative_response(query, parsed_query)
        elif parsed_query.intent == QueryIntent.RBM_REPORT:
            # Handle RBM report generation
            response = self._generate_rbm_response(query, parsed_query)
        else:
            # Execute the query
            result = self.executor.execute(parsed_query)
            
            # Try graph query if tabular query returned no results
            if result.row_count == 0 and self.graph:
                graph_result = self.executor.execute_graph_query(parsed_query)
                if graph_result.success and graph_result.row_count > 0:
                    result = graph_result
            
            # Generate the response
            response = self.generator.generate(parsed_query, result)
        
        # Add processing time
        response.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Record the conversation turn
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=query,
            parsed_query=parsed_query,
            response=response
        )
        session.add_turn(turn)
        
        return response
    
    def _apply_session_context(self, parsed_query: ParsedQuery, session: ConversationSession) -> ParsedQuery:
        """Apply context from previous conversation turns"""
        # If no entity filters and session has active filters, apply them
        if not parsed_query.entity_filters and session.active_filters:
            from .query_parser import EntityFilter
            for entity_name, values in session.active_filters.items():
                try:
                    entity_type = EntityType[entity_name]
                    parsed_query.entity_filters.append(EntityFilter(
                        entity_type=entity_type,
                        operator='in',
                        values=values
                    ))
                    parsed_query.parse_notes.append(f"Applied context filter: {entity_name}")
                except KeyError:
                    pass
        
        return parsed_query
    
    def _needs_clarification(self, parsed_query: ParsedQuery) -> bool:
        """Determine if the query needs clarification"""
        # Very low confidence
        if parsed_query.confidence < 0.3:
            return True
        
        # Unknown intent with no metrics
        if parsed_query.intent == QueryIntent.UNKNOWN and not parsed_query.primary_metric:
            return True
        
        return False
    
    def _generate_clarification_response(self, query: str, parsed_query: ParsedQuery) -> ConversationalResponse:
        """Generate a clarification request"""
        clarification = []
        
        if not parsed_query.primary_metric:
            clarification.append(self.CLARIFICATION_PROMPTS['no_metric'])
        
        if not parsed_query.group_by:
            clarification.append(self.CLARIFICATION_PROMPTS['no_entity'])
        
        if parsed_query.confidence < 0.3:
            clarification.append(self.CLARIFICATION_PROMPTS['low_confidence'])
        
        response = ConversationalResponse(
            query=query,
            understanding=f"I partially understood your query but need more details.",
            answer="\n\n".join(clarification),
            confidence=parsed_query.confidence,
            follow_up_questions=[
                "Which metric are you interested in?",
                "What time period should I analyze?",
                "Do you want to filter by site, country, or subject?"
            ]
        )
        
        return response
    
    def refine(self, refinement: str, session_id: str = None) -> ConversationalResponse:
        """
        Refine the previous query with additional constraints
        
        Args:
            refinement: Additional constraints or modifications
            session_id: Optional session ID
            
        Returns:
            Updated ConversationalResponse
        """
        session = self.sessions.get(session_id) or self.current_session
        
        if not session or not session.history:
            return self.ask(refinement, session_id)
        
        # Get the last query
        last_turn = session.history[-1]
        
        # Combine with refinement
        combined_query = f"{last_turn.user_query}, {refinement}"
        
        return self.ask(combined_query, session_id)
    
    def _generate_narrative_response(self, query: str, parsed_query: ParsedQuery) -> ConversationalResponse:
        """Generate a response for narrative generation requests"""
        try:
            # Import the narrative generator
            from ..narratives.patient_narrative_generator import PatientNarrativeGenerator
            
            # Initialize narrative generator
            narrative_gen = PatientNarrativeGenerator()
            
            # Extract patient/subject information from query or use defaults
            patient_id = None
            # Look for subject/patient mentions in entity filters
            for ef in parsed_query.entity_filters:
                if ef.entity_type == EntityType.SUBJECT:
                    patient_id = ef.values[0] if ef.values else None
                    break
            
            # Generate narrative
            if patient_id:
                narrative_obj = narrative_gen.generate_narrative(patient_id)
                narrative = narrative_obj.narrative_text
            else:
                # Generate a sample narrative if no specific patient mentioned
                narrative_obj = narrative_gen.generate_narrative("SAMPLE_PATIENT_001")
                narrative = narrative_obj.narrative_text
            
            response = ConversationalResponse(
                query=query,
                understanding="Generating a patient safety narrative based on clinical data",
                answer=f"Here's the generated patient safety narrative:\n\n{narrative}",
                confidence=0.9,
                follow_up_questions=[
                    "Would you like me to generate narratives for specific patients?",
                    "Do you need RBM reports for site monitoring?",
                    "Would you like to analyze safety trends across the study?"
                ]
            )
            
            # Add narrative-specific insights
            from .insight_generator import Insight
            response.insights = [
                Insight(
                    category="narrative",
                    title="Patient Safety Narrative Generated",
                    description="Automated narrative generation completed with clinical data integration and compliance markings.",
                    severity="info",
                    next_steps=[
                        "Review narrative for clinical accuracy",
                        "Add to patient safety database",
                        "Share with medical monitor if required"
                    ]
                )
            ]
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return ConversationalResponse(
                query=query,
                understanding="Narrative generation request",
                answer=f"I encountered an error while generating the patient safety narrative: {str(e)}. Please try again or contact support.",
                confidence=0.0
            )
    
    def _generate_rbm_response(self, query: str, parsed_query: ParsedQuery) -> ConversationalResponse:
        """Generate a response for RBM report requests"""
        try:
            # Import the RBM report generator
            from ..narratives.rbm_report_generator import RBMReportGenerator
            
            # Initialize RBM generator
            rbm_gen = RBMReportGenerator()
            
            # Extract site information from query or use defaults
            site_id = None
            # Look for site mentions in entity filters
            for ef in parsed_query.entity_filters:
                if ef.entity_type == EntityType.SITE:
                    site_id = ef.values[0] if ef.values else None
                    break
            
            # Generate RBM report
            if site_id:
                report_obj = rbm_gen.generate_monitoring_report(site_id)
                report = report_obj.to_markdown()
            else:
                # Generate a sample report if no specific site mentioned
                report_obj = rbm_gen.generate_monitoring_report("SAMPLE_SITE_001")
                report = report_obj.to_markdown()
            
            response = ConversationalResponse(
                query=query,
                understanding="Generating a Risk-Based Monitoring report for site oversight",
                answer=f"Here's the generated RBM monitoring report:\n\n{report}",
                confidence=0.9,
                follow_up_questions=[
                    "Would you like RBM reports for specific sites?",
                    "Do you need patient safety narratives?",
                    "Would you like to analyze monitoring trends across sites?"
                ]
            )
            
            # Add RBM-specific insights
            from .insight_generator import Insight
            response.insights = [
                Insight(
                    category="monitoring",
                    title="RBM Report Generated",
                    description="Risk-based monitoring report completed with prioritized findings and recommended actions.",
                    severity="info",
                    next_steps=[
                        "Review high-risk findings",
                        "Schedule follow-up visits as needed",
                        "Update monitoring plan based on findings"
                    ]
                )
            ]
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating RBM report: {e}")
            return ConversationalResponse(
                query=query,
                understanding="RBM report generation request",
                answer=f"I encountered an error while generating the RBM monitoring report: {str(e)}. Please try again or contact support.",
                confidence=0.0
            )
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of available data sources"""
        summary = {
            'sources': {},
            'total_records': 0
        }
        
        for name, df in self.data_sources.items():
            summary['sources'][name] = {
                'rows': len(df),
                'columns': df.columns.tolist(),
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist()
            }
            summary['total_records'] += len(df)
        
        if self.graph:
            summary['graph'] = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges()
            }
        
        return summary
    
    def export_session(self, session_id: str = None) -> str:
        """Export conversation session as JSON"""
        session = self.sessions.get(session_id) or self.current_session
        
        if not session:
            return json.dumps({'error': 'No session found'})
        
        export_data = {
            'session': session.to_dict(),
            'turns': [turn.to_dict() for turn in session.history]
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def suggest_queries(self) -> List[str]:
        """Suggest example queries based on available data"""
        suggestions = []
        
        # Based on data sources
        if 'cpid' in self.data_sources or any('cpid' in k.lower() for k in self.data_sources.keys()):
            suggestions.extend([
                "Show me sites with the highest missing visits",
                "Which sites have open queries trending up?",
                "Find anomalies in data quality across sites"
            ])
        
        if 'esae' in self.data_sources or any('sae' in k.lower() for k in self.data_sources.keys()):
            suggestions.extend([
                "Show me SAE summary by preferred term",
                "Which subjects have unresolved SAEs older than 30 days?",
                "What's the SAE reporting timeline?"
            ])
        
        if 'visit' in str(self.data_sources.keys()).lower():
            suggestions.extend([
                "Show me visits that are overdue",
                "What's the visit compliance rate by site?"
            ])
        
        # Default suggestions
        if not suggestions:
            suggestions = [
                "Show me top 10 sites by data quality",
                "What are the trending metrics this month?",
                "Find correlations between key metrics"
            ]
        
        return suggestions[:5]
    
    def get_help(self, topic: str = None) -> str:
        """Get help on using the conversational interface"""
        if topic and topic.lower() in self._knowledge_base.get('clinical_terminology', {}):
            term_def = self._knowledge_base['clinical_terminology'][topic.upper()]
            return f"**{topic.upper()}**: {term_def}"
        
        help_text = """
# Natural Language Query Help

## How to Ask Questions

I understand natural language queries about your clinical trial data. Here are some examples:

### Trend Analysis
- "Show me sites where missing visits is trending up over the last 3 snapshots"
- "What's the trend in open queries this month?"

### Rankings
- "Which are the top 5 sites by open queries?"
- "Show me the worst performing sites"

### Filtering
- "Show me all subjects in the US with missing visits"
- "List sites in EMEA with SSM below 70"

### Correlations
- "What correlates with high query counts?"
- "Find relationships between missing visits and data quality"

### Safety
- "Show me SAE summary"
- "Which SAEs are overdue?"

## Available Metrics
- Missing Visits, Missing Pages
- Open Queries, Query Aging
- SAE Count, Protocol Deviations
- Data Quality Index (DQI)
- Site Status Metric (SSM)
- Frozen/Locked CRFs

## Tips
- Be specific about the metric you're interested in
- Mention the time period if relevant
- Specify country or site filters as needed
        """
        return help_text


# Convenience function for quick usage
def create_engine(data_sources: Dict[str, pd.DataFrame] = None, graph=None) -> ConversationalEngine:
    """
    Create a ConversationalEngine instance with data sources
    
    Args:
        data_sources: Dictionary of DataFrames
        graph: Optional NetworkX graph
        
    Returns:
        Configured ConversationalEngine
    """
    engine = ConversationalEngine()
    if data_sources:
        engine.load_data(data_sources, graph)
    return engine
