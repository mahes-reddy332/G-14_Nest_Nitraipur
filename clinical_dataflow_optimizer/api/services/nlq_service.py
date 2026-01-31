"""
RAG and NLQ Service
Implements Retrieval-Augmented Generation and Natural Language Query processing
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import os
from pathlib import Path

# Import enhanced RAG system
from rag.enhanced_rag_system import EnhancedRAGSystem, QueryType
from core.security import InputValidator, ValidationError

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service for Retrieval-Augmented Generation using Enhanced RAG System
    """

    def __init__(self):
        self.rag_system = None
        self._initialized = False

    async def initialize(self, documents: List[str] = None):
        """Initialize the RAG system with clinical data"""
        if self._initialized:
            return

        try:
            # Get data path from environment or use default
            data_path = os.getenv("CLINICAL_DATA_PATH")
            if not data_path:
                # Use the QC Anonymized Study Files directory (in parent of clinical_dataflow_optimizer)
                data_path = Path(__file__).parent.parent.parent.parent / "QC Anonymized Study Files"

            if not data_path.exists():
                logger.warning(f"Data path {data_path} does not exist, RAG will use fallback knowledge")
                self._initialized = True
                return

            # Initialize enhanced RAG system
            self.rag_system = EnhancedRAGSystem(data_path)

            # Perform one-time ingestion if needed
            ingestion_performed = self.rag_system.initialize()

            if ingestion_performed:
                logger.info("Performed one-time data ingestion for RAG system")
            else:
                logger.info("Using cached knowledge graph for RAG system")

            self._initialized = True
            logger.info("Enhanced RAG system initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            self._initialized = True  # Mark as initialized to prevent repeated attempts

    async def query(self, question: str, query_type: QueryType = None) -> Dict[str, Any]:
        """Process a natural language query using enhanced RAG"""
        if not self.rag_system or not self._initialized:
            return self._fallback_query(question)

        try:
            # Create context with query type if provided
            context = {}
            if query_type:
                context['query_type'] = query_type.value

            # Process query using enhanced RAG system
            response = self.rag_system.query(question, context)

            return {
                "answer": response.get("answer", "No answer available"),
                "sources": self._format_sources(response.get("sources", [])),
                "confidence": response.get("confidence", 0.8),
                "query_type": response.get("query_type", "factual"),
                "insights": response.get("insights", []),
                "recommendations": response.get("recommendations", [])
            }

        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return self._fallback_query(question)

    def _classify_query_type(self, question: str) -> QueryType:
        """Classify the type of query for optimal processing"""
        question_lower = question.lower()

        # Diagnostic queries
        if any(word in question_lower for word in ["why", "problem", "issue", "error", "discrepancy"]):
            return QueryType.DIAGNOSTIC

        # Predictive queries
        elif any(word in question_lower for word in ["predict", "forecast", "risk", "trend", "future"]):
            return QueryType.PREDICTIVE

        # Prescriptive queries
        elif any(word in question_lower for word in ["recommend", "should", "action", "improve", "fix"]):
            return QueryType.PRESCRIPTIVE

        # Analytical queries
        elif any(word in question_lower for word in ["analyze", "compare", "correlation", "pattern", "trend"]):
            return QueryType.ANALYTICAL

        # Explanatory queries
        elif any(word in question_lower for word in ["explain", "how", "what is"]):
            return QueryType.EXPLANATORY

        # Default to factual
        else:
            return QueryType.FACTUAL

    def _fallback_query(self, question: str) -> Dict[str, Any]:
        """Fallback query processing when RAG system is not available"""
        # Enhanced fallback with clinical knowledge
        knowledge_base = self._get_clinical_knowledge_base()

        # Simple keyword matching for common queries
        question_lower = question.lower()
        relevant_info = []

        for category, info in knowledge_base.items():
            if any(keyword in question_lower for keyword in info.get("keywords", [])):
                relevant_info.extend(info.get("responses", []))

        if relevant_info:
            answer = " ".join(relevant_info[:2])  # Limit to first 2 relevant responses
        else:
            answer = "Based on clinical trial best practices, I recommend reviewing data quality metrics and ensuring proper reconciliation between clinical and safety databases."

        return {
            "answer": answer,
            "sources": [{"title": "Clinical Knowledge Base", "content": "Built-in clinical trial knowledge", "type": "knowledge_base"}],
            "confidence": 0.6,
            "query_type": "factual"
        }

    def _format_sources(self, sources: List[Any]) -> List[Dict[str, Any]]:
        """Format sources to ensure they are dictionaries"""
        formatted_sources = []
        for source in sources:
            if isinstance(source, dict):
                formatted_sources.append(source)
            elif isinstance(source, str):
                formatted_sources.append({
                    "title": source,
                    "content": source,
                    "type": "text"
                })
            else:
                formatted_sources.append({
                    "title": str(source),
                    "content": str(source),
                    "type": "unknown"
                })
        return formatted_sources

    def _get_clinical_knowledge_base(self) -> Dict[str, Any]:
        """Get clinical knowledge base for fallback queries"""
        return {
            "data_quality": {
                "keywords": ["quality", "dqi", "missing", "incomplete", "accuracy"],
                "responses": [
                    "Data quality in clinical trials is measured by completeness, accuracy, and timeliness. Key metrics include DQI (Data Quality Index), missing visit rates, and open query aging.",
                    "Common data quality issues include missing visits, incomplete CRF entries, and discrepancies between clinical and safety databases."
                ]
            },
            "safety": {
                "keywords": ["sae", "adverse", "safety", "serious"],
                "responses": [
                    "Serious Adverse Events (SAEs) require immediate reporting and reconciliation between clinical and safety databases.",
                    "SAE reconciliation ensures consistency between EDC systems and safety databases for regulatory compliance."
                ]
            },
            "coding": {
                "keywords": ["coding", "meddra", "whodd", "terminology"],
                "responses": [
                    "Medical coding uses standardized terminologies like MedDRA for adverse events and WHODD for medications.",
                    "Proper coding ensures consistent terminology across clinical trial data for analysis and reporting."
                ]
            },
            "sites": {
                "keywords": ["site", "investigator", "performance", "compliance"],
                "responses": [
                    "Site performance is monitored through metrics like SSM (Site Status Metric), visit compliance, and query resolution rates.",
                    "Site management involves proactive communication, query management, and risk identification."
                ]
            }
        }


class NLQService:
    """
    Natural Language Query service for clinical data
    """

    def __init__(self):
        self.rag_service = RAGService()
        self._initialized = False

    async def initialize(self):
        """Initialize NLQ processing"""
        if self._initialized:
            return

        await self.rag_service.initialize()
        self._initialized = True
        logger.info("NLQ service initialized")

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a natural language query about clinical data"""
        if not self._initialized:
            await self.initialize()

        # Enhance query with context
        validated_query = InputValidator.validate_query(query)
        enhanced_query = self._enhance_query(validated_query, context)

        # Use RAG to get answer
        rag_result = await self.rag_service.query(enhanced_query)

        sources = rag_result.get("sources", [])
        if not sources:
            sources = [
                {
                    "title": "Clinical Knowledge Base",
                    "content": "Generated response using built-in clinical knowledge.",
                    "type": "knowledge_base"
                }
            ]

        # Format response
        return {
            "answer": rag_result["answer"],
            "sources": sources,
            "confidence": rag_result["confidence"],
            "query_type": self._classify_query(query) or "general"
        }

    def _enhance_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Enhance the query with clinical context"""
        enhancements = []

        if context:
            if context.get("study_id"):
                enhancements.append(f"for study {context['study_id']}")
            if context.get("patient_id"):
                enhancements.append(f"for patient {context['patient_id']}")
            if context.get("site_id"):
                enhancements.append(f"for site {context['site_id']}")

        if enhancements:
            return f"{query} {' '.join(enhancements)}"
        return query

    def _classify_query(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["how many", "count", "number of"]):
            return "aggregation"
        elif any(word in query_lower for word in ["what", "which", "who"]):
            return "factual"
        elif any(word in query_lower for word in ["why", "explain", "reason"]):
            return "explanatory"
        elif any(word in query_lower for word in ["trend", "over time", "change"]):
            return "temporal"

    def _format_sources(self, sources: List[Any]) -> List[Dict[str, Any]]:
        """Format sources to ensure they are dictionaries"""
        formatted_sources = []
        for source in sources:
            if isinstance(source, dict):
                formatted_sources.append(source)
            elif isinstance(source, str):
                formatted_sources.append({
                    "title": source,
                    "content": source,
                    "type": "text"
                })
            else:
                formatted_sources.append({
                    "title": str(source),
                    "content": str(source),
                    "type": "unknown"
                })
        return formatted_sources

    async def get_status(self) -> Dict[str, Any]:
        """Get the status of the NLQ service"""
        return {
            "initialized": self._initialized,
            "rag_available": self.rag_service._initialized if self.rag_service else False,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "timestamp": datetime.now().isoformat()
        }