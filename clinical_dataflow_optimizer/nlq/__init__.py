"""
Natural Language Querying (NLQ) Module
======================================

Conversational Insight Engine powered by RAG (Retrieval-Augmented Generation)
for the Data Quality Team (DQT) and Medical Monitors.

Allows stakeholders to "talk" to their data using natural language queries
instead of writing SQL or SAS programs.

Components:
- QueryParser: Parses natural language to identify intent, entities, metrics
- QueryExecutor: Converts NL queries to Graph/SQL queries against Neural Mesh
- InsightGenerator: RAG-powered response generation with correlations
- ConversationalEngine: Main orchestration layer
- EnhancedQueryParser: LLM-powered query parsing with entity extraction
- EnhancedRAGQueryExecutor: RAG-enhanced query execution

Example Usage:
    >>> from nlq import ConversationalEngine
    >>> engine = ConversationalEngine(data_sources={'cpid': cpid_df})
    >>> response = engine.ask("Show me sites where missing visits is trending up")
    >>> print(response.to_markdown())
    
    # Or use the enhanced one-shot function:
    >>> from nlq import query_clinical_data
    >>> result = query_clinical_data("What sites have high missing visits?", {'cpid': cpid_df})
"""

from .query_parser import QueryParser, QueryIntent, ParsedQuery, MetricType, EntityType
from .query_executor import QueryExecutor, QueryResult
from .insight_generator import InsightGenerator, ConversationalResponse, Insight, InsightContext
from .conversational_engine import ConversationalEngine

# Enhanced NLQ components
from .enhanced_nlq_processor import (
    EnhancedQueryParser,
    EnhancedRAGQueryExecutor,
    ExtractedEntity,
    query_clinical_data
)

__all__ = [
    # Original components
    'QueryParser',
    'QueryIntent', 
    'ParsedQuery',
    'MetricType',
    'EntityType',
    'QueryExecutor',
    'QueryResult',
    'InsightGenerator',
    'InsightContext',
    'Insight',
    'ConversationalResponse',
    'ConversationalEngine',
    
    # Enhanced components
    'EnhancedQueryParser',
    'EnhancedRAGQueryExecutor',
    'ExtractedEntity',
    'query_clinical_data'
]
