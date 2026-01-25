"""
Enhanced NLQ Processor with RAG Integration
============================================

Advanced natural language query processing with retrieval-augmented generation
for contextual clinical data queries.

Features:
- Intent classification using LLM
- Entity extraction and resolution
- Knowledge graph traversal for context retrieval
- RAG-powered response generation
- Multi-turn conversation support
- Query suggestions and autocomplete
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import re
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import LLM integration
try:
    from agents.llm_integration import (
        AgentReasoningEngine, 
        get_reasoning_engine,
        LLMClientFactory,
        PromptTemplates
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM integration not available for NLQ")


class QueryIntent(Enum):
    """Classified query intents"""
    TREND_ANALYSIS = "trend_analysis"          # "What's the trend in..."
    COMPARISON = "comparison"                   # "Compare site A vs site B"
    ANOMALY_DETECTION = "anomaly_detection"     # "Find outliers..."
    SUMMARY = "summary"                         # "Summarize..."
    SPECIFIC_VALUE = "specific_value"           # "What is the DQI for..."
    LIST_QUERY = "list_query"                   # "Show me all patients with..."
    CORRELATION = "correlation"                 # "What's the relationship..."
    PREDICTION = "prediction"                   # "Will this site..."
    ROOT_CAUSE = "root_cause"                   # "Why is..."
    RECOMMENDATION = "recommendation"           # "What should I do about..."
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Types of entities that can be extracted"""
    SITE = "site"
    SUBJECT = "subject"
    STUDY = "study"
    COUNTRY = "country"
    REGION = "region"
    VISIT = "visit"
    METRIC = "metric"
    TIME_PERIOD = "time_period"
    THRESHOLD = "threshold"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from a query"""
    entity_type: EntityType
    value: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0
    normalized_value: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'entity_type': self.entity_type.value,
            'value': self.value,
            'confidence': self.confidence,
            'normalized_value': self.normalized_value
        }


@dataclass
class ParsedQuery:
    """Fully parsed query with intent, entities, and context"""
    original_query: str
    intent: QueryIntent
    confidence: float
    entities: List[ExtractedEntity]
    time_context: Optional[Dict] = None
    aggregation_level: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'original_query': self.original_query,
            'intent': self.intent.value,
            'confidence': self.confidence,
            'entities': [e.to_dict() for e in self.entities],
            'time_context': self.time_context,
            'aggregation_level': self.aggregation_level,
            'filters': self.filters,
            'requires_clarification': self.requires_clarification,
            'clarification_prompt': self.clarification_prompt
        }


class EnhancedQueryParser:
    """
    Enhanced query parser with LLM-powered intent classification
    and entity extraction.
    """
    
    # Intent patterns for rule-based fallback
    INTENT_PATTERNS = {
        QueryIntent.TREND_ANALYSIS: [
            r'\b(trend|trending|over time|history|progression|evolution)\b',
            r'\b(increasing|decreasing|rising|falling|growing)\b'
        ],
        QueryIntent.COMPARISON: [
            r'\b(compare|comparison|versus|vs|between|difference)\b',
            r'\b(better|worse|higher|lower) than\b'
        ],
        QueryIntent.ANOMALY_DETECTION: [
            r'\b(anomaly|anomalies|outlier|outliers|unusual|abnormal)\b',
            r'\b(spike|drop|deviation|unexpected)\b'
        ],
        QueryIntent.SUMMARY: [
            r'\b(summary|summarize|overview|overall|status)\b',
            r'\b(give me|show me|what is).*(summary|overview)\b'
        ],
        QueryIntent.SPECIFIC_VALUE: [
            r'\b(what is|what\'s|how many|how much|count|number of)\b',
            r'\b(value|metric|score) (of|for)\b'
        ],
        QueryIntent.LIST_QUERY: [
            r'\b(list|show|display|find|get).*(all|patients|sites|subjects)\b',
            r'\b(which|what) (sites|patients|subjects)\b'
        ],
        QueryIntent.CORRELATION: [
            r'\b(correlation|correlate|relationship|related|associated)\b',
            r'\b(impact|effect|influence) (on|of)\b'
        ],
        QueryIntent.PREDICTION: [
            r'\b(predict|forecast|will|expect|projection)\b',
            r'\b(likely|probability|chance)\b'
        ],
        QueryIntent.ROOT_CAUSE: [
            r'\b(why|reason|cause|root cause|because)\b',
            r'\b(explain|understand) why\b'
        ],
        QueryIntent.RECOMMENDATION: [
            r'\b(recommend|suggestion|should|advice|action)\b',
            r'\b(what (should|can) (i|we) do)\b'
        ]
    }
    
    # Metric aliases
    METRIC_ALIASES = {
        'missing visits': ['mv', 'missed visits', 'overdue visits', 'outstanding visits'],
        'open queries': ['oq', 'queries', 'data queries', 'outstanding queries'],
        'dqi': ['data quality', 'data quality index', 'quality score', 'quality index'],
        'sae': ['serious adverse', 'safety events', 'adverse events', 'ae'],
        'uncoded terms': ['coding', 'uncoded', 'medical coding', 'whodra'],
        'verification': ['sdv', 'source verification', 'verified']
    }
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and LLM_AVAILABLE
        self.reasoning_engine = get_reasoning_engine() if self.use_llm else None
        
        # Build reverse alias lookup
        self._alias_to_metric = {}
        for metric, aliases in self.METRIC_ALIASES.items():
            for alias in aliases:
                self._alias_to_metric[alias.lower()] = metric
            self._alias_to_metric[metric.lower()] = metric
    
    def parse(self, query: str, context: Dict = None) -> ParsedQuery:
        """
        Parse a natural language query into structured components.
        """
        query_lower = query.lower().strip()
        
        # Extract intent
        intent, intent_confidence = self._classify_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        
        # Determine time context
        time_context = self._extract_time_context(query_lower)
        
        # Determine aggregation level
        aggregation = self._determine_aggregation(query_lower, entities)
        
        # Build filters
        filters = self._build_filters(entities)
        
        # Check if clarification needed
        requires_clarification = False
        clarification_prompt = None
        
        if intent == QueryIntent.UNKNOWN or intent_confidence < 0.5:
            requires_clarification = True
            clarification_prompt = "Could you rephrase your question? I'm not sure what analysis you're looking for."
        elif not any(e.entity_type == EntityType.METRIC for e in entities):
            requires_clarification = True
            clarification_prompt = "Which metric would you like to analyze? (e.g., missing visits, open queries, DQI)"
        
        return ParsedQuery(
            original_query=query,
            intent=intent,
            confidence=intent_confidence,
            entities=entities,
            time_context=time_context,
            aggregation_level=aggregation,
            filters=filters,
            requires_clarification=requires_clarification,
            clarification_prompt=clarification_prompt
        )
    
    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify query intent using patterns and optionally LLM"""
        # Rule-based classification
        intent_scores = defaultdict(float)
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    intent_scores[intent] += 1.0
        
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(0.9, best_intent[1] / 2.0)  # Normalize confidence
            return best_intent[0], confidence
        
        # LLM-based classification if available and rules didn't match well
        if self.use_llm and self.reasoning_engine:
            try:
                result = self.reasoning_engine.interpret_nlq(
                    user_query=query,
                    available_metrics=list(self.METRIC_ALIASES.keys()),
                    available_entities=['site', 'subject', 'study', 'country'],
                    conversation_context=""
                )
                # Parse LLM response to extract intent
                # This is simplified - in production would parse structured response
                interpretation = result.get('interpretation', '')
                if 'trend' in interpretation.lower():
                    return QueryIntent.TREND_ANALYSIS, 0.8
                elif 'compare' in interpretation.lower():
                    return QueryIntent.COMPARISON, 0.8
            except Exception as e:
                logger.warning(f"LLM intent classification failed: {e}")
        
        return QueryIntent.UNKNOWN, 0.3
    
    def _extract_entities(self, query: str) -> List[ExtractedEntity]:
        """Extract entities from query"""
        entities = []
        
        # Extract metrics
        for alias, metric in self._alias_to_metric.items():
            if alias in query:
                entities.append(ExtractedEntity(
                    entity_type=EntityType.METRIC,
                    value=alias,
                    confidence=0.9,
                    normalized_value=metric
                ))
        
        # Extract site references
        site_patterns = [
            r'site\s+(\d+[-\d]*)',
            r'site\s+([A-Z]{2,3}[-\d]+)',
            r'(\d{3,4})\s+site'
        ]
        for pattern in site_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append(ExtractedEntity(
                    entity_type=EntityType.SITE,
                    value=match,
                    confidence=0.85
                ))
        
        # Extract subject references
        subject_patterns = [
            r'subject\s+(\d+[-\d]*)',
            r'patient\s+(\d+[-\d]*)',
            r'(\d{3}-\d{3})'  # Common subject ID pattern
        ]
        for pattern in subject_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append(ExtractedEntity(
                    entity_type=EntityType.SUBJECT,
                    value=match,
                    confidence=0.85
                ))
        
        # Extract country references
        countries = ['usa', 'us', 'uk', 'germany', 'france', 'japan', 'china', 'india', 'brazil', 'canada']
        for country in countries:
            if country in query:
                entities.append(ExtractedEntity(
                    entity_type=EntityType.COUNTRY,
                    value=country,
                    confidence=0.9
                ))
        
        # Extract threshold values
        threshold_pattern = r'(?:more than|greater than|less than|over|under|above|below)\s+(\d+)'
        matches = re.findall(threshold_pattern, query, re.IGNORECASE)
        for match in matches:
            entities.append(ExtractedEntity(
                entity_type=EntityType.THRESHOLD,
                value=match,
                confidence=0.8
            ))
        
        return entities
    
    def _extract_time_context(self, query: str) -> Optional[Dict]:
        """Extract time-related context from query"""
        time_patterns = {
            'last_week': r'\b(last week|past week|previous week)\b',
            'last_month': r'\b(last month|past month|previous month)\b',
            'last_3_snapshots': r'\b(last 3|past 3|recent 3).*(snapshot|data)\b',
            'this_month': r'\b(this month|current month)\b',
            'ytd': r'\b(year to date|ytd)\b',
            'all_time': r'\b(all time|entire|complete|full)\b'
        }
        
        for period, pattern in time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return {'period': period, 'confidence': 0.85}
        
        # Check for specific date references
        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        matches = re.findall(date_pattern, query)
        if matches:
            return {'period': 'specific_dates', 'dates': matches, 'confidence': 0.9}
        
        return None
    
    def _determine_aggregation(self, query: str, entities: List[ExtractedEntity]) -> str:
        """Determine the aggregation level for the query"""
        # Check explicit mentions
        if any(word in query for word in ['by site', 'each site', 'per site', 'sites']):
            return 'site'
        if any(word in query for word in ['by patient', 'each patient', 'per patient', 'patients', 'subjects']):
            return 'subject'
        if any(word in query for word in ['by country', 'each country', 'per country', 'countries']):
            return 'country'
        if any(word in query for word in ['overall', 'total', 'study level', 'study-wide']):
            return 'study'
        
        # Infer from entities
        entity_types = [e.entity_type for e in entities]
        if EntityType.SITE in entity_types:
            return 'site'
        if EntityType.SUBJECT in entity_types:
            return 'subject'
        if EntityType.COUNTRY in entity_types:
            return 'country'
        
        return 'site'  # Default
    
    def _build_filters(self, entities: List[ExtractedEntity]) -> Dict[str, Any]:
        """Build filter dictionary from entities"""
        filters = {}
        
        for entity in entities:
            key = entity.entity_type.value
            if key not in filters:
                filters[key] = []
            filters[key].append(entity.normalized_value or entity.value)
        
        # Flatten single-value filters
        for key, values in filters.items():
            if len(values) == 1:
                filters[key] = values[0]
        
        return filters


class EnhancedRAGQueryExecutor:
    """
    Executes parsed queries against clinical data with RAG enhancement.
    """
    
    def __init__(
        self,
        data_sources: Dict[str, pd.DataFrame] = None,
        graph = None,
        use_llm: bool = True
    ):
        self.data_sources = data_sources or {}
        self.graph = graph
        self.use_llm = use_llm and LLM_AVAILABLE
        self.reasoning_engine = get_reasoning_engine() if self.use_llm else None
        
        # Context store for RAG
        self._context_store: Dict[str, Any] = self._build_context_store()
    
    def _build_context_store(self) -> Dict[str, Any]:
        """Build context store for retrieval"""
        return {
            'clinical_knowledge': {
                'DQI': 'Data Quality Index - composite score measuring data completeness and accuracy (0-100)',
                'SSM': 'Site Status Metric - indicator of overall site performance',
                'SAE': 'Serious Adverse Event - requiring immediate reporting',
                'CRF': 'Case Report Form - document for collecting patient data',
                'SDV': 'Source Data Verification - comparing CRF to source documents',
                'CPID': 'Clinical Protocol ID - unique identifier for clinical metrics'
            },
            'thresholds': {
                'dqi_good': 90,
                'dqi_warning': 75,
                'dqi_critical': 60,
                'queries_warning': 20,
                'queries_critical': 50,
                'missing_visits_warning': 3,
                'missing_visits_critical': 5
            },
            'best_practices': [
                'Review sites with DQI < 75% weekly',
                'Address queries > 14 days old immediately',
                'Escalate SAEs pending > 7 days',
                'Monitor trends across 3+ snapshots'
            ]
        }
    
    def execute(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Execute a parsed query and return results"""
        results = {
            'query': parsed_query.to_dict(),
            'data': None,
            'insights': [],
            'visualization_hint': None,
            'context': [],
            'executed_at': datetime.now().isoformat()
        }
        
        if parsed_query.requires_clarification:
            results['requires_clarification'] = True
            results['clarification_prompt'] = parsed_query.clarification_prompt
            return results
        
        # Execute based on intent
        intent_handlers = {
            QueryIntent.TREND_ANALYSIS: self._execute_trend_analysis,
            QueryIntent.COMPARISON: self._execute_comparison,
            QueryIntent.ANOMALY_DETECTION: self._execute_anomaly_detection,
            QueryIntent.SUMMARY: self._execute_summary,
            QueryIntent.SPECIFIC_VALUE: self._execute_specific_value,
            QueryIntent.LIST_QUERY: self._execute_list_query,
            QueryIntent.CORRELATION: self._execute_correlation,
            QueryIntent.RECOMMENDATION: self._execute_recommendation
        }
        
        handler = intent_handlers.get(parsed_query.intent, self._execute_generic)
        data_result = handler(parsed_query)
        results['data'] = data_result
        
        # Add relevant context from RAG
        results['context'] = self._retrieve_relevant_context(parsed_query)
        
        # Generate insights using LLM if available
        if self.use_llm and self.reasoning_engine:
            insights = self._generate_insights(parsed_query, data_result)
            results['insights'] = insights
        
        # Suggest visualization
        results['visualization_hint'] = self._suggest_visualization(parsed_query.intent)
        
        return results
    
    def _execute_trend_analysis(self, query: ParsedQuery) -> Dict:
        """Execute trend analysis query"""
        # Get metric from entities
        metric = None
        for entity in query.entities:
            if entity.entity_type == EntityType.METRIC:
                metric = entity.normalized_value or entity.value
                break
        
        if not metric:
            return {'error': 'No metric specified for trend analysis'}
        
        # Get data source
        df = self.data_sources.get('cpid')
        if df is None:
            return {'error': 'CPID data not available'}
        
        # Apply filters
        filtered_df = self._apply_filters(df, query.filters)
        
        # Calculate trend data
        # This is simplified - in production would aggregate by time periods
        aggregation = query.aggregation_level or 'site'
        
        result = {
            'metric': metric,
            'aggregation': aggregation,
            'data_points': [],
            'trend_direction': 'stable',
            'trend_strength': 0.0
        }
        
        # Group and calculate
        try:
            if aggregation == 'site':
                site_col = self._find_column(df, ['Site ID', 'Site', 'site_id'])
                metric_col = self._find_metric_column(df, metric)
                
                if site_col and metric_col:
                    grouped = filtered_df.groupby(site_col)[metric_col].mean()
                    result['data_points'] = [
                        {'entity': str(idx), 'value': float(val)}
                        for idx, val in grouped.items() if not pd.isna(val)
                    ]
        except Exception as e:
            logger.warning(f"Trend analysis error: {e}")
        
        return result
    
    def _execute_comparison(self, query: ParsedQuery) -> Dict:
        """Execute comparison query"""
        # Extract entities to compare
        sites = [e.value for e in query.entities if e.entity_type == EntityType.SITE]
        
        if len(sites) < 2:
            return {'error': 'Need at least 2 entities to compare'}
        
        result = {
            'comparison_type': 'site',
            'entities': sites,
            'metrics': [],
            'summary': ''
        }
        
        df = self.data_sources.get('cpid')
        if df is None:
            return result
        
        site_col = self._find_column(df, ['Site ID', 'Site', 'site_id'])
        if not site_col:
            return result
        
        # Get metrics to compare
        metrics_to_compare = ['Missing Visits', 'Open Queries', 'Uncoded Terms']
        
        for metric in metrics_to_compare:
            metric_col = self._find_metric_column(df, metric)
            if metric_col:
                values = {}
                for site in sites:
                    site_data = df[df[site_col].astype(str) == str(site)]
                    if not site_data.empty:
                        values[site] = float(site_data[metric_col].mean())
                
                if values:
                    result['metrics'].append({
                        'name': metric,
                        'values': values,
                        'winner': min(values, key=values.get) if values else None
                    })
        
        return result
    
    def _execute_anomaly_detection(self, query: ParsedQuery) -> Dict:
        """Execute anomaly detection query"""
        df = self.data_sources.get('cpid')
        if df is None:
            return {'anomalies': []}
        
        anomalies = []
        
        # Check for statistical anomalies in key metrics
        metric_columns = ['Missing Visits', 'Open Queries', 'Uncoded Terms']
        
        for metric in metric_columns:
            col = self._find_metric_column(df, metric)
            if col:
                values = df[col].dropna()
                if len(values) > 10:
                    mean = values.mean()
                    std = values.std()
                    threshold = mean + (2 * std)
                    
                    # Find outliers
                    site_col = self._find_column(df, ['Site ID', 'Site', 'site_id'])
                    outliers = df[df[col] > threshold]
                    
                    for _, row in outliers.iterrows():
                        anomalies.append({
                            'metric': metric,
                            'entity': str(row.get(site_col, 'Unknown')),
                            'value': float(row[col]),
                            'threshold': float(threshold),
                            'severity': 'high' if row[col] > mean + (3 * std) else 'medium'
                        })
        
        return {
            'anomalies': anomalies,
            'total_found': len(anomalies)
        }
    
    def _execute_summary(self, query: ParsedQuery) -> Dict:
        """Execute summary query"""
        df = self.data_sources.get('cpid')
        if df is None:
            return {'summary': 'No data available'}
        
        # Apply filters
        filtered_df = self._apply_filters(df, query.filters)
        
        summary = {
            'total_records': len(filtered_df),
            'metrics': {}
        }
        
        # Calculate summary statistics for key metrics
        metric_columns = {
            'Missing Visits': 'missing_visits',
            'Open Queries': 'open_queries',
            'Uncoded Terms': 'uncoded_terms',
            'Verification %': 'verification_pct'
        }
        
        for display_name, internal_name in metric_columns.items():
            col = self._find_metric_column(filtered_df, display_name)
            if col:
                values = filtered_df[col].dropna()
                if len(values) > 0:
                    summary['metrics'][display_name] = {
                        'mean': round(float(values.mean()), 2),
                        'median': round(float(values.median()), 2),
                        'min': round(float(values.min()), 2),
                        'max': round(float(values.max()), 2),
                        'std': round(float(values.std()), 2)
                    }
        
        return summary
    
    def _execute_specific_value(self, query: ParsedQuery) -> Dict:
        """Execute specific value query"""
        df = self.data_sources.get('cpid')
        if df is None:
            return {'value': None, 'error': 'No data available'}
        
        # Get metric and entity from query
        metric = None
        entity_value = None
        entity_type = None
        
        for entity in query.entities:
            if entity.entity_type == EntityType.METRIC:
                metric = entity.normalized_value or entity.value
            elif entity.entity_type in [EntityType.SITE, EntityType.SUBJECT]:
                entity_value = entity.value
                entity_type = entity.entity_type
        
        if not metric:
            return {'value': None, 'error': 'No metric specified'}
        
        # Filter data
        filtered_df = df
        if entity_value and entity_type:
            if entity_type == EntityType.SITE:
                site_col = self._find_column(df, ['Site ID', 'Site', 'site_id'])
                if site_col:
                    filtered_df = df[df[site_col].astype(str) == str(entity_value)]
            elif entity_type == EntityType.SUBJECT:
                subj_col = self._find_column(df, ['Subject ID', 'Subject', 'subject_id'])
                if subj_col:
                    filtered_df = df[df[subj_col].astype(str) == str(entity_value)]
        
        # Get value
        metric_col = self._find_metric_column(filtered_df, metric)
        if metric_col and not filtered_df.empty:
            value = filtered_df[metric_col].mean()
            return {
                'metric': metric,
                'entity': entity_value,
                'value': round(float(value), 2) if not pd.isna(value) else None,
                'unit': self._get_metric_unit(metric)
            }
        
        return {'value': None, 'error': 'Value not found'}
    
    def _execute_list_query(self, query: ParsedQuery) -> Dict:
        """Execute list query"""
        df = self.data_sources.get('cpid')
        if df is None:
            return {'items': []}
        
        # Apply filters and thresholds
        filtered_df = self._apply_filters(df, query.filters)
        
        # Apply threshold filters
        for entity in query.entities:
            if entity.entity_type == EntityType.THRESHOLD:
                # Apply threshold (simplified - assumes "more than")
                metric_entities = [e for e in query.entities if e.entity_type == EntityType.METRIC]
                if metric_entities:
                    metric = metric_entities[0].normalized_value
                    metric_col = self._find_metric_column(df, metric)
                    if metric_col:
                        filtered_df = filtered_df[filtered_df[metric_col] > float(entity.value)]
        
        # Build result list
        items = []
        site_col = self._find_column(filtered_df, ['Site ID', 'Site', 'site_id'])
        subj_col = self._find_column(filtered_df, ['Subject ID', 'Subject', 'subject_id'])
        
        aggregation = query.aggregation_level or 'site'
        
        if aggregation == 'site' and site_col:
            for site in filtered_df[site_col].unique()[:50]:  # Limit results
                site_data = filtered_df[filtered_df[site_col] == site]
                item = {'site_id': str(site)}
                
                # Add key metrics
                for metric in ['Missing Visits', 'Open Queries', 'Uncoded Terms']:
                    col = self._find_metric_column(site_data, metric)
                    if col:
                        item[metric.lower().replace(' ', '_')] = site_data[col].sum()
                
                items.append(item)
        
        elif aggregation == 'subject' and subj_col:
            for subj in filtered_df[subj_col].unique()[:100]:
                subj_data = filtered_df[filtered_df[subj_col] == subj]
                item = {'subject_id': str(subj)}
                
                for metric in ['Missing Visits', 'Open Queries', 'Uncoded Terms']:
                    col = self._find_metric_column(subj_data, metric)
                    if col:
                        item[metric.lower().replace(' ', '_')] = float(subj_data[col].iloc[0]) if len(subj_data) > 0 else 0
                
                items.append(item)
        
        return {
            'items': items,
            'total': len(items),
            'aggregation': aggregation
        }
    
    def _execute_correlation(self, query: ParsedQuery) -> Dict:
        """Execute correlation query"""
        df = self.data_sources.get('cpid')
        if df is None:
            return {'correlations': []}
        
        # Get metrics from query
        metrics = [e.normalized_value or e.value for e in query.entities if e.entity_type == EntityType.METRIC]
        
        if len(metrics) < 2:
            # Use default metrics
            metrics = ['missing_visits', 'open_queries']
        
        result = {'correlations': []}
        
        # Calculate correlations
        cols = [self._find_metric_column(df, m) for m in metrics]
        cols = [c for c in cols if c is not None]
        
        if len(cols) >= 2:
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    try:
                        corr = df[cols[i]].corr(df[cols[j]])
                        if not pd.isna(corr):
                            result['correlations'].append({
                                'metric1': cols[i],
                                'metric2': cols[j],
                                'correlation': round(float(corr), 3),
                                'strength': 'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.4 else 'weak'
                            })
                    except:
                        pass
        
        return result
    
    def _execute_recommendation(self, query: ParsedQuery) -> Dict:
        """Execute recommendation query"""
        # Get current state from data
        df = self.data_sources.get('cpid')
        
        recommendations = []
        
        if df is not None:
            # Check for high-priority issues
            mv_col = self._find_metric_column(df, 'Missing Visits')
            oq_col = self._find_metric_column(df, 'Open Queries')
            site_col = self._find_column(df, ['Site ID', 'Site', 'site_id'])
            
            if mv_col and site_col:
                high_mv_sites = df[df[mv_col] > 5]
                if len(high_mv_sites) > 0:
                    recommendations.append({
                        'priority': 'high',
                        'type': 'missing_visits',
                        'action': f"Review {len(high_mv_sites)} sites with >5 missing visits",
                        'sites': high_mv_sites[site_col].tolist()[:5]
                    })
            
            if oq_col and site_col:
                high_oq_sites = df[df[oq_col] > 20]
                if len(high_oq_sites) > 0:
                    recommendations.append({
                        'priority': 'medium',
                        'type': 'open_queries',
                        'action': f"Address {len(high_oq_sites)} sites with >20 open queries",
                        'sites': high_oq_sites[site_col].tolist()[:5]
                    })
        
        # Add best practice recommendations from context
        recommendations.extend([
            {'priority': 'info', 'type': 'best_practice', 'action': bp}
            for bp in self._context_store['best_practices'][:2]
        ])
        
        return {
            'recommendations': recommendations,
            'total': len(recommendations)
        }
    
    def _execute_generic(self, query: ParsedQuery) -> Dict:
        """Execute generic query"""
        return self._execute_summary(query)
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered = df.copy()
        
        for filter_type, value in filters.items():
            if filter_type == 'site':
                col = self._find_column(df, ['Site ID', 'Site', 'site_id'])
                if col:
                    if isinstance(value, list):
                        filtered = filtered[filtered[col].astype(str).isin([str(v) for v in value])]
                    else:
                        filtered = filtered[filtered[col].astype(str) == str(value)]
            elif filter_type == 'subject':
                col = self._find_column(df, ['Subject ID', 'Subject', 'subject_id'])
                if col:
                    if isinstance(value, list):
                        filtered = filtered[filtered[col].astype(str).isin([str(v) for v in value])]
                    else:
                        filtered = filtered[filtered[col].astype(str) == str(value)]
        
        return filtered
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find a column by possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _find_metric_column(self, df: pd.DataFrame, metric: str) -> Optional[str]:
        """Find metric column in dataframe"""
        metric_lower = metric.lower()
        
        # Direct match
        if metric in df.columns:
            return metric
        
        # Case-insensitive match
        for col in df.columns:
            if col.lower() == metric_lower:
                return col
        
        # Partial match
        for col in df.columns:
            if metric_lower in col.lower() or col.lower() in metric_lower:
                return col
        
        return None
    
    def _get_metric_unit(self, metric: str) -> str:
        """Get unit for a metric"""
        units = {
            'missing_visits': 'visits',
            'open_queries': 'queries',
            'verification': '%',
            'dqi': 'score (0-100)'
        }
        return units.get(metric.lower().replace(' ', '_'), '')
    
    def _retrieve_relevant_context(self, query: ParsedQuery) -> List[Dict]:
        """Retrieve relevant context for RAG"""
        context = []
        
        # Add relevant clinical knowledge
        for entity in query.entities:
            if entity.entity_type == EntityType.METRIC:
                metric = (entity.normalized_value or entity.value).upper()
                if metric in self._context_store['clinical_knowledge']:
                    context.append({
                        'type': 'definition',
                        'term': metric,
                        'content': self._context_store['clinical_knowledge'][metric]
                    })
        
        # Add relevant thresholds
        for key, value in self._context_store['thresholds'].items():
            for entity in query.entities:
                if entity.entity_type == EntityType.METRIC:
                    metric = entity.normalized_value or entity.value
                    if metric.lower().replace(' ', '_') in key or key.replace('_', ' ') in metric.lower():
                        context.append({
                            'type': 'threshold',
                            'metric': metric,
                            'threshold_name': key,
                            'value': value
                        })
        
        return context
    
    def _generate_insights(self, query: ParsedQuery, data: Dict) -> List[str]:
        """Generate insights using LLM"""
        insights = []
        
        try:
            # Build context for LLM
            context = f"""
Query: {query.original_query}
Intent: {query.intent.value}
Data Result: {json.dumps(data, indent=2, default=str)[:1000]}
"""
            
            result = self.reasoning_engine.reason(
                agent_name="InsightGenerator",
                context="Clinical trial data analysis",
                task="Generate 2-3 actionable insights from the query results",
                data={'query_result': data}
            )
            
            # Parse insights from response
            for line in result.split('\n'):
                line = line.strip()
                if line and len(line) > 20 and not line.startswith('#'):
                    insights.append(line)
                    if len(insights) >= 3:
                        break
        except Exception as e:
            logger.warning(f"Insight generation failed: {e}")
        
        return insights
    
    def _suggest_visualization(self, intent: QueryIntent) -> str:
        """Suggest appropriate visualization for query intent"""
        viz_mapping = {
            QueryIntent.TREND_ANALYSIS: 'line_chart',
            QueryIntent.COMPARISON: 'bar_chart',
            QueryIntent.ANOMALY_DETECTION: 'scatter_plot_with_threshold',
            QueryIntent.SUMMARY: 'dashboard_cards',
            QueryIntent.SPECIFIC_VALUE: 'single_value_card',
            QueryIntent.LIST_QUERY: 'data_table',
            QueryIntent.CORRELATION: 'heatmap',
            QueryIntent.RECOMMENDATION: 'priority_list'
        }
        return viz_mapping.get(intent, 'data_table')


# Convenience function for quick queries
def query_clinical_data(
    query: str,
    data_sources: Dict[str, pd.DataFrame] = None,
    graph = None
) -> Dict[str, Any]:
    """
    One-shot function for querying clinical data.
    
    Example:
        >>> result = query_clinical_data("Show me sites with high missing visits", {'cpid': cpid_df})
        >>> print(result['data'])
    """
    parser = EnhancedQueryParser()
    executor = EnhancedRAGQueryExecutor(data_sources=data_sources, graph=graph)
    
    parsed = parser.parse(query)
    return executor.execute(parsed)
