"""
Insight Generator - RAG-Powered Natural Language Response Generation
=====================================================================

Takes query results and generates natural language insights with:
- Pattern recognition and correlation discovery
- Contextual recommendations
- Risk assessments and alerts
- Trend narratives
- Actionable next steps
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import pandas as pd
import numpy as np

from .query_parser import ParsedQuery, QueryIntent, MetricType, EntityType
from .query_executor import QueryResult

logger = logging.getLogger(__name__)


@dataclass
class InsightContext:
    """Context for generating insights"""
    study_id: str = "Unknown"
    therapeutic_area: str = "Unknown"
    phase: str = "Unknown"
    total_sites: int = 0
    total_subjects: int = 0
    data_cutoff_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'study_id': self.study_id,
            'therapeutic_area': self.therapeutic_area,
            'phase': self.phase,
            'total_sites': self.total_sites,
            'total_subjects': self.total_subjects,
            'data_cutoff_date': self.data_cutoff_date.isoformat() if self.data_cutoff_date else None
        }


@dataclass
class Insight:
    """A single insight or finding"""
    category: str  # 'finding', 'trend', 'correlation', 'risk', 'recommendation'
    severity: str  # 'info', 'warning', 'critical'
    title: str
    description: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    related_entities: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'category': self.category,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'supporting_data': self.supporting_data,
            'confidence': self.confidence,
            'related_entities': self.related_entities,
            'next_steps': self.next_steps
        }


@dataclass
class ConversationalResponse:
    """Complete response to a natural language query"""
    query: str
    understanding: str  # How we understood the query
    answer: str  # Main answer in natural language
    insights: List[Insight] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    data_summary: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'query': self.query,
            'understanding': self.understanding,
            'answer': self.answer,
            'insights': [i.to_dict() for i in self.insights],
            'visualizations': self.visualizations,
            'follow_up_questions': self.follow_up_questions,
            'data_summary': self.data_summary,
            'confidence': self.confidence,
            'processing_time_ms': self.processing_time_ms
        }
    
    def to_markdown(self) -> str:
        """Format response as readable markdown"""
        lines = []
        lines.append(f"## Query Understanding")
        lines.append(f"> {self.understanding}")
        lines.append("")
        lines.append(f"## Answer")
        lines.append(self.answer)
        lines.append("")
        
        if self.insights:
            lines.append(f"## Key Insights ({len(self.insights)})")
            for i, insight in enumerate(self.insights, 1):
                severity_icon = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🔴'}[insight.severity]
                lines.append(f"\n### {severity_icon} {insight.title}")
                lines.append(insight.description)
                
                if insight.next_steps:
                    lines.append("\n**Recommended Actions:**")
                    for step in insight.next_steps:
                        lines.append(f"- {step}")
        
        if self.follow_up_questions:
            lines.append("\n## You Might Also Want to Know")
            for q in self.follow_up_questions:
                lines.append(f"- {q}")
        
        lines.append(f"\n---")
        lines.append(f"*Confidence: {self.confidence:.1%} | Processing: {self.processing_time_ms:.0f}ms*")
        
        return "\n".join(lines)


class InsightGenerator:
    """
    Generates natural language insights from query results
    
    Uses pattern matching, statistical analysis, and domain knowledge
    to create actionable insights from clinical data.
    """
    
    # Clinical domain knowledge for contextualization
    METRIC_THRESHOLDS = {
        MetricType.MISSING_VISITS: {'warning': 5, 'critical': 10},
        MetricType.MISSING_PAGES: {'warning': 10, 'critical': 25},
        MetricType.OPEN_QUERIES: {'warning': 20, 'critical': 50},
        MetricType.UNCODED_TERMS: {'warning': 10, 'critical': 30},
        MetricType.SAE_COUNT: {'warning': 3, 'critical': 5},
        MetricType.DAYS_OUTSTANDING: {'warning': 14, 'critical': 30},
        MetricType.QUERY_AGING: {'warning': 14, 'critical': 30},
    }
    
    # Answer templates by intent
    ANSWER_TEMPLATES = {
        QueryIntent.TREND_ANALYSIS: {
            'up': "Based on the analysis, **{count} {entity_type}** show an **upward trend** in {metric}. {top_entities} are the most notable, with increases of {top_changes}.",
            'down': "Based on the analysis, **{count} {entity_type}** show a **downward trend** in {metric}. {top_entities} have improved the most, with decreases of {top_changes}.",
            'mixed': "The analysis shows **mixed trends** in {metric}: {up_count} {entity_type} trending up and {down_count} trending down."
        },
        QueryIntent.TOP_N: "Here are the **top {n} {entity_type}** by {metric}:\n\n{ranked_list}",
        QueryIntent.BOTTOM_N: "Here are the **{n} {entity_type} with lowest** {metric}:\n\n{ranked_list}",
        QueryIntent.AGGREGATION: "**Summary Statistics for {metric}:**\n- Total: {total}\n- Average: {mean:.1f}\n- Range: {min} to {max}\n- Records: {count}",
        QueryIntent.FILTER_LIST: "Found **{count} records** matching your criteria:\n\n{summary}",
        QueryIntent.CORRELATION: "Correlation analysis reveals **{count} significant relationships**:\n\n{correlations}",
        QueryIntent.SAFETY_CHECK: "**Safety Summary:**\n\n{summary}\n\n{alerts}",
        QueryIntent.ANOMALY_DETECTION: "Anomaly detection identified **{count} outliers** in {metric}:\n\n{anomalies}"
    }
    
    # Correlation explanations
    CORRELATION_INSIGHTS = {
        ('missing_visits', 'open_queries'): "Sites with more missing visits often have more open queries, suggesting data collection issues may cascade to data quality problems.",
        ('missing_visits', 'frozen_crfs'): "A negative correlation between missing visits and frozen CRFs indicates sites with fewer visits naturally have less data to freeze.",
        ('open_queries', 'days_outstanding'): "Sites with more open queries tend to have longer outstanding days, suggesting resource constraints.",
        ('sae_count', 'query_aging'): "Higher SAE counts correlating with query aging may indicate sites overwhelmed by safety reporting need additional support."
    }
    
    def __init__(self, context: InsightContext = None):
        """Initialize the insight generator"""
        self.context = context or InsightContext()
        self._knowledge_base = self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Build domain knowledge base for RAG"""
        return {
            'metrics': {
                'missing_visits': {
                    'description': 'Visits that were expected but not completed',
                    'impact': 'Affects data completeness and study integrity',
                    'actions': ['Check visit schedule', 'Contact site', 'Review protocol amendments']
                },
                'open_queries': {
                    'description': 'Data queries awaiting site response',
                    'impact': 'Delays database lock and analysis',
                    'actions': ['Prioritize aged queries', 'Schedule site calls', 'Review query text clarity']
                },
                'frozen_crfs': {
                    'description': 'Case Report Forms ready for review',
                    'impact': 'Indicates data cleaning progress',
                    'actions': ['Review for SDV', 'Check for remaining edits', 'Prepare for locking']
                }
            },
            'thresholds': self.METRIC_THRESHOLDS,
            'best_practices': [
                'Review high-risk sites weekly',
                'Address queries older than 14 days',
                'Escalate SAEs immediately',
                'Monitor trends over 3 snapshots minimum'
            ]
        }
    
    def generate(self, parsed_query: ParsedQuery, result: QueryResult) -> ConversationalResponse:
        """
        Generate a conversational response from query results
        
        Args:
            parsed_query: The parsed natural language query
            result: The query execution result
            
        Returns:
            ConversationalResponse with natural language answer and insights
        """
        start_time = datetime.now()
        
        response = ConversationalResponse(
            query=parsed_query.original_query,
            understanding=self._generate_understanding(parsed_query),
            answer=""
        )
        
        if not result.success:
            response.answer = f"I couldn't complete this analysis: {result.error_message}"
            response.confidence = 0.0
            return response
        
        # Generate main answer
        response.answer = self._generate_answer(parsed_query, result)
        
        # Generate insights
        response.insights = self._generate_insights(parsed_query, result)
        
        # Add data summary
        response.data_summary = self._generate_data_summary(result)
        
        # Generate follow-up questions
        response.follow_up_questions = self._generate_follow_ups(parsed_query, result)
        
        # Generate visualization suggestions
        response.visualizations = self._suggest_visualizations(parsed_query, result)
        
        # Calculate confidence
        response.confidence = self._calculate_response_confidence(parsed_query, result)
        
        # Calculate processing time
        response.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return response
    
    def _generate_understanding(self, parsed_query: ParsedQuery) -> str:
        """Generate human-readable query understanding"""
        parts = []
        
        # Intent
        intent_descriptions = {
            QueryIntent.TREND_ANALYSIS: "analyzing trends",
            QueryIntent.TOP_N: f"finding top {parsed_query.top_n or 'N'} items",
            QueryIntent.BOTTOM_N: f"finding bottom {parsed_query.top_n or 'N'} items",
            QueryIntent.AGGREGATION: "calculating aggregate statistics",
            QueryIntent.FILTER_LIST: "filtering and listing data",
            QueryIntent.COMPARISON: "comparing entities",
            QueryIntent.CORRELATION: "finding correlations",
            QueryIntent.SAFETY_CHECK: "checking safety data",
            QueryIntent.ANOMALY_DETECTION: "detecting anomalies",
            QueryIntent.DRILL_DOWN: "drilling down into details",
            QueryIntent.COMPLIANCE_CHECK: "checking compliance",
            QueryIntent.FORECAST: "forecasting future values"
        }
        
        intent_desc = intent_descriptions.get(parsed_query.intent, "analyzing")
        parts.append(f"I understand you want me to help with **{intent_desc}**")
        
        # Metric
        if parsed_query.primary_metric:
            metric_name = parsed_query.primary_metric.value.replace('_', ' ').title()
            parts.append(f"for **{metric_name}**")
        
        # Entity filters
        for ef in parsed_query.entity_filters:
            entity_name = ef.entity_type.name.lower()
            parts.append(f"filtered by {entity_name}: {', '.join(ef.values)}")
        
        # Time constraint
        if parsed_query.time_constraint:
            if parsed_query.time_constraint.type == 'relative':
                parts.append(f"over the last {parsed_query.time_constraint.value} {parsed_query.time_constraint.unit}")
        
        return " ".join(parts) + "."
    
    def _generate_answer(self, parsed_query: ParsedQuery, result: QueryResult) -> str:
        """Generate the main answer based on intent and results"""
        
        if parsed_query.intent == QueryIntent.TREND_ANALYSIS:
            return self._generate_trend_answer(parsed_query, result)
        elif parsed_query.intent == QueryIntent.TOP_N:
            return self._generate_top_n_answer(parsed_query, result, "top")
        elif parsed_query.intent == QueryIntent.BOTTOM_N:
            return self._generate_top_n_answer(parsed_query, result, "bottom")
        elif parsed_query.intent == QueryIntent.AGGREGATION:
            return self._generate_aggregation_answer(parsed_query, result)
        elif parsed_query.intent == QueryIntent.CORRELATION:
            return self._generate_correlation_answer(parsed_query, result)
        elif parsed_query.intent == QueryIntent.SAFETY_CHECK:
            return self._generate_safety_answer(parsed_query, result)
        elif parsed_query.intent == QueryIntent.ANOMALY_DETECTION:
            return self._generate_anomaly_answer(parsed_query, result)
        else:
            return self._generate_list_answer(parsed_query, result)
    
    def _generate_trend_answer(self, parsed_query: ParsedQuery, result: QueryResult) -> str:
        """Generate answer for trend analysis"""
        if not result.trends:
            return f"I analyzed {result.row_count} records but couldn't identify clear trends in the data."
        
        metric_name = parsed_query.primary_metric.value.replace('_', ' ') if parsed_query.primary_metric else "the metric"
        entity_type = parsed_query.group_by[0].name.lower() + 's' if parsed_query.group_by else "entities"
        
        up_trend = next((t for t in result.trends if t.get('direction') == 'up'), {})
        down_trend = next((t for t in result.trends if t.get('direction') == 'down'), {})
        
        lines = []
        
        if up_trend.get('entities'):
            up_count = len(up_trend['entities'])
            top_up = up_trend['entities'][:3]
            lines.append(f"📈 **{up_count} {entity_type}** show an **upward trend** in {metric_name}.")
            lines.append(f"   Most notable: {', '.join(map(str, top_up))}")
        
        if down_trend.get('entities'):
            down_count = len(down_trend['entities'])
            top_down = down_trend['entities'][:3]
            lines.append(f"📉 **{down_count} {entity_type}** show a **downward trend** in {metric_name}.")
            lines.append(f"   Most improved: {', '.join(map(str, top_down))}")
        
        # Check for trending filter
        for mf in parsed_query.metric_filters:
            if mf.operator == 'trending_up':
                if up_trend.get('entities'):
                    lines = [f"Found **{len(up_trend['entities'])} {entity_type}** where {metric_name} is **trending up**:"]
                    for i, (entity, change) in enumerate(zip(up_trend['entities'][:5], up_trend.get('changes', [])[:5]), 1):
                        change_str = f"+{change:.1f}" if isinstance(change, float) else f"+{change}"
                        lines.append(f"  {i}. **{entity}** ({change_str})")
                else:
                    lines = [f"No {entity_type} found with {metric_name} trending up."]
                break
        
        return "\n".join(lines) if lines else "No significant trends detected."
    
    def _generate_top_n_answer(self, parsed_query: ParsedQuery, result: QueryResult, direction: str) -> str:
        """Generate answer for top/bottom N queries"""
        n = parsed_query.top_n or 10
        metric_name = parsed_query.primary_metric.value.replace('_', ' ') if parsed_query.primary_metric else "the metric"
        entity_type = parsed_query.group_by[0].name.lower() + 's' if parsed_query.group_by else "items"
        
        lines = [f"**{direction.title()} {n} {entity_type} by {metric_name}:**\n"]
        
        if result.data is not None and len(result.data) > 0:
            # Try to get the relevant columns
            df = result.data
            if len(df.columns) >= 2:
                for i, row in df.head(n).iterrows():
                    values = list(row.values)
                    entity_val = values[0]
                    metric_val = values[1] if len(values) > 1 else 'N/A'
                    
                    if isinstance(metric_val, float):
                        metric_val = f"{metric_val:.1f}"
                    
                    lines.append(f"{i+1}. **{entity_val}**: {metric_val}")
        else:
            lines.append("No data available for this query.")
        
        return "\n".join(lines)
    
    def _generate_aggregation_answer(self, parsed_query: ParsedQuery, result: QueryResult) -> str:
        """Generate answer for aggregation queries"""
        lines = []
        
        for metric, stats in result.aggregations.items():
            metric_name = metric.replace('_', ' ').title()
            lines.append(f"**{metric_name} Statistics:**")
            lines.append(f"  - Total: {stats.get('sum', 'N/A'):,.0f}")
            lines.append(f"  - Average: {stats.get('mean', 0):.1f}")
            lines.append(f"  - Median: {stats.get('median', 0):.1f}")
            lines.append(f"  - Range: {stats.get('min', 0):.0f} to {stats.get('max', 0):.0f}")
            lines.append(f"  - Count: {stats.get('count', 0):,}")
            lines.append("")
        
        return "\n".join(lines) if lines else f"Analyzed {result.row_count} records."
    
    def _generate_correlation_answer(self, parsed_query: ParsedQuery, result: QueryResult) -> str:
        """Generate answer for correlation analysis"""
        if not result.correlations:
            return "No significant correlations found in the data."
        
        lines = [f"Found **{len(result.correlations)} significant correlations**:\n"]
        
        for corr in result.correlations[:5]:
            var1 = corr['variable_1'].replace('_', ' ').title()
            var2 = corr['variable_2'].replace('_', ' ').title()
            val = corr['correlation']
            strength = corr['strength']
            direction = "positive" if val > 0 else "negative"
            
            lines.append(f"- **{var1}** ↔ **{var2}**: {val:+.2f} ({strength} {direction})")
            
            # Add explanation if available
            key = (corr['variable_1'].lower(), corr['variable_2'].lower())
            if key in self.CORRELATION_INSIGHTS:
                lines.append(f"  *{self.CORRELATION_INSIGHTS[key]}*")
        
        return "\n".join(lines)
    
    def _generate_safety_answer(self, parsed_query: ParsedQuery, result: QueryResult) -> str:
        """Generate answer for safety-related queries"""
        lines = ["**Safety Data Summary:**\n"]
        
        if result.summary:
            lines.append(f"Total safety records: **{result.summary.get('total_safety_records', 0)}**")
        
        if result.alerts:
            lines.append("\n**⚠️ Alerts:**")
            for alert in result.alerts:
                lines.append(f"- {alert}")
        else:
            lines.append("\n✅ No critical safety alerts.")
        
        return "\n".join(lines)
    
    def _generate_anomaly_answer(self, parsed_query: ParsedQuery, result: QueryResult) -> str:
        """Generate answer for anomaly detection"""
        if result.row_count == 0:
            return "No anomalies detected in the data."
        
        metric_name = result.summary.get('metric', 'the metric').replace('_', ' ')
        
        lines = [
            f"**Anomaly Detection Results:**\n",
            f"Found **{result.row_count} outliers** in {metric_name}.",
            f"- Mean: {result.summary.get('mean', 0):.1f}",
            f"- Std Dev: {result.summary.get('std', 0):.1f}",
            f"- Threshold: >{result.summary.get('anomaly_threshold', '2 std')}"
        ]
        
        if result.alerts:
            lines.append("\n" + "\n".join(result.alerts))
        
        return "\n".join(lines)
    
    def _generate_list_answer(self, parsed_query: ParsedQuery, result: QueryResult) -> str:
        """Generate answer for filter/list queries"""
        count = result.row_count
        
        if count == 0:
            return "No records found matching your criteria."
        
        entity_type = parsed_query.group_by[0].name.lower() + 's' if parsed_query.group_by else "records"
        
        lines = [f"Found **{count} {entity_type}** matching your criteria.\n"]
        
        if result.data is not None and len(result.data) > 0:
            # Show first few results
            lines.append("**Sample Results:**")
            for i, (idx, row) in enumerate(result.data.head(5).iterrows()):
                summary = ", ".join([f"{k}: {v}" for k, v in list(row.items())[:3]])
                lines.append(f"  {i+1}. {summary}")
            
            if count > 5:
                lines.append(f"\n*...and {count - 5} more*")
        
        return "\n".join(lines)
    
    def _generate_insights(self, parsed_query: ParsedQuery, result: QueryResult) -> List[Insight]:
        """Generate actionable insights from the results"""
        insights = []
        
        # Check for threshold breaches
        insights.extend(self._check_threshold_insights(parsed_query, result))
        
        # Check for trend-based insights
        insights.extend(self._check_trend_insights(parsed_query, result))
        
        # Check for correlation-based insights
        insights.extend(self._check_correlation_insights(result))
        
        # Add contextual insights
        insights.extend(self._generate_contextual_insights(parsed_query, result))
        
        return insights
    
    def _check_threshold_insights(self, parsed_query: ParsedQuery, result: QueryResult) -> List[Insight]:
        """Check if any metrics breach thresholds"""
        insights = []
        
        if result.data is None:
            return insights
        
        for metric, thresholds in self.METRIC_THRESHOLDS.items():
            metric_col = metric.value if hasattr(metric, 'value') else str(metric)
            
            # Check if this metric is in the data
            matching_cols = [c for c in result.data.columns if metric_col.replace('_', ' ').lower() in c.lower()]
            
            for col in matching_cols:
                values = pd.to_numeric(result.data[col], errors='coerce').dropna()
                
                if len(values) == 0:
                    continue
                
                critical_count = (values > thresholds['critical']).sum()
                warning_count = (values > thresholds['warning']).sum() - critical_count
                
                if critical_count > 0:
                    insights.append(Insight(
                        category='risk',
                        severity='critical',
                        title=f"Critical {col.replace('_', ' ').title()} Threshold Breach",
                        description=f"**{critical_count} records** exceed the critical threshold ({thresholds['critical']}) for {col}. Immediate attention required.",
                        supporting_data={'count': int(critical_count), 'threshold': thresholds['critical']},
                        confidence=0.95,
                        next_steps=[
                            "Review affected records immediately",
                            "Escalate to study team if needed",
                            "Document root cause analysis"
                        ]
                    ))
                
                if warning_count > 0:
                    insights.append(Insight(
                        category='finding',
                        severity='warning',
                        title=f"{col.replace('_', ' ').title()} Warning",
                        description=f"**{warning_count} records** exceed the warning threshold ({thresholds['warning']}) for {col}.",
                        supporting_data={'count': int(warning_count), 'threshold': thresholds['warning']},
                        confidence=0.85,
                        next_steps=["Monitor closely", "Schedule review"]
                    ))
        
        return insights
    
    def _check_trend_insights(self, parsed_query: ParsedQuery, result: QueryResult) -> List[Insight]:
        """Generate insights from trend data"""
        insights = []
        
        if not result.trends:
            return insights
        
        for trend in result.trends:
            direction = trend.get('direction', 'unknown')
            entities = trend.get('entities', [])
            
            if len(entities) > 3 and direction == 'up':
                insights.append(Insight(
                    category='trend',
                    severity='warning',
                    title="Widespread Upward Trend Detected",
                    description=f"**{len(entities)} entities** show an upward trend. This may indicate a systemic issue requiring attention.",
                    supporting_data={'entity_count': len(entities), 'direction': direction},
                    confidence=0.8,
                    related_entities=entities[:5],
                    next_steps=[
                        "Investigate common factors",
                        "Check for protocol changes",
                        "Review site training compliance"
                    ]
                ))
        
        return insights
    
    def _check_correlation_insights(self, result: QueryResult) -> List[Insight]:
        """Generate insights from correlation data"""
        insights = []
        
        if not result.correlations:
            return insights
        
        # Find strong correlations
        strong_corrs = [c for c in result.correlations if c['strength'] == 'strong']
        
        for corr in strong_corrs:
            var1 = corr['variable_1'].replace('_', ' ').title()
            var2 = corr['variable_2'].replace('_', ' ').title()
            val = corr['correlation']
            
            direction = "increases" if val > 0 else "decreases"
            
            insights.append(Insight(
                category='correlation',
                severity='info',
                title=f"Strong Correlation: {var1} & {var2}",
                description=f"Strong correlation ({val:+.2f}) found: as {var1} {direction}, {var2} tends to follow.",
                supporting_data={'correlation': val, 'variables': [var1, var2]},
                confidence=0.9,
                next_steps=["Investigate causal relationship", "Consider for monitoring"]
            ))
        
        return insights
    
    def _generate_contextual_insights(self, parsed_query: ParsedQuery, result: QueryResult) -> List[Insight]:
        """Generate context-aware insights"""
        insights = []
        
        # Data completeness insight
        if result.row_count > 0 and result.data is not None:
            null_pcts = result.data.isnull().mean()
            high_null_cols = null_pcts[null_pcts > 0.2].index.tolist()
            
            if high_null_cols:
                insights.append(Insight(
                    category='finding',
                    severity='warning',
                    title="Data Completeness Issue",
                    description=f"**{len(high_null_cols)} columns** have >20% missing values: {', '.join(high_null_cols[:3])}",
                    supporting_data={'columns': high_null_cols},
                    confidence=0.95,
                    next_steps=["Review data collection process", "Check source system mappings"]
                ))
        
        return insights
    
    def _generate_data_summary(self, result: QueryResult) -> Dict[str, Any]:
        """Generate summary statistics for the data"""
        summary = {
            'row_count': result.row_count,
            'execution_time_ms': result.execution_time_ms
        }
        
        if result.data is not None:
            summary['columns'] = result.data.columns.tolist()
            summary['numeric_columns'] = result.data.select_dtypes(include=[np.number]).columns.tolist()
        
        summary.update(result.summary)
        
        return summary
    
    def _generate_follow_ups(self, parsed_query: ParsedQuery, result: QueryResult) -> List[str]:
        """Generate relevant follow-up questions"""
        follow_ups = []
        
        metric_name = parsed_query.primary_metric.value.replace('_', ' ') if parsed_query.primary_metric else "metrics"
        
        # Intent-based follow-ups
        if parsed_query.intent == QueryIntent.TREND_ANALYSIS:
            follow_ups.append(f"What's causing the trend in {metric_name}?")
            follow_ups.append("Show me the same analysis for the previous period")
            follow_ups.append("Which sites are outliers in this trend?")
        
        elif parsed_query.intent == QueryIntent.TOP_N:
            follow_ups.append(f"What do the bottom performers look like?")
            follow_ups.append(f"How do these compare to the study average?")
        
        elif parsed_query.intent == QueryIntent.SAFETY_CHECK:
            follow_ups.append("Show me SAE details by preferred term")
            follow_ups.append("What's the SAE resolution timeline?")
        
        # Generic follow-ups
        follow_ups.append("Show me the correlation between key metrics")
        follow_ups.append("Drill down into the worst-performing site")
        
        return follow_ups[:4]  # Limit to 4 suggestions
    
    def _suggest_visualizations(self, parsed_query: ParsedQuery, result: QueryResult) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations"""
        suggestions = []
        
        if parsed_query.intent == QueryIntent.TREND_ANALYSIS:
            suggestions.append({
                'type': 'line_chart',
                'title': 'Trend Over Time',
                'description': 'Line chart showing metric changes over snapshots'
            })
        
        elif parsed_query.intent in [QueryIntent.TOP_N, QueryIntent.BOTTOM_N]:
            suggestions.append({
                'type': 'bar_chart',
                'title': 'Ranked Bar Chart',
                'description': 'Horizontal bar chart showing ranked values'
            })
        
        elif parsed_query.intent == QueryIntent.CORRELATION:
            suggestions.append({
                'type': 'heatmap',
                'title': 'Correlation Heatmap',
                'description': 'Matrix showing correlation strengths'
            })
        
        elif parsed_query.intent == QueryIntent.COMPARISON:
            suggestions.append({
                'type': 'grouped_bar',
                'title': 'Comparison Chart',
                'description': 'Grouped bar chart for side-by-side comparison'
            })
        
        # Always suggest table view
        suggestions.append({
            'type': 'data_table',
            'title': 'Data Table',
            'description': 'Sortable table with all results'
        })
        
        return suggestions
    
    def _calculate_response_confidence(self, parsed_query: ParsedQuery, result: QueryResult) -> float:
        """Calculate overall response confidence"""
        confidence = parsed_query.confidence * 0.5  # Query parsing confidence
        
        if result.success:
            confidence += 0.3
        
        if result.row_count > 0:
            confidence += 0.1
        
        if result.data is not None and len(result.data) > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
