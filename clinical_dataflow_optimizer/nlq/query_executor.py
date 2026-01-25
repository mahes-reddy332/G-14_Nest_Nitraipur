"""
Query Executor - Converts Parsed Queries to Data Operations
============================================================

Takes ParsedQuery objects and executes them against:
- Pandas DataFrames (tabular data)
- NetworkX Graph (patient-centric graph)
- Time-series data (snapshots)

Returns structured QueryResult with data for visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .query_parser import (
    ParsedQuery, QueryIntent, MetricType, EntityType,
    EntityFilter, MetricFilter, TimeConstraint
)

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of executing a parsed query"""
    success: bool
    data: Optional[pd.DataFrame] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    aggregations: Dict[str, Any] = field(default_factory=dict)
    trends: List[Dict[str, Any]] = field(default_factory=list)
    correlations: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    row_count: int = 0
    error_message: Optional[str] = None
    query_executed: str = ""  # The equivalent "SQL-like" query
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'data': self.data.to_dict('records') if self.data is not None else None,
            'summary': self.summary,
            'aggregations': self.aggregations,
            'trends': self.trends,
            'correlations': self.correlations,
            'alerts': self.alerts,
            'execution_time_ms': self.execution_time_ms,
            'row_count': self.row_count,
            'error_message': self.error_message,
            'query_executed': self.query_executed
        }


class QueryExecutor:
    """
    Executes parsed queries against clinical data sources
    
    Converts natural language understanding into actual data operations.
    """
    
    # Column mappings for different data sources
    COLUMN_MAPPINGS = {
        'CPID_EDC_Metrics': {
            'site': ['Site ID', 'Site', 'SiteID'],
            'subject': ['Subject ID', 'Subject', 'SubjectID'],
            'country': ['Country', 'CountryCode'],
            'visit': ['Visit', 'Visit Name', 'Cycle'],
            'missing_visits': ['# Missing Visits', 'Missing Visits', 'MissingVisits'],
            'missing_pages': ['# Missing Pages', 'Missing Pages', 'MissingPages'],
            'open_queries': ['# Open Queries', 'Open Queries', 'OpenQueries'],
            'frozen_crfs': ['# CRFs Frozen', 'Frozen CRFs', 'FrozenCRFs'],
            'locked_crfs': ['# CRFs Locked', 'Locked CRFs', 'LockedCRFs'],
            'ssm': ['SSM', 'Site Status Metric', 'SiteStatusMetric']
        },
        'Visit_Projection_Tracker': {
            'site': ['Site ID', 'Site'],
            'subject': ['Subject ID', 'Subject'],
            'visit': ['Planned Visit', 'Visit', 'Visit Name'],
            'days_outstanding': ['Days Outstanding', 'DaysOutstanding'],
            'status': ['Status', 'Visit Status']
        },
        'eSAE_Dashboard': {
            'site': ['Site ID', 'Site'],
            'subject': ['Subject ID', 'Subject'],
            'sae_count': ['SAE Count', 'SAECount', 'Serious Adverse Events'],
            'sae_status': ['SAE Status', 'Status'],
            'days_since_onset': ['Days Since Onset', 'DaysSinceOnset']
        }
    }
    
    def __init__(self, data_sources: Dict[str, pd.DataFrame] = None, graph=None):
        """
        Initialize the query executor
        
        Args:
            data_sources: Dictionary of DataFrames keyed by source name
            graph: NetworkX graph for graph-based queries
        """
        self.data_sources = data_sources or {}
        self.graph = graph
        self._execution_log = []
    
    def set_data_source(self, name: str, df: pd.DataFrame):
        """Add or update a data source"""
        self.data_sources[name] = df
        logger.info(f"Data source '{name}' set with {len(df)} rows")
    
    def set_graph(self, graph):
        """Set the graph for graph-based queries"""
        self.graph = graph
    
    def execute(self, parsed_query: ParsedQuery) -> QueryResult:
        """
        Execute a parsed query and return results
        
        Args:
            parsed_query: The parsed natural language query
            
        Returns:
            QueryResult with data and analysis
        """
        start_time = datetime.now()
        result = QueryResult(success=False)
        
        try:
            # Determine which data source to use
            primary_source = self._select_primary_source(parsed_query)
            
            if primary_source not in self.data_sources:
                result.error_message = f"Data source '{primary_source}' not available"
                return result
            
            df = self.data_sources[primary_source].copy()
            
            # Build and execute the query
            result.query_executed = self._build_query_description(parsed_query, primary_source)
            
            # Apply entity filters
            df = self._apply_entity_filters(df, parsed_query.entity_filters, primary_source)
            
            # Apply metric filters
            df = self._apply_metric_filters(df, parsed_query.metric_filters, primary_source)
            
            # Apply time constraint
            if parsed_query.time_constraint:
                df = self._apply_time_constraint(df, parsed_query.time_constraint)
            
            # Execute based on intent
            if parsed_query.intent == QueryIntent.TREND_ANALYSIS:
                result = self._execute_trend_analysis(df, parsed_query, primary_source)
            elif parsed_query.intent == QueryIntent.TOP_N:
                result = self._execute_top_n(df, parsed_query, primary_source, ascending=False)
            elif parsed_query.intent == QueryIntent.BOTTOM_N:
                result = self._execute_top_n(df, parsed_query, primary_source, ascending=True)
            elif parsed_query.intent == QueryIntent.AGGREGATION:
                result = self._execute_aggregation(df, parsed_query, primary_source)
            elif parsed_query.intent == QueryIntent.COMPARISON:
                result = self._execute_comparison(df, parsed_query, primary_source)
            elif parsed_query.intent == QueryIntent.CORRELATION:
                result = self._execute_correlation(df, parsed_query, primary_source)
            elif parsed_query.intent == QueryIntent.SAFETY_CHECK:
                result = self._execute_safety_check(df, parsed_query, primary_source)
            elif parsed_query.intent == QueryIntent.ANOMALY_DETECTION:
                result = self._execute_anomaly_detection(df, parsed_query, primary_source)
            else:
                # Default: Filter and list
                result = self._execute_filter_list(df, parsed_query, primary_source)
            
            result.success = True
            result.query_executed = self._build_query_description(parsed_query, primary_source)
            
        except Exception as e:
            result.error_message = f"Execution error: {str(e)}"
            logger.error(f"Query execution failed: {e}", exc_info=True)
        
        finally:
            end_time = datetime.now()
            result.execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return result
    
    def _select_primary_source(self, parsed_query: ParsedQuery) -> str:
        """Select the primary data source based on query"""
        if parsed_query.data_sources:
            # Map friendly names to actual source names
            source_mapping = {
                'CPID_EDC_Metrics': 'CPID_EDC_Metrics',
                'Visit_Projection_Tracker': 'visit_tracker',
                'eSAE_Dashboard': 'esae_dashboard',
                'Compiled_EDRR': 'edrr',
                'GlobalCodingReport_MedDRA': 'meddra',
                'GlobalCodingReport_WHODD': 'whodd'
            }
            
            for source in parsed_query.data_sources:
                mapped = source_mapping.get(source, source)
                if mapped in self.data_sources:
                    return mapped
        
        # Default to CPID if available
        if 'cpid' in self.data_sources:
            return 'cpid'
        
        # Return first available source
        if self.data_sources:
            return list(self.data_sources.keys())[0]
        
        return 'CPID_EDC_Metrics'
    
    def _get_column(self, df: pd.DataFrame, field_type: str, source: str) -> Optional[str]:
        """Find the actual column name for a field type"""
        mappings = self.COLUMN_MAPPINGS.get(source, self.COLUMN_MAPPINGS.get('CPID_EDC_Metrics', {}))
        candidates = mappings.get(field_type, [field_type])
        
        for col in candidates:
            if col in df.columns:
                return col
        
        # Try case-insensitive match
        lower_cols = {c.lower(): c for c in df.columns}
        for col in candidates:
            if col.lower() in lower_cols:
                return lower_cols[col.lower()]
        
        return None
    
    def _apply_entity_filters(self, df: pd.DataFrame, filters: List[EntityFilter], source: str) -> pd.DataFrame:
        """Apply entity-based filters to the DataFrame"""
        for ef in filters:
            if ef.entity_type == EntityType.COUNTRY:
                col = self._get_column(df, 'country', source)
                if col:
                    df = df[df[col].astype(str).str.upper().isin([v.upper() for v in ef.values])]
            
            elif ef.entity_type == EntityType.SITE:
                col = self._get_column(df, 'site', source)
                if col:
                    # Extract site IDs from filter values
                    site_ids = []
                    for v in ef.values:
                        import re
                        match = re.search(r'\d+', str(v))
                        if match:
                            site_ids.append(match.group())
                        else:
                            site_ids.append(str(v))
                    df = df[df[col].astype(str).isin(site_ids)]
            
            elif ef.entity_type == EntityType.VISIT:
                col = self._get_column(df, 'visit', source)
                if col:
                    df = df[df[col].astype(str).str.contains('|'.join(ef.values), case=False, na=False)]
            
            elif ef.entity_type == EntityType.SUBJECT:
                col = self._get_column(df, 'subject', source)
                if col:
                    subject_ids = []
                    for v in ef.values:
                        import re
                        match = re.search(r'\d+', str(v))
                        if match:
                            subject_ids.append(match.group())
                        else:
                            subject_ids.append(str(v))
                    df = df[df[col].astype(str).isin(subject_ids)]
        
        return df
    
    def _apply_metric_filters(self, df: pd.DataFrame, filters: List[MetricFilter], source: str) -> pd.DataFrame:
        """Apply metric-based filters to the DataFrame"""
        for mf in filters:
            col = self._get_column(df, mf.metric.value, source)
            if col and mf.value is not None:
                if mf.operator == 'greater_than':
                    df = df[pd.to_numeric(df[col], errors='coerce') > mf.value]
                elif mf.operator == 'less_than':
                    df = df[pd.to_numeric(df[col], errors='coerce') < mf.value]
                elif mf.operator == 'equals':
                    df = df[pd.to_numeric(df[col], errors='coerce') == mf.value]
        
        return df
    
    def _apply_time_constraint(self, df: pd.DataFrame, time_constraint: TimeConstraint) -> pd.DataFrame:
        """Apply time-based filtering"""
        # Look for date/snapshot columns
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'snapshot' in c.lower()]
        
        if not date_cols:
            return df  # No time column found
        
        date_col = date_cols[0]
        
        # Convert to datetime if needed
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        if time_constraint.type == 'relative':
            if time_constraint.unit == 'snapshots':
                # Get last N unique dates
                unique_dates = df[date_col].dropna().unique()
                unique_dates = sorted(unique_dates)[-time_constraint.value:]
                df = df[df[date_col].isin(unique_dates)]
            elif time_constraint.unit == 'days':
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=time_constraint.value)
                df = df[df[date_col] >= cutoff]
            elif time_constraint.unit == 'weeks':
                cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=time_constraint.value)
                df = df[df[date_col] >= cutoff]
            elif time_constraint.unit == 'months':
                cutoff = pd.Timestamp.now() - pd.DateOffset(months=time_constraint.value)
                df = df[df[date_col] >= cutoff]
        
        return df
    
    def _execute_trend_analysis(self, df: pd.DataFrame, parsed_query: ParsedQuery, source: str) -> QueryResult:
        """Execute trend analysis query"""
        result = QueryResult(success=True)
        
        # Get metric column
        metric_col = None
        if parsed_query.primary_metric:
            metric_col = self._get_column(df, parsed_query.primary_metric.value, source)
        
        if not metric_col:
            # Find a numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                metric_col = numeric_cols[0]
        
        if not metric_col:
            result.error_message = "No metric column found for trend analysis"
            return result
        
        # Get grouping column
        group_col = None
        for entity in parsed_query.group_by:
            group_col = self._get_column(df, entity.name.lower(), source)
            if group_col:
                break
        
        if group_col:
            # Group by entity and calculate trends
            grouped = df.groupby(group_col)[metric_col].agg(['mean', 'std', 'count', 'first', 'last'])
            grouped['change'] = grouped['last'] - grouped['first']
            grouped['change_pct'] = (grouped['change'] / grouped['first'].replace(0, 1)) * 100
            
            # Identify trending up entities
            trending_up = grouped[grouped['change'] > 0].sort_values('change', ascending=False)
            trending_down = grouped[grouped['change'] < 0].sort_values('change')
            
            result.trends = [
                {
                    'direction': 'up',
                    'entities': trending_up.index.tolist()[:10],
                    'changes': trending_up['change'].tolist()[:10],
                    'percentages': trending_up['change_pct'].tolist()[:10]
                },
                {
                    'direction': 'down',
                    'entities': trending_down.index.tolist()[:10],
                    'changes': trending_down['change'].tolist()[:10],
                    'percentages': trending_down['change_pct'].tolist()[:10]
                }
            ]
            
            result.data = grouped.reset_index()
            result.row_count = len(grouped)
            
            # Check for trending filter
            for mf in parsed_query.metric_filters:
                if mf.operator == 'trending_up':
                    result.data = trending_up.reset_index()
                    result.row_count = len(trending_up)
                elif mf.operator == 'trending_down':
                    result.data = trending_down.reset_index()
                    result.row_count = len(trending_down)
        else:
            # Overall trend
            result.summary = {
                'metric': metric_col,
                'mean': df[metric_col].mean(),
                'std': df[metric_col].std(),
                'min': df[metric_col].min(),
                'max': df[metric_col].max(),
                'trend': 'up' if df[metric_col].iloc[-1] > df[metric_col].iloc[0] else 'down'
            }
            result.data = df
            result.row_count = len(df)
        
        return result
    
    def _execute_top_n(self, df: pd.DataFrame, parsed_query: ParsedQuery, source: str, ascending: bool) -> QueryResult:
        """Execute top/bottom N query"""
        result = QueryResult(success=True)
        
        n = parsed_query.top_n or 10
        
        # Get metric column
        metric_col = None
        if parsed_query.primary_metric:
            metric_col = self._get_column(df, parsed_query.primary_metric.value, source)
        
        if not metric_col:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                metric_col = numeric_cols[0]
        
        if not metric_col:
            result.data = df.head(n)
            result.row_count = min(n, len(df))
            return result
        
        # Get grouping column
        group_col = None
        for entity in parsed_query.group_by:
            group_col = self._get_column(df, entity.name.lower(), source)
            if group_col:
                break
        
        if group_col:
            # Aggregate by group first
            grouped = df.groupby(group_col)[metric_col].sum().sort_values(ascending=ascending)
            top_entities = grouped.head(n)
            
            result.data = pd.DataFrame({
                group_col: top_entities.index,
                metric_col: top_entities.values
            })
            result.row_count = len(result.data)
            
            result.summary = {
                'group_by': group_col,
                'metric': metric_col,
                'direction': 'bottom' if ascending else 'top',
                'n': n
            }
        else:
            # Just sort and take top N
            df_sorted = df.sort_values(metric_col, ascending=ascending)
            result.data = df_sorted.head(n)
            result.row_count = min(n, len(df))
        
        return result
    
    def _execute_aggregation(self, df: pd.DataFrame, parsed_query: ParsedQuery, source: str) -> QueryResult:
        """Execute aggregation query"""
        result = QueryResult(success=True)
        
        # Get metric columns
        metric_cols = []
        if parsed_query.primary_metric:
            col = self._get_column(df, parsed_query.primary_metric.value, source)
            if col:
                metric_cols.append(col)
        
        for metric in parsed_query.secondary_metrics:
            col = self._get_column(df, metric.value, source)
            if col and col not in metric_cols:
                metric_cols.append(col)
        
        if not metric_cols:
            metric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Get grouping
        group_col = None
        for entity in parsed_query.group_by:
            group_col = self._get_column(df, entity.name.lower(), source)
            if group_col:
                break
        
        aggregations = {}
        for col in metric_cols:
            aggregations[col] = {
                'sum': df[col].sum(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count()
            }
        
        result.aggregations = aggregations
        
        if group_col:
            grouped = df.groupby(group_col)[metric_cols].agg(['sum', 'mean', 'count'])
            result.data = grouped.reset_index()
        else:
            result.data = df
        
        result.row_count = len(df)
        result.summary = {
            'total_records': len(df),
            'metrics_aggregated': metric_cols
        }
        
        return result
    
    def _execute_comparison(self, df: pd.DataFrame, parsed_query: ParsedQuery, source: str) -> QueryResult:
        """Execute comparison query"""
        result = QueryResult(success=True)
        
        # Get comparison entities from filters
        entities_to_compare = []
        for ef in parsed_query.entity_filters:
            entities_to_compare.extend(ef.values)
        
        if len(entities_to_compare) < 2:
            result.error_message = "Need at least 2 entities to compare"
            return result
        
        # Get metric column
        metric_col = None
        if parsed_query.primary_metric:
            metric_col = self._get_column(df, parsed_query.primary_metric.value, source)
        
        # Get group column
        group_col = None
        for entity in parsed_query.group_by:
            group_col = self._get_column(df, entity.name.lower(), source)
            if group_col:
                break
        
        if group_col and metric_col:
            comparison_data = []
            for entity in entities_to_compare:
                entity_df = df[df[group_col].astype(str).str.contains(str(entity), case=False)]
                comparison_data.append({
                    'entity': entity,
                    'mean': entity_df[metric_col].mean(),
                    'sum': entity_df[metric_col].sum(),
                    'count': len(entity_df)
                })
            
            result.data = pd.DataFrame(comparison_data)
            result.row_count = len(comparison_data)
        else:
            result.data = df
            result.row_count = len(df)
        
        return result
    
    def _execute_correlation(self, df: pd.DataFrame, parsed_query: ParsedQuery, source: str) -> QueryResult:
        """Execute correlation analysis"""
        result = QueryResult(success=True)
        
        # Get numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            result.error_message = "Need at least 2 numeric columns for correlation"
            return result
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations
        correlations = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Upper triangle only
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        correlations.append({
                            'variable_1': col1,
                            'variable_2': col2,
                            'correlation': round(corr_val, 3),
                            'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                        })
        
        result.correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
        result.data = corr_matrix.reset_index()
        result.row_count = len(correlations)
        
        return result
    
    def _execute_safety_check(self, df: pd.DataFrame, parsed_query: ParsedQuery, source: str) -> QueryResult:
        """Execute safety-related query"""
        result = QueryResult(success=True)
        
        # Look for SAE-related columns
        sae_cols = [c for c in df.columns if 'sae' in c.lower() or 'safety' in c.lower() or 'adverse' in c.lower()]
        
        if not sae_cols and source != 'esae_dashboard':
            # Try to join with SAE data
            if 'esae_dashboard' in self.data_sources:
                df = self.data_sources['esae_dashboard'].copy()
                sae_cols = df.columns.tolist()
        
        # Add alerts for critical safety items
        alerts = []
        
        # Check for high SAE counts
        sae_count_col = self._get_column(df, 'sae_count', source)
        if sae_count_col:
            high_sae = df[pd.to_numeric(df[sae_count_col], errors='coerce') > 5]
            if len(high_sae) > 0:
                alerts.append(f"⚠️ {len(high_sae)} records with SAE count > 5")
        
        # Check for aged SAEs
        days_col = self._get_column(df, 'days_since_onset', source)
        if days_col:
            aged_sae = df[pd.to_numeric(df[days_col], errors='coerce') > 30]
            if len(aged_sae) > 0:
                alerts.append(f"🔴 {len(aged_sae)} SAEs aged > 30 days")
        
        result.alerts = alerts
        result.data = df
        result.row_count = len(df)
        result.summary = {
            'total_safety_records': len(df),
            'critical_alerts': len(alerts)
        }
        
        return result
    
    def _execute_anomaly_detection(self, df: pd.DataFrame, parsed_query: ParsedQuery, source: str) -> QueryResult:
        """Execute anomaly detection query"""
        result = QueryResult(success=True)
        
        # Get metric column
        metric_col = None
        if parsed_query.primary_metric:
            metric_col = self._get_column(df, parsed_query.primary_metric.value, source)
        
        if not metric_col:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                metric_col = numeric_cols[0]
        
        if not metric_col:
            result.error_message = "No numeric column found for anomaly detection"
            return result
        
        # Calculate z-scores
        values = pd.to_numeric(df[metric_col], errors='coerce')
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val > 0:
            df['z_score'] = (values - mean_val) / std_val
            anomalies = df[abs(df['z_score']) > 2]
            
            result.data = anomalies
            result.row_count = len(anomalies)
            
            result.summary = {
                'metric': metric_col,
                'mean': mean_val,
                'std': std_val,
                'anomaly_count': len(anomalies),
                'anomaly_threshold': '2 standard deviations'
            }
            
            result.alerts = [f"Found {len(anomalies)} anomalies in {metric_col}"]
        else:
            result.data = df
            result.row_count = 0
            result.summary = {'message': 'No variation in data, cannot detect anomalies'}
        
        return result
    
    def _execute_filter_list(self, df: pd.DataFrame, parsed_query: ParsedQuery, source: str) -> QueryResult:
        """Execute simple filter and list query"""
        result = QueryResult(success=True)
        
        # Apply top N if specified
        if parsed_query.top_n:
            df = df.head(parsed_query.top_n)
        
        result.data = df
        result.row_count = len(df)
        result.summary = {
            'total_records': len(df),
            'columns': df.columns.tolist()
        }
        
        return result
    
    def _build_query_description(self, parsed_query: ParsedQuery, source: str) -> str:
        """Build a human-readable query description"""
        parts = []
        
        # SELECT clause
        if parsed_query.primary_metric:
            parts.append(f"SELECT {parsed_query.primary_metric.value}")
        else:
            parts.append("SELECT *")
        
        # FROM clause
        parts.append(f"FROM {source}")
        
        # WHERE clause
        where_parts = []
        for ef in parsed_query.entity_filters:
            where_parts.append(f"{ef.entity_type.name} IN {ef.values}")
        
        for mf in parsed_query.metric_filters:
            if mf.value is not None:
                where_parts.append(f"{mf.metric.value} {mf.operator} {mf.value}")
            else:
                where_parts.append(f"{mf.metric.value} IS {mf.operator}")
        
        if where_parts:
            parts.append("WHERE " + " AND ".join(where_parts))
        
        # GROUP BY
        if parsed_query.group_by:
            parts.append(f"GROUP BY {', '.join([e.name for e in parsed_query.group_by])}")
        
        # ORDER BY
        if parsed_query.primary_metric:
            parts.append(f"ORDER BY {parsed_query.primary_metric.value} {parsed_query.sort_order.upper()}")
        
        # LIMIT
        if parsed_query.top_n:
            parts.append(f"LIMIT {parsed_query.top_n}")
        
        return "\n".join(parts)
    
    def execute_graph_query(self, parsed_query: ParsedQuery) -> QueryResult:
        """Execute query against the graph structure"""
        result = QueryResult(success=False)
        
        if not self.graph:
            result.error_message = "Graph not initialized"
            return result
        
        try:
            import networkx as nx
            
            # Get nodes of specific type
            target_type = None
            for entity in parsed_query.group_by:
                if entity == EntityType.SITE:
                    target_type = 'Site'
                elif entity == EntityType.SUBJECT:
                    target_type = 'Subject'
                elif entity == EntityType.VISIT:
                    target_type = 'Visit'
            
            if not target_type:
                target_type = 'Subject'
            
            # Find nodes of target type
            nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == target_type]
            
            # Extract node data
            node_data = []
            for node in nodes:
                attrs = dict(self.graph.nodes[node])
                attrs['node_id'] = node
                attrs['degree'] = self.graph.degree(node)
                attrs['neighbors'] = list(self.graph.neighbors(node))[:5]
                node_data.append(attrs)
            
            result.data = pd.DataFrame(node_data)
            result.row_count = len(node_data)
            result.success = True
            
            result.summary = {
                'node_type': target_type,
                'total_nodes': len(nodes),
                'graph_stats': {
                    'total_nodes': self.graph.number_of_nodes(),
                    'total_edges': self.graph.number_of_edges()
                }
            }
            
        except Exception as e:
            result.error_message = f"Graph query failed: {str(e)}"
        
        return result
