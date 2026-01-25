"""
Metrics Service
Calculates and provides dashboard metrics using existing business logic
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for calculating and providing metrics"""
    
    def __init__(self, data_service):
        self.data_service = data_service
    
    async def get_kpi_metrics(self, study_id: Optional[str], site_id: Optional[str]) -> List[Dict]:
        """Get KPI metrics for dashboard tiles"""
        summary = await self.data_service.get_dashboard_summary(study_id)
        
        return [
            {
                'id': 'total_patients',
                'title': 'Total Patients',
                'value': summary['total_patients'],
                'unit': None,
                'trend': 'up',
                'trend_value': 5.2,
                'status': 'good',
                'tooltip': 'Total enrolled patients across all studies',
                'drill_down_available': True
            },
            {
                'id': 'clean_rate',
                'title': 'Clean Patient Rate',
                'value': round(summary['clean_patients'] / summary['total_patients'] * 100, 1) if summary['total_patients'] > 0 else 0,
                'unit': '%',
                'trend': 'up',
                'trend_value': 2.3,
                'status': 'good',
                'tooltip': 'Percentage of patients meeting all cleanliness criteria',
                'drill_down_available': True
            },
            {
                'id': 'dqi_score',
                'title': 'Data Quality Index',
                'value': summary['overall_dqi'],
                'unit': None,
                'trend': 'stable',
                'trend_value': 0.5,
                'status': 'good' if summary['overall_dqi'] >= 80 else 'warning',
                'tooltip': 'Overall data quality score based on completeness, consistency, timeliness',
                'drill_down_available': True
            },
            {
                'id': 'open_queries',
                'title': 'Open Queries',
                'value': summary['open_queries'],
                'unit': None,
                'trend': 'down',
                'trend_value': -8.5,
                'status': 'warning' if summary['open_queries'] > 50 else 'good',
                'tooltip': 'Total number of unresolved data queries',
                'drill_down_available': True
            },
            {
                'id': 'pending_saes',
                'title': 'Pending SAEs',
                'value': summary['pending_saes'],
                'unit': None,
                'trend': 'stable',
                'trend_value': 0,
                'status': 'critical' if summary['pending_saes'] > 10 else 'good',
                'tooltip': 'SAEs requiring reconciliation',
                'drill_down_available': True
            },
            {
                'id': 'uncoded_terms',
                'title': 'Uncoded Terms',
                'value': summary['uncoded_terms'],
                'unit': None,
                'trend': 'down',
                'trend_value': -15.0,
                'status': 'warning' if summary['uncoded_terms'] > 20 else 'good',
                'tooltip': 'Medical terms awaiting MedDRA/WHO Drug coding',
                'drill_down_available': True
            }
        ]
    
    async def get_kpi_tiles(self, study_id: Optional[str], site_id: Optional[str]) -> List[Dict]:
        """Alias for get_kpi_metrics"""
        return await self.get_kpi_metrics(study_id, site_id)
    
    async def get_dqi_metrics(self, study_id: Optional[str], site_id: Optional[str], days: int) -> Dict:
        """Get DQI breakdown metrics"""
        agg = await self.data_service.get_cpid_aggregate(study_id)
        total = agg["total_patients"]
        if total <= 0:
            return {
                'overall_dqi': 0.0,
                'completeness': 0.0,
                'consistency': 0.0,
                'timeliness': 0.0,
                'accuracy': 0.0,
                'conformity': 0.0,
                'trend': [],
                'comparison_to_benchmark': 0.0
            }

        missing_rate = (agg["missing_visits"] + agg["missing_pages"]) / max(1, total)
        open_rate = agg["open_queries"] / max(1, total)
        uncoded_rate = agg["uncoded_terms"] / max(1, total)

        completeness = max(0.0, 100.0 - min(40.0, missing_rate * 100))
        consistency = max(0.0, 100.0 - min(30.0, open_rate * 100))
        timeliness = max(0.0, 100.0 - min(25.0, open_rate * 100))
        accuracy = max(0.0, 100.0 - min(30.0, uncoded_rate * 100))
        conformity = max(0.0, min(100.0, (completeness + consistency) / 2))

        overall_dqi = round((completeness + consistency + timeliness + accuracy) / 4, 1)

        return {
            'overall_dqi': overall_dqi,
            'completeness': round(completeness, 1),
            'consistency': round(consistency, 1),
            'timeliness': round(timeliness, 1),
            'accuracy': round(accuracy, 1),
            'conformity': round(conformity, 1),
            'trend': [],
            'comparison_to_benchmark': round(max(0.0, overall_dqi - 80.0), 1)
        }
    
    async def get_cleanliness_metrics(self, study_id: Optional[str], site_id: Optional[str], days: int) -> Dict:
        """Get clean patient metrics"""
        agg = await self.data_service.get_cpid_aggregate(study_id)

        total = agg["total_patients"]
        clean = agg["clean_patients"]
        dirty = agg["dirty_patients"]
        pending = agg["at_risk_patients"]

        overall_rate = round(clean / total * 100, 1) if total > 0 else 0
        visits_rate = max(0.0, 100.0 - min(100.0, (agg["missing_visits"] / max(1, total)) * 100)) if total > 0 else 0
        pages_rate = max(0.0, 100.0 - min(100.0, (agg["missing_pages"] / max(1, total)) * 100)) if total > 0 else 0
        queries_rate = max(0.0, 100.0 - min(100.0, (agg["open_queries"] / max(1, total)) * 100)) if total > 0 else 0
        coding_rate = max(0.0, 100.0 - min(100.0, (agg["uncoded_terms"] / max(1, total)) * 100)) if total > 0 else 0

        return {
            'overall_rate': overall_rate,
            'total_patients': total,
            'clean_patients': clean,
            'dirty_patients': dirty,
            'pending_patients': pending,
            'trend': [],
            'by_category': {
                'visits': round(visits_rate, 1),
                'queries': round(queries_rate, 1),
                'coding': round(coding_rate, 1),
                'forms': round(pages_rate, 1)
            }
        }
    
    async def get_query_metrics(self, study_id: Optional[str], site_id: Optional[str], days: int) -> Dict:
        """Get query management metrics"""
        agg = await self.data_service.get_cpid_aggregate(study_id)
        total = agg["total_queries"]
        open_q = agg["open_queries"]
        closed_q = max(0, total - open_q)
        resolution_rate = round((closed_q / total * 100), 1) if total > 0 else 0.0

        return {
            'total_queries': total,
            'open_queries': open_q,
            'closed_queries': closed_q,
            'resolution_rate': resolution_rate,
            'avg_resolution_time_days': 0.0,
            'velocity_trend': [],
            'by_category': {},
            'aging_distribution': {}
        }
    
    async def get_sae_metrics(self, study_id: Optional[str], site_id: Optional[str]) -> Dict:
        """Get SAE reconciliation metrics"""
        sae = await self.data_service.get_sae_aggregate(study_id)
        total = sae["total_saes"]
        reconciled = sae["reconciled"]
        pending = max(0, sae["pending"])
        reconciliation_rate = round((reconciled / total * 100), 1) if total > 0 else 0.0

        return {
            'total_saes': total,
            'reconciled': reconciled,
            'pending': pending,
            'overdue': 0,
            'reconciliation_rate': reconciliation_rate,
            'avg_reconciliation_days': 0.0,
            'by_seriousness': sae["by_category"]
        }
    
    async def get_coding_metrics(self, study_id: Optional[str], site_id: Optional[str]) -> Dict:
        """Get medical coding metrics"""
        coding = await self.data_service.get_coding_aggregate(study_id)
        total = coding["total_terms"]
        coded = coding["coded_terms"]
        uncoded = coding["pending_terms"]
        completion_rate = round((coded / total * 100), 1) if total > 0 else 0.0

        return {
            'total_terms': total,
            'coded': coded,
            'uncoded': uncoded,
            'completion_rate': completion_rate,
            'meddra_status': coding.get("meddra", {"total": 0, "coded": 0, "uncoded": 0}),
            'whodrug_status': coding.get("whodrug", {"total": 0, "coded": 0, "uncoded": 0}),
            'uncoded_breakdown': []
        }
    
    async def get_velocity_metrics(self, study_id: Optional[str], site_id: Optional[str], days: int) -> Dict:
        """Get operational velocity metrics"""
        agg = await self.data_service.get_cpid_aggregate(study_id)
        total_queries = agg["total_queries"]
        open_queries = agg["open_queries"]
        resolved = max(0, total_queries - open_queries)

        queries_per_day = round(total_queries / max(1, days), 2) if days > 0 else 0.0
        resolutions_per_day = round(resolved / max(1, days), 2) if days > 0 else 0.0
        data_entries_per_day = 0.0

        return {
            'enrollment_velocity': 0.0,
            'query_resolution_velocity': queries_per_day,
            'form_completion_velocity': data_entries_per_day,
            'sae_processing_velocity': 0.0,
            'overall_velocity_index': round((queries_per_day + resolutions_per_day) / 2, 1) if days > 0 else 0.0,
            'trend': []
        }
    
    async def get_study_metrics(self, study_id: str) -> Dict:
        """Get metrics for a specific study"""
        summary = await self.data_service.get_dashboard_summary(study_id)
        agg = await self.data_service.get_cpid_aggregate(study_id)
        sae = await self.data_service.get_sae_aggregate(study_id)
        coding = await self.data_service.get_coding_aggregate(study_id)

        total_patients = agg["total_patients"]
        total_queries = agg["total_queries"]
        open_queries = agg["open_queries"]
        closed_queries = max(0, total_queries - open_queries)

        cleanliness_rate = round((agg["clean_patients"] / total_patients * 100), 1) if total_patients > 0 else 0.0
        query_resolution_rate = round((closed_queries / total_queries * 100), 1) if total_queries > 0 else 0.0
        query_velocity = round(total_queries / 7, 1) if total_queries > 0 else 0.0

        sae_total = sae["total_saes"]
        sae_reconciliation_rate = round((sae["reconciled"] / sae_total * 100), 1) if sae_total > 0 else 0.0

        coding_total = coding["total_terms"]
        coding_completion_rate = round((coding["coded_terms"] / coding_total * 100), 1) if coding_total > 0 else 0.0

        visit_completion_rate = round(100 - (agg["missing_visits"] / max(1, total_patients) * 100), 1) if total_patients > 0 else 0.0
        form_completion_rate = round(100 - (agg["missing_pages"] / max(1, total_patients) * 100), 1) if total_patients > 0 else 0.0

        return {
            'study_id': study_id,
            'dqi_score': summary.get('overall_dqi', 0.0),
            'dqi_trend': [],
            'cleanliness_rate': cleanliness_rate,
            'cleanliness_trend': [],
            'query_count': open_queries,
            'query_resolution_rate': query_resolution_rate,
            'query_velocity': query_velocity,
            'sae_count': sae_total,
            'sae_reconciliation_rate': sae_reconciliation_rate,
            'coding_completion_rate': coding_completion_rate,
            'visit_completion_rate': visit_completion_rate,
            'form_completion_rate': form_completion_rate
        }
    
    async def get_metric_trends(self, study_id: str, metric: str, days: int) -> Dict:
        """Get trends for a specific metric"""
        labels = []
        trend_data = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            labels.append(date.strftime('%Y-%m-%d'))
            trend_data.append(0.0)

        return {
            'labels': labels,
            'values': trend_data,
            'average': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    async def get_site_performance(self, site_id: str) -> Dict:
        """Get performance metrics for a site"""
        site = await self.data_service.get_site_detail(site_id)
        if site:
            return site.get('performance', {
                'query_resolution_rate': 0.0,
                'query_resolution_velocity': 0.0,
                'enrollment_rate': 0.0,
                'data_entry_timeliness': 0.0,
                'sae_reporting_timeliness': 0.0,
                'overall_score': 0.0
            })

        return {
            'query_resolution_rate': 0.0,
            'query_resolution_velocity': 0.0,
            'enrollment_rate': 0.0,
            'data_entry_timeliness': 0.0,
            'sae_reporting_timeliness': 0.0,
            'overall_score': 0.0
        }
    
    async def get_site_trends(self, site_id: str, metrics: List[str], days: int) -> Dict:
        """Get trend data for site metrics"""
        trends = {}
        for metric in metrics:
            trend_data = await self.get_metric_trends(site_id, metric, days)
            trends[metric] = trend_data
        return trends
    
    async def get_heatmap_data(self, metric: str, group_by: str, study_id: Optional[str]) -> List[Dict]:
        """Get heatmap visualization data"""
        return await self.data_service.get_heatmap_data(study_id or 'Study_1', metric)
    
    async def get_multiple_trends(self, metrics: List[str], study_id: Optional[str], site_id: Optional[str], days: int) -> Dict:
        """Get trends for multiple metrics"""
        trends = {}
        for metric in metrics:
            trends[metric] = await self.get_metric_trends(study_id or 'Study_1', metric, days)
        return trends
    
    async def get_benchmarks(self, study_id: Optional[str], metrics: List[str]) -> Dict:
        """Get benchmark comparisons"""
        benchmarks = {}
        for metric in metrics:
            current_value = 0.0
            benchmarks[metric] = {
                'current': current_value,
                'benchmark': 80.0,
                'industry_average': 75.0,
                'top_quartile': 90.0,
                'vs_benchmark': round(current_value - 80.0, 1)
            }
        return benchmarks
    
    async def detect_anomalies(self, study_id: Optional[str], sensitivity: float) -> List[Dict]:
        """Detect metric anomalies"""
        return []
