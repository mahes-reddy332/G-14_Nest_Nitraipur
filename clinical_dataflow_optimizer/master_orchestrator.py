"""
Master Orchestrator for Clinical Dataflow Optimizer
====================================================

This module provides:
1. Unified management of ALL studies (knowledge graphs, features, RAG pipelines)
2. Cross-study analytics and comparisons
3. Natural Language Query interface across all data
4. Master dashboard generation with unified insights
5. Real-time query execution against all studies

Usage:
    orchestrator = MasterOrchestrator()
    orchestrator.load_all_studies()
    response = orchestrator.ask("Which sites across all studies have the highest risk?")
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_ingestion import ClinicalDataIngester, load_all_clinical_data
from core.metrics_calculator import (
    PatientTwinBuilder, SiteMetricsAggregator, 
    StudyMetricsAggregator, DataQualityIndexCalculator
)
from core.feature_engineering import SiteFeatureEngineer, engineer_study_features
from agents.agent_framework import SupervisorAgent
from visualization.dashboard import create_full_dashboard, DashboardVisualizer
from models.data_models import DigitalPatientTwin, SiteMetrics, StudyMetrics, RiskLevel

# Knowledge Graph components
from graph.knowledge_graph import ClinicalKnowledgeGraph, NodeType, EdgeType
from graph.graph_builder import ClinicalGraphBuilder, build_knowledge_graph_from_study
from graph.graph_queries import GraphQueryEngine, QueryCondition, QueryOperator
from graph.graph_analytics import GraphAnalytics, PatientRiskProfile, SiteRiskProfile

# NLQ components
from nlq.conversational_engine import ConversationalEngine
from nlq.query_parser import QueryParser, ParsedQuery
from nlq.insight_generator import InsightGenerator

# Real-time monitoring
from core.real_time_monitor import RealTimeDataMonitor

logger = logging.getLogger(__name__)


@dataclass
class StudyAnalysisResult:
    """Complete analysis result for a single study"""
    study_id: str
    twins: List[DigitalPatientTwin]
    site_metrics: Dict[str, SiteMetrics]
    study_metrics: StudyMetrics
    recommendations: List[Any]
    knowledge_graph: ClinicalKnowledgeGraph
    query_engine: GraphQueryEngine
    graph_analytics: GraphAnalytics
    feature_engineer: Optional[SiteFeatureEngineer]
    feature_matrix: Optional[pd.DataFrame]
    multi_hop_queries: Dict[str, Any]
    graph_stats: Dict[str, Any]
    
    def to_summary(self) -> Dict:
        return {
            'study_id': self.study_id,
            'total_patients': len(self.twins),
            'clean_patients': sum(1 for t in self.twins if t.clean_status),
            'total_sites': len(self.site_metrics),
            'global_dqi': self.study_metrics.global_dqi if self.study_metrics else 0,
            'graph_nodes': self.graph_stats.get('total_nodes', 0),
            'graph_edges': self.graph_stats.get('total_edges', 0),
            'total_recommendations': len(self.recommendations) if self.recommendations else 0
        }


@dataclass
class CrossStudyInsight:
    """Insight derived from cross-study analysis"""
    insight_type: str
    title: str
    description: str
    affected_studies: List[str]
    severity: str  # 'Critical', 'High', 'Medium', 'Low', 'Info'
    metrics: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'insight_type': self.insight_type,
            'title': self.title,
            'description': self.description,
            'affected_studies': self.affected_studies,
            'severity': self.severity,
            'metrics': self.metrics,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class MasterOrchestrator:
    """
    Central orchestration for all clinical trial data analysis
    
    Features:
    - Load and analyze ALL studies in parallel
    - Maintain knowledge graphs for each study
    - Execute cross-study queries via NLQ
    - Generate unified dashboards and reports
    - Feature engineering across all sites
    """
    
    def __init__(self, data_path: str = None):
        """Initialize the Master Orchestrator"""
        if data_path is None:
            data_path = str(Path(__file__).parent.parent / "QC Anonymized Study Files")
        
        self.data_path = Path(data_path)
        self.reports_path = self.data_path.parent / "reports"
        self.reports_path.mkdir(exist_ok=True)
        
        # Study analysis storage
        self.studies: Dict[str, StudyAnalysisResult] = {}
        self.cross_study_insights: List[CrossStudyInsight] = []
        
        # Unified NLQ engine
        self.conversational_engine: Optional[ConversationalEngine] = None
        self.unified_data_sources: Dict[str, pd.DataFrame] = {}
        
        # Feature storage
        self.all_features: Dict[str, Dict] = {}
        self.unified_feature_matrix: Optional[pd.DataFrame] = None
        
        # Real-time monitoring
        self.real_time_monitor = RealTimeDataMonitor(self.data_path)
        
        logger.info(f"MasterOrchestrator initialized with data path: {self.data_path}")
    
    def load_all_studies(self, parallel: bool = True, max_workers: int = 4) -> Dict[str, StudyAnalysisResult]:
        """
        Load and analyze ALL studies with full feature extraction
        
        Args:
            parallel: Use parallel processing for faster loading
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary of study results
        """
        logger.info("="*70)
        logger.info("MASTER ORCHESTRATOR - LOADING ALL STUDIES")
        logger.info("="*70)
        
        # Get all study folders
        study_folders = [
            f for f in self.data_path.iterdir() 
            if f.is_dir() and 'CPID' in f.name
        ]
        
        logger.info(f"Found {len(study_folders)} study folders")
        
        # Load all data first
        ingester, all_studies_data = load_all_clinical_data(str(self.data_path))
        
        # Analyze each study
        if parallel and len(study_folders) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._analyze_single_study, 
                        study_id, 
                        study_data
                    ): study_id
                    for study_id, study_data in all_studies_data.items()
                }
                
                for future in as_completed(futures):
                    study_id = futures[future]
                    try:
                        result = future.result()
                        if result:
                            self.studies[study_id] = result
                            logger.info(f"✓ Completed {study_id}")
                    except Exception as e:
                        logger.error(f"✗ Error processing {study_id}: {e}")
        else:
            for study_id, study_data in all_studies_data.items():
                try:
                    result = self._analyze_single_study(study_id, study_data)
                    if result:
                        self.studies[study_id] = result
                        logger.info(f"✓ Completed {study_id}")
                except Exception as e:
                    logger.error(f"✗ Error processing {study_id}: {e}")
        
        # Build cross-study insights
        self._generate_cross_study_insights()
        
        # Initialize unified NLQ engine
        self._initialize_unified_nlq()
        
        # Build unified feature matrix
        self._build_unified_feature_matrix()
        
        logger.info(f"\nLoaded {len(self.studies)} studies successfully")
        return self.studies
    
    def _analyze_single_study(
        self, 
        study_id: str, 
        study_data: Dict[str, pd.DataFrame]
    ) -> Optional[StudyAnalysisResult]:
        """Analyze a single study with full feature extraction"""
        
        if 'cpid_metrics' not in study_data or study_data['cpid_metrics'] is None:
            logger.warning(f"No CPID metrics for {study_id}, skipping")
            return None
        
        # Build components
        twin_builder = PatientTwinBuilder()
        site_aggregator = SiteMetricsAggregator()
        study_aggregator = StudyMetricsAggregator()
        supervisor = SupervisorAgent()
        
        # Build Digital Patient Twins
        twins = twin_builder.build_all_twins(study_data, study_id)
        if not twins:
            return None
        
        # Aggregate metrics
        site_metrics = site_aggregator.aggregate_site_metrics(twins, study_id)
        study_metrics = study_aggregator.aggregate_study_metrics(site_metrics, twins, study_id)
        
        # Run agent analysis
        agent_results = supervisor.run_analysis(twins, site_metrics, study_data, study_id)
        recommendations = agent_results.get('prioritized', [])
        
        # Build Knowledge Graph
        builder = ClinicalGraphBuilder(study_id=study_id)
        knowledge_graph = builder.build_from_study_data(study_data, study_id)
        query_engine = GraphQueryEngine(knowledge_graph)
        graph_analytics = GraphAnalytics(knowledge_graph)
        
        # Run multi-hop queries
        multi_hop_queries = self._run_multi_hop_queries(query_engine, study_id)
        
        # Feature engineering
        feature_engineer = None
        feature_matrix = None
        try:
            cpid_df = study_data.get('cpid_metrics')
            inact_df = study_data.get('inactivated_forms')
            if cpid_df is not None:
                feature_engineer, feature_matrix = engineer_study_features(
                    study_id=study_id,
                    cpid_metrics=cpid_df,
                    inactivated_forms=inact_df
                )
        except Exception as e:
            logger.warning(f"Feature engineering failed for {study_id}: {e}")
        
        # Get graph stats
        graph_stats = knowledge_graph.get_statistics()
        
        return StudyAnalysisResult(
            study_id=study_id,
            twins=twins,
            site_metrics=site_metrics,
            study_metrics=study_metrics,
            recommendations=recommendations,
            knowledge_graph=knowledge_graph,
            query_engine=query_engine,
            graph_analytics=graph_analytics,
            feature_engineer=feature_engineer,
            feature_matrix=feature_matrix,
            multi_hop_queries=multi_hop_queries,
            graph_stats=graph_stats
        )
    
    def _run_multi_hop_queries(self, query_engine: GraphQueryEngine, study_id: str) -> Dict[str, Any]:
        """Execute standard multi-hop queries on a study"""
        results = {
            'patients_needing_attention': [],
            'critical_patients': [],
            'issue_summary': {},
            'site_aggregations': {}
        }
        
        try:
            # Find patients needing attention (multi-issue patients)
            attention_patients = query_engine.find_patients_needing_attention()
            results['patients_needing_attention'] = [
                p.to_dict() for p in attention_patients[:50]  # Top 50
            ]
            
            # Get critical patients
            critical = [p for p in attention_patients if p.risk_score >= 70]
            results['critical_patients'] = [p.to_dict() for p in critical]
            
            # Issue summary
            results['issue_summary'] = query_engine.get_issue_summary()
            
            # Site aggregations
            results['site_aggregations'] = query_engine.aggregate_by_site()
            
        except Exception as e:
            logger.warning(f"Multi-hop query error for {study_id}: {e}")
        
        return results
    
    def _generate_cross_study_insights(self):
        """Generate insights from cross-study analysis"""
        self.cross_study_insights = []
        
        if not self.studies:
            return
        
        # Insight 1: Studies with lowest DQI
        dqi_ranking = sorted(
            [(sid, s.study_metrics.global_dqi if s.study_metrics else 0) 
             for sid, s in self.studies.items()],
            key=lambda x: x[1]
        )
        
        low_dqi_studies = [(s, d) for s, d in dqi_ranking if d < 85]
        if low_dqi_studies:
            self.cross_study_insights.append(CrossStudyInsight(
                insight_type='data_quality',
                title='Studies with Low Data Quality Index',
                description=f'{len(low_dqi_studies)} studies have DQI below 85%',
                affected_studies=[s for s, _ in low_dqi_studies],
                severity='High',
                metrics={s: d for s, d in low_dqi_studies},
                recommendations=[
                    'Prioritize data cleaning in affected studies',
                    'Review site training programs',
                    'Consider targeted monitoring visits'
                ]
            ))
        
        # Insight 2: Critical sites across all studies
        critical_sites = []
        for study_id, study in self.studies.items():
            for site_id, site in study.site_metrics.items():
                if site.risk_level == RiskLevel.CRITICAL:
                    critical_sites.append({
                        'study': study_id,
                        'site': site_id,
                        'dqi': site.data_quality_index
                    })
        
        if critical_sites:
            self.cross_study_insights.append(CrossStudyInsight(
                insight_type='site_risk',
                title='Critical Sites Requiring Immediate Attention',
                description=f'{len(critical_sites)} critical sites identified across all studies',
                affected_studies=list(set(s['study'] for s in critical_sites)),
                severity='Critical',
                metrics={'critical_sites': critical_sites[:20]},
                recommendations=[
                    'Conduct immediate site review',
                    'Escalate to clinical operations',
                    'Consider site remediation plan'
                ]
            ))
        
        # Insight 3: Studies approaching interim analysis readiness
        interim_ready = [
            (sid, s.study_metrics.global_clean_rate) 
            for sid, s in self.studies.items()
            if s.study_metrics and s.study_metrics.global_clean_rate >= 80
        ]
        
        if interim_ready:
            self.cross_study_insights.append(CrossStudyInsight(
                insight_type='milestone',
                title='Studies Near Interim Analysis Readiness',
                description=f'{len(interim_ready)} studies have ≥80% clean rate',
                affected_studies=[s for s, _ in interim_ready],
                severity='Info',
                metrics={s: r for s, r in interim_ready},
                recommendations=[
                    'Schedule data review meetings',
                    'Prepare interim analysis packages',
                    'Notify biostatistics team'
                ]
            ))
        
        # Insight 4: Common issues across studies
        issue_counts = {}
        for study_id, study in self.studies.items():
            for twin in study.twins:
                for item in twin.blocking_items:
                    issue_type = item.item_type
                    if issue_type not in issue_counts:
                        issue_counts[issue_type] = {'count': 0, 'studies': set()}
                    issue_counts[issue_type]['count'] += 1
                    issue_counts[issue_type]['studies'].add(study_id)
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
        if top_issues:
            self.cross_study_insights.append(CrossStudyInsight(
                insight_type='common_issues',
                title='Most Common Blocking Issues Across Portfolio',
                description='Top issues preventing patient clean status',
                affected_studies=list(set().union(*[i[1]['studies'] for i in top_issues])),
                severity='Medium',
                metrics={
                    issue: {'count': data['count'], 'studies': len(data['studies'])}
                    for issue, data in top_issues
                },
                recommendations=[
                    f'Address {top_issues[0][0]} as primary focus' if top_issues else '',
                    'Implement standardized query resolution workflows',
                    'Review site training on common issues'
                ]
            ))
    
    def _initialize_unified_nlq(self):
        """Initialize unified NLQ engine with all study data"""
        # Combine all data sources
        all_cpid = []
        all_sae = []
        all_twins_data = []
        
        for study_id, study in self.studies.items():
            # Convert twins to DataFrame
            for twin in study.twins:
                twin_dict = twin.to_dict()
                twin_dict['study_id'] = study_id
                all_twins_data.append(twin_dict)
        
        if all_twins_data:
            self.unified_data_sources['twins'] = pd.DataFrame(all_twins_data)
        
        # Initialize conversational engine
        self.conversational_engine = ConversationalEngine(
            data_sources=self.unified_data_sources
        )
        
        logger.info("Unified NLQ engine initialized")
    
    def _build_unified_feature_matrix(self):
        """Build unified feature matrix across all studies"""
        all_features = []
        
        for study_id, study in self.studies.items():
            if study.feature_matrix is not None:
                df = study.feature_matrix.copy()
                df['study_id'] = study_id
                all_features.append(df)
        
        if all_features:
            self.unified_feature_matrix = pd.concat(all_features, ignore_index=True)
            logger.info(f"Built unified feature matrix: {self.unified_feature_matrix.shape}")
    
    # ==================== Query Interface ====================
    
    def ask(self, question: str, study_filter: List[str] = None) -> Dict[str, Any]:
        """
        Natural language query interface across all studies
        
        Args:
            question: Natural language question
            study_filter: Optional list of study IDs to filter
            
        Returns:
            Response with answer, data, and visualizations
        """
        logger.info(f"Processing query: {question}")
        
        response = {
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'answer': '',
            'data': {},
            'studies_searched': [],
            'insights': []
        }
        
        # Determine relevant studies
        target_studies = study_filter or list(self.studies.keys())
        response['studies_searched'] = target_studies
        
        # Parse query intent
        query_lower = question.lower()
        
        # Route to appropriate handler
        if any(kw in query_lower for kw in ['risk', 'critical', 'attention', 'problem']):
            response = self._handle_risk_query(question, target_studies, response)
        elif any(kw in query_lower for kw in ['compare', 'comparison', 'versus', 'vs']):
            response = self._handle_comparison_query(question, target_studies, response)
        elif any(kw in query_lower for kw in ['trend', 'trending', 'change', 'over time']):
            response = self._handle_trend_query(question, target_studies, response)
        elif any(kw in query_lower for kw in ['site', 'sites']):
            response = self._handle_site_query(question, target_studies, response)
        elif any(kw in query_lower for kw in ['patient', 'subject', 'patients']):
            response = self._handle_patient_query(question, target_studies, response)
        elif any(kw in query_lower for kw in ['quality', 'dqi', 'clean']):
            response = self._handle_quality_query(question, target_studies, response)
        elif any(kw in query_lower for kw in ['summary', 'overview', 'status']):
            response = self._handle_summary_query(question, target_studies, response)
        else:
            response = self._handle_general_query(question, target_studies, response)
        
        return response
    
    def _handle_risk_query(self, question: str, studies: List[str], response: Dict) -> Dict:
        """Handle risk-related queries"""
        critical_data = []
        high_risk_patients = []
        high_risk_sites = []
        
        for study_id in studies:
            if study_id not in self.studies:
                continue
            
            study = self.studies[study_id]
            
            # Get critical patients from multi-hop queries
            critical = study.multi_hop_queries.get('critical_patients', [])
            for p in critical[:10]:
                p['study_id'] = study_id
                high_risk_patients.append(p)
            
            # Get high-risk sites
            for site_id, site in study.site_metrics.items():
                if site.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                    high_risk_sites.append({
                        'study_id': study_id,
                        'site_id': site_id,
                        'risk_level': site.risk_level.value,
                        'dqi': site.data_quality_index,
                        'open_queries': site.total_open_queries
                    })
        
        # Sort by risk
        high_risk_sites.sort(key=lambda x: x['dqi'])
        high_risk_patients.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
        
        response['data'] = {
            'high_risk_patients': high_risk_patients[:20],
            'high_risk_sites': high_risk_sites[:20],
            'total_critical_patients': len(high_risk_patients),
            'total_risk_sites': len(high_risk_sites)
        }
        
        # Generate answer
        response['answer'] = f"""## Risk Analysis Across {len(studies)} Studies

### High-Risk Overview
- **{len(high_risk_patients)}** critical patients requiring immediate attention
- **{len(high_risk_sites)}** sites with elevated risk levels

### Top Risk Sites
| Study | Site | Risk Level | DQI | Open Queries |
|-------|------|------------|-----|--------------|
"""
        for site in high_risk_sites[:10]:
            response['answer'] += f"| {site['study_id']} | {site['site_id']} | {site['risk_level']} | {site['dqi']:.1f}% | {site['open_queries']} |\n"
        
        response['insights'] = [
            f"Focus on {high_risk_sites[0]['site_id']} in {high_risk_sites[0]['study_id']} - lowest DQI" if high_risk_sites else "",
            f"{len([s for s in high_risk_sites if s['risk_level'] == 'Critical'])} sites at critical risk level"
        ]
        
        return response
    
    def _handle_comparison_query(self, question: str, studies: List[str], response: Dict) -> Dict:
        """Handle comparison queries between studies"""
        comparison_data = []
        
        for study_id in studies:
            if study_id not in self.studies:
                continue
            
            study = self.studies[study_id]
            sm = study.study_metrics
            
            comparison_data.append({
                'study_id': study_id,
                'total_patients': sm.total_patients if sm else 0,
                'clean_patients': sm.clean_patients if sm else 0,
                'clean_rate': sm.global_clean_rate if sm else 0,
                'global_dqi': sm.global_dqi if sm else 0,
                'total_sites': sm.total_sites if sm else 0,
                'sites_at_risk': sm.sites_at_risk if sm else 0
            })
        
        # Sort by DQI for ranking
        comparison_data.sort(key=lambda x: x['global_dqi'], reverse=True)
        
        response['data'] = {'comparison': comparison_data}
        
        response['answer'] = f"""## Study Comparison

| Study | Patients | Clean Rate | DQI | Sites | At Risk |
|-------|----------|------------|-----|-------|---------|
"""
        for comp in comparison_data:
            response['answer'] += f"| {comp['study_id']} | {comp['total_patients']} | {comp['clean_rate']:.1f}% | {comp['global_dqi']:.1f}% | {comp['total_sites']} | {comp['sites_at_risk']} |\n"
        
        # Best and worst performers
        if comparison_data:
            best = comparison_data[0]
            worst = comparison_data[-1]
            response['answer'] += f"\n### Key Findings\n"
            response['answer'] += f"- **Best performing:** {best['study_id']} (DQI: {best['global_dqi']:.1f}%)\n"
            response['answer'] += f"- **Needs attention:** {worst['study_id']} (DQI: {worst['global_dqi']:.1f}%)\n"
        
        return response
    
    def _handle_site_query(self, question: str, studies: List[str], response: Dict) -> Dict:
        """Handle site-related queries"""
        all_sites = []
        
        for study_id in studies:
            if study_id not in self.studies:
                continue
            
            study = self.studies[study_id]
            for site_id, site in study.site_metrics.items():
                all_sites.append({
                    'study_id': study_id,
                    'site_id': site_id,
                    'country': site.country or 'Unknown',
                    'patients': site.total_patients,
                    'dqi': site.data_quality_index,
                    'risk_level': site.risk_level.value,
                    'open_queries': site.total_open_queries
                })
        
        # Sort by DQI
        all_sites.sort(key=lambda x: x['dqi'])
        
        response['data'] = {
            'sites': all_sites,
            'total_sites': len(all_sites),
            'by_risk': {
                'Critical': len([s for s in all_sites if s['risk_level'] == 'Critical']),
                'High': len([s for s in all_sites if s['risk_level'] == 'High']),
                'Medium': len([s for s in all_sites if s['risk_level'] == 'Medium']),
                'Low': len([s for s in all_sites if s['risk_level'] == 'Low'])
            }
        }
        
        response['answer'] = f"""## Site Analysis Across {len(studies)} Studies

### Overview
- **Total Sites:** {len(all_sites)}
- **Critical:** {response['data']['by_risk']['Critical']}
- **High Risk:** {response['data']['by_risk']['High']}
- **Medium:** {response['data']['by_risk']['Medium']}
- **Low:** {response['data']['by_risk']['Low']}

### Sites Requiring Attention (Lowest DQI)
| Study | Site | Country | DQI | Risk | Open Queries |
|-------|------|---------|-----|------|--------------|
"""
        for site in all_sites[:15]:
            response['answer'] += f"| {site['study_id']} | {site['site_id']} | {site['country']} | {site['dqi']:.1f}% | {site['risk_level']} | {site['open_queries']} |\n"
        
        return response
    
    def _handle_patient_query(self, question: str, studies: List[str], response: Dict) -> Dict:
        """Handle patient-related queries"""
        all_patients = []
        
        for study_id in studies:
            if study_id not in self.studies:
                continue
            
            study = self.studies[study_id]
            attention_patients = study.multi_hop_queries.get('patients_needing_attention', [])
            
            for p in attention_patients[:20]:
                p['study_id'] = study_id
                all_patients.append(p)
        
        # Sort by risk score
        all_patients.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
        
        total_patients = sum(len(s.twins) for s in self.studies.values() if s.study_id in studies)
        clean_patients = sum(
            sum(1 for t in s.twins if t.clean_status) 
            for s in self.studies.values() if s.study_id in studies
        )
        
        response['data'] = {
            'attention_patients': all_patients[:30],
            'total_patients': total_patients,
            'clean_patients': clean_patients,
            'clean_rate': (clean_patients / total_patients * 100) if total_patients > 0 else 0
        }
        
        response['answer'] = f"""## Patient Analysis Across {len(studies)} Studies

### Overview
- **Total Patients:** {total_patients:,}
- **Clean Patients:** {clean_patients:,}
- **Clean Rate:** {response['data']['clean_rate']:.1f}%
- **Patients Needing Attention:** {len(all_patients)}

### High Priority Patients (Multi-Hop Analysis)
| Study | Subject | Site | Risk Score | Priority | Issues |
|-------|---------|------|------------|----------|--------|
"""
        for p in all_patients[:15]:
            issues = p.get('issue_count', 0)
            response['answer'] += f"| {p['study_id']} | {p.get('subject_id', 'N/A')} | {p.get('site_id', 'N/A')} | {p.get('risk_score', 0):.0f} | #{p.get('priority_rank', 'N/A')} | {issues} |\n"
        
        return response
    
    def _handle_quality_query(self, question: str, studies: List[str], response: Dict) -> Dict:
        """Handle data quality queries"""
        quality_data = []
        
        for study_id in studies:
            if study_id not in self.studies:
                continue
            
            study = self.studies[study_id]
            sm = study.study_metrics
            
            quality_data.append({
                'study_id': study_id,
                'dqi': sm.global_dqi if sm else 0,
                'clean_rate': sm.global_clean_rate if sm else 0,
                'interim_ready': sm.interim_analysis_ready if sm else False,
                'sites_at_risk': sm.sites_at_risk if sm else 0
            })
        
        quality_data.sort(key=lambda x: x['dqi'], reverse=True)
        
        response['data'] = {'quality': quality_data}
        
        avg_dqi = np.mean([q['dqi'] for q in quality_data]) if quality_data else 0
        interim_ready_count = sum(1 for q in quality_data if q['interim_ready'])
        
        response['answer'] = f"""## Data Quality Overview

### Portfolio Summary
- **Average DQI:** {avg_dqi:.1f}%
- **Studies Ready for Interim:** {interim_ready_count} / {len(quality_data)}

### Study Quality Ranking
| Study | DQI | Clean Rate | Interim Ready | Sites at Risk |
|-------|-----|------------|---------------|---------------|
"""
        for q in quality_data:
            ready = "✓" if q['interim_ready'] else "✗"
            response['answer'] += f"| {q['study_id']} | {q['dqi']:.1f}% | {q['clean_rate']:.1f}% | {ready} | {q['sites_at_risk']} |\n"
        
        return response
    
    def _handle_summary_query(self, question: str, studies: List[str], response: Dict) -> Dict:
        """Handle summary/overview queries"""
        total_patients = 0
        total_clean = 0
        total_sites = 0
        total_recommendations = 0
        
        study_summaries = []
        
        for study_id in studies:
            if study_id not in self.studies:
                continue
            
            study = self.studies[study_id]
            summary = study.to_summary()
            study_summaries.append(summary)
            
            total_patients += summary['total_patients']
            total_clean += summary['clean_patients']
            total_sites += summary['total_sites']
            total_recommendations += summary['total_recommendations']
        
        response['data'] = {
            'summaries': study_summaries,
            'totals': {
                'studies': len(study_summaries),
                'patients': total_patients,
                'clean_patients': total_clean,
                'sites': total_sites,
                'recommendations': total_recommendations
            }
        }
        
        clean_rate = (total_clean / total_patients * 100) if total_patients > 0 else 0
        
        response['answer'] = f"""## Portfolio Overview

### Key Metrics
- **Total Studies:** {len(study_summaries)}
- **Total Patients:** {total_patients:,}
- **Clean Patients:** {total_clean:,} ({clean_rate:.1f}%)
- **Total Sites:** {total_sites}
- **Active Recommendations:** {total_recommendations}

### Study Summary
| Study | Patients | Clean | DQI | Sites | Graph Nodes |
|-------|----------|-------|-----|-------|-------------|
"""
        for s in study_summaries:
            response['answer'] += f"| {s['study_id']} | {s['total_patients']} | {s['clean_patients']} | {s['global_dqi']:.1f}% | {s['total_sites']} | {s['graph_nodes']} |\n"
        
        return response
    
    def _handle_trend_query(self, question: str, studies: List[str], response: Dict) -> Dict:
        """Handle trend-related queries"""
        response['answer'] = """## Trend Analysis

*Note: Full trend analysis requires historical snapshots. Current analysis shows point-in-time metrics.*

"""
        return self._handle_summary_query(question, studies, response)
    
    def _handle_general_query(self, question: str, studies: List[str], response: Dict) -> Dict:
        """Handle general queries"""
        return self._handle_summary_query(question, studies, response)
    
    # ==================== Export Methods ====================
    
    def export_all_artifacts(self) -> Dict[str, str]:
        """Export all analysis artifacts for all studies"""
        export_paths = {}
        
        for study_id, study in self.studies.items():
            study_paths = self._export_study_artifacts(study_id, study)
            export_paths[study_id] = study_paths
        
        # Export cross-study artifacts
        cross_study_path = self._export_cross_study_artifacts()
        export_paths['cross_study'] = cross_study_path
        
        # Export master dashboard
        master_dashboard_path = self.generate_master_dashboard()
        export_paths['master_dashboard'] = master_dashboard_path
        
        return export_paths
    
    def _export_study_artifacts(self, study_id: str, study: StudyAnalysisResult) -> Dict[str, str]:
        """Export artifacts for a single study"""
        paths = {}
        
        # Knowledge Graph
        kg_path = self.reports_path / f"{study_id}_knowledge_graph.json"
        try:
            kg_data = {
                'study_id': study_id,
                'stats': study.graph_stats,
                'node_types': {
                    k.name: v for k, v in study.graph_stats.get('nodes_by_type', {}).items()
                } if 'nodes_by_type' in study.graph_stats else {}
            }
            with open(kg_path, 'w') as f:
                json.dump(kg_data, f, indent=2, default=str)
            paths['knowledge_graph'] = str(kg_path)
        except Exception as e:
            logger.warning(f"Failed to export KG for {study_id}: {e}")
        
        # Multi-hop queries
        mhq_path = self.reports_path / f"{study_id}_multi_hop_queries.json"
        try:
            with open(mhq_path, 'w') as f:
                json.dump(study.multi_hop_queries, f, indent=2, default=str)
            paths['multi_hop_queries'] = str(mhq_path)
        except Exception as e:
            logger.warning(f"Failed to export MHQ for {study_id}: {e}")
        
        # Feature matrix
        if study.feature_matrix is not None:
            fm_path = self.reports_path / f"{study_id}_feature_matrix.csv"
            study.feature_matrix.to_csv(fm_path, index=False)
            paths['feature_matrix'] = str(fm_path)
        
        # Features JSON
        if study.feature_engineer:
            feat_path = self.reports_path / f"{study_id}_features.json"
            with open(feat_path, 'w') as f:
                json.dump(study.feature_engineer.to_dict(), f, indent=2, default=str)
            paths['features'] = str(feat_path)
        
        # Patient network
        pn_path = self.reports_path / f"{study_id}_patient_network.csv"
        try:
            network_data = []
            for twin in study.twins:
                network_data.append({
                    'subject_id': twin.subject_id,
                    'site_id': twin.site_id,
                    'clean_status': twin.clean_status,
                    'risk_score': twin.data_quality_index,
                    'blocking_items': len(twin.blocking_items)
                })
            pd.DataFrame(network_data).to_csv(pn_path, index=False)
            paths['patient_network'] = str(pn_path)
        except Exception as e:
            logger.warning(f"Failed to export patient network for {study_id}: {e}")
        
        return paths
    
    def _export_cross_study_artifacts(self) -> Dict[str, str]:
        """Export cross-study analysis artifacts"""
        paths = {}
        
        # Cross-study insights
        insights_path = self.reports_path / "cross_study_insights.json"
        with open(insights_path, 'w') as f:
            json.dump([i.to_dict() for i in self.cross_study_insights], f, indent=2)
        paths['insights'] = str(insights_path)
        
        # Unified feature matrix
        if self.unified_feature_matrix is not None:
            ufm_path = self.reports_path / "unified_feature_matrix.csv"
            self.unified_feature_matrix.to_csv(ufm_path, index=False)
            paths['unified_features'] = str(ufm_path)
        
        # Portfolio summary
        summary_path = self.reports_path / "portfolio_summary.json"
        portfolio_summary = {
            'generated_at': datetime.now().isoformat(),
            'total_studies': len(self.studies),
            'studies': {sid: s.to_summary() for sid, s in self.studies.items()},
            'cross_study_insights': len(self.cross_study_insights)
        }
        with open(summary_path, 'w') as f:
            json.dump(portfolio_summary, f, indent=2, default=str)
        paths['portfolio_summary'] = str(summary_path)
        
        return paths
    
    def generate_master_dashboard(self) -> str:
        """Generate the master orchestration dashboard with unified insights"""
        output_path = self.reports_path / "master_dashboard.html"
        
        # Collect all study metrics
        study_data = []
        for study_id, study in self.studies.items():
            sm = study.study_metrics
            study_data.append({
                'study_id': study_id,
                'total_patients': sm.total_patients if sm else 0,
                'clean_patients': sm.clean_patients if sm else 0,
                'clean_rate': sm.global_clean_rate if sm else 0,
                'global_dqi': sm.global_dqi if sm else 0,
                'total_sites': sm.total_sites if sm else 0,
                'sites_at_risk': sm.sites_at_risk if sm else 0,
                'graph_nodes': study.graph_stats.get('total_nodes', 0),
                'graph_edges': study.graph_stats.get('total_edges', 0),
                'recommendations': len(study.recommendations) if study.recommendations else 0
            })
        
        # Calculate totals
        totals = {
            'studies': len(study_data),
            'patients': sum(s['total_patients'] for s in study_data),
            'clean_patients': sum(s['clean_patients'] for s in study_data),
            'sites': sum(s['total_sites'] for s in study_data),
            'recommendations': sum(s['recommendations'] for s in study_data)
        }
        totals['clean_rate'] = (totals['clean_patients'] / totals['patients'] * 100) if totals['patients'] > 0 else 0
        totals['avg_dqi'] = np.mean([s['global_dqi'] for s in study_data]) if study_data else 0
        
        # Generate HTML
        html_content = self._generate_master_dashboard_html(study_data, totals)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Master dashboard generated: {output_path}")
        return str(output_path)
    
    def _generate_master_dashboard_html(self, study_data: List[Dict], totals: Dict) -> str:
        """Generate the master dashboard HTML content"""
        
        # Sort studies by DQI for charts
        study_data_sorted = sorted(study_data, key=lambda x: x['global_dqi'])
        
        # Generate study cards HTML
        study_cards_html = ""
        for study in sorted(study_data, key=lambda x: x['study_id']):
            dqi_color = '#10b981' if study['global_dqi'] >= 85 else '#f59e0b' if study['global_dqi'] >= 70 else '#ef4444'
            study_cards_html += f'''
            <div class="study-card" onclick="navigateToStudy('{study['study_id']}')" data-study="{study['study_id']}">
                <div class="study-header">
                    <h3>{study['study_id']}</h3>
                    <span class="dqi-badge" style="background: {dqi_color};">{study['global_dqi']:.1f}%</span>
                </div>
                <div class="study-metrics">
                    <div class="metric">
                        <span class="metric-value">{study['total_patients']:,}</span>
                        <span class="metric-label">Patients</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">{study['clean_rate']:.1f}%</span>
                        <span class="metric-label">Clean Rate</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">{study['total_sites']}</span>
                        <span class="metric-label">Sites</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">{study['graph_nodes']:,}</span>
                        <span class="metric-label">Graph Nodes</span>
                    </div>
                </div>
                <div class="study-footer">
                    <span class="rec-count">{study['recommendations']} recommendations</span>
                    <span class="risk-sites">{study['sites_at_risk']} sites at risk</span>
                </div>
            </div>
            '''
        
        # Generate insights HTML
        insights_html = ""
        for insight in self.cross_study_insights[:6]:
            severity_color = {
                'Critical': '#ef4444',
                'High': '#f97316',
                'Medium': '#eab308',
                'Low': '#10b981',
                'Info': '#3b82f6'
            }.get(insight.severity, '#94a3b8')
            
            insights_html += f'''
            <div class="insight-card" data-severity="{insight.severity.lower()}">
                <div class="insight-header">
                    <span class="severity-badge" style="background: {severity_color};">{insight.severity}</span>
                    <span class="insight-type">{insight.insight_type}</span>
                </div>
                <h4>{insight.title}</h4>
                <p>{insight.description}</p>
                <div class="affected-studies">
                    {' '.join(f'<span class="study-tag">{s}</span>' for s in insight.affected_studies[:5])}
                </div>
            </div>
            '''
        
        # Chart data
        chart_labels = json.dumps([s['study_id'] for s in study_data_sorted])
        chart_dqi = json.dumps([s['global_dqi'] for s in study_data_sorted])
        chart_patients = json.dumps([s['total_patients'] for s in study_data_sorted])
        chart_clean_rates = json.dumps([s['clean_rate'] for s in study_data_sorted])
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Dashboard - Clinical Dataflow Optimizer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --border-color: #334155;
            --primary: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --purple: #8b5cf6;
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 24px 32px;
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        
        .header-content {{
            max-width: 1800px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .logo {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        
        .logo i {{
            font-size: 32px;
            color: var(--primary);
        }}
        
        .logo h1 {{
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary) 0%, var(--purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .logo span {{
            font-size: 12px;
            color: var(--text-muted);
            display: block;
        }}
        
        /* Query Bar */
        .query-section {{
            background: var(--bg-secondary);
            padding: 24px 32px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .query-container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        
        .query-bar {{
            display: flex;
            gap: 16px;
            align-items: center;
        }}
        
        .query-input {{
            flex: 1;
            padding: 16px 24px;
            border: 2px solid var(--border-color);
            border-radius: 12px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 16px;
            transition: all 0.3s ease;
        }}
        
        .query-input:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
        }}
        
        .query-btn {{
            padding: 16px 32px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--purple) 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
        }}
        
        .query-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
        }}
        
        .quick-queries {{
            margin-top: 16px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        .quick-query {{
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            color: var(--text-secondary);
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .quick-query:hover {{
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }}
        
        /* Main Content */
        .main-content {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 32px;
        }}
        
        /* KPI Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 20px;
            margin-bottom: 32px;
        }}
        
        .kpi-card {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border-color);
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        }}
        
        .kpi-icon {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px;
            font-size: 20px;
        }}
        
        .kpi-value {{
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        
        .kpi-label {{
            font-size: 13px;
            color: var(--text-muted);
        }}
        
        /* Charts Section */
        .charts-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }}
        
        .chart-card {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border-color);
        }}
        
        .chart-card h3 {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }}
        
        /* Insights Section */
        .insights-section {{
            margin-bottom: 32px;
        }}
        
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .section-title {{
            font-size: 20px;
            font-weight: 600;
        }}
        
        .insights-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }}
        
        .insight-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }}
        
        .insight-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }}
        
        .insight-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        
        .severity-badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            color: white;
        }}
        
        .insight-type {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        
        .insight-card h4 {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .insight-card p {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 12px;
        }}
        
        .affected-studies {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }}
        
        .study-tag {{
            padding: 4px 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 11px;
            color: var(--text-secondary);
        }}
        
        /* Studies Grid */
        .studies-section {{
            margin-bottom: 32px;
        }}
        
        .studies-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .study-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .study-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
            border-color: var(--primary);
        }}
        
        .study-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        
        .study-header h3 {{
            font-size: 18px;
            font-weight: 600;
        }}
        
        .dqi-badge {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            color: white;
        }}
        
        .study-metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }}
        
        .metric {{
            text-align: center;
        }}
        
        .metric-value {{
            display: block;
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .metric-label {{
            font-size: 11px;
            color: var(--text-muted);
        }}
        
        .study-footer {{
            display: flex;
            justify-content: space-between;
            padding-top: 12px;
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-muted);
        }}
        
        /* Query Response */
        .query-response {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--border-color);
            display: none;
        }}
        
        .query-response.visible {{
            display: block;
            animation: fadeIn 0.3s ease;
        }}
        
        .response-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        
        .response-content {{
            font-size: 14px;
            line-height: 1.7;
            color: var(--text-secondary);
        }}
        
        .response-content h2 {{
            font-size: 18px;
            margin-bottom: 12px;
            color: var(--text-primary);
        }}
        
        .response-content h3 {{
            font-size: 15px;
            margin: 16px 0 8px;
            color: var(--text-primary);
        }}
        
        .response-content table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
        }}
        
        .response-content th, .response-content td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .response-content th {{
            background: var(--bg-tertiary);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Loading */
        .loading {{
            display: none;
            text-align: center;
            padding: 40px;
        }}
        
        .loading.visible {{
            display: block;
        }}
        
        .spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid var(--border-color);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        /* Responsive */
        @media (max-width: 1200px) {{
            .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .charts-section {{ grid-template-columns: 1fr; }}
            .insights-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        
        @media (max-width: 768px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .insights-grid {{ grid-template-columns: 1fr; }}
            .study-metrics {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <div class="logo">
                <i class="fas fa-project-diagram"></i>
                <div>
                    <h1>Master Dashboard</h1>
                    <span>Clinical Dataflow Optimizer - Portfolio View</span>
                </div>
            </div>
            <div>
                <span style="color: var(--text-muted);">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
            </div>
        </div>
    </header>
    
    <!-- Query Section -->
    <section class="query-section">
        <div class="query-container">
            <div class="query-bar">
                <input type="text" class="query-input" id="queryInput" 
                       placeholder="Ask anything about your clinical trials... (e.g., 'Which sites have the highest risk?')">
                <button class="query-btn" onclick="submitQuery()">
                    <i class="fas fa-search"></i> Ask
                </button>
            </div>
            <div class="quick-queries">
                <span class="quick-query" onclick="setQuery('Show me all critical sites')">Critical Sites</span>
                <span class="quick-query" onclick="setQuery('Compare all studies')">Compare Studies</span>
                <span class="quick-query" onclick="setQuery('Which patients need attention?')">Patients at Risk</span>
                <span class="quick-query" onclick="setQuery('What is the data quality overview?')">Quality Overview</span>
                <span class="quick-query" onclick="setQuery('Give me a portfolio summary')">Portfolio Summary</span>
            </div>
        </div>
    </section>
    
    <!-- Main Content -->
    <main class="main-content">
        <!-- Query Response -->
        <div class="query-response" id="queryResponse">
            <div class="response-header">
                <h3><i class="fas fa-robot" style="color: var(--primary);"></i> AI Response</h3>
                <button onclick="closeResponse()" style="background: none; border: none; color: var(--text-muted); cursor: pointer;">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="response-content" id="responseContent"></div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing across {totals['studies']} studies...</p>
        </div>
        
        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon" style="background: rgba(59, 130, 246, 0.2); color: var(--primary);">
                    <i class="fas fa-flask"></i>
                </div>
                <div class="kpi-value">{totals['studies']}</div>
                <div class="kpi-label">Total Studies</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background: rgba(139, 92, 246, 0.2); color: var(--purple);">
                    <i class="fas fa-users"></i>
                </div>
                <div class="kpi-value">{totals['patients']:,}</div>
                <div class="kpi-label">Total Patients</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background: rgba(16, 185, 129, 0.2); color: var(--success);">
                    <i class="fas fa-check-circle"></i>
                </div>
                <div class="kpi-value">{totals['clean_patients']:,}</div>
                <div class="kpi-label">Clean Patients</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background: rgba(245, 158, 11, 0.2); color: var(--warning);">
                    <i class="fas fa-percentage"></i>
                </div>
                <div class="kpi-value">{totals['clean_rate']:.1f}%</div>
                <div class="kpi-label">Clean Rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background: rgba(6, 182, 212, 0.2); color: #06b6d4;">
                    <i class="fas fa-hospital"></i>
                </div>
                <div class="kpi-value">{totals['sites']:,}</div>
                <div class="kpi-label">Total Sites</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background: rgba(239, 68, 68, 0.2); color: var(--danger);">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="kpi-value">{totals['recommendations']:,}</div>
                <div class="kpi-label">Active Recommendations</div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-section">
            <div class="chart-card">
                <h3><i class="fas fa-chart-bar" style="color: var(--primary);"></i> Data Quality Index by Study</h3>
                <div id="dqiChart"></div>
            </div>
            <div class="chart-card">
                <h3><i class="fas fa-chart-pie" style="color: var(--purple);"></i> Patient Distribution</h3>
                <div id="patientChart"></div>
            </div>
        </div>
        
        <!-- Cross-Study Insights -->
        <section class="insights-section">
            <div class="section-header">
                <h2 class="section-title"><i class="fas fa-lightbulb" style="color: var(--warning);"></i> Cross-Study Insights</h2>
            </div>
            <div class="insights-grid">
                {insights_html}
            </div>
        </section>
        
        <!-- Studies Grid -->
        <section class="studies-section">
            <div class="section-header">
                <h2 class="section-title"><i class="fas fa-folder-open" style="color: var(--primary);"></i> All Studies</h2>
            </div>
            <div class="studies-grid">
                {study_cards_html}
            </div>
        </section>
    </main>
    
    <script>
        // Chart data
        const studyLabels = {chart_labels};
        const dqiData = {chart_dqi};
        const patientData = {chart_patients};
        const cleanRates = {chart_clean_rates};
        
        // DQI Bar Chart
        Plotly.newPlot('dqiChart', [{{
            x: studyLabels,
            y: dqiData,
            type: 'bar',
            marker: {{
                color: dqiData.map(d => d >= 85 ? '#10b981' : d >= 70 ? '#f59e0b' : '#ef4444'),
                line: {{ color: 'rgba(255,255,255,0.2)', width: 1 }}
            }},
            hovertemplate: '<b>%{{x}}</b><br>DQI: %{{y:.1f}}%<extra></extra>'
        }}], {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {{ color: '#cbd5e1', family: 'Inter' }},
            margin: {{ t: 20, r: 20, b: 60, l: 50 }},
            xaxis: {{
                tickangle: -45,
                gridcolor: '#334155'
            }},
            yaxis: {{
                title: 'DQI (%)',
                gridcolor: '#334155',
                range: [0, 100]
            }},
            shapes: [{{
                type: 'line',
                x0: -0.5, x1: studyLabels.length - 0.5,
                y0: 85, y1: 85,
                line: {{ color: '#10b981', width: 2, dash: 'dash' }}
            }}]
        }}, {{ responsive: true }});
        
        // Patient Distribution Pie
        Plotly.newPlot('patientChart', [{{
            labels: studyLabels,
            values: patientData,
            type: 'pie',
            hole: 0.5,
            marker: {{
                colors: ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', 
                         '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#84cc16', '#a855f7']
            }},
            textinfo: 'label+percent',
            textfont: {{ size: 11 }},
            hovertemplate: '<b>%{{label}}</b><br>%{{value:,}} patients<br>%{{percent}}<extra></extra>'
        }}], {{
            paper_bgcolor: 'transparent',
            font: {{ color: '#cbd5e1', family: 'Inter' }},
            margin: {{ t: 20, r: 20, b: 20, l: 20 }},
            showlegend: false,
            annotations: [{{
                text: '{totals["patients"]:,}<br>Total',
                x: 0.5, y: 0.5,
                font: {{ size: 16, color: '#f8fafc' }},
                showarrow: false
            }}]
        }}, {{ responsive: true }});
        
        // Query functions
        function setQuery(query) {{
            document.getElementById('queryInput').value = query;
            submitQuery();
        }}
        
        function submitQuery() {{
            const query = document.getElementById('queryInput').value;
            if (!query.trim()) return;
            
            document.getElementById('loading').classList.add('visible');
            document.getElementById('queryResponse').classList.remove('visible');
            
            // Simulate API call (in production, this would call the Python backend)
            setTimeout(() => {{
                processQuery(query);
            }}, 1000);
        }}
        
        function processQuery(query) {{
            document.getElementById('loading').classList.remove('visible');
            
            // Simple query routing for demo
            let response = '';
            const q = query.toLowerCase();
            
            if (q.includes('critical') || q.includes('risk')) {{
                response = generateRiskResponse();
            }} else if (q.includes('compare')) {{
                response = generateComparisonResponse();
            }} else if (q.includes('quality') || q.includes('dqi')) {{
                response = generateQualityResponse();
            }} else if (q.includes('summary') || q.includes('overview')) {{
                response = generateSummaryResponse();
            }} else {{
                response = generateSummaryResponse();
            }}
            
            document.getElementById('responseContent').innerHTML = response;
            document.getElementById('queryResponse').classList.add('visible');
        }}
        
        function generateRiskResponse() {{
            return `
                <h2>Risk Analysis</h2>
                <p>Analyzing risk factors across all {totals['studies']} studies...</p>
                <h3>Key Findings</h3>
                <ul>
                    <li>{totals['recommendations']:,} active recommendations require attention</li>
                    <li>Portfolio clean rate is {totals['clean_rate']:.1f}%</li>
                </ul>
                <p><em>Click on individual study cards below for detailed risk analysis.</em></p>
            `;
        }}
        
        function generateComparisonResponse() {{
            return `
                <h2>Study Comparison</h2>
                <p>Comparing all {totals['studies']} studies in the portfolio...</p>
                <h3>Performance Ranking (by DQI)</h3>
                <table>
                    <tr><th>Study</th><th>DQI</th><th>Patients</th><th>Clean Rate</th></tr>
                    ${{studyLabels.map((s, i) => `<tr><td>${{s}}</td><td>${{dqiData[i].toFixed(1)}}%</td><td>${{patientData[i].toLocaleString()}}</td><td>${{cleanRates[i].toFixed(1)}}%</td></tr>`).join('')}}
                </table>
            `;
        }}
        
        function generateQualityResponse() {{
            const avgDqi = dqiData.reduce((a, b) => a + b, 0) / dqiData.length;
            return `
                <h2>Data Quality Overview</h2>
                <h3>Portfolio Quality Metrics</h3>
                <ul>
                    <li><strong>Average DQI:</strong> ${{avgDqi.toFixed(1)}}%</li>
                    <li><strong>Total Clean Patients:</strong> {totals['clean_patients']:,} / {totals['patients']:,}</li>
                    <li><strong>Overall Clean Rate:</strong> {totals['clean_rate']:.1f}%</li>
                </ul>
                <h3>Studies Below Target (DQI < 85%)</h3>
                <ul>
                    ${{studyLabels.filter((s, i) => dqiData[i] < 85).map((s, i) => `<li>${{s}}: ${{dqiData[studyLabels.indexOf(s)].toFixed(1)}}%</li>`).join('') || '<li>All studies meeting target!</li>'}}
                </ul>
            `;
        }}
        
        function generateSummaryResponse() {{
            return `
                <h2>Portfolio Summary</h2>
                <h3>Key Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Studies</td><td>{totals['studies']}</td></tr>
                    <tr><td>Total Patients</td><td>{totals['patients']:,}</td></tr>
                    <tr><td>Clean Patients</td><td>{totals['clean_patients']:,}</td></tr>
                    <tr><td>Clean Rate</td><td>{totals['clean_rate']:.1f}%</td></tr>
                    <tr><td>Total Sites</td><td>{totals['sites']:,}</td></tr>
                    <tr><td>Active Recommendations</td><td>{totals['recommendations']:,}</td></tr>
                </table>
            `;
        }}
        
        function closeResponse() {{
            document.getElementById('queryResponse').classList.remove('visible');
        }}
        
        function navigateToStudy(studyId) {{
            const dashboardPath = studyId + '_dashboard.html';
            window.open(dashboardPath, '_blank');
        }}
        
        // Enter key support
        document.getElementById('queryInput').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') submitQuery();
        }});
    </script>
</body>
</html>'''
    
    def generate_all_study_dashboards(self) -> Dict[str, str]:
        """Generate enhanced dashboards for all studies"""
        dashboard_paths = {}
        
        for study_id, study in self.studies.items():
            try:
                output_path = str(self.reports_path / f"{study_id}_dashboard.html")
                
                viz = create_full_dashboard(
                    twins=study.twins,
                    site_metrics=study.site_metrics,
                    study_metrics=study.study_metrics,
                    recommendations=[r.to_dict() for r in study.recommendations] if study.recommendations else None,
                    output_path=output_path
                )
                
                dashboard_paths[study_id] = output_path
                logger.info(f"Generated dashboard for {study_id}")
                
            except Exception as e:
                logger.error(f"Failed to generate dashboard for {study_id}: {e}")
        
        return dashboard_paths
    
    def print_summary(self):
        """Print portfolio summary to console"""
        print("\n" + "="*70)
        print("MASTER ORCHESTRATOR - PORTFOLIO SUMMARY")
        print("="*70)
        
        total_patients = sum(len(s.twins) for s in self.studies.values())
        total_clean = sum(sum(1 for t in s.twins if t.clean_status) for s in self.studies.values())
        total_sites = sum(len(s.site_metrics) for s in self.studies.values())
        
        print(f"\n📊 Portfolio Overview")
        print(f"   Studies: {len(self.studies)}")
        print(f"   Patients: {total_patients:,}")
        print(f"   Clean: {total_clean:,} ({total_clean/total_patients*100:.1f}%)")
        print(f"   Sites: {total_sites:,}")
        
        print(f"\n📈 Study Rankings (by DQI)")
        rankings = sorted(
            [(sid, s.study_metrics.global_dqi if s.study_metrics else 0) 
             for sid, s in self.studies.items()],
            key=lambda x: x[1], reverse=True
        )
        for i, (sid, dqi) in enumerate(rankings[:10], 1):
            print(f"   {i}. {sid}: {dqi:.1f}%")
        
        print(f"\n🔍 Cross-Study Insights: {len(self.cross_study_insights)}")
        for insight in self.cross_study_insights[:3]:
            print(f"   [{insight.severity}] {insight.title}")
        
        print("\n" + "="*70)


    # Real-time monitoring methods
    def start_real_time_monitoring(self):
        """Start real-time data monitoring"""
        self.real_time_monitor.add_alert_callback(self._handle_status_alert)
        self.real_time_monitor.start_monitoring()
        logger.info("Real-time monitoring started")

    def stop_real_time_monitoring(self):
        """Stop real-time data monitoring"""
        self.real_time_monitor.stop_monitoring()
        logger.info("Real-time monitoring stopped")

    def get_real_time_status(self) -> Dict[str, Any]:
        """Get current real-time monitoring status"""
        return self.real_time_monitor.get_current_status_summary()

    def get_patient_status_history(self, subject_id: str) -> List:
        """Get status history for a specific patient"""
        return self.real_time_monitor.get_patient_status_history(subject_id)

    def _handle_status_alert(self, update):
        """Handle patient status change alerts"""
        logger.info(f"ALERT: Patient {update.subject_id} status changed: "
                   f"{update.previous_status.name} -> {update.new_status.name}")
        logger.info(f"Reason: {update.trigger_reason}")
        if update.longcat_explanation:
            logger.info(f"AI Analysis: {update.longcat_explanation}")


def main():
    """Main entry point for master orchestration"""
    
    # Initialize orchestrator
    orchestrator = MasterOrchestrator()
    
    # Load all studies
    orchestrator.load_all_studies(parallel=False)  # Sequential for stability
    
    # Print summary
    orchestrator.print_summary()
    
    # Export all artifacts
    print("\n📁 Exporting artifacts...")
    export_paths = orchestrator.export_all_artifacts()
    
    # Generate all dashboards
    print("\n📊 Generating dashboards...")
    dashboard_paths = orchestrator.generate_all_study_dashboards()
    
    print(f"\n✅ Master dashboard: {export_paths.get('master_dashboard', 'N/A')}")
    print(f"✅ Study dashboards: {len(dashboard_paths)}")
    
    # Demo query
    print("\n" + "="*70)
    print("QUERY INTERFACE DEMO")
    print("="*70)
    
    questions = [
        "Give me a portfolio summary",
        "Which sites have the highest risk?",
        "Compare all studies by data quality"
    ]
    
    for q in questions:
        print(f"\n❓ {q}")
        response = orchestrator.ask(q)
        print(response['answer'][:500] + "..." if len(response['answer']) > 500 else response['answer'])
    
    return orchestrator


if __name__ == "__main__":
    main()
