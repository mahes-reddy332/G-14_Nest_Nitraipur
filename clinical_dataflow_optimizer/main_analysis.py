"""
Clinical Dataflow Optimizer - Main Analysis Pipeline
Neural Clinical Data Mesh Implementation

This script orchestrates the complete analysis pipeline:
1. Data Ingestion from all study files
2. Knowledge Graph construction (using NetworkX as offline graph DB)
3. Digital Patient Twin construction
4. Clean Patient Status and DQI calculation
5. Multi-hop graph queries for complex analysis
6. Agentic AI analysis (Rex, Codex, Lia)
7. Dashboard generation and reporting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json
import logging
import sys

from clinical_dataflow_optimizer.core.data_ingestion import ClinicalDataIngester, load_all_clinical_data
from clinical_dataflow_optimizer.core.metrics_calculator import (
    PatientTwinBuilder, SiteMetricsAggregator, 
    StudyMetricsAggregator, DataQualityIndexCalculator,
    FeatureEnhancedTwinBuilder
)
from clinical_dataflow_optimizer.agents.agent_framework import SupervisorAgent
from clinical_dataflow_optimizer.visualization.dashboard import create_full_dashboard, DashboardVisualizer
from clinical_dataflow_optimizer.models.data_models import DigitalPatientTwin, SiteMetrics, StudyMetrics

# Import Knowledge Graph components (Neural Clinical Data Mesh)
from clinical_dataflow_optimizer.graph.knowledge_graph import ClinicalKnowledgeGraph, NodeType, EdgeType
from clinical_dataflow_optimizer.graph.graph_builder import ClinicalGraphBuilder, build_knowledge_graph_from_study
from clinical_dataflow_optimizer.graph.graph_queries import GraphQueryEngine, QueryCondition, QueryOperator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalDataflowAnalyzer:
    """
    Main orchestrator for clinical dataflow analysis
    Implements the Neural Clinical Data Mesh architecture
    
    Key Features:
    - Transforms flat CSV data into multi-dimensional knowledge graph
    - Patient-centric data model with Subject ID as central anchor
    - Multi-hop graph queries for complex analysis scenarios
    - Digital Patient Twin construction
    - AI Agent-based recommendations
    """
    
    def __init__(self, data_path: str, enable_graph: bool = True):
        self.data_path = Path(data_path)
        self.enable_graph = enable_graph
        self.ingester = None
        self.studies_data = {}
        self.study_results = {}
        
        # Knowledge Graph components (Neural Clinical Data Mesh)
        self.knowledge_graphs: Dict[str, ClinicalKnowledgeGraph] = {}
        self.query_engines: Dict[str, GraphQueryEngine] = {}
        
        # Initialize components
        self.twin_builder = FeatureEnhancedTwinBuilder()
        self.site_aggregator = SiteMetricsAggregator()
        self.study_aggregator = StudyMetricsAggregator()
        self.supervisor = SupervisorAgent()
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Execute the complete analysis pipeline for all studies
        """
        logger.info("="*60)
        logger.info("NEURAL CLINICAL DATA MESH - ANALYSIS STARTED")
        logger.info("="*60)
        
        # Step 1: Ingest all data
        logger.info("\n[STEP 1] DATA INGESTION")
        self.ingester, self.studies_data = load_all_clinical_data(str(self.data_path))
        
        # Step 2: Process each study
        for study_id, study_data in self.studies_data.items():
            logger.info(f"\n[STEP 2] PROCESSING {study_id}")
            
            # Build Knowledge Graph (Data Mesh transformation)
            if self.enable_graph:
                logger.info(f"  Building Knowledge Graph for {study_id}...")
                self._build_study_knowledge_graph(study_id, study_data)
            
            result = self._analyze_study(study_id, study_data)
            self.study_results[study_id] = result
        
        # Step 3: Generate aggregate report
        logger.info("\n[STEP 3] GENERATING AGGREGATE REPORT")
        aggregate_report = self._generate_aggregate_report()
        
        # Step 4: Run cross-study graph analysis
        if self.enable_graph:
            logger.info("\n[STEP 4] RUNNING GRAPH-BASED ANALYSIS")
            graph_insights = self._run_graph_analysis()
            aggregate_report['graph_insights'] = graph_insights
        
        return {
            'studies': self.study_results,
            'aggregate': aggregate_report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_study_knowledge_graph(
        self, 
        study_id: str, 
        study_data: Dict[str, pd.DataFrame]
    ) -> ClinicalKnowledgeGraph:
        """
        Build a knowledge graph for a study
        
        This transforms flat CSV data into a multi-dimensional graph structure
        where Patient is the central anchor node and all related data
        (visits, SAEs, coding issues, queries) are connected via edges.
        """
        try:
            builder = ClinicalGraphBuilder(study_id=study_id)
            graph = builder.build_from_study_data(study_data, study_id)
            
            self.knowledge_graphs[study_id] = graph
            self.query_engines[study_id] = GraphQueryEngine(graph)
            
            # Log statistics
            stats = graph.get_statistics()
            logger.info(f"  Graph built: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
            
            return graph
            
        except Exception as e:
            logger.error(f"  Error building knowledge graph for {study_id}: {e}")
            return None
    
    def _run_graph_analysis(self) -> Dict[str, Any]:
        """
        Run graph-based analysis across all studies
        
        This demonstrates the power of graph traversal over SQL joins
        for complex multi-hop queries.
        """
        insights = {
            'patients_needing_attention': {},
            'multi_issue_patients': {},
            'site_aggregations': {},
            'query_examples': []
        }
        
        for study_id, query_engine in self.query_engines.items():
            try:
                # Example 1: Find patients needing attention
                # (Missing Visit AND Open Query AND Uncoded Term)
                attention_patients = query_engine.find_patients_needing_attention()
                insights['patients_needing_attention'][study_id] = len(attention_patients)
                
                # Example 2: Get issue summary
                issue_summary = query_engine.get_issue_summary()
                insights['multi_issue_patients'][study_id] = {
                    'total_patients': issue_summary['total_patients'],
                    'clean_patients': issue_summary['clean_patients'],
                    '3_plus_issues': issue_summary['patients_by_issue_count']['3_plus_issues'],
                    'critical_count': len(issue_summary['critical_patients'])
                }
                
                # Example 3: Site aggregation via graph
                site_agg = query_engine.aggregate_by_site()
                insights['site_aggregations'][study_id] = {
                    site_id: {
                        'total_patients': data['total_patients'],
                        'clean_rate': data.get('clean_rate', 0),
                        'attention_needed': data.get('patients_needing_attention', 0)
                    }
                    for site_id, data in site_agg.items()
                }
                
            except Exception as e:
                logger.warning(f"Graph analysis error for {study_id}: {e}")
                continue
        
        # Add example query descriptions
        insights['query_examples'] = [
            {
                'name': 'Multi-Condition Patient Search',
                'description': 'Find patients with Missing Visit AND Open Query AND Uncoded Term',
                'sql_equivalent': '3 table JOIN with complex WHERE clause',
                'graph_approach': 'Simple neighbor traversal from Patient node'
            },
            {
                'name': 'Patient 360 View',
                'description': 'Get all related data for a patient',
                'sql_equivalent': 'Multiple UNION or separate queries',
                'graph_approach': 'Single BFS traversal from Patient node'
            },
            {
                'name': 'Risk-Based Prioritization',
                'description': 'Prioritize patients by issue severity across data sources',
                'sql_equivalent': 'Complex subqueries with scoring',
                'graph_approach': 'Graph centrality + neighbor property aggregation'
            }
        ]
        
        return insights
    
    def _analyze_study(self, study_id: str, study_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze a single study"""
        result = {
            'study_id': study_id,
            'twins': [],
            'site_metrics': {},
            'study_metrics': None,
            'recommendations': [],
            'summary': {},
            'graph_stats': None  # Knowledge graph statistics
        }
        
        # Check if we have CPID metrics
        if 'cpid_metrics' not in study_data or study_data['cpid_metrics'] is None:
            logger.warning(f"No CPID metrics found for {study_id}, skipping...")
            return result
        
        # Build Digital Patient Twins
        logger.info(f"  Building Digital Patient Twins for {study_id}...")
        twins = self.twin_builder.build_all_twins(study_data, study_id)
        result['twins'] = twins
        
        if len(twins) == 0:
            logger.warning(f"  No patients found for {study_id}")
            return result
        
        # Aggregate site metrics
        logger.info(f"  Aggregating site metrics...")
        site_metrics = self.site_aggregator.aggregate_site_metrics(twins, study_id)
        result['site_metrics'] = site_metrics
        
        # Calculate study-level metrics
        logger.info(f"  Calculating study-level metrics...")
        study_metrics = self.study_aggregator.aggregate_study_metrics(
            site_metrics, twins, study_id
        )
        result['study_metrics'] = study_metrics
        
        # Run AI agents
        logger.info(f"  Running AI Agent analysis...")
        agent_results = self.supervisor.run_analysis(
            twins, site_metrics, study_data, study_id,
            knowledge_graph=self.knowledge_graphs.get(study_id),
            query_engine=self.query_engines.get(study_id)
        )
        result['recommendations'] = agent_results['prioritized']
        
        # Update Knowledge Graph with calculated metrics from twins
        if self.enable_graph and study_id in self.knowledge_graphs:
            logger.info(f"  Updating Knowledge Graph with calculated metrics...")
            self._update_graph_with_twin_metrics(study_id, twins)
            
            # Re-save the updated graph
            self.knowledge_graphs[study_id].save(f"graph_data/{study_id}_graph")
        
        # Generate summary
        result['summary'] = self._generate_study_summary(
            twins, site_metrics, study_metrics, agent_results
        )
        
        logger.info(f"  Completed {study_id}: {len(twins)} patients, {len(site_metrics)} sites")
        
        return result
    
    def _update_graph_with_twin_metrics(
        self, 
        study_id: str, 
        twins: List[DigitalPatientTwin]
    ) -> None:
        """
        Update graph patient nodes with calculated metrics from digital twins
        
        This bridges the gap between relational analytics and graph queries
        by ensuring graph nodes contain the actual calculated metrics.
        """
        if study_id not in self.knowledge_graphs:
            return
            
        graph = self.knowledge_graphs[study_id]
        updated_count = 0
        
        for twin in twins:
            try:
                # Find patient node by subject_id
                patient_node_id = graph.get_patient_node_id(twin.subject_id)
                if not patient_node_id:
                    continue
                    
                # Update patient node with calculated metrics
                update_data = {
                    'clean_status': twin.clean_status,
                    'clean_percentage': twin.clean_percentage,
                    'data_quality_index': twin.data_quality_index,
                    'missing_visits': twin.missing_visits,
                    'missing_pages': twin.missing_pages,
                    'open_queries': twin.open_queries,
                    'total_queries': twin.total_queries,
                    'uncoded_terms': twin.uncoded_terms,
                    'verification_pct': twin.verification_percentage,
                    'reconciliation_issues': twin.reconciliation_issues,
                    'protocol_deviations': twin.protocol_deviations
                }
                
                # Update the node in the graph
                graph.update_patient_node(patient_node_id, update_data)
                updated_count += 1
                
            except Exception as e:
                logger.warning(f"Error updating graph node for patient {twin.subject_id}: {e}")
                
        logger.info(f"  Updated {updated_count} patient nodes in knowledge graph")
    
    def _generate_study_summary(
        self,
        twins: List[DigitalPatientTwin],
        site_metrics: Dict[str, SiteMetrics],
        study_metrics: StudyMetrics,
        agent_results: Dict
    ) -> Dict:
        """Generate summary statistics for a study"""
        
        clean_count = sum(1 for t in twins if t.clean_status)
        
        # Count blocking items by type
        blocking_summary = {}
        for twin in twins:
            for item in twin.blocking_items:
                cat = item.item_type
                if cat not in blocking_summary:
                    blocking_summary[cat] = 0
                blocking_summary[cat] += 1
        
        # Site risk distribution
        risk_dist = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        for site in site_metrics.values():
            risk_dist[site.risk_level.value] += 1
        
        return {
            'total_patients': len(twins),
            'clean_patients': clean_count,
            'clean_rate': round(clean_count / len(twins) * 100, 1) if twins else 0,
            'total_sites': len(site_metrics),
            'global_dqi': study_metrics.global_dqi,
            'interim_ready': study_metrics.interim_analysis_ready,
            'blocking_items_summary': blocking_summary,
            'site_risk_distribution': risk_dist,
            'agent_summary': self.supervisor.get_summary(),
            'top_issues': self._get_top_issues(twins)
        }
    
    def _get_top_issues(self, twins: List[DigitalPatientTwin]) -> List[Dict]:
        """Identify top issues across all patients"""
        
        # Aggregate metrics
        total_missing_visits = sum(t.missing_visits for t in twins)
        total_missing_pages = sum(t.missing_pages for t in twins)
        total_open_queries = sum(t.open_queries for t in twins)
        total_uncoded = sum(t.uncoded_terms for t in twins)
        total_recon = sum(t.reconciliation_issues for t in twins)
        
        issues = [
            {'issue': 'Open Queries', 'count': total_open_queries, 'severity': 'High'},
            {'issue': 'Missing Visits', 'count': total_missing_visits, 'severity': 'High'},
            {'issue': 'Missing Pages', 'count': total_missing_pages, 'severity': 'Medium'},
            {'issue': 'Uncoded Terms', 'count': total_uncoded, 'severity': 'Medium'},
            {'issue': 'Reconciliation Issues', 'count': total_recon, 'severity': 'Critical'}
        ]
        
        return sorted(issues, key=lambda x: x['count'], reverse=True)
    
    def _generate_aggregate_report(self) -> Dict:
        """Generate aggregate report across all studies"""
        
        total_patients = 0
        total_clean = 0
        total_sites = 0
        all_recommendations = []
        
        for study_id, result in self.study_results.items():
            if result['study_metrics']:
                total_patients += result['study_metrics'].total_patients
                total_clean += result['study_metrics'].clean_patients
                total_sites += result['study_metrics'].total_sites
            
            all_recommendations.extend(result['recommendations'])
        
        return {
            'total_studies': len(self.study_results),
            'total_patients': total_patients,
            'total_clean_patients': total_clean,
            'overall_clean_rate': round(total_clean / total_patients * 100, 1) if total_patients > 0 else 0,
            'total_sites': total_sites,
            'total_recommendations': len(all_recommendations),
            'critical_recommendations': sum(
                1 for r in all_recommendations 
                if hasattr(r, 'priority') and r.priority.name == 'CRITICAL'
            )
        }
    
    def generate_dashboards(self, output_dir: str = None) -> Dict[str, str]:
        """Generate interactive dashboards for all studies"""
        
        if output_dir is None:
            output_dir = self.data_path.parent / 'reports'
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        dashboard_paths = {}
        
        for study_id, result in self.study_results.items():
            if not result['twins']:
                continue
            
            try:
                viz = create_full_dashboard(
                    twins=result['twins'],
                    site_metrics=result['site_metrics'],
                    study_metrics=result['study_metrics'],
                    recommendations=[r.to_dict() for r in result['recommendations']] if result['recommendations'] else None,
                    output_path=str(output_path / f"{study_id}_dashboard.html")
                )
                dashboard_paths[study_id] = str(output_path / f"{study_id}_dashboard.html")
                logger.info(f"Generated dashboard for {study_id}")
            except Exception as e:
                logger.error(f"Error generating dashboard for {study_id}: {e}")
        
        return dashboard_paths
    
    def export_results_json(self, output_path: str = None) -> str:
        """Export all results to JSON"""
        
        if output_path is None:
            output_path = self.data_path.parent / 'reports' / 'analysis_results.json'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert results to serializable format
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'studies': {}
        }
        
        for study_id, result in self.study_results.items():
            study_export = {
                'study_id': study_id,
                'summary': result['summary'],
                'patients': [t.to_dict() for t in result['twins']],
                'sites': {sid: sm.to_dict() for sid, sm in result['site_metrics'].items()},
                'study_metrics': result['study_metrics'].to_dict() if result['study_metrics'] else None,
                'recommendations': [r.to_dict() for r in result['recommendations']]
            }
            export_data['studies'][study_id] = study_export
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported results to {output_path}")
        return str(output_path)
    
    def print_executive_summary(self):
        """Print executive summary to console"""
        
        print("\n" + "="*70)
        print("[REPORT] NEURAL CLINICAL DATA MESH - EXECUTIVE SUMMARY")
        print("="*70)
        
        for study_id, result in self.study_results.items():
            if not result['summary']:
                continue
            
            s = result['summary']
            print(f"\n{'-'*70}")
            print(f"[STUDY] {study_id}")
            print(f"{'-'*70}")
            
            # Key metrics
            print(f"\n  [KEY METRICS]:")
            print(f"     * Total Patients: {s['total_patients']}")
            print(f"     * Clean Patients: {s['clean_patients']} ({s['clean_rate']}%)")
            print(f"     * Global DQI: {s['global_dqi']}")
            print(f"     * Interim Analysis Ready: {'YES' if s['interim_ready'] else 'NO'}")
            
            # Site distribution
            print(f"\n  [SITE RISK DISTRIBUTION]:")
            for risk, count in s['site_risk_distribution'].items():
                marker = {'Critical': '[!]', 'High': '[H]', 'Medium': '[M]', 'Low': '[L]'}
                print(f"     {marker.get(risk, '[ ]')} {risk}: {count} sites")
            
            # Top issues
            print(f"\n  [TOP ISSUES]:")
            for issue in s['top_issues'][:5]:
                if issue['count'] > 0:
                    print(f"     * {issue['issue']}: {issue['count']}")
            
            # Agent recommendations
            agent_sum = s.get('agent_summary', {})
            print(f"\n  [AI AGENT RECOMMENDATIONS]:")
            print(f"     * Total Recommendations: {agent_sum.get('total_recommendations', 0)}")
            print(f"     * Auto-Executable: {agent_sum.get('auto_executable', 0)}")
            print(f"     * Requires Approval: {agent_sum.get('requires_approval', 0)}")
            
            by_priority = agent_sum.get('by_priority', {})
            if by_priority:
                print(f"     * Critical: {by_priority.get('Critical', 0)}")
                print(f"     * High: {by_priority.get('High', 0)}")
        
        print("\n" + "="*70)
        print("[COMPLETE] ANALYSIS COMPLETE")
        print("="*70 + "\n")
    
    # ==================== Knowledge Graph Query Methods ====================
    
    def query_patients_needing_attention(
        self, 
        study_id: str = None
    ) -> Dict[str, List[Dict]]:
        """
        Multi-hop graph query: Find patients with multiple issues
        
        This query demonstrates the power of the knowledge graph:
        "Show me all patients who have a Missing Visit AND an Open Query AND an Uncoded Term"
        
        In SQL, this requires joining 3+ tables with complex WHERE clauses.
        In the graph, it's a simple neighbor traversal from Patient nodes.
        """
        results = {}
        
        engines = {study_id: self.query_engines[study_id]} if study_id else self.query_engines
        
        for sid, engine in engines.items():
            try:
                attention_patients = engine.find_patients_needing_attention()
                results[sid] = [p.to_dict() for p in attention_patients]
            except Exception as e:
                logger.error(f"Query error for {sid}: {e}")
                results[sid] = []
        
        return results
    
    def get_patient_360_view(self, subject_id: str, study_id: str) -> Dict[str, Any]:
        """
        Get complete 360-degree view of a patient via graph traversal
        
        Returns all connected data: visits, SAEs, coding issues, queries
        """
        if study_id not in self.query_engines:
            return {'error': f'Study {study_id} not found'}
        
        return self.query_engines[study_id].get_patient_360_view(subject_id)
    
    def execute_custom_mesh_query(
        self,
        study_id: str,
        query_spec: Dict[str, Any]
    ) -> List[Dict]:
        """
        Execute a custom Data Mesh query
        
        Query spec format:
        {
            "patient_filters": [{"field": "missing_visits", "op": "gt", "value": 0}],
            "required_relationships": ["HAS_VISIT", "HAS_QUERY"],
            "neighbor_filters": {"HAS_QUERY": [{"field": "status", "op": "eq", "value": "Open"}]},
            "logic": "AND"
        }
        """
        if study_id not in self.query_engines:
            return []
        
        results = self.query_engines[study_id].execute_mesh_query(query_spec)
        return [r.to_dict() for r in results]
    
    def save_knowledge_graphs(self, output_dir: str = None) -> Dict[str, str]:
        """Save all knowledge graphs to disk"""
        if output_dir is None:
            output_dir = self.data_path.parent / 'graph_data'
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_paths = {}
        for study_id, graph in self.knowledge_graphs.items():
            try:
                filepath = output_path / f"{study_id}_graph"
                graph.save(filepath)
                saved_paths[study_id] = str(filepath)
                logger.info(f"Saved knowledge graph for {study_id}")
            except Exception as e:
                logger.error(f"Error saving graph for {study_id}: {e}")
        
        return saved_paths
    
    def get_graph_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all knowledge graphs"""
        return {
            study_id: graph.get_statistics()
            for study_id, graph in self.knowledge_graphs.items()
        }


def main():
    """Main entry point"""
    
    # Define data path
    DATA_PATH = Path(__file__).parent.parent / "QC Anonymized Study Files"
    
    if not DATA_PATH.exists():
        logger.error(f"Data path not found: {DATA_PATH}")
        return
    
    # Initialize analyzer with Knowledge Graph enabled
    analyzer = ClinicalDataflowAnalyzer(str(DATA_PATH), enable_graph=True)
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    # Print executive summary
    analyzer.print_executive_summary()
    
    # Print Knowledge Graph summary
    print("\n" + "="*70)
    print("[KNOWLEDGE GRAPH] NEURAL CLINICAL DATA MESH STATISTICS")
    print("="*70)
    
    for study_id, stats in analyzer.get_graph_statistics().items():
        print(f"\n{study_id}:")
        print(f"  Nodes: {stats.get('total_nodes', 0)}")
        print(f"  Edges: {stats.get('total_edges', 0)}")
        if 'nodes_by_type' in stats:
            print(f"  Node Types: {stats['nodes_by_type']}")
    
    # Demonstrate multi-hop query
    print("\n" + "-"*70)
    print("[QUERY] Patients Needing Attention (Multi-Hop Graph Query)")
    print("-"*70)
    attention_results = analyzer.query_patients_needing_attention()
    for study_id, patients in attention_results.items():
        if patients:
            print(f"\n{study_id}: {len(patients)} patients with multiple issues")
            for p in patients[:3]:  # Show top 3
                print(f"  - {p['subject_id']}: Risk Score {p['risk_score']}, Priority #{p['priority_rank']}")
    
    # Generate dashboards
    dashboard_paths = analyzer.generate_dashboards()
    
    # Export results
    json_path = analyzer.export_results_json()
    
    # Save knowledge graphs
    graph_paths = analyzer.save_knowledge_graphs()
    
    print(f"\n📁 Reports generated:")
    print(f"   JSON: {json_path}")
    for study_id, path in dashboard_paths.items():
        print(f"   Dashboard ({study_id}): {path}")
    print(f"\n📊 Knowledge Graphs saved:")
    for study_id, path in graph_paths.items():
        print(f"   Graph ({study_id}): {path}")
    
    return results


if __name__ == "__main__":
    main()
