"""
Generate all study artifacts (knowledge graphs, multi-hop queries, features)
and create a master dashboard for portfolio-wide insights.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clinical_dataflow_optimizer.core.data_ingestion import load_all_clinical_data
from clinical_dataflow_optimizer.core.metrics_calculator import (
    PatientTwinBuilder, SiteMetricsAggregator, StudyMetricsAggregator
)
from clinical_dataflow_optimizer.agents.agent_framework import SupervisorAgent
from clinical_dataflow_optimizer.graph.knowledge_graph import ClinicalKnowledgeGraph
from clinical_dataflow_optimizer.graph.graph_builder import ClinicalGraphBuilder
from clinical_dataflow_optimizer.graph.graph_queries import GraphQueryEngine
from clinical_dataflow_optimizer.graph.graph_analytics import GraphAnalytics


def generate_study_artifacts(study_id: str, study_data: Dict, output_dir: Path) -> Dict:
    """Generate all artifacts for a single study"""
    
    if 'cpid_metrics' not in study_data or study_data['cpid_metrics'] is None:
        logger.warning(f"No CPID metrics for {study_id}, skipping")
        return None
    
    logger.info(f"Processing {study_id}...")
    
    # Build Digital Twins
    twin_builder = PatientTwinBuilder()
    site_aggregator = SiteMetricsAggregator()
    study_aggregator = StudyMetricsAggregator()
    
    twins = twin_builder.build_all_twins(study_data, study_id)
    if not twins:
        return None
    
    # Aggregate metrics
    site_metrics = site_aggregator.aggregate_site_metrics(twins, study_id)
    study_metrics = study_aggregator.aggregate_study_metrics(site_metrics, twins, study_id)
    
    # Build Knowledge Graph
    builder = ClinicalGraphBuilder(study_id=study_id)
    knowledge_graph = builder.build_from_study_data(study_data, study_id)
    query_engine = GraphQueryEngine(knowledge_graph)
    graph_analytics = GraphAnalytics(knowledge_graph)
    
    # Run multi-hop queries
    multi_hop_queries = run_multi_hop_queries(query_engine, study_id)
    
    # Get patient risk profiles
    patient_risk_profiles = []
    site_risk_profiles = []
    
    # Handle twins - it might be a list or dict
    twins_dict = twins if isinstance(twins, dict) else {t.subject_id: t for t in twins if hasattr(t, 'subject_id')}
    
    try:
        for patient_id in list(twins_dict.keys())[:100]:  # Limit for performance
            profile = graph_analytics.get_patient_risk_profile(patient_id)
            if profile:
                patient_risk_profiles.append({
                    'patient_id': profile.patient_id,
                    'risk_score': profile.risk_score,
                    'risk_factors': profile.risk_factors[:3] if profile.risk_factors else [],
                    'recommendations': profile.recommendations[:2] if profile.recommendations else []
                })
        
        for site_id in list(site_metrics.keys())[:50]:
            profile = graph_analytics.get_site_risk_profile(site_id)
            if profile:
                site_risk_profiles.append({
                    'site_id': profile.site_id,
                    'risk_score': profile.risk_score,
                    'risk_factors': profile.risk_factors[:3] if profile.risk_factors else []
                })
    except Exception as e:
        logger.warning(f"Error getting risk profiles for {study_id}: {e}")
    
    # Get graph statistics
    graph_stats = knowledge_graph.get_statistics()
    
    # Calculate twin count
    twin_count = len(twins_dict) if isinstance(twins_dict, dict) else len(twins) if twins else 0
    
    # Save artifacts  
    artifacts = {
        'study_id': study_id,
        'patient_count': twin_count,
        'site_count': len(site_metrics) if isinstance(site_metrics, dict) else 0,
        'clean_rate': study_metrics.global_clean_rate if study_metrics else 0,
        'avg_dqi': study_metrics.global_dqi if study_metrics else 0,
        'open_queries': getattr(study_metrics, 'total_open_queries', 0) if study_metrics else 0,
        'pending_pages': getattr(study_metrics, 'total_pending_pages', 0) if study_metrics else 0,
        'graph_stats': graph_stats,
        'multi_hop_queries': multi_hop_queries,
        'patient_risk_profiles': patient_risk_profiles[:20],
        'site_risk_profiles': site_risk_profiles[:20]
    }
    
    # Save knowledge graph
    kg_path = output_dir / f"{study_id}_knowledge_graph.json"
    kg_data = {
        'study_id': study_id,
        'nodes': graph_stats.get('node_count', 0),
        'edges': graph_stats.get('edge_count', 0),
        'node_types': graph_stats.get('nodes_by_type', {}),
        'edge_types': graph_stats.get('edges_by_type', {}),
        'generated_at': datetime.now().isoformat()
    }
    with open(kg_path, 'w') as f:
        json.dump(kg_data, f, indent=2, default=str)
    logger.info(f"  Saved {kg_path.name}")
    
    # Save multi-hop queries
    mhq_path = output_dir / f"{study_id}_multi_hop_queries.json"
    with open(mhq_path, 'w') as f:
        json.dump(multi_hop_queries, f, indent=2, default=str)
    logger.info(f"  Saved {mhq_path.name}")
    
    # Save features
    features_path = output_dir / f"{study_id}_features.json"
    features_data = {
        'study_id': study_id,
        'patient_risk_profiles': patient_risk_profiles,
        'site_risk_profiles': site_risk_profiles,
        'study_metrics': {
            'clean_rate': study_metrics.global_clean_rate if study_metrics else 0,
            'avg_dqi': study_metrics.global_dqi if study_metrics else 0,
            'open_queries': getattr(study_metrics, 'total_open_queries', 0) if study_metrics else 0,
            'pending_pages': getattr(study_metrics, 'total_pending_pages', 0) if study_metrics else 0
        },
        'generated_at': datetime.now().isoformat()
    }
    with open(features_path, 'w') as f:
        json.dump(features_data, f, indent=2, default=str)
    logger.info(f"  Saved {features_path.name}")
    
    # Save feature matrix (site-level)
    feature_matrix_path = output_dir / f"{study_id}_feature_matrix.csv"
    site_features = []
    if isinstance(site_metrics, dict):
        for site_id, site in site_metrics.items():
            site_features.append({
                'study_id': study_id,
                'site_id': site_id,
                'patient_count': getattr(site, 'patient_count', 0),
                'clean_rate': getattr(site, 'clean_patient_rate', 0),
                'dqi': getattr(site, 'data_quality_index', 0),
                'open_queries': getattr(site, 'total_open_queries', 0),
                'pending_pages': getattr(site, 'total_pending_pages', 0),
                'risk_level': site.risk_level.value if hasattr(site, 'risk_level') and site.risk_level else 'Unknown'
            })
    if site_features:
        pd.DataFrame(site_features).to_csv(feature_matrix_path, index=False)
        logger.info(f"  Saved {feature_matrix_path.name}")
    
    # Save patient network
    network_path = output_dir / f"{study_id}_patient_network.csv"
    patient_network = []
    twins_items = list(twins_dict.items())[:500] if isinstance(twins_dict, dict) else []
    for patient_id, twin in twins_items:
        patient_network.append({
            'patient_id': patient_id,
            'site_id': getattr(twin, 'site_id', ''),
            'is_clean': getattr(twin, 'is_clean', False),
            'open_queries': getattr(twin, 'open_queries', 0),
            'pending_pages': getattr(twin, 'pending_pages', 0),
            'risk_score': getattr(twin, 'risk_score', 0)
        })
    if patient_network:
        pd.DataFrame(patient_network).to_csv(network_path, index=False)
        logger.info(f"  Saved {network_path.name}")
    
    return artifacts


def run_multi_hop_queries(query_engine: GraphQueryEngine, study_id: str) -> Dict:
    """Run standard multi-hop queries on the knowledge graph"""
    results = {}
    
    try:
        # Query 1: Patients needing attention
        patients = query_engine.find_patients_needing_attention()
        results['patients_needing_attention'] = [
            {'patient_id': p.patient_id, 'site_id': p.site_id, 'issues': p.related_issues[:3] if p.related_issues else []}
            for p in patients[:20]
        ]
    except Exception as e:
        results['patients_needing_attention'] = []
        logger.warning(f"Patients needing attention query failed for {study_id}: {e}")
    
    try:
        # Query 2: Non-clean patients
        non_clean = query_engine.find_non_clean_patients()
        results['non_clean_patients'] = [
            {'patient_id': p.patient_id, 'site_id': p.site_id}
            for p in non_clean[:20]
        ]
    except Exception as e:
        results['non_clean_patients'] = []
        logger.warning(f"Non-clean patients query failed for {study_id}: {e}")
    
    try:
        # Query 3: Patients with uncoded terms
        uncoded = query_engine.find_patients_with_uncoded_terms()
        results['patients_with_uncoded_terms'] = [
            {'patient_id': p.patient_id, 'site_id': p.site_id}
            for p in uncoded[:20]
        ]
    except Exception as e:
        results['patients_with_uncoded_terms'] = []
        logger.warning(f"Patients with uncoded terms query failed for {study_id}: {e}")
    
    try:
        # Query 4: Issue summary
        results['issue_summary'] = query_engine.get_issue_summary()
    except Exception as e:
        results['issue_summary'] = {}
        logger.warning(f"Issue summary query failed for {study_id}: {e}")
    
    try:
        # Query 5: Site aggregation
        results['site_aggregation'] = query_engine.aggregate_by_site()
    except Exception as e:
        results['site_aggregation'] = {}
        logger.warning(f"Site aggregation query failed for {study_id}: {e}")
    
    return results


def generate_master_dashboard(all_artifacts: List[Dict], output_dir: Path):
    """Generate a master dashboard aggregating all studies"""
    
    # Calculate portfolio-wide metrics
    total_patients = sum(a['patient_count'] for a in all_artifacts)
    total_sites = sum(a['site_count'] for a in all_artifacts)
    avg_clean_rate = sum(a['clean_rate'] for a in all_artifacts) / len(all_artifacts) if all_artifacts else 0
    avg_dqi = sum(a['avg_dqi'] for a in all_artifacts) / len(all_artifacts) if all_artifacts else 0
    total_open_queries = sum(a['open_queries'] for a in all_artifacts)
    total_pending_pages = sum(a['pending_pages'] for a in all_artifacts)
    
    # Sort studies by DQI for insights
    sorted_by_dqi = sorted(all_artifacts, key=lambda x: x['avg_dqi'])
    lowest_dqi_studies = sorted_by_dqi[:5]
    highest_dqi_studies = sorted_by_dqi[-5:][::-1]
    
    # Studies with most open queries
    sorted_by_queries = sorted(all_artifacts, key=lambda x: x['open_queries'], reverse=True)
    highest_query_studies = sorted_by_queries[:5]
    
    # Generate insights
    insights = []
    
    # Insight 1: Low DQI studies
    for study in lowest_dqi_studies:
        if study['avg_dqi'] < 70:
            insights.append({
                'type': 'warning',
                'study': study['study_id'],
                'message': f"DQI below threshold: {study['avg_dqi']:.1f}%",
                'severity': 'High' if study['avg_dqi'] < 50 else 'Medium'
            })
    
    # Insight 2: High query volume
    for study in highest_query_studies[:3]:
        if study['open_queries'] > 100:
            insights.append({
                'type': 'alert',
                'study': study['study_id'],
                'message': f"High open query volume: {study['open_queries']}",
                'severity': 'High'
            })
    
    # Insight 3: Portfolio health
    if avg_dqi >= 80:
        insights.append({
            'type': 'success',
            'study': 'Portfolio',
            'message': f"Portfolio DQI healthy at {avg_dqi:.1f}%",
            'severity': 'Low'
        })
    
    # Generate HTML
    html = generate_master_html(
        all_artifacts=all_artifacts,
        total_patients=total_patients,
        total_sites=total_sites,
        avg_clean_rate=avg_clean_rate,
        avg_dqi=avg_dqi,
        total_open_queries=total_open_queries,
        total_pending_pages=total_pending_pages,
        insights=insights,
        lowest_dqi_studies=lowest_dqi_studies,
        highest_dqi_studies=highest_dqi_studies
    )
    
    # Save dashboard
    dashboard_path = output_dir / "master_dashboard.html"
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Saved master dashboard: {dashboard_path}")
    
    # Save portfolio summary
    summary_path = output_dir / "portfolio_summary.json"
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_studies': len(all_artifacts),
        'total_patients': total_patients,
        'total_sites': total_sites,
        'avg_clean_rate': avg_clean_rate,
        'avg_dqi': avg_dqi,
        'total_open_queries': total_open_queries,
        'total_pending_pages': total_pending_pages,
        'insights': insights,
        'studies': [
            {
                'study_id': a['study_id'],
                'patient_count': a['patient_count'],
                'site_count': a['site_count'],
                'clean_rate': a['clean_rate'],
                'dqi': a['avg_dqi'],
                'open_queries': a['open_queries']
            }
            for a in all_artifacts
        ]
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved portfolio summary: {summary_path}")
    
    # Save cross-study insights
    insights_path = output_dir / "cross_study_insights.json"
    with open(insights_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'insights': insights,
            'lowest_dqi_studies': [s['study_id'] for s in lowest_dqi_studies],
            'highest_query_studies': [s['study_id'] for s in highest_query_studies],
            'portfolio_avg_dqi': avg_dqi,
            'portfolio_avg_clean_rate': avg_clean_rate
        }, f, indent=2)
    logger.info(f"Saved cross-study insights: {insights_path}")


def generate_master_html(all_artifacts, total_patients, total_sites, avg_clean_rate,
                         avg_dqi, total_open_queries, total_pending_pages, insights,
                         lowest_dqi_studies, highest_dqi_studies) -> str:
    """Generate the master dashboard HTML"""
    
    # Prepare chart data
    study_ids = [a['study_id'] for a in all_artifacts]
    clean_rates = [a['clean_rate'] for a in all_artifacts]
    dqi_values = [a['avg_dqi'] for a in all_artifacts]
    patient_counts = [a['patient_count'] for a in all_artifacts]
    query_counts = [a['open_queries'] for a in all_artifacts]
    
    # Generate insight cards HTML
    insight_cards = ""
    for insight in insights[:10]:
        color = {
            'warning': '#f39c12',
            'alert': '#e74c3c',
            'success': '#27ae60',
            'info': '#3498db'
        }.get(insight['type'], '#95a5a6')
        
        severity_badge = {
            'High': '<span style="background:#e74c3c;color:white;padding:2px 8px;border-radius:4px;font-size:11px;">HIGH</span>',
            'Medium': '<span style="background:#f39c12;color:white;padding:2px 8px;border-radius:4px;font-size:11px;">MEDIUM</span>',
            'Low': '<span style="background:#27ae60;color:white;padding:2px 8px;border-radius:4px;font-size:11px;">LOW</span>'
        }.get(insight['severity'], '')
        
        insight_cards += f'''
        <div style="background:white;border-radius:8px;padding:15px;margin-bottom:10px;border-left:4px solid {color};box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <strong style="color:#2c3e50;">{insight['study']}</strong>
                {severity_badge}
            </div>
            <p style="margin:8px 0 0 0;color:#7f8c8d;">{insight['message']}</p>
        </div>
        '''
    
    # Generate study cards HTML
    study_cards = ""
    for artifact in sorted(all_artifacts, key=lambda x: x['avg_dqi']):
        dqi_color = '#27ae60' if artifact['avg_dqi'] >= 80 else '#f39c12' if artifact['avg_dqi'] >= 60 else '#e74c3c'
        study_cards += f'''
        <div style="background:white;border-radius:8px;padding:15px;box-shadow:0 2px 4px rgba(0,0,0,0.1);min-width:250px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <h4 style="margin:0;color:#2c3e50;">{artifact['study_id']}</h4>
                <span style="background:{dqi_color};color:white;padding:4px 10px;border-radius:15px;font-size:12px;font-weight:bold;">{artifact['avg_dqi']:.1f}%</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div>
                    <small style="color:#7f8c8d;">Patients</small>
                    <p style="margin:0;font-size:18px;font-weight:bold;color:#2c3e50;">{artifact['patient_count']:,}</p>
                </div>
                <div>
                    <small style="color:#7f8c8d;">Sites</small>
                    <p style="margin:0;font-size:18px;font-weight:bold;color:#2c3e50;">{artifact['site_count']}</p>
                </div>
                <div>
                    <small style="color:#7f8c8d;">Clean Rate</small>
                    <p style="margin:0;font-size:18px;font-weight:bold;color:#2c3e50;">{artifact['clean_rate']:.1f}%</p>
                </div>
                <div>
                    <small style="color:#7f8c8d;">Open Queries</small>
                    <p style="margin:0;font-size:18px;font-weight:bold;color:#2c3e50;">{artifact['open_queries']}</p>
                </div>
            </div>
            <a href="{artifact['study_id']}_dashboard.html" style="display:block;margin-top:15px;text-align:center;padding:8px;background:#3498db;color:white;border-radius:5px;text-decoration:none;font-size:13px;">View Dashboard →</a>
        </div>
        '''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Trial Portfolio - Master Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f6fa; }}
        .header {{ background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 30px; }}
        .header h1 {{ margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .kpi-card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }}
        .kpi-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .kpi-label {{ color: #7f8c8d; margin-top: 5px; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .chart-card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .chart-card h3 {{ margin-bottom: 15px; color: #2c3e50; }}
        .insights-grid {{ display: grid; grid-template-columns: 1fr 2fr; gap: 20px; margin-bottom: 30px; }}
        .insights-panel {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .insights-panel h3 {{ margin-bottom: 15px; color: #2c3e50; }}
        .query-panel {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .query-panel h3 {{ margin-bottom: 15px; color: #2c3e50; }}
        .query-input {{ width: 100%; padding: 15px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 16px; margin-bottom: 15px; }}
        .query-input:focus {{ outline: none; border-color: #3498db; }}
        .query-examples {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }}
        .query-example {{ background: #ecf0f1; padding: 8px 15px; border-radius: 20px; cursor: pointer; font-size: 13px; transition: all 0.2s; }}
        .query-example:hover {{ background: #3498db; color: white; }}
        .query-result {{ background: #f8f9fa; border-radius: 8px; padding: 15px; min-height: 150px; }}
        .study-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .section-title {{ font-size: 24px; color: #2c3e50; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #3498db; }}
        @media (max-width: 768px) {{
            .insights-grid {{ grid-template-columns: 1fr; }}
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>🏥 Clinical Trial Portfolio Dashboard</h1>
            <p>Master orchestration view - {len(all_artifacts)} studies | {total_patients:,} patients | {total_sites:,} sites</p>
            <p style="font-size:12px;margin-top:10px;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
    
    <div class="container">
        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">{len(all_artifacts)}</div>
                <div class="kpi-label">Total Studies</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{total_patients:,}</div>
                <div class="kpi-label">Total Patients</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{total_sites:,}</div>
                <div class="kpi-label">Total Sites</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color: {'#27ae60' if avg_dqi >= 80 else '#f39c12' if avg_dqi >= 60 else '#e74c3c'};">{avg_dqi:.1f}%</div>
                <div class="kpi-label">Portfolio DQI</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color: {'#27ae60' if avg_clean_rate >= 80 else '#f39c12' if avg_clean_rate >= 60 else '#e74c3c'};">{avg_clean_rate:.1f}%</div>
                <div class="kpi-label">Avg Clean Rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color: {'#e74c3c' if total_open_queries > 1000 else '#f39c12' if total_open_queries > 500 else '#27ae60'};">{total_open_queries:,}</div>
                <div class="kpi-label">Open Queries</div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="chart-grid">
            <div class="chart-card">
                <h3>📊 Data Quality Index by Study</h3>
                <div id="dqi-chart"></div>
            </div>
            <div class="chart-card">
                <h3>👥 Patient Distribution by Study</h3>
                <div id="patient-chart"></div>
            </div>
        </div>
        
        <div class="chart-grid">
            <div class="chart-card">
                <h3>✅ Clean Rate Comparison</h3>
                <div id="clean-rate-chart"></div>
            </div>
            <div class="chart-card">
                <h3>❓ Open Queries by Study</h3>
                <div id="query-chart"></div>
            </div>
        </div>
        
        <!-- Insights and Query Panel -->
        <div class="insights-grid">
            <div class="insights-panel">
                <h3>🔔 Portfolio Insights</h3>
                {insight_cards if insight_cards else '<p style="color:#7f8c8d;">No critical insights at this time.</p>'}
            </div>
            <div class="query-panel">
                <h3>🔍 Ask Questions About Your Portfolio</h3>
                <input type="text" class="query-input" id="query-input" placeholder="Ask a question about your clinical trial portfolio...">
                <div class="query-examples">
                    <span class="query-example" onclick="setQuery('Which studies have the lowest DQI?')">Which studies have lowest DQI?</span>
                    <span class="query-example" onclick="setQuery('Show studies with most open queries')">Most open queries?</span>
                    <span class="query-example" onclick="setQuery('Which studies are ready for interim analysis?')">Ready for interim?</span>
                    <span class="query-example" onclick="setQuery('Compare all study clean rates')">Compare clean rates</span>
                    <span class="query-example" onclick="setQuery('Show critical sites across portfolio')">Critical sites?</span>
                </div>
                <div class="query-result" id="query-result">
                    <p style="color:#7f8c8d;">Ask a question about your portfolio above, or click one of the example queries.</p>
                </div>
            </div>
        </div>
        
        <!-- Study Cards -->
        <h2 class="section-title">📁 Individual Studies</h2>
        <div class="study-grid">
            {study_cards}
        </div>
    </div>
    
    <script>
        // Portfolio data for query handling
        const portfolioData = {json.dumps([{
            'study_id': a['study_id'],
            'patient_count': a['patient_count'],
            'site_count': a['site_count'],
            'clean_rate': a['clean_rate'],
            'dqi': a['avg_dqi'],
            'open_queries': a['open_queries'],
            'pending_pages': a['pending_pages']
        } for a in all_artifacts], indent=2)};
        
        // DQI Chart
        Plotly.newPlot('dqi-chart', [{{
            x: {json.dumps(study_ids)},
            y: {json.dumps(dqi_values)},
            type: 'bar',
            marker: {{
                color: {json.dumps(dqi_values)},
                colorscale: [[0, '#e74c3c'], [0.5, '#f39c12'], [1, '#27ae60']]
            }}
        }}], {{
            margin: {{t: 10, b: 80, l: 50, r: 20}},
            yaxis: {{title: 'DQI (%)', range: [0, 100]}},
            xaxis: {{tickangle: -45}}
        }}, {{responsive: true}});
        
        // Patient Distribution Chart
        Plotly.newPlot('patient-chart', [{{
            labels: {json.dumps(study_ids)},
            values: {json.dumps(patient_counts)},
            type: 'pie',
            hole: 0.4,
            textinfo: 'percent',
            hovertemplate: '%{{label}}: %{{value:,}} patients<extra></extra>'
        }}], {{
            margin: {{t: 10, b: 10, l: 10, r: 10}},
            showlegend: true,
            legend: {{orientation: 'h', y: -0.1}}
        }}, {{responsive: true}});
        
        // Clean Rate Chart
        Plotly.newPlot('clean-rate-chart', [{{
            x: {json.dumps(study_ids)},
            y: {json.dumps(clean_rates)},
            type: 'scatter',
            mode: 'markers+lines',
            marker: {{
                size: 12,
                color: {json.dumps(clean_rates)},
                colorscale: [[0, '#e74c3c'], [0.5, '#f39c12'], [1, '#27ae60']]
            }},
            line: {{color: '#3498db', width: 2}}
        }}], {{
            margin: {{t: 10, b: 80, l: 50, r: 20}},
            yaxis: {{title: 'Clean Rate (%)', range: [0, 100]}},
            xaxis: {{tickangle: -45}},
            shapes: [{{
                type: 'line',
                x0: 0,
                x1: 1,
                xref: 'paper',
                y0: 80,
                y1: 80,
                line: {{color: '#27ae60', width: 2, dash: 'dash'}}
            }}]
        }}, {{responsive: true}});
        
        // Open Queries Chart
        Plotly.newPlot('query-chart', [{{
            x: {json.dumps(study_ids)},
            y: {json.dumps(query_counts)},
            type: 'bar',
            marker: {{
                color: {json.dumps(query_counts)},
                colorscale: [[0, '#27ae60'], [0.5, '#f39c12'], [1, '#e74c3c']]
            }}
        }}], {{
            margin: {{t: 10, b: 80, l: 60, r: 20}},
            yaxis: {{title: 'Open Queries'}},
            xaxis: {{tickangle: -45}}
        }}, {{responsive: true}});
        
        // Query handling
        function setQuery(q) {{
            document.getElementById('query-input').value = q;
            handleQuery(q);
        }}
        
        document.getElementById('query-input').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                handleQuery(this.value);
            }}
        }});
        
        function handleQuery(query) {{
            const result = document.getElementById('query-result');
            const q = query.toLowerCase();
            
            if (q.includes('lowest dqi') || q.includes('low dqi')) {{
                const sorted = [...portfolioData].sort((a, b) => a.dqi - b.dqi).slice(0, 5);
                let html = '<h4 style="color:#2c3e50;margin-bottom:10px;">Studies with Lowest DQI:</h4><ul>';
                sorted.forEach(s => {{
                    html += `<li><strong>${{s.study_id}}</strong>: ${{s.dqi.toFixed(1)}}% DQI (${{s.patient_count.toLocaleString()}} patients)</li>`;
                }});
                html += '</ul>';
                result.innerHTML = html;
            }}
            else if (q.includes('most') && q.includes('quer')) {{
                const sorted = [...portfolioData].sort((a, b) => b.open_queries - a.open_queries).slice(0, 5);
                let html = '<h4 style="color:#2c3e50;margin-bottom:10px;">Studies with Most Open Queries:</h4><ul>';
                sorted.forEach(s => {{
                    html += `<li><strong>${{s.study_id}}</strong>: ${{s.open_queries.toLocaleString()}} open queries (${{s.dqi.toFixed(1)}}% DQI)</li>`;
                }});
                html += '</ul>';
                result.innerHTML = html;
            }}
            else if (q.includes('interim') || q.includes('ready')) {{
                const ready = portfolioData.filter(s => s.clean_rate >= 80);
                if (ready.length > 0) {{
                    let html = '<h4 style="color:#2c3e50;margin-bottom:10px;">Studies Ready for Interim Analysis (≥80% Clean Rate):</h4><ul>';
                    ready.forEach(s => {{
                        html += `<li><strong>${{s.study_id}}</strong>: ${{s.clean_rate.toFixed(1)}}% clean rate</li>`;
                    }});
                    html += '</ul>';
                    result.innerHTML = html;
                }} else {{
                    result.innerHTML = '<p style="color:#e74c3c;">No studies currently meet the 80% clean rate threshold for interim analysis.</p>';
                }}
            }}
            else if (q.includes('compare') && q.includes('clean')) {{
                const sorted = [...portfolioData].sort((a, b) => b.clean_rate - a.clean_rate);
                let html = '<h4 style="color:#2c3e50;margin-bottom:10px;">Clean Rate Comparison (All Studies):</h4><table style="width:100%;border-collapse:collapse;">';
                html += '<tr style="background:#ecf0f1;"><th style="padding:8px;text-align:left;">Study</th><th style="padding:8px;text-align:right;">Clean Rate</th><th style="padding:8px;text-align:right;">DQI</th></tr>';
                sorted.forEach(s => {{
                    const color = s.clean_rate >= 80 ? '#27ae60' : s.clean_rate >= 60 ? '#f39c12' : '#e74c3c';
                    html += `<tr><td style="padding:8px;">${{s.study_id}}</td><td style="padding:8px;text-align:right;color:${{color}};font-weight:bold;">${{s.clean_rate.toFixed(1)}}%</td><td style="padding:8px;text-align:right;">${{s.dqi.toFixed(1)}}%</td></tr>`;
                }});
                html += '</table>';
                result.innerHTML = html;
            }}
            else if (q.includes('critical') && q.includes('site')) {{
                const critical = portfolioData.filter(s => s.dqi < 50);
                if (critical.length > 0) {{
                    let html = '<h4 style="color:#2c3e50;margin-bottom:10px;">Studies with Critical Issues (DQI < 50%):</h4><ul>';
                    critical.forEach(s => {{
                        html += `<li style="color:#e74c3c;"><strong>${{s.study_id}}</strong>: ${{s.dqi.toFixed(1)}}% DQI, ${{s.open_queries}} queries</li>`;
                    }});
                    html += '</ul>';
                    result.innerHTML = html;
                }} else {{
                    result.innerHTML = '<p style="color:#27ae60;">✅ No studies with critical issues detected (all DQI ≥ 50%).</p>';
                }}
            }}
            else {{
                result.innerHTML = '<p>Query not recognized. Try one of the example queries above, or ask about:<ul><li>Studies with lowest/highest DQI</li><li>Open query counts</li><li>Clean rate comparisons</li><li>Interim analysis readiness</li><li>Critical sites/studies</li></ul></p>';
            }}
        }}
    </script>
</body>
</html>'''
    
    return html


def main():
    """Main entry point"""
    
    # Set paths
    base_path = Path(__file__).parent.parent
    data_path = base_path / "QC Anonymized Study Files"
    output_dir = base_path / "reports"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("GENERATING ALL STUDY ARTIFACTS")
    logger.info("=" * 70)
    
    # Load all studies
    logger.info("Loading all studies...")
    ingester, all_studies_data = load_all_clinical_data(str(data_path))
    
    logger.info(f"Loaded {len(all_studies_data)} studies")
    
    # Generate artifacts for each study
    all_artifacts = []
    for study_id, study_data in all_studies_data.items():
        try:
            artifacts = generate_study_artifacts(study_id, study_data, output_dir)
            if artifacts:
                all_artifacts.append(artifacts)
                logger.info(f"✓ Completed {study_id}")
        except Exception as e:
            logger.error(f"✗ Failed {study_id}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("=" * 70)
    logger.info(f"Generated artifacts for {len(all_artifacts)} studies")
    logger.info("=" * 70)
    
    # Generate master dashboard
    if all_artifacts:
        logger.info("Generating master dashboard...")
        generate_master_dashboard(all_artifacts, output_dir)
    
    logger.info("=" * 70)
    logger.info("COMPLETE!")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
