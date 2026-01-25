"""
Clinical Dataflow Visualization Dashboard
Interactive visualizations for the Neural Clinical Data Mesh
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import threading
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.data_models import (
    DigitalPatientTwin, SiteMetrics, StudyMetrics, RiskLevel
)
from core.real_time_monitor import RealTimeDataMonitor, LiveCleanlinessEngine





class DashboardVisualizer:
    """
    Creates interactive visualizations for clinical trial data
    """
    
    # Color scheme aligned with risk levels
    COLORS = {
        'Critical': '#FF0000',
        'High': '#FF6600',
        'Medium': '#FFCC00',
        'Low': '#00CC00',
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'background': '#f8f9fa'
    }
    
    def __init__(self):
        self.figures = {}
    
    def create_dqi_heatmap(
        self,
        site_metrics: Dict[str, SiteMetrics],
        title: str = "Site Data Quality Index (DQI) Heatmap"
    ) -> go.Figure:
        """Create a heatmap showing DQI scores by site"""
        
        sites = []
        dqi_scores = []
        risk_levels = []
        countries = []
        
        for site_id, metrics in site_metrics.items():
            sites.append(site_id)
            dqi_scores.append(metrics.data_quality_index)
            risk_levels.append(metrics.risk_level.value)
            countries.append(metrics.country or 'Unknown')
        
        df = pd.DataFrame({
            'Site': sites,
            'DQI': dqi_scores,
            'Risk Level': risk_levels,
            'Country': countries
        })
        
        # Sort by DQI
        df = df.sort_values('DQI', ascending=True)
        
        # Create color scale based on DQI
        colors = []
        for dqi in df['DQI']:
            if dqi < 50:
                colors.append(self.COLORS['Critical'])
            elif dqi < 75:
                colors.append(self.COLORS['High'])
            elif dqi < 90:
                colors.append(self.COLORS['Medium'])
            else:
                colors.append(self.COLORS['Low'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['DQI'],
                y=df['Site'],
                orientation='h',
                marker_color=colors,
                text=df['DQI'].round(1),
                textposition='outside',
                hovertemplate=(
                    '<b>%{y}</b><br>' +
                    'DQI: %{x:.1f}<br>' +
                    '<extra></extra>'
                )
            )
        ])
        
        # Add threshold lines
        fig.add_vline(x=90, line_dash="dash", line_color="green", annotation_text="Low Risk (90)")
        fig.add_vline(x=75, line_dash="dash", line_color="orange", annotation_text="Medium Risk (75)")
        fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="High Risk (50)")
        
        fig.update_layout(
            title=title,
            xaxis_title="Data Quality Index (DQI)",
            yaxis_title="Site ID",
            height=max(400, len(sites) * 25),
            showlegend=False
        )
        
        self.figures['dqi_heatmap'] = fig
        return fig
    
    def create_clean_patient_progress(
        self,
        twins: List[DigitalPatientTwin],
        title: str = "Clean Patient Status Distribution"
    ) -> go.Figure:
        """Create an enhanced visualization of clean patient progress"""
        
        # Calculate clean percentage buckets
        buckets = {
            '✓ 100% Clean': 0,
            '90-99%': 0,
            '75-89%': 0,
            '50-74%': 0,
            '< 50%': 0
        }
        
        for twin in twins:
            pct = twin.clean_percentage
            if pct >= 100:
                buckets['✓ 100% Clean'] += 1
            elif pct >= 90:
                buckets['90-99%'] += 1
            elif pct >= 75:
                buckets['75-89%'] += 1
            elif pct >= 50:
                buckets['50-74%'] += 1
            else:
                buckets['< 50%'] += 1
        
        colors = [
            '#10b981',  # Clean - emerald
            '#34d399',  # Almost clean - light emerald
            '#fbbf24',  # Warning - amber
            '#f97316',  # Concern - orange
            '#ef4444'   # Critical - red
        ]
        
        # Calculate percentages
        total = len(twins)
        percentages = [(v / total * 100) if total > 0 else 0 for v in buckets.values()]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(buckets.keys()),
                values=list(buckets.values()),
                marker=dict(
                    colors=colors,
                    line=dict(color='white', width=3)
                ),
                hole=0.55,
                textinfo='label+percent',
                textfont=dict(size=12, family='Inter'),
                hovertemplate='<b>%{label}</b><br>%{value} patients<br>%{percent}<extra></extra>',
                pull=[0.05 if k == '✓ 100% Clean' else 0 for k in buckets.keys()]
            )
        ])
        
        # Add annotation in center
        clean = sum(1 for t in twins if t.clean_status)
        clean_pct = (clean / total * 100) if total > 0 else 0
        center_color = '#10b981' if clean_pct >= 80 else '#f59e0b' if clean_pct >= 50 else '#ef4444'
        
        fig.add_annotation(
            text=f"<b style='font-size:32px;color:{center_color}'>{clean}</b><br><span style='font-size:14px;color:#64748b'>of {total}<br>Clean</span>",
            showarrow=False,
            font_size=14
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=18, color='#1e293b', family='Inter'),
                x=0.5
            ),
            height=450,
            paper_bgcolor='white',
            font=dict(family='Inter'),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5
            )
        )
        
        self.figures['clean_patient_progress'] = fig
        return fig
    
    def create_site_scatter_plot(
        self,
        site_metrics: Dict[str, SiteMetrics],
        title: str = "Site Risk Quadrant Analysis"
    ) -> go.Figure:
        """Create an enhanced scatter plot of sites by enrollment and issue count"""
        
        data = []
        for site_id, metrics in site_metrics.items():
            data.append({
                'Site': site_id,
                'Patients': metrics.total_patients,
                'Issues': metrics.total_open_queries + metrics.total_missing_visits,
                'DQI': metrics.data_quality_index,
                'Risk': metrics.risk_level.value,
                'Country': metrics.country or 'Unknown'
            })
        
        if not data:
            fig = go.Figure()
            fig.add_annotation(text="No site data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            self.figures['site_scatter'] = fig
            return fig
            
        df = pd.DataFrame(data)
        
        # Enhanced color map
        color_map = {
            'Critical': '#ef4444',
            'High': '#f97316',
            'Medium': '#eab308',
            'Low': '#10b981'
        }
        
        fig = go.Figure()
        
        # Add traces per risk level for better legend
        for risk in ['Critical', 'High', 'Medium', 'Low']:
            risk_df = df[df['Risk'] == risk]
            if len(risk_df) > 0:
                fig.add_trace(go.Scatter(
                    x=risk_df['Patients'],
                    y=risk_df['Issues'],
                    mode='markers',
                    name=f'{risk} Risk',
                    marker=dict(
                        size=risk_df['DQI'] / 5 + 10,
                        color=color_map[risk],
                        line=dict(color='white', width=2),
                        opacity=0.85
                    ),
                    text=risk_df['Site'],
                    customdata=risk_df[['Country', 'DQI']].values,
                    hovertemplate='<b>%{text}</b><br>Country: %{customdata[0]}<br>Patients: %{x}<br>Issues: %{y}<br>DQI: %{customdata[1]:.1f}<extra></extra>'
                ))
        
        # Add quadrant lines and labels
        max_patients = df['Patients'].max() if len(df) > 0 else 10
        max_issues = df['Issues'].max() if len(df) > 0 else 10
        mid_patients = max_patients / 2
        mid_issues = max_issues / 2
        
        # Quadrant dividers
        fig.add_hline(y=mid_issues, line_dash="dot", line_color="#cbd5e1", line_width=1)
        fig.add_vline(x=mid_patients, line_dash="dot", line_color="#cbd5e1", line_width=1)
        
        # Quadrant labels with background
        quadrants = [
            {'x': max_patients * 0.75, 'y': max_issues * 0.85, 'text': '⚠️ High Volume<br>High Risk', 'color': '#ef4444'},
            {'x': max_patients * 0.75, 'y': max_issues * 0.15, 'text': '✓ High Volume<br>Low Risk', 'color': '#10b981'},
            {'x': max_patients * 0.25, 'y': max_issues * 0.85, 'text': '⚠️ Low Volume<br>High Risk', 'color': '#f97316'},
            {'x': max_patients * 0.25, 'y': max_issues * 0.15, 'text': '✓ Low Volume<br>Low Risk', 'color': '#10b981'}
        ]
        
        for q in quadrants:
            fig.add_annotation(
                x=q['x'], y=q['y'],
                text=q['text'],
                showarrow=False,
                font=dict(size=11, color=q['color'], family='Inter'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=q['color'],
                borderwidth=1,
                borderpad=4
            )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=18, color='#1e293b', family='Inter'),
                x=0.5
            ),
            xaxis_title="Number of Patients",
            yaxis_title="Total Issues (Queries + Missing Visits)",
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            font=dict(family='Inter', color='#475569'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            xaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0'),
            yaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0')
        )
        
        self.figures['site_scatter'] = fig
        return fig
    
    def create_blocking_items_breakdown(
        self,
        twins: List[DigitalPatientTwin],
        title: str = "Blocking Items Analysis"
    ) -> go.Figure:
        """Create breakdown of what's blocking clean patient status - Enhanced"""
        
        blocking_categories = {}
        severity_counts = {'Critical': {}, 'High': {}, 'Medium': {}, 'Low': {}}
        
        for twin in twins:
            for item in twin.blocking_items:
                category = item.item_type
                severity = item.severity if hasattr(item, 'severity') else 'Medium'
                if category not in blocking_categories:
                    blocking_categories[category] = 0
                blocking_categories[category] += 1
                
                # Track severity per category
                if category not in severity_counts.get(severity, {}):
                    if severity in severity_counts:
                        severity_counts[severity][category] = 0
                if severity in severity_counts:
                    severity_counts[severity][category] = severity_counts[severity].get(category, 0) + 1
        
        if not blocking_categories:
            # Create a positive message chart
            fig = go.Figure()
            fig.add_annotation(
                text="<b>🎉 Excellent!</b><br>No Blocking Items Found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=24, color='#10b981')
            )
            fig.update_layout(height=400, paper_bgcolor='white', plot_bgcolor='white')
            self.figures['blocking_items'] = fig
            return fig
        
        # Sort by count
        sorted_cats = dict(sorted(blocking_categories.items(), key=lambda x: x[1], reverse=True)[:10])
        categories = list(sorted_cats.keys())
        counts = list(sorted_cats.values())
        
        # Create color gradient based on count
        max_count = max(counts) if counts else 1
        colors = [f'rgba(239, 68, 68, {0.3 + 0.7 * c / max_count})' for c in counts]
        
        fig = go.Figure()
        
        # Horizontal bar chart
        fig.add_trace(go.Bar(
            y=categories,
            x=counts,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#dc2626', width=1)
            ),
            text=[f'{c} patients' for c in counts],
            textposition='auto',
            textfont=dict(color='white', size=12, family='Inter'),
            hovertemplate='<b>%{y}</b><br>%{x} patients affected<extra></extra>'
        ))
        
        # Add percentage annotations
        total = len(twins)
        for i, (cat, count) in enumerate(zip(categories, counts)):
            pct = (count / total) * 100
            fig.add_annotation(
                x=count + max_count * 0.02,
                y=cat,
                text=f'{pct:.1f}%',
                showarrow=False,
                font=dict(size=11, color='#64748b'),
                xanchor='left'
            )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=18, color='#1e293b', family='Inter'),
                x=0.5
            ),
            xaxis_title="Number of Patients Affected",
            yaxis_title="",
            height=max(350, len(categories) * 45),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Inter', color='#475569'),
            margin=dict(l=150, r=80),
            yaxis=dict(autorange='reversed')
        )
        
        self.figures['blocking_items'] = fig
        return fig
    
    def create_agent_recommendations_summary(
        self,
        recommendations: List[Dict],
        title: str = "AI Agent Recommendations Summary"
    ) -> go.Figure:
        """Create visualization of agent recommendations"""
        
        # Handle empty or None recommendations
        if not recommendations or len(recommendations) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="<b>No Recommendations</b><br><br>All data quality checks passed!<br>No AI agent actions required.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color='#10b981', family='Inter'),
                align='center'
            )
            fig.update_layout(
                title=dict(
                    text=f"<b>{title}</b>",
                    font=dict(size=18, color='#1e293b', family='Inter'),
                    x=0.5
                ),
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            self.figures['agent_recommendations'] = fig
            return fig
        
        # Count by agent
        by_agent = {}
        by_priority = {}
        by_action = {}
        
        for rec in recommendations:
            # Ensure we handle both objects and dicts
            if hasattr(rec, 'to_dict'):
                rec = rec.to_dict()
            
            agent = rec.get('agent', 'Unknown')
            priority = rec.get('priority', 'Medium')
            action = rec.get('action_type', 'Unknown')
            
            by_agent[agent] = by_agent.get(agent, 0) + 1
            by_priority[priority] = by_priority.get(priority, 0) + 1
            by_action[action] = by_action.get(action, 0) + 1
        
        # Create subplots with better spacing
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['<b>By Agent</b>', '<b>By Priority</b>', '<b>By Action Type</b>'],
            specs=[[{'type': 'pie'}, {'type': 'pie'}, {'type': 'bar'}]],
            horizontal_spacing=0.08
        )
        
        # Agent colors - modern palette
        agent_colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899']
        
        # By Agent - Enhanced donut
        fig.add_trace(
            go.Pie(
                labels=list(by_agent.keys()),
                values=list(by_agent.values()),
                name='Agent',
                hole=0.5,
                marker=dict(
                    colors=agent_colors[:len(by_agent)],
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textfont=dict(size=11),
                hovertemplate='<b>%{label}</b><br>%{value} recommendations<br>%{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # By Priority - Enhanced with pull effect on critical
        priority_colors = {
            'CRITICAL': '#ef4444',
            'HIGH': '#f97316',
            'MEDIUM': '#eab308',
            'LOW': '#10b981'
        }
        pull_values = [0.1 if p == 'CRITICAL' else 0.05 if p == 'HIGH' else 0 for p in by_priority.keys()]
        
        fig.add_trace(
            go.Pie(
                labels=list(by_priority.keys()),
                values=list(by_priority.values()),
                name='Priority',
                marker=dict(
                    colors=[priority_colors.get(p, '#94a3b8') for p in by_priority.keys()],
                    line=dict(color='white', width=2)
                ),
                hole=0.5,
                pull=pull_values,
                textinfo='label+value',
                textfont=dict(size=11, color='white'),
                hovertemplate='<b>%{label}</b><br>%{value} items<extra></extra>'
            ),
            row=1, col=2
        )
        
        # By Action Type - Horizontal bar with gradient
        action_values = list(by_action.values())
        max_val = max(action_values) if action_values else 1
        colors = [f'rgba(59, 130, 246, {0.4 + 0.6 * v / max_val})' for v in action_values]
        
        fig.add_trace(
            go.Bar(
                y=list(by_action.keys()),
                x=action_values,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='#2563eb', width=1)
                ),
                text=action_values,
                textposition='auto',
                textfont=dict(color='white', size=12),
                hovertemplate='<b>%{y}</b>: %{x} recommendations<extra></extra>'
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=18, color='#1e293b', family='Inter'),
                x=0.5
            ),
            height=400,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Inter', color='#475569')
        )
        
        # Style subplot titles
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=13, color='#374151', family='Inter')
        
        self.figures['agent_recommendations'] = fig
        return fig
    
    def create_study_overview_dashboard(
        self,
        study_metrics: StudyMetrics,
        title: str = "Study Overview"
    ) -> go.Figure:
        """Create a comprehensive study overview dashboard with enhanced visuals"""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                '<b>Global DQI Score</b>', '<b>Clean Patient Rate</b>', '<b>Sites by Risk Level</b>',
                '<b>Clean Patient Progress</b>', '<b>Site Performance</b>', '<b>Patient Distribution</b>'
            ],
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'pie'}],
                [{'type': 'indicator'}, {'type': 'bar'}, {'type': 'pie'}]
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.15
        )
        
        # Determine DQI color
        dqi = study_metrics.global_dqi
        dqi_color = '#10b981' if dqi >= 85 else '#f59e0b' if dqi >= 70 else '#ef4444'
        
        # Global DQI Gauge - Enhanced
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=study_metrics.global_dqi,
                number={'font': {'size': 48, 'color': dqi_color}, 'suffix': ''},
                title={'text': "", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#64748b'},
                    'bar': {'color': dqi_color, 'thickness': 0.75},
                    'bgcolor': '#f1f5f9',
                    'borderwidth': 2,
                    'bordercolor': '#e2e8f0',
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [50, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [70, 85], 'color': 'rgba(234, 179, 8, 0.2)'},
                        {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': '#10b981', 'width': 4},
                        'thickness': 0.8,
                        'value': 85
                    }
                }
            ),
            row=1, col=1
        )
        
        # Clean Patient Rate Gauge - Enhanced (realistic clinical trial thresholds)
        clean_rate = study_metrics.global_clean_rate
        # Use realistic thresholds for clinical trials (clean rates are typically lower during active collection)
        clean_color = '#10b981' if clean_rate >= 30 else '#f59e0b' if clean_rate >= 10 else '#ef4444'
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=clean_rate,
                number={'suffix': '%', 'font': {'size': 42, 'color': clean_color}},
                title={'text': "", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2},
                    'bar': {'color': clean_color, 'thickness': 0.75},
                    'bgcolor': '#f1f5f9',
                    'borderwidth': 2,
                    'bordercolor': '#e2e8f0',
                    'steps': [
                        {'range': [0, 10], 'color': 'rgba(239, 68, 68, 0.15)'},
                        {'range': [10, 30], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [30, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': '#10b981', 'width': 4},
                        'thickness': 0.8,
                        'value': 30  # Realistic target for active trials
                    }
                }
            ),
            row=1, col=2
        )
        
        # Sites by Risk Pie - Enhanced with gradients
        risk_data = study_metrics.sites_by_risk
        risk_colors = ['#ef4444', '#f97316', '#eab308', '#10b981']  # Critical, High, Medium, Low
        
        fig.add_trace(
            go.Pie(
                labels=list(risk_data.keys()),
                values=list(risk_data.values()),
                marker=dict(
                    colors=risk_colors,
                    line=dict(color='white', width=3)
                ),
                hole=0.5,
                textinfo='label+value',
                textfont=dict(size=12, color='white'),
                hovertemplate='<b>%{label}</b><br>%{value} sites<br>%{percent}<extra></extra>',
                pull=[0.05 if k == 'Critical' else 0 for k in risk_data.keys()]
            ),
            row=1, col=3
        )
        
        # Clean Patient Progress - Show actual clean count with percentage
        clean_patients = study_metrics.clean_patients
        total_patients = study_metrics.total_patients
        clean_pct = round((clean_patients / total_patients * 100), 1) if total_patients > 0 else 0
        needs_attention = total_patients - clean_patients
        
        # Color based on percentage
        progress_color = '#10b981' if clean_pct >= 30 else '#f59e0b' if clean_pct >= 10 else '#ef4444'
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=clean_patients,
                number={'font': {'size': 56, 'color': progress_color}, 'suffix': ''},
                title={'text': f"<span style='font-size:14px;color:#64748b'>of {total_patients:,} patients<br><b style='color:{progress_color}'>{clean_pct}% Clean</b></span>", 'font': {'size': 14}}
            ),
            row=2, col=1
        )
        
        # Site Performance Bar - Horizontal with better colors
        at_risk = study_metrics.sites_at_risk
        stable = study_metrics.total_sites - study_metrics.sites_at_risk
        
        fig.add_trace(
            go.Bar(
                y=['At Risk', 'Stable'],
                x=[at_risk, stable],
                orientation='h',
                marker=dict(
                    color=['#ef4444', '#10b981'],
                    line=dict(color='white', width=2)
                ),
                text=[f'{at_risk} sites', f'{stable} sites'],
                textposition='inside',
                textfont=dict(color='white', size=14, family='Inter'),
                hovertemplate='<b>%{y}</b>: %{x} sites<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Patient Distribution Pie - Enhanced donut
        clean_pct = study_metrics.clean_patients
        not_clean = study_metrics.total_patients - study_metrics.clean_patients
        
        fig.add_trace(
            go.Pie(
                labels=['Clean', 'Needs Attention'],
                values=[clean_pct, not_clean],
                marker=dict(
                    colors=['#10b981', '#f59e0b'],
                    line=dict(color='white', width=3)
                ),
                hole=0.6,
                textinfo='percent+label',
                textfont=dict(size=12),
                hovertemplate='<b>%{label}</b><br>%{value} patients<br>%{percent}<extra></extra>'
            ),
            row=2, col=3
        )
        
        # Update layout with modern styling
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>: {study_metrics.study_id}",
                font=dict(size=20, color='#1e293b', family='Inter'),
                x=0.5,
                y=0.98
            ),
            height=650,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', color='#475569'),
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        # Update subplot title styling
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color='#374151', family='Inter')
        
        self.figures['study_overview'] = fig
        return fig
    
    def create_patient_timeline(
        self,
        twin: DigitalPatientTwin,
        title: str = None
    ) -> go.Figure:
        """Create a timeline visualization for a single patient"""
        
        title = title or f"Patient Timeline: {twin.subject_id}"
        
        # Create timeline events
        events = []
        
        # Add blocking items as events
        for i, item in enumerate(twin.blocking_items):
            events.append({
                'Event': item.item_type,
                'Description': item.description,
                'Severity': item.severity,
                'Position': i
            })
        
        if not events:
            events.append({
                'Event': 'Clean',
                'Description': 'No blocking items',
                'Severity': 'Low',
                'Position': 0
            })
        
        df = pd.DataFrame(events)
        
        severity_colors = {
            'Critical': self.COLORS['Critical'],
            'High': self.COLORS['High'],
            'Medium': self.COLORS['Medium'],
            'Low': self.COLORS['Low']
        }
        
        fig = go.Figure()
        
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Position']],
                y=[0],
                mode='markers+text',
                marker=dict(
                    size=30,
                    color=severity_colors.get(row['Severity'], '#999'),
                    symbol='circle'
                ),
                text=row['Event'],
                textposition='top center',
                hovertemplate=f"<b>{row['Event']}</b><br>{row['Description']}<extra></extra>"
            ))
        
        # Add patient info annotation
        fig.add_annotation(
            x=0, y=0.5,
            text=(
                f"<b>Subject:</b> {twin.subject_id}<br>"
                f"<b>Site:</b> {twin.site_id}<br>"
                f"<b>Status:</b> {twin.status.value}<br>"
                f"<b>DQI:</b> {twin.data_quality_index:.1f}<br>"
                f"<b>Clean %:</b> {twin.clean_percentage:.1f}%"
            ),
            showarrow=False,
            xanchor='left',
            align='left',
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1
        )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=300,
            yaxis=dict(visible=False, range=[-0.5, 1]),
            xaxis=dict(visible=False)
        )
        
        self.figures['patient_timeline'] = fig
        return fig
    
    def export_all_figures_html(self, output_path: str, study_metrics: 'StudyMetrics' = None, 
                                 recommendations: List[Dict] = None, twins: List['DigitalPatientTwin'] = None,
                                 site_metrics: Dict[str, 'SiteMetrics'] = None):
        """Export all figures to a single professional HTML dashboard with enhanced UI/UX"""
        
        # Extract metrics for executive summary
        total_patients = study_metrics.total_patients if study_metrics else 0
        clean_patients = study_metrics.clean_patients if study_metrics else 0
        global_dqi = study_metrics.global_dqi if study_metrics else 0
        clean_rate = study_metrics.global_clean_rate if study_metrics else 0
        total_sites = study_metrics.total_sites if study_metrics else 0
        sites_at_risk = study_metrics.sites_at_risk if study_metrics else 0
        study_id = study_metrics.study_id if study_metrics else "Unknown"
        
        # Calculate recommendation stats
        total_recs = len(recommendations) if recommendations else 0
        critical_recs = sum(1 for r in (recommendations or []) if r.get('priority', '').upper() == 'CRITICAL')
        high_recs = sum(1 for r in (recommendations or []) if r.get('priority', '').upper() == 'HIGH')
        medium_recs = sum(1 for r in (recommendations or []) if r.get('priority', '').upper() == 'MEDIUM')
        low_recs = total_recs - critical_recs - high_recs - medium_recs
        
        # Calculate blocking items summary
        blocking_summary = {}
        if twins:
            for twin in twins:
                for item in twin.blocking_items:
                    cat = item.item_type
                    blocking_summary[cat] = blocking_summary.get(cat, 0) + 1
        top_blockers = sorted(blocking_summary.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate site distribution by country
        country_dist = {}
        if site_metrics:
            for site_id, metrics in site_metrics.items():
                country = metrics.country or 'Unknown'
                if country not in country_dist:
                    country_dist[country] = {'count': 0, 'at_risk': 0}
                country_dist[country]['count'] += 1
                if metrics.risk_level.value in ['Critical', 'High']:
                    country_dist[country]['at_risk'] += 1
        
        # Calculate patient stats by status
        patient_by_status = {'Complete': 0, 'Ongoing': 0, 'Discontinued': 0}
        if twins:
            for twin in twins:
                status = twin.status.value if hasattr(twin, 'status') else 'Ongoing'
                if status in patient_by_status:
                    patient_by_status[status] += 1
                else:
                    patient_by_status['Ongoing'] += 1
        
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Clinical Data Mesh - {study_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            /* Light Theme */
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f1f5f9;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --primary: #3b82f6;
            --primary-light: #60a5fa;
            --primary-dark: #2563eb;
            --primary-glow: rgba(59, 130, 246, 0.15);
            --success: #10b981;
            --success-light: #34d399;
            --success-glow: rgba(16, 185, 129, 0.15);
            --warning: #f59e0b;
            --warning-light: #fbbf24;
            --warning-glow: rgba(245, 158, 11, 0.15);
            --danger: #ef4444;
            --danger-light: #f87171;
            --danger-glow: rgba(239, 68, 68, 0.15);
            --info: #06b6d4;
            --purple: #8b5cf6;
            --purple-glow: rgba(139, 92, 246, 0.15);
            --gradient-primary: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --gradient-success: linear-gradient(135deg, #10b981 0%, #34d399 100%);
            --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        /* Dark Theme */
        [data-theme="dark"] {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --border-color: #334155;
            --primary-glow: rgba(59, 130, 246, 0.25);
            --success-glow: rgba(16, 185, 129, 0.25);
            --warning-glow: rgba(245, 158, 11, 0.25);
            --danger-glow: rgba(239, 68, 68, 0.25);
            --purple-glow: rgba(139, 92, 246, 0.25);
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            min-height: 100vh;
            color: var(--text-primary);
            transition: background var(--transition-normal), color var(--transition-normal);
        }}
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: var(--bg-tertiary);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: var(--text-muted);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-secondary);
        }}
        
        /* Header */
        .header {{
            background: var(--gradient-primary);
            color: white;
            padding: 0;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: var(--shadow-xl);
        }}
        
        .header-content {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header-title {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        
        .header-logo {{
            width: 48px;
            height: 48px;
            background: rgba(255,255,255,0.2);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            backdrop-filter: blur(10px);
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        .header-title h1 {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        
        .header-title .study-badge {{
            background: rgba(255,255,255,0.2);
            padding: 8px 20px;
            border-radius: 25px;
            font-size: 13px;
            font-weight: 600;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }}
        
        .header-meta {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        
        .header-btn {{
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 8px;
            border: none;
        }}
        
        .header-btn.ghost {{
            background: rgba(255,255,255,0.15);
            color: white;
            backdrop-filter: blur(10px);
        }}
        
        .header-btn.ghost:hover {{
            background: rgba(255,255,255,0.25);
            transform: translateY(-1px);
        }}
        
        .theme-toggle {{
            width: 44px;
            height: 44px;
            border-radius: 12px;
            border: none;
            background: rgba(255,255,255,0.15);
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all var(--transition-fast);
            backdrop-filter: blur(10px);
        }}
        
        .theme-toggle:hover {{
            background: rgba(255,255,255,0.25);
            transform: rotate(15deg);
        }}
        
        /* Navigation */
        .nav-bar {{
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 0 32px;
            position: sticky;
            top: 80px;
            z-index: 999;
            box-shadow: var(--shadow-sm);
            transition: background var(--transition-normal), border-color var(--transition-normal);
        }}
        
        .nav-content {{
            max-width: 1800px;
            margin: 0 auto;
            display: flex;
            gap: 4px;
            overflow-x: auto;
            padding: 8px 0;
        }}
        
        .nav-btn {{
            padding: 12px 20px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 10px;
            transition: all var(--transition-fast);
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 8px;
            position: relative;
        }}
        
        .nav-btn:hover {{
            background: var(--bg-tertiary);
            color: var(--primary);
        }}
        
        .nav-btn.active {{
            background: var(--primary-glow);
            color: var(--primary);
        }}
        
        .nav-btn .badge {{
            background: var(--danger);
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 700;
            animation: badgePulse 2s infinite;
        }}
        
        @keyframes badgePulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}
        
        /* Search Bar */
        .search-bar {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-left: auto;
            padding-left: 24px;
        }}
        
        .search-input {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 10px 16px 10px 40px;
            font-size: 14px;
            color: var(--text-primary);
            width: 240px;
            transition: all var(--transition-fast);
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
            width: 300px;
        }}
        
        .search-wrapper {{
            position: relative;
        }}
        
        .search-wrapper i {{
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
        }}
        
        /* Main Content */
        .main-content {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 32px;
        }}
        
        /* Executive Summary */
        .executive-summary {{
            background: var(--bg-secondary);
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 28px;
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--border-color);
            transition: background var(--transition-normal), border-color var(--transition-normal);
        }}
        
        .summary-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .summary-header h2 {{
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .summary-header h2 i {{
            color: var(--primary);
            font-size: 22px;
        }}
        
        .status-badge {{
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 13px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .status-badge.good {{
            background: var(--success-glow);
            color: var(--success);
            border: 1px solid var(--success);
        }}
        
        .status-badge.warning {{
            background: var(--warning-glow);
            color: var(--warning);
            border: 1px solid var(--warning);
        }}
        
        .status-badge.danger {{
            background: var(--danger-glow);
            color: var(--danger);
            border: 1px solid var(--danger);
        }}
        
        /* KPI Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
        }}
        
        .kpi-card {{
            background: var(--bg-tertiary);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: all var(--transition-normal);
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }}
        
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--gradient-primary);
            opacity: 0;
            transition: opacity var(--transition-fast);
        }}
        
        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }}
        
        .kpi-card:hover::before {{
            opacity: 1;
        }}
        
        .kpi-icon {{
            width: 56px;
            height: 56px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px;
            font-size: 24px;
            transition: transform var(--transition-fast);
        }}
        
        .kpi-card:hover .kpi-icon {{
            transform: scale(1.1);
        }}
        
        .kpi-icon.blue {{ background: var(--primary-glow); color: var(--primary); }}
        .kpi-icon.green {{ background: var(--success-glow); color: var(--success); }}
        .kpi-icon.yellow {{ background: var(--warning-glow); color: var(--warning); }}
        .kpi-icon.red {{ background: var(--danger-glow); color: var(--danger); }}
        .kpi-icon.purple {{ background: var(--purple-glow); color: var(--purple); }}
        
        .kpi-value {{
            font-size: 32px;
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: 6px;
            letter-spacing: -1px;
        }}
        
        .kpi-label {{
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 500;
        }}
        
        .kpi-trend {{
            font-size: 12px;
            margin-top: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 20px;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }}
        
        .kpi-trend.up {{ 
            color: var(--success); 
            background: var(--success-glow);
        }}
        .kpi-trend.down {{ 
            color: var(--danger); 
            background: var(--danger-glow);
        }}
        
        /* Progress Ring */
        .progress-ring {{
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }}
        
        .progress-ring svg {{
            transform: rotate(-90deg);
        }}
        
        .progress-ring-value {{
            position: absolute;
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
        }}
        
        /* Alert Banner */
        .alert-banner {{
            background: var(--warning-glow);
            border: 1px solid var(--warning);
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 28px;
            display: flex;
            align-items: center;
            gap: 16px;
            animation: slideDown 0.5s ease;
        }}
        
        @keyframes slideDown {{
            from {{
                opacity: 0;
                transform: translateY(-10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .alert-banner.danger {{
            background: var(--danger-glow);
            border-color: var(--danger);
        }}
        
        .alert-banner i {{
            font-size: 28px;
            color: var(--warning);
        }}
        
        .alert-banner.danger i {{
            color: var(--danger);
        }}
        
        .alert-content h4 {{
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 6px;
            color: var(--text-primary);
        }}
        
        .alert-content p {{
            font-size: 14px;
            color: var(--text-secondary);
        }}
        
        /* Section */
        .section {{
            display: none;
            animation: fadeInUp 0.4s ease;
        }}
        
        .section.active {{
            display: block;
        }}
        
        @keyframes fadeInUp {{
            from {{ 
                opacity: 0; 
                transform: translateY(20px); 
            }}
            to {{ 
                opacity: 1; 
                transform: translateY(0); 
            }}
        }}
        
        .section-title {{
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-title i {{
            color: var(--primary);
            font-size: 22px;
        }}
        
        /* Chart Cards */
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
            gap: 28px;
            margin-bottom: 28px;
        }}
        
        .chart-card {{
            background: var(--bg-secondary);
            border-radius: 20px;
            padding: 28px;
            box-shadow: var(--shadow-md);
            transition: all var(--transition-normal);
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }}
        
        .chart-card::after {{
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 20px;
            background: linear-gradient(135deg, transparent 0%, rgba(255,255,255,0.05) 100%);
            pointer-events: none;
        }}
        
        .chart-card:hover {{
            transform: translateY(-4px);
            box-shadow: var(--shadow-xl);
        }}
        
        .chart-card.full-width {{
            grid-column: 1 / -1;
        }}
        
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .chart-header h3 {{
            font-size: 17px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .chart-header h3 i {{
            font-size: 20px;
        }}
        
        .chart-actions {{
            display: flex;
            gap: 10px;
        }}
        
        .chart-btn {{
            padding: 8px 16px;
            border: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            border-radius: 10px;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .chart-btn:hover {{
            background: var(--primary-glow);
            color: var(--primary);
            border-color: var(--primary);
        }}
        
        /* Recommendations Table */
        .rec-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 14px;
        }}
        
        .rec-table th {{
            background: var(--bg-tertiary);
            padding: 16px 20px;
            text-align: left;
            font-weight: 600;
            color: var(--text-primary);
            border-bottom: 2px solid var(--border-color);
        }}
        
        .rec-table th:first-child {{
            border-radius: 12px 0 0 0;
        }}
        
        .rec-table th:last-child {{
            border-radius: 0 12px 0 0;
        }}
        
        .rec-table td {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-secondary);
            transition: background var(--transition-fast);
        }}
        
        .rec-table tr:hover td {{
            background: var(--bg-tertiary);
        }}
        
        .priority-badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .priority-badge.critical {{ 
            background: var(--danger-glow); 
            color: var(--danger); 
            border: 1px solid var(--danger);
        }}
        .priority-badge.high {{ 
            background: rgba(249, 115, 22, 0.15); 
            color: #ea580c; 
            border: 1px solid #f97316;
        }}
        .priority-badge.medium {{ 
            background: var(--warning-glow); 
            color: var(--warning); 
            border: 1px solid var(--warning);
        }}
        .priority-badge.low {{ 
            background: var(--success-glow); 
            color: var(--success); 
            border: 1px solid var(--success);
        }}
        
        /* Top Blockers */
        .blocker-list {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        
        .blocker-item {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px 20px;
            background: var(--bg-tertiary);
            border-radius: 14px;
            transition: all var(--transition-fast);
            border: 1px solid transparent;
        }}
        
        .blocker-item:hover {{
            border-color: var(--danger);
            transform: translateX(4px);
        }}
        
        .blocker-rank {{
            width: 36px;
            height: 36px;
            background: var(--gradient-danger);
            color: white;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 700;
            flex-shrink: 0;
        }}
        
        .blocker-info {{
            flex: 1;
            min-width: 0;
        }}
        
        .blocker-name {{
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }}
        
        .blocker-count {{
            font-size: 13px;
            color: var(--text-muted);
        }}
        
        .blocker-bar {{
            width: 120px;
            height: 10px;
            background: var(--bg-primary);
            border-radius: 6px;
            overflow: hidden;
            flex-shrink: 0;
        }}
        
        .blocker-bar-fill {{
            height: 100%;
            background: var(--gradient-danger);
            border-radius: 6px;
            transition: width 1s ease;
        }}
        
        /* Tooltip */
        .tooltip {{
            position: relative;
            cursor: help;
        }}
        
        .tooltip::after {{
            content: attr(data-tooltip);
            position: absolute;
            bottom: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%);
            background: var(--text-primary);
            color: var(--bg-primary);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: all var(--transition-fast);
            z-index: 1000;
        }}
        
        .tooltip:hover::after {{
            opacity: 1;
            visibility: visible;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 32px;
            color: var(--text-muted);
            font-size: 13px;
            border-top: 1px solid var(--border-color);
            margin-top: 48px;
            background: var(--bg-secondary);
        }}
        
        .footer-logo {{
            font-size: 24px;
            margin-bottom: 12px;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        /* Loading Spinner */
        .loading {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 60px;
        }}
        
        .spinner {{
            width: 48px;
            height: 48px;
            border: 4px solid var(--border-color);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        /* Responsive */
        @media (max-width: 1200px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media (max-width: 768px) {{
            .header-content {{
                flex-direction: column;
                gap: 16px;
                padding: 16px;
            }}
            
            .header-meta {{
                flex-wrap: wrap;
                justify-content: center;
            }}
            
            .kpi-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .main-content {{
                padding: 16px;
            }}
            
            .nav-content {{
                padding: 0;
            }}
            
            .search-bar {{
                display: none;
            }}
        }}
        
        /* Print Styles */
        @media print {{
            body {{
                background: white;
                color: black;
            }}
            .header {{ 
                position: relative;
                background: #2563eb;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .nav-bar {{ display: none; }}
            .theme-toggle {{ display: none; }}
            .section {{ 
                display: block !important; 
                page-break-inside: avoid; 
            }}
            .chart-card {{ 
                break-inside: avoid;
                box-shadow: none;
                border: 1px solid #ddd;
            }}
        }}
        
        /* Glassmorphism Effects */
        .glass {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        /* Micro-interactions */
        .interactive {{
            transition: all var(--transition-fast);
        }}
        
        .interactive:active {{
            transform: scale(0.98);
        }}
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <div class="header-title">
                <div class="header-logo">
                    <i class="fas fa-dna"></i>
                </div>
                <div>
                    <h1>Neural Clinical Data Mesh</h1>
                    <span style="font-size: 12px; opacity: 0.8;">AI-Powered Clinical Intelligence Platform</span>
                </div>
                <span class="study-badge">{study_id}</span>
            </div>
            <div class="header-meta">
                <span><i class="fas fa-calendar-alt"></i> {datetime.now().strftime('%B %d, %Y')}</span>
                <span><i class="fas fa-clock"></i> {datetime.now().strftime('%H:%M')}</span>
                <button class="theme-toggle glass interactive" onclick="toggleTheme()" title="Toggle Dark Mode">
                    <i class="fas fa-moon"></i>
                </button>
                <button class="glass interactive" onclick="window.print()" title="Print Dashboard" style="background: rgba(255,255,255,0.2); border: none; padding: 10px 18px; border-radius: 10px; color: white; cursor: pointer; font-weight: 500;">
                    <i class="fas fa-print"></i> Export
                </button>
            </div>
        </div>
    </header>
    
    <!-- Navigation -->
    <nav class="nav-bar">
        <div class="nav-content">
            <div class="nav-buttons">
                <button class="nav-btn active" onclick="showSection('overview')">
                    <i class="fas fa-chart-pie"></i> Overview
                </button>
                <button class="nav-btn" onclick="showSection('sites')">
                    <i class="fas fa-building"></i> Sites
                    <span class="badge">{total_sites}</span>
                </button>
                <button class="nav-btn" onclick="showSection('patients')">
                    <i class="fas fa-users"></i> Patients
                    <span class="badge">{total_patients}</span>
                </button>
                <button class="nav-btn" onclick="showSection('recommendations')">
                    <i class="fas fa-lightbulb"></i> AI Insights
                    {f'<span class="badge danger">{critical_recs}</span>' if critical_recs > 0 else f'<span class="badge">{total_recs}</span>'}
                </button>
                <button class="nav-btn" onclick="showSection('quality')">
                    <i class="fas fa-shield-alt"></i> Quality
                    <span class="badge {'success' if global_dqi >= 85 else 'warning' if global_dqi >= 70 else 'danger'}">{global_dqi:.0f}%</span>
                </button>
            </div>
            <div class="search-bar">
                <i class="fas fa-search"></i>
                <input type="text" id="globalSearch" placeholder="Search patients, sites, issues..." onkeyup="filterDashboard(this.value)">
            </div>
        </div>
    </nav>
    
    <!-- Main Content -->
    <main class="main-content">
        <!-- Executive Summary (Always Visible) -->
        <div class="executive-summary animate__animated animate__fadeIn">
            <div class="summary-header">
                <h2><i class="fas fa-chart-line"></i> Executive Summary</h2>
                <div style="display: flex; gap: 12px; align-items: center;">
                    <span class="status-badge {'good' if global_dqi >= 85 else 'warning' if global_dqi >= 70 else 'danger'}">
                        <i class="fas fa-{'check-circle' if global_dqi >= 85 else 'exclamation-triangle' if global_dqi >= 70 else 'times-circle'}"></i>
                        {'Excellent' if global_dqi >= 90 else 'On Track' if global_dqi >= 85 else 'Needs Attention' if global_dqi >= 70 else 'Critical'}
                    </span>
                    <span style="font-size: 12px; color: var(--text-muted);">Last Updated: Just Now</span>
                </div>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card interactive" onclick="showSection('patients')">
                    <div class="kpi-icon blue"><i class="fas fa-users"></i></div>
                    <div class="kpi-value">{total_patients:,}</div>
                    <div class="kpi-label">Total Patients</div>
                    <div class="kpi-progress">
                        <div class="kpi-progress-fill" style="width: 100%; background: var(--gradient-primary);"></div>
                    </div>
                </div>
                <div class="kpi-card interactive" onclick="showSection('patients')">
                    <div class="kpi-icon green"><i class="fas fa-user-check"></i></div>
                    <div class="kpi-value">{clean_patients:,}</div>
                    <div class="kpi-label">Clean Patients</div>
                    <div class="kpi-progress">
                        <div class="kpi-progress-fill" style="width: {(clean_patients/total_patients*100) if total_patients > 0 else 0:.0f}%; background: var(--gradient-success);"></div>
                    </div>
                    <div class="kpi-trend {'up' if clean_rate >= 80 else 'down'}">
                        <i class="fas fa-{'arrow-up' if clean_rate >= 80 else 'arrow-down'}"></i>
                        {clean_rate:.1f}% clean rate
                    </div>
                </div>
                <div class="kpi-card interactive" onclick="showSection('quality')">
                    <div class="kpi-icon {'green' if global_dqi >= 85 else 'yellow' if global_dqi >= 70 else 'red'}">
                        <i class="fas fa-tachometer-alt"></i>
                    </div>
                    <div class="kpi-value">{global_dqi:.1f}</div>
                    <div class="kpi-label">Global DQI Score</div>
                    <div class="kpi-progress">
                        <div class="kpi-progress-fill" style="width: {global_dqi:.0f}%; background: var({'--gradient-success' if global_dqi >= 85 else '--gradient-warning' if global_dqi >= 70 else '--gradient-danger'});"></div>
                    </div>
                </div>
                <div class="kpi-card interactive" onclick="showSection('sites')">
                    <div class="kpi-icon purple"><i class="fas fa-hospital"></i></div>
                    <div class="kpi-value">{total_sites}</div>
                    <div class="kpi-label">Total Sites</div>
                    <div class="kpi-trend {'up' if sites_at_risk == 0 else 'down'}">
                        <i class="fas fa-exclamation-triangle"></i>
                        {sites_at_risk} at risk
                    </div>
                </div>
                <div class="kpi-card interactive" onclick="showSection('recommendations')">
                    <div class="kpi-icon {'green' if total_recs == 0 else 'red' if critical_recs > 0 else 'yellow'}">
                        <i class="fas fa-brain"></i>
                    </div>
                    <div class="kpi-value">{total_recs}</div>
                    <div class="kpi-label">AI Recommendations</div>
                    {f'<div class="kpi-trend down"><i class="fas fa-exclamation-circle"></i> {critical_recs} critical</div>' if critical_recs > 0 else ''}
                </div>
            </div>
        </div>
        
        {f'''<div class="alert-banner {'danger' if critical_recs > 0 else ''} animate__animated animate__fadeInDown">
            <i class="fas fa-{'exclamation-triangle' if critical_recs > 0 else 'info-circle'}" style="font-size: 24px;"></i>
            <div class="alert-content">
                <h4>{'Critical Issues Detected' if critical_recs > 0 else 'Action Items Available'}</h4>
                <p>{critical_recs} critical and {high_recs} high priority insights from AI agents need attention.</p>
            </div>
            <button class="interactive" onclick="showSection('recommendations')" style="background: rgba(255,255,255,0.2); border: none; padding: 10px 20px; border-radius: 8px; color: inherit; cursor: pointer; font-weight: 600;">
                Review Now <i class="fas fa-arrow-right"></i>
            </button>
        </div>''' if total_recs > 0 else ''}
        
        <!-- Overview Section -->
        <section id="overview" class="section active">
            <div class="chart-grid">
                <div class="chart-card">
                    <div class="chart-header">
                        <h3><i class="fas fa-chart-bar" style="color: var(--primary); margin-right: 8px;"></i> Study Overview</h3>
                        <div class="chart-actions">
                            <button class="chart-action-btn tooltip" data-tooltip="Refresh Data"><i class="fas fa-sync-alt"></i></button>
                            <button class="chart-action-btn tooltip" data-tooltip="Expand"><i class="fas fa-expand"></i></button>
                        </div>
                    </div>
                    <div id="study_overview"></div>
                </div>
                <div class="chart-card">
                    <div class="chart-header">
                        <h3><i class="fas fa-exclamation-circle" style="color: var(--danger); margin-right: 8px;"></i> Top Blockers</h3>
                        <span class="badge danger">{len(top_blockers) if top_blockers else 0}</span>
                    </div>
                    {self._generate_blockers_html(top_blockers, twins)}
                </div>
            </div>
        </section>
        
        <!-- Sites Section -->
        <section id="sites" class="section">
            <h2 class="section-title"><i class="fas fa-building"></i> Site Analytics</h2>
            <div class="chart-grid">
                <div class="chart-card full-width">
                    <div class="chart-header">
                        <h3><i class="fas fa-th-large" style="color: var(--primary); margin-right: 8px;"></i> Site Quality Heatmap</h3>
                        <div class="chart-actions">
                            <button class="chart-action-btn tooltip" data-tooltip="Download CSV"><i class="fas fa-download"></i></button>
                            <button class="chart-action-btn tooltip" data-tooltip="Expand"><i class="fas fa-expand"></i></button>
                        </div>
                    </div>
                    <div id="dqi_heatmap"></div>
                </div>
                <div class="chart-card">
                    <div class="chart-header">
                        <h3><i class="fas fa-chart-scatter" style="color: var(--info); margin-right: 8px;"></i> Risk Quadrant</h3>
                    </div>
                    <div id="site_scatter"></div>
                </div>
            </div>
        </section>
        
        <!-- Patients Section -->
        <section id="patients" class="section">
            <h2 class="section-title"><i class="fas fa-users"></i> Patient Analytics</h2>
            <div class="chart-grid">
                <div class="chart-card">
                    <div class="chart-header">
                        <h3><i class="fas fa-chart-pie" style="color: var(--success); margin-right: 8px;"></i> Clean Patient Progress</h3>
                        <div class="chart-actions">
                            <button class="chart-action-btn tooltip" data-tooltip="Download"><i class="fas fa-download"></i></button>
                        </div>
                    </div>
                    <div id="clean_patient_progress"></div>
                </div>
                <div class="chart-card">
                    <div class="chart-header">
                        <h3><i class="fas fa-ban" style="color: var(--danger); margin-right: 8px;"></i> Blocking Categories</h3>
                    </div>
                    <div id="blocking_items"></div>
                </div>
            </div>
        </section>
        
        <!-- Recommendations Section -->
        <section id="recommendations" class="section">
            <h2 class="section-title"><i class="fas fa-brain"></i> AI Insights & Recommendations</h2>
            <div class="chart-grid">
                <div class="chart-card full-width">
                    <div class="chart-header">
                        <h3><i class="fas fa-robot" style="color: var(--primary); margin-right: 8px;"></i> Agent Analysis Overview</h3>
                        <div style="display: flex; gap: 12px; align-items: center;">
                            <select id="priorityFilter" class="filter-select" onchange="filterRecommendations()">
                                <option value="all">All Priorities</option>
                                <option value="critical">Critical Only</option>
                                <option value="high">High & Above</option>
                            </select>
                        </div>
                    </div>
                    <div id="agent_recommendations"></div>
                </div>
                <div class="chart-card full-width">
                    <div class="chart-header">
                        <h3><i class="fas fa-list-alt" style="color: var(--warning); margin-right: 8px;"></i> Action Items</h3>
                        <span class="badge">{total_recs} total</span>
                    </div>
                    {self._generate_recommendations_table(recommendations)}
                </div>
            </div>
        </section>
        
        <!-- Quality Section -->
        <section id="quality" class="section">
            <h2 class="section-title"><i class="fas fa-shield-alt"></i> Data Quality Intelligence</h2>
            <div class="chart-grid">
                <div class="chart-card full-width">
                    <div class="chart-header">
                        <h3><i class="fas fa-clipboard-check" style="color: var(--success); margin-right: 8px;"></i> Quality Metrics Dashboard</h3>
                    </div>
                    {self._generate_quality_summary(study_metrics, site_metrics)}
                </div>
            </div>
        </section>
    </main>
    
    <!-- Footer -->
    <footer class="footer">
        <div class="footer-logo">
            <i class="fas fa-dna"></i> Neural Clinical Data Mesh
        </div>
        <p>AI-Powered Clinical Trial Intelligence Platform</p>
        <p style="margin-top: 8px; opacity: 0.7;">Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        <p style="margin-top: 4px; font-size: 11px; opacity: 0.5;">&copy; {datetime.now().year} Clinical Dataflow Optimizer</p>
    </footer>
    
    <!-- Plotly Charts -->
    <script>
'''
        # Add Plotly figures
        for name, fig in self.figures.items():
            html_content += f'''
        var {name}_data = {fig.to_json()};
        if (document.getElementById('{name}')) {{
            Plotly.newPlot('{name}', {name}_data.data, {name}_data.layout, {{responsive: true, displayModeBar: false}});
        }}
'''
        
        html_content += '''
        // Theme Toggle
        function toggleTheme() {
            const body = document.body;
            const currentTheme = body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            body.setAttribute('data-theme', newTheme);
            localStorage.setItem('dashboard-theme', newTheme);
            
            // Update icon
            const icon = document.querySelector('.theme-toggle i');
            icon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            
            // Update Plotly charts for theme
            updateChartsTheme(newTheme);
        }
        
        // Load saved theme
        const savedTheme = localStorage.getItem('dashboard-theme') || 'light';
        if (savedTheme === 'dark') {
            document.body.setAttribute('data-theme', 'dark');
            document.querySelector('.theme-toggle i').className = 'fas fa-sun';
        }
        
        function updateChartsTheme(theme) {
            const bgColor = theme === 'dark' ? '#1e293b' : '#ffffff';
            const textColor = theme === 'dark' ? '#e2e8f0' : '#1e293b';
            const gridColor = theme === 'dark' ? '#334155' : '#e2e8f0';
            
            document.querySelectorAll('[id]').forEach(el => {
                if (el.data) {
                    Plotly.relayout(el.id, {
                        'paper_bgcolor': bgColor,
                        'plot_bgcolor': bgColor,
                        'font.color': textColor,
                        'xaxis.gridcolor': gridColor,
                        'yaxis.gridcolor': gridColor
                    });
                }
            });
        }
        
        // Navigation
        function showSection(sectionId) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(sectionId).classList.add('active');
            event.target.closest('.nav-btn').classList.add('active');
            
            // Resize charts after section becomes visible
            setTimeout(() => {
                window.dispatchEvent(new Event('resize'));
            }, 100);
        }
        
        // Global Search
        function filterDashboard(query) {
            const q = query.toLowerCase();
            document.querySelectorAll('.rec-table tbody tr').forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(q) ? '' : 'none';
            });
            document.querySelectorAll('.blocker-item').forEach(item => {
                const text = item.textContent.toLowerCase();
                item.style.display = text.includes(q) ? '' : 'none';
            });
        }
        
        // Filter Recommendations
        function filterRecommendations() {
            const filter = document.getElementById('priorityFilter').value;
            document.querySelectorAll('.rec-table tbody tr').forEach(row => {
                const badge = row.querySelector('.priority-badge');
                if (!badge) return;
                const priority = badge.textContent.toLowerCase();
                if (filter === 'all') {
                    row.style.display = '';
                } else if (filter === 'critical') {
                    row.style.display = priority === 'critical' ? '' : 'none';
                } else if (filter === 'high') {
                    row.style.display = (priority === 'critical' || priority === 'high') ? '' : 'none';
                }
            });
        }
        
        // Animate elements on scroll
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate__animated', 'animate__fadeInUp');
                }
            });
        }, { threshold: 0.1 });
        
        document.querySelectorAll('.chart-card, .kpi-card').forEach(el => observer.observe(el));
        
        // Resize charts on window resize
        window.addEventListener('resize', function() {
            document.querySelectorAll('[id]').forEach(el => {
                if (el.data) {
                    Plotly.Plots.resize(el);
                }
            });
        });
        
        // Initialize tooltips
        document.querySelectorAll('[data-tooltip]').forEach(el => {
            el.addEventListener('mouseenter', function() {
                this.style.position = 'relative';
            });
        });
    </script>
    
    <style>
        /* Filter Select */
        .filter-select {
            padding: 8px 16px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 13px;
            cursor: pointer;
        }
        
        .filter-select:focus {
            outline: none;
            border-color: var(--primary);
        }
        
        /* Chart action button */
        .chart-action-btn {
            width: 32px;
            height: 32px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-muted);
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .chart-action-btn:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        .chart-actions {
            display: flex;
            gap: 8px;
        }
        
        /* Badge in nav */
        .nav-btn .badge {
            margin-left: 6px;
            padding: 2px 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            font-size: 11px;
        }
        
        .badge.success { background: var(--success-glow); color: var(--success); }
        .badge.warning { background: var(--warning-glow); color: var(--warning); }
        .badge.danger { background: var(--danger-glow); color: var(--danger); animation: badgePulse 2s infinite; }
        
        /* Nav buttons container */
        .nav-buttons {
            display: flex;
            gap: 8px;
        }
    </style>
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_blockers_html(self, top_blockers: List, twins: List) -> str:
        """Generate HTML for top blockers section"""
        if not top_blockers:
            return '''<div style="text-align: center; padding: 60px 20px;">
                <i class="fas fa-check-circle" style="font-size: 64px; color: var(--success); margin-bottom: 20px; display: block;"></i>
                <h3 style="color: var(--text-primary); margin-bottom: 8px;">Excellent Data Quality!</h3>
                <p style="color: var(--text-muted);">No blocking items found - all patients are clean.</p>
            </div>'''
        
        max_count = top_blockers[0][1] if top_blockers else 1
        total_patients = len(twins) if twins else 1
        
        html = '<div class="blocker-list">'
        for i, (blocker, count) in enumerate(top_blockers, 1):
            pct = (count / max_count) * 100
            impact_pct = (count / total_patients) * 100
            severity = 'critical' if impact_pct > 20 else 'high' if impact_pct > 10 else 'medium' if impact_pct > 5 else 'low'
            html += f'''
            <div class="blocker-item" data-severity="{severity}">
                <div class="blocker-rank">{i}</div>
                <div class="blocker-info">
                    <div class="blocker-name">{blocker}</div>
                    <div class="blocker-count">
                        <i class="fas fa-users" style="margin-right: 4px;"></i>
                        {count:,} patients ({impact_pct:.1f}% impact)
                    </div>
                </div>
                <div class="blocker-bar">
                    <div class="blocker-bar-fill" style="width: {pct}%;"></div>
                </div>
            </div>
            '''
        html += '</div>'
        return html
    
    def _generate_recommendations_table(self, recommendations: List[Dict]) -> str:
        """Generate HTML table for recommendations"""
        if not recommendations:
            return '''<div style="text-align: center; padding: 60px 20px;">
                <i class="fas fa-sparkles" style="font-size: 64px; color: var(--success); margin-bottom: 20px; display: block;"></i>
                <h3 style="color: var(--text-primary); margin-bottom: 8px;">All Systems Go!</h3>
                <p style="color: var(--text-muted);">No action items - all quality checks passed.</p>
            </div>'''
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        sorted_recs = sorted(recommendations, key=lambda x: priority_order.get(x.get('priority', 'LOW').upper(), 4))
        
        html = '''
        <div style="overflow-x: auto;">
            <table class="rec-table">
                <thead>
                    <tr>
                        <th>Priority</th>
                        <th>Agent</th>
                        <th>Target</th>
                        <th>Action</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for rec in sorted_recs[:20]:  # Limit to top 20
            priority = rec.get('priority', 'LOW').upper()
            html += f'''
                <tr>
                    <td><span class="priority-badge {priority.lower()}">{priority}</span></td>
                    <td><strong>{rec.get('agent', 'Unknown')}</strong></td>
                    <td>{rec.get('target_type', '')} {rec.get('target_id', '')}</td>
                    <td>{rec.get('action_type', 'Review')}</td>
                    <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;">{rec.get('description', '')[:100]}...</td>
                </tr>
            '''
        
        html += '''
                </tbody>
            </table>
        </div>
        '''
        
        if recommendations and len(recommendations) > 20:
            html += f'<p style="text-align: center; margin-top: 16px; color: var(--gray-500);">Showing 20 of {len(recommendations)} recommendations</p>'
        
        return html
    
    def _generate_quality_summary(self, study_metrics: 'StudyMetrics', site_metrics: Dict[str, 'SiteMetrics']) -> str:
        """Generate quality summary HTML"""
        if not study_metrics:
            return '<p style="color: var(--text-muted);">No metrics available</p>'
        
        # Calculate risk distribution
        risk_dist = study_metrics.sites_by_risk if study_metrics else {}
        
        html = f'''
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px;">
            <div style="background: var(--bg-tertiary); padding: 24px; border-radius: 16px; border: 1px solid var(--border-color);">
                <h4 style="margin-bottom: 20px; color: var(--text-primary); display: flex; align-items: center; gap: 10px;">
                    <span style="width: 36px; height: 36px; background: var(--primary-glow); border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-chart-line" style="color: var(--primary);"></i>
                    </span>
                    Overall Metrics
                </h4>
                <div style="display: flex; flex-direction: column; gap: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; background: var(--bg-primary); border-radius: 10px;">
                        <span style="color: var(--text-muted);">Global DQI</span>
                        <strong style="font-size: 18px; color: var({'--success' if study_metrics.global_dqi >= 85 else '--warning' if study_metrics.global_dqi >= 70 else '--danger'});">{study_metrics.global_dqi:.1f}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; background: var(--bg-primary); border-radius: 10px;">
                        <span style="color: var(--text-muted);">Clean Rate</span>
                        <strong style="font-size: 18px; color: var({'--success' if study_metrics.global_clean_rate >= 80 else '--warning' if study_metrics.global_clean_rate >= 60 else '--danger'});">{study_metrics.global_clean_rate:.1f}%</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; background: var(--bg-primary); border-radius: 10px;">
                        <span style="color: var(--text-muted);">Interim Analysis</span>
                        <span style="display: flex; align-items: center; gap: 6px;">
                            <i class="fas fa-{'check-circle' if study_metrics.interim_analysis_ready else 'times-circle'}" style="color: var({'--success' if study_metrics.interim_analysis_ready else '--danger'});"></i>
                            <strong style="color: var({'--success' if study_metrics.interim_analysis_ready else '--danger'});">{'Ready' if study_metrics.interim_analysis_ready else 'Not Ready'}</strong>
                        </span>
                    </div>
                </div>
            </div>
            <div style="background: var(--bg-tertiary); padding: 24px; border-radius: 16px; border: 1px solid var(--border-color);">
                <h4 style="margin-bottom: 20px; color: var(--text-primary); display: flex; align-items: center; gap: 10px;">
                    <span style="width: 36px; height: 36px; background: var(--warning-glow); border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-hospital" style="color: var(--warning);"></i>
                    </span>
                    Site Risk Distribution
                </h4>
                <div style="display: flex; flex-direction: column; gap: 12px;">
        '''
        
        risk_colors = {'Critical': '--danger', 'High': '--warning', 'Medium': '#f97316', 'Low': '--success'}
        for risk, count in risk_dist.items():
            color_var = risk_colors.get(risk, '--text-muted')
            html += f'''
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; background: var(--bg-primary); border-radius: 10px;">
                        <span style="display: flex; align-items: center; gap: 10px; color: var(--text-secondary);">
                            <span style="width: 14px; height: 14px; background: var({color_var}); border-radius: 4px;"></span>
                            {risk} Risk
                        </span>
                        <strong style="font-size: 16px; color: var({color_var});">{count}</strong>
                    </div>
            '''
        
        html += '''
                </div>
            </div>
        </div>
        '''
        
        return html

    # ==========================================================================
    # ENHANCED SCIENTIFIC QUESTIONS VISUALIZATIONS
    # ==========================================================================
    
    def create_top_offenders_visualization(
        self,
        offenders_data: Dict[str, Any],
        title: str = "Visit Adherence: Top 10 Offenders"
    ) -> go.Figure:
        """
        Create visualization for Top 10 Offenders (Missing Visits)
        
        Answers: "Which sites/patients have the most missing visits?"
        """
        site_offenders = offenders_data.get('top_10_sites', [])
        
        if not site_offenders:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No offenders data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Extract data
        sites = [o.get('entity_id', 'Unknown') for o in site_offenders]
        days_outstanding = [o.get('max_days_outstanding', 0) for o in site_offenders]
        missing_visits = [o.get('missing_visits_count', 0) for o in site_offenders]
        priority_scores = [o.get('priority_score', 0) for o in site_offenders]
        
        # Create color scale based on days outstanding
        colors = []
        for days in days_outstanding:
            if days >= 60:
                colors.append(self.COLORS['Critical'])
            elif days >= 30:
                colors.append(self.COLORS['High'])
            elif days >= 14:
                colors.append(self.COLORS['Medium'])
            else:
                colors.append(self.COLORS['Low'])
        
        # Create figure with dual y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for days outstanding
        fig.add_trace(
            go.Bar(
                x=sites,
                y=days_outstanding,
                name='Max Days Outstanding',
                marker_color=colors,
                text=days_outstanding,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Days Outstanding: %{y}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Line for missing visits
        fig.add_trace(
            go.Scatter(
                x=sites,
                y=missing_visits,
                name='Missing Visits',
                mode='lines+markers',
                line=dict(color=self.COLORS['secondary'], width=3),
                marker=dict(size=10),
                hovertemplate='<b>%{x}</b><br>Missing Visits: %{y}<extra></extra>'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sub>Prioritized by Days Outstanding</sub>",
                x=0.5
            ),
            xaxis_title="Site ID",
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Days Outstanding", secondary_y=False)
        fig.update_yaxes(title_text="Missing Visits Count", secondary_y=True)
        
        self.figures['top_offenders'] = fig
        return fig
    
    def create_non_conformance_heatmap(
        self,
        heatmap_data: Dict[str, Any],
        title: str = "Non-Conformant Data Heatmap"
    ) -> go.Figure:
        """
        Create geographic heatmap of non-conformant data
        
        Answers: "Where are the highest rates of non-conformant data?"
        """
        hotspots = heatmap_data.get('hotspots_by_site', [])
        
        if not hotspots:
            fig = go.Figure()
            fig.add_annotation(text="No heatmap data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Extract data
        sites = [h.get('region_id', 'Unknown') for h in hotspots[:15]]  # Top 15
        non_conformant = [h.get('non_conformant_pages', 0) for h in hotspots[:15]]
        rates = [h.get('non_conformance_rate', 0) for h in hotspots[:15]]
        dqi_scores = [h.get('dqi_score', 0) for h in hotspots[:15]]
        
        # Sort by non-conformance rate
        sorted_data = sorted(zip(sites, non_conformant, rates, dqi_scores), 
                            key=lambda x: x[2], reverse=True)
        sites = [d[0] for d in sorted_data]
        non_conformant = [d[1] for d in sorted_data]
        rates = [d[2] for d in sorted_data]
        
        # Create color scale
        colors = []
        for rate in rates:
            if rate >= 0.10:
                colors.append('#FF0000')  # Red
            elif rate >= 0.05:
                colors.append('#FF6600')  # Orange
            elif rate >= 0.02:
                colors.append('#FFCC00')  # Yellow
            else:
                colors.append('#00CC00')  # Green
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=sites,
            x=non_conformant,
            orientation='h',
            marker_color=colors,
            text=[f"{r*100:.1f}%" for r in rates],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Non-Conformant Pages: %{x}<br>Rate: %{text}<extra></extra>'
        ))
        
        # Add threshold lines
        max_val = max(non_conformant) if non_conformant else 10
        fig.add_vline(x=10, line_dash="dash", line_color="red", 
                     annotation_text="Critical (10+)", annotation_position="top right")
        fig.add_vline(x=5, line_dash="dash", line_color="orange", 
                     annotation_text="High (5+)", annotation_position="top right")
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sub>Sites with Highest Non-Conformance Rates</sub>",
                x=0.5
            ),
            xaxis_title="# Non-Conformant Pages",
            yaxis_title="Site ID",
            height=max(400, len(sites) * 30),
            showlegend=False
        )
        
        self.figures['non_conformance_heatmap'] = fig
        return fig
    
    def create_delta_engine_visualization(
        self,
        delta_data: Dict[str, Any],
        title: str = "Site Intervention Analysis"
    ) -> go.Figure:
        """
        Create visualization for Delta Engine analysis
        
        Answers: "Which sites require immediate attention?"
        Shows sites with DQI < 75 AND negative velocity
        """
        site_metrics = delta_data.get('site_metrics', [])
        flagged_sites = delta_data.get('flagged_sites', [])
        immediate = delta_data.get('immediate_intervention_required', [])
        
        if not site_metrics:
            fig = go.Figure()
            fig.add_annotation(text="No delta analysis data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create scatter plot: DQI vs Velocity
        sites = [m.get('site_id', 'Unknown') for m in site_metrics]
        dqi_scores = [m.get('current_dqi', 0) for m in site_metrics]
        velocities = [m.get('velocity', 0) for m in site_metrics]
        trends = [m.get('trend', 'stable') for m in site_metrics]
        requires_intervention = [m.get('requires_intervention', False) for m in site_metrics]
        
        # Determine colors based on intervention need
        colors = []
        for i, req in enumerate(requires_intervention):
            if req:
                colors.append(self.COLORS['Critical'])
            elif velocities[i] < -5:
                colors.append(self.COLORS['High'])
            elif velocities[i] < 0:
                colors.append(self.COLORS['Medium'])
            else:
                colors.append(self.COLORS['Low'])
        
        # Determine sizes based on magnitude
        sizes = [max(10, abs(v) * 2 + 10) for v in velocities]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dqi_scores,
            y=velocities,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=2, color='white')
            ),
            text=sites,
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate=(
                '<b>%{text}</b><br>'
                'DQI: %{x:.1f}<br>'
                'Velocity: %{y:.2f}/week<br>'
                '<extra></extra>'
            )
        ))
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
        fig.add_vline(x=75, line_dash="dash", line_color="red", 
                     annotation_text="DQI < 75: Intervention Zone")
        fig.add_hline(y=-5, line_dash="dash", line_color="orange",
                     annotation_text="Negative Velocity Warning")
        
        # Add danger zone shading
        fig.add_shape(
            type="rect",
            x0=0, x1=75, y0=-20, y1=0,
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(width=0)
        )
        
        # Add annotations for quadrants
        fig.add_annotation(x=37.5, y=-10, text="⚠️ IMMEDIATE INTERVENTION", 
                          showarrow=False, font=dict(color="red", size=12))
        fig.add_annotation(x=87.5, y=5, text="✓ Healthy Sites", 
                          showarrow=False, font=dict(color="green", size=11))
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sub>Sites with DQI < 75 AND Negative Velocity Require Immediate Attention</sub>",
                x=0.5
            ),
            xaxis_title="Current DQI Score",
            yaxis_title="DQI Velocity (change per week)",
            height=550,
            showlegend=False
        )
        
        self.figures['delta_engine'] = fig
        return fig
    
    def create_global_cleanliness_gauge(
        self,
        cleanliness_data: Dict[str, Any],
        title: str = "Interim Analysis Readiness"
    ) -> go.Figure:
        """
        Create Global Cleanliness Meter visualization
        
        Answers: "Is the snapshot clean enough for interim analysis?"
        Outputs definitive YES/NO
        """
        answer = cleanliness_data.get('definitive_answer', 'NO')
        percentage = cleanliness_data.get('overall_clean_percentage', 0)
        clean_count = cleanliness_data.get('clean_patient_count', 0)
        total_count = cleanliness_data.get('total_patient_count', 1)
        ci = cleanliness_data.get('confidence_interval', {'lower': 0, 'upper': 0})
        
        # Determine color based on answer
        if answer == 'YES':
            bar_color = '#00CC00'
            status_color = 'green'
        elif answer == 'CONDITIONAL':
            bar_color = '#FFCC00'
            status_color = '#B8860B'
        else:
            bar_color = '#FF0000'
            status_color = 'red'
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
            column_widths=[0.6, 0.4]
        )
        
        # Main gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=percentage,
                number={'suffix': '%', 'font': {'size': 40}},
                title={'text': f"<b>Clean Patient Rate</b><br><span style='font-size:14px;color:{status_color}'>Answer: {answer}</span>"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': bar_color, 'thickness': 0.75},
                    'bgcolor': 'white',
                    'borderwidth': 2,
                    'bordercolor': 'gray',
                    'steps': [
                        {'range': [0, 80], 'color': '#fee2e2'},
                        {'range': [80, 100], 'color': '#dcfce7'}
                    ],
                    'threshold': {
                        'line': {'color': 'green', 'width': 4},
                        'thickness': 0.8,
                        'value': 80
                    }
                }
            ),
            row=1, col=1
        )
        
        # Patient count indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=clean_count,
                number={'font': {'size': 36}},
                title={'text': f"<b>Clean Patients</b><br><span style='font-size:12px'>of {total_count} total</span>"},
                delta={'reference': int(total_count * 0.8), 'relative': False, 'valueformat': '.0f'}
            ),
            row=1, col=2
        )
        
        # Add confidence interval annotation
        fig.add_annotation(
            x=0.3, y=-0.1,
            xref='paper', yref='paper',
            text=f"95% CI: {ci.get('lower', 0):.1f}% - {ci.get('upper', 0):.1f}%",
            showarrow=False,
            font=dict(size=12, color='gray')
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sub>Global Cleanliness Meter - Power Threshold: 80%</sub>",
                x=0.5
            ),
            height=400
        )
        
        self.figures['global_cleanliness'] = fig
        return fig
    
    def create_roi_dashboard(
        self,
        roi_data: Dict[str, Any],
        title: str = "ROI & Efficiency Metrics"
    ) -> go.Figure:
        """
        Create ROI metrics dashboard
        
        Shows efficiency gains, quality improvements, and financial impact
        """
        efficiency = roi_data.get('efficiency_metrics', {})
        quality = roi_data.get('quality_metrics', {})
        speed = roi_data.get('speed_metrics', {})
        financial = roi_data.get('financial_impact', {})
        
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
            ],
            subplot_titles=[
                'Query Automation Rate', 'DQI Improvement', 'Time Saved',
                'Hours Saved', 'Operational Savings', 'Total Value'
            ]
        )
        
        # Automation rate
        automation_rate = efficiency.get('automation_rate', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=automation_rate,
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#2563eb'},
                    'threshold': {'line': {'color': 'green', 'width': 3}, 'thickness': 0.75, 'value': 70}
                }
            ),
            row=1, col=1
        )
        
        # DQI improvement
        dqi_improvement = quality.get('dqi_improvement', 0)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=quality.get('current_dqi', 0),
                number={'suffix': ' DQI'},
                delta={'reference': quality.get('baseline_dqi', 75), 'valueformat': '+.1f'}
            ),
            row=1, col=2
        )
        
        # Time saved
        months_saved = speed.get('months_saved', 0)
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=months_saved,
                number={'suffix': ' months', 'font': {'color': 'green' if months_saved > 0 else 'red'}}
            ),
            row=1, col=3
        )
        
        # Hours saved
        hours_saved = efficiency.get('hours_saved', 0)
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=hours_saved,
                number={'suffix': ' hrs'}
            ),
            row=2, col=1
        )
        
        # Operational savings
        op_savings = financial.get('operational_cost_savings', 0)
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=op_savings,
                number={'prefix': '$', 'valueformat': ',.0f'}
            ),
            row=2, col=2
        )
        
        # Total value
        total_value = financial.get('total_value', 0)
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_value,
                number={'prefix': '$', 'valueformat': ',.0f', 'font': {'color': '#10b981', 'size': 36}}
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5),
            height=500
        )
        
        self.figures['roi_dashboard'] = fig
        return fig


def generate_ai_recommendations(
    twins: List[DigitalPatientTwin],
    site_metrics: Dict[str, SiteMetrics],
    study_metrics: StudyMetrics
) -> List[Dict]:
    """
    Generate AI-driven recommendations based on study data analysis.
    This simulates what the agent framework would produce.
    """
    recommendations = []
    
    # Analyze study-level issues
    total_patients = study_metrics.total_patients
    clean_rate = study_metrics.clean_patient_rate if hasattr(study_metrics, 'clean_patient_rate') else 0
    
    # 1. Site-level recommendations from site metrics
    for site_id, sm in site_metrics.items():
        site_dqi = sm.dqi_score if hasattr(sm, 'dqi_score') else 85
        site_patients = sm.total_patients if hasattr(sm, 'total_patients') else 0
        open_queries = sm.open_queries if hasattr(sm, 'open_queries') else 0
        missing_visits = sm.missing_visits if hasattr(sm, 'missing_visits') else 0
        
        # Critical: Low DQI sites
        if site_dqi < 70:
            recommendations.append({
                'agent': 'Lia (Site Liaison)',
                'action_type': 'Alert CRA',
                'priority': 'CRITICAL',
                'target': f'Site {site_id}',
                'title': f'Critical DQI Alert - Site {site_id}',
                'description': f'Site {site_id} has DQI of {site_dqi:.1f}% which is below critical threshold (70%). Immediate CRA intervention required.'
            })
        elif site_dqi < 80:
            recommendations.append({
                'agent': 'Lia (Site Liaison)',
                'action_type': 'Query to Site',
                'priority': 'HIGH',
                'target': f'Site {site_id}',
                'title': f'Data Quality Improvement Needed - Site {site_id}',
                'description': f'Site {site_id} DQI is {site_dqi:.1f}%. Schedule site training and implement quality improvement plan.'
            })
        
        # High open queries
        if open_queries > 10:
            priority = 'CRITICAL' if open_queries > 25 else 'HIGH'
            recommendations.append({
                'agent': 'Rex (Reconciliation)',
                'action_type': 'Query to DM',
                'priority': priority,
                'target': f'Site {site_id}',
                'title': f'Query Backlog - Site {site_id}',
                'description': f'Site {site_id} has {open_queries} open queries. Prioritize query resolution with Data Management team.'
            })
        
        # Missing visits
        if missing_visits > 5:
            recommendations.append({
                'agent': 'Lia (Site Liaison)',
                'action_type': 'Send Reminder',
                'priority': 'MEDIUM',
                'target': f'Site {site_id}',
                'title': f'Visit Completion Required - Site {site_id}',
                'description': f'Site {site_id} has {missing_visits} missing visit records. Send automated reminder to site coordinator.'
            })
    
    # 2. Patient-level recommendations from twins
    critical_patients = []
    high_risk_patients = []
    reconciliation_issues = []
    
    for twin in twins:
        # Check for blocking items
        if hasattr(twin, 'blocking_items') and twin.blocking_items:
            blocking_count = len(twin.blocking_items) if isinstance(twin.blocking_items, list) else 0
            
            if blocking_count >= 5:
                critical_patients.append(twin)
            elif blocking_count >= 3:
                high_risk_patients.append(twin)
        
        # Check reconciliation issues
        if hasattr(twin, 'reconciliation_issues') and twin.reconciliation_issues > 0:
            reconciliation_issues.append(twin)
        
        # Check SAE records needing attention
        if hasattr(twin, 'sae_records') and twin.sae_records:
            for sae in twin.sae_records if isinstance(twin.sae_records, list) else []:
                if hasattr(sae, 'status') and sae.status in ['Pending', 'Under Review']:
                    recommendations.append({
                        'agent': 'Rex (Reconciliation)',
                        'action_type': 'Alert Safety Team',
                        'priority': 'CRITICAL',
                        'target': f'Patient {twin.subject_id}',
                        'title': f'SAE Review Pending - {twin.subject_id}',
                        'description': f'SAE record for patient {twin.subject_id} at site {twin.site_id} requires safety team review.'
                    })
    
    # Aggregate patient-level findings
    if len(critical_patients) > 0:
        recommendations.append({
            'agent': 'Rex (Reconciliation)',
            'action_type': 'Alert CRA',
            'priority': 'CRITICAL',
            'target': f'{len(critical_patients)} Patients',
            'title': f'Critical Patient Data Issues',
            'description': f'{len(critical_patients)} patients have 5+ blocking items requiring immediate attention. Review and prioritize data cleanup.'
        })
    
    if len(high_risk_patients) > 0:
        recommendations.append({
            'agent': 'Codex (Coding)',
            'action_type': 'Query to DM',
            'priority': 'HIGH',
            'target': f'{len(high_risk_patients)} Patients',
            'title': f'High-Risk Patient Data Review',
            'description': f'{len(high_risk_patients)} patients have 3-4 blocking items. Schedule data review session with DM team.'
        })
    
    if len(reconciliation_issues) > 0:
        recommendations.append({
            'agent': 'Rex (Reconciliation)',
            'action_type': 'Query to Safety',
            'priority': 'HIGH',
            'target': f'{len(reconciliation_issues)} Records',
            'title': f'Safety-EDC Reconciliation Required',
            'description': f'{len(reconciliation_issues)} patient records have reconciliation discrepancies between Safety and EDC databases.'
        })
    
    # 3. Study-level recommendations
    if clean_rate < 20:
        recommendations.append({
            'agent': 'Lia (Site Liaison)',
            'action_type': 'Escalate',
            'priority': 'CRITICAL',
            'target': 'Study Leadership',
            'title': 'Critical Clean Patient Rate',
            'description': f'Study clean patient rate is {clean_rate:.1f}%, significantly below target. Escalate to study leadership for resource allocation.'
        })
    elif clean_rate < 40:
        recommendations.append({
            'agent': 'Lia (Site Liaison)',
            'action_type': 'Alert CRA',
            'priority': 'HIGH',
            'target': 'All CRAs',
            'title': 'Low Clean Patient Rate',
            'description': f'Study clean patient rate is {clean_rate:.1f}%. Implement study-wide data quality improvement initiative.'
        })
    
    # Sort by priority
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'LOW'), 4))
    
    return recommendations


def create_full_dashboard(
    twins: List[DigitalPatientTwin],
    site_metrics: Dict[str, SiteMetrics],
    study_metrics: StudyMetrics,
    recommendations: List = None,
    output_path: str = None
) -> DashboardVisualizer:
    """
    Create a complete dashboard with all visualizations
    """
    viz = DashboardVisualizer()
    
    # Auto-generate recommendations if not provided
    if recommendations is None:
        recommendations = generate_ai_recommendations(twins, site_metrics, study_metrics)
    
    # Create all visualizations
    viz.create_study_overview_dashboard(study_metrics)
    viz.create_dqi_heatmap(site_metrics)
    viz.create_clean_patient_progress(twins)
    viz.create_site_scatter_plot(site_metrics)
    viz.create_blocking_items_breakdown(twins)
    
    if recommendations:
        viz.create_agent_recommendations_summary(recommendations)
    
    # Export if path provided - pass additional data for enhanced dashboard
    if output_path:
        viz.export_all_figures_html(
            output_path, 
            study_metrics=study_metrics, 
            recommendations=recommendations,
            twins=twins,
            site_metrics=site_metrics
        )
    
    return viz


class RealTimeDashboardVisualizer(DashboardVisualizer):
    """
    Enhanced dashboard with real-time updates and live cleanliness tracking
    """

    def __init__(self, real_time_monitor: Optional[RealTimeDataMonitor] = None):
        super().__init__()
        self.real_time_monitor = real_time_monitor
        self.live_cleanliness_engine = LiveCleanlinessEngine()
        self.last_update_time = datetime.now()
        self.update_thread: Optional[threading.Thread] = None
        self.is_real_time_enabled = False

    def enable_real_time_updates(self, update_interval: int = 30):
        """Enable real-time dashboard updates"""
        self.is_real_time_enabled = True

        def update_loop():
            while self.is_real_time_enabled:
                try:
                    # Trigger dashboard refresh
                    self._refresh_real_time_data()
                    time.sleep(update_interval)
                except Exception as e:
                    print(f"Real-time update error: {e}")
                    time.sleep(60)  # Wait longer on error

        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()

    def disable_real_time_updates(self):
        """Disable real-time dashboard updates"""
        self.is_real_time_enabled = False
        if self.update_thread:
            self.update_thread.join(timeout=5)

    def _refresh_real_time_data(self):
        """Refresh real-time data components"""
        if self.real_time_monitor:
            # This would trigger data re-ingestion and status recalculation
            self.last_update_time = datetime.now()

    def create_live_cleanliness_dashboard(
        self,
        twins: List[DigitalPatientTwin],
        title: str = "Live Patient Cleanliness Dashboard"
    ) -> go.Figure:
        """Create real-time cleanliness dashboard with drill-down capabilities"""

        # Evaluate live cleanliness for all patients
        cleanliness_data = []
        for twin in twins:
            result = self.live_cleanliness_engine.evaluate_patient_cleanliness(twin)

            cleanliness_data.append({
                'subject_id': twin.subject_id,
                'site_id': twin.site_id,
                'cleanliness_score': result['cleanliness_score'],
                'is_clean': result['is_clean'],
                'blocking_factors': result['blocking_factors'],
                'rule_results': result['rule_results']
            })

        df = pd.DataFrame(cleanliness_data)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Real-Time Cleanliness Distribution',
                'Cleanliness Score Histogram',
                'Blocking Factors Summary',
                'Site Cleanliness Performance'
            ),
            specs=[
                [{"type": "pie"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )

        # 1. Cleanliness Distribution Pie Chart
        clean_count = df['is_clean'].sum()
        dirty_count = len(df) - clean_count

        fig.add_trace(
            go.Pie(
                labels=['Clean', 'Dirty'],
                values=[clean_count, dirty_count],
                marker_colors=['#10b981', '#ef4444'],
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>%{value} patients<br>%{percent}<extra></extra>',
                pull=[0.05, 0]
            ),
            row=1, col=1
        )

        # 2. Cleanliness Score Histogram
        fig.add_trace(
            go.Histogram(
                x=df['cleanliness_score'],
                nbinsx=20,
                marker_color='#3b82f6',
                hovertemplate='<b>Score Range: %{x}</b><br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # 3. Blocking Factors Summary
        blocking_counts = {}
        for factors in df['blocking_factors']:
            for factor in factors:
                blocking_counts[factor] = blocking_counts.get(factor, 0) + 1

        if blocking_counts:
            factors_df = pd.DataFrame(list(blocking_counts.items()), columns=['Factor', 'Count'])
            factors_df = factors_df.sort_values('Count', ascending=True)

            fig.add_trace(
                go.Bar(
                    x=factors_df['Count'],
                    y=factors_df['Factor'],
                    orientation='h',
                    marker_color='#f59e0b',
                    hovertemplate='<b>%{y}</b><br>%{x} patients<extra></extra>'
                ),
                row=2, col=1
            )

        # 4. Site Performance Scatter
        site_performance = df.groupby('site_id').agg({
            'cleanliness_score': 'mean',
            'is_clean': 'mean'  # Percentage clean
        }).reset_index()

        fig.add_trace(
            go.Scatter(
                x=site_performance['cleanliness_score'],
                y=site_performance['is_clean'] * 100,
                mode='markers+text',
                text=site_performance['site_id'],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=site_performance['cleanliness_score'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Avg Cleanliness Score")
                ),
                hovertemplate=(
                    '<b>Site %{text}</b><br>' +
                    'Avg Cleanliness: %{x:.1f}%<br>' +
                    'Clean Rate: %{y:.1f}%<extra></extra>'
                )
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><span style='font-size:12px;color:#64748b'>Last updated: {datetime.now().strftime('%H:%M:%S')}</span>",
                font=dict(size=16, family='Inter'),
                x=0.5
            ),
            height=800,
            showlegend=False,
            font=dict(family='Inter')
        )

        # Add real-time indicator
        fig.add_annotation(
            text="🔴 LIVE" if self.is_real_time_enabled else "⚫ STATIC",
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            font=dict(size=12, color="#ef4444" if self.is_real_time_enabled else "#6b7280"),
            bgcolor="white",
            bordercolor="#e5e7eb",
            borderwidth=1,
            borderpad=4
        )

        return fig

    def create_operational_velocity_dashboard(
        self,
        ovi_metrics: List[Dict],
        title: str = "Operational Velocity Index (OVI) Dashboard"
    ) -> go.Figure:
        """Create operational velocity dashboard"""

        if not ovi_metrics:
            # Create empty dashboard
            fig = go.Figure()
            fig.add_annotation(
                text="No operational velocity data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig

        df = pd.DataFrame(ovi_metrics)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'OVI Ranking by Site',
                'Query Resolution Times',
                'Velocity Trends',
                'Performance Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "histogram"}]
            ]
        )

        # 1. OVI Ranking
        top_sites = df.head(10)  # Top 10 sites
        fig.add_trace(
            go.Bar(
                x=top_sites['ovi_score'],
                y=top_sites['site_id'],
                orientation='h',
                marker_color='#10b981',
                hovertemplate=(
                    '<b>%{y}</b><br>' +
                    'OVI Score: %{x:.1f}<br>' +
                    'Rank: %{customdata}<extra></extra>'
                ),
                customdata=top_sites['rank']
            ),
            row=1, col=1
        )

        # 2. Query Resolution Times
        fig.add_trace(
            go.Bar(
                x=df['site_id'],
                y=df['avg_resolution_hours'],
                marker_color='#f59e0b',
                hovertemplate=(
                    '<b>%{x}</b><br>' +
                    'Avg Resolution: %{y:.1f} hours<extra></extra>'
                )
            ),
            row=1, col=2
        )

        # 3. Velocity Trends
        trend_values = [float(trend.strip('%')) for trend in df['velocity_trend']]
        colors = ['#10b981' if t > 0 else '#ef4444' for t in trend_values]

        fig.add_trace(
            go.Scatter(
                x=df['site_id'],
                y=trend_values,
                mode='markers+lines',
                marker=dict(color=colors, size=8),
                line=dict(color='#6b7280', width=2),
                hovertemplate=(
                    '<b>%{x}</b><br>' +
                    'Trend: %{y:+.1f}%<extra></extra>'
                )
            ),
            row=2, col=1
        )

        # 4. Performance Distribution
        fig.add_trace(
            go.Histogram(
                x=df['ovi_score'],
                nbinsx=10,
                marker_color='#3b82f6',
                hovertemplate='<b>OVI Range: %{x}</b><br>Site Count: %{y}<extra></extra>'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=16, family='Inter'),
                x=0.5
            ),
            height=800,
            showlegend=False,
            font=dict(family='Inter')
        )

        # Add trend line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", row=2, col=1)

        return fig

    def create_patient_drilldown_view(
        self,
        twin: DigitalPatientTwin,
        cleanliness_result: Dict,
        title: str = None
    ) -> go.Figure:
        """Create detailed patient drill-down view"""

        if title is None:
            title = f"Patient {twin.subject_id} - Cleanliness Analysis"

        # Create rule evaluation results
        rules_data = []
        for rule_name, rule_result in cleanliness_result['rule_results'].items():
            rules_data.append({
                'rule': rule_name,
                'passed': rule_result['passed'],
                'weight': rule_result['weight'],
                'description': rule_result['description']
            })

        rules_df = pd.DataFrame(rules_data)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cleanliness Score & Status',
                'Rule Evaluation Results',
                'Blocking Factors',
                'Patient Metrics Overview'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "table"}],
                [{"type": "bar" if cleanliness_result['blocking_factors'] else "pie"}, {"type": "bar"}]
            ]
        )

        # 1. Cleanliness Score Indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=cleanliness_result['cleanliness_score'],
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10b981" if cleanliness_result['is_clean'] else "#ef4444"},
                    'steps': [
                        {'range': [0, 50], 'color': "#fee2e2"},
                        {'range': [50, 75], 'color': "#fef3c7"},
                        {'range': [75, 100], 'color': "#d1fae5"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                },
                title={'text': f"Status: {'Clean' if cleanliness_result['is_clean'] else 'Dirty'}"}
            ),
            row=1, col=1
        )

        # 2. Rule Evaluation Table
        colors = ['#10b981' if passed else '#ef4444' for passed in rules_df['passed']]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Rule', 'Status', 'Weight', 'Description'],
                    fill_color='#f8f9fa',
                    align='left',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[
                        rules_df['rule'],
                        ['✅ Passed' if p else '❌ Failed' for p in rules_df['passed']],
                        [f"{w:.1f}" for w in rules_df['weight']],
                        rules_df['description']
                    ],
                    fill_color=[colors],
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=1, col=2
        )

        # 3. Blocking Factors or Clean Status
        if cleanliness_result['blocking_factors']:
            blocking_factors = cleanliness_result['blocking_factors'][:5]  # Top 5
            fig.add_trace(
                go.Bar(
                    x=list(range(len(blocking_factors))),
                    y=[1] * len(blocking_factors),  # Dummy values for labeling
                    text=blocking_factors,
                    textposition='inside',
                    marker_color='#ef4444',
                    showlegend=False,
                    hovertemplate='<b>Blocking Factor:</b><br>%{text}<extra></extra>'
                ),
                row=2, col=1
            )
        else:
            # Show clean celebration
            fig.add_trace(
                go.Pie(
                    labels=['Clean'],
                    values=[100],
                    marker_colors=['#10b981'],
                    textinfo='label+percent',
                    hovertemplate='<b>Status: Clean</b><br>All rules passed<extra></extra>'
                ),
                row=2, col=1
            )

        # 4. Patient Metrics Overview
        metrics = {
            'Missing Visits': twin.missing_visits,
            'Open Queries': twin.open_queries,
            'Uncoded Terms': twin.uncoded_terms,
            'SAE Reconciliation': 0 if twin.sae_reconciliation_confirmed else 1
        }

        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=['#ef4444' if v > 0 else '#10b981' for v in metrics.values()],
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=16, family='Inter'),
                x=0.5
            ),
            height=800,
            font=dict(family='Inter')
        )

        return fig
