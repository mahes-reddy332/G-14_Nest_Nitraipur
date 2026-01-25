"""
Graph Analytics Module for Neural Clinical Data Mesh
Advanced analytics leveraging the graph structure for clinical insights

This module provides:
- Network analysis metrics (centrality, clustering)
- Pattern detection across the knowledge graph
- Anomaly detection using graph properties
- Predictive risk scoring using graph features
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

from .knowledge_graph import (
    ClinicalKnowledgeGraph, NodeType, EdgeType,
    PatientNode, GraphNode
)
from .graph_queries import GraphQueryEngine, PatientQueryResult

logger = logging.getLogger(__name__)


@dataclass
class PatientRiskProfile:
    """Comprehensive risk profile for a patient based on graph analysis"""
    subject_id: str
    study_id: str
    
    # Graph-based metrics
    connection_count: int = 0
    issue_diversity: int = 0  # Number of different issue types
    
    # Centrality scores (how connected/important this patient is in the network)
    degree_centrality: float = 0.0
    
    # Risk scores
    composite_risk_score: float = 0.0
    data_quality_risk: float = 0.0
    safety_risk: float = 0.0
    compliance_risk: float = 0.0
    
    # Flags
    is_critical: bool = False
    requires_immediate_attention: bool = False
    
    # Related issues breakdown
    issues: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'study_id': self.study_id,
            'connection_count': self.connection_count,
            'issue_diversity': self.issue_diversity,
            'degree_centrality': round(self.degree_centrality, 4),
            'composite_risk_score': round(self.composite_risk_score, 2),
            'data_quality_risk': round(self.data_quality_risk, 2),
            'safety_risk': round(self.safety_risk, 2),
            'compliance_risk': round(self.compliance_risk, 2),
            'is_critical': self.is_critical,
            'requires_immediate_attention': self.requires_immediate_attention,
            'issues': self.issues
        }


@dataclass
class SiteRiskProfile:
    """Risk profile for a clinical site based on aggregated patient graph data"""
    site_id: str
    study_id: str
    
    # Patient counts
    total_patients: int = 0
    clean_patients: int = 0
    at_risk_patients: int = 0
    critical_patients: int = 0
    
    # Aggregated risk scores
    average_patient_risk: float = 0.0
    max_patient_risk: float = 0.0
    
    # Issue patterns
    common_issues: List[Tuple[str, int]] = field(default_factory=list)
    
    # Site-level risk assessment
    site_risk_level: str = "Low"
    requires_intervention: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'study_id': self.study_id,
            'total_patients': self.total_patients,
            'clean_patients': self.clean_patients,
            'at_risk_patients': self.at_risk_patients,
            'critical_patients': self.critical_patients,
            'clean_rate': round((self.clean_patients / max(self.total_patients, 1)) * 100, 1),
            'average_patient_risk': round(self.average_patient_risk, 2),
            'max_patient_risk': round(self.max_patient_risk, 2),
            'common_issues': self.common_issues[:5],
            'site_risk_level': self.site_risk_level,
            'requires_intervention': self.requires_intervention
        }


class GraphAnalytics:
    """
    Advanced graph analytics for clinical trial data
    
    Leverages the NetworkX graph structure to compute:
    - Patient risk profiles using graph metrics
    - Site risk aggregations
    - Pattern detection across the network
    - Anomaly identification
    """
    
    def __init__(self, graph: ClinicalKnowledgeGraph):
        self.graph = graph
        self.query_engine = GraphQueryEngine(graph)
        self._centrality_cache: Dict[str, float] = {}
    
    # ==================== Patient Risk Analysis ====================
    
    def analyze_patient_risk(self, subject_id: str) -> PatientRiskProfile:
        """
        Compute comprehensive risk profile for a patient using graph metrics
        """
        patient_node = self.graph.get_patient_node(subject_id)
        if not patient_node:
            return None
        
        attrs = patient_node.attributes
        node_id = patient_node.node_id
        
        profile = PatientRiskProfile(
            subject_id=subject_id,
            study_id=self.graph.study_id
        )
        
        # Get graph-based metrics
        neighbors = self.graph.get_patient_neighbors(subject_id)
        
        # Count connections (edges to issue nodes)
        total_connections = sum(len(nodes) for nodes in neighbors.values())
        profile.connection_count = total_connections
        
        # Count issue diversity (how many different types of issues)
        issue_types = [et for et, nodes in neighbors.items() if nodes]
        profile.issue_diversity = len(issue_types)
        
        # Calculate centrality (if not cached)
        if node_id not in self._centrality_cache:
            self._compute_centrality_metrics()
        profile.degree_centrality = self._centrality_cache.get(node_id, 0.0)
        
        # Calculate risk components
        profile.data_quality_risk = self._calculate_data_quality_risk(attrs)
        profile.safety_risk = self._calculate_safety_risk(attrs, neighbors)
        profile.compliance_risk = self._calculate_compliance_risk(attrs, neighbors)
        
        # Composite risk score (weighted average)
        profile.composite_risk_score = (
            profile.data_quality_risk * 0.3 +
            profile.safety_risk * 0.4 +
            profile.compliance_risk * 0.3
        )
        
        # Issue breakdown
        profile.issues = {
            'missing_visits': attrs.get('missing_visits', 0),
            'open_queries': attrs.get('open_queries', 0),
            'uncoded_terms': attrs.get('uncoded_terms', 0),
            'reconciliation_issues': attrs.get('reconciliation_issues', 0),
            'protocol_deviations': attrs.get('protocol_deviations', 0),
            'connected_sae_count': len(neighbors.get(EdgeType.HAS_ADVERSE_EVENT, [])),
            'connected_coding_issues': len(neighbors.get(EdgeType.HAS_CODING_ISSUE, []))
        }
        
        # Determine criticality
        profile.is_critical = profile.composite_risk_score >= 70
        profile.requires_immediate_attention = (
            profile.is_critical or 
            attrs.get('reconciliation_issues', 0) > 0 or
            profile.safety_risk >= 80
        )
        
        return profile
    
    def _calculate_data_quality_risk(self, attrs: Dict) -> float:
        """Calculate data quality risk component"""
        risk = 0.0
        
        # Missing visits penalty
        missing_visits = attrs.get('missing_visits', 0)
        risk += min(missing_visits * 10, 30)
        
        # Open queries penalty
        open_queries = attrs.get('open_queries', 0)
        risk += min(open_queries * 5, 30)
        
        # Verification percentage (inverse)
        verification = attrs.get('verification_pct', 100)
        if verification < 100:
            risk += (100 - verification) * 0.4
        
        return min(risk, 100)
    
    def _calculate_safety_risk(self, attrs: Dict, neighbors: Dict) -> float:
        """Calculate safety risk component"""
        risk = 0.0
        
        # Reconciliation issues (critical)
        recon_issues = attrs.get('reconciliation_issues', 0)
        risk += min(recon_issues * 25, 50)
        
        # SAE count from graph
        sae_count = len(neighbors.get(EdgeType.HAS_ADVERSE_EVENT, []))
        risk += min(sae_count * 15, 40)
        
        # Check SAE statuses
        for sae_node in neighbors.get(EdgeType.HAS_ADVERSE_EVENT, []):
            status = sae_node.attributes.get('review_status', '')
            if 'pending' in status.lower():
                risk += 10
        
        return min(risk, 100)
    
    def _calculate_compliance_risk(self, attrs: Dict, neighbors: Dict) -> float:
        """Calculate compliance risk component"""
        risk = 0.0
        
        # Protocol deviations
        pds = attrs.get('protocol_deviations', 0)
        risk += min(pds * 15, 40)
        
        # Uncoded terms
        uncoded = attrs.get('uncoded_terms', 0)
        risk += min(uncoded * 8, 30)
        
        # Missing pages
        missing_pages = attrs.get('missing_pages', 0)
        risk += min(missing_pages * 5, 25)
        
        # Outstanding visits from graph
        visit_nodes = neighbors.get(EdgeType.HAS_VISIT, [])
        overdue_visits = sum(
            1 for v in visit_nodes 
            if v.attributes.get('days_outstanding', 0) > 30
        )
        risk += min(overdue_visits * 10, 30)
        
        return min(risk, 100)
    
    def _compute_centrality_metrics(self):
        """Compute and cache centrality metrics for all nodes"""
        try:
            centrality = nx.degree_centrality(self.graph.graph)
            self._centrality_cache = centrality
        except Exception as e:
            logger.warning(f"Error computing centrality: {e}")
            self._centrality_cache = {}
    
    # ==================== Site Risk Analysis ====================
    
    def analyze_site_risk(self, site_id: str) -> SiteRiskProfile:
        """Compute risk profile for a clinical site"""
        profile = SiteRiskProfile(
            site_id=site_id,
            study_id=self.graph.study_id
        )
        
        # Get all patients at this site
        patient_nodes = self.graph.get_nodes_by_type(NodeType.PATIENT)
        site_patients = [
            p for p in patient_nodes 
            if p.attributes.get('site_id') == site_id
        ]
        
        profile.total_patients = len(site_patients)
        
        if profile.total_patients == 0:
            return profile
        
        # Analyze each patient
        risk_scores = []
        issue_counter = defaultdict(int)
        
        for patient in site_patients:
            attrs = patient.attributes
            subject_id = attrs.get('subject_id')
            
            # Check clean status
            if attrs.get('clean_status', False):
                profile.clean_patients += 1
            
            # Get patient risk
            patient_risk = self.analyze_patient_risk(subject_id)
            if patient_risk:
                risk_scores.append(patient_risk.composite_risk_score)
                
                if patient_risk.is_critical:
                    profile.critical_patients += 1
                elif patient_risk.composite_risk_score >= 50:
                    profile.at_risk_patients += 1
                
                # Track issues
                for issue_type, count in patient_risk.issues.items():
                    if count > 0:
                        issue_counter[issue_type] += count
        
        # Aggregate risk scores
        if risk_scores:
            profile.average_patient_risk = np.mean(risk_scores)
            profile.max_patient_risk = np.max(risk_scores)
        
        # Common issues
        profile.common_issues = sorted(
            issue_counter.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Determine site risk level
        if profile.critical_patients >= 3 or profile.average_patient_risk >= 70:
            profile.site_risk_level = "Critical"
            profile.requires_intervention = True
        elif profile.at_risk_patients >= 5 or profile.average_patient_risk >= 50:
            profile.site_risk_level = "High"
            profile.requires_intervention = True
        elif profile.average_patient_risk >= 30:
            profile.site_risk_level = "Medium"
        else:
            profile.site_risk_level = "Low"
        
        return profile
    
    def analyze_all_sites(self) -> Dict[str, SiteRiskProfile]:
        """Analyze risk for all sites in the graph"""
        site_nodes = self.graph.get_nodes_by_type(NodeType.SITE)
        
        return {
            site.attributes.get('site_id'): self.analyze_site_risk(
                site.attributes.get('site_id')
            )
            for site in site_nodes
        }
    
    # ==================== Pattern Detection ====================
    
    def detect_issue_patterns(self) -> Dict[str, Any]:
        """Detect patterns in issues across the knowledge graph"""
        patterns = {
            'issue_correlations': {},
            'site_patterns': {},
            'temporal_patterns': {},
            'high_risk_clusters': []
        }
        
        # Analyze issue co-occurrence
        patient_nodes = self.graph.get_nodes_by_type(NodeType.PATIENT)
        
        issue_pairs = defaultdict(int)
        for patient in patient_nodes:
            attrs = patient.attributes
            issues = []
            
            if attrs.get('missing_visits', 0) > 0:
                issues.append('missing_visits')
            if attrs.get('open_queries', 0) > 0:
                issues.append('open_queries')
            if attrs.get('uncoded_terms', 0) > 0:
                issues.append('uncoded_terms')
            if attrs.get('reconciliation_issues', 0) > 0:
                issues.append('reconciliation_issues')
            
            # Count co-occurrences
            for i, issue1 in enumerate(issues):
                for issue2 in issues[i+1:]:
                    pair = tuple(sorted([issue1, issue2]))
                    issue_pairs[pair] += 1
        
        patterns['issue_correlations'] = dict(issue_pairs)
        
        # Site-level patterns
        site_profiles = self.analyze_all_sites()
        high_risk_sites = [
            site_id for site_id, profile in site_profiles.items()
            if profile.site_risk_level in ['Critical', 'High']
        ]
        
        patterns['site_patterns'] = {
            'high_risk_site_count': len(high_risk_sites),
            'high_risk_sites': high_risk_sites
        }
        
        return patterns
    
    def find_similar_patients(
        self, 
        subject_id: str, 
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find patients with similar issue profiles using graph structure
        
        This leverages the graph to find patients with similar connected issues.
        """
        target_node = self.graph.get_patient_node(subject_id)
        if not target_node:
            return []
        
        target_neighbors = self.graph.get_patient_neighbors(subject_id)
        target_edge_types = set(et for et, nodes in target_neighbors.items() if nodes)
        target_attrs = target_node.attributes
        
        similarities = []
        
        for patient in self.graph.get_nodes_by_type(NodeType.PATIENT):
            if patient.attributes.get('subject_id') == subject_id:
                continue
            
            other_id = patient.attributes.get('subject_id')
            other_neighbors = self.graph.get_patient_neighbors(other_id)
            other_edge_types = set(et for et, nodes in other_neighbors.items() if nodes)
            
            # Calculate Jaccard similarity of edge types
            if target_edge_types or other_edge_types:
                intersection = len(target_edge_types & other_edge_types)
                union = len(target_edge_types | other_edge_types)
                edge_similarity = intersection / union if union > 0 else 0
            else:
                edge_similarity = 1.0  # Both have no issues
            
            # Calculate attribute similarity
            attr_similarity = self._calculate_attribute_similarity(
                target_attrs, patient.attributes
            )
            
            # Combined similarity
            combined = (edge_similarity * 0.6) + (attr_similarity * 0.4)
            similarities.append((other_id, combined))
        
        # Return top N similar patients
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def _calculate_attribute_similarity(
        self, 
        attrs1: Dict, 
        attrs2: Dict
    ) -> float:
        """Calculate similarity between two sets of patient attributes"""
        # Compare numeric attributes
        numeric_attrs = [
            'missing_visits', 'open_queries', 'uncoded_terms',
            'reconciliation_issues', 'protocol_deviations'
        ]
        
        diffs = []
        for attr in numeric_attrs:
            v1 = attrs1.get(attr, 0)
            v2 = attrs2.get(attr, 0)
            max_val = max(v1, v2, 1)
            diff = abs(v1 - v2) / max_val
            diffs.append(1 - diff)
        
        return np.mean(diffs) if diffs else 0.0
    
    # ==================== Network Statistics ====================
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        stats = self.graph.get_statistics()
        
        # Add advanced metrics
        G = self.graph.graph
        
        if G.number_of_nodes() > 0:
            # Components
            if G.is_directed():
                stats['weakly_connected_components'] = nx.number_weakly_connected_components(G)
            
            # Average clustering (for undirected approximation)
            try:
                undirected = G.to_undirected()
                stats['average_clustering'] = round(nx.average_clustering(undirected), 4)
            except:
                stats['average_clustering'] = 0
            
            # Degree distribution summary
            degrees = [d for n, d in G.degree()]
            stats['degree_stats'] = {
                'min': min(degrees),
                'max': max(degrees),
                'mean': round(np.mean(degrees), 2),
                'median': np.median(degrees)
            }
        
        return stats
    
    def export_risk_report(self) -> Dict[str, Any]:
        """Export comprehensive risk report"""
        report = {
            'study_id': self.graph.study_id,
            'generated_at': datetime.now().isoformat(),
            'network_statistics': self.get_network_statistics(),
            'site_profiles': {},
            'patient_profiles': [],
            'patterns': self.detect_issue_patterns()
        }
        
        # Site profiles
        for site_id, profile in self.analyze_all_sites().items():
            report['site_profiles'][site_id] = profile.to_dict()
        
        # Patient profiles (top 20 highest risk)
        patient_risks = []
        for patient in self.graph.get_nodes_by_type(NodeType.PATIENT):
            subject_id = patient.attributes.get('subject_id')
            profile = self.analyze_patient_risk(subject_id)
            if profile:
                patient_risks.append(profile)
        
        patient_risks.sort(key=lambda x: x.composite_risk_score, reverse=True)
        report['patient_profiles'] = [p.to_dict() for p in patient_risks[:20]]
        
        return report
