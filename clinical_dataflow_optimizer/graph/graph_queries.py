"""
Graph Query Engine for Neural Clinical Data Mesh
Enables complex multi-hop queries that relational databases struggle to perform efficiently

Key Capabilities:
- Multi-condition patient queries (e.g., patients with Missing Visit AND Open Query AND Uncoded Term)
- Graph pattern matching
- Aggregation queries across the knowledge graph
- Risk-based patient prioritization
- Data mesh analytics
"""

import networkx as nx
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .knowledge_graph import (
    ClinicalKnowledgeGraph, NodeType, EdgeType,
    PatientNode, EventNode, DiscrepancyNode, SAENode, CodingTermNode,
    GraphNode, QueryStatus, CodingStatus
)

logger = logging.getLogger(__name__)


class QueryOperator(Enum):
    """Operators for query conditions"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    CONTAINS = "contains"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


@dataclass
class QueryCondition:
    """Represents a single query condition"""
    field: str
    operator: QueryOperator
    value: Any
    
    def evaluate(self, node_attributes: Dict[str, Any]) -> bool:
        """Evaluate this condition against node attributes"""
        actual_value = node_attributes.get(self.field)
        
        if self.operator == QueryOperator.EXISTS:
            return actual_value is not None
        if self.operator == QueryOperator.NOT_EXISTS:
            return actual_value is None
        
        if actual_value is None:
            return False
        
        if self.operator == QueryOperator.EQUALS:
            return actual_value == self.value
        elif self.operator == QueryOperator.NOT_EQUALS:
            return actual_value != self.value
        elif self.operator == QueryOperator.GREATER_THAN:
            return actual_value > self.value
        elif self.operator == QueryOperator.LESS_THAN:
            return actual_value < self.value
        elif self.operator == QueryOperator.GREATER_EQUAL:
            return actual_value >= self.value
        elif self.operator == QueryOperator.LESS_EQUAL:
            return actual_value <= self.value
        elif self.operator == QueryOperator.CONTAINS:
            return str(self.value).lower() in str(actual_value).lower()
        elif self.operator == QueryOperator.IN:
            return actual_value in self.value
        elif self.operator == QueryOperator.NOT_IN:
            return actual_value not in self.value
        
        return False


@dataclass
class PatientQueryResult:
    """Result of a patient query"""
    subject_id: str
    node_id: str
    patient_attributes: Dict[str, Any]
    matched_conditions: List[str]
    related_nodes: Dict[str, List[GraphNode]] = field(default_factory=dict)
    risk_score: float = 0.0
    priority_rank: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'node_id': self.node_id,
            'attributes': self.patient_attributes,
            'matched_conditions': self.matched_conditions,
            'related_nodes_count': {k: len(v) for k, v in self.related_nodes.items()},
            'risk_score': round(self.risk_score, 2),
            'priority_rank': self.priority_rank
        }


class GraphQueryEngine:
    """
    Query engine for the Clinical Knowledge Graph
    
    Enables complex multi-hop queries like:
    "Show me all patients who have a Missing Visit (from Tracker) 
    AND an Open Safety Query (from CPID) 
    AND an Uncoded Concomitant Medication (from WHODRA)"
    
    This is a simple traversal of patient node neighbors in the graph,
    whereas in SQL this would require joining three distinct tables.
    """
    
    def __init__(self, graph: ClinicalKnowledgeGraph):
        self.graph = graph
        self.query_cache: Dict[str, List[PatientQueryResult]] = {}
    
    # ==================== Core Query Methods ====================
    
    def find_patients_with_conditions(
        self,
        patient_conditions: List[QueryCondition] = None,
        must_have_edges: List[EdgeType] = None,
        neighbor_conditions: Dict[EdgeType, List[QueryCondition]] = None,
        logic: str = "AND"  # AND or OR
    ) -> List[PatientQueryResult]:
        """
        Find patients matching specified conditions
        
        This is the primary multi-hop query method that enables complex queries
        impossible to express efficiently in SQL.
        
        Args:
            patient_conditions: Conditions on patient node attributes
            must_have_edges: Edge types the patient must have
            neighbor_conditions: Conditions on neighbor nodes via specific edge types
            logic: "AND" (all conditions must match) or "OR" (any condition matches)
            
        Returns:
            List of PatientQueryResult matching the criteria
        """
        results = []
        patient_nodes = self.graph.get_nodes_by_type(NodeType.PATIENT)
        
        for patient_node in patient_nodes:
            if not hasattr(patient_node, 'subject_id'):
                # Get subject_id from attributes
                subject_id = patient_node.attributes.get('subject_id', '')
            else:
                subject_id = patient_node.subject_id
            
            matched_conditions = []
            all_conditions_met = True
            any_condition_met = False
            related_nodes = {}
            
            # Check patient node conditions
            if patient_conditions:
                for condition in patient_conditions:
                    if condition.evaluate(patient_node.attributes):
                        matched_conditions.append(f"patient.{condition.field}")
                        any_condition_met = True
                    elif logic == "AND":
                        all_conditions_met = False
                        break
            
            # Check required edge types
            if must_have_edges and all_conditions_met:
                neighbors = self.graph.get_patient_neighbors(subject_id)
                for edge_type in must_have_edges:
                    if edge_type in neighbors and len(neighbors[edge_type]) > 0:
                        matched_conditions.append(f"has_{edge_type.value}")
                        related_nodes[edge_type.value] = neighbors[edge_type]
                        any_condition_met = True
                    elif logic == "AND":
                        all_conditions_met = False
                        break
            
            # Check neighbor conditions
            if neighbor_conditions and all_conditions_met:
                neighbors = self.graph.get_patient_neighbors(subject_id)
                for edge_type, conditions in neighbor_conditions.items():
                    neighbor_nodes = neighbors.get(edge_type, [])
                    
                    matching_neighbors = []
                    for neighbor in neighbor_nodes:
                        neighbor_matches = all(
                            cond.evaluate(neighbor.attributes) 
                            for cond in conditions
                        )
                        if neighbor_matches:
                            matching_neighbors.append(neighbor)
                    
                    if matching_neighbors:
                        condition_desc = f"{edge_type.value}[{len(matching_neighbors)} matches]"
                        matched_conditions.append(condition_desc)
                        related_nodes[edge_type.value] = matching_neighbors
                        any_condition_met = True
                    elif logic == "AND":
                        all_conditions_met = False
                        break
            
            # Determine if patient matches
            should_include = (logic == "AND" and all_conditions_met) or \
                           (logic == "OR" and any_condition_met)
            
            if should_include and matched_conditions:
                result = PatientQueryResult(
                    subject_id=subject_id,
                    node_id=patient_node.node_id,
                    patient_attributes=patient_node.attributes,
                    matched_conditions=matched_conditions,
                    related_nodes=related_nodes
                )
                results.append(result)
        
        return results
    
    def find_patients_needing_attention(self) -> List[PatientQueryResult]:
        """
        The flagship multi-hop query:
        Find all patients who have:
        - A Missing Visit (from Visit Tracker) AND
        - An Open Safety Query (from CPID) AND  
        - An Uncoded Term (from GlobalCodingReport)
        
        This demonstrates the power of graph traversal over SQL joins.
        """
        # Define the conditions
        patient_conditions = [
            QueryCondition('missing_visits', QueryOperator.GREATER_THAN, 0),
            QueryCondition('open_queries', QueryOperator.GREATER_THAN, 0),
            QueryCondition('uncoded_terms', QueryOperator.GREATER_THAN, 0)
        ]
        
        # Execute query
        results = self.find_patients_with_conditions(
            patient_conditions=patient_conditions,
            must_have_edges=[EdgeType.HAS_VISIT, EdgeType.HAS_QUERY, EdgeType.HAS_CODING_ISSUE],
            logic="AND"
        )
        
        # Calculate risk scores
        for result in results:
            result.risk_score = self._calculate_attention_risk_score(result)
        
        # Sort by risk score (highest first)
        results.sort(key=lambda x: x.risk_score, reverse=True)
        
        # Assign priority ranks
        for i, result in enumerate(results):
            result.priority_rank = i + 1
        
        return results
    
    def find_patients_with_missing_visits(
        self,
        min_days_outstanding: int = 0
    ) -> List[PatientQueryResult]:
        """Find patients with missing visits"""
        patient_conditions = [
            QueryCondition('missing_visits', QueryOperator.GREATER_THAN, 0)
        ]
        
        neighbor_conditions = {}
        if min_days_outstanding > 0:
            neighbor_conditions[EdgeType.HAS_VISIT] = [
                QueryCondition('days_outstanding', QueryOperator.GREATER_EQUAL, min_days_outstanding),
                QueryCondition('is_missing', QueryOperator.EQUALS, True)
            ]
        
        return self.find_patients_with_conditions(
            patient_conditions=patient_conditions,
            neighbor_conditions=neighbor_conditions if neighbor_conditions else None,
            logic="AND"
        )
    
    def find_patients_with_open_queries(
        self,
        query_type: str = None,
        min_days_open: int = 0
    ) -> List[PatientQueryResult]:
        """Find patients with open queries"""
        patient_conditions = [
            QueryCondition('open_queries', QueryOperator.GREATER_THAN, 0)
        ]
        
        neighbor_conditions = {
            EdgeType.HAS_QUERY: [
                QueryCondition('status', QueryOperator.EQUALS, 'Open')
            ]
        }
        
        if query_type:
            neighbor_conditions[EdgeType.HAS_QUERY].append(
                QueryCondition('query_type', QueryOperator.EQUALS, query_type)
            )
        
        if min_days_open > 0:
            neighbor_conditions[EdgeType.HAS_QUERY].append(
                QueryCondition('days_open', QueryOperator.GREATER_EQUAL, min_days_open)
            )
        
        return self.find_patients_with_conditions(
            patient_conditions=patient_conditions,
            neighbor_conditions=neighbor_conditions,
            logic="AND"
        )
    
    def find_patients_with_uncoded_terms(
        self,
        coding_dictionary: str = None
    ) -> List[PatientQueryResult]:
        """Find patients with uncoded medical terms"""
        patient_conditions = [
            QueryCondition('uncoded_terms', QueryOperator.GREATER_THAN, 0)
        ]
        
        neighbor_conditions = {
            EdgeType.HAS_CODING_ISSUE: [
                QueryCondition('coding_status', QueryOperator.EQUALS, 'UnCoded')
            ]
        }
        
        if coding_dictionary:
            neighbor_conditions[EdgeType.HAS_CODING_ISSUE].append(
                QueryCondition('coding_dictionary', QueryOperator.EQUALS, coding_dictionary)
            )
        
        return self.find_patients_with_conditions(
            patient_conditions=patient_conditions,
            neighbor_conditions=neighbor_conditions,
            logic="AND"
        )
    
    def find_patients_with_sae_issues(
        self,
        review_status: str = "Pending",
        min_days_pending: int = 0
    ) -> List[PatientQueryResult]:
        """Find patients with SAE issues requiring attention"""
        neighbor_conditions = {
            EdgeType.HAS_ADVERSE_EVENT: [
                QueryCondition('review_status', QueryOperator.CONTAINS, review_status)
            ]
        }
        
        if min_days_pending > 0:
            neighbor_conditions[EdgeType.HAS_ADVERSE_EVENT].append(
                QueryCondition('days_pending', QueryOperator.GREATER_EQUAL, min_days_pending)
            )
        
        return self.find_patients_with_conditions(
            must_have_edges=[EdgeType.HAS_ADVERSE_EVENT],
            neighbor_conditions=neighbor_conditions,
            logic="AND"
        )
    
    def find_patients_with_reconciliation_issues(self) -> List[PatientQueryResult]:
        """Find patients with EDC/Safety reconciliation issues"""
        patient_conditions = [
            QueryCondition('reconciliation_issues', QueryOperator.GREATER_THAN, 0)
        ]
        
        return self.find_patients_with_conditions(
            patient_conditions=patient_conditions,
            logic="AND"
        )
    
    # ==================== Complex Multi-Hop Queries ====================
    
    def execute_mesh_query(
        self,
        query_spec: Dict[str, Any]
    ) -> List[PatientQueryResult]:
        """
        Execute a Data Mesh query from a specification dictionary
        
        This allows for dynamic query construction from external sources
        (e.g., UI filters, configuration files, API requests)
        
        Query spec format:
        {
            "patient_filters": [
                {"field": "missing_visits", "op": "gt", "value": 0}
            ],
            "required_relationships": ["HAS_VISIT", "HAS_QUERY"],
            "neighbor_filters": {
                "HAS_QUERY": [
                    {"field": "status", "op": "eq", "value": "Open"}
                ]
            },
            "logic": "AND"
        }
        """
        # Parse patient conditions
        patient_conditions = []
        for spec in query_spec.get('patient_filters', []):
            condition = QueryCondition(
                field=spec['field'],
                operator=QueryOperator(spec['op']),
                value=spec['value']
            )
            patient_conditions.append(condition)
        
        # Parse required edges
        must_have_edges = []
        for edge_name in query_spec.get('required_relationships', []):
            try:
                edge_type = EdgeType(edge_name)
                must_have_edges.append(edge_type)
            except ValueError:
                logger.warning(f"Unknown edge type: {edge_name}")
        
        # Parse neighbor conditions
        neighbor_conditions = {}
        for edge_name, specs in query_spec.get('neighbor_filters', {}).items():
            try:
                edge_type = EdgeType(edge_name)
                conditions = [
                    QueryCondition(
                        field=spec['field'],
                        operator=QueryOperator(spec['op']),
                        value=spec['value']
                    )
                    for spec in specs
                ]
                neighbor_conditions[edge_type] = conditions
            except ValueError:
                logger.warning(f"Unknown edge type in neighbor filters: {edge_name}")
        
        return self.find_patients_with_conditions(
            patient_conditions=patient_conditions or None,
            must_have_edges=must_have_edges or None,
            neighbor_conditions=neighbor_conditions or None,
            logic=query_spec.get('logic', 'AND')
        )
    
    def find_patients_at_site(
        self,
        site_id: str,
        additional_conditions: List[QueryCondition] = None
    ) -> List[PatientQueryResult]:
        """Find all patients at a specific site"""
        conditions = [
            QueryCondition('site_id', QueryOperator.EQUALS, site_id)
        ]
        if additional_conditions:
            conditions.extend(additional_conditions)
        
        return self.find_patients_with_conditions(
            patient_conditions=conditions,
            logic="AND"
        )
    
    def find_clean_patients(self) -> List[PatientQueryResult]:
        """Find all patients with clean status"""
        return self.find_patients_with_conditions(
            patient_conditions=[
                QueryCondition('clean_status', QueryOperator.EQUALS, True)
            ],
            logic="AND"
        )
    
    def find_non_clean_patients(self) -> List[PatientQueryResult]:
        """Find all patients without clean status"""
        return self.find_patients_with_conditions(
            patient_conditions=[
                QueryCondition('clean_status', QueryOperator.EQUALS, False)
            ],
            logic="AND"
        )
    
    # ==================== Aggregation Queries ====================
    
    def aggregate_by_site(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate patient data by site"""
        patient_nodes = self.graph.get_nodes_by_type(NodeType.PATIENT)
        
        site_aggregates = {}
        for patient in patient_nodes:
            site_id = patient.attributes.get('site_id', 'Unknown')
            
            if site_id not in site_aggregates:
                site_aggregates[site_id] = {
                    'total_patients': 0,
                    'clean_patients': 0,
                    'total_open_queries': 0,
                    'total_missing_visits': 0,
                    'total_uncoded_terms': 0,
                    'total_reconciliation_issues': 0,
                    'patients_needing_attention': 0,
                    'average_dqi': 0.0,
                    'dqi_sum': 0.0
                }
            
            agg = site_aggregates[site_id]
            agg['total_patients'] += 1
            agg['clean_patients'] += 1 if patient.attributes.get('clean_status', False) else 0
            agg['total_open_queries'] += patient.attributes.get('open_queries', 0)
            agg['total_missing_visits'] += patient.attributes.get('missing_visits', 0)
            agg['total_uncoded_terms'] += patient.attributes.get('uncoded_terms', 0)
            agg['total_reconciliation_issues'] += patient.attributes.get('reconciliation_issues', 0)
            agg['dqi_sum'] += patient.attributes.get('data_quality_index', 100.0)
            
            # Check if patient needs attention (has multiple issues)
            issues = sum([
                1 if patient.attributes.get('missing_visits', 0) > 0 else 0,
                1 if patient.attributes.get('open_queries', 0) > 0 else 0,
                1 if patient.attributes.get('uncoded_terms', 0) > 0 else 0
            ])
            if issues >= 2:
                agg['patients_needing_attention'] += 1
        
        # Calculate averages
        for site_id, agg in site_aggregates.items():
            if agg['total_patients'] > 0:
                agg['average_dqi'] = round(agg['dqi_sum'] / agg['total_patients'], 1)
                agg['clean_rate'] = round(
                    (agg['clean_patients'] / agg['total_patients']) * 100, 1
                )
            del agg['dqi_sum']
        
        return site_aggregates
    
    def aggregate_by_country(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate patient data by country"""
        patient_nodes = self.graph.get_nodes_by_type(NodeType.PATIENT)
        
        country_aggregates = {}
        for patient in patient_nodes:
            country = patient.attributes.get('country', 'Unknown')
            
            if country not in country_aggregates:
                country_aggregates[country] = {
                    'total_patients': 0,
                    'clean_patients': 0,
                    'sites': set(),
                    'average_dqi': 0.0,
                    'dqi_sum': 0.0
                }
            
            agg = country_aggregates[country]
            agg['total_patients'] += 1
            agg['clean_patients'] += 1 if patient.attributes.get('clean_status', False) else 0
            agg['sites'].add(patient.attributes.get('site_id', 'Unknown'))
            agg['dqi_sum'] += patient.attributes.get('data_quality_index', 100.0)
        
        # Calculate averages and convert sets
        for country, agg in country_aggregates.items():
            if agg['total_patients'] > 0:
                agg['average_dqi'] = round(agg['dqi_sum'] / agg['total_patients'], 1)
                agg['clean_rate'] = round(
                    (agg['clean_patients'] / agg['total_patients']) * 100, 1
                )
            agg['total_sites'] = len(agg['sites'])
            del agg['sites']
            del agg['dqi_sum']
        
        return country_aggregates
    
    def get_issue_summary(self) -> Dict[str, Any]:
        """Get summary of all issues across the knowledge graph"""
        patient_nodes = self.graph.get_nodes_by_type(NodeType.PATIENT)
        
        summary = {
            'total_patients': len(patient_nodes),
            'clean_patients': 0,
            'patients_with_issues': 0,
            'issue_breakdown': {
                'missing_visits': 0,
                'open_queries': 0,
                'uncoded_terms': 0,
                'reconciliation_issues': 0,
                'protocol_deviations': 0
            },
            'patients_by_issue_count': {
                '0_issues': 0,
                '1_issue': 0,
                '2_issues': 0,
                '3_plus_issues': 0
            },
            'critical_patients': []
        }
        
        for patient in patient_nodes:
            attrs = patient.attributes
            
            if attrs.get('clean_status', False):
                summary['clean_patients'] += 1
                summary['patients_by_issue_count']['0_issues'] += 1
                continue
            
            summary['patients_with_issues'] += 1
            
            issue_count = 0
            if attrs.get('missing_visits', 0) > 0:
                summary['issue_breakdown']['missing_visits'] += 1
                issue_count += 1
            if attrs.get('open_queries', 0) > 0:
                summary['issue_breakdown']['open_queries'] += 1
                issue_count += 1
            if attrs.get('uncoded_terms', 0) > 0:
                summary['issue_breakdown']['uncoded_terms'] += 1
                issue_count += 1
            if attrs.get('reconciliation_issues', 0) > 0:
                summary['issue_breakdown']['reconciliation_issues'] += 1
                issue_count += 1
            if attrs.get('protocol_deviations', 0) > 0:
                summary['issue_breakdown']['protocol_deviations'] += 1
                issue_count += 1
            
            if issue_count == 1:
                summary['patients_by_issue_count']['1_issue'] += 1
            elif issue_count == 2:
                summary['patients_by_issue_count']['2_issues'] += 1
            else:
                summary['patients_by_issue_count']['3_plus_issues'] += 1
                # Track critical patients
                summary['critical_patients'].append({
                    'subject_id': attrs.get('subject_id'),
                    'site_id': attrs.get('site_id'),
                    'issue_count': issue_count
                })
        
        return summary
    
    # ==================== Risk Scoring ====================
    
    def _calculate_attention_risk_score(self, result: PatientQueryResult) -> float:
        """Calculate a risk score for prioritization"""
        score = 0.0
        attrs = result.patient_attributes
        
        # Missing visits (weighted by count)
        missing_visits = attrs.get('missing_visits', 0)
        score += min(missing_visits * 10, 30)  # Max 30 points
        
        # Open queries (weighted by count)
        open_queries = attrs.get('open_queries', 0)
        score += min(open_queries * 5, 25)  # Max 25 points
        
        # Uncoded terms
        uncoded_terms = attrs.get('uncoded_terms', 0)
        score += min(uncoded_terms * 5, 20)  # Max 20 points
        
        # Reconciliation issues (critical)
        recon_issues = attrs.get('reconciliation_issues', 0)
        score += min(recon_issues * 15, 30)  # Max 30 points
        
        # Protocol deviations
        pds = attrs.get('protocol_deviations', 0)
        score += min(pds * 10, 20)  # Max 20 points
        
        # Verification percentage (inverse)
        verification = attrs.get('verification_pct', 100)
        if verification < 75:
            score += (75 - verification) * 0.5  # Max ~37 points
        
        # Bonus for multiple related issues
        related_count = len(result.related_nodes)
        if related_count >= 3:
            score *= 1.2  # 20% bonus for 3+ issue types
        
        return min(score, 100)  # Cap at 100
    
    def prioritize_patients(
        self,
        patients: List[PatientQueryResult] = None
    ) -> List[PatientQueryResult]:
        """Prioritize patients by risk score"""
        if patients is None:
            patients = self.find_patients_with_conditions(
                patient_conditions=[
                    QueryCondition('clean_status', QueryOperator.EQUALS, False)
                ]
            )
        
        for patient in patients:
            patient.risk_score = self._calculate_attention_risk_score(patient)
        
        patients.sort(key=lambda x: x.risk_score, reverse=True)
        
        for i, patient in enumerate(patients):
            patient.priority_rank = i + 1
        
        return patients
    
    # ==================== Utility Methods ====================
    
    def get_patient_360_view(self, subject_id: str) -> Dict[str, Any]:
        """
        Get a complete 360-degree view of a patient
        
        This is the ultimate graph traversal - gathering all related
        information for a single patient from across the knowledge graph.
        """
        patient_node = self.graph.get_patient_node(subject_id)
        if not patient_node:
            return {'error': f'Patient {subject_id} not found'}
        
        neighbors = self.graph.get_patient_neighbors(subject_id)
        
        view = {
            'patient': patient_node.attributes,
            'visits': [],
            'queries': [],
            'sae_events': [],
            'coding_issues': [],
            'statistics': self.graph.get_patient_statistics(subject_id)
        }
        
        # Process visits
        for visit_node in neighbors.get(EdgeType.HAS_VISIT, []):
            view['visits'].append(visit_node.attributes)
        
        # Process queries
        for query_node in neighbors.get(EdgeType.HAS_QUERY, []):
            view['queries'].append(query_node.attributes)
        
        # Process SAE events
        for sae_node in neighbors.get(EdgeType.HAS_ADVERSE_EVENT, []):
            view['sae_events'].append(sae_node.attributes)
        
        # Process coding issues
        for coding_node in neighbors.get(EdgeType.HAS_CODING_ISSUE, []):
            view['coding_issues'].append(coding_node.attributes)
        
        return view
    
    def export_query_results_to_dataframe(
        self,
        results: List[PatientQueryResult]
    ):
        """Export query results to a pandas DataFrame for analysis"""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not available, returning dict list instead")
            return [r.to_dict() for r in results]
        
        rows = []
        for result in results:
            row = {
                'subject_id': result.subject_id,
                'site_id': result.patient_attributes.get('site_id'),
                'country': result.patient_attributes.get('country'),
                'clean_status': result.patient_attributes.get('clean_status'),
                'missing_visits': result.patient_attributes.get('missing_visits', 0),
                'open_queries': result.patient_attributes.get('open_queries', 0),
                'uncoded_terms': result.patient_attributes.get('uncoded_terms', 0),
                'reconciliation_issues': result.patient_attributes.get('reconciliation_issues', 0),
                'dqi': result.patient_attributes.get('data_quality_index', 100),
                'risk_score': result.risk_score,
                'priority_rank': result.priority_rank,
                'matched_conditions': ', '.join(result.matched_conditions)
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def find_patients_with_multiple_issues(self, min_issues: int = 2) -> List[PatientQueryResult]:
        """
        Find patients with multiple types of issues (diverse problem patterns)
        """
        results = []
        
        for patient_node in self.graph.get_nodes_by_type(NodeType.PATIENT):
            attrs = patient_node.attributes
            issues = []
            
            if attrs.get('missing_visits', 0) > 0:
                issues.append('missing_visits')
            if attrs.get('open_queries', 0) > 0:
                issues.append('open_queries')
            if attrs.get('uncoded_terms', 0) > 0:
                issues.append('uncoded_terms')
            if attrs.get('protocol_deviations', 0) > 0:
                issues.append('protocol_deviations')
            if attrs.get('reconciliation_issues', 0) > 0:
                issues.append('reconciliation_issues')
            
            if len(issues) >= min_issues:
                result = PatientQueryResult(
                    subject_id=attrs.get('subject_id', ''),
                    node_id=patient_node.node_id,
                    patient_attributes=attrs,
                    matched_conditions=issues,
                    risk_score=len(issues) * 2,  # Risk based on issue diversity
                    related_nodes={}
                )
                result.issues = issues
                results.append(result)
        
        # Sort by issue count (most problematic first)
        results.sort(key=lambda x: len(x.issues), reverse=True)
        return results
    
    def find_sites_with_high_risk_clusters(self) -> List[Dict]:
        """
        Find sites with clusters of high-risk patients
        """
        site_risks = {}
        
        for patient_node in self.graph.get_nodes_by_type(NodeType.PATIENT):
            attrs = patient_node.attributes
            site_id = attrs.get('site_id')
            if not site_id:
                continue
            
            # Calculate patient risk
            risk_score = (
                attrs.get('missing_visits', 0) * 2 +
                attrs.get('open_queries', 0) * 1.5 +
                attrs.get('uncoded_terms', 0) * 1 +
                attrs.get('protocol_deviations', 0) * 3
            )
            
            if site_id not in site_risks:
                site_risks[site_id] = {
                    'site_id': site_id,
                    'total_patients': 0,
                    'high_risk_patients': 0,
                    'total_risk_score': 0,
                    'avg_risk_score': 0
                }
            
            site_risks[site_id]['total_patients'] += 1
            site_risks[site_id]['total_risk_score'] += risk_score
            if risk_score > 5:
                site_risks[site_id]['high_risk_patients'] += 1
        
        # Calculate averages and filter high-risk sites
        high_risk_sites = []
        for site_data in site_risks.values():
            if site_data['total_patients'] > 0:
                site_data['avg_risk_score'] = site_data['total_risk_score'] / site_data['total_patients']
                
                # Site is high-risk if >30% patients are high-risk or avg risk > 3
                risk_ratio = site_data['high_risk_patients'] / site_data['total_patients']
                if risk_ratio > 0.3 or site_data['avg_risk_score'] > 3:
                    high_risk_sites.append(site_data)
        
        # Sort by risk ratio
        high_risk_sites.sort(key=lambda x: x['high_risk_patients'] / x['total_patients'], reverse=True)
        return high_risk_sites
    
    def find_temporal_issue_patterns(self) -> List[Dict]:
        """
        Find temporal correlations between visits and issues
        """
        patterns = []
        
        for patient_node in self.graph.get_nodes_by_type(NodeType.PATIENT):
            patient_attrs = patient_node.attributes
            subject_id = patient_attrs.get('subject_id')
            
            # Get connected visits
            visit_nodes = []
            for neighbor_id in self.graph.graph.neighbors(patient_node.node_id):
                neighbor_node = self.graph.node_index.get(neighbor_id)
                if neighbor_node and neighbor_node.node_type == NodeType.EVENT:
                    visit_nodes.append(neighbor_node)
            
            # Look for overdue visits with open queries
            overdue_visits = [v for v in visit_nodes if v.attributes.get('days_outstanding', 0) > 30]
            has_open_queries = patient_attrs.get('open_queries', 0) > 0
            
            if overdue_visits and has_open_queries:
                pattern = {
                    'subject_id': subject_id,
                    'site_id': patient_attrs.get('site_id'),
                    'overdue_visits': len(overdue_visits),
                    'max_days_overdue': max((v.attributes.get('days_outstanding', 0) for v in overdue_visits), default=0),
                    'open_queries': patient_attrs.get('open_queries', 0),
                    'pattern_type': 'overdue_visit_with_queries'
                }
                patterns.append(pattern)
        
        # Sort by severity
        patterns.sort(key=lambda x: x['max_days_overdue'] * x['open_queries'], reverse=True)
        return patterns


class FederatedGraphQueryEngine:
    """
    Federated query engine for cross-study graph analytics
    
    Enables queries across multiple clinical studies while maintaining
    study-level isolation and performance.
    """
    
    def __init__(self, study_graphs: Dict[str, ClinicalKnowledgeGraph], 
                 study_query_engines: Dict[str, GraphQueryEngine]):
        self.study_graphs = study_graphs
        self.study_query_engines = study_query_engines
        self.study_ids = list(study_graphs.keys())
    
    def query_cross_study_patient_patterns(self, pattern_criteria: Dict) -> Dict[str, Any]:
        """
        Query for patient patterns across all studies
        
        Args:
            pattern_criteria: Dictionary with query criteria like:
                - min_open_queries: int
                - min_uncoded_terms: int
                - min_missing_visits: int
                - risk_threshold: float
        """
        results = {
            'total_studies': len(self.study_ids),
            'study_results': {},
            'cross_study_aggregates': {},
            'patterns_identified': []
        }
        
        # Query each study
        for study_id, query_engine in self.study_query_engines.items():
            try:
                study_results = self._query_single_study_patterns(query_engine, pattern_criteria)
                results['study_results'][study_id] = study_results
            except Exception as e:
                logger.error(f"Error querying study {study_id}: {e}")
                results['study_results'][study_id] = {'error': str(e)}
        
        # Calculate cross-study aggregates
        results['cross_study_aggregates'] = self._calculate_cross_study_aggregates(results['study_results'])
        
        # Identify cross-study patterns
        results['patterns_identified'] = self._identify_cross_study_patterns(results['study_results'])
        
        return results
    
    def _query_single_study_patterns(self, query_engine: GraphQueryEngine, criteria: Dict) -> Dict:
        """Query patterns within a single study"""
        results = {
            'high_risk_patients': [],
            'multi_issue_patients': [],
            'site_risk_clusters': [],
            'total_patients': 0,
            'at_risk_patients': 0
        }
        
        # Get all patients
        all_patients = query_engine.graph.get_nodes_by_type(NodeType.PATIENT)
        results['total_patients'] = len(all_patients)
        
        # Find high-risk patients
        for patient_node in all_patients:
            attrs = patient_node.attributes
            
            risk_score = (
                attrs.get('missing_visits', 0) * 2 +
                attrs.get('open_queries', 0) * 1.5 +
                attrs.get('uncoded_terms', 0) * 1 +
                attrs.get('protocol_deviations', 0) * 3
            )
            
            if risk_score >= criteria.get('risk_threshold', 5):
                results['high_risk_patients'].append({
                    'subject_id': attrs.get('subject_id'),
                    'site_id': attrs.get('site_id'),
                    'country': attrs.get('country'),
                    'risk_score': risk_score,
                    'issues': self._count_patient_issues(attrs)
                })
                results['at_risk_patients'] += 1
        
        # Find multi-issue patients
        multi_issue = query_engine.find_patients_with_multiple_issues(
            min_issues=criteria.get('min_issue_types', 2)
        )
        results['multi_issue_patients'] = [{
            'subject_id': p.subject_id,
            'issue_count': len(p.issues),
            'issues': p.issues
        } for p in multi_issue]
        
        # Find site risk clusters
        site_clusters = query_engine.find_sites_with_high_risk_clusters()
        results['site_risk_clusters'] = site_clusters
        
        return results
    
    def _calculate_cross_study_aggregates(self, study_results: Dict) -> Dict:
        """Calculate aggregates across all studies"""
        aggregates = {
            'total_patients': 0,
            'total_at_risk_patients': 0,
            'total_high_risk_patients': 0,
            'total_multi_issue_patients': 0,
            'total_site_clusters': 0,
            'studies_with_issues': 0,
            'avg_risk_per_study': 0,
            'country_risk_distribution': {},
            'site_risk_distribution': {}
        }
        
        study_risks = []
        
        for study_id, results in study_results.items():
            if 'error' in results:
                continue
                
            aggregates['total_patients'] += results.get('total_patients', 0)
            aggregates['total_at_risk_patients'] += results.get('at_risk_patients', 0)
            aggregates['total_high_risk_patients'] += len(results.get('high_risk_patients', []))
            aggregates['total_multi_issue_patients'] += len(results.get('multi_issue_patients', []))
            aggregates['total_site_clusters'] += len(results.get('site_risk_clusters', []))
            
            if results.get('at_risk_patients', 0) > 0:
                aggregates['studies_with_issues'] += 1
            
            # Country and site distributions
            for patient in results.get('high_risk_patients', []):
                country = patient.get('country', 'Unknown')
                site = patient.get('site_id', 'Unknown')
                
                aggregates['country_risk_distribution'][country] = \
                    aggregates['country_risk_distribution'].get(country, 0) + 1
                aggregates['site_risk_distribution'][site] = \
                    aggregates['site_risk_distribution'].get(site, 0) + 1
            
            # Study-level risk
            if results.get('total_patients', 0) > 0:
                study_risk = results.get('at_risk_patients', 0) / results.get('total_patients', 0)
                study_risks.append(study_risk)
        
        if study_risks:
            aggregates['avg_risk_per_study'] = sum(study_risks) / len(study_risks)
        
        return aggregates
    
    def _identify_cross_study_patterns(self, study_results: Dict) -> List[Dict]:
        """Identify patterns that span multiple studies"""
        patterns = []
        
        # Pattern 1: Consistent high-risk countries across studies
        country_consistency = {}
        for study_id, results in study_results.items():
            if 'error' in results:
                continue
            for patient in results.get('high_risk_patients', []):
                country = patient.get('country', 'Unknown')
                if country not in country_consistency:
                    country_consistency[country] = []
                country_consistency[country].append(study_id)
        
        for country, studies in country_consistency.items():
            if len(set(studies)) > 1:  # Appears in multiple studies
                patterns.append({
                    'pattern_type': 'cross_study_country_risk',
                    'country': country,
                    'studies_affected': list(set(studies)),
                    'severity': 'High' if len(set(studies)) > 2 else 'Medium',
                    'description': f'Country {country} shows high-risk patterns across {len(set(studies))} studies'
                })
        
        # Pattern 2: Study performance comparison
        study_performance = {}
        for study_id, results in study_results.items():
            if 'error' in results:
                continue
            total = results.get('total_patients', 0)
            at_risk = results.get('at_risk_patients', 0)
            if total > 0:
                risk_ratio = at_risk / total
                study_performance[study_id] = risk_ratio
        
        if len(study_performance) > 1:
            avg_risk = sum(study_performance.values()) / len(study_performance)
            high_performers = [s for s, r in study_performance.items() if r < avg_risk * 0.7]
            low_performers = [s for s, r in study_performance.items() if r > avg_risk * 1.3]
            
            if high_performers:
                patterns.append({
                    'pattern_type': 'high_performing_studies',
                    'studies': high_performers,
                    'avg_risk_ratio': avg_risk,
                    'description': f'Studies {high_performers} show significantly better performance than average'
                })
            
            if low_performers:
                patterns.append({
                    'pattern_type': 'low_performing_studies',
                    'studies': low_performers,
                    'avg_risk_ratio': avg_risk,
                    'description': f'Studies {low_performers} show significantly worse performance than average'
                })
        
        return patterns
    
    def _count_patient_issues(self, attrs: Dict) -> Dict[str, int]:
        """Count different types of issues for a patient"""
        return {
            'missing_visits': attrs.get('missing_visits', 0),
            'open_queries': attrs.get('open_queries', 0),
            'uncoded_terms': attrs.get('uncoded_terms', 0),
            'protocol_deviations': attrs.get('protocol_deviations', 0),
            'reconciliation_issues': attrs.get('reconciliation_issues', 0)
        }
