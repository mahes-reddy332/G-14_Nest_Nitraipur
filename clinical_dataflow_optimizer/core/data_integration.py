"""
Data Integration Layer: From Tables to Graphs
Unified patient-centric data model for the Neural Clinical Data Mesh

Instead of treating the nine files as separate tables to be joined via VLOOKUPs or SQL JOINS,
this module ingests them into a graph database structure where Subject ID acts as the 
central anchor node.

Key Capabilities:
- Semantic network transformation from flat CSV structures
- Multi-hop query execution (impossible in traditional SQL efficiently)
- Graph-based pattern detection and anomaly identification
- Patient-centric data model with rich relationships

Architecture:
- Node: Patient (central anchor) - from CPID_EDC_Metrics
- Node: Site - aggregated from patient data  
- Node: Event/Visit - from Visit Projection Tracker
- Node: Discrepancy - from CPID_EDC_Metrics (queries)
- Node: SAE - from SAE Dashboard
- Node: CodingTerm - from GlobalCodingReport_MedDRA/WHODRA

- Edge: HAS_VISIT (Patient -> Event)
- Edge: HAS_ADVERSE_EVENT (Patient -> SAE) with Review Status, Action Status
- Edge: HAS_CODING_ISSUE (Patient -> CodingTerm) with Verbatim Term, Coding Status
- Edge: HAS_QUERY (Patient -> Discrepancy)
- Edge: ENROLLED_AT (Patient -> Site)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.knowledge_graph import (
    ClinicalKnowledgeGraph, NodeType, EdgeType,
    PatientNode, SiteNode, EventNode, DiscrepancyNode, 
    SAENode, CodingTermNode, GraphEdge, QueryStatus, CodingStatus
)
from graph.graph_builder import ClinicalGraphBuilder
from graph.graph_queries import (
    GraphQueryEngine, QueryCondition, QueryOperator, PatientQueryResult
)
from graph.graph_analytics import GraphAnalytics, PatientRiskProfile, SiteRiskProfile
from core.data_ingestion import ClinicalDataIngester
from models.data_models import DigitalPatientTwin, BlockingItem, RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class MultiHopQueryResult:
    """Result of a multi-hop graph query"""
    query_name: str
    query_description: str
    patient_count: int
    patients: List[PatientQueryResult]
    execution_time_ms: float
    sql_equivalent_complexity: str  # Description of SQL complexity
    conditions_matched: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'query_name': self.query_name,
            'description': self.query_description,
            'patient_count': self.patient_count,
            'patients': [p.to_dict() for p in self.patients[:10]],  # Limit output
            'execution_time_ms': round(self.execution_time_ms, 2),
            'sql_equivalent': self.sql_equivalent_complexity,
            'conditions_matched': self.conditions_matched
        }


@dataclass 
class GraphStatistics:
    """Statistics about the knowledge graph"""
    study_id: str
    total_nodes: int = 0
    total_edges: int = 0
    node_counts: Dict[str, int] = field(default_factory=dict)
    edge_counts: Dict[str, int] = field(default_factory=dict)
    average_patient_connections: float = 0.0
    max_patient_connections: int = 0
    connected_components: int = 0
    graph_density: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'study_id': self.study_id,
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'node_breakdown': self.node_counts,
            'edge_breakdown': self.edge_counts,
            'avg_patient_connections': round(self.average_patient_connections, 2),
            'max_patient_connections': self.max_patient_connections,
            'connected_components': self.connected_components,
            'graph_density': round(self.graph_density, 6)
        }


class ClinicalDataMesh:
    """
    The Clinical Data Mesh - Main interface for graph-based clinical data analysis
    
    This class provides the complete data integration layer that transforms
    flat CSV/Excel files into a semantic graph network, enabling complex
    multi-hop queries that relational databases cannot efficiently perform.
    
    Example multi-hop query:
    "Show me all patients who have:
     - A Missing Visit (from Visit Tracker) AND
     - An Open Safety Query (from CPID) AND
     - An Uncoded Concomitant Medication (from WHODRA)"
     
    In SQL: Requires joining 3+ tables with mismatched keys
    In Graph: Simple traversal of patient node's neighbors
    """
    
    def __init__(self, study_id: str = ""):
        self.study_id = study_id
        self.graph: Optional[ClinicalKnowledgeGraph] = None
        self.builder: Optional[ClinicalGraphBuilder] = None
        self.query_engine: Optional[GraphQueryEngine] = None
        self.analytics: Optional[GraphAnalytics] = None
        self._is_built = False
        self._study_data: Dict[str, pd.DataFrame] = {}  # Store loaded data sources
    
    @property
    def data_sources(self) -> Dict[str, pd.DataFrame]:
        """Access the loaded study data sources (SAE, EDRR, etc.)"""
        return self._study_data
    
    def build_from_study_folder(
        self,
        study_folder: Path,
        study_id: str = None
    ) -> 'ClinicalDataMesh':
        """
        Build the knowledge graph from a study folder containing all data files
        
        Args:
            study_folder: Path to study folder with Excel files
            study_id: Optional study identifier
            
        Returns:
            Self for method chaining
        """
        import time
        start_time = time.time()
        
        if study_id:
            self.study_id = study_id
        elif not self.study_id:
            self.study_id = study_folder.name.split('_')[0].replace(' ', '_')
        
        logger.info(f"Building Clinical Data Mesh for {self.study_id}")
        logger.info(f"Source folder: {study_folder}")
        
        # Initialize components
        self.builder = ClinicalGraphBuilder(study_id=self.study_id)
        
        # Create data ingester to load files
        ingester = ClinicalDataIngester(study_folder.parent)
        
        # Load all available data files
        study_data = self._load_study_files(study_folder, ingester)
        self._study_data = study_data  # Store for agent access
        
        # Build the knowledge graph
        self.graph = self.builder.build_from_study_data(
            study_data=study_data,
            study_id=self.study_id
        )
        
        # Initialize query engine and analytics
        self.query_engine = GraphQueryEngine(self.graph)
        self.analytics = GraphAnalytics(self.graph)
        
        self._is_built = True
        build_time = (time.time() - start_time) * 1000
        
        logger.info(f"Data Mesh built in {build_time:.1f}ms")
        logger.info(f"Graph statistics: {self.get_statistics().to_dict()}")
        
        return self
    
    def _load_study_files(
        self, 
        study_folder: Path, 
        ingester: ClinicalDataIngester
    ) -> Dict[str, pd.DataFrame]:
        """Load all study files from the folder"""
        study_data = {}
        
        # File patterns to look for
        file_patterns = {
            'cpid_metrics': '*CPID_EDC_Metrics*.xlsx',
            'visit_tracker': '*Visit*Projection*.xlsx',
            'sae_dashboard': '*eSAE*.xlsx',
            'meddra_coding': '*MedDRA*.xlsx',
            'whodra_coding': '*WHODD*.xlsx',
            'compiled_edrr': '*EDRR*.xlsx',
            'inactivated_forms': '*Inactivated*.xlsx',
            'missing_pages': '*Missing_Pages*.xlsx'
        }
        
        for data_key, pattern in file_patterns.items():
            files = list(study_folder.glob(pattern))
            if files:
                file_path = files[0]
                logger.info(f"Loading {data_key}: {file_path.name}")
                
                try:
                    # Use appropriate loader based on file type
                    if data_key == 'cpid_metrics':
                        df = ingester.load_cpid_metrics(file_path)
                    elif data_key == 'visit_tracker':
                        df = ingester.load_visit_tracker(file_path)
                    elif data_key == 'sae_dashboard':
                        df = ingester.load_sae_dashboard(file_path)
                    elif data_key in ['meddra_coding', 'whodra_coding']:
                        coding_type = 'MedDRA' if 'meddra' in data_key else 'WHODRA'
                        df = ingester.load_coding_report(file_path, coding_type)
                    elif data_key == 'compiled_edrr':
                        df = ingester.load_compiled_edrr(file_path)
                    elif data_key == 'inactivated_forms':
                        df = ingester.load_inactivated_forms(file_path)
                    elif data_key == 'missing_pages':
                        df = ingester.load_missing_pages(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    
                    if df is not None and len(df) > 0:
                        study_data[data_key] = df
                        logger.info(f"  Loaded {len(df)} rows")
                except Exception as e:
                    logger.error(f"Error loading {data_key}: {e}")
        
        return study_data
    
    def get_statistics(self) -> GraphStatistics:
        """Get comprehensive statistics about the knowledge graph"""
        if not self._is_built:
            return GraphStatistics(study_id=self.study_id)
        
        stats = GraphStatistics(study_id=self.study_id)
        
        # Basic counts
        stats.total_nodes = self.graph.graph.number_of_nodes()
        stats.total_edges = self.graph.graph.number_of_edges()
        
        # Node counts by type
        for node_type in NodeType:
            count = len(self.graph._node_type_cache.get(node_type, set()))
            if count > 0:
                stats.node_counts[node_type.value] = count
        
        # Edge counts by type
        edge_type_counts = {}
        for u, v, data in self.graph.graph.edges(data=True):
            edge_type = data.get('edge_type', 'Unknown')
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        stats.edge_counts = edge_type_counts
        
        # Patient connection statistics
        patient_nodes = self.graph.get_nodes_by_type(NodeType.PATIENT)
        if patient_nodes:
            connection_counts = []
            for patient in patient_nodes:
                neighbors = self.graph.get_patient_neighbors(patient.attributes.get('subject_id', ''))
                total_connections = sum(len(v) for v in neighbors.values())
                connection_counts.append(total_connections)
            
            if connection_counts:
                stats.average_patient_connections = np.mean(connection_counts)
                stats.max_patient_connections = max(connection_counts)
        
        # Graph density
        if stats.total_nodes > 1:
            stats.graph_density = self.graph.graph.number_of_edges() / (
                stats.total_nodes * (stats.total_nodes - 1)
            )
        
        return stats
    
    # ==================== Multi-Hop Query Methods ====================
    
    def query_patients_needing_attention(self) -> MultiHopQueryResult:
        """
        THE FLAGSHIP MULTI-HOP QUERY
        
        Find all patients who have:
        - A Missing Visit (from Visit Tracker) AND
        - An Open Safety Query (from CPID) AND
        - An Uncoded Term (from GlobalCodingReport)
        
        This demonstrates the power of graph traversal over SQL.
        
        SQL Equivalent would require:
        ```sql
        SELECT p.* 
        FROM patients p
        INNER JOIN visit_tracker v ON p.subject_id = v.subject_id AND v.is_missing = TRUE
        INNER JOIN cpid_queries q ON p.subject_id = q.subject_id AND q.status = 'Open' AND q.type = 'Safety'
        INNER JOIN coding_report c ON p.subject_id = c.subject_id AND c.status = 'Uncoded'
        WHERE p.study_id = ?
        ```
        
        In Graph: Simple neighbor traversal in O(n) where n = number of patient connections
        """
        import time
        start = time.time()
        
        results = self.query_engine.find_patients_needing_attention()
        
        execution_time = (time.time() - start) * 1000
        
        return MultiHopQueryResult(
            query_name="Patients Needing Immediate Attention",
            query_description=(
                "Patients with Missing Visit AND Open Query AND Uncoded Term - "
                "the three-way intersection that traditional SQL cannot efficiently compute"
            ),
            patient_count=len(results),
            patients=results,
            execution_time_ms=execution_time,
            sql_equivalent_complexity=(
                "3-table INNER JOIN with index scans on subject_id; "
                "O(n*m*k) worst case where n,m,k are table sizes"
            ),
            conditions_matched=[
                "missing_visits > 0",
                "open_queries > 0 (Safety)",
                "uncoded_terms > 0"
            ]
        )
    
    def query_patients_with_visit_and_query_issues(
        self,
        min_days_outstanding: int = 30,
        min_open_queries: int = 1
    ) -> MultiHopQueryResult:
        """
        Find patients with overdue visits AND open queries
        
        Graph query traverses:
        Patient -> HAS_VISIT -> Event (days_outstanding > threshold)
        Patient -> HAS_QUERY -> Discrepancy (status = Open)
        """
        import time
        start = time.time()
        
        results = self.query_engine.find_patients_with_conditions(
            patient_conditions=[
                QueryCondition('missing_visits', QueryOperator.GREATER_THAN, 0),
                QueryCondition('open_queries', QueryOperator.GREATER_EQUAL, min_open_queries)
            ],
            neighbor_conditions={
                EdgeType.HAS_VISIT: [
                    QueryCondition('days_outstanding', QueryOperator.GREATER_EQUAL, min_days_outstanding)
                ]
            },
            logic="AND"
        )
        
        execution_time = (time.time() - start) * 1000
        
        return MultiHopQueryResult(
            query_name="Visit and Query Issues",
            query_description=f"Patients with visits overdue >{min_days_outstanding} days AND >{min_open_queries} open queries",
            patient_count=len(results),
            patients=results,
            execution_time_ms=execution_time,
            sql_equivalent_complexity="2-table JOIN with conditional filters",
            conditions_matched=[
                f"days_outstanding >= {min_days_outstanding}",
                f"open_queries >= {min_open_queries}"
            ]
        )
    
    def query_patients_with_sae_and_coding_issues(
        self,
        review_status: str = "Pending"
    ) -> MultiHopQueryResult:
        """
        Find patients with SAE review pending AND uncoded terms
        
        This is critical for safety reporting - identifies patients
        where both safety events and medical coding are incomplete.
        
        Graph query traverses:
        Patient -> HAS_ADVERSE_EVENT -> SAE (review_status = Pending)
        Patient -> HAS_CODING_ISSUE -> CodingTerm (status = Uncoded)
        """
        import time
        start = time.time()
        
        results = self.query_engine.find_patients_with_conditions(
            patient_conditions=[
                QueryCondition('uncoded_terms', QueryOperator.GREATER_THAN, 0)
            ],
            must_have_edges=[EdgeType.HAS_ADVERSE_EVENT, EdgeType.HAS_CODING_ISSUE],
            neighbor_conditions={
                EdgeType.HAS_ADVERSE_EVENT: [
                    QueryCondition('review_status', QueryOperator.CONTAINS, review_status)
                ],
                EdgeType.HAS_CODING_ISSUE: [
                    QueryCondition('coding_status', QueryOperator.EQUALS, 'UnCoded')
                ]
            },
            logic="AND"
        )
        
        execution_time = (time.time() - start) * 1000
        
        return MultiHopQueryResult(
            query_name="SAE and Coding Issues",
            query_description=f"Patients with SAE review '{review_status}' AND uncoded medical terms",
            patient_count=len(results),
            patients=results,
            execution_time_ms=execution_time,
            sql_equivalent_complexity="3-table JOIN (patients, sae_dashboard, coding_report)",
            conditions_matched=[
                f"sae.review_status contains '{review_status}'",
                "coding.status = 'UnCoded'"
            ]
        )
    
    def query_patients_by_multi_criteria(
        self,
        has_missing_visit: bool = False,
        has_open_query: bool = False,
        has_uncoded_term: bool = False,
        has_sae: bool = False,
        has_reconciliation_issue: bool = False,
        logic: str = "AND"
    ) -> MultiHopQueryResult:
        """
        Flexible multi-criteria query builder
        
        Allows any combination of conditions to be searched,
        demonstrating the flexibility of graph-based queries.
        
        Args:
            has_missing_visit: Include patients with missing visits
            has_open_query: Include patients with open queries
            has_uncoded_term: Include patients with uncoded terms
            has_sae: Include patients with SAE records
            has_reconciliation_issue: Include patients with EDRR issues
            logic: "AND" (all conditions) or "OR" (any condition)
        """
        import time
        start = time.time()
        
        patient_conditions = []
        must_have_edges = []
        conditions_desc = []
        
        if has_missing_visit:
            patient_conditions.append(
                QueryCondition('missing_visits', QueryOperator.GREATER_THAN, 0)
            )
            must_have_edges.append(EdgeType.HAS_VISIT)
            conditions_desc.append("Missing Visit")
        
        if has_open_query:
            patient_conditions.append(
                QueryCondition('open_queries', QueryOperator.GREATER_THAN, 0)
            )
            must_have_edges.append(EdgeType.HAS_QUERY)
            conditions_desc.append("Open Query")
        
        if has_uncoded_term:
            patient_conditions.append(
                QueryCondition('uncoded_terms', QueryOperator.GREATER_THAN, 0)
            )
            must_have_edges.append(EdgeType.HAS_CODING_ISSUE)
            conditions_desc.append("Uncoded Term")
        
        if has_sae:
            must_have_edges.append(EdgeType.HAS_ADVERSE_EVENT)
            conditions_desc.append("SAE Record")
        
        if has_reconciliation_issue:
            patient_conditions.append(
                QueryCondition('reconciliation_issues', QueryOperator.GREATER_THAN, 0)
            )
            conditions_desc.append("Reconciliation Issue")
        
        results = self.query_engine.find_patients_with_conditions(
            patient_conditions=patient_conditions or None,
            must_have_edges=must_have_edges or None,
            logic=logic
        )
        
        execution_time = (time.time() - start) * 1000
        
        return MultiHopQueryResult(
            query_name=f"Multi-Criteria Query ({logic})",
            query_description=f"Patients matching: {f' {logic} '.join(conditions_desc)}",
            patient_count=len(results),
            patients=results,
            execution_time_ms=execution_time,
            sql_equivalent_complexity=f"{len(conditions_desc)}-way JOIN with {logic} logic",
            conditions_matched=conditions_desc
        )
    
    def execute_cypher_like_query(self, query_spec: Dict) -> MultiHopQueryResult:
        """
        Execute a Cypher-like query specification
        
        This mirrors the query language used by Neo4j, allowing
        external systems to send query specs that get translated
        to graph traversals.
        
        Query spec format:
        {
            "patient_filters": [
                {"field": "missing_visits", "op": "gt", "value": 0}
            ],
            "required_relationships": ["HAS_VISIT", "HAS_QUERY"],
            "neighbor_filters": {
                "HAS_QUERY": [{"field": "status", "op": "eq", "value": "Open"}]
            },
            "logic": "AND"
        }
        """
        import time
        start = time.time()
        
        results = self.query_engine.execute_mesh_query(query_spec)
        
        execution_time = (time.time() - start) * 1000
        
        return MultiHopQueryResult(
            query_name="Custom Cypher-like Query",
            query_description=json.dumps(query_spec, indent=2)[:200],
            patient_count=len(results),
            patients=results,
            execution_time_ms=execution_time,
            sql_equivalent_complexity="Dynamic query - complexity varies",
            conditions_matched=list(query_spec.get('required_relationships', []))
        )
    
    # ==================== Digital Patient Twin Methods ====================
    
    def get_digital_twin(self, subject_id: str) -> Optional[DigitalPatientTwin]:
        """
        Get the Digital Patient Twin for a specific patient
        
        The Digital Twin is the unified, machine-readable representation
        that serves as single source of truth for UI and AI agents.
        
        Args:
            subject_id: The patient's subject identifier
            
        Returns:
            DigitalPatientTwin object or None if not found
        """
        if not self._is_built:
            return None
        
        from core.digital_twin import DigitalTwinFactory
        factory = DigitalTwinFactory(self.graph)
        return factory.create_twin(subject_id)
    
    def get_all_digital_twins(self) -> List[DigitalPatientTwin]:
        """
        Get Digital Patient Twins for all patients
        
        Returns:
            List of DigitalPatientTwin objects
        """
        if not self._is_built:
            return []
        
        from core.digital_twin import DigitalTwinFactory
        factory = DigitalTwinFactory(self.graph)
        return factory.create_all_twins()
    
    def get_ai_readable_twin(self, subject_id: str) -> Optional[Dict]:
        """
        Get a simplified, AI-agent-readable twin representation
        
        This format is optimized for machine consumption by AI agents:
        {
          "subject_id": "101-001",
          "status": "Ongoing",
          "clean_status": false,
          "blocking_items": [...],
          "risk_metrics": {
            "query_aging_index": 0.8,
            "protocol_deviation_count": 2,
            "manipulation_risk_score": "High"
          }
        }
        
        Args:
            subject_id: The patient's subject identifier
            
        Returns:
            Dictionary in AI-readable format
        """
        if not self._is_built:
            return None
        
        from core.digital_twin import DigitalTwinFactory
        factory = DigitalTwinFactory(self.graph)
        return factory.get_ai_readable_twin(subject_id)
    
    def get_all_ai_readable_twins(self) -> List[Dict]:
        """Get AI-readable twins for all patients"""
        if not self._is_built:
            return []
        
        from core.digital_twin import DigitalTwinFactory
        factory = DigitalTwinFactory(self.graph)
        return factory.get_all_ai_readable_twins()
    
    def export_digital_twins_json(self, output_path: Path) -> int:
        """
        Export all Digital Patient Twins to a JSON file
        
        Args:
            output_path: Path to output JSON file
            
        Returns:
            Number of twins exported
        """
        if not self._is_built:
            raise ValueError("Graph not built yet")
        
        from core.digital_twin import DigitalTwinFactory
        factory = DigitalTwinFactory(self.graph)
        twins = factory.create_all_twins()
        factory.export_twins_json(twins, output_path)
        
        logger.info(f"Exported {len(twins)} Digital Twins to {output_path}")
        return len(twins)
    
    # ==================== Analytics Methods ====================
    
    def get_patient_risk_profile(self, subject_id: str) -> Optional[PatientRiskProfile]:
        """Get comprehensive risk profile for a patient"""
        if not self._is_built:
            return None
        return self.analytics.analyze_patient_risk(subject_id)
    
    def get_site_risk_profiles(self) -> List[SiteRiskProfile]:
        """Get risk profiles for all sites"""
        if not self._is_built:
            return []
        return self.analytics.analyze_all_sites()
    
    def get_high_risk_patients(self, risk_threshold: float = 50.0) -> List[PatientRiskProfile]:
        """Get all patients above a risk threshold"""
        if not self._is_built:
            return []
        return self.analytics.get_high_risk_patients(risk_threshold)
    
    # ==================== Export Methods ====================
    
    def export_graph_json(self, output_path: Path) -> None:
        """Export the knowledge graph to JSON format"""
        if not self._is_built:
            raise ValueError("Graph not built yet")
        
        export_data = {
            'study_id': self.study_id,
            'statistics': self.get_statistics().to_dict(),
            'nodes': [],
            'edges': []
        }
        
        # Export nodes
        for node_id, node in self.graph.node_index.items():
            export_data['nodes'].append(node.to_dict())
        
        # Export edges  
        for u, v, data in self.graph.graph.edges(data=True):
            export_data['edges'].append({
                'source': u,
                'target': v,
                **data
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Graph exported to {output_path}")
    
    def export_patient_network_csv(self, output_path: Path) -> None:
        """Export patient-centric network view to CSV"""
        if not self._is_built:
            raise ValueError("Graph not built yet")
        
        rows = []
        patients = self.graph.get_nodes_by_type(NodeType.PATIENT)
        
        for patient in patients:
            subject_id = patient.attributes.get('subject_id', '')
            neighbors = self.graph.get_patient_neighbors(subject_id)
            
            row = {
                'subject_id': subject_id,
                'site_id': patient.attributes.get('site_id', ''),
                'status': patient.attributes.get('status', ''),
                'clean_status': patient.attributes.get('clean_status', False),
                'missing_visits': patient.attributes.get('missing_visits', 0),
                'missing_pages': patient.attributes.get('missing_pages', 0),
                'open_queries': patient.attributes.get('open_queries', 0),
                'uncoded_terms': patient.attributes.get('uncoded_terms', 0),
                'visit_connections': len(neighbors.get(EdgeType.HAS_VISIT, [])),
                'query_connections': len(neighbors.get(EdgeType.HAS_QUERY, [])),
                'sae_connections': len(neighbors.get(EdgeType.HAS_ADVERSE_EVENT, [])),
                'coding_connections': len(neighbors.get(EdgeType.HAS_CODING_ISSUE, [])),
                'total_connections': sum(len(v) for v in neighbors.values())
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Patient network exported to {output_path}")


def build_clinical_data_mesh(
    study_folder: Union[str, Path],
    study_id: str = None
) -> ClinicalDataMesh:
    """
    Convenience function to build a Clinical Data Mesh from a study folder
    
    Args:
        study_folder: Path to study folder containing Excel files
        study_id: Optional study identifier
        
    Returns:
        Built ClinicalDataMesh instance
    """
    mesh = ClinicalDataMesh(study_id=study_id or "")
    mesh.build_from_study_folder(Path(study_folder), study_id)
    return mesh
