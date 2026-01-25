"""
Clinical Knowledge Graph - Core Graph Database Implementation
Transforms flat CSV clinical trial data into a multi-dimensional knowledge graph
using NetworkX as an offline graph database (replacing Neo4j/Amazon Neptune)

Architecture:
- Patient node is the central anchor
- Events, Discrepancies, SAEs, and Coding Issues are connected via edges
- Enables complex multi-hop traversals that relational databases cannot efficiently perform
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the clinical knowledge graph"""
    PATIENT = "Patient"
    SITE = "Site"
    EVENT = "Event"
    VISIT = "Visit"
    DISCREPANCY = "Discrepancy"
    QUERY = "Query"
    SAE = "SAE"
    CODING_TERM = "CodingTerm"
    MEDDRA_TERM = "MedDRATerm"
    WHODRA_TERM = "WHODRATerm"
    PROTOCOL_DEVIATION = "ProtocolDeviation"
    CRF_PAGE = "CRFPage"
    STUDY = "Study"
    COUNTRY = "Country"
    REGION = "Region"


class EdgeType(Enum):
    """Types of edges (relationships) in the clinical knowledge graph"""
    # Patient-centric relationships
    HAS_VISIT = "HAS_VISIT"
    HAS_ADVERSE_EVENT = "HAS_ADVERSE_EVENT"
    HAS_CODING_ISSUE = "HAS_CODING_ISSUE"
    HAS_QUERY = "HAS_QUERY"
    HAS_DISCREPANCY = "HAS_DISCREPANCY"
    HAS_PROTOCOL_DEVIATION = "HAS_PROTOCOL_DEVIATION"
    HAS_CRF_PAGE = "HAS_CRF_PAGE"
    
    # Site relationships
    ENROLLED_AT = "ENROLLED_AT"
    BELONGS_TO_STUDY = "BELONGS_TO_STUDY"
    
    # Geographic relationships
    LOCATED_IN_COUNTRY = "LOCATED_IN_COUNTRY"
    LOCATED_IN_REGION = "LOCATED_IN_REGION"
    
    # Temporal relationships
    OCCURRED_ON = "OCCURRED_ON"
    SCHEDULED_FOR = "SCHEDULED_FOR"
    
    # Safety relationships
    REQUIRES_RECONCILIATION = "REQUIRES_RECONCILIATION"
    RELATED_TO_SAE = "RELATED_TO_SAE"
    
    # Coding relationships
    CODED_AS = "CODED_AS"
    REQUIRES_CODING = "REQUIRES_CODING"
    
    # Query relationships
    GENERATED_QUERY = "GENERATED_QUERY"
    ANSWERS_QUERY = "ANSWERS_QUERY"


class QueryStatus(Enum):
    """Status of queries/discrepancies"""
    OPEN = "Open"
    ANSWERED = "Answered"
    CLOSED = "Closed"
    CANCELLED = "Cancelled"


class CodingStatus(Enum):
    """Medical coding status"""
    UNCODED = "UnCoded"
    CODED = "Coded"
    PENDING_REVIEW = "Pending Review"
    CLARIFICATION_NEEDED = "Clarification Needed"


@dataclass
class GraphNode:
    """Base class for all graph nodes"""
    node_id: str = ""
    node_type: NodeType = NodeType.PATIENT
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'attributes': self.attributes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class PatientNode(GraphNode):
    """
    Patient Node - Central entity in the knowledge graph
    Attributes derived from CPID_EDC_Metrics
    """
    subject_id: str = ""
    site_id: str = ""
    study_id: str = ""
    country: str = ""
    region: str = ""
    status: str = "Unknown"
    clean_status: bool = False
    clean_percentage: float = 0.0
    data_quality_index: float = 100.0
    
    # Metrics snapshot
    missing_visits: int = 0
    missing_pages: int = 0
    open_queries: int = 0
    total_queries: int = 0
    uncoded_terms: int = 0
    verification_pct: float = 0.0
    reconciliation_issues: int = 0
    protocol_deviations: int = 0
    
    def __post_init__(self):
        self.node_type = NodeType.PATIENT
        self.node_id = f"patient_{self.study_id}_{self.subject_id}"
        self.attributes = {
            'subject_id': self.subject_id,
            'site_id': self.site_id,
            'study_id': self.study_id,
            'country': self.country,
            'region': self.region,
            'status': self.status,
            'clean_status': self.clean_status,
            'clean_percentage': self.clean_percentage,
            'data_quality_index': self.data_quality_index,
            'missing_visits': self.missing_visits,
            'missing_pages': self.missing_pages,
            'open_queries': self.open_queries,
            'total_queries': self.total_queries,
            'uncoded_terms': self.uncoded_terms,
            'verification_pct': self.verification_pct,
            'reconciliation_issues': self.reconciliation_issues,
            'protocol_deviations': self.protocol_deviations
        }


@dataclass
class SiteNode(GraphNode):
    """
    Site Node - Clinical trial site
    Aggregates patient data at site level
    """
    site_id: str = ""
    study_id: str = ""
    country: str = ""
    region: str = ""
    total_patients: int = 0
    clean_patients: int = 0
    data_quality_index: float = 100.0
    risk_level: str = "Low"
    
    def __post_init__(self):
        self.node_type = NodeType.SITE
        self.node_id = f"site_{self.study_id}_{self.site_id}"
        self.attributes = {
            'site_id': self.site_id,
            'study_id': self.study_id,
            'country': self.country,
            'region': self.region,
            'total_patients': self.total_patients,
            'clean_patients': self.clean_patients,
            'data_quality_index': self.data_quality_index,
            'risk_level': self.risk_level
        }


@dataclass
class EventNode(GraphNode):
    """
    Event/Visit Node - Represents clinical trial visits
    Attributes from Visit Projection Tracker
    """
    event_id: str = ""
    subject_id: str = ""
    study_id: str = ""
    visit_name: str = ""
    projected_date: Optional[datetime] = None
    actual_date: Optional[datetime] = None
    days_outstanding: int = 0
    status: str = "Pending"
    is_missing: bool = False
    
    def __post_init__(self):
        self.node_type = NodeType.EVENT
        self.node_id = f"event_{self.study_id}_{self.subject_id}_{self.event_id}"
        self.attributes = {
            'event_id': self.event_id,
            'subject_id': self.subject_id,
            'study_id': self.study_id,
            'visit_name': self.visit_name,
            'projected_date': self.projected_date.isoformat() if self.projected_date else None,
            'actual_date': self.actual_date.isoformat() if self.actual_date else None,
            'days_outstanding': self.days_outstanding,
            'status': self.status,
            'is_missing': self.is_missing
        }


@dataclass
class DiscrepancyNode(GraphNode):
    """
    Discrepancy/Query Node - Captures data friction points
    Attributes from CPID_EDC_Metrics and query data
    """
    query_id: str = ""
    subject_id: str = ""
    study_id: str = ""
    query_type: str = "General"  # DM, Clinical, Safety, Coding
    status: QueryStatus = QueryStatus.OPEN
    days_open: int = 0
    form_name: str = ""
    field_name: str = ""
    query_text: str = ""
    response_text: str = ""
    severity: str = "Medium"
    
    def __post_init__(self):
        self.node_type = NodeType.DISCREPANCY
        self.node_id = f"query_{self.study_id}_{self.query_id}"
        self.attributes = {
            'query_id': self.query_id,
            'subject_id': self.subject_id,
            'study_id': self.study_id,
            'query_type': self.query_type,
            'status': self.status.value if isinstance(self.status, QueryStatus) else self.status,
            'days_open': self.days_open,
            'form_name': self.form_name,
            'field_name': self.field_name,
            'query_text': self.query_text,
            'severity': self.severity
        }


@dataclass
class SAENode(GraphNode):
    """
    Serious Adverse Event Node - Safety data
    Attributes from SAE Dashboard
    """
    sae_id: str = ""
    subject_id: str = ""
    study_id: str = ""
    site_id: str = ""
    event_type: str = ""
    event_date: Optional[datetime] = None
    review_status: str = "Pending"
    action_status: str = "Open"
    requires_reconciliation: bool = False
    discrepancy_id: str = ""
    days_pending: int = 0
    severity: str = "Serious"
    
    def __post_init__(self):
        self.node_type = NodeType.SAE
        self.node_id = f"sae_{self.study_id}_{self.sae_id}"
        self.attributes = {
            'sae_id': self.sae_id,
            'subject_id': self.subject_id,
            'study_id': self.study_id,
            'site_id': self.site_id,
            'event_type': self.event_type,
            'event_date': self.event_date.isoformat() if self.event_date else None,
            'review_status': self.review_status,
            'action_status': self.action_status,
            'requires_reconciliation': self.requires_reconciliation,
            'discrepancy_id': self.discrepancy_id,
            'days_pending': self.days_pending,
            'severity': self.severity
        }


@dataclass
class CodingTermNode(GraphNode):
    """
    Coding Term Node - Medical coding data
    Attributes from GlobalCodingReport_MedDRA or GlobalCodingReport_WHODRA
    """
    term_id: str = ""
    subject_id: str = ""
    study_id: str = ""
    verbatim_term: str = ""
    coded_term: str = ""
    coding_dictionary: str = "MedDRA"  # MedDRA or WHODRA
    coding_status: CodingStatus = CodingStatus.UNCODED
    context: str = ""  # Where the term appears (AE, MedHx, ConMed)
    requires_review: bool = True
    days_uncoded: int = 0
    
    def __post_init__(self):
        self.node_type = NodeType.CODING_TERM
        self.node_id = f"term_{self.study_id}_{self.term_id}"
        self.attributes = {
            'term_id': self.term_id,
            'subject_id': self.subject_id,
            'study_id': self.study_id,
            'verbatim_term': self.verbatim_term,
            'coded_term': self.coded_term,
            'coding_dictionary': self.coding_dictionary,
            'coding_status': self.coding_status.value if isinstance(self.coding_status, CodingStatus) else self.coding_status,
            'context': self.context,
            'requires_review': self.requires_review,
            'days_uncoded': self.days_uncoded
        }


@dataclass
class GraphEdge:
    """Represents an edge (relationship) in the knowledge graph"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'edge_type': self.edge_type.value,
            'properties': self.properties,
            'weight': self.weight,
            'created_at': self.created_at.isoformat()
        }


class ClinicalKnowledgeGraph:
    """
    Clinical Knowledge Graph - The core graph database implementation
    
    This class transforms flat CSV clinical trial data into a multi-dimensional
    knowledge graph using NetworkX. It provides:
    - Patient-centric data model with Subject ID as the central anchor
    - Multi-hop query capabilities
    - Graph analytics and traversals
    - Persistence and serialization
    
    The graph structure enables complex queries like:
    "Show me all patients who have a Missing Visit AND an Open Safety Query 
    AND an Uncoded Concomitant Medication"
    """
    
    def __init__(self, study_id: str = ""):
        self.study_id = study_id
        self.graph = nx.MultiDiGraph()  # Directed multigraph for multiple edge types
        self.node_index: Dict[str, GraphNode] = {}
        self.edge_index: Dict[str, List[GraphEdge]] = {}
        self._node_type_cache: Dict[NodeType, Set[str]] = {nt: set() for nt in NodeType}
        self._patient_index: Dict[str, str] = {}  # subject_id -> node_id mapping
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        logger.info(f"Initialized ClinicalKnowledgeGraph for study: {study_id}")
    
    # ==================== Node Operations ====================
    
    def add_node(self, node: GraphNode) -> str:
        """
        Add a node to the knowledge graph
        
        Args:
            node: GraphNode instance to add
            
        Returns:
            The node_id of the added node
        """
        node_id = node.node_id
        
        # Add to NetworkX graph
        self.graph.add_node(
            node_id,
            node_type=node.node_type.value,
            **node.attributes
        )
        
        # Update indices
        self.node_index[node_id] = node
        self._node_type_cache[node.node_type].add(node_id)
        
        # Special handling for patient nodes
        if node.node_type == NodeType.PATIENT and hasattr(node, 'subject_id'):
            self._patient_index[node.subject_id] = node_id
        
        self.updated_at = datetime.now()
        return node_id
    
    def add_patient(self, patient_node: PatientNode) -> str:
        """Add a patient node (convenience method)"""
        return self.add_node(patient_node)
    
    def add_site(self, site_node: SiteNode) -> str:
        """Add a site node (convenience method)"""
        return self.add_node(site_node)
    
    def add_event(self, event_node: EventNode) -> str:
        """Add an event/visit node (convenience method)"""
        return self.add_node(event_node)
    
    def add_discrepancy(self, discrepancy_node: DiscrepancyNode) -> str:
        """Add a discrepancy/query node (convenience method)"""
        return self.add_node(discrepancy_node)
    
    def add_sae(self, sae_node: SAENode) -> str:
        """Add an SAE node (convenience method)"""
        return self.add_node(sae_node)
    
    def add_coding_term(self, coding_node: CodingTermNode) -> str:
        """Add a coding term node (convenience method)"""
        return self.add_node(coding_node)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID"""
        return self.node_index.get(node_id)
    
    def get_patient_node(self, subject_id: str) -> Optional[PatientNode]:
        """Get a patient node by subject ID"""
        node_id = self._patient_index.get(subject_id)
        if node_id:
            return self.node_index.get(node_id)
        return None
    
    def get_patient_node_id(self, subject_id: str) -> Optional[str]:
        """Get a patient node ID by subject ID"""
        return self._patient_index.get(subject_id)
    
    def update_patient_node(self, node_id: str, attributes: Dict[str, Any]) -> bool:
        """Update patient node attributes with calculated metrics"""
        return self.update_node(node_id, attributes)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type"""
        node_ids = self._node_type_cache.get(node_type, set())
        return [self.node_index[nid] for nid in node_ids if nid in self.node_index]
    
    def update_node(self, node_id: str, attributes: Dict[str, Any]) -> bool:
        """Update node attributes"""
        if node_id not in self.node_index:
            return False
        
        node = self.node_index[node_id]
        node.attributes.update(attributes)
        node.updated_at = datetime.now()
        
        # Update NetworkX graph
        self.graph.nodes[node_id].update(attributes)
        self.updated_at = datetime.now()
        
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the graph"""
        if node_id not in self.node_index:
            return False
        
        node = self.node_index[node_id]
        
        # Remove from indices
        self._node_type_cache[node.node_type].discard(node_id)
        if node.node_type == NodeType.PATIENT and hasattr(node, 'subject_id'):
            self._patient_index.pop(node.subject_id, None)
        
        # Remove from NetworkX (also removes all connected edges)
        self.graph.remove_node(node_id)
        del self.node_index[node_id]
        
        self.updated_at = datetime.now()
        return True
    
    # ==================== Edge Operations ====================
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """
        Add an edge (relationship) to the knowledge graph
        
        Args:
            edge: GraphEdge instance to add
            
        Returns:
            True if edge was added successfully
        """
        if edge.source_id not in self.node_index:
            logger.warning(f"Source node {edge.source_id} not found in graph")
            return False
        if edge.target_id not in self.node_index:
            logger.warning(f"Target node {edge.target_id} not found in graph")
            return False
        
        # Add to NetworkX graph
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            **edge.properties
        )
        
        # Update edge index
        edge_key = f"{edge.source_id}_{edge.target_id}"
        if edge_key not in self.edge_index:
            self.edge_index[edge_key] = []
        self.edge_index[edge_key].append(edge)
        
        self.updated_at = datetime.now()
        return True
    
    def connect_patient_to_visit(
        self,
        patient_id: str,
        event_id: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """Create HAS_VISIT relationship"""
        edge = GraphEdge(
            source_id=patient_id,
            target_id=event_id,
            edge_type=EdgeType.HAS_VISIT,
            properties=properties or {}
        )
        return self.add_edge(edge)
    
    def connect_patient_to_sae(
        self,
        patient_id: str,
        sae_id: str,
        review_status: str = "Pending",
        action_status: str = "Open"
    ) -> bool:
        """Create HAS_ADVERSE_EVENT relationship"""
        edge = GraphEdge(
            source_id=patient_id,
            target_id=sae_id,
            edge_type=EdgeType.HAS_ADVERSE_EVENT,
            properties={
                'review_status': review_status,
                'action_status': action_status
            }
        )
        return self.add_edge(edge)
    
    def connect_patient_to_coding_issue(
        self,
        patient_id: str,
        term_id: str,
        verbatim_term: str = "",
        coding_status: str = "UnCoded"
    ) -> bool:
        """Create HAS_CODING_ISSUE relationship"""
        edge = GraphEdge(
            source_id=patient_id,
            target_id=term_id,
            edge_type=EdgeType.HAS_CODING_ISSUE,
            properties={
                'verbatim_term': verbatim_term,
                'coding_status': coding_status
            }
        )
        return self.add_edge(edge)
    
    def connect_patient_to_query(
        self,
        patient_id: str,
        query_id: str,
        query_type: str = "General",
        status: str = "Open"
    ) -> bool:
        """Create HAS_QUERY relationship"""
        edge = GraphEdge(
            source_id=patient_id,
            target_id=query_id,
            edge_type=EdgeType.HAS_QUERY,
            properties={
                'query_type': query_type,
                'status': status
            }
        )
        return self.add_edge(edge)
    
    def connect_patient_to_site(self, patient_id: str, site_id: str) -> bool:
        """Create ENROLLED_AT relationship"""
        edge = GraphEdge(
            source_id=patient_id,
            target_id=site_id,
            edge_type=EdgeType.ENROLLED_AT,
            properties={}
        )
        return self.add_edge(edge)
    
    def get_edges(
        self,
        source_id: str = None,
        target_id: str = None,
        edge_type: EdgeType = None
    ) -> List[GraphEdge]:
        """Get edges matching the specified criteria"""
        results = []
        
        if source_id and target_id:
            edge_key = f"{source_id}_{target_id}"
            edges = self.edge_index.get(edge_key, [])
            if edge_type:
                edges = [e for e in edges if e.edge_type == edge_type]
            results.extend(edges)
        elif source_id:
            for edge_key, edges in self.edge_index.items():
                if edge_key.startswith(f"{source_id}_"):
                    for edge in edges:
                        if edge_type is None or edge.edge_type == edge_type:
                            results.append(edge)
        elif target_id:
            for edge_key, edges in self.edge_index.items():
                if edge_key.endswith(f"_{target_id}"):
                    for edge in edges:
                        if edge_type is None or edge.edge_type == edge_type:
                            results.append(edge)
        else:
            for edges in self.edge_index.values():
                for edge in edges:
                    if edge_type is None or edge.edge_type == edge_type:
                        results.append(edge)
        
        return results
    
    # ==================== Graph Traversal Operations ====================
    
    def get_patient_neighbors(
        self,
        subject_id: str,
        edge_types: List[EdgeType] = None
    ) -> Dict[EdgeType, List[GraphNode]]:
        """
        Get all neighbors of a patient node
        
        This is the key graph operation that enables multi-hop queries.
        For a given patient, returns all connected nodes grouped by relationship type.
        """
        patient_node_id = self._patient_index.get(subject_id)
        if not patient_node_id:
            return {}
        
        neighbors: Dict[EdgeType, List[GraphNode]] = {et: [] for et in EdgeType}
        
        # Get outgoing edges
        for _, target, data in self.graph.out_edges(patient_node_id, data=True):
            edge_type_str = data.get('edge_type', '')
            try:
                edge_type = EdgeType(edge_type_str)
                if edge_types is None or edge_type in edge_types:
                    if target in self.node_index:
                        neighbors[edge_type].append(self.node_index[target])
            except ValueError:
                continue
        
        return {k: v for k, v in neighbors.items() if v}
    
    def traverse_from_patient(
        self,
        subject_id: str,
        max_depth: int = 2
    ) -> Dict[int, List[GraphNode]]:
        """
        BFS traversal from a patient node up to max_depth
        
        Returns nodes grouped by their depth from the patient
        """
        patient_node_id = self._patient_index.get(subject_id)
        if not patient_node_id:
            return {}
        
        visited = {patient_node_id}
        current_level = [patient_node_id]
        results: Dict[int, List[GraphNode]] = {0: [self.node_index[patient_node_id]]}
        
        for depth in range(1, max_depth + 1):
            next_level = []
            for node_id in current_level:
                for neighbor in self.graph.neighbors(node_id):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            
            if next_level:
                results[depth] = [
                    self.node_index[nid] for nid in next_level 
                    if nid in self.node_index
                ]
            current_level = next_level
        
        return results
    
    def find_path(
        self,
        source_subject_id: str,
        target_node_id: str
    ) -> Optional[List[str]]:
        """Find shortest path between a patient and another node"""
        source_node_id = self._patient_index.get(source_subject_id)
        if not source_node_id:
            return None
        
        try:
            path = nx.shortest_path(self.graph, source_node_id, target_node_id)
            return path
        except nx.NetworkXNoPath:
            return None
    
    # ==================== Graph Statistics ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'nodes_by_type': {},
            'study_id': self.study_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        for node_type in NodeType:
            count = len(self._node_type_cache.get(node_type, set()))
            if count > 0:
                stats['nodes_by_type'][node_type.value] = count
        
        # Calculate additional metrics
        if self.graph.number_of_nodes() > 0:
            stats['density'] = nx.density(self.graph)
            stats['avg_degree'] = sum(d for n, d in self.graph.degree()) / self.graph.number_of_nodes()
        
        return stats
    
    def get_patient_statistics(self, subject_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific patient"""
        patient_node_id = self._patient_index.get(subject_id)
        if not patient_node_id or patient_node_id not in self.node_index:
            return {}
        
        patient_node = self.node_index[patient_node_id]
        neighbors = self.get_patient_neighbors(subject_id)
        
        stats = {
            'subject_id': subject_id,
            'node_id': patient_node_id,
            'attributes': patient_node.attributes,
            'connections': {},
            'total_connections': 0
        }
        
        for edge_type, nodes in neighbors.items():
            stats['connections'][edge_type.value] = len(nodes)
            stats['total_connections'] += len(nodes)
        
        return stats
    
    # ==================== Persistence Operations ====================
    
    def save(self, filepath: Union[str, Path]) -> bool:
        """
        Save the knowledge graph to disk
        
        Uses pickle for NetworkX graph and JSON for metadata
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save NetworkX graph
            graph_path = filepath.with_suffix('.gpickle')
            with open(graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
            
            # Save metadata and indices
            metadata = {
                'study_id': self.study_id,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
                'patient_index': self._patient_index,
                'statistics': self.get_statistics()
            }
            
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved knowledge graph to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ClinicalKnowledgeGraph':
        """Load a knowledge graph from disk"""
        filepath = Path(filepath)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(study_id=metadata.get('study_id', ''))
        instance.created_at = datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat()))
        instance._patient_index = metadata.get('patient_index', {})
        
        # Load NetworkX graph
        graph_path = filepath.with_suffix('.gpickle')
        with open(graph_path, 'rb') as f:
            instance.graph = pickle.load(f)
        
        # Rebuild node index from graph
        for node_id, attrs in instance.graph.nodes(data=True):
            node_type_str = attrs.get('node_type', 'Patient')
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.PATIENT
            
            node = GraphNode(
                node_id=node_id,
                node_type=node_type,
                attributes=attrs
            )
            instance.node_index[node_id] = node
            instance._node_type_cache[node_type].add(node_id)
        
        logger.info(f"Loaded knowledge graph from {filepath}")
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary"""
        return {
            'study_id': self.study_id,
            'statistics': self.get_statistics(),
            'nodes': [node.to_dict() for node in self.node_index.values()],
            'edges': [edge.to_dict() for edges in self.edge_index.values() for edge in edges]
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ClinicalKnowledgeGraph(study_id='{self.study_id}', "
            f"nodes={stats['total_nodes']}, edges={stats['total_edges']})"
        )
