"""
Digital Patient Twin Module
Creates unified, machine-readable patient representations for AI agents

The Digital Patient Twin is the single source of truth for:
1. Dashboard UI rendering
2. AI/ML model input features  
3. Agentic AI decision making
4. Risk-Based Monitoring (RBM) engine

Output Format:
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
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_models import (
    DigitalPatientTwin, BlockingItem, RiskMetrics,
    PatientStatus, RiskLevel, QueryStatus, CodingStatus
)
from graph.knowledge_graph import (
    ClinicalKnowledgeGraph, NodeType, EdgeType,
    PatientNode, SiteNode, EventNode, DiscrepancyNode, 
    SAENode, CodingTermNode
)

logger = logging.getLogger(__name__)


class BlockingItemType(Enum):
    """Types of items that can block clean patient status"""
    MISSING_VISIT = "Missing Visit"
    MISSING_PAGE = "Missing Page"
    OPEN_QUERY = "Open Query"
    UNCODED_TERM = "Uncoded Term"
    SAE_PENDING = "SAE Pending Review"
    RECONCILIATION = "Reconciliation Issue"
    VERIFICATION = "Verification Incomplete"
    NON_CONFORMANT = "Non-Conformant Data"
    PROTOCOL_DEVIATION = "Protocol Deviation"


class BlockingSeverity(Enum):
    """Severity levels for blocking items"""
    CRITICAL = "Critical"    # Blocks data lock
    HIGH = "High"            # Requires immediate attention
    MEDIUM = "Medium"        # Should be addressed soon
    LOW = "Low"              # Can be addressed during routine work


@dataclass
class DigitalTwinConfig:
    """Configuration for Digital Patient Twin generation"""
    # Clean status thresholds
    max_missing_visits: int = 0
    max_missing_pages: int = 0
    max_open_queries: int = 0
    max_uncoded_terms: int = 0
    min_verification_pct: float = 100.0
    max_reconciliation_issues: int = 0
    
    # Risk thresholds
    query_aging_warning_days: int = 14
    query_aging_critical_days: int = 30
    high_risk_composite_threshold: float = 60.0
    
    # Blocking item severity mappings
    severity_weights: Dict[BlockingItemType, BlockingSeverity] = field(default_factory=lambda: {
        BlockingItemType.MISSING_VISIT: BlockingSeverity.HIGH,
        BlockingItemType.MISSING_PAGE: BlockingSeverity.MEDIUM,
        BlockingItemType.OPEN_QUERY: BlockingSeverity.MEDIUM,
        BlockingItemType.UNCODED_TERM: BlockingSeverity.LOW,
        BlockingItemType.SAE_PENDING: BlockingSeverity.CRITICAL,
        BlockingItemType.RECONCILIATION: BlockingSeverity.HIGH,
        BlockingItemType.VERIFICATION: BlockingSeverity.MEDIUM,
        BlockingItemType.NON_CONFORMANT: BlockingSeverity.HIGH,
        BlockingItemType.PROTOCOL_DEVIATION: BlockingSeverity.HIGH,
    })


class DigitalTwinFactory:
    """
    Factory for creating Digital Patient Twins from knowledge graph data
    
    The factory traverses the patient-centric graph to collect all relevant
    data and compute derived metrics, producing a single consolidated JSON
    object per patient.
    
    This unified view enables:
    - AI agents to "read" patient status without human intervention
    - Dashboard UI to render complete patient information
    - RBM engine to prioritize monitoring activities
    """
    
    def __init__(
        self, 
        graph: Union[ClinicalKnowledgeGraph, nx.MultiDiGraph],
        config: Optional[DigitalTwinConfig] = None
    ):
        self.graph = graph
        self.config = config or DigitalTwinConfig()
        self._feature_cache: Dict[str, Dict] = {}
        self._is_networkx = isinstance(graph, nx.MultiDiGraph)
    
    def _create_patient_node_from_nx(self, node_data: Dict) -> PatientNode:
        """Create a PatientNode from NetworkX node data"""
        return PatientNode(
            node_id=node_data.get('node_id', ''),
            attributes=node_data
        )
    
    def _get_patient_neighbors_nx(self, subject_id: str) -> Dict[EdgeType, List]:
        """Get patient neighbors from NetworkX graph"""
        # Find patient node
        patient_node_id = None
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'Patient' and node_data.get('subject_id') == subject_id:
                patient_node_id = node_id
                break
        
        if not patient_node_id:
            return {}
        
        neighbors = {et: [] for et in EdgeType}
        
        # Get outgoing edges
        for _, target, edge_data in self.graph.out_edges(patient_node_id, data=True):
            edge_type_str = edge_data.get('edge_type', '')
            try:
                edge_type = EdgeType(edge_type_str)
                target_data = self.graph.nodes[target]
                # Create appropriate node type based on target node type
                node = self._create_node_from_nx(target_data)
                if node:
                    neighbors[edge_type].append(node)
            except ValueError:
                continue
        
        return neighbors
    
    def _create_node_from_nx(self, node_data: Dict) -> Optional[Any]:
        """Create appropriate node type from NetworkX node data"""
        node_type = node_data.get('node_type', '')
        if node_type == 'Patient':
            return PatientNode(node_id=node_data.get('node_id', ''), attributes=node_data)
        elif node_type == 'Site':
            return SiteNode(node_id=node_data.get('node_id', ''), attributes=node_data)
        elif node_type == 'Event':
            return EventNode(node_id=node_data.get('node_id', ''), attributes=node_data)
        elif node_type == 'Discrepancy':
            return DiscrepancyNode(node_id=node_data.get('node_id', ''), attributes=node_data)
        elif node_type == 'SAE':
            return SAENode(node_id=node_data.get('node_id', ''), attributes=node_data)
        elif node_type == 'CodingTerm':
            return CodingTermNode(node_id=node_data.get('node_id', ''), attributes=node_data)
        else:
            # Generic node for unknown types
            return type('GenericNode', (), {'node_id': node_data.get('node_id', ''), 'attributes': node_data})()
    
    def create_twin(self, subject_id: str) -> Optional[DigitalPatientTwin]:
        """
        Create a Digital Patient Twin for a specific patient
        
        This method traverses all graph relationships from the patient node
        to collect blocking items, calculate risk metrics, and produce
        a unified patient representation.
        
        Args:
            subject_id: The patient's subject identifier
            
        Returns:
            DigitalPatientTwin object or None if patient not found
        """
        # Find patient node in graph
        patient_node = self._find_patient_node(subject_id)
        if not patient_node:
            logger.warning(f"Patient {subject_id} not found in graph")
            return None
        
        # Extract base patient attributes
        attrs = patient_node.attributes
        
        # Get all connected data via graph traversal
        if self._is_networkx:
            neighbors = self._get_patient_neighbors_nx(subject_id)
        else:
            neighbors = self.graph.get_patient_neighbors(subject_id)
        
        # Build blocking items list
        blocking_items = self._identify_blocking_items(patient_node, neighbors)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(patient_node, neighbors, blocking_items)
        
        # Determine clean status
        clean_status = len(blocking_items) == 0
        clean_percentage = self._calculate_clean_percentage(patient_node, blocking_items)
        
        # Parse status
        status_str = attrs.get('status', 'Unknown')
        try:
            status = PatientStatus(status_str) if status_str in [e.value for e in PatientStatus] else PatientStatus.UNKNOWN
        except:
            status = PatientStatus.UNKNOWN
        
        # Build outstanding visits list
        outstanding_visits = self._extract_outstanding_visits(neighbors)
        
        # Build SAE records list
        sae_records = self._extract_sae_records(neighbors)
        
        # Build uncoded terms list
        uncoded_terms_list = self._extract_uncoded_terms(neighbors)
        
        # Create the Digital Twin
        twin = DigitalPatientTwin(
            subject_id=subject_id,
            site_id=str(attrs.get('site_id', '')),
            study_id=str(attrs.get('study_id', '')),
            country=str(attrs.get('country', '')),
            region=str(attrs.get('region', '')),
            status=status,
            clean_status=clean_status,
            clean_percentage=clean_percentage,
            blocking_items=blocking_items,
            missing_visits=int(attrs.get('missing_visits', 0)),
            missing_pages=int(attrs.get('missing_pages', 0)),
            open_queries=int(attrs.get('open_queries', 0)),
            total_queries=int(attrs.get('total_queries', 0)),
            uncoded_terms=int(attrs.get('uncoded_terms', 0)),
            coded_terms=int(attrs.get('coded_terms', 0)),
            verification_pct=float(attrs.get('verification_pct', 0)),
            forms_verified=int(attrs.get('forms_verified', 0)),
            expected_visits=int(attrs.get('expected_visits', 0)),
            pages_entered=int(attrs.get('pages_entered', 0)),
            non_conformant_pages=int(attrs.get('non_conformant_pages', 0)),
            reconciliation_issues=int(attrs.get('reconciliation_issues', 0)),
            protocol_deviations=int(attrs.get('protocol_deviations', 0)),
            risk_metrics=risk_metrics,
            data_quality_index=float(attrs.get('data_quality_index', 100.0)),
            outstanding_visits=outstanding_visits,
            sae_records=sae_records,
            uncoded_terms_list=uncoded_terms_list,
            last_updated=datetime.now()
        )
        
        return twin
    
    def create_all_twins(self) -> List[DigitalPatientTwin]:
        """Create Digital Patient Twins for all patients in the graph"""
        twins = []
        
        patient_nodes = self.graph.get_nodes_by_type(NodeType.PATIENT)
        
        for patient_node in patient_nodes:
            subject_id = patient_node.attributes.get('subject_id', '')
            if subject_id:
                twin = self.create_twin(subject_id)
                if twin:
                    twins.append(twin)
        
        logger.info(f"Created {len(twins)} Digital Patient Twins")
        return twins
    
    def export_twins_json(self, twins: List[DigitalPatientTwin], output_path: Path) -> None:
        """Export all twins to a JSON file"""
        study_id = twins[0].study_id if twins else "unknown"
        export_data = {
            'study_id': study_id,
            'generated_at': datetime.now().isoformat(),
            'patient_count': len(twins),
            'twins': [twin.to_dict() for twin in twins]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(twins)} Digital Twins to {output_path}")
    
    def export_twin_json(self, twin: DigitalPatientTwin, output_path: Path) -> None:
        """Export a single twin to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(twin.to_dict(), f, indent=2, default=str)
    
    def get_ai_readable_twin(self, subject_id: str) -> Optional[Dict]:
        """
        Get a simplified, AI-agent-readable twin representation
        
        This format is optimized for machine consumption by AI agents,
        matching the exact specification:
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
        """
        twin = self.create_twin(subject_id)
        if not twin:
            return None
        
        return {
            "subject_id": twin.subject_id,
            "status": twin.status.value,
            "clean_status": twin.clean_status,
            "blocking_items": [item.to_dict() for item in twin.blocking_items],
            "risk_metrics": {
                "query_aging_index": round(twin.risk_metrics.query_aging_index, 2),
                "protocol_deviation_count": twin.protocol_deviations,
                "manipulation_risk_score": twin.risk_metrics.manipulation_risk_score,
                "composite_risk_score": round(twin.risk_metrics.composite_risk_score, 1),
                "velocity_index": round(twin.risk_metrics.net_velocity, 3),
                "data_density": round(twin.risk_metrics.data_density_score, 4),
                "requires_intervention": twin.risk_metrics.requires_intervention
            }
        }
    
    def get_all_ai_readable_twins(self) -> List[Dict]:
        """Get AI-readable twins for all patients"""
        results = []
        patient_nodes = self.graph.get_nodes_by_type(NodeType.PATIENT)
        
        for patient_node in patient_nodes:
            subject_id = patient_node.attributes.get('subject_id', '')
            if subject_id:
                ai_twin = self.get_ai_readable_twin(subject_id)
                if ai_twin:
                    results.append(ai_twin)
        
        return results
    
    # ==================== Private Helper Methods ====================
    
    def _find_patient_node(self, subject_id: str) -> Optional[PatientNode]:
        """Find patient node in graph by subject ID"""
        if self._is_networkx:
            # Work with NetworkX graph
            for node_id, node_data in self.graph.nodes(data=True):
                if node_data.get('node_type') == 'Patient' and node_data.get('subject_id') == subject_id:
                    # Create a PatientNode-like object from NetworkX node data
                    return self._create_patient_node_from_nx(node_data)
            return None
        else:
            # Original ClinicalKnowledgeGraph implementation
            for node in self.graph.get_nodes_by_type(NodeType.PATIENT):
                if node.attributes.get('subject_id') == subject_id:
                    return node
            return None
    
    def _identify_blocking_items(
        self, 
        patient_node: PatientNode,
        neighbors: Dict[EdgeType, List]
    ) -> List[BlockingItem]:
        """
        Identify all items blocking clean patient status
        
        Traverses patient's graph connections to find:
        - Missing visits from Visit Tracker
        - Open queries from CPID
        - Uncoded terms from GlobalCodingReport
        - Pending SAEs from SAE Dashboard
        - Reconciliation issues from EDRR
        """
        blocking_items = []
        attrs = patient_node.attributes
        
        # Check missing visits
        missing_visits = int(attrs.get('missing_visits', 0))
        if missing_visits > self.config.max_missing_visits:
            blocking_items.append(BlockingItem(
                item_type=BlockingItemType.MISSING_VISIT.value,
                description=f"{missing_visits} missing visit(s) detected",
                source_file="Visit Projection Tracker",
                severity=self.config.severity_weights[BlockingItemType.MISSING_VISIT].value,
                days_outstanding=self._get_max_visit_days(neighbors)
            ))
        
        # Check missing pages
        missing_pages = int(attrs.get('missing_pages', 0))
        if missing_pages > self.config.max_missing_pages:
            blocking_items.append(BlockingItem(
                item_type=BlockingItemType.MISSING_PAGE.value,
                description=f"{missing_pages} missing page(s) detected",
                source_file="CPID_EDC_Metrics",
                severity=self.config.severity_weights[BlockingItemType.MISSING_PAGE].value
            ))
        
        # Check open queries
        open_queries = int(attrs.get('open_queries', 0))
        if open_queries > self.config.max_open_queries:
            blocking_items.append(BlockingItem(
                item_type=BlockingItemType.OPEN_QUERY.value,
                description=f"{open_queries} open query(ies) pending response",
                source_file="CPID_EDC_Metrics",
                severity=self.config.severity_weights[BlockingItemType.OPEN_QUERY].value,
                days_outstanding=self._get_max_query_age(neighbors)
            ))
        
        # Check uncoded terms
        uncoded_terms = int(attrs.get('uncoded_terms', 0))
        if uncoded_terms > self.config.max_uncoded_terms:
            blocking_items.append(BlockingItem(
                item_type=BlockingItemType.UNCODED_TERM.value,
                description=f"{uncoded_terms} term(s) require medical coding",
                source_file="GlobalCodingReport",
                severity=self.config.severity_weights[BlockingItemType.UNCODED_TERM].value
            ))
        
        # Check SAE records for pending review
        pending_sae_count = self._count_pending_saes(neighbors)
        if pending_sae_count > 0:
            blocking_items.append(BlockingItem(
                item_type=BlockingItemType.SAE_PENDING.value,
                description=f"{pending_sae_count} SAE(s) pending review or action",
                source_file="SAE Dashboard",
                severity=self.config.severity_weights[BlockingItemType.SAE_PENDING].value
            ))
        
        # Check reconciliation issues
        reconciliation_issues = int(attrs.get('reconciliation_issues', 0))
        if reconciliation_issues > self.config.max_reconciliation_issues:
            blocking_items.append(BlockingItem(
                item_type=BlockingItemType.RECONCILIATION.value,
                description=f"{reconciliation_issues} reconciliation issue(s)",
                source_file="Compiled_EDRR",
                severity=self.config.severity_weights[BlockingItemType.RECONCILIATION].value
            ))
        
        # Check verification percentage
        verification_pct = float(attrs.get('verification_pct', 0))
        if verification_pct < self.config.min_verification_pct:
            blocking_items.append(BlockingItem(
                item_type=BlockingItemType.VERIFICATION.value,
                description=f"Verification at {verification_pct:.1f}% (target: {self.config.min_verification_pct}%)",
                source_file="CPID_EDC_Metrics",
                severity=self.config.severity_weights[BlockingItemType.VERIFICATION].value
            ))
        
        # Check non-conformant pages
        non_conformant = int(attrs.get('non_conformant_pages', 0))
        if non_conformant > 0:
            blocking_items.append(BlockingItem(
                item_type=BlockingItemType.NON_CONFORMANT.value,
                description=f"{non_conformant} non-conformant page(s)",
                source_file="CPID_EDC_Metrics",
                severity=self.config.severity_weights[BlockingItemType.NON_CONFORMANT].value
            ))
        
        # Check protocol deviations
        deviations = int(attrs.get('protocol_deviations', 0))
        if deviations > 0:
            blocking_items.append(BlockingItem(
                item_type=BlockingItemType.PROTOCOL_DEVIATION.value,
                description=f"{deviations} protocol deviation(s) recorded",
                source_file="CPID_EDC_Metrics",
                severity=self.config.severity_weights[BlockingItemType.PROTOCOL_DEVIATION].value
            ))
        
        # Sort by severity (Critical first)
        severity_order = {
            BlockingSeverity.CRITICAL.value: 0,
            BlockingSeverity.HIGH.value: 1,
            BlockingSeverity.MEDIUM.value: 2,
            BlockingSeverity.LOW.value: 3
        }
        blocking_items.sort(key=lambda x: severity_order.get(x.severity, 99))
        
        return blocking_items
    
    def _calculate_risk_metrics(
        self,
        patient_node: PatientNode,
        neighbors: Dict[EdgeType, List],
        blocking_items: List[BlockingItem]
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the patient
        
        Integrates the three engineered features:
        1. Operational Velocity Index
        2. Normalized Data Density
        3. Manipulation Risk Score
        """
        attrs = patient_node.attributes
        
        # Get pre-computed values from patient node if available
        risk_metrics = RiskMetrics()
        
        # Query Aging Index - based on max query age
        max_query_age = self._get_max_query_age(neighbors)
        if max_query_age:
            # Normalize to 0-1 scale (0=no aging, 1=critical aging)
            risk_metrics.query_aging_index = min(max_query_age / self.config.query_aging_critical_days, 1.0)
        
        # Protocol deviation count
        risk_metrics.protocol_deviation_count = int(attrs.get('protocol_deviations', 0))
        
        # Safety signals from SAE
        risk_metrics.safety_signal_count = len(neighbors.get(EdgeType.HAS_ADVERSE_EVENT, []))
        
        # Visit compliance
        expected_visits = int(attrs.get('expected_visits', 1))
        missing_visits = int(attrs.get('missing_visits', 0))
        if expected_visits > 0:
            risk_metrics.visit_compliance_rate = ((expected_visits - missing_visits) / expected_visits) * 100
        
        # Feature 1: Velocity Index (from patient attributes if computed)
        risk_metrics.resolution_velocity = float(attrs.get('resolution_velocity', 0))
        risk_metrics.accumulation_velocity = float(attrs.get('accumulation_velocity', 0))
        risk_metrics.net_velocity = risk_metrics.resolution_velocity - risk_metrics.accumulation_velocity
        risk_metrics.is_bottleneck = risk_metrics.net_velocity < 0
        
        # Feature 2: Data Density
        pages_entered = int(attrs.get('pages_entered', 1))
        total_queries = int(attrs.get('total_queries', 0))
        if pages_entered > 0:
            risk_metrics.data_density_score = total_queries / pages_entered
        risk_metrics.query_density_normalized = min(risk_metrics.data_density_score / 0.15, 1.0)  # Normalize to 15% max
        
        # Feature 3: Manipulation Risk Score
        manipulation_value = float(attrs.get('manipulation_risk_value', 0))
        risk_metrics.manipulation_risk_value = manipulation_value
        risk_metrics.inactivation_rate = float(attrs.get('inactivation_rate', 0))
        
        # Determine manipulation risk level
        if manipulation_value >= 80:
            risk_metrics.manipulation_risk_score = "Critical"
        elif manipulation_value >= 60:
            risk_metrics.manipulation_risk_score = "High"
        elif manipulation_value >= 40:
            risk_metrics.manipulation_risk_score = "Elevated"
        elif manipulation_value >= 20:
            risk_metrics.manipulation_risk_score = "Moderate"
        else:
            risk_metrics.manipulation_risk_score = "Low"
        
        # Calculate composite risk score (weighted average)
        composite = (
            risk_metrics.query_aging_index * 25 +  # 25% weight
            (risk_metrics.protocol_deviation_count > 0) * 20 +  # 20% weight
            (risk_metrics.safety_signal_count > 0) * 20 +  # 20% weight
            (1 - risk_metrics.visit_compliance_rate / 100) * 15 +  # 15% weight
            risk_metrics.query_density_normalized * 10 +  # 10% weight
            (risk_metrics.manipulation_risk_value / 100) * 10  # 10% weight
        )
        risk_metrics.composite_risk_score = min(composite * 100, 100)
        
        # Determine if intervention needed
        risk_metrics.requires_intervention = (
            risk_metrics.composite_risk_score >= self.config.high_risk_composite_threshold or
            risk_metrics.manipulation_risk_score in ["Critical", "High"] or
            len(blocking_items) >= 3  # Multiple blocking items
        )
        
        return risk_metrics
    
    def _calculate_clean_percentage(
        self,
        patient_node: PatientNode,
        blocking_items: List[BlockingItem]
    ) -> float:
        """Calculate percentage completion towards clean status"""
        total_criteria = 9  # Total clean status criteria
        
        # Count criteria met
        criteria_met = total_criteria - len(set(item.item_type for item in blocking_items))
        
        return round((criteria_met / total_criteria) * 100, 1)
    
    def _get_max_visit_days(self, neighbors: Dict[EdgeType, List]) -> Optional[int]:
        """Get maximum days outstanding for visits"""
        max_days = 0
        for visit in neighbors.get(EdgeType.HAS_VISIT, []):
            if isinstance(visit, dict):
                days = visit.get('days_outstanding', 0)
            else:
                days = getattr(visit, 'days_outstanding', 0) if hasattr(visit, 'days_outstanding') else 0
            if days and days > max_days:
                max_days = days
        return max_days if max_days > 0 else None
    
    def _get_max_query_age(self, neighbors: Dict[EdgeType, List]) -> Optional[int]:
        """Get maximum age of open queries in days"""
        max_age = 0
        for query in neighbors.get(EdgeType.HAS_QUERY, []):
            if isinstance(query, dict):
                age = query.get('days_open', 0)
            else:
                age = getattr(query, 'days_open', 0) if hasattr(query, 'days_open') else 0
            if age and age > max_age:
                max_age = age
        return max_age if max_age > 0 else None
    
    def _count_pending_saes(self, neighbors: Dict[EdgeType, List]) -> int:
        """Count SAEs with pending review or open action"""
        pending_count = 0
        for sae in neighbors.get(EdgeType.HAS_ADVERSE_EVENT, []):
            if isinstance(sae, dict):
                review_status = str(sae.get('review_status', '')).lower()
                action_status = str(sae.get('action_status', '')).lower()
            else:
                review_status = str(getattr(sae, 'review_status', '')).lower() if hasattr(sae, 'review_status') else ''
                action_status = str(getattr(sae, 'action_status', '')).lower() if hasattr(sae, 'action_status') else ''
            
            if 'pending' in review_status or 'open' in action_status:
                pending_count += 1
        
        return pending_count
    
    def _extract_outstanding_visits(self, neighbors: Dict[EdgeType, List]) -> List[Dict]:
        """Extract outstanding visits as list of dicts"""
        visits = []
        for visit in neighbors.get(EdgeType.HAS_VISIT, []):
            if isinstance(visit, dict):
                if visit.get('days_outstanding', 0) > 0:
                    visits.append({
                        'visit_name': visit.get('visit_name', 'Unknown'),
                        'days_outstanding': visit.get('days_outstanding', 0),
                        'status': visit.get('status', 'Unknown')
                    })
            elif hasattr(visit, 'attributes'):
                attrs = visit.attributes if hasattr(visit, 'attributes') else {}
                if attrs.get('days_outstanding', 0) > 0:
                    visits.append({
                        'visit_name': attrs.get('visit_name', 'Unknown'),
                        'days_outstanding': attrs.get('days_outstanding', 0),
                        'status': attrs.get('status', 'Unknown')
                    })
        return visits
    
    def _extract_sae_records(self, neighbors: Dict[EdgeType, List]) -> List[Dict]:
        """Extract SAE records as list of dicts"""
        sae_list = []
        for sae in neighbors.get(EdgeType.HAS_ADVERSE_EVENT, []):
            if isinstance(sae, dict):
                sae_list.append({
                    'sae_id': sae.get('sae_id', ''),
                    'review_status': sae.get('review_status', ''),
                    'action_status': sae.get('action_status', '')
                })
            elif hasattr(sae, 'attributes'):
                attrs = sae.attributes if hasattr(sae, 'attributes') else {}
                sae_list.append({
                    'sae_id': attrs.get('sae_id', ''),
                    'review_status': attrs.get('review_status', ''),
                    'action_status': attrs.get('action_status', '')
                })
        return sae_list
    
    def _extract_uncoded_terms(self, neighbors: Dict[EdgeType, List]) -> List[Dict]:
        """Extract uncoded terms as list of dicts"""
        terms = []
        for term in neighbors.get(EdgeType.HAS_CODING_ISSUE, []):
            if isinstance(term, dict):
                if term.get('coding_status', '').lower() == 'uncoded':
                    terms.append({
                        'verbatim_term': term.get('verbatim_term', ''),
                        'dictionary': term.get('dictionary', ''),
                        'context': term.get('context', '')
                    })
            elif hasattr(term, 'attributes'):
                attrs = term.attributes if hasattr(term, 'attributes') else {}
                if str(attrs.get('coding_status', '')).lower() == 'uncoded':
                    terms.append({
                        'verbatim_term': attrs.get('verbatim_term', ''),
                        'dictionary': attrs.get('dictionary', ''),
                        'context': attrs.get('context', '')
                    })
        return terms


def create_digital_twins(
    graph: ClinicalKnowledgeGraph,
    config: Optional[DigitalTwinConfig] = None
) -> Tuple[List[DigitalPatientTwin], DigitalTwinFactory]:
    """
    Convenience function to create all Digital Patient Twins
    
    Args:
        graph: The Clinical Knowledge Graph
        config: Optional configuration
        
    Returns:
        Tuple of (list of twins, factory instance)
    """
    factory = DigitalTwinFactory(graph, config)
    twins = factory.create_all_twins()
    return twins, factory
