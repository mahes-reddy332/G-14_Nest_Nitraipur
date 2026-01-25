"""
Clinical Graph Builder - Transforms flat CSV data into Knowledge Graph
Implements the data integration layer from the Neural Clinical Data Mesh architecture

This module ingests the provided CSV snapshots and transforms them from flat files
into a multi-dimensional knowledge graph, moving beyond the traditional "Data Lake"
to an active "Data Mesh" architecture.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import logging
import uuid
import hashlib

from .knowledge_graph import (
    ClinicalKnowledgeGraph, NodeType, EdgeType,
    PatientNode, SiteNode, EventNode, DiscrepancyNode, 
    SAENode, CodingTermNode, GraphEdge
)

logger = logging.getLogger(__name__)


class ClinicalGraphBuilder:
    """
    Transforms flat CSV clinical trial data into a multi-dimensional knowledge graph
    
    Ingestion and Mapping Logic:
    - Node: Patient (central entity) - from CPID_EDC_Metrics
    - Node: Site - aggregated from patient data
    - Node: Event/Visit - from Visit Projection Tracker
    - Node: Discrepancy/Query - from CPID_EDC_Metrics
    - Node: SAE - from SAE Dashboard
    - Node: CodingTerm - from GlobalCodingReport_MedDRA/WHODRA
    
    Edge relationships:
    - HAS_VISIT: Patient -> Event
    - HAS_ADVERSE_EVENT: Patient -> SAE
    - HAS_CODING_ISSUE: Patient -> CodingTerm
    - HAS_QUERY: Patient -> Discrepancy
    - ENROLLED_AT: Patient -> Site
    """
    
    def __init__(self, study_id: str = ""):
        self.study_id = study_id
        self.graph = ClinicalKnowledgeGraph(study_id=study_id)
        self._site_cache: Dict[str, str] = {}  # site_id -> node_id
        self._patient_cache: Dict[str, str] = {}  # subject_id -> node_id
        self._stats = {
            'patients_created': 0,
            'sites_created': 0,
            'visits_created': 0,
            'queries_created': 0,
            'saes_created': 0,
            'coding_terms_created': 0,
            'edges_created': 0
        }
    
    def build_from_study_data(
        self,
        study_data: Dict[str, pd.DataFrame],
        study_id: str = None
    ) -> ClinicalKnowledgeGraph:
        """
        Build the complete knowledge graph from study data
        
        Args:
            study_data: Dictionary containing DataFrames:
                - cpid_metrics: CPID_EDC_Metrics data
                - visit_tracker: Visit Projection Tracker data
                - sae_dashboard: SAE Dashboard data
                - meddra_coding: MedDRA coding report
                - whodra_coding: WHODRA coding report
                - compiled_edrr: EDRR data
            study_id: Override study ID
            
        Returns:
            Populated ClinicalKnowledgeGraph
        """
        if study_id:
            self.study_id = study_id
            self.graph.study_id = study_id
        
        logger.info(f"Building knowledge graph for {self.study_id}")
        
        # Step 1: Build patient nodes (central entities)
        if 'cpid_metrics' in study_data and study_data['cpid_metrics'] is not None:
            self._build_patient_nodes(study_data['cpid_metrics'])
        
        # Step 2: Build site nodes (aggregated)
        self._build_site_nodes()
        
        # Step 3: Build event/visit nodes
        if 'visit_tracker' in study_data and study_data['visit_tracker'] is not None:
            self._build_visit_nodes(study_data['visit_tracker'])
        
        # Step 4: Build SAE nodes
        if 'sae_dashboard' in study_data and study_data['sae_dashboard'] is not None:
            self._build_sae_nodes(study_data['sae_dashboard'])
        
        # Step 5: Build coding term nodes (MedDRA)
        if 'meddra_coding' in study_data and study_data['meddra_coding'] is not None:
            self._build_coding_nodes(study_data['meddra_coding'], 'MedDRA')
        
        # Step 6: Build coding term nodes (WHODRA)
        if 'whodra_coding' in study_data and study_data['whodra_coding'] is not None:
            self._build_coding_nodes(study_data['whodra_coding'], 'WHODRA')
        
        # Step 7: Build reconciliation/EDRR nodes
        if 'compiled_edrr' in study_data and study_data['compiled_edrr'] is not None:
            self._build_edrr_nodes(study_data['compiled_edrr'])
        
        # Log statistics
        logger.info(f"Graph build complete for {self.study_id}:")
        for key, value in self._stats.items():
            logger.info(f"  {key}: {value}")
        
        return self.graph
    
    def _safe_get(self, row: pd.Series, columns: List[str], default: Any = None) -> Any:
        """Safely get value from row trying multiple column names"""
        for col in columns:
            if col in row.index:
                value = row[col]
                if pd.notna(value):
                    return value
        return default
    
    def _safe_numeric(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to numeric"""
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to integer"""
        return int(self._safe_numeric(value, float(default)))
    
    def _generate_id(self, prefix: str, *parts) -> str:
        """Generate a unique ID from parts"""
        combined = '_'.join(str(p) for p in parts if p)
        if not combined:
            combined = uuid.uuid4().hex[:8]
        # Create a hash for consistent IDs
        hash_val = hashlib.md5(combined.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_val}"
    
    # ==================== Patient Node Builder ====================
    
    def _build_patient_nodes(self, cpid_df: pd.DataFrame) -> int:
        """
        Build patient nodes from CPID_EDC_Metrics
        
        The Patient node is the central anchor in the knowledge graph.
        """
        count = 0
        
        # Column mapping for patient attributes
        subject_cols = ['subject_id', 'Subject ID', 'SubjectID', 'Subject_ID']
        site_cols = ['site_id', 'Site ID', 'SiteID', 'Site_ID', 'Site']
        country_cols = ['country', 'Country', 'COUNTRY']
        region_cols = ['region', 'Region', 'REGION']
        status_cols = ['status', 'Subject Status', 'Status', 'SUBJECT_STATUS']
        
        for idx, row in cpid_df.iterrows():
            try:
                subject_id = self._safe_get(row, subject_cols)
                if not subject_id or pd.isna(subject_id):
                    continue
                
                subject_id = str(subject_id).strip()
                site_id = str(self._safe_get(row, site_cols, 'Unknown')).strip()
                
                # Create patient node
                patient_node = PatientNode(
                    subject_id=subject_id,
                    site_id=site_id,
                    study_id=self.study_id,
                    country=str(self._safe_get(row, country_cols, '')),
                    region=str(self._safe_get(row, region_cols, '')),
                    status=str(self._safe_get(row, status_cols, 'Unknown')),
                    clean_status=False,  # Will be calculated later
                    clean_percentage=0.0,
                    data_quality_index=100.0,
                    missing_visits=self._safe_int(self._safe_get(row, ['missing_visits', 'Missing Visits', '# Missing Visits'])),
                    missing_pages=self._safe_int(self._safe_get(row, ['missing_pages', 'Missing Page', '# Missing Pages'])),
                    open_queries=self._safe_int(self._safe_get(row, ['open_queries', '# Open Queries', 'Open Queries'])),
                    total_queries=self._safe_int(self._safe_get(row, ['total_queries', '# Total Queries', 'Total Queries'])),
                    uncoded_terms=self._safe_int(self._safe_get(row, ['uncoded_terms', '# Uncoded Terms', 'Uncoded Terms'])),
                    verification_pct=self._safe_numeric(self._safe_get(row, ['verification_pct', 'Data Verification %', 'Verification %'])),
                    reconciliation_issues=self._safe_int(self._safe_get(row, ['reconciliation_issues', '# Reconciliation Issues', 'Recon Issues'])),
                    protocol_deviations=self._safe_int(self._safe_get(row, ['protocol_deviations', '# PDs Confirmed', 'PDs Confirmed']))
                )
                
                # Calculate clean status
                patient_node.clean_status = self._calculate_clean_status(patient_node)
                patient_node.clean_percentage = self._calculate_clean_percentage(patient_node)
                patient_node.data_quality_index = self._calculate_dqi(patient_node)
                
                # Update attributes with calculated values
                patient_node.attributes['clean_status'] = patient_node.clean_status
                patient_node.attributes['clean_percentage'] = patient_node.clean_percentage
                patient_node.attributes['data_quality_index'] = patient_node.data_quality_index
                
                # Add to graph
                node_id = self.graph.add_patient(patient_node)
                self._patient_cache[subject_id] = node_id
                count += 1
                
            except Exception as e:
                logger.warning(f"Error creating patient node at row {idx}: {e}")
                continue
        
        self._stats['patients_created'] = count
        logger.info(f"Created {count} patient nodes")
        return count
    
    def _calculate_clean_status(self, patient: PatientNode) -> bool:
        """Calculate if patient has clean status"""
        return (
            patient.missing_visits == 0 and
            patient.missing_pages == 0 and
            patient.open_queries == 0 and
            patient.uncoded_terms == 0 and
            patient.reconciliation_issues == 0 and
            patient.verification_pct >= 75.0  # Relaxed threshold
        )
    
    def _calculate_clean_percentage(self, patient: PatientNode) -> float:
        """Calculate percentage towards clean status"""
        total_checks = 6
        passed = 0
        
        if patient.missing_visits == 0:
            passed += 1
        if patient.missing_pages == 0:
            passed += 1
        if patient.open_queries == 0:
            passed += 1
        if patient.uncoded_terms == 0:
            passed += 1
        if patient.reconciliation_issues == 0:
            passed += 1
        if patient.verification_pct >= 75.0:
            passed += 1
        
        return round((passed / total_checks) * 100, 1)
    
    def _calculate_dqi(self, patient: PatientNode) -> float:
        """Calculate Data Quality Index"""
        dqi = 100.0
        
        # Penalize for issues
        dqi -= min(patient.missing_visits * 5, 20)
        dqi -= min(patient.open_queries * 2, 20)
        dqi -= min(patient.uncoded_terms * 3, 15)
        dqi -= min(patient.reconciliation_issues * 10, 25)
        
        # Penalize for low verification
        if patient.verification_pct < 100:
            dqi -= (100 - patient.verification_pct) * 0.2
        
        return max(round(dqi, 1), 0)
    
    # ==================== Site Node Builder ====================
    
    def _build_site_nodes(self) -> int:
        """Build site nodes by aggregating patient data"""
        count = 0
        site_data: Dict[str, Dict] = {}
        
        # Aggregate from patient nodes
        for patient_node in self.graph.get_nodes_by_type(NodeType.PATIENT):
            attrs = patient_node.attributes
            site_id = attrs.get('site_id', 'Unknown')
            
            if site_id not in site_data:
                site_data[site_id] = {
                    'site_id': site_id,
                    'study_id': self.study_id,
                    'country': attrs.get('country', ''),
                    'region': attrs.get('region', ''),
                    'total_patients': 0,
                    'clean_patients': 0,
                    'dqi_sum': 0.0
                }
            
            site_data[site_id]['total_patients'] += 1
            if attrs.get('clean_status', False):
                site_data[site_id]['clean_patients'] += 1
            site_data[site_id]['dqi_sum'] += attrs.get('data_quality_index', 100.0)
        
        # Create site nodes with enhanced analytics
        for site_id, data in site_data.items():
            avg_dqi = data['dqi_sum'] / data['total_patients'] if data['total_patients'] > 0 else 100.0
            
            # Calculate enhanced site metrics
            site_metrics = self._calculate_enhanced_site_metrics(site_id, data)
            
            site_node = SiteNode(
                site_id=site_id,
                study_id=self.study_id,
                country=data['country'],
                region=data['region'],
                total_patients=data['total_patients'],
                clean_patients=data['clean_patients'],
                data_quality_index=round(avg_dqi, 1),
                risk_level=self._calculate_site_risk(avg_dqi, data)
            )
            
            # Add enhanced metrics to site node attributes
            site_node.attributes.update(site_metrics)
            
            node_id = self.graph.add_site(site_node)
            self._site_cache[site_id] = node_id
            count += 1
            
            # Connect patients to site
            for patient_node in self.graph.get_nodes_by_type(NodeType.PATIENT):
                if patient_node.attributes.get('site_id') == site_id:
                    self.graph.connect_patient_to_site(patient_node.node_id, node_id)
                    self._stats['edges_created'] += 1
        
        self._stats['sites_created'] = count
        logger.info(f"Created {count} site nodes with enhanced analytics")
        return count
    
    def _calculate_site_risk(self, avg_dqi: float, data: Dict) -> str:
        """Calculate site risk level"""
        if avg_dqi < 50:
            return "Critical"
        elif avg_dqi < 75:
            return "High"
        elif avg_dqi < 90:
            return "Medium"
        return "Low"
    
    def _calculate_enhanced_site_metrics(self, site_id: str, data: Dict) -> Dict[str, Any]:
        """
        Calculate sophisticated site-level graph analytics
        
        Includes connectivity metrics, risk patterns, and predictive indicators
        """
        # Get all patients for this site
        site_patients = [
            p for p in self.graph.get_nodes_by_type(NodeType.PATIENT)
            if p.attributes.get('site_id') == site_id
        ]
        
        if not site_patients:
            return {}
        
        # Calculate patient connectivity metrics
        total_connections = 0
        issue_diversity = set()
        risk_scores = []
        
        for patient in site_patients:
            # Count connections (edges) for this patient
            patient_connections = len(self.graph.graph.edges(patient.node_id))
            total_connections += patient_connections
            
            # Track issue types
            if patient.attributes.get('missing_visits', 0) > 0:
                issue_diversity.add('missing_visits')
            if patient.attributes.get('open_queries', 0) > 0:
                issue_diversity.add('open_queries')
            if patient.attributes.get('uncoded_terms', 0) > 0:
                issue_diversity.add('uncoded_terms')
            if patient.attributes.get('protocol_deviations', 0) > 0:
                issue_diversity.add('protocol_deviations')
            
            # Calculate patient risk score
            risk_score = (
                patient.attributes.get('missing_visits', 0) * 2 +
                patient.attributes.get('open_queries', 0) * 1.5 +
                patient.attributes.get('uncoded_terms', 0) * 1 +
                patient.attributes.get('protocol_deviations', 0) * 3
            )
            risk_scores.append(risk_score)
        
        # Calculate site-level metrics
        avg_connections = total_connections / len(site_patients) if site_patients else 0
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        max_risk_score = max(risk_scores) if risk_scores else 0
        
        # Risk pattern analysis
        high_risk_patients = sum(1 for score in risk_scores if score > 5)
        critical_patients = sum(1 for score in risk_scores if score > 10)
        
        # Connectivity analysis
        well_connected_patients = sum(1 for patient in site_patients 
                                    if len(self.graph.graph.edges(patient.node_id)) > avg_connections)
        
        return {
            'avg_patient_connections': round(avg_connections, 2),
            'issue_diversity_score': len(issue_diversity),
            'avg_risk_score': round(avg_risk_score, 2),
            'max_risk_score': round(max_risk_score, 2),
            'high_risk_patients': high_risk_patients,
            'critical_patients': critical_patients,
            'well_connected_patients': well_connected_patients,
            'connectivity_ratio': round(well_connected_patients / len(site_patients), 2) if site_patients else 0,
            'risk_patterns': list(issue_diversity),
            'site_network_density': round(total_connections / (len(site_patients) ** 2), 4) if site_patients else 0
        }
    
    # ==================== Visit/Event Node Builder ====================
    
    def _build_visit_nodes(self, visit_df: pd.DataFrame) -> int:
        """Build event/visit nodes from Visit Projection Tracker"""
        count = 0
        
        subject_cols = ['subject_id', 'Subject ID', 'SubjectID']
        visit_cols = ['visit_name', 'Visit Name', 'VisitName', 'Visit']
        date_cols = ['projected_date', 'Projected Date', 'ProjectedDate']
        days_cols = ['days_outstanding', '# Days Outstanding', 'Days Outstanding']
        
        for idx, row in visit_df.iterrows():
            try:
                subject_id = self._safe_get(row, subject_cols)
                if not subject_id or pd.isna(subject_id):
                    continue
                
                subject_id = str(subject_id).strip()
                
                # Skip if patient not in graph
                patient_node_id = self._patient_cache.get(subject_id)
                if not patient_node_id:
                    continue
                
                visit_name = str(self._safe_get(row, visit_cols, f'Visit_{idx}'))
                days_outstanding = self._safe_int(self._safe_get(row, days_cols))
                
                # Parse projected date
                projected_date = None
                date_val = self._safe_get(row, date_cols)
                if date_val and pd.notna(date_val):
                    try:
                        if isinstance(date_val, datetime):
                            projected_date = date_val
                        else:
                            projected_date = pd.to_datetime(date_val)
                    except:
                        pass
                
                event_id = self._generate_id('visit', subject_id, visit_name)
                
                event_node = EventNode(
                    event_id=event_id,
                    subject_id=subject_id,
                    study_id=self.study_id,
                    visit_name=visit_name,
                    projected_date=projected_date,
                    days_outstanding=days_outstanding,
                    status="Overdue" if days_outstanding > 0 else "On Track",
                    is_missing=days_outstanding > 30
                )
                
                # Add to graph
                visit_node_id = self.graph.add_event(event_node)
                
                # Connect to patient
                self.graph.connect_patient_to_visit(
                    patient_node_id,
                    visit_node_id,
                    {'days_outstanding': days_outstanding}
                )
                self._stats['edges_created'] += 1
                count += 1
                
            except Exception as e:
                logger.warning(f"Error creating visit node at row {idx}: {e}")
                continue
        
        self._stats['visits_created'] = count
        logger.info(f"Created {count} visit nodes")
        return count
    
    # ==================== SAE Node Builder ====================
    
    def _build_sae_nodes(self, sae_df: pd.DataFrame) -> int:
        """Build SAE nodes from SAE Dashboard"""
        count = 0
        
        subject_cols = ['subject_id', 'Subject ID', 'SubjectID', 'Patient ID']
        site_cols = ['site_id', 'Site ID', 'SiteID', 'Site']
        review_cols = ['review_status', 'Review Status', 'ReviewStatus', 'DM Review']
        action_cols = ['action_status', 'Action Status', 'ActionStatus']
        discrepancy_cols = ['discrepancy_id', 'Discrepancy ID', 'DiscrepancyID']
        
        for idx, row in sae_df.iterrows():
            try:
                subject_id = self._safe_get(row, subject_cols)
                if not subject_id or pd.isna(subject_id):
                    continue
                
                subject_id = str(subject_id).strip()
                
                # Skip if patient not in graph
                patient_node_id = self._patient_cache.get(subject_id)
                if not patient_node_id:
                    continue
                
                sae_id = self._generate_id('sae', subject_id, idx)
                review_status = str(self._safe_get(row, review_cols, 'Pending'))
                action_status = str(self._safe_get(row, action_cols, 'Open'))
                
                sae_node = SAENode(
                    sae_id=sae_id,
                    subject_id=subject_id,
                    study_id=self.study_id,
                    site_id=str(self._safe_get(row, site_cols, '')),
                    review_status=review_status,
                    action_status=action_status,
                    requires_reconciliation='pending' in review_status.lower() or 'open' in action_status.lower(),
                    discrepancy_id=str(self._safe_get(row, discrepancy_cols, ''))
                )
                
                # Add to graph
                sae_node_id = self.graph.add_sae(sae_node)
                
                # Connect to patient with edge properties
                self.graph.connect_patient_to_sae(
                    patient_node_id,
                    sae_node_id,
                    review_status=review_status,
                    action_status=action_status
                )
                self._stats['edges_created'] += 1
                count += 1
                
            except Exception as e:
                logger.warning(f"Error creating SAE node at row {idx}: {e}")
                continue
        
        self._stats['saes_created'] = count
        logger.info(f"Created {count} SAE nodes")
        return count
    
    # ==================== Coding Term Node Builder ====================
    
    def _build_coding_nodes(self, coding_df: pd.DataFrame, dictionary: str) -> int:
        """Build coding term nodes from MedDRA or WHODRA reports"""
        count = 0
        
        subject_cols = ['subject_id', 'Subject ID', 'SubjectID']
        verbatim_cols = ['verbatim_term', 'Verbatim Term', 'VerbatimTerm', 'Verbatim']
        status_cols = ['coding_status', 'Coding Status', 'CodingStatus', 'Status']
        coded_cols = ['coded_term', 'Coded Term', 'CodedTerm', 'LLT', 'PT']
        context_cols = ['context', 'Context', 'Form', 'Source']
        
        for idx, row in coding_df.iterrows():
            try:
                subject_id = self._safe_get(row, subject_cols)
                if not subject_id or pd.isna(subject_id):
                    continue
                
                subject_id = str(subject_id).strip()
                
                # Skip if patient not in graph
                patient_node_id = self._patient_cache.get(subject_id)
                if not patient_node_id:
                    continue
                
                verbatim_term = str(self._safe_get(row, verbatim_cols, ''))
                coding_status = str(self._safe_get(row, status_cols, 'UnCoded'))
                
                # Only create nodes for uncoded terms (these are the "issues")
                if 'coded' in coding_status.lower() and 'uncoded' not in coding_status.lower():
                    continue
                
                term_id = self._generate_id('term', subject_id, verbatim_term[:20], idx)
                
                coding_node = CodingTermNode(
                    term_id=term_id,
                    subject_id=subject_id,
                    study_id=self.study_id,
                    verbatim_term=verbatim_term,
                    coded_term=str(self._safe_get(row, coded_cols, '')),
                    coding_dictionary=dictionary,
                    coding_status='UnCoded',
                    context=str(self._safe_get(row, context_cols, '')),
                    requires_review=True
                )
                
                # Add to graph
                term_node_id = self.graph.add_coding_term(coding_node)
                
                # Connect to patient
                self.graph.connect_patient_to_coding_issue(
                    patient_node_id,
                    term_node_id,
                    verbatim_term=verbatim_term,
                    coding_status='UnCoded'
                )
                self._stats['edges_created'] += 1
                count += 1
                
            except Exception as e:
                logger.warning(f"Error creating coding node at row {idx}: {e}")
                continue
        
        self._stats['coding_terms_created'] += count
        logger.info(f"Created {count} {dictionary} coding term nodes")
        return count
    
    # ==================== EDRR/Reconciliation Node Builder ====================
    
    def _build_edrr_nodes(self, edrr_df: pd.DataFrame) -> int:
        """Build reconciliation discrepancy nodes from EDRR data"""
        count = 0
        
        subject_cols = ['subject_id', 'Subject ID', 'SubjectID']
        
        for idx, row in edrr_df.iterrows():
            try:
                subject_id = self._safe_get(row, subject_cols)
                if not subject_id or pd.isna(subject_id):
                    continue
                
                subject_id = str(subject_id).strip()
                
                # Skip if patient not in graph
                patient_node_id = self._patient_cache.get(subject_id)
                if not patient_node_id:
                    continue
                
                query_id = self._generate_id('edrr', subject_id, idx)
                
                discrepancy_node = DiscrepancyNode(
                    query_id=query_id,
                    subject_id=subject_id,
                    study_id=self.study_id,
                    query_type="Reconciliation",
                    status='Open',
                    severity="Critical"
                )
                
                # Add to graph
                query_node_id = self.graph.add_discrepancy(discrepancy_node)
                
                # Connect to patient
                self.graph.connect_patient_to_query(
                    patient_node_id,
                    query_node_id,
                    query_type="Reconciliation",
                    status="Open"
                )
                self._stats['edges_created'] += 1
                count += 1
                
            except Exception as e:
                logger.warning(f"Error creating EDRR node at row {idx}: {e}")
                continue
        
        self._stats['queries_created'] += count
        logger.info(f"Created {count} EDRR/reconciliation nodes")
        return count
    
    # ==================== Utility Methods ====================
    
    def get_build_statistics(self) -> Dict[str, int]:
        """Get statistics from the build process"""
        return self._stats.copy()
    
    def get_graph(self) -> ClinicalKnowledgeGraph:
        """Get the built knowledge graph"""
        return self.graph
    
    def save_graph(self, filepath: str) -> bool:
        """Save the knowledge graph to disk"""
        return self.graph.save(filepath)


def build_knowledge_graph_from_study(
    study_data: Dict[str, pd.DataFrame],
    study_id: str
) -> Tuple[ClinicalKnowledgeGraph, Dict[str, int]]:
    """
    Convenience function to build knowledge graph from study data
    
    Args:
        study_data: Dictionary of DataFrames
        study_id: Study identifier
        
    Returns:
        Tuple of (ClinicalKnowledgeGraph, build_statistics)
    """
    builder = ClinicalGraphBuilder(study_id=study_id)
    graph = builder.build_from_study_data(study_data, study_id)
    return graph, builder.get_build_statistics()
