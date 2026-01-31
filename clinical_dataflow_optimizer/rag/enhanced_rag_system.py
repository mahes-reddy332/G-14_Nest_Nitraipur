"""
Enhanced RAG Pipeline with Knowledge Graph Integration
======================================================

One-time ingestion of CSV data into knowledge graph, with agent-integrated query processing.

Architecture:
1. KnowledgeGraphBuilder: One-time ingestion and graph construction
2. EnhancedRAGPipeline: Query processing using knowledge graph
3. AgentRAGIntegration: Agent-enhanced response generation
4. QueryRouter: Intelligent routing based on query type and requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import json
import pickle
import hashlib
from enum import Enum

from graph.knowledge_graph import (
    ClinicalKnowledgeGraph, NodeType, EdgeType,
    PatientNode, SAENode, EventNode, CodingTermNode
)
from nlq.conversational_engine import ConversationalEngine
from agents.agent_framework import SupervisorAgent, ReconciliationAgent, CodingAgent, SiteLiaisonAgent
from core.longcat_integration import longcat_client
from config.settings import DEFAULT_AGENT_CONFIG

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the RAG system can handle"""
    FACTUAL = "factual"           # Direct data retrieval
    ANALYTICAL = "analytical"     # Trends, patterns, correlations
    DIAGNOSTIC = "diagnostic"     # Problem identification and root cause
    PREDICTIVE = "predictive"     # Forecasting and risk assessment
    PRESCRIPTIVE = "prescriptive" # Recommendations and actions
    EXPLANATORY = "explanatory"   # Why questions and interpretations


class KnowledgeGraphBuilder:
    """
    One-time ingestion system that builds and maintains the knowledge graph
    from all CSV data sources.
    """

    def __init__(self, data_path: Path, graph_path: Optional[Path] = None):
        self.data_path = Path(data_path)
        self.graph_path = graph_path or (data_path.parent / "knowledge_graph.pkl")
        self.knowledge_graph = ClinicalKnowledgeGraph()
        self.ingestion_timestamp = None
        self.data_hash = None

    def should_reingest(self) -> bool:
        """Check if data has changed and requires reingestion"""
        if not self.graph_path.exists():
            return True

        # Calculate hash of all CSV files
        current_hash = self._calculate_data_hash()
        if current_hash != self.data_hash:
            return True

        return False

    def _calculate_data_hash(self) -> str:
        """Calculate MD5 hash of all CSV files to detect changes"""
        hasher = hashlib.md5()
        csv_files = list(self.data_path.rglob("*.csv")) + list(self.data_path.rglob("*.xlsx"))

        for file_path in sorted(csv_files):
            hasher.update(str(file_path).encode())
            hasher.update(file_path.stat().st_mtime_ns.to_bytes(8, 'big'))

        return hasher.hexdigest()

    def ingest_all_data(self, force_reingest: bool = False) -> bool:
        """
        Perform one-time ingestion of all CSV data into knowledge graph.
        Returns True if ingestion was performed, False if using cached graph.
        """
        if not force_reingest and not self.should_reingest():
            logger.info("Data unchanged, loading cached knowledge graph")
            self._load_cached_graph()
            return False

        logger.info("Starting one-time data ingestion into knowledge graph")

        # Clear existing graph
        self.knowledge_graph = ClinicalKnowledgeGraph()

        # Discover all studies
        study_dirs = [d for d in self.data_path.iterdir() if d.is_dir() and d.name.startswith("Study")]
        logger.info(f"Found {len(study_dirs)} study directories")

        total_patients = 0
        total_relationships = 0

        for study_dir in study_dirs:
            study_id = study_dir.name
            logger.info(f"Processing study: {study_id}")

            # Load all data files for this study
            study_data = self._load_study_data(study_dir)

            if study_data:
                patients_added, relationships_added = self._build_study_graph(study_data, study_id)
                total_patients += patients_added
                total_relationships += relationships_added

        # Save ingestion metadata
        self.ingestion_timestamp = datetime.now()
        self.data_hash = self._calculate_data_hash()

        # Cache the graph
        self._save_graph_cache()

        logger.info(f"Knowledge graph built successfully:")
        logger.info(f"  - Total patients: {total_patients}")
        logger.info(f"  - Total relationships: {total_relationships}")
        logger.info(f"  - Graph saved to: {self.graph_path}")

        return True

    def _load_study_data(self, study_dir: Path) -> Optional[Dict[str, pd.DataFrame]]:
        """Load all data files for a specific study"""
        study_data = {}

        # Define expected file patterns with more flexible matching
        file_mappings = {
            'cpid_data': ['CPID_EDC_Metrics', 'CPID_EDC'],
            'sae_data': ['eSAE', 'SAE', 'Safety'],
            'visit_data': ['Visit', 'Projection', 'Tracker'],
            'coding_data': ['GlobalCodingReport', 'Coding'],
            'audit_data': ['Inactivated', 'Audit']
        }

        # Get all Excel files in the directory
        excel_files = list(study_dir.glob("*.xlsx")) + list(study_dir.glob("*.xls"))

        for file_path in excel_files:
            file_name = file_path.name.lower()

            for data_type, keywords in file_mappings.items():
                if any(keyword.lower() in file_name for keyword in keywords):
                    try:
                        df = pd.read_excel(file_path)
                        study_data[data_type] = df
                        logger.debug(f"Loaded {data_type}: {file_path.name} ({len(df)} rows)")
                        break  # Found a match, move to next file
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path.name}: {e}")

        return study_data if study_data else None

    def _build_study_graph(self, study_data: Dict[str, pd.DataFrame], study_id: str) -> Tuple[int, int]:
        """Build knowledge graph for a specific study"""
        patients_added = 0
        relationships_added = 0

        # First pass: Create all nodes
        logger.info(f"Creating nodes for study {study_id}")

        # Create SAE nodes
        if 'sae_data' in study_data:
            sae_df = study_data['sae_data']
            for _, sae_row in sae_df.iterrows():
                try:
                    # SAE ID should be just the discrepancy ID - node class adds prefix
                    sae_id = str(sae_row.get('Discrepancy ID', 'UNK'))
                    sae_node = SAENode(
                        sae_id=sae_id,
                        subject_id=str(sae_row.get('Patient ID', '')),
                        study_id=study_id,
                        site_id=str(sae_row.get('Site', '')),
                        event_type=sae_row.get('Form Name', ''),
                        review_status=sae_row.get('Review Status', 'Pending'),
                        action_status=sae_row.get('Action Status', 'Open'),
                        discrepancy_id=str(sae_row.get('Discrepancy ID', ''))
                    )
                    self.knowledge_graph.add_sae(sae_node)
                except Exception as e:
                    logger.warning(f"Error creating SAE node: {e}")

        # Create Event nodes
        if 'visit_data' in study_data:
            visit_df = study_data['visit_data']
            for _, visit_row in visit_df.iterrows():
                try:
                    # Try multiple column names for patient ID
                    subject_id = None
                    for col in ['Subject', 'Subject_ID', 'SUBJECT_ID', 'Patient ID']:
                        if col in visit_row and pd.notna(visit_row[col]):
                            subject_id = str(visit_row[col]).strip()
                            # Normalize subject ID by removing "Subject " prefix if present
                            if subject_id.startswith('Subject '):
                                subject_id = subject_id.replace('Subject ', '')
                            break

                    if not subject_id:
                        continue

                    # Event ID should be just the visit name - node class adds subject_id
                    event_id = str(visit_row.get('Visit', 'UNK'))
                    event_node = EventNode(
                        event_id=event_id,
                        subject_id=subject_id,
                        study_id=study_id,
                        visit_name=visit_row.get('Visit', ''),
                        projected_date=pd.to_datetime(visit_row.get('Projected Date'), errors='coerce'),
                        days_outstanding=visit_row.get('# Days Outstanding', 0),
                        status='Overdue' if visit_row.get('# Days Outstanding', 0) > 0 else 'On Track'
                    )
                    self.knowledge_graph.add_event(event_node)
                except Exception as e:
                    logger.warning(f"Error creating Event node: {e}")

        # Create Coding Term nodes
        if 'coding_data' in study_data:
            coding_df = study_data['coding_data']
            for _, coding_row in coding_df.iterrows():
                try:
                    # Try multiple column names for patient ID
                    subject_id = None
                    for col in ['Subject_ID', 'Subject', 'SUBJECT_ID', 'Patient ID']:
                        if col in coding_row and pd.notna(coding_row[col]):
                            subject_id = str(coding_row[col]).strip()
                            # Normalize subject ID by removing "Subject " prefix if present
                            if subject_id.startswith('Subject '):
                                subject_id = subject_id.replace('Subject ', '')
                            break

                    if not subject_id:
                        continue

                    term_id = f"{subject_id}_{hash(coding_row.get('Term', ''))}"
                    coding_node = CodingTermNode(
                        term_id=term_id,
                        subject_id=subject_id,
                        study_id=study_id,
                        verbatim_term=coding_row.get('Term', ''),
                        coded_term=coding_row.get('Code', ''),
                        coding_dictionary=coding_row.get('Dictionary', 'MedDRA'),
                        coding_status=coding_row.get('Coding_Status', 'UnCoded')
                    )
                    self.knowledge_graph.add_coding_term(coding_node)
                except Exception as e:
                    logger.warning(f"Error creating Coding Term node: {e}")

        # Second pass: Create patient nodes and relationships
        logger.info(f"Creating patient nodes and relationships for study {study_id}")

        # Extract patient information
        if 'cpid_data' in study_data:
            patients_df = study_data['cpid_data']

            for _, patient_row in patients_df.iterrows():
                try:
                    # Try multiple possible column names for patient ID
                    patient_id = None
                    for col in ['Subject ID', 'Subject_ID', 'SUBJECT_ID', 'Subject']:
                        if col in patient_row and pd.notna(patient_row[col]):
                            patient_id = str(patient_row[col]).strip()
                            # Normalize patient ID by removing "Subject " prefix if present
                            if patient_id.startswith('Subject '):
                                patient_id = patient_id.replace('Subject ', '')
                            break

                    if not patient_id or patient_id == 'nan':
                        continue

                    # Try multiple possible column names for site ID
                    site_id = ''
                    for col in ['Site ID', 'Site_ID', 'SITE_ID', 'Site']:
                        if col in patient_row and pd.notna(patient_row[col]):
                            site_id = str(patient_row[col]).strip()
                            break

                    if site_id == 'nan':
                        site_id = ''

                    # Add patient node
                    patient_node = PatientNode(
                        subject_id=patient_id,
                        site_id=site_id,
                        study_id=study_id,
                        country=patient_row.get('Country'),
                        region=patient_row.get('Region'),
                        status=patient_row.get('Subject Status (Source: PRIMARY Form)'),
                        clean_status=False,  # Will be calculated later
                        clean_percentage=0.0,  # Will be calculated later
                        data_quality_index=100.0  # Will be calculated later
                    )
                    self.knowledge_graph.add_patient(patient_node)
                    patients_added += 1

                    # Add relationships
                    relationships_added += self._add_patient_relationships(
                        patient_id, site_id, study_id, study_data
                    )

                except Exception as e:
                    logger.warning(f"Error processing patient {patient_id}: {e}")
                    continue

        return patients_added, relationships_added

    def _add_patient_relationships(self, patient_id: str, site_id: str, study_id: str,
                                 study_data: Dict[str, pd.DataFrame]) -> int:
        """Add all relationships for a patient"""
        relationships = 0
        
        # Construct the proper patient node ID (matching PatientNode.__post_init__)
        patient_node_id = f"patient_{study_id}_{patient_id}"

        # Add site relationship (if site exists)
        if site_id:
            # For now, we'll create a simple edge - in a full implementation we'd create site nodes
            pass  # Sites would need their own nodes

        # Add study relationship (if study exists)
        if study_id:
            # For now, we'll create a simple edge - in a full implementation we'd create study nodes
            pass  # Studies would need their own nodes

        # Add SAE relationships
        if 'sae_data' in study_data:
            sae_df = study_data['sae_data']
            # Try multiple column names for patient ID
            patient_col = None
            for col in ['Patient ID', 'Patient_ID', 'PATIENT_ID', 'Subject', 'Subject_ID']:
                if col in sae_df.columns:
                    patient_col = col
                    break

            if patient_col:
                # Normalize patient IDs in the dataframe for comparison
                normalized_patient_ids = sae_df[patient_col].astype(str).str.strip()
                normalized_patient_ids = normalized_patient_ids.str.replace('^Subject ', '', regex=True)
                patient_saes = sae_df[normalized_patient_ids == patient_id]

                for _, sae_row in patient_saes.iterrows():
                    sae_node_id = f"sae_{study_id}_{sae_row.get('Discrepancy ID', 'UNK')}"
                    # Check if SAE node exists before connecting
                    if sae_node_id in self.knowledge_graph.node_index:
                        if self.knowledge_graph.connect_patient_to_sae(
                            patient_id=patient_node_id,
                            sae_id=sae_node_id,
                            review_status=sae_row.get('Review Status', 'Pending'),
                            action_status=sae_row.get('Action Status', 'Open')
                        ):
                            relationships += 1

        # Add visit relationships
        if 'visit_data' in study_data:
            visit_df = study_data['visit_data']
            # Try multiple column names for patient ID
            patient_col = None
            for col in ['Subject', 'Subject_ID', 'SUBJECT_ID', 'Patient ID']:
                if col in visit_df.columns:
                    patient_col = col
                    break

            if patient_col:
                # Normalize patient IDs in the data for comparison
                visit_df_normalized = visit_df.copy()
                visit_df_normalized[patient_col] = visit_df_normalized[patient_col].astype(str).apply(
                    lambda x: x.replace('Subject ', '') if x.startswith('Subject ') else x
                )
                
                patient_visits = visit_df_normalized[visit_df_normalized[patient_col].astype(str) == patient_id]

                for _, visit_row in patient_visits.iterrows():
                    visit_name = visit_row.get('Visit', 'UNK')
                    event_node_id = f"event_{study_id}_{patient_id}_{visit_name}"
                    # Check if event node exists before connecting
                    if event_node_id in self.knowledge_graph.node_index:
                        if self.knowledge_graph.connect_patient_to_visit(
                            patient_id=patient_node_id,
                            event_id=event_node_id
                        ):
                            relationships += 1

        # Add coding relationships
        if 'coding_data' in study_data:
            coding_df = study_data['coding_data']
            # Try multiple column names for patient ID
            patient_col = None
            for col in ['Subject_ID', 'Subject', 'SUBJECT_ID', 'Patient ID']:
                if col in coding_df.columns:
                    patient_col = col
                    break

            if patient_col:
                # Normalize patient IDs in the data for comparison
                coding_df_normalized = coding_df.copy()
                coding_df_normalized[patient_col] = coding_df_normalized[patient_col].astype(str).apply(
                    lambda x: x.replace('Subject ', '') if x.startswith('Subject ') else x
                )
                
                patient_coding = coding_df_normalized[coding_df_normalized[patient_col].astype(str) == patient_id]

                for _, coding_row in patient_coding.iterrows():
                    term_hash = hash(coding_row.get('Term', ''))
                    coding_node_id = f"term_{study_id}_{patient_id}_{term_hash}"
                    # Check if coding node exists before connecting
                    if coding_node_id in self.knowledge_graph.node_index:
                        if self.knowledge_graph.connect_patient_to_coding_issue(
                            patient_id=patient_node_id,
                            term_id=coding_node_id,
                            verbatim_term=coding_row.get('Term', ''),
                            coding_status=coding_row.get('Coding_Status', 'UnCoded')
                        ):
                            relationships += 1

        return relationships

    def _save_graph_cache(self):
        """Save the knowledge graph to disk for future use"""
        cache_data = {
            'graph': self.knowledge_graph,
            'ingestion_timestamp': self.ingestion_timestamp,
            'data_hash': self.data_hash,
            'metadata': {
                'node_count': len(self.knowledge_graph.graph.nodes()),
                'edge_count': len(self.knowledge_graph.graph.edges()),
                'node_types': self._get_node_type_counts(),
                'edge_types': self._get_edge_type_counts()
            }
        }

        with open(self.graph_path, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"Knowledge graph cached to {self.graph_path}")

    def _load_cached_graph(self):
        """Load cached knowledge graph from disk"""
        try:
            with open(self.graph_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.knowledge_graph = cache_data['graph']
            self.ingestion_timestamp = cache_data['ingestion_timestamp']
            self.data_hash = cache_data['data_hash']

            metadata = cache_data['metadata']
            logger.info(f"Loaded cached knowledge graph:")
            logger.info(f"  - Nodes: {metadata['node_count']}")
            logger.info(f"  - Edges: {metadata['edge_count']}")
            logger.info(f"  - Created: {self.ingestion_timestamp}")

        except Exception as e:
            logger.error(f"Failed to load cached graph: {e}")
            # Force reingestion
            self.ingest_all_data(force_reingest=True)

    def _get_node_type_counts(self) -> Dict[str, int]:
        """Get count of each node type in the graph"""
        counts = {}
        for node_id, node_data in self.knowledge_graph.graph.nodes(data=True):
            node_type = node_data.get('type', 'Unknown')
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts

    def _get_edge_type_counts(self) -> Dict[str, int]:
        """Get count of each edge type in the graph"""
        counts = {}
        for _, _, edge_data in self.knowledge_graph.graph.edges(data=True):
            edge_type = edge_data.get('type', 'Unknown')
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts


@dataclass
class RAGContext:
    """Context information for RAG processing"""
    query_type: QueryType
    entities_mentioned: List[str]
    metrics_requested: List[str]
    time_constraints: Dict[str, Any]
    filters_applied: Dict[str, Any]
    knowledge_graph_context: Dict[str, Any]
    agent_insights: Dict[str, Any] = field(default_factory=dict)


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline that uses the knowledge graph for query processing
    and integrates with AI agents for intelligent response generation.
    """

    def __init__(self, knowledge_graph_builder: KnowledgeGraphBuilder):
        self.kg_builder = knowledge_graph_builder
        self.conversational_engine = ConversationalEngine()

        # Load data sources into conversational engine
        self._load_data_into_conversational_engine()

        # Initialize agents
        self.supervisor_agent = SupervisorAgent()
        self.reconciliation_agent = ReconciliationAgent()
        self.coding_agent = CodingAgent()
        self.liaison_agent = SiteLiaisonAgent()

    def _load_data_into_conversational_engine(self):
        """Load data sources from knowledge graph builder into conversational engine"""
        try:
            # Load data from the first study as an example
            study_dirs = [d for d in self.kg_builder.data_path.iterdir() if d.is_dir() and d.name.startswith("Study")]
            if study_dirs:
                study_data = self.kg_builder._load_study_data(study_dirs[0])
                if study_data:
                    # Load the data into conversational engine
                    self.conversational_engine.load_data(study_data, self.kg_builder.knowledge_graph.graph)
                    logger.info(f"Loaded data sources into conversational engine: {list(study_data.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load data into conversational engine: {e}")

    def process_query(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query using the enhanced RAG pipeline.

        Args:
            user_query: Natural language query from user
            context: Optional context information (conversation history, user preferences, etc.)

        Returns:
            Comprehensive response with answer, insights, and agent recommendations
        """
        start_time = datetime.now()

        try:
            # Step 1: Analyze query and extract context
            rag_context = self._analyze_query(user_query, context)

            # Step 2: Query knowledge graph for relevant information
            graph_results = self._query_knowledge_graph(rag_context)

            # Step 3: Get agent insights and recommendations
            agent_insights = self._get_agent_insights(rag_context, graph_results)

            # Step 4: Generate comprehensive response using conversational engine
            response = self._generate_response(user_query, rag_context, graph_results, agent_insights)

            # Step 5: Add agent-enhanced recommendations
            final_response = self._enhance_with_agent_recommendations(response, agent_insights)

            # Add metadata
            final_response['metadata'] = {
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'query_type': rag_context.query_type.value,
                'knowledge_graph_nodes_used': len(graph_results.get('nodes', [])),
                'agents_consulted': list(agent_insights.keys()),
                'ingestion_timestamp': self.kg_builder.ingestion_timestamp.isoformat() if self.kg_builder.ingestion_timestamp else None
            }

            return final_response

        except Exception as e:
            logger.error(f"Error processing query '{user_query}': {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': "I apologize, but I encountered an error while processing your query. Please try rephrasing your question.",
                'metadata': {
                    'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'error_type': type(e).__name__
                }
            }

    def _analyze_query(self, user_query: str, context: Optional[Dict[str, Any]]) -> RAGContext:
        """Analyze the query to extract context and requirements"""
        # Use conversational engine to parse the query
        parsed_query = self.conversational_engine.parser.parse(user_query)

        # Determine query type
        query_type = self._classify_query_type(user_query, parsed_query)

        # Extract entities and metrics
        entities_mentioned = self._extract_entities(parsed_query)
        metrics_requested = self._extract_metrics(parsed_query)

        # Extract time constraints
        time_constraints = self._extract_time_constraints(parsed_query)

        # Extract filters
        filters_applied = self._extract_filters(parsed_query)

        # Get knowledge graph context
        knowledge_graph_context = self._get_graph_context(entities_mentioned, query_type)

        return RAGContext(
            query_type=query_type,
            entities_mentioned=entities_mentioned,
            metrics_requested=metrics_requested,
            time_constraints=time_constraints,
            filters_applied=filters_applied,
            knowledge_graph_context=knowledge_graph_context
        )

    def _classify_query_type(self, user_query: str, parsed_query) -> QueryType:
        """Classify the type of query"""
        query_lower = user_query.lower()

        # Check for prescriptive queries (recommendations, actions)
        if any(word in query_lower for word in ['should', 'recommend', 'suggest', 'what to do', 'how to']):
            return QueryType.PRESCRIPTIVE

        # Check for predictive queries
        if any(word in query_lower for word in ['will', 'predict', 'forecast', 'risk of', 'likely to']):
            return QueryType.PREDICTIVE

        # Check for diagnostic queries
        if any(word in query_lower for word in ['why', 'cause', 'reason', 'problem', 'issue']):
            return QueryType.DIAGNOSTIC

        # Check for analytical queries
        if any(word in query_lower for word in ['trend', 'pattern', 'correlation', 'compare', 'analysis']):
            return QueryType.ANALYTICAL

        # Check for explanatory queries
        if any(word in query_lower for word in ['explain', 'what is', 'how does', 'meaning']):
            return QueryType.EXPLANATORY

        # Default to factual
        return QueryType.FACTUAL

    def _extract_entities(self, parsed_query) -> List[str]:
        """Extract entity mentions from parsed query"""
        entities = []

        # Extract from query filters and entities
        if hasattr(parsed_query, 'entities'):
            for entity in parsed_query.entities:
                if hasattr(entity, 'type') and hasattr(entity, 'value'):
                    entities.append(f"{entity.type}:{entity.value}")

        return entities

    def _extract_metrics(self, parsed_query) -> List[str]:
        """Extract metrics requested from parsed query"""
        metrics = []

        if hasattr(parsed_query, 'metrics'):
            for metric in parsed_query.metrics:
                if hasattr(metric, 'type'):
                    metrics.append(metric.type.value)

        return metrics

    def _extract_time_constraints(self, parsed_query) -> Dict[str, Any]:
        """Extract time constraints from parsed query"""
        time_constraints = {}

        if hasattr(parsed_query, 'time_constraints'):
            for constraint in parsed_query.time_constraints:
                if hasattr(constraint, 'field') and hasattr(constraint, 'operator') and hasattr(constraint, 'value'):
                    time_constraints[constraint.field] = {
                        'operator': constraint.operator,
                        'value': constraint.value
                    }

        return time_constraints

    def _extract_filters(self, parsed_query) -> Dict[str, Any]:
        """Extract filters from parsed query"""
        filters = {}

        if hasattr(parsed_query, 'filters'):
            for filter_obj in parsed_query.filters:
                if hasattr(filter_obj, 'field') and hasattr(filter_obj, 'value'):
                    filters[filter_obj.field] = filter_obj.value

        return filters

    def _get_graph_context(self, entities: List[str], query_type: QueryType) -> Dict[str, Any]:
        """Get relevant context from knowledge graph"""
        context = {
            'relevant_nodes': [],
            'central_entities': [],
            'relationship_patterns': []
        }

        # Find central entities in the graph
        for entity_spec in entities:
            if ':' in entity_spec:
                entity_type, entity_value = entity_spec.split(':', 1)

                # Search for nodes of this type with this value
                matching_nodes = []
                for node_id, node_data in self.kg_builder.knowledge_graph.graph.nodes(data=True):
                    if (node_data.get('type') == entity_type and
                        str(node_data.get('id', '')).lower() == entity_value.lower()):
                        matching_nodes.append((node_id, node_data))

                if matching_nodes:
                    context['central_entities'].extend(matching_nodes[:5])  # Top 5 matches

        return context

    def _query_knowledge_graph(self, rag_context: RAGContext) -> Dict[str, Any]:
        """Query the knowledge graph for relevant information"""
        results = {
            'nodes': [],
            'relationships': [],
            'subgraphs': [],
            'statistics': {}
        }

        # Query based on context
        if rag_context.knowledge_graph_context['central_entities']:
            # Start from central entities and explore
            central_nodes = rag_context.knowledge_graph_context['central_entities']

            for node_id, node_data in central_nodes:
                # Get connected nodes and relationships
                connected = self._get_connected_entities(node_id, max_depth=2)
                results['nodes'].extend(connected['nodes'])
                results['relationships'].extend(connected['relationships'])

        # Remove duplicates
        results['nodes'] = list(set(results['nodes']))
        results['relationships'] = list(set(results['relationships']))

        # Calculate statistics
        results['statistics'] = {
            'total_nodes': len(results['nodes']),
            'total_relationships': len(results['relationships']),
            'node_types': self._count_node_types(results['nodes']),
            'relationship_types': self._count_relationship_types(results['relationships'])
        }

        return results

    def _get_connected_entities(self, node_id: str, max_depth: int = 2) -> Dict[str, List]:
        """Get entities connected to a central node"""
        nodes = set([node_id])
        relationships = set()

        current_level = {node_id}
        for depth in range(max_depth):
            next_level = set()

            for current_node in current_level:
                # Get all neighbors
                neighbors = list(self.kg_builder.knowledge_graph.graph.neighbors(current_node))
                next_level.update(neighbors)

                # Get relationships
                for neighbor in neighbors:
                    edge_data = self.kg_builder.knowledge_graph.graph.get_edge_data(current_node, neighbor)
                    if edge_data:
                        relationships.add((current_node, neighbor, edge_data.get('type', 'Unknown')))

            nodes.update(next_level)
            current_level = next_level

        return {
            'nodes': list(nodes),
            'relationships': list(relationships)
        }

    def _count_node_types(self, nodes: List[str]) -> Dict[str, int]:
        """Count node types in a list of nodes"""
        counts = {}
        for node_id in nodes:
            node_data = self.kg_builder.knowledge_graph.graph.nodes.get(node_id, {})
            node_type = node_data.get('type', 'Unknown')
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts

    def _count_relationship_types(self, relationships: List[Tuple]) -> Dict[str, int]:
        """Count relationship types"""
        counts = {}
        for _, _, rel_type in relationships:
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts

    def _get_agent_insights(self, rag_context: RAGContext, graph_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights from relevant agents"""
        agent_insights = {}

        # Route to appropriate agents based on query type and content
        if rag_context.query_type in [QueryType.DIAGNOSTIC, QueryType.PRESCRIPTIVE]:
            # Use supervisor agent for complex analysis
            try:
                supervisor_insights = self.supervisor_agent.analyze_cross_study_patterns(
                    graph_results.get('nodes', [])
                )
                agent_insights['supervisor'] = supervisor_insights
            except Exception as e:
                logger.warning(f"Supervisor agent error: {e}")

        if any('SAE' in str(node) for node in graph_results.get('nodes', [])):
            # Use reconciliation agent for safety data
            try:
                reconciliation_insights = self.reconciliation_agent.analyze_reconciliation_status(
                    graph_results.get('nodes', [])
                )
                agent_insights['reconciliation'] = reconciliation_insights
            except Exception as e:
                logger.warning(f"Reconciliation agent error: {e}")

        if any('CODING' in str(node) for node in graph_results.get('nodes', [])):
            # Use coding agent for coding issues
            try:
                coding_insights = self.coding_agent.analyze_coding_patterns(
                    graph_results.get('nodes', [])
                )
                agent_insights['coding'] = coding_insights
            except Exception as e:
                logger.warning(f"Coding agent error: {e}")

        if any('VISIT' in str(node) for node in graph_results.get('nodes', [])):
            # Use liaison agent for visit compliance
            try:
                liaison_insights = self.liaison_agent.analyze_visit_compliance(
                    graph_results.get('nodes', [])
                )
                agent_insights['liaison'] = liaison_insights
            except Exception as e:
                logger.warning(f"Liaison agent error: {e}")

        return agent_insights

    def _generate_response(self, user_query: str, rag_context: RAGContext,
                          graph_results: Dict[str, Any], agent_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive response using conversational engine"""

        # Create enhanced context for conversational engine
        enhanced_context = {
            'rag_context': rag_context,
            'graph_results': graph_results,
            'agent_insights': agent_insights,
            'query_type': rag_context.query_type.value
        }

        # Use conversational engine to generate response
        try:
            response = self.conversational_engine.ask(user_query)

            # Convert to dict if it's an object
            if hasattr(response, 'to_dict'):
                response = response.to_dict()
            elif not isinstance(response, dict):
                response = {'answer': str(response)}

        except Exception as e:
            logger.warning(f"Conversational engine error: {e}")
            response = {
                'answer': "I found relevant information but had trouble formatting the response.",
                'success': True
            }

        return response

    def _enhance_with_agent_recommendations(self, response: Dict[str, Any],
                                          agent_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance response with agent recommendations"""
        if not agent_insights:
            return response

        # Add agent recommendations to response
        recommendations = []
        insights = []

        for agent_name, agent_data in agent_insights.items():
            if isinstance(agent_data, dict):
                if 'recommendations' in agent_data:
                    recommendations.extend(agent_data['recommendations'])
                if 'insights' in agent_data:
                    insights.extend(agent_data['insights'])

        # Add to response
        response['agent_recommendations'] = recommendations
        response['agent_insights'] = insights

        # Enhance answer with agent context if using LongCat
        if longcat_client.is_available():
            try:
                enhanced_answer = longcat_client.enhance_reasoning(
                    response.get('answer', ''),
                    context=f"Agent insights: {json.dumps(agent_insights)}"
                )
                if enhanced_answer:
                    response['answer'] = enhanced_answer
                    response['enhanced_by_longcat'] = True
            except Exception as e:
                logger.warning(f"LongCat enhancement error: {e}")

        return response


class AgentRAGIntegration:
    """
    Integration layer between agents and RAG pipeline
    """

    def __init__(self):
        self.agent_cache = {}
        self.last_agent_calls = {}

    def get_agent_context(self, query_type: QueryType, entities: List[str]) -> Dict[str, Any]:
        """Get relevant agent context for a query"""
        context = {
            'relevant_agents': [],
            'agent_priorities': [],
            'context_data': {}
        }

        # Determine which agents are relevant based on query type and entities
        if query_type == QueryType.PRESCRIPTIVE:
            context['relevant_agents'].extend(['supervisor', 'liaison'])
            context['agent_priorities'] = ['supervisor', 'liaison']

        elif query_type == QueryType.DIAGNOSTIC:
            context['relevant_agents'].extend(['supervisor', 'reconciliation'])
            context['agent_priorities'] = ['supervisor', 'reconciliation']

        elif query_type == QueryType.PREDICTIVE:
            context['relevant_agents'].extend(['supervisor'])
            context['agent_priorities'] = ['supervisor']

        # Check entity types
        entity_types = [e.split(':')[0] if ':' in e else e for e in entities]

        if 'SAE' in entity_types or 'ADVERSE_EVENT' in entity_types:
            if 'reconciliation' not in context['relevant_agents']:
                context['relevant_agents'].append('reconciliation')

        if 'CODING' in entity_types or 'TERM' in entity_types:
            if 'coding' not in context['relevant_agents']:
                context['relevant_agents'].append('coding')

        if 'VISIT' in entity_types:
            if 'liaison' not in context['relevant_agents']:
                context['relevant_agents'].append('liaison')

        return context

    def coordinate_agent_responses(self, agent_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate and prioritize agent responses"""
        coordinated = {
            'primary_recommendations': [],
            'supporting_insights': [],
            'conflicting_views': [],
            'consensus_items': []
        }

        # Extract recommendations from all agents
        all_recommendations = []
        for agent_name, insights in agent_insights.items():
            if isinstance(insights, dict) and 'recommendations' in insights:
                for rec in insights['recommendations']:
                    all_recommendations.append({
                        'agent': agent_name,
                        'recommendation': rec,
                        'priority': insights.get('priority', 'medium')
                    })

        # Prioritize recommendations
        high_priority = [r for r in all_recommendations if r['priority'] == 'high']
        medium_priority = [r for r in all_recommendations if r['priority'] == 'medium']
        low_priority = [r for r in all_recommendations if r['priority'] == 'low']

        coordinated['primary_recommendations'] = high_priority + medium_priority[:3]
        coordinated['supporting_insights'] = low_priority

        return coordinated


class QueryRouter:
    """
    Intelligent router that determines the best processing path for queries
    """

    def __init__(self, rag_pipeline: EnhancedRAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.query_history = []
        self.performance_metrics = {}

    def route_query(self, user_query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route query to appropriate processing pipeline
        """
        start_time = datetime.now()

        # Analyze query complexity and requirements
        query_analysis = self._analyze_query_complexity(user_query)

        # Determine processing strategy
        strategy = self._determine_processing_strategy(query_analysis, user_context)

        # Execute query using determined strategy
        if strategy == 'full_rag':
            result = self.rag_pipeline.process_query(user_query, user_context)
        elif strategy == 'agent_direct':
            result = self._process_agent_direct(user_query, query_analysis)
        elif strategy == 'graph_only':
            result = self._process_graph_only(user_query, query_analysis)
        else:
            result = self._process_simple_query(user_query)

        # Record performance
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self._record_performance_metrics(user_query, strategy, processing_time, result.get('success', False))

        # Add routing metadata
        result['routing'] = {
            'strategy': strategy,
            'complexity_score': query_analysis['complexity_score'],
            'processing_time_ms': processing_time
        }

        return result

    def _analyze_query_complexity(self, user_query: str) -> Dict[str, Any]:
        """Analyze the complexity of a query"""
        analysis = {
            'complexity_score': 0,
            'requires_agents': False,
            'requires_graph': False,
            'time_sensitivity': False,
            'entity_count': 0,
            'metric_count': 0
        }

        query_lower = user_query.lower()

        # Check for complexity indicators
        complexity_indicators = [
            'why', 'how', 'what if', 'suggest', 'recommend',
            'analyze', 'compare', 'trend', 'correlation',
            'risk', 'predict', 'forecast'
        ]

        for indicator in complexity_indicators:
            if indicator in query_lower:
                analysis['complexity_score'] += 2

        # Check for agent-requiring keywords
        agent_keywords = ['should', 'recommend', 'action', 'fix', 'resolve']
        if any(keyword in query_lower for keyword in agent_keywords):
            analysis['requires_agents'] = True
            analysis['complexity_score'] += 3

        # Check for graph-requiring queries
        graph_keywords = ['relationship', 'connected', 'related', 'network', 'pattern']
        if any(keyword in query_lower for keyword in graph_keywords):
            analysis['requires_graph'] = True
            analysis['complexity_score'] += 2

        # Check for clinical trial data keywords - these should use the knowledge graph
        clinical_keywords = [
            'patient', 'study', 'coding', 'sae', 'visit', 'adverse', 'event',
            'term', 'issue', 'completion', 'rate', 'average', 'count',
            'how many', 'show me', 'list', 'find', 'identify'
        ]
        if any(keyword in query_lower for keyword in clinical_keywords):
            analysis['requires_graph'] = True
            analysis['complexity_score'] += 3  # Boost score for clinical queries

        # Count entities and metrics (simplified)
        analysis['entity_count'] = len([w for w in user_query.split() if w[0].isupper()])
        analysis['metric_count'] = len([w for w in user_query.split() if '%' in w or 'count' in w.lower()])

        analysis['complexity_score'] += analysis['entity_count'] + analysis['metric_count']

        return analysis

    def _determine_processing_strategy(self, query_analysis: Dict[str, Any],
                                     user_context: Optional[Dict[str, Any]]) -> str:
        """Determine the best processing strategy"""

        # Clinical trial queries should use full RAG pipeline
        if query_analysis['requires_graph'] or 'clinical' in str(query_analysis).lower():
            return 'full_rag'

        if query_analysis['complexity_score'] >= 8:
            return 'full_rag'
        elif query_analysis['requires_agents']:
            return 'agent_direct'
        elif query_analysis['requires_graph']:
            return 'graph_only'
        else:
            return 'simple'

    def _process_agent_direct(self, user_query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process query directly through agents"""
        # Simplified agent-direct processing
        return {
            'success': True,
            'answer': f"Based on agent analysis: {user_query}",
            'agent_direct': True
        }

    def _process_graph_only(self, user_query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using only knowledge graph"""
        # Simplified graph-only processing
        return {
            'success': True,
            'answer': f"Based on knowledge graph analysis: {user_query}",
            'graph_only': True
        }

    def _process_simple_query(self, user_query: str) -> Dict[str, Any]:
        """Process simple factual queries"""
        return {
            'success': True,
            'answer': f"Simple query response: {user_query}",
            'simple': True
        }

    def _record_performance_metrics(self, query: str, strategy: str, processing_time: float, success: bool):
        """Record performance metrics for analysis"""
        self.performance_metrics[query] = {
            'strategy': strategy,
            'processing_time': processing_time,
            'success': success,
            'timestamp': datetime.now()
        }

        # Keep only last 100 queries
        if len(self.performance_metrics) > 100:
            oldest = min(self.performance_metrics.keys(),
                        key=lambda k: self.performance_metrics[k]['timestamp'])
            del self.performance_metrics[oldest]


# Main interface class
class EnhancedRAGSystem:
    """
    Main interface for the enhanced RAG system with knowledge graph integration
    """

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)

        # Initialize components
        self.kg_builder = KnowledgeGraphBuilder(data_path)
        self.rag_pipeline = EnhancedRAGPipeline(self.kg_builder)
        self.query_router = QueryRouter(self.rag_pipeline)

        # Ingestion state
        self.is_ingested = False

    def initialize(self, force_reingest: bool = False) -> bool:
        """
        Initialize the RAG system by ingesting data into knowledge graph.
        Returns True if ingestion was performed, False if using cached data.
        """
        logger.info("Initializing Enhanced RAG System...")

        ingested = self.kg_builder.ingest_all_data(force_reingest=force_reingest)
        self.is_ingested = True

        if ingested:
            logger.info("✅ Knowledge graph built from fresh data")
        else:
            logger.info("✅ Knowledge graph loaded from cache")

        return ingested

    def query(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a natural language query using the enhanced RAG system.

        Args:
            user_query: The user's question or request
            context: Optional context (conversation history, user preferences, etc.)

        Returns:
            Comprehensive response with answer, insights, and metadata
        """
        if not self.is_ingested:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        # Route and process the query
        result = self.query_router.route_query(user_query, context)

        # Add system metadata
        result['system_info'] = {
            'rag_system': 'Enhanced Knowledge Graph RAG',
            'ingestion_timestamp': self.kg_builder.ingestion_timestamp.isoformat() if self.kg_builder.ingestion_timestamp else None,
            'graph_nodes': len(self.kg_builder.knowledge_graph.graph.nodes()),
            'graph_edges': len(self.kg_builder.knowledge_graph.graph.edges())
        }

        return result

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        return {
            'initialized': self.is_ingested,
            'ingestion_timestamp': self.kg_builder.ingestion_timestamp.isoformat() if self.kg_builder.ingestion_timestamp else None,
            'graph_statistics': {
                'nodes': len(self.kg_builder.knowledge_graph.graph.nodes()),
                'edges': len(self.kg_builder.knowledge_graph.graph.edges()),
                'node_types': self.kg_builder._get_node_type_counts(),
                'edge_types': self.kg_builder._get_edge_type_counts()
            },
            'performance_metrics': {
                'total_queries_processed': len(self.query_router.performance_metrics),
                'average_processing_time': self._calculate_avg_processing_time()
            }
        }

    def _calculate_avg_processing_time(self) -> float:
        """Calculate average query processing time"""
        if not self.query_router.performance_metrics:
            return 0.0

        times = [metrics['processing_time'] for metrics in self.query_router.performance_metrics.values()]
        return sum(times) / len(times)

    def rebuild_knowledge_graph(self) -> bool:
        """Force rebuild of the knowledge graph"""
        logger.info("Forcing knowledge graph rebuild...")
        return self.initialize(force_reingest=True)