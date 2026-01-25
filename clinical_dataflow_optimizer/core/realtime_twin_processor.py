"""
Real-Time Digital Twin Processor
================================

Provides real-time processing and updates for Digital Patient Twins
using NetworkX graph processing and WebSocket streaming.

Features:
- Dynamic twin generation from live data streams
- Real-time graph updates and traversal
- WebSocket broadcasting for UI updates
- Event-driven twin evolution
- Caching with intelligent invalidation
"""

import json
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd
import numpy as np
import networkx as nx

from core.digital_twin import (
    DigitalTwinFactory, 
    DigitalTwinConfig,
    BlockingItemType,
    BlockingSeverity
)
from models.data_models import DigitalPatientTwin, RiskMetrics, PatientStatus
from graph.knowledge_graph import ClinicalKnowledgeGraph, NodeType, EdgeType

logger = logging.getLogger(__name__)


@dataclass
class TwinUpdateEvent:
    """Event representing a twin state change"""
    subject_id: str
    event_type: str  # created, updated, deleted
    changes: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    
    def to_dict(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'event_type': self.event_type,
            'changes': self.changes,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class TwinCache:
    """
    Intelligent cache for Digital Patient Twins with TTL and invalidation.
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.ttl = timedelta(seconds=ttl_seconds)
        self._access_count: Dict[str, int] = defaultdict(int)
        self._invalidation_callbacks: List[Callable] = []
    
    def get(self, subject_id: str) -> Optional[DigitalPatientTwin]:
        """Get cached twin if valid"""
        with self._lock:
            if subject_id in self._cache:
                entry = self._cache[subject_id]
                if datetime.now() - entry['cached_at'] < self.ttl:
                    self._access_count[subject_id] += 1
                    return entry['twin']
                else:
                    del self._cache[subject_id]
        return None
    
    def set(self, subject_id: str, twin: DigitalPatientTwin):
        """Cache a twin"""
        with self._lock:
            self._cache[subject_id] = {
                'twin': twin,
                'cached_at': datetime.now(),
                'hash': self._compute_hash(twin)
            }
    
    def invalidate(self, subject_id: str):
        """Invalidate a specific twin"""
        with self._lock:
            if subject_id in self._cache:
                del self._cache[subject_id]
                
        # Notify callbacks
        for callback in self._invalidation_callbacks:
            try:
                callback(subject_id)
            except Exception as e:
                logger.warning(f"Invalidation callback error: {e}")
    
    def invalidate_all(self):
        """Invalidate all cached twins"""
        with self._lock:
            self._cache.clear()
    
    def add_invalidation_callback(self, callback: Callable[[str], None]):
        """Add callback to be notified on invalidation"""
        self._invalidation_callbacks.append(callback)
    
    def _compute_hash(self, twin: DigitalPatientTwin) -> str:
        """Compute hash of twin state for change detection"""
        data = json.dumps(twin.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(data.encode()).hexdigest()
    
    def has_changed(self, subject_id: str, new_twin: DigitalPatientTwin) -> bool:
        """Check if twin has changed from cached version"""
        with self._lock:
            if subject_id not in self._cache:
                return True
            old_hash = self._cache[subject_id].get('hash', '')
            new_hash = self._compute_hash(new_twin)
            return old_hash != new_hash
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'total_accesses': sum(self._access_count.values()),
                'hot_twins': sorted(
                    self._access_count.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }


class RealTimeTwinProcessor:
    """
    Real-time processor for Digital Patient Twins.
    
    Provides:
    - Dynamic twin creation from streaming data
    - Graph-based twin evolution
    - WebSocket broadcasting
    - Event emission for downstream consumers
    """
    
    def __init__(
        self,
        graph: Optional[nx.MultiDiGraph] = None,
        config: Optional[DigitalTwinConfig] = None,
        cache_ttl: int = 300
    ):
        self.graph = graph or nx.MultiDiGraph()
        self.config = config or DigitalTwinConfig()
        self.cache = TwinCache(cache_ttl)
        self.factory: Optional[DigitalTwinFactory] = None
        
        # Event subscribers for real-time updates
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # WebSocket connections for broadcasting
        self._websocket_connections: Set = set()
        
        # Processing state
        self._processing = False
        self._update_queue = asyncio.Queue() if asyncio.get_event_loop().is_running() else None
        
        # Metrics
        self._metrics = {
            'twins_created': 0,
            'twins_updated': 0,
            'events_emitted': 0,
            'graph_updates': 0
        }
        
        logger.info("RealTimeTwinProcessor initialized")
    
    def set_graph(self, graph: nx.MultiDiGraph):
        """Update the knowledge graph"""
        self.graph = graph
        self.factory = DigitalTwinFactory(graph, self.config)
        self.cache.invalidate_all()
        self._metrics['graph_updates'] += 1
        logger.info(f"Graph updated with {graph.number_of_nodes()} nodes")
    
    def get_twin(self, subject_id: str, force_refresh: bool = False) -> Optional[DigitalPatientTwin]:
        """
        Get a Digital Patient Twin, using cache when available.
        """
        if not force_refresh:
            cached = self.cache.get(subject_id)
            if cached:
                return cached
        
        # Create fresh twin from graph
        if self.factory is None:
            self.factory = DigitalTwinFactory(self.graph, self.config)
        
        twin = self.factory.create_twin(subject_id)
        
        if twin:
            # Check for changes and emit events
            if self.cache.has_changed(subject_id, twin):
                self._emit_update_event(subject_id, twin)
            
            self.cache.set(subject_id, twin)
            self._metrics['twins_created'] += 1
        
        return twin
    
    def get_all_twins(self, study_id: Optional[str] = None) -> List[DigitalPatientTwin]:
        """Get all twins, optionally filtered by study"""
        twins = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'Patient':
                subject_id = node_data.get('subject_id', '')
                
                if study_id and node_data.get('study_id') != study_id:
                    continue
                
                if subject_id:
                    twin = self.get_twin(subject_id)
                    if twin:
                        twins.append(twin)
        
        return twins
    
    def update_from_dataframe(
        self,
        df: pd.DataFrame,
        data_type: str,
        subject_id_col: str = 'Subject ID'
    ):
        """
        Update twins from a DataFrame (e.g., new CPID data).
        This triggers real-time updates.
        """
        if df.empty:
            return
        
        affected_subjects = set()
        
        for _, row in df.iterrows():
            subject_id = str(row.get(subject_id_col, ''))
            if not subject_id:
                continue
            
            affected_subjects.add(subject_id)
            
            # Update graph node with new data
            self._update_graph_node(subject_id, data_type, row.to_dict())
        
        # Invalidate cache for affected subjects
        for subject_id in affected_subjects:
            self.cache.invalidate(subject_id)
        
        # Emit batch update event
        self._emit_batch_update(list(affected_subjects), data_type)
        
        logger.info(f"Updated {len(affected_subjects)} twins from {data_type} data")
    
    def _update_graph_node(self, subject_id: str, data_type: str, data: Dict):
        """Update graph node with new data"""
        # Find or create patient node
        patient_node_id = None
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'Patient' and node_data.get('subject_id') == subject_id:
                patient_node_id = node_id
                break
        
        if patient_node_id is None:
            # Create new patient node
            patient_node_id = f"patient_{subject_id}"
            self.graph.add_node(patient_node_id, node_type='Patient', subject_id=subject_id)
        
        # Update node attributes based on data type
        if data_type == 'cpid':
            self._update_cpid_attributes(patient_node_id, data)
        elif data_type == 'sae':
            self._update_sae_attributes(patient_node_id, data)
        elif data_type == 'coding':
            self._update_coding_attributes(patient_node_id, data)
        
        self._metrics['graph_updates'] += 1
    
    def _update_cpid_attributes(self, node_id: str, data: Dict):
        """Update patient node with CPID data"""
        # Map CPID columns to node attributes
        attr_mapping = {
            'Missing Visits': 'missing_visits',
            'Missing Pages': 'missing_pages',
            'Open Queries': 'open_queries',
            'Total Queries': 'total_queries',
            'Uncoded Terms': 'uncoded_terms',
            'Coded Terms': 'coded_terms',
            'Verification %': 'verification_pct',
            'Data Quality Index': 'data_quality_index'
        }
        
        attrs = self.graph.nodes[node_id]
        for src_col, dest_attr in attr_mapping.items():
            if src_col in data and not pd.isna(data[src_col]):
                attrs[dest_attr] = data[src_col]
        
        attrs['last_updated'] = datetime.now().isoformat()
    
    def _update_sae_attributes(self, node_id: str, data: Dict):
        """Update patient node with SAE data"""
        attrs = self.graph.nodes[node_id]
        
        # Create or update SAE record
        sae_records = attrs.get('sae_records', [])
        sae_id = data.get('Discrepancy ID', data.get('SAE ID', ''))
        
        # Find and update or add
        found = False
        for i, record in enumerate(sae_records):
            if record.get('sae_id') == sae_id:
                sae_records[i] = {
                    'sae_id': sae_id,
                    'review_status': data.get('Review Status', 'Unknown'),
                    'action_status': data.get('Action Status', 'Unknown'),
                    'form_name': data.get('Form Name', ''),
                    'updated_at': datetime.now().isoformat()
                }
                found = True
                break
        
        if not found:
            sae_records.append({
                'sae_id': sae_id,
                'review_status': data.get('Review Status', 'Unknown'),
                'action_status': data.get('Action Status', 'Unknown'),
                'form_name': data.get('Form Name', ''),
                'created_at': datetime.now().isoformat()
            })
        
        attrs['sae_records'] = sae_records
        attrs['last_updated'] = datetime.now().isoformat()
    
    def _update_coding_attributes(self, node_id: str, data: Dict):
        """Update patient node with coding data"""
        attrs = self.graph.nodes[node_id]
        
        # Update uncoded terms
        coding_status = data.get('Coding Status', '')
        if 'uncoded' in coding_status.lower():
            attrs['uncoded_terms'] = attrs.get('uncoded_terms', 0) + 1
        elif 'coded' in coding_status.lower():
            attrs['coded_terms'] = attrs.get('coded_terms', 0) + 1
    
    def _emit_update_event(self, subject_id: str, twin: DigitalPatientTwin):
        """Emit update event for a twin"""
        event = TwinUpdateEvent(
            subject_id=subject_id,
            event_type='updated',
            changes={
                'clean_status': twin.clean_status,
                'risk_level': twin.risk_metrics.composite_risk_score if twin.risk_metrics else 0,
                'blocking_count': len(twin.blocking_items),
                'dqi': twin.data_quality_index
            }
        )
        
        # Notify subscribers
        for subscriber in self._subscribers.get('twin_update', []):
            try:
                subscriber(event)
            except Exception as e:
                logger.warning(f"Subscriber error: {e}")
        
        # Broadcast to WebSockets
        asyncio.create_task(self._broadcast_websocket(event))
        
        self._metrics['events_emitted'] += 1
    
    def _emit_batch_update(self, subject_ids: List[str], source: str):
        """Emit batch update notification"""
        event = {
            'event_type': 'batch_update',
            'affected_subjects': subject_ids,
            'source': source,
            'timestamp': datetime.now().isoformat()
        }
        
        for subscriber in self._subscribers.get('batch_update', []):
            try:
                subscriber(event)
            except Exception as e:
                logger.warning(f"Batch subscriber error: {e}")
    
    async def _broadcast_websocket(self, event: TwinUpdateEvent):
        """Broadcast event to all WebSocket connections"""
        if not self._websocket_connections:
            return
        
        message = event.to_json()
        dead_connections = set()
        
        for ws in self._websocket_connections:
            try:
                await ws.send(message)
            except Exception:
                dead_connections.add(ws)
        
        # Remove dead connections
        self._websocket_connections -= dead_connections
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to twin events"""
        self._subscribers[event_type].append(callback)
    
    def register_websocket(self, websocket):
        """Register WebSocket for real-time updates"""
        self._websocket_connections.add(websocket)
    
    def unregister_websocket(self, websocket):
        """Unregister WebSocket"""
        self._websocket_connections.discard(websocket)
    
    def get_metrics(self) -> Dict:
        """Get processor metrics"""
        return {
            **self._metrics,
            'cache_stats': self.cache.get_stats(),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'websocket_connections': len(self._websocket_connections)
        }
    
    # ==================== Real-Time Evolution ====================
    
    def evolve_twin(
        self,
        subject_id: str,
        changes: Dict[str, Any],
        source: str = "external"
    ) -> Optional[DigitalPatientTwin]:
        """
        Evolve a twin based on new information.
        This is the core method for dynamic twin updates.
        """
        # Find patient node
        patient_node_id = None
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'Patient' and node_data.get('subject_id') == subject_id:
                patient_node_id = node_id
                break
        
        if patient_node_id is None:
            logger.warning(f"Cannot evolve twin - subject {subject_id} not found")
            return None
        
        # Apply changes to graph node
        attrs = self.graph.nodes[patient_node_id]
        for key, value in changes.items():
            attrs[key] = value
        attrs['last_updated'] = datetime.now().isoformat()
        attrs['update_source'] = source
        
        # Invalidate cache
        self.cache.invalidate(subject_id)
        
        # Generate fresh twin
        twin = self.get_twin(subject_id, force_refresh=True)
        
        if twin:
            self._metrics['twins_updated'] += 1
            
            # Emit evolution event
            event = TwinUpdateEvent(
                subject_id=subject_id,
                event_type='evolved',
                changes=changes,
                source=source
            )
            
            for subscriber in self._subscribers.get('twin_evolved', []):
                try:
                    subscriber(event)
                except Exception as e:
                    logger.warning(f"Evolution subscriber error: {e}")
        
        return twin
    
    def get_twin_history(self, subject_id: str, limit: int = 10) -> List[Dict]:
        """Get recent history of twin changes (if tracking enabled)"""
        # This would connect to an event store in production
        # For now, return empty list
        return []
    
    def compute_twin_trajectory(
        self,
        subject_id: str,
        lookahead_days: int = 7
    ) -> Dict:
        """
        Predict twin state trajectory based on current trends.
        Used for proactive intervention recommendations.
        """
        twin = self.get_twin(subject_id)
        if not twin:
            return {}
        
        # Simple trajectory prediction based on current metrics
        trajectory = {
            'subject_id': subject_id,
            'current_state': {
                'clean_status': twin.clean_status,
                'dqi': twin.data_quality_index,
                'risk_score': twin.risk_metrics.composite_risk_score if twin.risk_metrics else 0
            },
            'predictions': []
        }
        
        # Calculate velocity (change rate) if we have history
        velocity = twin.risk_metrics.net_velocity if twin.risk_metrics else 0
        
        # Project forward
        for day in range(1, lookahead_days + 1):
            predicted_dqi = max(0, min(100, twin.data_quality_index + (velocity * day * 10)))
            trajectory['predictions'].append({
                'day': day,
                'predicted_dqi': round(predicted_dqi, 1),
                'predicted_clean': predicted_dqi >= 95 and len(twin.blocking_items) == 0,
                'confidence': max(0.3, 0.9 - (day * 0.1))
            })
        
        return trajectory


# Global singleton
_twin_processor: Optional[RealTimeTwinProcessor] = None


def get_twin_processor() -> RealTimeTwinProcessor:
    """Get or create the global twin processor"""
    global _twin_processor
    if _twin_processor is None:
        _twin_processor = RealTimeTwinProcessor()
    return _twin_processor


def set_twin_processor(processor: RealTimeTwinProcessor):
    """Set the global twin processor"""
    global _twin_processor
    _twin_processor = processor
