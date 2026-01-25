"""
Real-time Service
Manages WebSocket connections and real-time updates
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import asyncio
import json
import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept and store WebSocket connection"""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def disconnect_all(self):
        """Disconnect all WebSocket connections"""
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.close()
                except Exception:
                    pass
            self.active_connections.clear()
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_to_connection(self, websocket: WebSocket, message: Dict):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")
            self.disconnect(websocket)


class RealtimeService:
    """
    Service for managing real-time updates and subscriptions
    """
    
    def __init__(self):
        self.subscriptions: Dict[WebSocket, Dict] = {}
        self.pending_alerts: List[Dict] = []
        self.latest_updates: List[Dict] = []
        self._background_task = None
        self._running = False
        self._update_queue = asyncio.Queue()
    
    async def start_background_tasks(self):
        """Start background update tasks (Idempotent)"""
        if self._running and self._background_task and not self._background_task.done():
            logger.warning("Real-time service already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._process_updates())
        logger.info("Real-time service background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks"""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("Real-time service background tasks stopped")
    
    async def _process_updates(self):
        """Background task to process and distribute updates"""
        while self._running:
            try:
                # Check for updates periodically
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing updates: {e}")
    
    def _generate_update(self) -> Optional[Dict]:
        """Generate a simulated update for demo purposes"""
        import random
        
        update_types = ['patient_status', 'query_update', 'metric_change', 'alert']
        update_type = random.choice(update_types)
        
        if update_type == 'patient_status':
            return {
                'type': 'patient_status',
                'patient_id': f'P{random.randint(1, 1000):04d}',
                'previous_status': random.choice(['clean', 'dirty']),
                'new_status': random.choice(['clean', 'dirty']),
                'cleanliness_score': round(random.uniform(50, 100), 1),
                'timestamp': datetime.now().isoformat()
            }
        elif update_type == 'query_update':
            return {
                'type': 'query_update',
                'query_id': f'Q{random.randint(1, 500):04d}',
                'status': random.choice(['opened', 'answered', 'closed']),
                'patient_id': f'P{random.randint(1, 1000):04d}',
                'timestamp': datetime.now().isoformat()
            }
        elif update_type == 'metric_change':
            return {
                'type': 'metric_change',
                'metric': random.choice(['dqi', 'cleanliness', 'query_count']),
                'entity_type': random.choice(['study', 'site']),
                'entity_id': f'Entity_{random.randint(1, 20)}',
                'old_value': round(random.uniform(70, 85), 1),
                'new_value': round(random.uniform(75, 90), 1),
                'timestamp': datetime.now().isoformat()
            }
        elif update_type == 'alert':
            alert = self._generate_alert()
            self.pending_alerts.append(alert)
            return {
                'type': 'new_alert',
                'alert': alert,
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _generate_alert(self) -> Dict:
        """Generate a simulated alert"""
        import random
        
        severities = ['critical', 'high', 'medium', 'low']
        categories = ['data_quality', 'safety', 'operational', 'compliance']
        
        return {
            'alert_id': f'ALT{random.randint(1, 10000):05d}',
            'category': random.choice(categories),
            'severity': random.choice(severities),
            'status': 'new',
            'title': f'Alert: {random.choice(["Query backlog increasing", "SAE pending reconciliation", "Data entry delay", "Missing visit data"])}',
            'description': 'Automated alert generated by monitoring system',
            'source': 'monitoring_agent',
            'affected_entity': {
                'type': random.choice(['study', 'site', 'patient']),
                'id': f'Entity_{random.randint(1, 100)}'
            },
            'details': {},
            'created_at': datetime.now().isoformat(),
            'updated_at': None,
            'acknowledged_by': None,
            'resolved_at': None,
            'actions_taken': []
        }
    
    async def add_subscription(self, websocket: WebSocket, subscription: Dict):
        """Add subscription for a WebSocket connection"""
        self.subscriptions[websocket] = subscription
        logger.info(f"Added subscription: {subscription}")
    
    async def remove_subscriptions(self, websocket: WebSocket):
        """Remove all subscriptions for a WebSocket"""
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
    
    async def get_pending_alerts(self) -> List[Dict]:
        """Get and clear pending alerts"""
        alerts = self.pending_alerts.copy()
        self.pending_alerts.clear()
        return alerts
    
    async def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        # In a real implementation, this would update the alert in the database
        logger.info(f"Alert {alert_id} acknowledged")
    
    async def get_latest_updates(self) -> List[Dict]:
        """Get latest updates"""
        updates = self.latest_updates.copy()
        self.latest_updates.clear()
        return updates
    
    async def push_update(self, update: Dict):
        """Push an update to the queue"""
        await self._update_queue.put(update)
    
    async def notify_patient_status_change(self, patient_id: str, old_status: str, 
                                           new_status: str, cleanliness_score: float):
        """Notify subscribers of patient status change"""
        update = {
            'type': 'patient_status_change',
            'patient_id': patient_id,
            'previous_status': old_status,
            'new_status': new_status,
            'cleanliness_score': cleanliness_score,
            'timestamp': datetime.now().isoformat()
        }
        self.latest_updates.append(update)
        await self.push_update(update)
    
    async def notify_metric_change(self, metric: str, entity_type: str, 
                                   entity_id: str, old_value: float, new_value: float):
        """Notify subscribers of metric change"""
        update = {
            'type': 'metric_change',
            'metric': metric,
            'entity_type': entity_type,
            'entity_id': entity_id,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': datetime.now().isoformat()
        }
        self.latest_updates.append(update)
        await self.push_update(update)
    
    async def notify_new_alert(self, alert: Dict):
        """Notify subscribers of new alert"""
        update = {
            'type': 'new_alert',
            'alert': alert,
            'timestamp': datetime.now().isoformat()
        }
        self.pending_alerts.append(alert)
        self.latest_updates.append(update)
        await self.push_update(update)
