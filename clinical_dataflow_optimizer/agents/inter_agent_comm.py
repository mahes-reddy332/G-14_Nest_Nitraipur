"""
Inter-Agent Communication Protocol
===================================

Event-driven communication system for multi-agent coordination.
Implements publish-subscribe pattern with priority queuing and
message persistence for audit trails.

Features:
- Event-driven activation (data changes, thresholds, triggers)
- Priority-based message queuing
- Inter-agent communication protocols
- Message persistence and audit logging
- Circuit breaker for agent failures
"""

import json
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from queue import PriorityQueue
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can trigger agent actions"""
    # Data events
    DATA_CHANGED = auto()
    DATA_INGESTED = auto()
    DATA_QUALITY_ALERT = auto()
    
    # Threshold events
    THRESHOLD_EXCEEDED = auto()
    THRESHOLD_APPROACHING = auto()
    
    # Safety events
    SAE_DETECTED = auto()
    ZOMBIE_SAE_DETECTED = auto()
    RECONCILIATION_ISSUE = auto()
    
    # Coding events
    UNCODED_TERM_DETECTED = auto()
    AMBIGUOUS_CODING = auto()
    CODING_COMPLETED = auto()
    
    # Site events
    VISIT_UPCOMING = auto()
    VISIT_MISSED = auto()
    SITE_PERFORMANCE_ALERT = auto()
    
    # Agent events
    AGENT_RECOMMENDATION = auto()
    AGENT_ACTION_REQUIRED = auto()
    AGENT_ESCALATION = auto()
    
    # System events
    SYSTEM_HEALTH_CHECK = auto()
    AUDIT_REQUIRED = auto()


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1  # Immediate processing
    HIGH = 2      # Process within minutes
    MEDIUM = 3    # Process within hour
    LOW = 4       # Batch processing OK
    BACKGROUND = 5  # When idle


@dataclass(order=True)
class AgentMessage:
    """
    Message for inter-agent communication.
    Comparable for priority queue ordering.
    """
    priority: int
    message_id: str = field(compare=False)
    event_type: EventType = field(compare=False)
    source_agent: str = field(compare=False)
    target_agent: Optional[str] = field(compare=False, default=None)  # None = broadcast
    payload: Dict = field(compare=False, default_factory=dict)
    created_at: datetime = field(compare=False, default_factory=datetime.now)
    expires_at: Optional[datetime] = field(compare=False, default=None)
    requires_ack: bool = field(compare=False, default=False)
    correlation_id: Optional[str] = field(compare=False, default=None)
    
    def to_dict(self) -> Dict:
        return {
            'message_id': self.message_id,
            'priority': self.priority,
            'event_type': self.event_type.name,
            'source_agent': self.source_agent,
            'target_agent': self.target_agent,
            'payload': self.payload,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'requires_ack': self.requires_ack,
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMessage':
        return cls(
            priority=data['priority'],
            message_id=data['message_id'],
            event_type=EventType[data['event_type']],
            source_agent=data['source_agent'],
            target_agent=data.get('target_agent'),
            payload=data.get('payload', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            requires_ack=data.get('requires_ack', False),
            correlation_id=data.get('correlation_id')
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class EventSubscription:
    """Represents a subscription to an event type"""
    subscriber_id: str
    event_types: Set[EventType]
    handler: Callable[[AgentMessage], Any]
    filter_fn: Optional[Callable[[AgentMessage], bool]] = None
    priority_filter: Optional[int] = None  # Only receive if priority <= this
    
    def should_receive(self, message: AgentMessage) -> bool:
        """Check if subscription should receive this message"""
        # Check event type
        if message.event_type not in self.event_types:
            return False
        
        # Check priority filter
        if self.priority_filter and message.priority > self.priority_filter:
            return False
        
        # Check custom filter
        if self.filter_fn and not self.filter_fn(message):
            return False
        
        return True


class MessageBus:
    """
    Central message bus for inter-agent communication.
    Implements publish-subscribe pattern with priority queuing.
    """
    
    def __init__(self, enable_persistence: bool = True, log_dir: str = None):
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._message_queue = PriorityQueue()
        self._processed_messages: List[Dict] = []
        self._pending_acks: Dict[str, AgentMessage] = {}
        self._lock = threading.Lock()
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        
        # Persistence
        self.enable_persistence = enable_persistence
        self.log_dir = log_dir or "./audit_logs"
        
        # Statistics
        self.stats = {
            'messages_published': 0,
            'messages_delivered': 0,
            'messages_expired': 0,
            'acks_received': 0,
            'errors': 0
        }
    
    def subscribe(
        self,
        subscriber_id: str,
        event_types: Set[EventType],
        handler: Callable[[AgentMessage], Any],
        filter_fn: Optional[Callable[[AgentMessage], bool]] = None,
        priority_filter: Optional[int] = None
    ) -> str:
        """Subscribe to events"""
        subscription = EventSubscription(
            subscriber_id=subscriber_id,
            event_types=event_types,
            handler=handler,
            filter_fn=filter_fn,
            priority_filter=priority_filter
        )
        
        with self._lock:
            self._subscriptions[subscriber_id] = subscription
        
        logger.info(f"Agent {subscriber_id} subscribed to {[e.name for e in event_types]}")
        return subscriber_id
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe from events"""
        with self._lock:
            if subscriber_id in self._subscriptions:
                del self._subscriptions[subscriber_id]
                return True
        return False
    
    def publish(
        self,
        event_type: EventType,
        source_agent: str,
        payload: Dict,
        priority: MessagePriority = MessagePriority.MEDIUM,
        target_agent: Optional[str] = None,
        requires_ack: bool = False,
        ttl_minutes: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Publish a message to the bus"""
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        expires_at = None
        if ttl_minutes:
            expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
        
        message = AgentMessage(
            priority=priority.value,
            message_id=message_id,
            event_type=event_type,
            source_agent=source_agent,
            target_agent=target_agent,
            payload=payload,
            requires_ack=requires_ack,
            expires_at=expires_at,
            correlation_id=correlation_id
        )
        
        # Add to queue
        self._message_queue.put(message)
        self.stats['messages_published'] += 1
        
        # Log for audit
        if self.enable_persistence:
            self._persist_message(message)
        
        logger.debug(f"Published {event_type.name} from {source_agent} (priority: {priority.name})")
        return message_id
    
    def _persist_message(self, message: AgentMessage):
        """Persist message for audit trail"""
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_file = os.path.join(
            self.log_dir,
            f"messages_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(message.to_dict()) + '\n')
        except Exception as e:
            logger.warning(f"Failed to persist message: {e}")
    
    def acknowledge(self, message_id: str, ack_data: Dict = None):
        """Acknowledge a message that required acknowledgment"""
        with self._lock:
            if message_id in self._pending_acks:
                del self._pending_acks[message_id]
                self.stats['acks_received'] += 1
                logger.debug(f"Message {message_id} acknowledged")
    
    def process_next(self) -> bool:
        """Process next message in queue. Returns False if queue is empty."""
        if self._message_queue.empty():
            return False
        
        try:
            message = self._message_queue.get_nowait()
        except:
            return False
        
        # Check expiration
        if message.is_expired():
            self.stats['messages_expired'] += 1
            logger.debug(f"Message {message.message_id} expired, skipping")
            return True
        
        # Track pending acks
        if message.requires_ack:
            with self._lock:
                self._pending_acks[message.message_id] = message
        
        # Deliver to subscribers
        with self._lock:
            subscriptions = list(self._subscriptions.values())
        
        delivered_count = 0
        for subscription in subscriptions:
            # Check if targeted
            if message.target_agent and message.target_agent != subscription.subscriber_id:
                continue
            
            if subscription.should_receive(message):
                try:
                    subscription.handler(message)
                    delivered_count += 1
                except Exception as e:
                    logger.error(f"Error delivering to {subscription.subscriber_id}: {e}")
                    self.stats['errors'] += 1
        
        self.stats['messages_delivered'] += delivered_count
        
        # Archive processed message
        self._processed_messages.append({
            'message': message.to_dict(),
            'delivered_to': delivered_count,
            'processed_at': datetime.now().isoformat()
        })
        
        return True
    
    def start_processor(self, poll_interval: float = 0.1):
        """Start background message processor"""
        if self._running:
            return
        
        self._running = True
        
        def processor_loop():
            while self._running:
                try:
                    if not self.process_next():
                        import time
                        time.sleep(poll_interval)
                except Exception as e:
                    logger.error(f"Message processor error: {e}")
        
        self._processor_thread = threading.Thread(target=processor_loop, daemon=True)
        self._processor_thread.start()
        logger.info("Message bus processor started")
    
    def stop_processor(self):
        """Stop background message processor"""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=5)
        logger.info("Message bus processor stopped")
    
    def get_stats(self) -> Dict:
        """Get message bus statistics"""
        return {
            **self.stats,
            'queue_size': self._message_queue.qsize(),
            'active_subscriptions': len(self._subscriptions),
            'pending_acks': len(self._pending_acks)
        }


class EventTrigger:
    """
    Event-driven trigger system for agent activation.
    Monitors data changes and thresholds to emit events.
    """
    
    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self._thresholds: Dict[str, Dict] = {}
        self._watchers: Dict[str, Callable] = {}
        self._last_values: Dict[str, Any] = {}
    
    def register_threshold(
        self,
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float,
        comparison: str = 'gt'  # gt, lt, eq
    ):
        """Register a threshold for monitoring"""
        self._thresholds[metric_name] = {
            'warning': warning_threshold,
            'critical': critical_threshold,
            'comparison': comparison
        }
    
    def check_threshold(
        self,
        metric_name: str,
        value: float,
        context: Dict = None
    ) -> Optional[EventType]:
        """Check if value exceeds threshold and emit event if needed"""
        if metric_name not in self._thresholds:
            return None
        
        threshold = self._thresholds[metric_name]
        comparison = threshold['comparison']
        
        def compare(val, thresh):
            if comparison == 'gt':
                return val > thresh
            elif comparison == 'lt':
                return val < thresh
            else:
                return val == thresh
        
        event_type = None
        priority = MessagePriority.LOW
        
        if compare(value, threshold['critical']):
            event_type = EventType.THRESHOLD_EXCEEDED
            priority = MessagePriority.CRITICAL
        elif compare(value, threshold['warning']):
            event_type = EventType.THRESHOLD_APPROACHING
            priority = MessagePriority.HIGH
        
        if event_type:
            self.bus.publish(
                event_type=event_type,
                source_agent='EventTrigger',
                payload={
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'context': context or {}
                },
                priority=priority
            )
        
        return event_type
    
    def check_data_change(
        self,
        data_key: str,
        new_value: Any,
        context: Dict = None
    ) -> bool:
        """Check if data has changed and emit event"""
        old_value = self._last_values.get(data_key)
        self._last_values[data_key] = new_value
        
        if old_value is None:
            return False  # First time seeing this key
        
        if old_value != new_value:
            self.bus.publish(
                event_type=EventType.DATA_CHANGED,
                source_agent='EventTrigger',
                payload={
                    'data_key': data_key,
                    'old_value': old_value,
                    'new_value': new_value,
                    'context': context or {}
                },
                priority=MessagePriority.MEDIUM
            )
            return True
        
        return False
    
    def emit_safety_event(
        self,
        event_type: EventType,
        subject_id: str,
        site_id: str,
        study_id: str,
        details: Dict,
        priority: MessagePriority = MessagePriority.HIGH
    ):
        """Emit a safety-related event"""
        if event_type not in [EventType.SAE_DETECTED, EventType.ZOMBIE_SAE_DETECTED, 
                             EventType.RECONCILIATION_ISSUE]:
            logger.warning(f"Event type {event_type} is not a safety event")
        
        self.bus.publish(
            event_type=event_type,
            source_agent='EventTrigger',
            payload={
                'subject_id': subject_id,
                'site_id': site_id,
                'study_id': study_id,
                **details
            },
            priority=priority,
            requires_ack=True,
            ttl_minutes=60 * 24  # 24 hour TTL for safety events
        )


class AgentCoordinator:
    """
    Coordinates multi-agent activities and ensures proper sequencing
    of agent responses to events.
    """
    
    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.event_trigger = EventTrigger(message_bus)
        self._agents: Dict[str, Any] = {}
        self._agent_status: Dict[str, str] = {}
        self._workflows: Dict[str, List[str]] = {}
        
        # Register default workflows
        self._register_default_workflows()
    
    def _register_default_workflows(self):
        """Register default agent coordination workflows"""
        # Zombie SAE workflow: Rex -> Site Query -> EDRR Update
        self._workflows['zombie_sae'] = ['Rex', 'Supervisor', 'AuditTrail']
        
        # Ambiguous coding workflow: Codex -> Site Query -> Learning Update
        self._workflows['ambiguous_coding'] = ['Codex', 'Supervisor', 'LearningEngine']
        
        # Site performance workflow: Lia -> CRA Alert -> Escalation
        self._workflows['site_performance'] = ['Lia', 'Supervisor', 'AlertEngine']
    
    def register_agent(self, agent_id: str, agent_instance: Any):
        """Register an agent with the coordinator"""
        self._agents[agent_id] = agent_instance
        self._agent_status[agent_id] = 'idle'
        
        # Auto-subscribe agent to relevant events
        if hasattr(agent_instance, 'get_subscribed_events'):
            events = agent_instance.get_subscribed_events()
            self.bus.subscribe(
                subscriber_id=agent_id,
                event_types=events,
                handler=lambda msg: self._handle_agent_message(agent_id, msg)
            )
        
        logger.info(f"Agent {agent_id} registered with coordinator")
    
    def _handle_agent_message(self, agent_id: str, message: AgentMessage):
        """Handle incoming message for an agent"""
        agent = self._agents.get(agent_id)
        if not agent:
            return
        
        self._agent_status[agent_id] = 'processing'
        
        try:
            if hasattr(agent, 'handle_event'):
                result = agent.handle_event(message.event_type, message.payload)
                
                # If agent produces recommendation, publish it
                if result and isinstance(result, dict) and result.get('recommendations'):
                    self.bus.publish(
                        event_type=EventType.AGENT_RECOMMENDATION,
                        source_agent=agent_id,
                        payload=result,
                        priority=MessagePriority.MEDIUM,
                        correlation_id=message.message_id
                    )
        finally:
            self._agent_status[agent_id] = 'idle'
    
    def trigger_workflow(self, workflow_name: str, initial_payload: Dict):
        """Trigger a predefined workflow"""
        if workflow_name not in self._workflows:
            logger.warning(f"Unknown workflow: {workflow_name}")
            return
        
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
        agents = self._workflows[workflow_name]
        
        if not agents:
            return
        
        # Start workflow by notifying first agent
        first_agent = agents[0]
        self.bus.publish(
            event_type=EventType.AGENT_ACTION_REQUIRED,
            source_agent='AgentCoordinator',
            target_agent=first_agent,
            payload={
                'workflow_id': workflow_id,
                'workflow_name': workflow_name,
                'step': 1,
                'total_steps': len(agents),
                **initial_payload
            },
            priority=MessagePriority.HIGH,
            correlation_id=workflow_id
        )
        
        logger.info(f"Started workflow {workflow_name} ({workflow_id})")
        return workflow_id
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all registered agents"""
        return self._agent_status.copy()


# Global singleton instances
_message_bus: Optional[MessageBus] = None
_coordinator: Optional[AgentCoordinator] = None


def get_message_bus() -> MessageBus:
    """Get or create the global message bus"""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
        _message_bus.start_processor()
    return _message_bus


def get_coordinator() -> AgentCoordinator:
    """Get or create the global agent coordinator"""
    global _coordinator
    if _coordinator is None:
        _coordinator = AgentCoordinator(get_message_bus())
    return _coordinator


def shutdown():
    """Shutdown communication system"""
    global _message_bus, _coordinator
    if _message_bus:
        _message_bus.stop_processor()
    _message_bus = None
    _coordinator = None
