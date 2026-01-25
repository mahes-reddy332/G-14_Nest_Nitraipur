"""
Alert Service
Alert CRUD operations, acknowledgement/resolution workflow
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'


class AlertCategory(str, Enum):
    DATA_QUALITY = 'data_quality'
    SAFETY = 'safety'
    OPERATIONAL = 'operational'
    COMPLIANCE = 'compliance'
    SYSTEM = 'system'


class AlertStatus(str, Enum):
    NEW = 'new'
    ACKNOWLEDGED = 'acknowledged'
    IN_PROGRESS = 'in_progress'
    RESOLVED = 'resolved'
    DISMISSED = 'dismissed'


class AlertService:
    """
    Service for managing alerts and notifications
    """
    
    def __init__(self):
        self.alerts: Dict[str, Dict] = {}
        self._alert_counter = 0
        self._initialized = False
    
    async def initialize(self):
        """Initialize alert service with some sample alerts"""
        if self._initialized:
            return
        
        # Generate sample alerts
        sample_alerts = [
            {
                'category': AlertCategory.DATA_QUALITY.value,
                'severity': AlertSeverity.CRITICAL.value,
                'title': 'Critical DQI Drop Detected',
                'description': 'Site SITE_023 DQI dropped below 50% in the last 24 hours',
                'source': 'data_quality_monitor',
                'affected_entity': {'type': 'site', 'id': 'SITE_023'},
                'details': {'current_dqi': 48.5, 'previous_dqi': 72.3}
            },
            {
                'category': AlertCategory.SAFETY.value,
                'severity': AlertSeverity.HIGH.value,
                'title': 'SAE Pending Reconciliation',
                'description': '5 SAE records pending reconciliation for more than 48 hours',
                'source': 'sae_monitor',
                'affected_entity': {'type': 'sae_records', 'id': 'multiple'},
                'details': {'count': 5, 'oldest_pending_hours': 72}
            },
            {
                'category': AlertCategory.OPERATIONAL.value,
                'severity': AlertSeverity.MEDIUM.value,
                'title': 'Query Backlog Increasing',
                'description': 'Open queries increased by 25% in the last week',
                'source': 'query_monitor',
                'affected_entity': {'type': 'study', 'id': 'Study_1'},
                'details': {'current_count': 156, 'previous_count': 125}
            },
            {
                'category': AlertCategory.COMPLIANCE.value,
                'severity': AlertSeverity.HIGH.value,
                'title': 'Protocol Deviation Trend',
                'description': 'Increasing protocol deviations at 3 sites',
                'source': 'compliance_monitor',
                'affected_entity': {'type': 'sites', 'id': 'multiple'},
                'details': {'sites': ['SITE_012', 'SITE_034', 'SITE_056']}
            },
            {
                'category': AlertCategory.DATA_QUALITY.value,
                'severity': AlertSeverity.LOW.value,
                'title': 'Minor Data Entry Discrepancies',
                'description': '12 minor discrepancies detected in visit data',
                'source': 'data_validator',
                'affected_entity': {'type': 'visits', 'id': 'multiple'},
                'details': {'count': 12}
            }
        ]
        
        for alert_data in sample_alerts:
            await self.create_alert(**alert_data)
        
        self._initialized = True
        logger.info("Alert service initialized with sample alerts")
    
    async def create_alert(self,
                           category: str,
                           severity: str,
                           title: str,
                           description: str,
                           source: str,
                           affected_entity: Dict,
                           details: Optional[Dict] = None) -> Dict:
        """Create a new alert"""
        self._alert_counter += 1
        alert_id = f'ALT{self._alert_counter:05d}'
        
        alert = {
            'alert_id': alert_id,
            'category': category,
            'severity': severity,
            'status': AlertStatus.NEW.value,
            'title': title,
            'description': description,
            'source': source,
            'affected_entity': affected_entity,
            'details': details or {},
            'created_at': datetime.now().isoformat(),
            'updated_at': None,
            'acknowledged_by': None,
            'acknowledged_at': None,
            'resolved_at': None,
            'resolved_by': None,
            'resolution_notes': None,
            'actions_taken': []
        }
        
        self.alerts[alert_id] = alert
        logger.info(f"Created alert {alert_id}: {title}")
        
        return alert
    
    async def get_alerts(self,
                          filters: Optional[Dict[str, Any]] = None,
                          limit: int = 50,
                          offset: int = 0) -> List[Dict]:
        """Get alerts with optional filters
        
        Args:
            filters: Dict with optional keys: study_id, site_id, category, severity, status, unacknowledged_only
            limit: Maximum number of alerts to return
            offset: Number of alerts to skip
        """
        alerts = list(self.alerts.values())
        filters = filters or {}
        
        # Apply filters
        if filters.get('status'):
            alerts = [a for a in alerts if a.get('status') == filters['status']]
        
        if filters.get('severity'):
            alerts = [a for a in alerts if a.get('severity') == filters['severity']]
        
        if filters.get('category'):
            alerts = [a for a in alerts if a.get('category') == filters['category']]
        
        if filters.get('study_id'):
            alerts = [a for a in alerts if a.get('affected_entity', {}).get('type') == 'study' 
                     and a.get('affected_entity', {}).get('id') == filters['study_id']]
        
        if filters.get('site_id'):
            alerts = [a for a in alerts if a.get('affected_entity', {}).get('type') == 'site' 
                     and a.get('affected_entity', {}).get('id') == filters['site_id']]
        
        if filters.get('unacknowledged_only'):
            alerts = [a for a in alerts if a.get('status') == 'new']
        
        # Sort by severity and creation time
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        alerts.sort(key=lambda x: (
            severity_order.get(x.get('severity', 'low'), 3),
            x.get('created_at', '')
        ))
        
        return alerts[offset:offset + limit]
    
    async def get_alert(self, alert_id: str) -> Optional[Dict]:
        """Get a specific alert by ID"""
        return self.alerts.get(alert_id)
    
    async def get_alert_summary(self,
                                  study_id: Optional[str] = None,
                                  site_id: Optional[str] = None) -> Dict:
        """Get summary of alerts by status and severity
        
        Args:
            study_id: Optional filter by study ID
            site_id: Optional filter by site ID
        """
        alerts = list(self.alerts.values())
        
        # Apply entity filters if provided
        if study_id:
            alerts = [a for a in alerts if a.get('affected_entity', {}).get('type') == 'study' 
                     and a.get('affected_entity', {}).get('id') == study_id]
        if site_id:
            alerts = [a for a in alerts if a.get('affected_entity', {}).get('type') == 'site' 
                     and a.get('affected_entity', {}).get('id') == site_id]
        
        # Count by status
        by_status = {status.value: 0 for status in AlertStatus}
        for alert in alerts:
            status = alert.get('status', 'new')
            if status in by_status:
                by_status[status] += 1
        
        # Count by severity
        by_severity = {severity.value: 0 for severity in AlertSeverity}
        for alert in alerts:
            if alert.get('status') not in ['resolved', 'dismissed']:
                severity = alert.get('severity', 'low')
                if severity in by_severity:
                    by_severity[severity] += 1
        
        # Count by category
        by_category = {category.value: 0 for category in AlertCategory}
        for alert in alerts:
            if alert.get('status') not in ['resolved', 'dismissed']:
                category = alert.get('category', 'system')
                if category in by_category:
                    by_category[category] += 1
        
        # Count unacknowledged and critical unresolved
        unacknowledged = sum(1 for a in alerts if a.get('status') == 'new')
        critical_unresolved = sum(1 for a in alerts 
                                   if a.get('severity') == 'critical' 
                                   and a.get('status') not in ['resolved', 'dismissed'])
        
        return {
            'total': len(alerts),
            'by_severity': by_severity,
            'by_category': by_category,
            'by_status': by_status,
            'unacknowledged': unacknowledged,
            'critical_unresolved': critical_unresolved
        }
    
    async def get_critical_alerts(self,
                                    study_id: Optional[str] = None,
                                    unresolved_only: bool = True) -> List[Dict]:
        """Get all critical severity alerts
        
        Args:
            study_id: Optional filter by study ID
            unresolved_only: If True, only return unresolved alerts
        """
        alerts = [
            a for a in self.alerts.values()
            if a.get('severity') == 'critical'
        ]
        
        if unresolved_only:
            alerts = [a for a in alerts if a.get('status') not in ['resolved', 'dismissed']]
        
        if study_id:
            alerts = [a for a in alerts if a.get('affected_entity', {}).get('type') == 'study' 
                     and a.get('affected_entity', {}).get('id') == study_id]
        
        return alerts
    
    async def get_recent_alerts(self,
                                  hours: int = 24,
                                  study_id: Optional[str] = None,
                                  limit: int = 50) -> List[Dict]:
        """Get alerts from the last N hours
        
        Args:
            hours: Look back period in hours
            study_id: Optional filter by study ID
            limit: Maximum number of alerts to return
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = []
        for alert in self.alerts.values():
            created_at = datetime.fromisoformat(alert.get('created_at', datetime.now().isoformat()))
            if created_at >= cutoff:
                if study_id:
                    if (alert.get('affected_entity', {}).get('type') == 'study' 
                        and alert.get('affected_entity', {}).get('id') == study_id):
                        recent.append(alert)
                else:
                    recent.append(alert)
        
        recent.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return recent[:limit]
    
    async def acknowledge_alert(self,
                                  alert_id: str,
                                  acknowledged_by: str,
                                  note: Optional[str] = None) -> Optional[Dict]:
        """Acknowledge an alert
        
        Args:
            alert_id: The alert ID to acknowledge
            acknowledged_by: User acknowledging the alert
            note: Optional acknowledgement note
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return None
        
        alert['status'] = AlertStatus.ACKNOWLEDGED.value
        alert['acknowledged_by'] = acknowledged_by
        alert['acknowledged_at'] = datetime.now().isoformat()
        alert['updated_at'] = datetime.now().isoformat()
        if note:
            alert['acknowledgement_note'] = note
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return alert
    
    async def resolve_alert(self, 
                            alert_id: str, 
                            resolved_by: str,
                            resolution_notes: Optional[str] = None,
                            actions_taken: Optional[List[str]] = None) -> Optional[Dict]:
        """Resolve an alert
        
        Args:
            alert_id: The alert ID to resolve
            resolved_by: User resolving the alert
            resolution_notes: Optional resolution notes
            actions_taken: Optional list of actions taken
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return None
        
        alert['status'] = AlertStatus.RESOLVED.value
        alert['resolved_by'] = resolved_by
        alert['resolved_at'] = datetime.now().isoformat()
        alert['resolution_notes'] = resolution_notes
        alert['updated_at'] = datetime.now().isoformat()
        if actions_taken:
            alert['actions_taken'].extend(actions_taken)
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return alert
    
    async def dismiss_alert(self, alert_id: str, dismissed_by: str, reason: str) -> Optional[Dict]:
        """Dismiss an alert"""
        alert = self.alerts.get(alert_id)
        if not alert:
            return None
        
        alert['status'] = AlertStatus.DISMISSED.value
        alert['resolved_by'] = dismissed_by
        alert['resolved_at'] = datetime.now().isoformat()
        alert['resolution_notes'] = f"Dismissed: {reason}"
        alert['updated_at'] = datetime.now().isoformat()
        
        logger.info(f"Alert {alert_id} dismissed by {dismissed_by}")
        return alert
    
    async def add_action(self, alert_id: str, action: str, performed_by: str) -> Optional[Dict]:
        """Add an action taken on an alert"""
        alert = self.alerts.get(alert_id)
        if not alert:
            return None
        
        action_entry = {
            'action': action,
            'performed_by': performed_by,
            'performed_at': datetime.now().isoformat()
        }
        
        alert['actions_taken'].append(action_entry)
        alert['updated_at'] = datetime.now().isoformat()
        
        if alert['status'] == AlertStatus.ACKNOWLEDGED.value:
            alert['status'] = AlertStatus.IN_PROGRESS.value
        
        logger.info(f"Added action to alert {alert_id}: {action}")
        return alert
    
    async def get_alert_history(self, 
                                 entity_type: Optional[str] = None,
                                 entity_id: Optional[str] = None,
                                 days: int = 30,
                                 category: Optional[str] = None) -> List[Dict]:
        """Get alert history for the past N days
        
        Args:
            entity_type: Optional entity type filter (study, site, patient)
            entity_id: Optional entity ID filter
            days: Number of days to look back
            category: Optional category filter
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        history = []
        for alert in self.alerts.values():
            created_at = datetime.fromisoformat(alert.get('created_at', datetime.now().isoformat()))
            if created_at >= cutoff:
                # Apply entity filter if provided
                if entity_type and entity_id:
                    affected = alert.get('affected_entity', {})
                    if affected.get('type') != entity_type or affected.get('id') != entity_id:
                        continue
                
                if category is None or alert.get('category') == category:
                    history.append(alert)
        
        history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return history
    
    async def get_alert_trends(self,
                                 study_id: Optional[str] = None,
                                 days: int = 7,
                                 group_by: str = 'day') -> Dict:
        """Get alert trends over the past N days
        
        Args:
            study_id: Optional filter by study ID
            days: Number of days to analyze
            group_by: Grouping method (day, week, category, severity)
        """
        # Note: study_id and group_by can be used to customize trending
        # For now, we implement basic daily trends
        trends = {}
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            trends[date] = {
                'total': 0,
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'resolved': 0
            }
        
        for alert in self.alerts.values():
            created_at = datetime.fromisoformat(alert.get('created_at', datetime.now().isoformat()))
            date_key = created_at.strftime('%Y-%m-%d')
            
            if date_key in trends:
                trends[date_key]['total'] += 1
                severity = alert.get('severity', 'low')
                if severity in trends[date_key]:
                    trends[date_key][severity] += 1
                
                if alert.get('status') == 'resolved':
                    trends[date_key]['resolved'] += 1
        
        return trends
    
    async def get_escalated_alerts(self) -> List[Dict]:
        """Get alerts that need escalation"""
        escalated = []
        now = datetime.now()
        
        for alert in self.alerts.values():
            if alert.get('status') in ['resolved', 'dismissed']:
                continue
            
            created_at = datetime.fromisoformat(alert.get('created_at', now.isoformat()))
            age_hours = (now - created_at).total_seconds() / 3600
            
            # Escalation rules
            should_escalate = False
            if alert.get('severity') == 'critical' and age_hours > 4 and alert.get('status') == 'new':
                should_escalate = True
            elif alert.get('severity') == 'high' and age_hours > 24 and alert.get('status') == 'new':
                should_escalate = True
            elif age_hours > 72 and alert.get('status') not in ['resolved', 'dismissed']:
                should_escalate = True
            
            if should_escalate:
                alert_copy = alert.copy()
                alert_copy['escalation_reason'] = f'Alert open for {age_hours:.1f} hours'
                escalated.append(alert_copy)
        
        return escalated
