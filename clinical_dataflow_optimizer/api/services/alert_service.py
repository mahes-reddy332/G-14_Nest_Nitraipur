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
        self._last_evaluated_at: Optional[datetime] = None
        self._evaluation_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize alert service"""
        if self._initialized:
            return

        self._initialized = True
        await self._evaluate_alerts_if_needed(force=True)
        logger.info("Alert service initialized")

    async def _evaluate_alerts_if_needed(self, force: bool = False, study_id: Optional[str] = None) -> None:
        """Evaluate rules and generate alerts based on current data."""
        from api.config import get_initialized_data_service, get_settings

        now = datetime.utcnow()
        settings = get_settings()
        interval = max(30, settings.alert_check_interval)

        if not force and self._last_evaluated_at:
            elapsed = (now - self._last_evaluated_at).total_seconds()
            if elapsed < interval:
                return

        async with self._evaluation_lock:
            if not force and self._last_evaluated_at:
                elapsed = (datetime.utcnow() - self._last_evaluated_at).total_seconds()
                if elapsed < interval:
                    return

            data_service = await get_initialized_data_service()
            studies = await data_service.get_all_studies()

            if study_id:
                studies = [s for s in studies if s.get('study_id') == study_id]

            active_fingerprints = {
                a.get('details', {}).get('fingerprint')
                for a in self.alerts.values()
                if a.get('status') not in [AlertStatus.RESOLVED.value, AlertStatus.DISMISSED.value]
            }

            for study in studies:
                study_id_value = study.get('study_id')
                if not study_id_value:
                    continue

                total_patients = study.get('total_patients', 0) or 0
                dqi_score = float(study.get('dqi_score', 0) or 0)
                cleanliness_rate = float(study.get('cleanliness_rate', 0) or 0)
                open_queries = int(study.get('open_queries', 0) or 0)
                pending_saes = int(study.get('pending_saes', 0) or 0)
                uncoded_terms = int(study.get('uncoded_terms', 0) or 0)

                # Study-level alerts
                if dqi_score < 75:
                    severity = AlertSeverity.CRITICAL.value if dqi_score < 60 else AlertSeverity.HIGH.value
                    fingerprint = f"study:{study_id_value}:dqi:{severity}"
                    if fingerprint not in active_fingerprints:
                        await self.create_alert(
                            category=AlertCategory.DATA_QUALITY.value,
                            severity=severity,
                            title="Low Data Quality Index",
                            description=f"Study {study_id_value} has DQI {dqi_score:.1f}, below target.",
                            source="rule_engine",
                            affected_entity={"type": "study", "id": study_id_value},
                            details={
                                "metric": "dqi_score",
                                "value": dqi_score,
                                "threshold": 75,
                                "fingerprint": fingerprint,
                            }
                        )
                        active_fingerprints.add(fingerprint)

                if cleanliness_rate < 85:
                    severity = AlertSeverity.HIGH.value if cleanliness_rate < 75 else AlertSeverity.MEDIUM.value
                    fingerprint = f"study:{study_id_value}:cleanliness:{severity}"
                    if fingerprint not in active_fingerprints:
                        await self.create_alert(
                            category=AlertCategory.OPERATIONAL.value,
                            severity=severity,
                            title="Cleanliness Rate Below Target",
                            description=f"Study {study_id_value} cleanliness rate is {cleanliness_rate:.1f}%.",
                            source="rule_engine",
                            affected_entity={"type": "study", "id": study_id_value},
                            details={
                                "metric": "cleanliness_rate",
                                "value": cleanliness_rate,
                                "threshold": 85,
                                "fingerprint": fingerprint,
                            }
                        )
                        active_fingerprints.add(fingerprint)

                query_threshold_high = max(50, int(total_patients * 0.5))
                query_threshold_medium = max(20, int(total_patients * 0.2))
                if open_queries >= query_threshold_medium:
                    severity = AlertSeverity.HIGH.value if open_queries >= query_threshold_high else AlertSeverity.MEDIUM.value
                    fingerprint = f"study:{study_id_value}:queries:{severity}"
                    if fingerprint not in active_fingerprints:
                        await self.create_alert(
                            category=AlertCategory.OPERATIONAL.value,
                            severity=severity,
                            title="High Volume of Open Queries",
                            description=f"Study {study_id_value} has {open_queries} open queries.",
                            source="rule_engine",
                            affected_entity={"type": "study", "id": study_id_value},
                            details={
                                "metric": "open_queries",
                                "value": open_queries,
                                "threshold": query_threshold_medium,
                                "fingerprint": fingerprint,
                            }
                        )
                        active_fingerprints.add(fingerprint)

                if pending_saes > 0:
                    severity = AlertSeverity.CRITICAL.value if pending_saes >= 5 else AlertSeverity.HIGH.value
                    fingerprint = f"study:{study_id_value}:sae:{severity}"
                    if fingerprint not in active_fingerprints:
                        await self.create_alert(
                            category=AlertCategory.SAFETY.value,
                            severity=severity,
                            title=f"Pending SAE Reconciliation - {study_id_value}",
                            description=f"Study {study_id_value} has {pending_saes} pending SAE records.",
                            source="rule_engine",
                            affected_entity={"type": "study", "id": study_id_value},
                            details={
                                "metric": "pending_saes",
                                "value": pending_saes,
                                "threshold": 1,
                                "fingerprint": fingerprint,
                            }
                        )
                        active_fingerprints.add(fingerprint)

                if uncoded_terms > max(30, int(total_patients * 0.3)):
                    severity = AlertSeverity.MEDIUM.value
                    fingerprint = f"study:{study_id_value}:coding:{severity}"
                    if fingerprint not in active_fingerprints:
                        await self.create_alert(
                            category=AlertCategory.COMPLIANCE.value,
                            severity=severity,
                            title=f"Uncoded Terms - {study_id_value}",
                            description=f"Study {study_id_value} has {uncoded_terms} uncoded terms pending.",
                            source="rule_engine",
                            affected_entity={"type": "study", "id": study_id_value},
                            details={
                                "metric": "uncoded_terms",
                                "value": uncoded_terms,
                                "threshold": max(30, int(total_patients * 0.3)),
                                "fingerprint": fingerprint,
                            }
                        )
                        active_fingerprints.add(fingerprint)

                # Site-level alerts (top offenders)
                sites = await data_service.get_sites({"study_id": study_id_value}, "dqi_score", "asc")
                flagged_sites = [
                    s for s in sites
                    if float(s.get('dqi_score', 100) or 100) < 75 or int(s.get('open_queries', 0) or 0) > 20
                ]
                for site in flagged_sites[:3]:
                    site_id_value = site.get('site_id') or site.get('site_name')
                    if not site_id_value:
                        continue
                    site_dqi = float(site.get('dqi_score', 0) or 0)
                    site_queries = int(site.get('open_queries', 0) or 0)
                    if site_dqi < 75:
                        severity = AlertSeverity.HIGH.value if site_dqi < 60 else AlertSeverity.MEDIUM.value
                        fingerprint = f"site:{site_id_value}:dqi:{severity}"
                        if fingerprint not in active_fingerprints:
                            await self.create_alert(
                                category=AlertCategory.DATA_QUALITY.value,
                                severity=severity,
                                title=f"Site DQI Below Target - {site_id_value}",
                                description=f"Site {site_id_value} has DQI {site_dqi:.1f}.",
                                source="rule_engine",
                                affected_entity={"type": "site", "id": site_id_value},
                                details={
                                    "metric": "dqi_score",
                                    "value": site_dqi,
                                    "threshold": 75,
                                    "study_id": study_id_value,
                                    "fingerprint": fingerprint,
                                }
                            )
                            active_fingerprints.add(fingerprint)

                    if site_queries > 20:
                        severity = AlertSeverity.MEDIUM.value if site_queries < 50 else AlertSeverity.HIGH.value
                        fingerprint = f"site:{site_id_value}:queries:{severity}"
                        if fingerprint not in active_fingerprints:
                            await self.create_alert(
                                category=AlertCategory.OPERATIONAL.value,
                                severity=severity,
                                title="Site Query Backlog",
                                description=f"Site {site_id_value} has {site_queries} open queries.",
                                source="rule_engine",
                                affected_entity={"type": "site", "id": site_id_value},
                                details={
                                    "metric": "open_queries",
                                    "value": site_queries,
                                    "threshold": 20,
                                    "study_id": study_id_value,
                                    "fingerprint": fingerprint,
                                }
                            )
                            active_fingerprints.add(fingerprint)

            self._last_evaluated_at = datetime.utcnow()
    
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
        await self._evaluate_alerts_if_needed(study_id=(filters or {}).get('study_id'))
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
        await self._evaluate_alerts_if_needed(study_id=study_id)
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
        await self._evaluate_alerts_if_needed(study_id=study_id)
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
        await self._evaluate_alerts_if_needed(study_id=study_id)
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
