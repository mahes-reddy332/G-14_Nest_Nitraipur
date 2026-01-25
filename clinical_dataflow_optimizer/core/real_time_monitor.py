"""
Real-Time Data Monitor for Neural Clinical Data Mesh
Continuously monitors data streams and updates patient status dynamically
"""

import time
import threading
import logging
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field
import asyncio
import websockets
import json

from config.settings import DEFAULT_AGENT_CONFIG
from core.data_ingestion import ClinicalDataIngester
from core.data_quality_index import DataQualityIndexCalculator
from models.data_models import DigitalPatientTwin, SiteMetrics
from agents.agent_framework import ReconciliationAgent, CodingAgent, SiteLiaisonAgent
from core.longcat_integration import longcat_client
from core.quality_cockpit import CleanPatientStatus

logger = logging.getLogger(__name__)


@dataclass
class DataStream:
    """Represents a data stream source"""
    name: str
    path: Path
    last_modified: datetime = None
    check_interval: int = 60  # seconds
    data_type: str = "excel"  # excel, csv, json, api
    api_endpoint: Optional[str] = None
    headers: Dict = field(default_factory=dict)


@dataclass
class PatientStatusUpdate:
    """Tracks patient status changes"""
    subject_id: str
    previous_status: bool
    new_status: bool
    timestamp: datetime
    trigger_reason: str
    longcat_explanation: str = ""
    blocking_factors: List[str] = field(default_factory=list)
    cleanliness_score: float = 0.0


@dataclass
class OperationalVelocityMetrics:
    """Tracks operational velocity across sites"""
    site_id: str
    avg_query_resolution_time: float  # hours
    resolution_velocity_trend: float  # improvement rate
    operational_velocity_index: float  # composite score
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class LiveCleanlinessRule:
    """Dynamic cleanliness rule with real-time evaluation"""
    rule_name: str
    condition: str  # Python expression to evaluate
    weight: float = 1.0
    blocking: bool = True  # If True, failure blocks clean status
    description: str = ""
    last_evaluated: datetime = None
    evaluation_result: bool = True


class LiveCleanlinessEngine:
    """
    Dynamic boolean logic engine for real-time patient cleanliness calculation
    """

    def __init__(self):
        self.rules: Dict[str, LiveCleanlinessRule] = {
            'no_missing_visits': LiveCleanlinessRule(
                rule_name='No Missing Visits',
                condition='twin.missing_visits == 0',
                weight=0.25,
                blocking=True,
                description='Patient has no missing scheduled visits'
            ),
            'no_open_queries': LiveCleanlinessRule(
                rule_name='No Open Queries',
                condition='twin.open_queries == 0',
                weight=0.25,
                blocking=True,
                description='Patient has no outstanding data queries'
            ),
            'no_uncoded_terms': LiveCleanlinessRule(
                rule_name='No Uncoded Terms',
                condition='twin.uncoded_terms == 0',
                weight=0.20,
                blocking=True,
                description='All adverse event terms are properly coded'
            ),
            'sae_reconciliation': LiveCleanlinessRule(
                rule_name='SAE Reconciliation Confirmed',
                condition='twin.sae_reconciliation_confirmed',
                weight=0.20,
                blocking=True,
                description='SAE records match between safety and clinical databases'
            ),
            'data_verification': LiveCleanlinessRule(
                rule_name='Data Verification Complete',
                condition='twin.verification_percentage >= 75.0',
                weight=0.10,
                blocking=False,
                description='Required data verification percentage achieved'
            )
        }

    def evaluate_patient_cleanliness(self, twin: DigitalPatientTwin) -> Dict[str, Any]:
        """
        Evaluate all cleanliness rules for a patient
        Returns detailed status with blocking factors
        """
        results = {}
        total_score = 0.0
        max_score = sum(rule.weight for rule in self.rules.values())
        blocking_factors = []

        for rule_name, rule in self.rules.items():
            try:
                # Evaluate the condition
                condition_result = eval(rule.condition, {'twin': twin})
                rule.evaluation_result = condition_result
                rule.last_evaluated = datetime.now()

                results[rule_name] = {
                    'passed': condition_result,
                    'weight': rule.weight,
                    'blocking': rule.blocking,
                    'description': rule.description
                }

                if condition_result:
                    total_score += rule.weight
                elif rule.blocking:
                    blocking_factors.append(rule.description)

            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
                results[rule_name] = {
                    'passed': False,
                    'weight': rule.weight,
                    'blocking': rule.blocking,
                    'description': f"Error: {str(e)}"
                }
                if rule.blocking:
                    blocking_factors.append(f"Rule evaluation error: {rule_name}")

        cleanliness_percentage = (total_score / max_score * 100) if max_score > 0 else 0
        is_clean = len(blocking_factors) == 0

        return {
            'is_clean': is_clean,
            'cleanliness_score': cleanliness_percentage,
            'blocking_factors': blocking_factors,
            'rule_results': results,
            'total_score': total_score,
            'max_score': max_score
        }


class RealTimeDataMonitor:
    """
    Real-time data monitoring system for continuous patient status updates
    """

    def __init__(self, base_path: Path, check_interval: int = 300):  # 5 minutes default
        self.base_path = Path(base_path)
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Enhanced data streams with more frequent checking
        self.data_streams: Dict[str, DataStream] = {
            'cpid_metrics': DataStream(
                name='CPID_EDC_Metrics',
                path=self.base_path / "**/CPID_EDC_Metrics*.xlsx",
                check_interval=60  # Check every minute for real-time
            ),
            'sae_dashboard': DataStream(
                name='SAE_Dashboard',
                path=self.base_path / "**/eSAE*.xlsx",
                check_interval=30  # Check every 30 seconds
            ),
            'visit_tracker': DataStream(
                name='Visit_Tracker',
                path=self.base_path / "**/Visit_Projection*.xlsx",
                check_interval=120  # Check every 2 minutes
            ),
            'meddra_coding': DataStream(
                name='MedDRA_Coding',
                path=self.base_path / "**/GlobalCodingReport_MedDRA*.xlsx",
                check_interval=180  # Check every 3 minutes
            ),
            'whodra_coding': DataStream(
                name='WHODRA_Coding',
                path=self.base_path / "**/GlobalCodingReport_WHODD*.xlsx",
                check_interval=180  # Check every 3 minutes
            ),
            'audit_trail': DataStream(
                name='Audit_Trail',
                path=self.base_path / "**/AuditTrail*.xlsx",
                check_interval=60  # Check every minute for manipulation detection
            )
        }

        # Initialize enhanced components
        self.data_ingester = ClinicalDataIngester(self.base_path)
        self.dqi_calculator = DataQualityIndexCalculator()
        self.cleanliness_engine = LiveCleanlinessEngine()

        # Enhanced agent swarm
        self.agents = {
            'rex': ReconciliationAgent(),
            'codex': CodingAgent(),
            'lia': SiteLiaisonAgent()
        }

        # Status tracking with enhanced metrics
        self.patient_status_history: Dict[str, List[PatientStatusUpdate]] = {}
        self.last_check_times: Dict[str, datetime] = {}
        self.alert_callbacks: List[Callable] = []

        # New real-time components
        self.operational_velocity_metrics: Dict[str, OperationalVelocityMetrics] = {}
        self.websocket_clients: set = set()
        self.last_broadcast_time: datetime = datetime.now()

        # WebSocket server for real-time UI updates
        self.websocket_thread: Optional[threading.Thread] = None
        self.websocket_server = None

        # Initialize last modified times
        self._initialize_stream_states()

    def add_alert_callback(self, callback: Callable):
        """Add a callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable):
        """Remove a callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def _initialize_stream_states(self):
        """Initialize last modified times for all data streams"""
        for stream_name, stream in self.data_streams.items():
            try:
                # Find the most recent file matching the pattern
                matching_files = list(self.base_path.glob(str(stream.path).replace('**/', '')))
                if matching_files:
                    latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
                    stream.last_modified = datetime.fromtimestamp(latest_file.stat().st_mtime)
                    self.last_check_times[stream_name] = datetime.now()
                    logger.info(f"Initialized {stream_name}: {latest_file.name} @ {stream.last_modified}")
            except Exception as e:
                logger.warning(f"Failed to initialize {stream_name}: {e}")

    def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time UI updates"""
        def websocket_handler():
            async def handler(websocket, path):
                self.websocket_clients.add(websocket)
                logger.info(f"WebSocket client connected: {len(self.websocket_clients)} total")

                try:
                    async for message in websocket:
                        # Handle incoming messages from UI
                        try:
                            data = json.loads(message)
                            if data.get('type') == 'ping':
                                await websocket.send(json.dumps({'type': 'pong'}))
                        except json.JSONDecodeError:
                            pass
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.websocket_clients.discard(websocket)
                    logger.info(f"WebSocket client disconnected: {len(self.websocket_clients)} remaining")

            # Start the WebSocket server
            start_server = websockets.serve(handler, host, port)
            asyncio.run(start_server.wait_closed())

        self.websocket_thread = threading.Thread(target=websocket_handler, daemon=True)
        self.websocket_thread.start()
        logger.info(f"WebSocket server started on ws://{host}:{port}")

    def broadcast_update(self, update_type: str, data: Dict):
        """Broadcast real-time update to all connected WebSocket clients"""
        if not self.websocket_clients:
            return

        message = {
            'type': update_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }

        # Send to all clients (non-blocking)
        for client in self.websocket_clients.copy():
            try:
                asyncio.create_task(client.send(json.dumps(message)))
            except Exception as e:
                logger.warning(f"Failed to send update to client: {e}")
                self.websocket_clients.discard(client)

    def _evaluate_live_cleanliness(self, twins: List[DigitalPatientTwin]) -> Dict[str, Dict]:
        """Evaluate live cleanliness for all patients"""
        cleanliness_results = {}

        for twin in twins:
            result = self.cleanliness_engine.evaluate_patient_cleanliness(twin)

            # Update twin with live cleanliness data
            twin.clean_status = result['is_clean']
            twin.clean_percentage = result['cleanliness_score']
            twin.blocking_factors = result['blocking_factors']

            cleanliness_results[twin.subject_id] = result

            # Check for status changes
            self._check_status_change(twin, result)

        return cleanliness_results

    def _check_status_change(self, twin: DigitalPatientTwin, cleanliness_result: Dict):
        """Check if patient status has changed and trigger alerts"""
        subject_id = twin.subject_id
        new_status = cleanliness_result['is_clean']
        cleanliness_score = cleanliness_result['cleanliness_score']
        blocking_factors = cleanliness_result['blocking_factors']

        # Get previous status
        history = self.patient_status_history.get(subject_id, [])
        previous_status = history[-1].new_status if history else None

        if previous_status != new_status:
            # Status change detected
            trigger_reason = "Cleanliness rule evaluation"
            if blocking_factors:
                trigger_reason += f": {', '.join(blocking_factors[:2])}"

            # Get LongCat explanation for the status change
            longcat_explanation = ""
            try:
                context = f"Patient {subject_id} status change: {'Clean' if new_status else 'Dirty'}"
                task = "Explain the reason for this status change and suggest next actions"
                data = {
                    'previous_status': 'Clean' if previous_status else 'Dirty',
                    'new_status': 'Clean' if new_status else 'Dirty',
                    'blocking_factors': blocking_factors,
                    'cleanliness_score': cleanliness_score
                }
                longcat_explanation = longcat_client.generate_agent_reasoning(context, task, data)
            except Exception as e:
                logger.warning(f"LongCat explanation failed: {e}")

            # Create status update record
            update = PatientStatusUpdate(
                subject_id=subject_id,
                previous_status=previous_status,
                new_status=new_status,
                timestamp=datetime.now(),
                trigger_reason=trigger_reason,
                longcat_explanation=longcat_explanation,
                blocking_factors=blocking_factors,
                cleanliness_score=cleanliness_score
            )

            # Store in history
            if subject_id not in self.patient_status_history:
                self.patient_status_history[subject_id] = []
            self.patient_status_history[subject_id].append(update)

            # Broadcast real-time update
            self.broadcast_update('patient_status_change', {
                'subject_id': subject_id,
                'previous_status': previous_status,
                'new_status': new_status,
                'cleanliness_score': cleanliness_score,
                'blocking_factors': blocking_factors,
                'longcat_explanation': longcat_explanation,
                'timestamp': update.timestamp.isoformat()
            })

            # Trigger agent actions for status changes
            self._trigger_agent_response(update, twin)

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

    def _trigger_agent_response(self, status_update: PatientStatusUpdate, twin: DigitalPatientTwin):
        """Trigger appropriate agent responses based on status changes"""
        if status_update.new_status == False:  # Patient became dirty
            # Trigger Rex for reconciliation issues
            if any('reconciliation' in factor.lower() for factor in status_update.blocking_factors):
                try:
                    rex_recommendations = self.agents['rex'].analyze([twin])
                    if rex_recommendations:
                        self.broadcast_update('agent_recommendations', {
                            'agent': 'rex',
                            'recommendations': [r.__dict__ for r in rex_recommendations[:3]],  # Top 3
                            'trigger': 'patient_status_change'
                        })
                except Exception as e:
                    logger.error(f"Rex agent analysis failed: {e}")

            # Trigger Lia for visit compliance issues
            if any('visit' in factor.lower() for factor in status_update.blocking_factors):
                try:
                    lia_recommendations = self.agents['lia'].analyze([twin])
                    if lia_recommendations:
                        self.broadcast_update('agent_recommendations', {
                            'agent': 'lia',
                            'recommendations': [r.__dict__ for r in lia_recommendations[:3]],  # Top 3
                            'trigger': 'patient_status_change'
                        })
                except Exception as e:
                    logger.error(f"Lia agent analysis failed: {e}")

    def calculate_operational_velocity_metrics(self, site_metrics: Dict[str, SiteMetrics]):
        """Calculate operational velocity metrics for all sites"""
        for site_id, metrics in site_metrics.items():
            # Calculate average query resolution time (mock data for now)
            avg_resolution_time = 48.0  # hours - would be calculated from actual data

            # Calculate resolution velocity trend (improvement rate)
            velocity_trend = 0.05  # 5% improvement - would be calculated from historical data

            # Calculate composite OVI score
            ovi_score = min(100, max(0, 100 - (avg_resolution_time / 24) + (velocity_trend * 1000)))

            self.operational_velocity_metrics[site_id] = OperationalVelocityMetrics(
                site_id=site_id,
                avg_query_resolution_time=avg_resolution_time,
                resolution_velocity_trend=velocity_trend,
                operational_velocity_index=ovi_score
            )

    def get_operational_velocity_ranking(self) -> List[Dict]:
        """Get sites ranked by operational velocity"""
        metrics = list(self.operational_velocity_metrics.values())
        metrics.sort(key=lambda x: x.operational_velocity_index, reverse=True)

        return [{
            'site_id': m.site_id,
            'ovi_score': round(m.operational_velocity_index, 1),
            'avg_resolution_hours': round(m.avg_query_resolution_time, 1),
            'velocity_trend': f"{m.resolution_velocity_trend*100:+.1f}%",
            'rank': idx + 1
        } for idx, m in enumerate(metrics)]

    def _check_stream_updates(self, stream: DataStream) -> bool:
        """Check if a data stream has been updated"""
        try:
            matching_files = list(self.base_path.glob(str(stream.path).replace('**/', '')))
            if not matching_files:
                return False

            latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
            current_modified = datetime.fromtimestamp(latest_file.stat().st_mtime)

            if stream.last_modified is None or current_modified > stream.last_modified:
                stream.last_modified = current_modified
                logger.info(f"Data update detected: {stream.name} - {latest_file.name}")
                return True

            return False
        except Exception as e:
            logger.error(f"Error checking stream {stream.name}: {e}")
            return False

    def _process_data_updates(self):
        """Process updates from all data streams"""
        updated_streams = []

        for stream_name, stream in self.data_streams.items():
            if self._check_stream_updates(stream):
                updated_streams.append(stream_name)

        if updated_streams:
            logger.info(f"Processing updates from: {', '.join(updated_streams)}")

            # Re-ingest data
            try:
                studies = self.data_ingester.discover_studies()
                for study in studies:
                    self.data_ingester.load_study_data(study)

                # Update patient statuses
                self._update_patient_statuses(updated_streams)

            except Exception as e:
                logger.error(f"Error processing data updates: {e}")

    def _update_patient_statuses(self, updated_streams: List[str]):
        """Update patient statuses based on new data"""
        try:
            # Get current patient data
            for study_id, study_data in self.data_ingester.studies.items():
                if 'cpid_metrics' not in study_data:
                    continue

                df = study_data['cpid_metrics']

                for _, patient_row in df.iterrows():
                    subject_id = str(patient_row.get('Subject ID', ''))
                    if not subject_id:
                        continue

                    # Calculate current clean status
                    current_status = self._calculate_patient_clean_status(subject_id, study_data)

                    # Check for status change
                    previous_status = self._get_last_patient_status(subject_id)

                    if previous_status != current_status:
                        # Status changed - create update record
                        update = PatientStatusUpdate(
                            subject_id=subject_id,
                            previous_status=previous_status,
                            new_status=current_status,
                            timestamp=datetime.now(),
                            trigger_reason=f"Data update in: {', '.join(updated_streams)}"
                        )

                        # Generate LongCat explanation
                        update.longcat_explanation = self._generate_status_change_explanation(
                            subject_id, previous_status, current_status, study_data
                        )

                        # Store update
                        if subject_id not in self.patient_status_history:
                            self.patient_status_history[subject_id] = []
                        self.patient_status_history[subject_id].append(update)

                        # Trigger alerts
                        self._trigger_alerts(update)

                        logger.info(f"Patient {subject_id} status changed: {previous_status} -> {current_status}")

        except Exception as e:
            logger.error(f"Error updating patient statuses: {e}")

    def _calculate_patient_clean_status(self, subject_id: str, study_data: Dict) -> bool:
        """Calculate clean patient status based on current data"""
        try:
            # Get patient metrics
            cpid_df = study_data.get('cpid_metrics', pd.DataFrame())
            patient_data = cpid_df[cpid_df['Subject ID'] == subject_id]

            if patient_data.empty:
                return False

            patient = patient_data.iloc[0]

            # Check blocking criteria
            open_queries = patient.get('# Open Queries', 0) or 0
            uncoded_terms = patient.get('# Uncoded Terms', 0) or 0
            missing_visits = patient.get('# Missing Visits', 0) or 0
            reconciliation_issues = patient.get('# SAE Reconciliation Issues', 0) or 0

            # Apply clean thresholds
            from config.settings import DEFAULT_CLEAN_THRESHOLDS

            if (open_queries > DEFAULT_CLEAN_THRESHOLDS.max_open_queries or
                uncoded_terms > DEFAULT_CLEAN_THRESHOLDS.max_uncoded_terms or
                missing_visits > 0 or
                reconciliation_issues > DEFAULT_CLEAN_THRESHOLDS.max_reconciliation_issues):
                return False

            return True

        except Exception as e:
            logger.error(f"Error calculating status for {subject_id}: {e}")
            return False

    def _get_last_patient_status(self, subject_id: str) -> bool:
        """Get the last known status for a patient"""
        if subject_id in self.patient_status_history and self.patient_status_history[subject_id]:
            return self.patient_status_history[subject_id][-1].new_status
        return False  # Default to not clean

    def _generate_status_change_explanation(self, subject_id: str, old_status: bool,
                                          new_status: bool, study_data: Dict) -> str:
        """Generate explanation for status change using LongCat"""
        try:
            status_names = {True: "CLEAN", False: "DIRTY"}
            context = f"Patient {subject_id} status changed from {status_names[old_status]} to {status_names[new_status]}"

            data_summary = {
                'subject_id': subject_id,
                'old_status': status_names[old_status],
                'new_status': status_names[new_status],
                'timestamp': datetime.now().isoformat()
            }

            # Add relevant metrics
            cpid_df = study_data.get('cpid_metrics', pd.DataFrame())
            patient_data = cpid_df[cpid_df['Subject ID'] == subject_id]
            if not patient_data.empty:
                patient = patient_data.iloc[0]
                data_summary.update({
                    'open_queries': patient.get('# Open Queries', 0),
                    'uncoded_terms': patient.get('# Uncoded Terms', 0),
                    'missing_visits': patient.get('# Missing Visits', 0),
                    'reconciliation_issues': patient.get('# SAE Reconciliation Issues', 0)
                })

            return longcat_client.explain_anomaly(data_summary, context)

        except Exception as e:
            logger.error(f"Failed to generate status change explanation: {e}")
            return f"Status changed from {old_status.name} to {new_status.name} due to data updates"

    def _trigger_alerts(self, update: PatientStatusUpdate):
        """Trigger alert callbacks for status changes"""
        for callback in self.alert_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def start_monitoring(self):
        """Start the real-time monitoring thread"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Real-time data monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Real-time data monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._process_data_updates()

                # Run agent analysis periodically (every 30 minutes)
                if datetime.now().minute % 30 == 0 and datetime.now().second < 10:
                    self._run_agent_analysis()

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

            time.sleep(self.check_interval)

    def _run_agent_analysis(self):
        """Run periodic agent analysis"""
        try:
            for agent_name, agent in self.agents.items():
                recommendations = agent.analyze(self.data_ingester.studies)
                if recommendations:
                    logger.info(f"{agent_name} generated {len(recommendations)} recommendations")

                    # Process recommendations with LongCat enhancement
                    for rec in recommendations:
                        enhanced_rationale = agent.enhance_reasoning_with_longcat(
                            f"Agent {agent_name} recommendation analysis",
                            f"Review and enhance recommendation: {rec.title}",
                            {
                                'subject_id': rec.subject_id,
                                'action_type': rec.action_type.value,
                                'priority': rec.priority.name,
                                'original_rationale': rec.rationale
                            }
                        )
                        rec.rationale = f"{rec.rationale}\n\nAI Enhanced Analysis: {enhanced_rationale}"

        except Exception as e:
            logger.error(f"Agent analysis failed: {e}")

    def get_patient_status_history(self, subject_id: str) -> List[PatientStatusUpdate]:
        """Get status history for a patient"""
        return self.patient_status_history.get(subject_id, [])

    def get_current_status_summary(self) -> Dict[str, Any]:
        """Get summary of current patient statuses"""
        summary = {
            'total_patients': len(self.patient_status_history),
            'clean_patients': 0,
            'dirty_patients': 0,
            'recent_changes': []
        }

        for subject_id, history in self.patient_status_history.items():
            if history:
                latest = history[-1]
                if latest.new_status == CleanPatientStatus.CLEAN:
                    summary['clean_patients'] += 1
                elif latest.new_status == CleanPatientStatus.DIRTY:
                    summary['dirty_patients'] += 1

                # Add recent changes (last 24 hours)
                if datetime.now() - latest.timestamp < timedelta(hours=24):
                    summary['recent_changes'].append({
                        'subject_id': subject_id,
                        'change': f"{latest.previous_status.name} -> {latest.new_status.name}",
                        'reason': latest.trigger_reason,
                        'timestamp': latest.timestamp.isoformat()
                    })

        return summary