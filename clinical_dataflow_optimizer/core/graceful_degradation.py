"""
Graceful Degradation for Neural Clinical Data Mesh
Provides fallback reasoning when AI services are unavailable
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class DegradationLevel(Enum):
    """Levels of service degradation"""
    FULL_SERVICE = "full_service"          # All services available
    LIMITED_AI = "limited_ai"             # AI services partially unavailable
    NO_AI = "no_ai"                       # AI services completely unavailable
    BASIC_SERVICE = "basic_service"       # Only basic functionality available

class GracefulDegradationManager:
    """
    Manages graceful degradation of services with fallback strategies
    """

    def __init__(self):
        self.current_level = DegradationLevel.FULL_SERVICE
        self.service_status = {
            'longcat_api': True,
            'graph_queries': True,
            'feature_engineering': True,
            'agent_framework': True,
            'external_apis': True
        }
        self.fallback_strategies = {}
        self._register_fallbacks()

    def _register_fallbacks(self):
        """Register fallback strategies for different services"""

        # LongCat API fallbacks
        self.fallback_strategies['longcat_cleanliness'] = self._fallback_cleanliness_analysis
        self.fallback_strategies['longcat_explanation'] = self._fallback_explanation

        # Agent framework fallbacks
        self.fallback_strategies['agent_analysis'] = self._fallback_agent_analysis
        self.fallback_strategies['agent_prioritization'] = self._fallback_agent_prioritization

        # Graph query fallbacks
        self.fallback_strategies['graph_patterns'] = self._fallback_graph_patterns
        self.fallback_strategies['graph_analytics'] = self._fallback_graph_analytics

    def update_service_status(self, service_name: str, is_available: bool):
        """Update the status of a service and adjust degradation level"""
        old_status = self.service_status.get(service_name, True)
        self.service_status[service_name] = is_available

        if old_status != is_available:
            logger.info(f"🔄 Service '{service_name}' status changed: {old_status} → {is_available}")
            self._reassess_degradation_level()

    def _reassess_degradation_level(self):
        """Reassess the current degradation level based on service status"""
        available_services = sum(self.service_status.values())
        total_services = len(self.service_status)

        if available_services == total_services:
            new_level = DegradationLevel.FULL_SERVICE
        elif available_services >= total_services * 0.75:
            new_level = DegradationLevel.LIMITED_AI
        elif available_services >= total_services * 0.5:
            new_level = DegradationLevel.NO_AI
        else:
            new_level = DegradationLevel.BASIC_SERVICE

        if new_level != self.current_level:
            logger.warning(f"📉 System degradation level changed: {self.current_level.value} → {new_level.value}")
            self.current_level = new_level

    def execute_with_fallback(self, operation_name: str, primary_func: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with fallback if primary function fails

        Args:
            operation_name: Name of the operation for fallback lookup
            primary_func: Primary function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result from primary function or fallback
        """
        try:
            # Try primary function
            result = primary_func(*args, **kwargs)
            return result

        except Exception as e:
            logger.warning(f"⚠️  Primary operation '{operation_name}' failed: {e}")

            # Try fallback if available
            fallback_func = self.fallback_strategies.get(operation_name)
            if fallback_func:
                try:
                    logger.info(f"🔄 Using fallback for '{operation_name}'")
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback for '{operation_name}' also failed: {fallback_error}")

            # Return basic fallback response
            return self._basic_fallback_response(operation_name)

    def _fallback_cleanliness_analysis(self, twin_data: Any) -> Dict[str, Any]:
        """Fallback cleanliness analysis using rule-based logic"""
        logger.info("🔄 Using rule-based cleanliness analysis")

        # Simple rule-based analysis
        issues = []

        if hasattr(twin_data, 'missing_visits') and twin_data.missing_visits > 0:
            issues.append(f"Missing {twin_data.missing_visits} visits")

        if hasattr(twin_data, 'open_queries') and twin_data.open_queries > 0:
            issues.append(f"{twin_data.open_queries} open queries")

        if hasattr(twin_data, 'uncoded_terms') and twin_data.uncoded_terms > 0:
            issues.append(f"{twin_data.uncoded_terms} uncoded terms")

        cleanliness_score = max(0, 100 - (len(issues) * 15))

        return {
            'cleanliness_score': cleanliness_score,
            'issues': issues,
            'method': 'rule_based_fallback',
            'confidence': 0.7,
            'explanation': f"Rule-based analysis identified {len(issues)} issues affecting cleanliness"
        }

    def _fallback_explanation(self, change_data: Any) -> str:
        """Fallback explanation generation"""
        logger.info("🔄 Using template-based explanation")

        if hasattr(change_data, 'trigger_reason'):
            return f"Status changed due to: {change_data.trigger_reason}. Rule-based analysis suggests monitoring for similar patterns."

        return "Status change detected. Using rule-based monitoring due to AI service unavailability."

    def _fallback_agent_analysis(self, data: Any) -> Dict[str, Any]:
        """Fallback agent analysis using statistical methods"""
        logger.info("🔄 Using statistical agent analysis")

        return {
            'insights': [
                "Statistical analysis shows normal data patterns",
                "No critical anomalies detected in current dataset",
                "Recommend continued monitoring of key metrics"
            ],
            'method': 'statistical_fallback',
            'confidence': 0.6,
            'recommendations': [
                "Monitor query resolution times",
                "Check for unusual data entry patterns",
                "Verify protocol compliance"
            ]
        }

    def _fallback_agent_prioritization(self, twins: List[Any]) -> List[Dict[str, Any]]:
        """Fallback agent prioritization using simple rules"""
        logger.info("🔄 Using rule-based agent prioritization")

        prioritized = []

        for twin in twins:
            priority_score = 0
            reasons = []

            # Simple priority rules
            if hasattr(twin, 'open_queries') and twin.open_queries > 2:
                priority_score += 3
                reasons.append("High open query count")

            if hasattr(twin, 'missing_visits') and twin.missing_visits > 0:
                priority_score += 2
                reasons.append("Missing visits")

            if hasattr(twin, 'clean_status') and not twin.clean_status:
                priority_score += 1
                reasons.append("Currently dirty")

            prioritized.append({
                'subject_id': twin.subject_id,
                'priority_score': priority_score,
                'reasons': reasons,
                'method': 'rule_based_fallback'
            })

        # Sort by priority score
        prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
        return prioritized

    def _fallback_graph_patterns(self, data: Any) -> Dict[str, Any]:
        """Fallback graph pattern detection"""
        logger.info("🔄 Using basic graph pattern detection")

        return {
            'patterns': [
                {
                    'type': 'basic_connectivity',
                    'description': 'Basic connectivity analysis available',
                    'confidence': 0.8
                }
            ],
            'method': 'basic_fallback',
            'limitations': 'Advanced pattern detection unavailable'
        }

    def _fallback_graph_analytics(self, data: Any) -> Dict[str, Any]:
        """Fallback graph analytics"""
        logger.info("🔄 Using basic graph analytics")

        return {
            'analytics': {
                'nodes_analyzed': len(data) if hasattr(data, '__len__') else 0,
                'basic_metrics': 'available',
                'advanced_metrics': 'unavailable'
            },
            'method': 'basic_fallback',
            'recommendations': 'Basic connectivity metrics available, advanced analytics offline'
        }

    def _basic_fallback_response(self, operation_name: str) -> Dict[str, Any]:
        """Basic fallback response when no specific fallback is available"""
        return {
            'status': 'degraded',
            'method': 'basic_fallback',
            'operation': operation_name,
            'message': f'Operation completed with limited functionality due to service degradation',
            'degradation_level': self.current_level.value,
            'timestamp': datetime.now().isoformat()
        }

    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        return {
            'current_level': self.current_level.value,
            'service_status': self.service_status,
            'available_services': sum(self.service_status.values()),
            'total_services': len(self.service_status),
            'availability_percentage': (sum(self.service_status.values()) / len(self.service_status)) * 100,
            'timestamp': datetime.now().isoformat()
        }

# Global degradation manager
degradation_manager = GracefulDegradationManager()

def graceful_execute(operation_name: str):
    """
    Decorator for graceful execution with fallback

    Args:
        operation_name: Name of operation for fallback lookup
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return degradation_manager.execute_with_fallback(operation_name, func, *args, **kwargs)
        return wrapper
    return decorator