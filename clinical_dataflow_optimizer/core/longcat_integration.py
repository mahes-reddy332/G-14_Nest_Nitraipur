"""
LongCat AI Integration for Neural Clinical Data Mesh
Provides AI-powered reasoning and narrative generation capabilities
"""

import requests
import json
import logging
import time
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import threading
from dataclasses import dataclass, field

from config.settings import LongCatConfig, DEFAULT_LONGCAT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for API resilience"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5  # Open circuit after 5 failures
    recovery_timeout: int = 30  # Recovery time in seconds

    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        # Open circuit after threshold consecutive failures
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def should_attempt_request(self) -> bool:
        """Check if request should be attempted"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            # Check if recovery timeout has passed
            if self.last_failure_time and (datetime.now() - self.last_failure_time) > timedelta(seconds=self.recovery_timeout):
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True

    def is_open(self) -> bool:
        """Check if circuit breaker is currently open"""
        return self.state == "OPEN"

    def record_success(self) -> None:
        """Record successful request"""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def record_failure(self) -> None:
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class LongCatClient:
    """
    Client for interacting with LongCat AI API
    Supports both OpenAI-compatible and Anthropic-compatible endpoints
    Enhanced with circuit breaker, proper timeouts, and graceful degradation
    """

    def __init__(self, config: LongCatConfig = None):
        self.config = config or DEFAULT_LONGCAT_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        })

        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreakerState()

        # Response cache for graceful degradation
        self.response_cache: Dict[str, Dict] = {}
        self.cache_lock = threading.Lock()

        # Connection pool configuration for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0,  # We handle retries ourselves
            pool_block=False
        )
        self.session.mount('https://', adapter)

    def _make_request(self, endpoint: str, payload: Dict) -> Dict:
        """
        Make HTTP request to LongCat API with enhanced retry logic and circuit breaker

        Key improvements:
        - Separate connect_timeout (10s) and read_timeout (45s)
        - Exponential backoff with jitter
        - Circuit breaker pattern
        - Structured logging
        """
        url = f"{self.config.base_url}{endpoint}"

        # Check circuit breaker
        if not self.circuit_breaker.should_attempt_request():
            logger.warning(f"Circuit breaker OPEN - skipping LongCat request to {endpoint}")
            raise requests.exceptions.RequestException("Circuit breaker is OPEN")

        # Separate timeouts for better control
        connect_timeout = 10  # 10 seconds for connection establishment
        read_timeout = 45     # 45 seconds for reading response

        for attempt in range(self.config.retry_attempts):
            start_time = time.time()

            try:
                logger.info(f"LongCat API request attempt {attempt + 1}/{self.config.retry_attempts} to {endpoint}")

                response = self.session.post(
                    url,
                    json=payload,
                    timeout=(connect_timeout, read_timeout)  # (connect, read) tuple
                )

                response.raise_for_status()
                elapsed = time.time() - start_time

                # Record success in circuit breaker
                self.circuit_breaker.record_success()

                logger.info(f"LongCat API request successful in {elapsed:.2f}s")
                return response.json()

            except requests.exceptions.Timeout as e:
                elapsed = time.time() - start_time
                logger.warning(f"LongCat API timeout (attempt {attempt + 1}): {e} (elapsed: {elapsed:.2f}s)")

                # Record failure in circuit breaker
                self.circuit_breaker.record_failure()

                if attempt < self.config.retry_attempts - 1:
                    # Exponential backoff with jitter (1-3 seconds additional randomness)
                    base_delay = 2 ** attempt
                    jitter = random.uniform(1, 3)
                    delay = base_delay + jitter
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.retry_attempts} attempts failed with timeout")
                    raise e

            except requests.exceptions.ConnectionError as e:
                elapsed = time.time() - start_time
                logger.warning(f"LongCat API connection error (attempt {attempt + 1}): {e} (elapsed: {elapsed:.2f}s)")

                # Record failure in circuit breaker
                self.circuit_breaker.record_failure()

                if attempt < self.config.retry_attempts - 1:
                    base_delay = 2 ** attempt
                    jitter = random.uniform(1, 3)
                    delay = base_delay + jitter
                    logger.info(f"Retrying connection in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.retry_attempts} attempts failed with connection error")
                    raise e

            except requests.exceptions.RequestException as e:
                elapsed = time.time() - start_time
                logger.warning(f"LongCat API request failed (attempt {attempt + 1}): {e} (elapsed: {elapsed:.2f}s)")

                # Record failure in circuit breaker
                self.circuit_breaker.record_failure()

                if attempt < self.config.retry_attempts - 1:
                    base_delay = 2 ** attempt
                    jitter = random.uniform(1, 3)
                    delay = base_delay + jitter
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.retry_attempts} attempts failed")
                    raise e

    def chat_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """
        OpenAI-compatible chat completion with circuit breaker and caching
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            logger.warning("Circuit breaker is open, skipping LongCat request")
            raise Exception("Circuit breaker is open - LongCat service unavailable")

        # Create cache key from messages
        cache_key = f"chat_{hash(json.dumps(messages, sort_keys=True))}"

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        payload = {
            "model": kwargs.get('model', self.config.model),
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "stream": False
        }

        if self.config.enable_thinking and 'thinking' in kwargs.get('model', '').lower():
            payload["enable_thinking"] = True
            payload["thinking_budget"] = kwargs.get('thinking_budget', self.config.thinking_budget)

        try:
            response = self._make_request("/openai/v1/chat/completions", payload)

            # Cache successful response
            self._cache_response(cache_key, response)

            # Record success for circuit breaker
            self.circuit_breaker.record_success()

            return response

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")

            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()

            raise e

    def anthropic_completion(self, messages: List[Dict], system: str = None, **kwargs) -> Dict:
        """
        Anthropic-compatible message completion
        """
        payload = {
            "model": kwargs.get('model', self.config.model),
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "stream": False
        }

        if system:
            payload["system"] = system

        if self.config.enable_thinking and 'thinking' in kwargs.get('model', '').lower():
            payload["enable_thinking"] = True
            payload["thinking_budget"] = kwargs.get('thinking_budget', self.config.thinking_budget)

        return self._make_request("/anthropic/v1/messages", payload)

    def generate_agent_reasoning(self, context: str, task: str, data: Dict) -> str:
        """
        Generate enhanced reasoning for AI agents using LongCat
        Includes graceful degradation and caching for resilience
        """
        # Create cache key from inputs
        cache_key = f"reasoning_{hash((context, task, json.dumps(data, sort_keys=True)))}"

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        system_prompt = """You are an expert clinical data management AI assistant.
        Analyze the provided context and data to provide insightful reasoning and recommendations.
        Focus on data quality, patient safety, and trial integrity."""

        # Truncate data if too large to prevent timeout
        data_str = json.dumps(data, indent=2)
        if len(data_str) > 4000:  # Limit data size
            # Keep only essential fields for large payloads
            truncated_data = {
                k: v for k, v in data.items()
                if k in ['previous_status', 'new_status', 'blocking_factors', 'cleanliness_score', 'subject_id']
            }
            data_str = json.dumps(truncated_data, indent=2)
            logger.info(f"Truncated reasoning payload from {len(json.dumps(data, indent=2))} to {len(data_str)} characters")

        user_prompt = f"""
        Context: {context}

        Task: {task}

        Data: {data_str}

        Please provide detailed reasoning and actionable recommendations.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.chat_completion(messages, model=self.config.thinking_model, enable_thinking=True)
            reasoning = response['choices'][0]['message']['content']

            # Cache successful response
            self._cache_response(cache_key, reasoning)

            return reasoning

        except Exception as e:
            logger.error(f"Failed to generate agent reasoning: {e}")

            # Graceful degradation - provide basic reasoning without AI
            return self._generate_fallback_reasoning(context, task, data)

    def _generate_fallback_reasoning(self, context: str, task: str, data: Dict) -> str:
        """
        Generate basic reasoning when LongCat is unavailable
        Provides minimal but functional agent reasoning
        """
        logger.info("Using fallback reasoning due to LongCat unavailability")

        fallback_reasoning = f"""
        AUTOMATED ANALYSIS (LongCat Unavailable)

        Context: {context}

        Analysis:
        """

        # Basic pattern matching for common scenarios
        if 'status change' in context.lower():
            old_status = data.get('previous_status', 'Unknown')
            new_status = data.get('new_status', 'Unknown')
            blocking_factors = data.get('blocking_factors', [])

            fallback_reasoning += f"""
            - Patient status changed from '{old_status}' to '{new_status}'
            - This indicates a {'improvement' if new_status == 'Clean' else 'deterioration'} in data quality
            """

            if blocking_factors:
                fallback_reasoning += f"""
            - Blocking factors identified: {', '.join(blocking_factors[:3])}
            - Immediate attention required for these issues
            """

        elif 'query' in context.lower():
            fallback_reasoning += """
            - Open queries detected requiring resolution
            - Query aging analysis suggests timeliness issues
            - Recommend prioritizing oldest queries
            """

        fallback_reasoning += f"""

        Recommendations:
        - Review patient data manually for detailed analysis
        - Consider automated data quality checks
        - Monitor for similar patterns across patients

        Note: This is automated analysis due to AI service unavailability.
        """

        return fallback_reasoning

    def generate_narrative(self, patient_data: Dict, issues: List[str], recommendations: List[str]) -> str:
        """
        Generate human-readable narratives for patient status and issues
        Includes graceful degradation for resilience
        """
        # Create cache key
        cache_key = f"narrative_{hash((json.dumps(patient_data, sort_keys=True), str(issues), str(recommendations)))}"

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        system_prompt = """You are a clinical trial expert. Generate clear, professional narratives
        explaining patient status, data quality issues, and recommended actions."""

        # Truncate patient data if too large
        patient_data_str = json.dumps(patient_data, indent=2)
        if len(patient_data_str) > 3000:
            # Keep essential fields only
            essential_fields = ['subject_id', 'site_id', 'status', 'clean_status', 'clean_percentage']
            truncated_data = {k: v for k, v in patient_data.items() if k in essential_fields}
            patient_data_str = json.dumps(truncated_data, indent=2)
            logger.info(f"Truncated narrative patient data from {len(json.dumps(patient_data, indent=2))} to {len(patient_data_str)} characters")

        narrative_prompt = f"""
        Patient Data: {patient_data_str}

        Identified Issues: {', '.join(issues) if issues else 'None identified'}

        Recommendations: {', '.join(recommendations) if recommendations else 'No specific recommendations'}

        Generate a comprehensive narrative explaining the patient's current status,
        the significance of any issues, and the rationale for recommendations.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": narrative_prompt}
        ]

        try:
            response = self.chat_completion(messages)
            narrative = response['choices'][0]['message']['content']

            # Cache successful response
            self._cache_response(cache_key, narrative)

            return narrative

        except Exception as e:
            logger.error(f"Failed to generate narrative: {e}")

            # Graceful degradation
            return self._generate_fallback_narrative(patient_data, issues, recommendations)

    def _generate_fallback_narrative(self, patient_data: Dict, issues: List[str], recommendations: List[str]) -> str:
        """Generate basic narrative when LongCat is unavailable"""
        logger.info("Using fallback narrative generation due to LongCat unavailability")

        subject_id = patient_data.get('subject_id', 'Unknown')
        status = patient_data.get('status', 'Unknown')
        clean_status = patient_data.get('clean_status', False)

        narrative = f"""
        PATIENT STATUS NARRATIVE (Automated Analysis)

        Patient {subject_id} is currently in {status} status.

        Current Assessment:
        - Data Quality: {'Clean' if clean_status else 'Requires Attention'}
        """

        if issues:
            narrative += f"""
        - Issues Identified: {', '.join(issues[:3])}{'...' if len(issues) > 3 else ''}
        """

        if recommendations:
            narrative += f"""
        - Recommended Actions: {', '.join(recommendations[:3])}{'...' if len(recommendations) > 3 else ''}
        """

        narrative += """

        Note: This is an automated summary due to AI service unavailability.
        Please review patient data manually for complete analysis.
        """

        return narrative

    def explain_anomaly(self, anomaly_data: Dict, context: str) -> str:
        """
        Explain detected anomalies in clinical data with graceful degradation
        """
        # Create cache key
        cache_key = f"anomaly_{hash((json.dumps(anomaly_data, sort_keys=True), context))}"

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        system_prompt = """You are a data quality expert in clinical trials.
        Explain anomalies in clear, actionable terms."""

        # Truncate anomaly data if too large
        anomaly_data_str = json.dumps(anomaly_data, indent=2)
        if len(anomaly_data_str) > 3000:
            # Keep essential fields only
            essential_fields = ['type', 'severity', 'description', 'affected_fields']
            truncated_data = {k: v for k, v in anomaly_data.items() if k in essential_fields}
            anomaly_data_str = json.dumps(truncated_data, indent=2)
            logger.info(f"Truncated anomaly data from {len(json.dumps(anomaly_data, indent=2))} to {len(anomaly_data_str)} characters")

        explanation_prompt = f"""
        Context: {context[:2000] if len(context) > 2000 else context}

        Anomaly Details: {anomaly_data_str}

        Explain what this anomaly means, its potential impact on the trial,
        and suggest investigation steps.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": explanation_prompt}
        ]

        try:
            response = self.chat_completion(messages, model=self.config.thinking_model, enable_thinking=True)
            explanation = response['choices'][0]['message']['content']

            # Cache successful response
            self._cache_response(cache_key, explanation)

            return explanation

        except Exception as e:
            logger.error(f"Failed to explain anomaly: {e}")

            # Graceful degradation
            return self._generate_fallback_anomaly_explanation(anomaly_data, context)

    def _generate_fallback_anomaly_explanation(self, anomaly_data: Dict, context: str) -> str:
        """Generate basic anomaly explanation when LongCat is unavailable"""
        logger.info("Using fallback anomaly explanation due to LongCat unavailability")

        anomaly_type = anomaly_data.get('type', 'Unknown')
        severity = anomaly_data.get('severity', 'Unknown')
        description = anomaly_data.get('description', 'No description available')

        explanation = f"""
        ANOMALY EXPLANATION (Automated Analysis)

        Anomaly Type: {anomaly_type}
        Severity: {severity}
        Description: {description}

        Context: {context[:500]}{'...' if len(context) > 500 else ''}

        Potential Impact:
        - Data quality may be compromised
        - Trial integrity could be affected
        - Manual review recommended

        Recommended Actions:
        - Review source data for accuracy
        - Verify data entry procedures
        - Consult with clinical team
        - Document findings for audit trail

        Note: This is an automated summary due to AI service unavailability.
        Please perform detailed analysis manually.
        """

        return explanation

    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if available and not expired"""
        if cache_key in self.response_cache:
            cached_item = self.response_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.config.cache_ttl:
                logger.info(f"Cache hit for key: {cache_key}")
                return cached_item['response']
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: Dict) -> None:
        """Cache response with timestamp"""
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        logger.debug(f"Cached response for key: {cache_key}")


# Global instance
longcat_client = LongCatClient()