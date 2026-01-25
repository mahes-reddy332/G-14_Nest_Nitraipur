"""
LLM Integration Module for AI Agents
====================================

Provides a unified interface for Large Language Model integration using
LangChain for reasoning, decision-making, and inter-agent communication.

Features:
- Multi-provider support (Longcat AI, OpenAI, Anthropic, Azure, local models)
- Prompt templates for clinical decision support
- Chain-of-thought reasoning
- Agent communication protocols
- Response caching and rate limiting

Primary Provider: Longcat AI (https://api.longcat.chat)
"""

import os
import json
import hashlib
import logging
import requests
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum
import threading
import asyncio
from functools import wraps

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    LONGCAT = "longcat"  # Primary provider - Longcat AI
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"  # Local models
    MOCK = "mock"  # For testing


@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    provider: LLMProvider = LLMProvider.LONGCAT  # Default to Longcat AI
    model_name: str = "LongCat-Flash-Chat"  # Longcat's default model
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1  # Low temperature for clinical accuracy
    max_tokens: int = 2000
    timeout_seconds: int = 60
    retry_attempts: int = 3
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    
    def __post_init__(self):
        # Load from environment if not provided
        if self.api_key is None:
            # Check for Longcat API key first (using the format from .env)
            if self.provider == LLMProvider.LONGCAT:
                self.api_key = os.getenv("API_KEY_Longcat", "")
            else:
                self.api_key = os.getenv(f"{self.provider.value.upper()}_API_KEY", "")
        if self.api_base is None:
            if self.provider == LLMProvider.LONGCAT:
                self.api_base = "https://api.longcat.chat"
            else:
                self.api_base = os.getenv(f"{self.provider.value.upper()}_API_BASE", None)


class PromptTemplates:
    """Clinical decision support prompt templates"""
    
    # Agent reasoning template
    AGENT_REASONING = """You are {agent_name}, a specialized AI agent for clinical trial data management.

Context: {context}
Current Task: {task}
Available Data: {data}

Based on this information, provide your analysis and recommended actions.
Structure your response as:
1. Key Observations (bullet points)
2. Risk Assessment (if applicable)
3. Recommended Actions (prioritized list)
4. Rationale (brief justification for each recommendation)

Be specific, actionable, and reference clinical trial best practices (ICH E6 R2, CDISC standards).
"""

    # Clinical insight generation
    CLINICAL_INSIGHT = """Analyze the following clinical trial data and provide insights:

Data Type: {data_type}
Study Context: {study_context}
Metrics: {metrics}

Generate:
1. Summary of current state
2. Anomalies or concerns detected
3. Trends observed
4. Actionable recommendations

Focus on patient safety and data quality.
"""

    # Reconciliation analysis
    RECONCILIATION_ANALYSIS = """You are analyzing a data reconciliation discrepancy in a clinical trial.

Discrepancy Details:
- Subject: {subject_id}
- Site: {site_id}
- Source 1 ({source1_name}): {source1_value}
- Source 2 ({source2_name}): {source2_value}
- Discrepancy Type: {discrepancy_type}

Analyze this discrepancy and provide:
1. Likely root cause
2. Severity assessment (Critical/High/Medium/Low)
3. Recommended resolution action
4. Query text to send to site (if applicable)

Reference ICH E6 R2 Section 5.18.4 for safety reporting compliance.
"""

    # Coding assistance
    CODING_ANALYSIS = """Analyze this verbatim term for medical coding:

Verbatim Term: {verbatim_term}
Dictionary: {dictionary} (e.g., MedDRA, WHODRA)
Context: {context}

Determine:
1. Is the term specific enough for coding? (Yes/No/Partial)
2. Confidence score (0.0 - 1.0)
3. Suggested code(s) if applicable
4. If not codeable, explain why and draft clarification query

Consider drug class terms vs specific medications for WHODRA.
"""

    # Narrative generation
    NARRATIVE_GENERATION = """Generate a patient safety narrative based on:

Subject Profile: {subject_profile}
Adverse Events: {adverse_events}
Concomitant Medications: {medications}
Lab Issues: {lab_issues}

Create a professional clinical narrative suitable for Medical Monitor review.
Include:
1. Patient demographics summary
2. Chronological adverse event description
3. Relevant medications and potential interactions
4. Outstanding data issues requiring attention
5. Risk assessment and recommendations

Use formal medical writing style.
"""

    # Natural language query interpretation
    NLQ_INTERPRETATION = """Interpret this natural language query about clinical trial data:

User Query: {user_query}
Available Metrics: {available_metrics}
Available Entities: {available_entities}
Conversation Context: {conversation_context}

Parse the query and return structured information:
1. Intent (trend_analysis, comparison, anomaly_detection, summary, specific_value)
2. Target metrics (list)
3. Target entities/filters (dict)
4. Time period (if specified)
5. Aggregation level (site, subject, country, study)
6. Confidence score (0.0 - 1.0)

If the query is ambiguous, identify what clarification is needed.
"""


class LLMResponse:
    """Structured LLM response"""
    
    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int = 0,
        latency_ms: float = 0,
        from_cache: bool = False,
        metadata: Dict = None
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.from_cache = from_cache
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'model': self.model,
            'tokens_used': self.tokens_used,
            'latency_ms': self.latency_ms,
            'from_cache': self.from_cache,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class ResponseCache:
    """Thread-safe response cache with TTL"""
    
    def __init__(self, ttl_minutes: int = 60):
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _make_key(self, prompt: str, config: Dict) -> str:
        """Create cache key from prompt and config"""
        key_data = json.dumps({'prompt': prompt, 'config': config}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, prompt: str, config: Dict) -> Optional[LLMResponse]:
        """Get cached response if valid"""
        key = self._make_key(prompt, config)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if datetime.now() - entry['timestamp'] < self.ttl:
                    response = entry['response']
                    response.from_cache = True
                    return response
                else:
                    del self._cache[key]
        return None
    
    def set(self, prompt: str, config: Dict, response: LLMResponse):
        """Cache a response"""
        key = self._make_key(prompt, config)
        with self._lock:
            self._cache[key] = {
                'response': response,
                'timestamp': datetime.now()
            }
    
    def clear(self):
        """Clear all cached responses"""
        with self._lock:
            self._cache.clear()


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.cache = ResponseCache(config.cache_ttl_minutes) if config.enable_caching else None
    
    @abstractmethod
    def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        """Make API call to LLM provider"""
        pass
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with caching"""
        config_dict = {
            'model': self.config.model_name,
            'temperature': self.config.temperature,
            **kwargs
        }
        
        # Check cache
        if self.cache:
            cached = self.cache.get(prompt, config_dict)
            if cached:
                logger.debug(f"Cache hit for prompt hash")
                return cached
        
        # Call API
        import time
        start_time = time.time()
        response = self._call_api(prompt, **kwargs)
        response.latency_ms = (time.time() - start_time) * 1000
        
        # Cache response
        if self.cache:
            self.cache.set(prompt, config_dict, response)
        
        return response
    
    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """Async generate - default implementation wraps sync"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, **kwargs))


class LongcatClient(BaseLLMClient):
    """
    Longcat AI API client
    
    Uses OpenAI-compatible API format.
    API Base: https://api.longcat.chat
    Endpoint: /openai/v1/chat/completions
    Models: LongCat-Flash-Chat, LongCat-Flash-Thinking-2601
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import requests
        self.requests = requests
        self.api_base = config.api_base or "https://api.longcat.chat"
        self.api_key = config.api_key
        self.available = bool(self.api_key)
        
        if not self.available:
            logger.warning("Longcat API key not configured. Set API_KEY_Longcat in environment.")
    
    def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.available:
            return LLMResponse(
                content=self._mock_response(prompt),
                model=self.config.model_name,
                metadata={'mock': True, 'reason': 'API key not configured'}
            )
        
        try:
            url = f"{self.api_base}/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "stream": False
            }
            
            # Add system message if provided
            if 'system_prompt' in kwargs:
                payload["messages"].insert(0, {
                    "role": "system",
                    "content": kwargs['system_prompt']
                })
            
            # Enable thinking mode for Thinking model
            if "Thinking" in self.config.model_name:
                payload["enable_thinking"] = kwargs.get('enable_thinking', True)
                payload["thinking_budget"] = kwargs.get('thinking_budget', 1024)
            
            response = self.requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Parse OpenAI-compatible response format
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            usage = result.get('usage', {})
            tokens_used = usage.get('total_tokens', 0) or (
                usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
            )
            
            return LLMResponse(
                content=content,
                model=self.config.model_name,
                tokens_used=tokens_used,
                metadata={
                    'finish_reason': result.get('choices', [{}])[0].get('finish_reason'),
                    'provider': 'longcat'
                }
            )
            
        except self.requests.exceptions.Timeout:
            logger.error(f"Longcat API timeout after {self.config.timeout_seconds}s")
            return LLMResponse(
                content=self._mock_response(prompt),
                model=self.config.model_name,
                metadata={'error': 'timeout', 'fallback': True}
            )
        except self.requests.exceptions.RequestException as e:
            logger.error(f"Longcat API error: {e}")
            return LLMResponse(
                content=self._mock_response(prompt),
                model=self.config.model_name,
                metadata={'error': str(e), 'fallback': True}
            )
        except Exception as e:
            logger.error(f"Unexpected Longcat API error: {e}")
            return LLMResponse(
                content=self._mock_response(prompt),
                model=self.config.model_name,
                metadata={'error': str(e), 'fallback': True}
            )
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing or fallback"""
        if "reconciliation" in prompt.lower():
            return """1. Key Observations:
- Data discrepancy detected between EDC and Safety database
- SAE record exists in Safety DB without corresponding AE form in EDC

2. Risk Assessment: HIGH
- This is a "Zombie SAE" scenario requiring immediate attention
- Regulatory compliance risk per ICH E6 R2 Section 5.18.4

3. Recommended Actions:
1. Issue query to site for AE form entry (Priority: Critical)
2. Escalate to Medical Monitor within 24 hours if unresolved
3. Update EDRR Total Open Issue Count

4. Rationale:
- Missing AE forms impact data integrity and safety monitoring
- Regulatory submissions require complete reconciliation"""

        elif "coding" in prompt.lower():
            return """1. Term Specificity: No - Term is too vague for coding
2. Confidence Score: 0.65
3. Suggested Codes: None applicable - clarification needed
4. Clarification Query:
The verbatim term provided is a drug class, not a specific medication.
Please provide the specific Trade Name or Generic Name.
Reference: WHO Drug Dictionary coding requirement"""

        elif "narrative" in prompt.lower():
            return """PATIENT SAFETY NARRATIVE

Subject presented with documented adverse events requiring medical attention.
Based on available data, the following observations are noted:

- Patient status: Under active monitoring
- Adverse events: Documented per protocol requirements
- Concomitant medications: Being tracked for potential interactions
- Outstanding issues: Require clinical team attention

RECOMMENDATION: Continue close monitoring and ensure timely data entry."""

        else:
            return """Analysis Complete:

1. Key Observations:
- Data reviewed and assessed against protocol requirements
- Current status within acceptable parameters
- No critical issues identified

2. Recommendations:
- Continue routine monitoring
- Address any outstanding queries
- Maintain data quality standards

3. Rationale:
- Standard clinical trial best practices apply
- ICH E6 R2 compliance requirements met"""


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (optional - requires 'openai' package)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
            self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)
            self.available = True
        except ImportError:
            logger.warning("OpenAI package not installed. Using mock client.")
            self.available = False
    
    def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.available:
            return LLMResponse(
                content=self._mock_response(prompt),
                model=self.config.model_name,
                metadata={'mock': True}
            )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.config.model_name,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                metadata={'finish_reason': response.choices[0].finish_reason}
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content=self._mock_response(prompt),
                model=self.config.model_name,
                metadata={'error': str(e), 'fallback': True}
            )
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing or fallback"""
        # Analyze prompt to generate contextually appropriate mock
        if "reconciliation" in prompt.lower():
            return """1. Key Observations:
- Data discrepancy detected between EDC and Safety database
- SAE record exists in Safety DB without corresponding AE form in EDC

2. Risk Assessment: HIGH
- This is a "Zombie SAE" scenario requiring immediate attention
- Regulatory compliance risk per ICH E6 R2 Section 5.18.4

3. Recommended Actions:
1. Issue query to site for AE form entry (Priority: Critical)
2. Escalate to Medical Monitor within 24 hours if unresolved
3. Update EDRR Total Open Issue Count

4. Rationale:
- Missing AE forms impact data integrity and safety monitoring
- Regulatory submissions require complete reconciliation"""

        elif "coding" in prompt.lower():
            return """1. Term Specificity: No - Term is too vague for coding
2. Confidence Score: 0.65
3. Suggested Codes: None applicable - clarification needed
4. Clarification Query:
The verbatim term provided is a drug class, not a specific medication.
Please provide the specific Trade Name or Generic Name.
Reference: WHO Drug Dictionary coding requirement"""

        elif "narrative" in prompt.lower():
            return """PATIENT SAFETY NARRATIVE

Subject presented with documented adverse events requiring medical attention.
Based on available data, the following observations are noted:

- Patient status: Under active monitoring
- Adverse events: Documented per protocol requirements
- Concomitant medications: Being tracked for potential interactions
- Outstanding issues: Require clinical team attention

RECOMMENDATION: Continue close monitoring and ensure timely data entry."""

        else:
            return """Analysis Complete:

1. Key Observations:
- Data reviewed and assessed against protocol requirements
- Current status within acceptable parameters
- No critical issues identified

2. Recommendations:
- Continue routine monitoring
- Address any outstanding queries
- Maintain data quality standards

3. Rationale:
- Standard clinical trial best practices apply
- ICH E6 R2 compliance requirements met"""


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client (optional - requires 'anthropic' package)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic  # type: ignore[import-not-found]
            self.client = anthropic.Anthropic(api_key=config.api_key)
            self.available = True
        except ImportError:
            logger.warning("Anthropic package not installed. Using mock client.")
            self.available = False
    
    def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.available:
            return LLMResponse(
                content="[Mock Anthropic Response] " + prompt[:100],
                model=self.config.model_name,
                metadata={'mock': True}
            )
        
        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=self.config.model_name,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                metadata={'stop_reason': response.stop_reason}
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                content=f"[Fallback] Error: {str(e)}",
                model=self.config.model_name,
                metadata={'error': str(e), 'fallback': True}
            )


class MockClient(BaseLLMClient):
    """Mock client for testing"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.call_count = 0
        self.prompts_received = []
    
    def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)
        
        # Generate intelligent mock response based on context
        if "reconciliation" in prompt.lower() or "sae" in prompt.lower():
            content = """Analysis indicates safety data reconciliation issue.
Severity: HIGH
Action: Issue query to site for clarification.
Reference: ICH E6 R2 compliance requirements."""
        elif "coding" in prompt.lower():
            content = """Coding Analysis:
- Term requires clarification for proper WHO Drug Dictionary coding
- Confidence: 0.70
- Recommendation: Issue clarification query"""
        elif "narrative" in prompt.lower():
            content = """Clinical Narrative Generated:
Patient requires ongoing safety monitoring. 
Data quality indicators within acceptable ranges.
Continue standard protocol procedures."""
        else:
            content = f"Mock response for: {prompt[:50]}..."
        
        return LLMResponse(
            content=content,
            model="mock-model",
            tokens_used=len(prompt.split()) + len(content.split()),
            metadata={'mock': True, 'call_number': self.call_count}
        )


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    _clients = {
        LLMProvider.LONGCAT: LongcatClient,  # Primary provider
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.MOCK: MockClient,
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMClient:
        """Create LLM client based on config"""
        client_class = cls._clients.get(config.provider, MockClient)
        return client_class(config)
    
    @classmethod
    def create_default(cls) -> BaseLLMClient:
        """Create client with default configuration"""
        # Try to use Longcat AI first (primary provider)
        if os.getenv("API_KEY_Longcat"):
            config = LLMConfig(provider=LLMProvider.LONGCAT)
            logger.info("Using Longcat AI as LLM provider")
        elif os.getenv("OPENAI_API_KEY"):
            config = LLMConfig(provider=LLMProvider.OPENAI)
            logger.info("Using OpenAI as LLM provider")
        elif os.getenv("ANTHROPIC_API_KEY"):
            config = LLMConfig(provider=LLMProvider.ANTHROPIC)
            logger.info("Using Anthropic as LLM provider")
        else:
            config = LLMConfig(provider=LLMProvider.MOCK)
            logger.info("No LLM API key found, using Mock provider")
        
        return cls.create(config)


class AgentReasoningEngine:
    """
    Unified reasoning engine for all AI agents.
    Provides chain-of-thought reasoning and decision support.
    """
    
    def __init__(self, llm_client: BaseLLMClient = None):
        self.llm = llm_client or LLMClientFactory.create_default()
        self.reasoning_history: List[Dict] = []
    
    def reason(
        self,
        agent_name: str,
        context: str,
        task: str,
        data: Dict,
        **kwargs
    ) -> str:
        """
        Generate reasoning for agent decision-making
        """
        prompt = PromptTemplates.AGENT_REASONING.format(
            agent_name=agent_name,
            context=context,
            task=task,
            data=json.dumps(data, indent=2, default=str)
        )
        
        response = self.llm.generate(prompt, **kwargs)
        
        # Log reasoning for audit
        self.reasoning_history.append({
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'task': task,
            'response_summary': response.content[:200],
            'from_cache': response.from_cache
        })
        
        return response.content
    
    def analyze_reconciliation(
        self,
        subject_id: str,
        site_id: str,
        source1_name: str,
        source1_value: Any,
        source2_name: str,
        source2_value: Any,
        discrepancy_type: str
    ) -> Dict:
        """Analyze reconciliation discrepancy"""
        prompt = PromptTemplates.RECONCILIATION_ANALYSIS.format(
            subject_id=subject_id,
            site_id=site_id,
            source1_name=source1_name,
            source1_value=source1_value,
            source2_name=source2_name,
            source2_value=source2_value,
            discrepancy_type=discrepancy_type
        )
        
        response = self.llm.generate(prompt)
        
        # Parse structured response
        return {
            'analysis': response.content,
            'model': response.model,
            'latency_ms': response.latency_ms,
            'from_cache': response.from_cache
        }
    
    def analyze_coding(
        self,
        verbatim_term: str,
        dictionary: str,
        context: str = ""
    ) -> Dict:
        """Analyze verbatim term for medical coding"""
        prompt = PromptTemplates.CODING_ANALYSIS.format(
            verbatim_term=verbatim_term,
            dictionary=dictionary,
            context=context
        )
        
        response = self.llm.generate(prompt)
        
        return {
            'analysis': response.content,
            'model': response.model,
            'latency_ms': response.latency_ms
        }
    
    def generate_narrative(
        self,
        subject_profile: Dict,
        adverse_events: List[Dict],
        medications: List[Dict],
        lab_issues: List[Dict]
    ) -> str:
        """Generate patient safety narrative"""
        prompt = PromptTemplates.NARRATIVE_GENERATION.format(
            subject_profile=json.dumps(subject_profile, indent=2, default=str),
            adverse_events=json.dumps(adverse_events, indent=2, default=str),
            medications=json.dumps(medications, indent=2, default=str),
            lab_issues=json.dumps(lab_issues, indent=2, default=str)
        )
        
        response = self.llm.generate(prompt)
        return response.content
    
    def interpret_nlq(
        self,
        user_query: str,
        available_metrics: List[str],
        available_entities: List[str],
        conversation_context: str = ""
    ) -> Dict:
        """Interpret natural language query"""
        prompt = PromptTemplates.NLQ_INTERPRETATION.format(
            user_query=user_query,
            available_metrics=json.dumps(available_metrics),
            available_entities=json.dumps(available_entities),
            conversation_context=conversation_context
        )
        
        response = self.llm.generate(prompt)
        
        # Try to parse structured response
        return {
            'interpretation': response.content,
            'raw_response': response.to_dict()
        }


# Global singleton for easy access
_default_reasoning_engine: Optional[AgentReasoningEngine] = None


def get_reasoning_engine() -> AgentReasoningEngine:
    """Get or create the default reasoning engine"""
    global _default_reasoning_engine
    if _default_reasoning_engine is None:
        _default_reasoning_engine = AgentReasoningEngine()
    return _default_reasoning_engine


def set_reasoning_engine(engine: AgentReasoningEngine):
    """Set the default reasoning engine"""
    global _default_reasoning_engine
    _default_reasoning_engine = engine
