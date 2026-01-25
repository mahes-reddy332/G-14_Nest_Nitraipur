# Neural Clinical Data Mesh - AI Agents
"""
Package initialization for agents module.

This module provides the AI agent framework for clinical trial data management:

Agents:
- Rex (ReconciliationAgent): Safety database reconciliation
- Codex (CodingAgent): Medical coding assistance  
- Lia (SiteLiaisonAgent): Site management and visit compliance
- Supervisor (SupervisorAgent): Orchestration and escalation

Components:
- LLM Integration: Multi-provider LLM support for agent reasoning
- Inter-Agent Communication: Event-driven messaging between agents
- Agent Reasoning Engine: Chain-of-thought decision support
"""

from .agent_framework import (
    BaseAgent,
    ReconciliationAgent,
    CodingAgent, 
    SiteLiaisonAgent,
    SupervisorAgent,
    AgentRecommendation,
    ActionType,
    ActionPriority
)

from .llm_integration import (
    LLMConfig,
    LLMProvider,
    LLMClientFactory,
    AgentReasoningEngine,
    PromptTemplates,
    get_reasoning_engine,
    set_reasoning_engine
)

from .inter_agent_comm import (
    MessageBus,
    EventType,
    MessagePriority,
    AgentMessage,
    EventTrigger,
    AgentCoordinator,
    get_message_bus,
    get_coordinator,
    shutdown as shutdown_comms
)

__all__ = [
    # Agents
    'BaseAgent',
    'ReconciliationAgent',
    'CodingAgent',
    'SiteLiaisonAgent',
    'SupervisorAgent',
    'AgentRecommendation',
    'ActionType',
    'ActionPriority',
    
    # LLM Integration
    'LLMConfig',
    'LLMProvider',
    'LLMClientFactory',
    'AgentReasoningEngine',
    'PromptTemplates',
    'get_reasoning_engine',
    'set_reasoning_engine',
    
    # Inter-Agent Communication
    'MessageBus',
    'EventType',
    'MessagePriority',
    'AgentMessage',
    'EventTrigger',
    'AgentCoordinator',
    'get_message_bus',
    'get_coordinator',
    'shutdown_comms'
]
