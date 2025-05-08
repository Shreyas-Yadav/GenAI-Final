"""
agent_factory.py - Factory for creating agents with different LLM providers

This module provides a flexible way to create agents with different LLM backends,
allowing for experimentation with various models for different agent roles.
"""

from typing import Dict, Any, Optional, List, Callable
from llama_index.core.llms import LLM
from llama_index.llms.openrouter import OpenRouter
# from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

class AgentFactory:
    """Factory class for creating agents with different LLM backends."""
    
    @staticmethod
    def create_llm(
        provider: str,
        model_name: str,
        api_key: str,
        **kwargs
    ) -> LLM:
        """
        Create an LLM instance based on the provider and model name.
        
        Args:
            provider: The LLM provider ("openrouter", "anthropic", "openai")
            model_name: The specific model to use
            api_key: API key for the provider
            **kwargs: Additional parameters for the LLM
            
        Returns:
            An LLM instance
        """
        if provider.lower() == "openrouter":
            return OpenRouter(
                api_key=api_key,
                model=model_name,
                **kwargs
            )
        # elif provider.lower() == "anthropic":
        #     return Anthropic(
        #         api_key=api_key,
        #         model_name=model_name,
        #         **kwargs
        #     )
        elif provider.lower() == "openai":
            return OpenAI(
                api_key=api_key,
                model=model_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def create_agent_configuration() -> Dict[str, Dict[str, Any]]:
        """
        Create a configuration for different agent types with recommended LLM settings.
        
        Returns:
            Dictionary mapping agent types to their recommended configurations
        """
        return {
            "controller": {
                "provider": "openrouter",
                "model": "google/gemini-2.0-flash-001",
                "max_tokens": 4000,
                "temperature": 0.2,  # More focused for task routing
                "max_retries": 3
            },
            "repo": {
                "provider": "openrouter",
                "model": "google/gemini-2.0-flash-001",
                "max_tokens": 2000,
                "temperature": 0.1,  # Very focused for repo operations
                "max_retries": 3
            },
            "issues": {
                "provider": "openrouter",
                "model": "google/gemini-2.0-flash-001",
                "max_tokens": 2000,
                "temperature": 0.1,  # Very focused for issue operations
                "max_retries": 3
            },
            "content": {
                "provider": "openrouter",
                "model": "google/gemini-2.0-flash-001",
                "max_tokens": 8000,  # Higher for file content operations
                "temperature": 0.1,
                "max_retries": 3
            },
            "search": {
                "provider": "openrouter",
                "model": "google/gemini-2.0-flash-001",
                "max_tokens": 2000,
                "temperature": 0.1,
                "max_retries": 3
            },
            "branch": {
                "provider": "openrouter",
                "model": "google/gemini-2.0-flash-001",
                "max_tokens": 2000,
                "temperature": 0.1,  # Very focused for branch operations
                "max_retries": 3
            }
        }
    
    @staticmethod
    def create_agents_from_config(
        config: Dict[str, Dict[str, Any]],
        api_keys: Dict[str, str],
        agent_constructor: Callable,
        controller_constructor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Create all agents based on the configuration.
        
        Args:
            config: Agent configuration dictionary
            api_keys: Dictionary mapping provider names to API keys
            agent_constructor: Constructor function for specialized agents
            controller_constructor: Constructor function for the controller agent
            
        Returns:
            Dictionary of created agents
        """
        agents = {}
        
        # Create specialized agents
        for agent_type, agent_config in config.items():
            if agent_type == "controller":
                continue
                
            provider = agent_config["provider"]
            api_key = api_keys.get(provider)
            
            if not api_key:
                raise ValueError(f"API key not found for provider: {provider}")
            
            # Create the LLM
            llm = AgentFactory.create_llm(
                provider=provider,
                model_name=agent_config["model"],
                api_key=api_key,
                max_tokens=agent_config.get("max_tokens", 100000),
                temperature=agent_config.get("temperature", 0.2),
                max_retries=agent_config.get("max_retries", 5)
            )
            
            # Create the agent
            agents[agent_type] = agent_constructor(
                agent_type=agent_type,
                llm=llm,
                verbose=True
            )
        
        # Create controller agent if provided
        if controller_constructor and "controller" in config:
            controller_config = config["controller"]
            provider = controller_config["provider"]
            api_key = api_keys.get(provider)
            
            if not api_key:
                raise ValueError(f"API key not found for provider: {provider}")
            
            # Create the LLM for the controller
            controller_llm = AgentFactory.create_llm(
                provider=provider,
                model_name=controller_config["model"],
                api_key=api_key,
                max_tokens=controller_config.get("max_tokens", 4000),
                temperature=controller_config.get("temperature", 0.2),
                max_retries=controller_config.get("max_retries", 3)
            )
            
            # Create the controller agent with specialized agents
            agents["controller"] = controller_constructor(
                llm=controller_llm,
                specialized_agents=agents,
                verbose=True
            )
        
        return agents