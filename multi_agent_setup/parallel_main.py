"""
parallel_main.py - Entry point for the GitHub Multi-Agent System with parallel processing

This script initializes the ParallelControllerAgent and provides an interactive
console interface for users to interact with the multi-agent system that can
execute tasks in parallel.
"""

import os
import asyncio
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Import agent components
from agents import RepoAgent, IssuesAgent, ContentAgent, SearchAgent, BranchAgent
from parallel_agents import ParallelControllerAgent
from agent_factory import AgentFactory
from controller_prompt import add_controller_prompt
from github_tools import init_github

# Load environment variables
load_dotenv()

# Collect API keys
api_keys = {
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY")
}

# Verify essential API keys are available
if not api_keys["openrouter"]:
    raise EnvironmentError("OPENROUTER_API_KEY not found in environment")

# Set up GitHub client
github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN_NEW")
if not github_token:
    raise EnvironmentError("GITHUB_PERSONAL_ACCESS_TOKEN not found in environment")

init_github(token=github_token)

def create_specialized_agent(agent_type: str, llm: Any, verbose: bool = True) -> Any:
    """Create a specialized agent based on its type."""
    if agent_type == "repo":
        return RepoAgent(llm, verbose)
    elif agent_type == "issues":
        return IssuesAgent(llm, verbose)
    elif agent_type == "content":
        return ContentAgent(llm, verbose)
    elif agent_type == "search":
        return SearchAgent(llm, verbose)
    elif agent_type == "branch":
        return BranchAgent(llm, verbose)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

async def interactive_chat():
    """Run an interactive chat session with the parallel multi-agent system."""
    # Get agent configuration
    config = AgentFactory.create_agent_configuration()
    
    # Create all agents using the factory
    specialized_agents = {}
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
            max_tokens=agent_config.get("max_tokens", 4000),
            temperature=agent_config.get("temperature", 0.2),
            max_retries=agent_config.get("max_retries", 3)
        )
        
        # Create the agent
        specialized_agents[agent_type] = create_specialized_agent(
            agent_type=agent_type,
            llm=llm,
            verbose=True
        )
    
    # Create the parallel controller agent
    controller_config = config["controller"]
    provider = controller_config["provider"]
    api_key = api_keys.get(provider)
    
    controller_llm = AgentFactory.create_llm(
        provider=provider,
        model_name=controller_config["model"],
        api_key=api_key,
        max_tokens=controller_config.get("max_tokens", 4000),
        temperature=controller_config.get("temperature", 0.2),
        max_retries=controller_config.get("max_retries", 3)
    )
    
    # Initialize the parallel controller
    parallel_controller = ParallelControllerAgent(
        llm=controller_llm,
        specialized_agents=specialized_agents,
        max_workers=4,  # Adjust based on your CPU capacity
        verbose=True
    )
    
    print("GitHub Multi-Agent System with parallel processing ready. Type 'exit' to quit.")
    print("\nAvailable agents:")
    for agent_type, agent in specialized_agents.items():
        print(f" - {agent}")
    print(" - Parallel Controller: Orchestrates agents with parallel execution")
    print("\nSample complex queries that benefit from parallel processing:")
    print(" - 'Find security issues in my repositories and create tickets for each'")
    print(" - 'Analyze the code quality in all my repositories'")
    print(" - 'Create issues for all TODOs in Python files across my repos'")
    print("\n")
    
    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        
        # Process the request with the parallel controller
        start_time = asyncio.get_event_loop().time()
        response = parallel_controller.chat(user_input)
        end_time = asyncio.get_event_loop().time()
        
        print("🤖 Agent:", response)
        print(f"Processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(interactive_chat())