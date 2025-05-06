"""
advanced_main.py - Enhanced entry point for the GitHub Multi-Agent System

This script uses the AgentFactory to create agents with different LLM backends,
providing a more flexible and configurable system.
"""

import os
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv

# Import agent components
from agents import RepoAgent, IssuesAgent, ContentAgent, SearchAgent, ControllerAgent
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
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def create_controller(llm: Any, specialized_agents: Dict[str, Any], verbose: bool = True) -> ControllerAgent:
    """Create a controller agent with the provided specialized agents."""
    controller = ControllerAgent(llm, verbose)
    # Replace the controller's default agents with the ones provided
    controller.agents = specialized_agents
    return controller

async def interactive_chat():
    """Run an interactive chat session with the multi-agent system."""
    # Get agent configuration
    config = AgentFactory.create_agent_configuration()
    
    # Create all agents using the factory
    agents = AgentFactory.create_agents_from_config(
        config=config,
        api_keys=api_keys,
        agent_constructor=create_specialized_agent,
        controller_constructor=create_controller
    )
    
    # Get the controller agent
    controller = agents["controller"]
    
    print("GitHub Multi-Agent System ready. Type 'exit' to quit.")
    print("Available agents:")
    for agent_type, agent in controller.agents.items():
        print(f" - {agent}")
    print(" - Controller Agent: Orchestrates all agents (default)")
    print("\n")
    
    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        
        # Add the controller prompt to the user input
        augmented_input = add_controller_prompt(user_input)
        
        # Pass the augmented input to the controller agent
        response = controller.chat(augmented_input)
        print("🤖 Agent:", response)

if __name__ == "__main__":
    asyncio.run(interactive_chat())