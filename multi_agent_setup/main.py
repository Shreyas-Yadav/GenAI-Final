"""
Main entry point for the GitHub Multi-Agent System

This script initializes the ControllerAgent and provides an interactive
console interface for users to interact with the multi-agent system.
"""

import os
import asyncio
from dotenv import load_dotenv
from llama_index.llms.openrouter import OpenRouter

# Import the controller agent
from agents import ControllerAgent
from github_tools import init_github

# Load environment variables
load_dotenv()

# Set up OpenRouter LLM
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY not found in environment")

# Set up GitHub client
github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN_NEW")
if not github_token:
    raise EnvironmentError("GITHUB_PERSONAL_ACCESS_TOKEN not found in environment")

init_github(token=github_token)

# Initialize the LLM
llm = OpenRouter(
    api_key=OPENROUTER_API_KEY,
    model="google/gemini-2.0-flash-001",  # or your preferred model
    max_tokens=100000,
    max_retries=5,
)

# Create the controller agent
controller = ControllerAgent(llm=llm, verbose=True)

async def interactive_chat():
    """Run an interactive chat session with the multi-agent system."""
    print("GitHub Multi-Agent System ready. Type 'exit' to quit.")
    print("Available agents:")
    print(" - Repo Agent: Repository creation and management")
    print(" - Issues Agent: Issue and PR tracking and management")
    print(" - Content Agent: File content operations and management")
    print(" - Search Agent: Searching GitHub repositories, issues, and PRs")
    print(" - Controller Agent: Orchestrates all agents (default)")
    print("\n")
    
    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        
        # Pass the user input to the controller agent
        response = controller.chat(user_input)
        print("🤖 Agent:", response)

if __name__ == "__main__":
    asyncio.run(interactive_chat())