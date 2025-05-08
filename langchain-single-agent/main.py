# Basic usage - create and use the toolkit directly
from master import create_github_toolkit

toolkit = create_github_toolkit()
repos = toolkit.list_repos()
print(f"Found {len(repos)} repositories")

# Agent usage - create an agent and interact with it
from master import create_github_agent

agent = create_github_agent(model_name="gpt-4-turbo",verbose=True,temperature=0.0)
response = agent.invoke({"input": "List my repositories and show me the README of the first one"})
print(response.output)

# Run interactive mode
import asyncio
from master import interactive_chat

asyncio.run(interactive_chat())