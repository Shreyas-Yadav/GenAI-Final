# multi_agent_experiment.py - Runs the multi-agent system with hardcoded questions
import os
import asyncio
from dotenv import load_dotenv
from agents import RepoAgent, IssuesAgent, ContentAgent, SearchAgent, ControllerAgent
from agent_factory import AgentFactory
from controller_prompt import add_controller_prompt
from github_tools import init_github
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

# Load environment variables
load_dotenv()

# Pre-defined test questions
TEST_QUESTIONS = [
    "Create a repository under my github account puranjaigarg783 of the same as that of a battlestar galactica character. If a repository of the name of a star trek character already exists, chosse a different one, create an issue for adding a README for creating a calculator then commit code and readme for it close the issue after.",
]

# Set Phoenix project name for the experiment
project_name = "multi-agent-experiment-4"
os.environ["PHOENIX_PROJECT_NAME"] = project_name
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

# Initialize OpenTelemetry tracing with the experiment name
tracer_provider = register(auto_instrument=True, project_name=project_name)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

def create_specialized_agent(agent_type: str, llm: object, verbose: bool = True) -> object:
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

def create_controller(llm: object, specialized_agents: dict, verbose: bool = True) -> ControllerAgent:
    """Create a controller agent with the provided specialized agents."""
    controller = ControllerAgent(llm, verbose)
    # Replace the controller's default agents with the ones provided
    controller.agents = specialized_agents
    return controller

async def run_test_questions():
    # Initialize GitHub client
    print("Setting up GitHub token...")
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        raise EnvironmentError("GITHUB_PERSONAL_ACCESS_TOKEN_NEW not found in environment")
    init_github(token=github_token)
    
    # Collect API keys
    api_keys = {
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY")
    }

    # Verify essential API keys are available
    if not api_keys["openrouter"]:
        raise EnvironmentError("OPENROUTER_API_KEY not found in environment")
    
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
    
    print(f"Running test questions with project name: {project_name}")
    print("Available agents:")
    for agent_type, agent in controller.agents.items():
        print(f" - {agent}")
    print(" - Controller Agent: Orchestrates all agents")
    
    # Run each test question with the controller agent
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n\n=== QUESTION {i}/{len(TEST_QUESTIONS)}: {question} ===\n")
        
        # Add the controller prompt to the question
        augmented_question = add_controller_prompt(question)
        
        # Pass the augmented question to the controller agent
        response = controller.chat(augmented_question)
        print(f"🤖 Agent Response: {response}")
        print("\n=== END OF RESPONSE ===\n")
        await asyncio.sleep(2)  # Small delay between questions
    
    print("All test questions completed!")
    print(f"Spans have been recorded in Phoenix under project: {project_name}")
    print("Run eval.py with the same project name to evaluate the spans")

if __name__ == "__main__":
    asyncio.run(run_test_questions())