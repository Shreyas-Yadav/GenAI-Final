# experiment.py - Runs the agent with hardcoded questions
import os
import asyncio
from dotenv import load_dotenv
from master import agent, init_github
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

# Load environment variables
load_dotenv()

# Pre-defined test questions  
TEST_QUESTIONS = [
    "Create a repository name of character in battlestar galactica after checking that the repository already doesnt exist, create an issue for adding a README for creating a calculator then commit code and readme for it close the issue after.",
]

# Set Phoenix project name for the experiment
project_name = "single-agent-experiment-4"
os.environ["PHOENIX_PROJECT_NAME"] = project_name
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

# Initialize OpenTelemetry tracing with the experiment name
tracer_provider = register(auto_instrument=True, project_name=project_name)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

async def run_test_questions():
    # Initialize GitHub client
    print("Setting up GitHub token...")
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        raise EnvironmentError("GITHUB_PERSONAL_ACCESS_TOKEN not found in environment")
    init_github(token=github_token)
    
    # Run each test question with the agent
    print(f"Running test questions with project name: {project_name}")
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n\n=== QUESTION {i}/{len(TEST_QUESTIONS)}: {question} ===\n")
        response = agent.chat(question)
        print(f"🤖 Agent Response: {response}")
        print("\n=== END OF RESPONSE ===\n")
        await asyncio.sleep(2)  # Small delay between questions
    
    print("All test questions completed!")
    print(f"Spans have been recorded in Phoenix under project: {project_name}")
    print("Run eval7.py with the same project name to evaluate the spans")

if __name__ == "__main__":
    asyncio.run(run_test_questions())