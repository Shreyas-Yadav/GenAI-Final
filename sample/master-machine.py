# master_agent.py
# pip install requests llama-index openai

import os
import requests
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
CLONE_SERVER  = os.environ.get("CLONE_URL", "http://127.0.0.1:8000")

def call_clone_agent(message: str) -> str:
    """Send `message` to the remote clone agent and return its reply."""
    resp = requests.post(
        f"{CLONE_SERVER}/agent/message",
        json={"message": message},
    )
    resp.raise_for_status()
    return resp.json()["reply"]

def build_master_agent():
    llm = OpenRouter(model="google/gemini-2.0-flash-001", api_key=OPENROUTER_API_KEY)
    
    echo_tool = FunctionTool.from_defaults(
        fn=call_clone_agent,
        name="call_clone_agent",
        description="Send a message to the clone agent server and return its response."
    )

    agent = ReActAgent.from_tools(tools=[echo_tool],llm=llm,verbose=True)
    return agent

if __name__ == "__main__":
    master = build_master_agent()
    while True:
        user_input = input("You> ")
        if user_input.lower() in {"quit", "exit"}:
            break
        # the agent may choose to call the remote tool
        response = master.chat(user_input)
        print("Master>", response)
