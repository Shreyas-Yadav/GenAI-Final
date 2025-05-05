# clone_server.py
# pip install fastapi uvicorn llama-index openai

import os
from typing import List, Optional, Any
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from dotenv import load_dotenv
load_dotenv()
from github import Github, GithubException, Auth

gh: Optional[Github] = None               # Will hold the authenticated client
DEFAULT_OWNER: str = os.getenv("GH_DEFAULT_OWNER", "Shreyas-Yadav")
DEFAULT_PER_PAGE: int = 100

def init_github(token: str) -> None:
    """Initialise the global PyGitHub client (run once)."""
    global gh
    gh = Github(auth=Auth.Token(token))


def _client() -> Github:
    if gh is None:
        raise RuntimeError("GitHub client not initialized — call init_github(<token>) first")
    return gh



def list_my_repos() -> List[str]:
    """Return the full names ("owner/repo") of all repos visible to the user."""
    try:
        return [r.full_name for r in _client().get_user().get_repos()]
    except GithubException as e:
        raise RuntimeError(f"GitHub API error listing user repos: {e.data.get('message', str(e))}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error listing user repos: {e}")



OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]


# Build a simple echo agent
def build_clone_agent():
    llm = OpenRouter(model="openai/gpt-4o-mini", api_key=OPENROUTER_API_KEY)

    list_repo_tool = FunctionTool.from_defaults(
        fn=list_my_repos,
        name="list_my_repos",
        description="Return all repositories the authenticated user can access",
    )

    agent = ReActAgent.from_tools(
        tools=[list_repo_tool],
        llm=llm,
        verbose=True,
    )

    return agent

init_github(token=os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"])
clone_agent = build_clone_agent()

class Message(BaseModel):
    message: str

app = FastAPI()

@app.post("/agent/message")
async def receive(msg: Message):
    # delegate to the LlamaIndex agent
    reply = clone_agent.chat(msg.message)
    return {"reply": reply}

# To run on port 8000:
# uvicorn clone_server:app --port 8000 --reload
