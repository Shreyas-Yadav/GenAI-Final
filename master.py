import os
from typing import List

from dotenv import load_dotenv

# ─── LlamaIndex imports ──────────────────────────────────────────────────────
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openrouter import OpenRouter

from github_tools import init_github

# ─── GitHub helper functions (10 endpoints) ──────────────────────────────────
from github_tools import (
    list_my_repos, read_file, commit_file, list_issues, open_issue, list_prs, create_pr, list_commits, search_repos, search_issues
)



# ─── Environment & LLM setup ─────────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY not found in environment")


github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if not github_token:
    raise EnvironmentError("GITHUB_PERSONAL_ACCESS_TOKEN not found in environment")
init_github(token=github_token)  # set up GitHub client with your token

llm = OpenRouter(
    api_key=OPENROUTER_API_KEY,
    model="google/gemini-2.0-flash-001",        # pick any OpenRouter‑hosted chat model you like
    max_tokens=2048,
)




# ─── Wrap GitHub helpers as LlamaIndex tools ─────────────────────────────────
tools: List[FunctionTool] = [
    FunctionTool.from_defaults(
        fn=list_my_repos,
        name="list_my_repos",
        description="Return all repositories the authenticated user can access",
    ),
    FunctionTool.from_defaults(
        fn=read_file,
        name="get_file_content",
        description="Fetch the raw content of a file from a GitHub repository",
    ),
    FunctionTool.from_defaults(
        fn=commit_file,
        name="update_file_content",
        description="Create or update a file in a GitHub repository (single commit)",
    ),
    FunctionTool.from_defaults(
        fn=list_issues,
        name="list_issues",
        description="List issues in a repository (optionally filtered by state/labels)",
    ),
    FunctionTool.from_defaults(
        fn=open_issue,
        name="create_issue",
        description="Open a new issue in a repository",
    ),
    FunctionTool.from_defaults(
        fn=list_prs,
        name="list_pull_requests",
        description="List pull requests in a repository (optionally filtered by state)",
    ),
    FunctionTool.from_defaults(
        fn=create_pr,
        name="create_pull_request",
        description="Create a pull request from a branch to the default branch",
    ),
    FunctionTool.from_defaults(
        fn=list_commits,
        name="list_commits",
        description="List commits on a branch (or the default branch) of a repository",
    ),
    FunctionTool.from_defaults(
        fn=search_repos,
        name="search_repositories",
        description="Search public repositories with a GitHub search query",
    ),
    FunctionTool.from_defaults(
        fn=search_issues,
        name="search_issues",
        description="Search issues and pull requests across GitHub with a query",
    ),
  
]

# ─── Build the ReAct agent ───────────────────────────────────────────────────
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,        # prints reasoning to stdout — great for debugging
)

# # ─── Convenience entry point (optional) ──────────────────────────────────────
# if __name__ == "__main__":
#     # quick sanity‑check: ask the agent to list your repositories
#     response = agent.chat("Show me my repositories, newest first.")
#     print(response)
