"""
master.py - Main orchestration module for GitHub agent tools

This module provides a unified interface for agents to interact with GitHub,
building on the foundational github_tools.py utilities. It converts the function-based
approach into a class-based system with async support for better integration with
agent frameworks and provides agent creation functionality.
"""

import os
import asyncio
from typing import List, Dict, Optional, Any, Union, Callable
from functools import wraps
import inspect
from rich import print
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langchain imports
from langchain.agents import AgentExecutor,create_react_agent 
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.callbacks.manager import CallbackManager

# Import all functions from github_tools
from github_tools import (
    init_github, list_my_repos, read_file, close_issue, commit_file,
    list_issues, open_issue, list_prs, create_pr, list_commits,
    search_repos, search_issues, create_repo, list_repo_files,
    merge_branches, check_merge_status, create_branch
)

class GitHubToolkit:
    """
    A class-based wrapper around github_tools.py that provides:
    1. Method-based access to all GitHub operations
    2. Automatic initialization with token
    3. Async versions of all methods for use with async frameworks
    4. Logging and error tracking
    
    This serves as the core toolkit for GitHub operations that can be used
    directly or exposed as tools for AI agents.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub toolkit.
        
        Args:
            token (str, optional): GitHub access token. If not provided, 
                                  will look for GITHUB_TOKEN environment variable.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token not provided and GITHUB_TOKEN environment variable not set")
        
        # Initialize the GitHub client
        init_github(self.token)
        
        # Operation tracking
        self.operation_history = []
        
    def log_operation(self, operation: str, success: bool, details: Dict = None):
        """Record an operation in the history log."""
        self.operation_history.append({
            "operation": operation,
            "success": success,
            "details": details or {}
        })
        
    # ---- Repository Operations ----
    
    def list_repos(self) -> List[str]:
        """List all repositories accessible to the authenticated user."""
        try:
            repos = list_my_repos()
            self.log_operation("list_repos", True, {"count": len(repos)})
            return repos
        except Exception as e:
            self.log_operation("list_repos", False, {"error": str(e)})
            raise
    
    def create_repository(self, name: str, description: str = "", private: bool = False) -> str:
        """Create a new GitHub repository."""
        try:
            repo_name = create_repo(name, description, private)
            self.log_operation("create_repository", True, {"repo": repo_name})
            return repo_name
        except Exception as e:
            self.log_operation("create_repository", False, {"error": str(e)})
            raise
    
    def search_repositories(self, query: str, sort: str = "stars", limit: int = 10) -> List[str]:
        """Search for GitHub repositories matching the query."""
        try:
            repos = search_repos(query, sort=sort, per_page=limit)
            self.log_operation("search_repositories", True, {"query": query, "count": len(repos)})
            return repos
        except Exception as e:
            self.log_operation("search_repositories", False, {"error": str(e)})
            raise
    
    # ---- File Operations ----
    
    def get_file_content(self, repo: str, path: str, branch: str = "main", owner: Optional[str] = None) -> str:
        """Get the content of a file from a GitHub repository."""
        try:
            content = read_file(repo, path, branch, owner)
            self.log_operation("get_file_content", True, {"repo": repo, "path": path})
            return content
        except Exception as e:
            self.log_operation("get_file_content", False, {"error": str(e)})
            raise
    
    def update_file(
        self, repo: str, path: str, message: str, content: str, 
        branch: str = "main", owner: Optional[str] = None
    ) -> str:
        """Create or update a file in a GitHub repository."""
        try:
            sha = commit_file(repo, path, message, content, branch, owner)
            self.log_operation("update_file", True, {"repo": repo, "path": path})
            return sha
        except Exception as e:
            self.log_operation("update_file", False, {"error": str(e)})
            raise
    
    def list_files(self, repo: str, branch: str = "main", owner: Optional[str] = None) -> List[str]:
        """List all files in a GitHub repository."""
        try:
            files = list_repo_files(repo, branch, owner)
            self.log_operation("list_files", True, {"repo": repo, "count": len(files)})
            return files
        except Exception as e:
            self.log_operation("list_files", False, {"error": str(e)})
            raise
            
    # ---- Issue Operations ----
    
    def list_repo_issues(self, repo: str, owner: Optional[str] = None, state: str = "open") -> List[Dict]:
        """List issues in a GitHub repository."""
        try:
            issues = list_issues(repo, owner, state)
            self.log_operation("list_repo_issues", True, {"repo": repo, "count": len(issues)})
            return issues
        except Exception as e:
            self.log_operation("list_repo_issues", False, {"error": str(e)})
            raise
            
    def create_issue(self, repo: str, title: str, body: str = "", labels: Optional[List[str]] = None, 
                    owner: Optional[str] = None) -> int:
        """Create a new issue in a GitHub repository."""
        try:
            issue_number = open_issue(repo, title, body, labels, owner)
            self.log_operation("create_issue", True, {"repo": repo, "issue_number": issue_number})
            return issue_number
        except Exception as e:
            self.log_operation("create_issue", False, {"error": str(e)})
            raise
            
    def close_repo_issue(self, repo: str, issue_number: int, owner: Optional[str] = None) -> str:
        """Close an issue in a GitHub repository."""
        try:
            result = close_issue(repo, issue_number, owner)
            self.log_operation("close_repo_issue", True, {"repo": repo, "issue_number": issue_number})
            return result
        except Exception as e:
            self.log_operation("close_repo_issue", False, {"error": str(e)})
            raise
            
    # ---- Pull Request Operations ----
    
    def list_pull_requests(self, repo: str, state: str = "open", owner: Optional[str] = None, 
                          base: Optional[str] = None) -> List[Any]:
        """List pull requests in a GitHub repository."""
        try:
            prs = list_prs(repo, state, owner, base)
            self.log_operation("list_pull_requests", True, {"repo": repo, "count": len(prs)})
            return prs
        except Exception as e:
            self.log_operation("list_pull_requests", False, {"error": str(e)})
            raise
            
    def create_pull_request(self, repo: str, title: str, head: str, base: str = "main", 
                           body: str = "", draft: bool = False, owner: Optional[str] = None) -> int:
        """Create a new pull request in a GitHub repository."""
        try:
            pr_number = create_pr(repo, title, head, base, body, draft, owner)
            self.log_operation("create_pull_request", True, {
                "repo": repo, "pr_number": pr_number, "head": head, "base": base
            })
            return pr_number
        except Exception as e:
            self.log_operation("create_pull_request", False, {"error": str(e)})
            raise
            
    # ---- Branch Operations ----
    
    def create_new_branch(self, repo: str, branch_name: str, source_branch: str = "main", 
                         owner: Optional[str] = None) -> dict:
        """Create a new branch in a GitHub repository."""
        try:
            result = create_branch(repo, branch_name, source_branch, owner)
            self.log_operation("create_new_branch", True, {
                "repo": repo, "branch": branch_name, "source": source_branch
            })
            return result
        except Exception as e:
            self.log_operation("create_new_branch", False, {"error": str(e)})
            raise
            
    def check_merge_compatibility(self, repo: str, head: str, base: str = "main", 
                                 owner: Optional[str] = None) -> Dict[str, Union[bool, str]]:
        """Check if branches can be merged without conflicts."""
        try:
            result = check_merge_status(repo, head, base, owner)
            self.log_operation("check_merge_compatibility", True, {
                "repo": repo, "head": head, "base": base, "mergeable": result["mergeable"]
            })
            return result
        except Exception as e:
            self.log_operation("check_merge_compatibility", False, {"error": str(e)})
            raise
            
    def merge_branch(self, repo: str, head: str, base: str = "main", commit_message: Optional[str] = None,
                    owner: Optional[str] = None, merge_method: str = "merge") -> Dict[str, Any]:
        """Merge one branch into another."""
        try:
            result = merge_branches(repo, head, base, commit_message, owner, merge_method)
            self.log_operation("merge_branch", True, {
                "repo": repo, "head": head, "base": base, "method": merge_method
            })
            return result
        except Exception as e:
            self.log_operation("merge_branch", False, {"error": str(e)})
            raise
            
    # ---- Commit Operations ----
    
    def list_repo_commits(self, repo: str, branch: str = "main", owner: Optional[str] = None, 
                         per_page: int = 100) -> List[str]:
        """List commits in a GitHub repository."""
        try:
            commits = list_commits(repo, branch, owner, per_page)
            self.log_operation("list_repo_commits", True, {"repo": repo, "count": len(commits)})
            return commits
        except Exception as e:
            self.log_operation("list_repo_commits", False, {"error": str(e)})
            raise
            
    # ---- Search Operations ----
    
    def search_github_issues(self, query: str, sort: str = "updated", order: str = "desc", 
                            per_page: int = 30) -> List[int]:
        """Search issues and pull requests across GitHub."""
        try:
            issues = search_issues(query, sort, order, per_page)
            self.log_operation("search_github_issues", True, {"query": query, "count": len(issues)})
            return issues
        except Exception as e:
            self.log_operation("search_github_issues", False, {"error": str(e)})
            raise
    
    # ---- Async versions of methods ----
    
    async def async_list_repos(self) -> List[str]:
        """Async version of list_repos."""
        return await asyncio.to_thread(self.list_repos)
        
    async def async_get_file_content(self, repo: str, path: str, branch: str = "main", 
                                    owner: Optional[str] = None) -> str:
        """Async version of get_file_content."""
        return await asyncio.to_thread(self.get_file_content, repo, path, branch, owner)
    
    async def async_update_file(self, repo: str, path: str, message: str, content: str,
                               branch: str = "main", owner: Optional[str] = None) -> str:
        """Async version of update_file."""
        return await asyncio.to_thread(self.update_file, repo, path, message, content, branch, owner)
        
    # Helper method to generate async versions of all methods
    @classmethod
    def _create_async_methods(cls):
        """
        Dynamically create async versions of all sync methods that don't already have async equivalents.
        This allows for easy use in async contexts.
        """
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('_') and not name.startswith('async_') and name != 'log_operation':
                # Create an async version of this method
                async_method_name = f"async_{name}"
                
                # Skip if async version already exists
                if hasattr(cls, async_method_name):
                    continue
                
                @wraps(method)
                async def async_wrapper(self, *args, **kwargs):
                    method_to_call = getattr(self, name)
                    return await asyncio.to_thread(method_to_call, *args, **kwargs)
                
                # Set the correct name and add to class
                async_wrapper.__name__ = async_method_name
                setattr(cls, async_method_name, async_wrapper)
                
# Create helper function for instantiating the toolkit with environment variables
def create_github_toolkit(token: Optional[str] = None) -> GitHubToolkit:
    """
    Create and return an instance of GitHubToolkit.
    
    Args:
        token (str, optional): GitHub token. If not provided, will look for 
                              GITHUB_TOKEN or GITHUB_PERSONAL_ACCESS_TOKEN_NEW env var.
                              
    Returns:
        GitHubToolkit: Initialized GitHub toolkit instance
    """
    token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN_NEW")
    if not token:
        raise ValueError("GitHub token not provided and not found in environment variables")
    
    return GitHubToolkit(token)


def create_github_tools() -> List[Tool]:
    """
    Create and return a list of Langchain tools for GitHub operations.
    
    Returns:
        List[Tool]: List of Langchain tools ready for use in an agent
    """
    return [
        Tool(
            name="close_issue",
            func=close_issue,
            description="Close an issue in a repository. Args: repo, issue_number, owner (optional)"
        ),
        Tool(
            name="list_my_repos",
            func=list_my_repos,
            description="Return all repositories the authenticated user can access"
        ),
        Tool(
            name="get_file_content",
            func=read_file,
            description="Fetch the raw content of a file from a GitHub repository. Args: repo, path, branch (optional), owner (optional)"
        ),
        Tool(
            name="update_file_content",
            func=commit_file,
            description="Create or update a file in a GitHub repository (single commit). Args: repo, path, message, content, branch (optional), owner (optional)"
        ),
        Tool(
            name="list_issues",
            func=list_issues,
            description="List issues in a repository (optionally filtered by state). Args: repo, owner (optional), state (optional)"
        ),
        Tool(
            name="create_issue",
            func=open_issue,
            description="Open a new issue in a repository. Args: repo, title, body (optional), labels (optional), owner (optional)"
        ),
        Tool(
            name="list_pull_requests",
            func=list_prs,
            description="List pull requests in a repository (optionally filtered by state). Args: repo, state (optional), owner (optional), base (optional)"
        ),
        Tool(
            name="create_pull_request",
            func=create_pr,
            description="Create a pull request from a branch to the default branch. Args: repo, title, head, base (optional), body (optional), draft (optional), owner (optional)"
        ),
        Tool(
            name="list_commits",
            func=list_commits,
            description="List commits on a branch (or the default branch) of a repository. Args: repo, branch (optional), owner (optional), per_page (optional)"
        ),
        Tool(
            name="search_repositories",
            func=search_repos,
            description="Search public repositories with a GitHub search query. Args: query, sort (optional), order (optional), per_page (optional)"
        ),
        Tool(
            name="search_issues",
            func=search_issues,
            description="Search issues and pull requests across GitHub with a query. Args: query, sort (optional), order (optional), per_page (optional)"
        ),
        Tool(
            name="create_repository",
            func=create_repo,
            description="Create a new GitHub repository for the authenticated user. Args: name, description (optional), private (optional), has_issues (optional), has_wiki (optional), auto_init (optional)"
        ),
        Tool(
            name="list_repo_files",
            func=list_repo_files,
            description="List files in a GitHub repository. Args: repo, branch (optional), owner (optional)"
        ),
        Tool(
            name="create_branch",
            func=create_branch,
            description="Create a new branch in a GitHub repository based on an existing branch. Args: repo, branch_name, source_branch (optional), owner (optional)"
        ),
        Tool(
            name="check_merge_compatibility",
            func=check_merge_status,
            description="Check if one branch can be merged into another branch without conflicts. Args: repo, head, base (optional), owner (optional)"
        ),
        Tool(
            name="merge_branches",
            func=merge_branches,
            description="Merge changes from one branch into another (head branch into base branch). Args: repo, head, base (optional), commit_message (optional), owner (optional), merge_method (optional)"
        ),
    ]


def create_github_agent(model_name: str = "google/gemini-2.0-flash-001", temperature: float = 0.0, verbose: bool = False):
    """
    Create and return a Langchain agent with GitHub tools.
    
    Args:
        model_name (str): The name of the LLM model to use
        temperature (float): Temperature setting for the LLM
        verbose (bool): Whether to enable verbose mode
        
    Returns:
        AgentExecutor: A ready-to-use agent with GitHub tools
    """
    # Initialize GitHub if not already done
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN_NEW") or os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise EnvironmentError("GitHub token not found in environment variables")
    
    init_github(token=github_token)
    
    # Create GitHub tools
    tools = create_github_tools()
    
    # Set up callback manager for tracing
    callback_manager = CallbackManager([ConsoleCallbackHandler()]) if verbose else None
    
    # Initialize LLM based on available API keys
    api_key = os.getenv("OPENAI_API_KEY") 
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if openrouter_key:
        # OpenRouter configuration
        llm = ChatOpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            model="google/gemini-2.0-flash-001" if "gemini" in model_name.lower() else model_name,
            temperature=temperature,
            callback_manager=callback_manager,
            max_completion_tokens=10000
        )
    elif api_key and "gpt" in model_name.lower():
        # OpenAI configuration
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            callback_manager=callback_manager
        )
    elif anthropic_key and "claude" in model_name.lower():
        # Anthropic configuration
        llm = ChatAnthropic(
            model=model_name, 
            temperature=temperature,
            callback_manager=callback_manager
        )
    else:
        # Default to OpenAI with appropriate warning
        available_key = api_key or anthropic_key or openrouter_key
        if not available_key:
            raise EnvironmentError("No API key found for OpenAI, Anthropic, or OpenRouter")
        
        print(f"Warning: Using default OpenAI model ({model_name}) as fallback")
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            callback_manager=callback_manager
        )
    
    # Agent system prompt
    system_prompt = """You are a helpful GitHub assistant that can help users manage their GitHub repositories.
    You have access to various GitHub operations like creating repositories, managing files, issues, pull requests, and more.
    Always use the appropriate tools to complete the user's request. Be thorough in your explanations.
    When dealing with code, verify it carefully before committing to repositories.
    
    For repository operations, unless the user specifies an owner, assume the default owner.
    
    TOOLS:
    ------
    You have access to the following tools:

    {tools}

    To use a tool, please use the following format:
    
    Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        
        
        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
    
    """
        
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(system_prompt)
        
        # Create agent
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        
    )

        # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=15
    )
        
    return agent_executor


# Run the interactive chat if executed directly
async def interactive_chat():
    """Run an interactive chat session with the GitHub agent."""
    # Initialize the agent
    
    agent = create_github_agent()
    
    
    print("GitHub-Agent ready. Type 'exit' to quit.")
    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        
        response = agent.invoke({"input": user_input})
        
        # Extract response text
        if hasattr(response, "output"):
            print("🤖 Agent:", response.output)
        else:
            print("🤖 Agent:", response)


if __name__ == "__main__":
    asyncio.run(interactive_chat())