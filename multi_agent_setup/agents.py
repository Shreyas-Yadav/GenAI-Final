"""
agents.py - Specialized agents for GitHub operations

This module defines several specialized agents that work together in a
multi-agent system for GitHub operations:

1. RepoAgent - Repository management (create, list repos)
2. IssuesAgent - Issue tracking (create, list, close issues)
3. ContentAgent - File operations (read, write, list files)
4. SearchAgent - Search operations (find repos, issues)
5. ControllerAgent - Orchestrates the specialized agents

Each agent has specific tools and responsibilities, with the ControllerAgent
serving as the coordinator.
"""

import os
from typing import List, Dict, Any, Optional

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.memory import ChatMemoryBuffer

# Import GitHub tools
from github_tools import (
    list_my_repos, read_file, commit_file, list_issues, 
    open_issue, list_prs, create_pr, list_commits, 
    search_repos, search_issues, create_repo, 
    close_issue, list_repo_files
)

class BaseAgent:
    """Base class for all specialized agents with common functionality."""
    
    def __init__(
        self, 
        llm: OpenRouter,
        tools: List[FunctionTool],
        name: str,
        description: str,
        verbose: bool = False
    ):
        self.name = name
        self.description = description
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        
        # Create the ReAct agent with the specified tools
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=verbose,
            memory=self.memory,
            max_iterations=10
        )

    def chat(self, message: str) -> str:
        """Process a message and return the response."""
        return self.agent.chat(message)
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class RepoAgent(BaseAgent):
    """Agent specialized in repository management."""
    
    def __init__(self, llm: OpenRouter, verbose: bool = False):
        # Tools specific to repository management
        tools = [
            FunctionTool.from_defaults(
                fn=list_my_repos,
                name="list_my_repos",
                description="Return all repositories the authenticated user can access",
            ),
            FunctionTool.from_defaults(
                fn=create_repo,
                name="create_repository",
                description="Create a new GitHub repository for the authenticated user",
            ),
            FunctionTool.from_defaults(
                fn=list_commits,
                name="list_commits",
                description="List commits on a branch (or the default branch) of a repository",
            ),
        ]
        
        super().__init__(
            llm=llm,
            tools=tools,
            name="RepoAgent",
            description="Specialized in repository creation and management",
            verbose=verbose
        )


class IssuesAgent(BaseAgent):
    """Agent specialized in issue tracking."""
    
    def __init__(self, llm: OpenRouter, verbose: bool = False):
        # Tools specific to issue management
        tools = [
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
                fn=close_issue,
                name="close_issue",
                description="Close an issue in a repository",
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
        ]
        
        super().__init__(
            llm=llm,
            tools=tools,
            name="IssuesAgent",
            description="Specialized in issue and PR tracking and management",
            verbose=verbose
        )


class ContentAgent(BaseAgent):
    """Agent specialized in file content operations."""
    
    def __init__(self, llm: OpenRouter, verbose: bool = False):
        # Tools specific to file content management
        tools = [
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
                fn=list_repo_files,
                name="list_repo_files",
                description="List files in a GitHub repository",
            ),
        ]
        
        super().__init__(
            llm=llm,
            tools=tools,
            name="ContentAgent",
            description="Specialized in file content operations and management",
            verbose=verbose
        )


class SearchAgent(BaseAgent):
    """Agent specialized in search operations."""
    
    def __init__(self, llm: OpenRouter, verbose: bool = False):
        # Tools specific to search operations
        tools = [
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
        
        super().__init__(
            llm=llm,
            tools=tools,
            name="SearchAgent",
            description="Specialized in searching GitHub repositories, issues, and PRs",
            verbose=verbose
        )


class ControllerAgent:
    """
    Orchestrator agent that delegates tasks to specialized agents.
    
    This agent analyzes user requests, determines which specialized agent
    should handle them, and coordinates the flow of information between agents.
    """
    
    def __init__(self, llm: OpenRouter, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        
        # Initialize specialized agents
        self.repo_agent = RepoAgent(llm, verbose)
        self.issues_agent = IssuesAgent(llm, verbose)
        self.content_agent = ContentAgent(llm, verbose)
        self.search_agent = SearchAgent(llm, verbose)
        
        # Create a registry of agents and their capabilities
        self.agents = {
            "repo": self.repo_agent,
            "issues": self.issues_agent,
            "content": self.content_agent,
            "search": self.search_agent
        }
        
        # Memory for the controller
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
        
        # Create tools for the controller to delegate to other agents
        delegation_tools = [self._create_delegation_tool(name, agent) 
                          for name, agent in self.agents.items()]
        
        # Controller's own direct tools for simple tasks
        controller_tools = [
            FunctionTool.from_defaults(
                fn=self._list_agents,
                name="list_agents",
                description="List all available specialized agents and their capabilities",
            ),
        ]
        
        # Create the controller agent with delegation tools
        self.agent = ReActAgent.from_tools(
            tools=delegation_tools + controller_tools,
            llm=llm,
            verbose=verbose,
            memory=self.memory,
            max_iterations=15
        )
    
    def _create_delegation_tool(self, agent_name: str, agent: BaseAgent) -> FunctionTool:
        """Create a tool for delegating tasks to a specialized agent."""
        
        def delegate_to_agent(query: str) -> str:
            """Delegate a query to a specialized agent and return its response."""
            if self.verbose:
                print(f"Delegating to {agent_name} agent: {query}")
            response = agent.chat(query)
            return response
        
        # Name the tool based on the agent
        function_name = f"delegate_to_{agent_name}_agent"
        delegate_to_agent.__name__ = function_name
        
        # Create and return the function tool
        return FunctionTool.from_defaults(
            fn=delegate_to_agent,
            name=function_name,
            description=f"Delegate a task to the {agent_name} agent. {agent.description}",
        )
    
    def _list_agents(self) -> str:
        """List all available specialized agents and their capabilities."""
        return "\n".join([f"- {agent}" for agent in self.agents.values()])
    
    def chat(self, message: str) -> str:
        """Process a user message and return the coordinated response."""
        # Ask the controller to analyze and delegate the task
        system_prompt = """
        You are a GitHub operations controller. Your job is to:
        1. Analyze the user's request to understand what they need
        2. Determine which specialized agent(s) should handle the request
        3. Delegate the relevant parts of the request to those agents
        4. Synthesize the results into a coherent response

        Available agents:
        - RepoAgent: Repository creation and management
        - IssuesAgent: Issue and PR tracking and management
        - ContentAgent: File content operations and management
        - SearchAgent: Searching GitHub repositories, issues, and PRs

        Use the appropriate delegation function for each agent.
        """
        
        # Prepend the system prompt to the user's message
        augmented_message = f"{system_prompt}\n\nUser request: {message}"
        
        # Process the request with the controller agent
        response = self.agent.chat(augmented_message)
        
        return response