"""
parallel_agents.py - Enhanced controller agent with parallel processing capabilities

This module extends the ControllerAgent to support parallel execution of tasks
across multiple specialized agents, improving efficiency for complex operations.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import time

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.memory import ChatMemoryBuffer

# Import from existing modules
from agents import BaseAgent
from controller_prompt import CONTROLLER_SYSTEM_PROMPT

class Task:
    """Represents a task to be executed by an agent."""
    
    def __init__(
        self, 
        agent_name: str, 
        query: str, 
        priority: int = 1,
        dependencies: List[str] = None
    ):
        self.id = f"task_{int(time.time() * 1000)}_{id(self)}"
        self.agent_name = agent_name
        self.query = query
        self.priority = priority
        self.dependencies = dependencies or []
        self.result = None
        self.completed = False
        self.failed = False
        self.error = None
    
    def __repr__(self) -> str:
        status = "completed" if self.completed else "failed" if self.failed else "pending"
        return f"Task(id={self.id}, agent={self.agent_name}, status={status}, priority={self.priority})"


class ParallelControllerAgent:
    """
    Enhanced controller agent that can process tasks in parallel.
    
    This agent can identify independent subtasks in a complex request
    and execute them concurrently when possible.
    """
    
    def __init__(
        self, 
        llm: OpenRouter, 
        specialized_agents: Dict[str, BaseAgent] = None,
        max_workers: int = 4,
        verbose: bool = False
    ):
        self.llm = llm
        self.verbose = verbose
        self.max_workers = max_workers
        
        # Initialize specialized agents if not provided
        if specialized_agents:
            self.agents = specialized_agents
        else:
            from agents import RepoAgent, IssuesAgent, ContentAgent, SearchAgent, BranchAgent
            self.agents = {
                "repo": RepoAgent(llm, verbose),
                "issues": IssuesAgent(llm, verbose),
                "content": ContentAgent(llm, verbose),
                "search": SearchAgent(llm, verbose),
                "branch": BranchAgent(llm, verbose)
            }
        
        # Memory for the controller
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
        
        # Create task planning tools
        self.task_planning_tools = [
            FunctionTool.from_defaults(
                fn=self._plan_tasks,
                name="plan_parallel_tasks",
                description="Plan a complex task as a series of parallel and sequential subtasks",
            ),
            FunctionTool.from_defaults(
                fn=self._execute_task_plan,
                name="execute_task_plan",
                description="Execute a planned set of tasks, potentially in parallel",
            ),
        ]
        
        # Create the task planner agent
        self.planner = ReActAgent.from_tools(
            tools=self.task_planning_tools,
            llm=llm,
            verbose=verbose,
            memory=self.memory,
            max_iterations=10
        )
        
        # ThreadPoolExecutor for parallel task execution
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task history
        self.task_history = []
    
    def _plan_tasks(self, request: str) -> str:
        """
        Analyze a complex request and break it into subtasks.
        
        Returns:
            JSON string representing the task plan with dependencies
        """
        # Example plan structure
        plan_prompt = f"""
        Analyze this request: "{request}"
        
        Break it down into subtasks for different agents. Each subtask should include:
        1. The agent to execute it (repo, issues, content, search)
        2. The specific query for that agent
        3. Priority (1=highest, 3=lowest)
        4. Dependencies (IDs of tasks that must complete first)
        
        Return a valid JSON array of task objects with these fields.
        
        Example format:
        [
          {{
            "agent": "search",
            "query": "Search for repositories with security issues",
            "priority": 1,
            "dependencies": []
          }},
          {{
            "agent": "issues",
            "query": "Create an issue in repo X about the security vulnerability",
            "priority": 2,
            "dependencies": ["task_1"]
          }}
        ]
        
        Create tasks for: {request}
        """
        
        # Use the LLM to plan tasks
        response = self.llm.complete(plan_prompt)
        
        if self.verbose:
            print(f"Task plan: {response}")
        
        # Note: In production code, add validation and error handling for the JSON response
        
        return response
    
    def _execute_task_plan(self, plan_json: str) -> str:
        """
        Execute a task plan, potentially running independent tasks in parallel.
        
        Args:
            plan_json: JSON string containing the task plan
            
        Returns:
            Summary of the execution results
        """
        import json
        
        try:
            # Parse the plan
            tasks_data = json.loads(plan_json)
            
            # Create Task objects
            tasks = []
            task_map = {}  # For looking up tasks by ID
            
            for i, task_data in enumerate(tasks_data):
                task = Task(
                    agent_name=task_data.get("agent"),
                    query=task_data.get("query"),
                    priority=task_data.get("priority", 1),
                    dependencies=[f"task_{j+1}" for j in range(i) if j+1 in task_data.get("dependencies", [])]
                )
                tasks.append(task)
                task_map[f"task_{i+1}"] = task
            
            # Update task dependencies with actual task IDs
            for task in tasks:
                resolved_deps = []
                for dep in task.dependencies:
                    if dep in task_map:
                        resolved_deps.append(task_map[dep].id)
                task.dependencies = resolved_deps
            
            # Execute the plan
            results = self._execute_tasks_parallel(tasks)
            
            # Compile results
            summary = ["Task execution results:"]
            for task in tasks:
                status = "✅ Completed" if task.completed else "❌ Failed"
                summary.append(f"{status}: {task.agent_name} - {task.query}")
                if task.completed:
                    # Truncate very long results for readability
                    result_text = task.result
                    if len(result_text) > 500:
                        result_text = result_text[:497] + "..."
                    summary.append(f"  Result: {result_text}")
                else:
                    summary.append(f"  Error: {task.error}")
            
            return "\n".join(summary)
            
        except Exception as e:
            return f"Error executing task plan: {str(e)}"
    
    async def _execute_single_task(self, task: Task) -> None:
        """Execute a single task with the appropriate agent."""
        try:
            if task.agent_name not in self.agents:
                raise ValueError(f"Unknown agent: {task.agent_name}")
            
            if self.verbose:
                print(f"Executing task: {task}")
            
            # Get the agent
            agent = self.agents[task.agent_name]
            
            # Execute the task
            result = agent.chat(task.query)
            
            # Update task status
            task.result = result
            task.completed = True
            
            if self.verbose:
                print(f"Task completed: {task.id}")
            
        except Exception as e:
            task.failed = True
            task.error = str(e)
            if self.verbose:
                print(f"Task failed: {task.id}, Error: {str(e)}")
    
    async def _execute_tasks_async(self, tasks: List[Task]) -> Dict[str, Any]:
        """
        Execute tasks with dependency management.
        
        This function respects dependencies between tasks and executes
        independent tasks concurrently.
        """
        # Track pending and completed tasks
        pending = {task.id: task for task in tasks}
        completed = {}
        
        # While there are pending tasks
        while pending:
            # Find tasks that can be executed (all dependencies satisfied)
            ready_tasks = []
            for task_id, task in list(pending.items()):
                if all(dep_id in completed for dep_id in task.dependencies):
                    ready_tasks.append(task)
                    del pending[task_id]
            
            if not ready_tasks:
                if pending:
                    # This indicates a circular dependency
                    raise ValueError("Circular dependency detected in tasks")
                break
            
            # Sort by priority
            ready_tasks.sort(key=lambda t: t.priority)
            
            # Execute tasks concurrently
            await asyncio.gather(
                *(self._execute_single_task(task) for task in ready_tasks)
            )
            
            # Add to completed
            for task in ready_tasks:
                completed[task.id] = task
        
        return completed
    
    def _execute_tasks_parallel(self, tasks: List[Task]) -> Dict[str, Any]:
        """Execute tasks in parallel using asyncio."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._execute_tasks_async(tasks))
        finally:
            loop.close()
    
    def chat(self, message: str) -> str:
        """Process a user message with parallel task execution when appropriate."""
        # First, determine if this is a complex request that benefits from parallelization
        complexity_check_prompt = f"""
        {CONTROLLER_SYSTEM_PROMPT}
        
        Analyze this request: "{message}"
        
        Is this a complex request that would benefit from parallel processing?
        Complex requests typically involve multiple independent operations like:
        - Searching multiple repositories simultaneously
        - Creating multiple issues or PRs
        - Reading and analyzing multiple files
        
        Answer 'yes' if parallelization would help, 'no' otherwise.
        """
        
        complexity_assessment = self.llm.complete(complexity_check_prompt).text.strip().lower()

        
        
        if "yes" in complexity_assessment and len(message.split()) > 5:
            if self.verbose:
                print("Using parallel processing for this request")
            
            # Plan the tasks
            plan = self._plan_tasks(message)
            
            # Execute the plan
            execution_results = self._execute_task_plan(plan)
            
            # Generate a summary response
            summary_prompt = f"""
            {CONTROLLER_SYSTEM_PROMPT}
            
            The user asked: "{message}"
            
            The system executed these tasks in parallel:
            {execution_results}
            
            Synthesize a clear, concise response that answers the user's original request
            and summarizes what was done. Don't mention the parallel processing unless relevant.
            """
            
            return self.llm.complete(summary_prompt)
        else:
            # For simpler requests, just use the standard controller approach
            if self.verbose:
                print("Using sequential processing for this request")
            
            # Determine which agent should handle this
            routing_prompt = f"""
            {CONTROLLER_SYSTEM_PROMPT}
            
            Analyze this request: "{message}"
            
            Which agent should handle this request? Choose one:
            - repo: for repository management
            - issues: for issue and PR operations
            - content: for file content operations
            - search: for search operations
            
            Answer with just the agent name.
            """
            
            agent_name = self.llm.complete(routing_prompt).strip().lower()
            
            if agent_name in self.agents:
                if self.verbose:
                    print(f"Delegating to {agent_name} agent")
                return self.agents[agent_name].chat(message)
            else:
                # If unclear, use repo agent as fallback
                return self.agents["repo"].chat(message)