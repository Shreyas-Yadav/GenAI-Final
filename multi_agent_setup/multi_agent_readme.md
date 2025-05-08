# GitHub Multi-Agent System

A sophisticated multi-agent system for GitHub operations that leverages specialized agents to perform various tasks efficiently.

## Overview

This project implements a multi-agent system for interacting with GitHub, where each agent specializes in specific GitHub operations. The system uses a controller agent to orchestrate these specialized agents, delegating tasks based on their capabilities.

## Architecture

### Core Components

1. **Specialized Agents**

   - **RepoAgent**: Repository creation and management
   - **IssuesAgent**: Issue and PR tracking and management
   - **ContentAgent**: File content operations and management
   - **SearchAgent**: Searching GitHub repositories, issues, and PRs
   - **BranchAgent**: Branch creation, merge status checking, and branch merging operations

2. **Controller Agent**

   - Analyzes user requests to determine which specialized agent(s) should handle them
   - Breaks down complex requests into sub-tasks for different agents
   - Coordinates the flow of information between agents
   - Synthesizes responses from multiple agents into coherent answers

3. **Agent Factory**

   - Creates agents with different LLM backends (OpenRouter, OpenAI, etc.)
   - Provides flexible configuration for different agent roles

4. **GitHub Tools**
   - Utility wrappers around GitHub REST endpoints
   - Handles authentication, error handling, and response formatting

### System Flow

1. User submits a request to the controller agent
2. Controller analyzes the request and determines which specialized agent(s) to use
3. Controller delegates tasks to the appropriate specialized agents
4. Specialized agents execute their tasks using GitHub tools
5. Controller synthesizes the results and presents a coherent response to the user

## Entry Points

The system provides three different entry points:

1. **main.py**: Basic entry point with a simple controller agent
2. **advance_main.py**: Enhanced entry point with more flexible agent configuration

## Setup

### Prerequisites

- Python 3.8+
- GitHub Personal Access Token with the following permissions:
  - `repo` (Full control of private repositories)
  - `read:org` (Read organization information)
  - `user` (Update all user data)
- OpenRouter API Key (or other LLM provider keys)

### Dependencies

The project relies on the following key Python packages:

- **llama_index**: Core framework for building LLM-powered applications
- **PyGithub**: Python library to access the GitHub API
- **python-dotenv**: For loading environment variables from .env files
- **asyncio**: For asynchronous programming
- **requests**: For making HTTP requests

Additional dependencies may be required based on the LLM providers you choose to use.

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   GITHUB_PERSONAL_ACCESS_TOKEN_NEW=your_github_token
   OPENAI_API_KEY=your_openai_api_key (optional)
   ANTHROPIC_API_KEY=your_anthropic_api_key (optional)
   GITHUB_REPO_OWNER=your_github_username
   ```

## Usage

### Basic Usage

Run the basic version:

```bash
python main.py
```

This starts an interactive chat session where you can input GitHub-related requests.

### Advanced Usage

Run the advanced version with configurable LLM backends:

```bash
python advance_main.py
```

## Example Requests

- "List all my repositories"
- "Create a new repository called 'test-repo'"
- "Find issues with the label 'bug' in my repositories"
- "Read the README.md file from my project"
- "Create a pull request from branch 'feature' to 'main'"
- "Search for repositories related to machine learning"

## Components in Detail

### Specialized Agents

Each specialized agent is built on top of the `BaseAgent` class and is equipped with specific tools for its domain:

#### RepoAgent

- List repositories
- Create repositories
- List commits

#### IssuesAgent

- List issues
- Create issues
- Close issues
- List pull requests
- Create pull requests

#### ContentAgent

- Read file content
- Update file content
- List repository files

#### SearchAgent

- Search repositories
- Search issues

#### BranchAgent

- Create branches
- Check merge status
- Merge branches

### Controller Agent

The controller agent uses a system prompt to understand how to orchestrate the specialized agents. It analyzes user requests, determines which specialized agent(s) should handle them, and coordinates the flow of information.

## Technical Implementation

### LLM Integration

The system uses the LlamaIndex framework to integrate with various LLM providers:

- OpenRouter (default)
- OpenAI
- Anthropic (commented out in the code)

### GitHub API Integration

The system uses the PyGithub library to interact with the GitHub API, with custom wrappers for error handling and response formatting.

### Memory Usage

The system uses ChatMemoryBuffer with token limits to manage memory:

- Controller agent: 8000 tokens
- Specialized agents: 4000 tokens

For processing large repositories or files, consider adjusting these limits in the agent initialization.

## Limitations and Known Issues

- The system requires valid API keys for both GitHub and an LLM provider
- Rate limits may apply for both GitHub API and LLM API calls
- Very large repositories or files may cause performance issues
- The Anthropic integration is commented out in the code and requires uncommenting to use
- Error handling for invalid JSON responses from LLMs could be improved
