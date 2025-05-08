# GitHub Agent

A powerful, conversational AI agent for automating GitHub operations through natural language commands.

## Overview

This project implements a GitHub automation agent using LlamaIndex's ReAct framework and OpenRouter's LLM API. The agent can perform a wide range of GitHub operations through simple conversational commands, making repository management more accessible and efficient.

## Features

### GitHub API Integration

- Complete wrapper around essential GitHub REST endpoints
- Consistent error handling with helpful error messages
- Authentication via GitHub Personal Access Token

### Available GitHub Operations

1. **Repository Management**

   - List repositories accessible to the authenticated user (`list_my_repos`)
   - Create new repositories (`create_repo`)
   - List files in a repository (`list_repo_files`)

2. **File Operations**

   - Read file content from repositories (`read_file`)
   - Create or update files with commit messages (`commit_file`)

3. **Issue Management**

   - List issues with filtering options (`list_issues`)
   - Create new issues with title, body, and labels (`open_issue`)
   - Close existing issues (`close_issue`)
   - Search issues across GitHub (`search_issues`)

4. **Pull Request Operations**

   - List pull requests with filtering options (`list_prs`)
   - Create pull requests between branches (`create_pr`)

5. **Branch Operations**

   - Create new branches from existing ones (`create_branch`)
   - Check merge compatibility between branches (`check_merge_status`)
   - Merge branches with different merge strategies (`merge_branches`)

6. **Commit Management**

   - List commits on a branch (`list_commits`)

7. **Search Capabilities**
   - Search repositories across GitHub (`search_repos`)
   - Search issues and PRs across GitHub (`search_issues`)

### AI Agent Capabilities

- Natural language understanding for GitHub operations
- ReAct (Reasoning and Acting) framework for step-by-step problem solving
- Interactive chat interface for user commands
- Verbose mode for debugging and understanding agent reasoning

## Technical Architecture

The project consists of three main components:

1. **github_tools.py**: Utility wrappers around GitHub REST endpoints

   - Typed signatures mirroring REST parameters
   - Consistent error handling
   - Shared global GitHub client

2. **master.py**: Agent configuration and setup

   - Initializes GitHub client with authentication
   - Configures OpenRouter LLM
   - Wraps GitHub functions as LlamaIndex tools
   - Creates ReAct agent with tools

3. **main.py**: Interactive interface
   - Provides chat-based interface for user interaction
   - Handles user input and agent responses

## Setup and Requirements

### Prerequisites

- Python 3.8+
- GitHub Personal Access Token
- OpenRouter API Key

### Environment Variables

The following environment variables need to be set:

- `GITHUB_PERSONAL_ACCESS_TOKEN_NEW`: Your GitHub personal access token
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `GITHUB_REPO_OWNER`: (Optional) Default GitHub repository owner

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   GITHUB_PERSONAL_ACCESS_TOKEN_NEW=your_github_token
   OPENROUTER_API_KEY=your_openrouter_key
   GITHUB_REPO_OWNER=your_github_username
   ```

## Usage

Run the interactive chat interface:

```
python main.py
```

### Example Commands

- "Show me my repositories, newest first"
- "Create a new repository called 'test-repo' with a README"
- "List all open issues in my repository 'project-x'"
- "Create a new branch called 'feature/login' in 'my-app' repository"
- "Commit a new file called 'README.md' to my 'docs' repository with content 'Hello World'"
- "Open an issue in 'bug-tracker' titled 'Fix login page' with description 'The login page is not working correctly'"
- "Check if branch 'feature/x' can be merged into 'main' in repository 'project-y'"

## Advanced Configuration

The agent uses Google's Gemini 2.0 Flash model by default, but you can modify `master.py` to use any model supported by OpenRouter:

```python
llm = OpenRouter(
    api_key=OPENROUTER_API_KEY,
    model="your-preferred-model",  # Change to any OpenRouter-hosted model
    max_tokens=100000,
    max_retries=5,
)
```

## Extending the Agent

To add new GitHub operations:

1. Implement the function in `github_tools.py`
2. Add the function to the `tools` list in `github_tools.py`
3. Create a FunctionTool wrapper in `master.py`
4. Add the tool to the agent's tools list

## Limitations

- The agent is limited to 15 iterations per request
- Some complex operations may require multiple steps
- Rate limits apply based on your GitHub account type

## Future Enhancements

- Support for GitHub Enterprise
- Integration with other Git providers
- Enhanced error recovery
- Support for more complex GitHub workflows
- Web interface for easier interaction
