# GitHub Agent Evaluation Framework

This project implements a comprehensive evaluation framework for GitHub agents built on LlamaIndex with Arize Phoenix telemetry. The framework addresses a critical gap in agent system development by providing robust, automated evaluation of agent reasoning, tool selection, and execution efficiency.

## Project Structure

The project is organized into three main directories:

1. **single_agent_setup/** - Contains the implementation and evaluation of a single GitHub agent
2. **multi_agent_setup/** - Contains the implementation and evaluation of a multi-agent system
3. **communication_agents/** - Contains agent implementations for communication between machines

## Features

- **ReAct Framework Integration**: Evaluates agents that follow the Reasoning + Acting paradigm
- **Multi-dimensional Evaluation**: Assesses reasoning, tool selection, and execution efficiency
- **LLM-Based Evaluation**: Uses GPT-4o to evaluate agent performance
- **Phoenix Integration**: Visualizes and logs evaluation results

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`
- GitHub Personal Access Token
- OpenRouter API Key

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd GenAI-Final
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following variables:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token
   PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
   ```

## Running the Single Agent Experiment

To run the single agent experiment with predefined test questions:

```bash
cd single_agent_setup
python experiment.py
```

This will run the agent with predefined test questions and record the results in Phoenix.

## Running the Interactive Single Agent

For an interactive chat session:

```bash
cd single_agent_setup
python main.py
```

## Running the Multi-Agent Experiment

To run the multi-agent system with predefined test questions:

```bash
cd multi_agent_setup
python multi_experiment.py
```

This will run the multi-agent system with predefined test questions and record the results in Phoenix.

## Running the Evaluation

To evaluate the agent's performance:

```bash
cd single_agent_setup
python evaluate_github_agent.py
```

This will run the evaluation framework and log the results to Phoenix.

## Evaluation Methodology

The framework implements three complementary evaluation dimensions:

1. **Thought-to-Tool Evaluation**: Assesses whether the agent chose the right GitHub tool based on its stated reasoning.
2. **Query-to-Thought Evaluation**: Evaluates if the agent properly understands and plans to address the user's query.
3. **Sequence Optimality Evaluation**: Assesses whether the agent took the most efficient path to accomplish the task.

## Visualization & Integration

Results are:
1. Saved as CSV files for detailed analysis
2. Logged to Phoenix for visualization
3. Displayed in console output for immediate review

## License

[Specify your license here]

## Contributors

[List contributors here]