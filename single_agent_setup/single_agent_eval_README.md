# GitHub Agent Evaluation Framework

## Introduction

This presentation introduces a comprehensive evaluation framework for GitHub agents built on LlamaIndex with Arize Phoenix telemetry. Our framework addresses a critical gap in agent system development: robust, automated evaluation of agent reasoning, tool selection, and execution efficiency.

### Why Agent Evaluation Matters

- Agents perform complex reasoning and make tool-calling decisions that affect outcomes
- Traditional metrics (success/failure) don't capture reasoning quality
- Poor intermediate decisions can lead to brittle agents prone to failure
- Understanding agent thinking enables targeted improvements

## Architecture Overview

```mermaid
flowchart TD
    %% Main components
    User(User) -->|"Queries (GitHub tasks)"| Agent
    Agent[GitHub Agent\n(LlamaIndex/ReAct)] -->|"Execute API calls"| GitHubAPI[GitHub API]
    GitHubAPI -->|"API responses"| Agent
    
    %% Telemetry capture
    Agent -->|"Emit trace data"| PhoenixTracing[Arize Phoenix\nTracing]
    PhoenixTracing -->|"Store spans"| PhoenixDB[(Phoenix DB)]
    
    %% Evaluation system
    PhoenixDB -->|"Query spans"| SpanQuery[Span Query DSL]
    SpanQuery -->|"Raw LLM spans"| ReactExtractor[ReAct Component\nExtractor]
    ReactExtractor -->|"Thoughts, tools,\naction inputs"| EvalPrep[Evaluation\nData Preparation]
    
    %% Three evaluations
    EvalPrep -->|"Individual steps"| T2TEval[Thought-to-Tool\nEvaluation]
    EvalPrep -->|"Individual steps"| Q2TEval[Query-to-Thought\nEvaluation]
    EvalPrep -->|"Grouped sequences"| SeqEval[Sequence Optimality\nEvaluation]
    
    %% GPT-4o evaluator
    T2TEval -->|"Prompt with data"| GPT4o[GPT-4o\nEvaluator]
    Q2TEval -->|"Prompt with data"| GPT4o
    SeqEval -->|"Prompt with data"| GPT4o
    
    %% Results processing
    GPT4o -->|"Evaluation results"| ResultProc[Results Processor]
    ResultProc -->|"Metrics and explanations"| Results[(Evaluation\nResults)]
    
    %% Outputs
    Results -->|"Log evaluations"| PhoenixViz[Phoenix\nVisualization]
    Results -->|"Detailed data"| CSV[CSV Reports]
    Results -->|"Summary statistics"| ConsoleOut[Console Output]
    
    %% Component styling
    classDef agent fill:#d9f7be,stroke:#389e0d
    classDef telemetry fill:#fff2e8,stroke:#d46b08
    classDef extraction fill:#e6f7ff,stroke:#096dd9
    classDef evaluation fill:#f6ffed,stroke:#52c41a
    classDef results fill:#f9f0ff,stroke:#722ed1
    
    class Agent agent
    class PhoenixTracing,PhoenixDB telemetry
    class SpanQuery,ReactExtractor,EvalPrep extraction
    class T2TEval,Q2TEval,SeqEval,GPT4o evaluation
    class ResultProc,Results,PhoenixViz,CSV,ConsoleOut results
```

Our evaluation framework implements a three-stage process:

1. **Trace Collection**: Capture agent interactions via Phoenix telemetry
2. **ReAct Extraction**: Parse thought processes and tool calls
3. **Multi-dimensional Evaluation**: Assess reasoning, tool selection, and execution efficiency

## ReAct Framework & Data Collection

The framework evaluates agents that follow the ReAct (Reasoning + Acting) paradigm:

```
Thought → Action → Action Input → Observation → Thought → ...
```

### Example ReAct Flow

```
Thought: I need to check if the user has any existing repositories first
Action: list_my_repos
Action Input: {}
Observation: [{"name": "project-alpha", "description": "A test project", ...}]
Thought: Now I'll check for open issues in the project-alpha repository
Action: list_issues
Action Input: {"repo": "project-alpha"}
...
```

### Data Extraction Process

```python
def extract_react_components(content):
    """Parse ReAct format to extract thought, tool call, and action input"""
    components = {
        "thought": "",
        "tool_call": "No tool used", 
        "action_input": ""
    }
    
    # Parse content line by line to extract components
    lines = content.split('\n')
    
    # Extract thought - capture everything between "Thought:" and "Action:"
    thought_lines = []
    recording_thought = False
    
    for line in lines:
        line = line.strip()
        if line.startswith("Thought:"):
            recording_thought = True
            # Get content after "Thought:" on same line
            rest_of_line = line[len("Thought:"):].strip()
            if rest_of_line:
                thought_lines.append(rest_of_line)
        elif recording_thought and line.startswith("Action:"):
            recording_thought = False
        elif recording_thought:
            thought_lines.append(line)
    
    components["thought"] = " ".join(thought_lines).strip()
    
    # Extract tool call and action input similarly...
    
    return components
```

The system fetches LLM spans from Phoenix, extracts the ReAct components (thoughts, tool calls, action inputs), and prepares this data for evaluation.

## Evaluation Methodology

Our framework implements three complementary evaluation dimensions:

### 1. Thought-to-Tool Evaluation

Assesses whether the agent chose the right GitHub tool based on its stated reasoning.

**Key Question**: Does the tool selection match the agent's reasoning about what it needs to do?

**Evaluation Template Excerpt**:
```
[BEGIN DATA]
************
[Original Question]: {original_question}
************
[Agent's Thought]: {thought}
************
[Tool Called]: {tool_call}
************
[Tool Action Input]: {action_input}
[END DATA]

Your task is to:
1. Identify what the agent is trying to accomplish in its current thought
2. Determine if the chosen tool is appropriate for accomplishing that specific goal
3. Make sure the parameters in the action input match what's needed for that goal

Response: "correct" or "incorrect"
```

**Example Evaluation**:
- **Thought**: "I need to check if the repository exists"
- **Tool**: `list_my_repos`
- **Evaluation**: ✓ Correct (tool matches intent)

### 2. Query-to-Thought Evaluation

Evaluates if the agent properly understands and plans to address the user's query.

**Key Question**: Does the agent's thinking align with what the user is asking for?

**Evaluation Template Excerpt**:
```
[BEGIN DATA]
************
[User Query]: {original_question}
************
[Agent's Current Thought]: {thought}
[END DATA]

Evaluate whether the agent's current thought reflects an understanding of and proper approach to the user's query.

"correct" means the thought:
- Shows the agent understands the user's request
- Is working on a relevant part of the overall task
- Is taking a logical step toward completing the request

Response: "correct" or "incorrect"
```

**Example Evaluation**:
- **Query**: "Create a new repo with a README"
- **Thought**: "I'll first check existing repos to avoid duplicates"
- **Evaluation**: ✓ Correct (demonstrates proactive planning)

### 3. Sequence Optimality Evaluation

Assesses whether the agent took the most efficient path to accomplish the task.

**Key Question**: Did the agent use an efficient sequence of steps without redundancy?

**Evaluation Template Excerpt**:
```
[BEGIN DATA]
************
[User Query]: {original_question}
************
[Complete Thought Sequence]:
{thought_sequence}
************
[Tool Call Sequence]:
{tool_sequence}
[END DATA]

Evaluate whether the agent used the most efficient sequence of thoughts and tool calls to accomplish the task.

"optimal" means the sequence represents the most efficient approach with no unnecessary steps.
"suboptimal" means a more efficient sequence could have been used.
```

**Example Evaluation**:
- **Query**: "Create an issue for bug #123"
- **Sequence**: Checks repos → Checks if issue exists → Creates issue
- **Evaluation**: ✓ Optimal (necessary verification steps)

## Technical Implementation

### LLM-Based Evaluation

The framework uses LLM-based evaluation (GPT-4o) to assess agent performance:

```python
# Create evaluation models
eval_model = OpenAIModel(model="gpt-4o")

# Run THOUGHT-TO-TOOL evaluation
thought_tool_template = ClassificationTemplate(
    rails=rails,
    template=IMPROVED_TOOL_CALLING_TEMPLATE,
    explanation_template=IMPROVED_TOOL_CALLING_TEMPLATE,
    scores=[1, 0]
)

thought_tool_classifications = llm_classify(
    data=input_data,
    template=thought_tool_template,
    model=eval_model,
    rails=rails,
    provide_explanation=True,
)
```

### Tool Definitions

The system defines GitHub tools with enhanced descriptions:

```python
# Create enhanced tool definitions with examples
tool_definitions = ""
for tool in tools:
    tool_name = tool.__name__
    tool_desc = tool.__doc__ or "No description available"
    
    # Add enhanced descriptions for common tools
    if tool_name == "update_file_content":
        tool_desc += " This tool is used to create or update files in a repository, including README files."
    elif tool_name == "commit_file":
        tool_desc += " This tool is used to commit code or text files to a repository."
    
    tool_definitions += f"\n{tool_name}: {tool_desc}\n"
```

### Sequence Data Preparation

For sequence evaluation, the system groups interactions by query:

```python
def prepare_sequence_data(eval_df):
    # Group by the original query to get all steps for each task
    grouped = eval_df.groupby('question_text')
    
    sequence_data = []
    
    for query, group in grouped:
        # Sort by timestamp if available, otherwise use the dataframe order
        sorted_group = group.reset_index()
        
        # Collect all thoughts and tools in sequence
        thought_sequence = "\n".join([
            f"Step {i+1}: {row['thought']}" 
            for i, (_, row) in enumerate(sorted_group.iterrows())
        ])
        
        tool_sequence = "\n".join([
            f"Step {i+1}: {row['tool_call']} - {row['action_input'][:50]}..." 
            for i, (_, row) in enumerate(sorted_group.iterrows())
        ])
        
        sequence_data.append({
            'original_question': query,
            'thought_sequence': thought_sequence,
            'tool_sequence': tool_sequence
        })
    
    return pd.DataFrame(sequence_data)
```

## Results & Analysis

### Evaluation Metrics

The framework provides multiple metrics:

1. **Thought-to-Tool Accuracy**: Percentage of tool selections that match agent reasoning
2. **Query-to-Thought Accuracy**: Percentage of thoughts that properly align with user queries
3. **Sequence Optimality**: Percentage of interactions that follow efficient paths
4. **Combined Accuracy**: Percentage of interactions that perform well across all dimensions
5. **Tool Usage Breakdown**: Frequency of each tool's usage

### Visualization & Integration

Results are:
1. Saved as CSV files for detailed analysis
2. Logged to Phoenix for visualization
3. Displayed in console output for immediate review

```python
# Log to Phoenix
px.Client().log_evaluations(
    SpanEvaluations(eval_name="GitHub Tool Calling Eval (Thought-to-Tool)", 
                    dataframe=thought_tool_classifications),
)

px.Client().log_evaluations(
    SpanEvaluations(eval_name="GitHub Query Interpretation Eval (Query-to-Thought)", 
                    dataframe=query_thought_classifications),
)

px.Client().log_evaluations(
    SpanEvaluations(eval_name="GitHub Sequence Optimality Eval", 
                    dataframe=sequence_classifications),
)
```

### Sample Output Analysis

```
Query-to-Thought accuracy: 92.35%
Thought-to-Tool accuracy: 87.56%
Sequence Optimality accuracy: 78.43%
Combined accuracy (both correct): 73.42%

Tool Usage Breakdown:
  list_my_repos: 15 uses
  read_file: 12 uses
  commit_file: 8 uses
  list_issues: 7 uses
  open_issue: 5 uses
  ...
```

## Technical Deep Dive: ReAct Extraction

The extraction of ReAct components is critical to reliable evaluation:

```python
# Function to safely extract React components from the LLM output messages
def get_react_components(row):
    try:
        outputs = row['outputs']
        
        # Handle structure where it's a list of message objects
        if isinstance(outputs, list) and len(outputs) > 0:
            message = outputs[0].get('message', {})
            content = message.get('content', '')
            return extract_react_components(content)
            
        # Handle structure where it's a string that needs parsing
        elif isinstance(outputs, str):
            try:
                parsed_outputs = json.loads(outputs)
                if isinstance(parsed_outputs, list) and len(parsed_outputs) > 0:
                    message = parsed_outputs[0].get('message', {})
                    content = message.get('content', '')
                    return extract_react_components(content)
            except:
                pass
        
        # Check the output field if exists
        if 'output' in row:
            output = row['output']
            if isinstance(output, dict) and 'value' in output:
                content = output.get('value', '')
                return extract_react_components(content)
                
        return {"thought": "", "tool_call": "No tool used", "action_input": ""}
    except Exception as e:
        print(f"Error extracting React components: {str(e)}")
        return {"thought": "", "tool_call": "No tool used", "action_input": ""}
```

The code handles multiple potential data formats:
- Structured message objects
- JSON strings
- Direct output values

## Technical Deep Dive: Evaluation Templates

The evaluation templates are carefully designed to ensure consistent, high-quality assessments:

```python
IMPROVED_TOOL_CALLING_TEMPLATE = """
You are an evaluation assistant looking at a conversational GitHub agent that uses tools to complete tasks. You're analyzing whether the agent chose the right GitHub tool based on its stated reasoning.

The agent follows a ReAct format: Thought → Action → Action Input → Observation. The "Thought" section contains the agent's reasoning, "Action" is the tool it chose, and "Action Input" contains the parameters for that tool.

Your job is to determine if the "Action" (tool choice) makes sense given the "Thought" (agent's reasoning).

    [BEGIN DATA]
    ************
    [Original Question]: {original_question}
    ************
    [Agent's Thought]: {thought}
    ************
    [Tool Called]: {tool_call}
    ************
    [Tool Action Input]: {action_input}
    [END DATA]

Your task is to:
1. Identify what the agent is trying to accomplish in its current thought
2. Determine if the chosen tool is appropriate for accomplishing that specific goal
3. Make sure the parameters in the action input match what's needed for that goal

Your response must be a single word, either "correct" or "incorrect".

[Available GitHub Tools]: {tool_definitions}

Please provide a clear EXPLANATION of your reasoning first, then end with the LABEL.
"""
```

This template:
1. Clearly defines the evaluation task
2. Provides all necessary context
3. Structures the response format
4. Gives access to tool definitions for informed assessment

## Future Directions & Extensions

### Potential Improvements

1. **Fine-grained Tool Parameter Evaluation**:
   - Evaluate parameter quality and completeness
   - Detect invalid parameter combinations

2. **Multi-step Planning Evaluation**:
   - Assess the agent's ability to create and follow plans
   - Evaluate handling of error conditions and recovery

3. **User Alignment Metrics**:
   - Measure how well agent actions align with user intent beyond query interpretation
   - Evaluate personalization to user preferences

4. **Cross-agent Comparative Analysis**:
   - Benchmark multiple agent implementations against each other
   - Identify technique and architecture advantages

### Integration Opportunities

1. **Continuous Evaluation Pipeline**:
   - Automate evaluation in CI/CD pipelines
   - Track performance across agent versions

2. **Real-time Agent Monitoring**:
   - Live evaluation of production agent systems
   - Alert on performance degradation

3. **Training Signal Integration**:
   - Use evaluation results to generate training data
   - Close the loop with RLHF or other learning approaches

## Conclusion

This GitHub Agent Evaluation Framework provides:

1. **Multi-dimensional assessment** of agent reasoning, tool selection, and efficiency
2. **Detailed, explainable metrics** through LLM-based evaluation
3. **Integration with Phoenix** for comprehensive visualization and analysis
4. **Extensible architecture** for additional evaluation dimensions

By evaluating not just outcomes but reasoning quality, the framework enables targeted improvements to agent systems and deeper understanding of agent behavior.

---

## References

1. ReAct: Synergizing Reasoning and Acting in Language Models (Yao, et al.)
2. LlamaIndex Agent Framework Documentation
3. Arize Phoenix Telemetry Documentation
4. GPT-4 Technical Report (OpenAI)
5. Evaluating LLM Reasoning (Zaib, et al.)