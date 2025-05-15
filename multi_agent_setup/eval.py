import os
import nest_asyncio
import pandas as pd
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure we have the necessary API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")

import nest_asyncio
import pandas as pd

from phoenix.evals import (
    TOOL_CALLING_PROMPT_RAILS_MAP,
    OpenAIModel,
    llm_classify,
)

nest_asyncio.apply()

import phoenix as px
from phoenix.otel import register

# Set up Phoenix project
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
project_name = "multi-agent-experiment-4"  # Set your project name here
os.environ["PHOENIX_PROJECT_NAME"] = project_name

tracer_provider = register(auto_instrument=True, project_name=project_name)

from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery

# Query for LLM spans
query = (
    SpanQuery()
    .where(
        # Filter for the `LLM` span kind
        "span_kind == 'LLM'",
    )
    .select(
        # Extract and rename the following span attributes
        span_kind="span_kind",
        question="llm.input_messages",
        outputs="llm.output_messages",
        input="input",
        output="output"
    )
)

print("Fetching spans from Phoenix...")
trace_df = px.Client().query_spans(query, project_name=project_name)
print(f"Found {len(trace_df)} LLM spans")

# Improved function to extract the thought, tool call, and action input from the output message content
def extract_react_components(content):
    """Parse ReAct format to extract thought, tool call, and action input with better robustness"""
    if not content:
        return {"thought": "", "tool_call": "No tool used", "action_input": ""}
    
    components = {
        "thought": "",
        "tool_call": "No tool used", 
        "action_input": ""
    }
    
    try:
        # Handle different line endings and whitespace
        content = content.replace('\r\n', '\n').replace('\r', '\n')
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
        
        # Extract tool call
        for line in lines:
            line = line.strip()
            if line.startswith("Action:"):
                components["tool_call"] = line[len("Action:"):].strip()
                break
        
        # Extract action input
        action_input_lines = []
        recording_action_input = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("Action Input:"):
                recording_action_input = True
                # Get content after "Action Input:" on same line
                rest_of_line = line[len("Action Input:"):].strip()
                if rest_of_line:
                    action_input_lines.append(rest_of_line)
            elif recording_action_input and line.startswith("Observation:"):
                recording_action_input = False
            elif recording_action_input:
                action_input_lines.append(line)
        
        components["action_input"] = " ".join(action_input_lines).strip()
        
        # Print full content for debugging if extraction produces empty thought
        if components["tool_call"] != "No tool used" and not components["thought"]:
            print("WARNING: Extracted a tool call without a thought. Raw content:")
            print(content)
            # Use a minimum placeholder thought to avoid empty thought
            components["thought"] = f"Using {components['tool_call']} tool"
    
    except Exception as e:
        print(f"Error extracting React components: {e}")
        print(f"Problematic content: {content[:200]}...")
    
    return components

# Function to diagnose thought extraction issues
def diagnose_thought_extraction(trace_df):
    """Analyze why thoughts might not be extracted properly"""
    print("\n=== THOUGHT EXTRACTION DIAGNOSIS ===")
    
    # Check if any thoughts are missing
    empty_thoughts = trace_df[trace_df['thought'] == '']
    if len(empty_thoughts) > 0:
        print(f"Found {len(empty_thoughts)} rows with empty thoughts")
        
        # Sample one problematic row
        if len(empty_thoughts) > 0:
            sample_row = empty_thoughts.iloc[0]
            print("\nSample row with missing thought:")
            print(f"Tool call: {sample_row['tool_call']}")
            
            # Look at raw output
            print("\nRaw output for this row:")
            outputs = sample_row['outputs']
            if isinstance(outputs, str):
                print(outputs[:500])
            elif isinstance(outputs, list) and len(outputs) > 0:
                print(str(outputs[0])[:500])
            else:
                print(f"Output type: {type(outputs)}")
    
    # Sample of successful extractions
    successful = trace_df[trace_df['thought'] != '']
    if len(successful) > 0:
        print(f"\nSuccessfully extracted {len(successful)} thoughts")
        sample = successful.iloc[0]
        print("\nSample successful extraction:")
        print(f"Thought: {sample['thought'][:200]}...")
        print(f"Tool: {sample['tool_call']}")
    
    print("\n=== END DIAGNOSIS ===\n")
    
    return

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

# Function to extract user's question
def extract_question(row):
    try:
        question = row['question']
        
        # Handle structure where it's a list of message objects
        if isinstance(question, list):
            for msg in question:
                if isinstance(msg, dict) and 'message' in msg:
                    if msg['message'].get('role') == 'user':
                        return msg['message'].get('content', '')
        
        # Handle input field if we couldn't extract from question
        if 'input' in row and isinstance(row['input'], dict):
            input_val = row['input'].get('value', '')
            if input_val:
                try:
                    parsed = json.loads(input_val)
                    if isinstance(parsed, dict) and 'messages' in parsed:
                        for msg in parsed['messages']:
                            if isinstance(msg, str) and 'role=<MessageRole.USER' in msg:
                                # Extract text between text=' and '
                                parts = msg.split("text='")
                                if len(parts) > 1:
                                    text_part = parts[1]
                                    end_idx = text_part.find("')")
                                    if end_idx > 0:
                                        return text_part[:end_idx]
                except:
                    pass
        
        return "Unknown input"
    except Exception as e:
        print(f"Error extracting question: {str(e)}")
        return "Unknown input"

# Apply the extraction functions
trace_df['question_text'] = trace_df.apply(extract_question, axis=1)

# Extract the React components
react_components = trace_df.apply(get_react_components, axis=1)
trace_df['thought'] = react_components.apply(lambda x: x['thought'])
trace_df['tool_call'] = react_components.apply(lambda x: x['tool_call'])
trace_df['action_input'] = react_components.apply(lambda x: x['action_input'])

# Run diagnosis on thought extraction
diagnose_thought_extraction(trace_df)

# Sample of extracted components for debugging
print("\nSample of extracted components:")
for i, row in trace_df.head().iterrows():
    print(f"Row {i}:")
    print(f"  Question: {row['question_text'][:50]}...")
    print(f"  Thought: {row['thought'][:100]}...")
    print(f"  Tool: {row['tool_call']}")
    print(f"  Action Input: {row['action_input'][:50]}...")
    print()

# Filter out rows with no tool used and drop duplicates
eval_df = trace_df[trace_df['tool_call'] != "No tool used"].drop_duplicates(subset=['thought', 'tool_call'])
print(f"Filtered to {len(eval_df)} valid tool call examples")

# Create counts of the different types of tools/agents called
tool_counts = eval_df['tool_call'].value_counts()
print("\nTool/Agent Call Distribution:")
for tool, count in tool_counts.items():
    print(f"  {tool}: {count} calls")

# Save the raw data to CSV for debugging
eval_df.to_csv('multi_agent_raw_tool_calls.csv', index=False)
print("Raw tool call data saved to 'multi_agent_raw_tool_calls.csv'")

if len(eval_df) == 0:
    print("No valid tool calls found in the spans. Make sure your agent has run with tracing enabled.")
else:
    # Import tool definitions
    try:
        from github_tools import (
            list_my_repos, read_file, commit_file, list_issues, open_issue, 
            list_prs, create_pr, list_commits, search_repos, search_issues,
            create_repo, close_issue, list_repo_files, update_file_content
        )
        
        # Define all tools - including update_file_content
        tools = [
            list_my_repos, read_file, commit_file, list_issues, open_issue, 
            list_prs, create_pr, list_commits, search_repos, search_issues,
            create_repo, close_issue, list_repo_files, update_file_content
        ]
    except ImportError:
        # If update_file_content is not available, create a mock for it
        from github_tools import (
            list_my_repos, read_file, commit_file, list_issues, open_issue, 
            list_prs, create_pr, list_commits, search_repos, search_issues,
            create_repo, close_issue, list_repo_files
        )
        
        # Define a mock update_file_content function
        def update_file_content(repo, path, message, content, owner=None):
            """Creates or updates a file in a repository with the given content. Used for creating or updating files like README.md."""
            pass
        
        # Add the mock to the tools list
        tools = [
            list_my_repos, read_file, commit_file, list_issues, open_issue, 
            list_prs, create_pr, list_commits, search_repos, search_issues,
            create_repo, close_issue, list_repo_files, update_file_content
        ]

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
        elif tool_name == "create_repo":
            tool_desc += " This tool is used to create a new GitHub repository."
        elif tool_name == "open_issue":
            tool_desc += " This tool is used to create new issues in a repository."
        elif tool_name == "close_issue":
            tool_desc += " This tool is used to close existing issues in a repository."
            
        tool_definitions += f"\n{tool_name}: {tool_desc}\n"
    
    # Add agent delegation as tools
    agent_tools = {
        "delegate_to_repo_agent": "Delegates a repository-related task to the RepoAgent which specializes in creating and managing repositories.",
        "delegate_to_issues_agent": "Delegates an issue-related task to the IssuesAgent which specializes in creating, managing, and closing issues.",
        "delegate_to_content_agent": "Delegates a content-related task to the ContentAgent which specializes in creating, reading, and modifying files and content.",
        "delegate_to_search_agent": "Delegates a search-related task to the SearchAgent which specializes in searching for repositories, issues, and other GitHub entities.",
    }
    
    # Add agent delegations to the tool definitions
    for agent_name, agent_desc in agent_tools.items():
        tool_definitions += f"\n{agent_name}: {agent_desc}\n"

    print("Tool and agent definitions created")
    
    # Check if agent delegation appears in tool calls
    agent_delegation_calls = [call for call in eval_df['tool_call'] if call.startswith('delegate_to_')]
    if agent_delegation_calls:
        print(f"\nNOTE: Found {len(agent_delegation_calls)} agent delegation calls")
        print("Treating agent delegation as specialized tool calls")
    
    # Create an improved evaluation template for thought-to-tool that includes agent delegations
    IMPROVED_TOOL_CALLING_TEMPLATE = """
    You are an evaluation assistant looking at a conversational GitHub multi-agent system that uses tools and agent delegations to complete tasks. You're analyzing whether the agent chose the right tool or agent delegation based on its stated reasoning.

    The agent follows a ReAct format: Thought → Action → Action Input → Observation. The "Thought" section contains the agent's reasoning, "Action" is the tool or agent delegation it chose, and "Action Input" contains the parameters for that action.

    Your job is to determine if the "Action" (tool or agent delegation) makes sense given the "Thought" (agent's reasoning).

        [BEGIN DATA]
        ************
        [Original Question]: {original_question}
        ************
        [Agent's Thought]: {thought}
        ************
        [Tool/Agent Called]: {tool_call}
        ************
        [Action Input]: {action_input}
        [END DATA]

    Your task is to:
    1. Identify what the agent is trying to accomplish in its current thought
    2. Determine if the chosen tool or agent delegation is appropriate for accomplishing that specific goal
    3. Make sure the parameters in the action input match what's needed for that goal

    Your response must be a single word, either "correct" or "incorrect".

    "correct" means:
    - The tool/agent choice directly addresses what the agent said it wanted to do in its thought
    - The parameters in the action input are appropriate for that specific task
    - If delegating to another agent, the delegation is to the appropriate specialized agent for the task

    "incorrect" means:
    - The tool/agent choice doesn't match what the agent said it wanted to do in its thought
    - The parameters are missing important information or contain incorrect values
    - If delegating to another agent, a different agent would have been more appropriate

    [Available GitHub Tools and Agent Delegations]: {tool_definitions}

    Please provide a clear EXPLANATION of your reasoning first, then end with the LABEL. Focus only on whether the tool/agent choice makes sense given the agent's thought.

    Example response:
    ************
    EXPLANATION: Looking at the agent's thought, it wants to [what the agent is trying to do]. The tool/agent it selected is [tool/agent name], which [analysis of why this choice is right or wrong for this specific purpose]. The parameters it provided [analysis of whether parameters match the intent].

    Based on this analysis, the tool/agent call is...

    LABEL: "correct" or "incorrect"
    ************

    EXPLANATION:
    """

    # Create Query-to-Thought evaluation template for multi-agent
    QUERY_TO_THOUGHT_TEMPLATE = """
    You are an evaluation assistant analyzing a GitHub multi-agent system's understanding of user requests.
    
    Your task is to determine if the agent's thought is properly aligned with the user's query.
    
    [BEGIN DATA]
    ************
    [User Query]: {original_question}
    ************
    [Agent's Current Thought]: {thought}
    [END DATA]
    
    Evaluate whether the agent's current thought reflects an understanding of and proper approach to the user's query.
    
    Your response must be EXACTLY one of these two words: "correct" or "incorrect".
    
    "correct" means the thought:
    - Shows the agent understands the user's request
    - Is working on a relevant part of the overall task
    - Is taking a logical step toward completing the request
    - Recognizes when to use specialized agents for specific subtasks
    
    "incorrect" means the thought:
    - Misunderstands what the user wants
    - Is focused on something irrelevant
    - Is proceeding in a way that won't help fulfill the request
    - Fails to recognize when to use specialized agents for specific subtasks
    
    First provide your EXPLANATION analyzing how well the thought aligns with the query, then provide your LABEL, following this format exactly:
    
    EXPLANATION: [Your analysis here...]
    
    LABEL: [MUST BE EXACTLY "correct" OR "incorrect"]
    """
    
    # Define the Sequence Optimality template for multi-agent systems
    SEQUENCE_OPTIMALITY_TEMPLATE = """
    You are evaluating whether a GitHub multi-agent system took the optimal sequence of steps to accomplish a user's task.
    
    [BEGIN DATA]
    ************
    [User Query]: {original_question}
    ************
    [Complete Thought Sequence]:
    {thought_sequence}
    ************
    [Tool and Agent Delegation Sequence]:
    {tool_sequence}
    [END DATA]
    
    Evaluate whether the multi-agent system used the most efficient sequence of thoughts, tool calls, and agent delegations to accomplish the task.
    
    Your response must be EXACTLY one of these two words: "optimal" or "suboptimal".
    
    "optimal" means:
    - The sequence represents the most efficient approach with no unnecessary steps
    - Agent delegations were used appropriately to handle specialized tasks
    - The work was distributed efficiently among specialized agents
    - The order of operations makes logical sense
    
    "suboptimal" means:
    - There are unnecessary steps or redundant operations
    - Agent delegations were missing when they would have been useful, or used inappropriately
    - The wrong agents were used for certain tasks
    - The workflow could have been more efficiently organized
    
    Please provide a clear EXPLANATION of your reasoning, then end with the LABEL.
    
    EXPLANATION:
    
    LABEL: [MUST BE EXACTLY "optimal" OR "suboptimal"]
    """
    
    # Create evaluation models
    eval_model = OpenAIModel(model="gpt-4o")
    print("Using GPT-4o for evaluation")

    # Prepare the data for llm_classify with correct variable names
    input_data = pd.DataFrame({
        'original_question': eval_df['question_text'],
        'thought': eval_df['thought'],
        'tool_call': eval_df['tool_call'],
        'action_input': eval_df['action_input'],
        'tool_definitions': tool_definitions
    })
    
    # Print sample evaluation input for debugging
    print("\nSample evaluation input:")
    for i, row in input_data.head().iterrows():
        print(f"Row {i}:")
        print(f"  Original question: {row['original_question'][:50]}...")
        print(f"  Thought: {row['thought'][:100]}...")
        print(f"  Tool/Agent: {row['tool_call']}")
        print(f"  Action Input: {row['action_input'][:50]}...")
        print()
    
    # Save the input data to CSV for debugging
    input_data.to_csv('multi_agent_eval_input_data.csv', index=False)
    print("Evaluation input data saved to 'multi_agent_eval_input_data.csv'")
    
    # Get rails for classification
    rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())

    # Run THOUGHT-TO-TOOL evaluation
    print("\nRunning Thought-to-Tool/Agent Evaluation...")
    from phoenix.evals import ClassificationTemplate
    
    # Create a custom template with our improved prompt
    thought_tool_template = ClassificationTemplate(
        rails=rails,
        template=IMPROVED_TOOL_CALLING_TEMPLATE,
        explanation_template=IMPROVED_TOOL_CALLING_TEMPLATE,
        scores={"correct": 1, "incorrect": 0}
    )
    
    thought_tool_classifications = llm_classify(
        data=input_data,
        template=thought_tool_template,
        model=eval_model,
        rails=rails,
        provide_explanation=True,
    )
    
    # Add score column
    thought_tool_classifications["score"] = thought_tool_classifications.apply(
        lambda x: 1 if x["label"] == "correct" else 0, axis=1
    )
    
    # Save the evaluation results to CSV
    thought_tool_classifications.to_csv('multi_agent_thought_tool_evaluations.csv', index=False)
    print("Thought-to-Tool/Agent evaluation results saved to 'multi_agent_thought_tool_evaluations.csv'")
    
    # Calculate overall accuracy for thought-to-tool
    tt_accuracy = thought_tool_classifications["score"].mean() * 100
    print(f"\nThought-to-Tool/Agent accuracy: {tt_accuracy:.2f}%")
    
    # Calculate accuracy for agent delegations vs regular tool calls
    agent_delegation_indices = input_data[input_data['tool_call'].str.startswith('delegate_to_')].index
    tool_call_indices = input_data[~input_data['tool_call'].str.startswith('delegate_to_')].index
    
    if len(agent_delegation_indices) > 0:
        agent_delegation_accuracy = thought_tool_classifications.loc[agent_delegation_indices, "score"].mean() * 100
        print(f"Agent Delegation accuracy: {agent_delegation_accuracy:.2f}% ({len(agent_delegation_indices)} examples)")
    
    if len(tool_call_indices) > 0:
        tool_call_accuracy = thought_tool_classifications.loc[tool_call_indices, "score"].mean() * 100
        print(f"Tool Call accuracy: {tool_call_accuracy:.2f}% ({len(tool_call_indices)} examples)")
    
    # Run QUERY-TO-THOUGHT evaluation
    print("\nRunning Query-to-Thought Evaluation...")
    
    # Create subset of data for query-to-thought eval
    query_thought_data = pd.DataFrame({
        'original_question': input_data['original_question'],
        'thought': input_data['thought'],
    })

    # Setup template for query-to-thought evaluation
    query_thought_template = ClassificationTemplate(
        rails=rails,
        template=QUERY_TO_THOUGHT_TEMPLATE,
        explanation_template=QUERY_TO_THOUGHT_TEMPLATE,
        scores={"correct": 1, "incorrect": 0}
    )
    
    # Run evaluation with clearer labels
    query_thought_classifications = llm_classify(
        data=query_thought_data,
        template=query_thought_template,
        model=eval_model,
        rails=rails,
        provide_explanation=True,
    )

    # The score calculation
    query_thought_classifications["score"] = query_thought_classifications.apply(
        lambda x: 1 if x["label"] == "correct" else 0, axis=1
    )
    
    # Save the query-to-thought evaluation results to CSV
    query_thought_classifications.to_csv('multi_agent_query_thought_evaluations.csv', index=False)
    print("Query-to-Thought evaluation results saved to 'multi_agent_query_thought_evaluations.csv'")
    
    # Calculate overall accuracy for query-to-thought
    qt_accuracy = query_thought_classifications["score"].mean() * 100
    print(f"\nQuery-to-Thought accuracy: {qt_accuracy:.2f}%")
    
    # Calculate accuracy for different action types (agent delegations vs. regular tool calls)
    if len(agent_delegation_indices) > 0:
        qt_agent_delegation_accuracy = query_thought_classifications.loc[agent_delegation_indices, "score"].mean() * 100
        print(f"Query-to-Thought accuracy for Agent Delegations: {qt_agent_delegation_accuracy:.2f}%")
    
    if len(tool_call_indices) > 0:
        qt_tool_call_accuracy = query_thought_classifications.loc[tool_call_indices, "score"].mean() * 100
        print(f"Query-to-Thought accuracy for Tool Calls: {qt_tool_call_accuracy:.2f}%")

    # Add the Sequence Optimality Evaluation
    print("\nRunning Sequence Optimality Evaluation...")
    
    # Sequence preparation function
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
    
    # Run sequence evaluation
    sequence_data = prepare_sequence_data(eval_df)
    
    sequence_template = ClassificationTemplate(
        rails=rails,
        template=SEQUENCE_OPTIMALITY_TEMPLATE,
        explanation_template=SEQUENCE_OPTIMALITY_TEMPLATE,
        scores={"optimal": 1, "suboptimal": 0}
    )
    
    sequence_classifications = llm_classify(
        data=sequence_data,
        template=sequence_template,
        model=eval_model,
        rails=rails,
        provide_explanation=True,
    )
    
    sequence_classifications["score"] = sequence_classifications.apply(
        lambda x: 1 if x["label"] == "optimal" else 0, axis=1
    )
    
    # Save the sequence evaluation results to CSV
    sequence_classifications.to_csv('multi_agent_sequence_optimality_evaluations.csv', index=False)
    print("Sequence Optimality evaluation results saved to 'multi_agent_sequence_optimality_evaluations.csv'")
    
    # Calculate overall sequence optimality
    seq_accuracy = sequence_classifications["score"].mean() * 100
    print(f"\nSequence Optimality accuracy: {seq_accuracy:.2f}%")
    
    # Add this before logging to Phoenix
    sequence_classifications = sequence_classifications.reset_index(drop=True)
    
    # Map each sequence evaluation to a valid span_id - use proper column names
    # First print column names to debug
    print("Columns in sequence_classifications:", sequence_classifications.columns.tolist())
    
    # Create mapping manually
    span_id_map = {}
    for i, row in eval_df.groupby('question_text').first().reset_index().iterrows():
        question = row['question_text']
        span_id = eval_df[eval_df['question_text'] == question].index[0]
        span_id_map[question] = span_id
    
    # Take the first column (which should be the question) and use it for mapping
    first_col = sequence_classifications.columns[0]  # Use the first column, which should be the question
    sequence_classifications['context.span_id'] = sequence_classifications[first_col].map(span_id_map)
    sequence_classifications = sequence_classifications.set_index('context.span_id')

    # Log to Phoenix
    px.Client().log_evaluations(
        SpanEvaluations(eval_name="Multi-Agent GitHub Sequence Optimality Eval", dataframe=sequence_classifications),
    )
    print("Sequence Optimality evaluation results logged to Phoenix")
    
    # Display sequence evaluation results
    print("\nSequence Optimality Evaluation:")
    for i in range(len(sequence_classifications)):
        print("Task " + str(i+1) + ":")
        
        # Get query
        query = sequence_data.iloc[i]['original_question']
        if isinstance(query, str) and len(query) > 100:
            query = query[:100] + "..."
        
        # Get evaluation data
        label = sequence_classifications.iloc[i]['label']
        score = sequence_classifications.iloc[i]['score']
        explanation = "No explanation provided"
        if 'explanation' in sequence_classifications.columns:
            explanation = sequence_classifications.iloc[i]['explanation']
            if isinstance(explanation, str) and len(explanation) > 150:
                explanation = explanation[:150] + "..."
        
        print("  Query: " + str(query))
        print("  Evaluation: " + str(label) + " (Score: " + str(score) + ")")
        print("  Explanation: " + str(explanation))
        print()
    
    # Display detailed evaluation results for both evaluations
    print("\n=== DETAILED EVALUATION RESULTS ===")
    
    print("\nThought-to-Tool/Agent Evaluation:")
    for i in range(len(thought_tool_classifications)):
        print("Example " + str(i+1) + ":")
        
        # Safely get data
        thought = ""
        tool_call = ""
        if i < len(input_data):
            thought = input_data.iloc[i]['thought']
            if isinstance(thought, str) and len(thought) > 100:
                thought = thought[:100] + "..."
            tool_call = input_data.iloc[i]['tool_call']
        
        # Get evaluation data
        label = ""
        score = ""
        explanation = ""
        if i < len(thought_tool_classifications):
            label = thought_tool_classifications.iloc[i]['label']
            score = thought_tool_classifications.iloc[i]['score']
            if 'explanation' in thought_tool_classifications.columns:
                explanation = thought_tool_classifications.iloc[i]['explanation']
                if isinstance(explanation, str) and len(explanation) > 150:
                    explanation = explanation[:150] + "..."
        
        # Highlight whether this is a tool call or agent delegation
        is_delegation = "delegate_to_" in str(tool_call)
        call_type = "Agent Delegation" if is_delegation else "Tool Call"
        
        print(f"  {call_type}: {tool_call}")
        print("  Thought: " + str(thought))
        print("  Evaluation: " + str(label) + " (Score: " + str(score) + ")")
        print("  Explanation: " + str(explanation))
        print()
    
    print("\nQuery-to-Thought Evaluation:")
    for i in range(len(query_thought_classifications)):
        print("Example " + str(i+1) + ":")
        
        # Safely get data
        query = ""
        thought = ""
        tool_call = ""
        if i < len(input_data):
            query = input_data.iloc[i]['original_question']
            if isinstance(query, str) and len(query) > 100:
                query = query[:100] + "..."
            thought = input_data.iloc[i]['thought']
            if isinstance(thought, str) and len(thought) > 100:
                thought = thought[:100] + "..."
            tool_call = input_data.iloc[i]['tool_call']
        
        # Get evaluation data
        label = ""
        score = ""
        explanation = ""
        if i < len(query_thought_classifications):
            label = query_thought_classifications.iloc[i]['label']
            score = query_thought_classifications.iloc[i]['score']
            if 'explanation' in query_thought_classifications.columns:
                explanation = query_thought_classifications.iloc[i]['explanation']
                if isinstance(explanation, str) and len(explanation) > 150:
                    explanation = explanation[:150] + "..."
        
        # Highlight whether this is a tool call or agent delegation
        is_delegation = "delegate_to_" in str(tool_call)
        call_type = "Agent Delegation" if is_delegation else "Tool Call"
        
        print(f"  {call_type}: {tool_call}")
        print("  Query: " + str(query))
        print("  Thought: " + str(thought))
        print("  Evaluation: " + str(label) + " (Score: " + str(score) + ")")
        print("  Explanation: " + str(explanation))
        print()
    
    # Log the evaluations to Phoenix
    px.Client().log_evaluations(
        SpanEvaluations(eval_name="Multi-Agent GitHub Tool Calling Eval (Thought-to-Tool)", dataframe=thought_tool_classifications),
    )
    print("\nThought-to-Tool/Agent evaluation results logged to Phoenix")
    
    px.Client().log_evaluations(
        SpanEvaluations(eval_name="Multi-Agent GitHub Query Interpretation Eval (Query-to-Thought)", dataframe=query_thought_classifications),
    )
    print("Query-to-Thought evaluation results logged to Phoenix")
    
    # Display tool and agent delegation usage breakdown
    tool_usage = eval_df['tool_call'].value_counts()
    print("\nTool and Agent Usage Breakdown:")
    for tool, count in tool_usage.items():
        is_delegation = "delegate_to_" in tool
        call_type = "Agent Delegation" if is_delegation else "Tool Call"
        print(f"  {call_type}: {tool} - {count} uses")
    
    # Create combined results with both evaluations
    try:
        # Create a more detailed evaluation results dataframe
        combined_results = pd.DataFrame({
            'original_question': input_data['original_question'],
            'thought': input_data['thought'],
            'tool_call': input_data['tool_call'],
            'action_input': input_data['action_input'],
            'is_delegation': input_data['tool_call'].str.startswith('delegate_to_'),
            'thought_tool_eval': thought_tool_classifications['label'],
            'thought_tool_score': thought_tool_classifications['score'],
            'thought_tool_explanation': thought_tool_classifications['explanation'],
            'query_thought_eval': query_thought_classifications['label'],
            'query_thought_score': query_thought_classifications['score'],
            'query_thought_explanation': query_thought_classifications['explanation']
        })
        
        # Save to CSV
        combined_results.to_csv('multi_agent_combined_evaluation_results.csv', index=False)
        print("\nCombined detailed results saved to 'multi_agent_combined_evaluation_results.csv'")
        
        # Calculate combined accuracy (both evaluations correct)
        combined_accuracy = (combined_results['thought_tool_score'] & combined_results['query_thought_score']).mean() * 100
        print(f"\nCombined accuracy (both correct): {combined_accuracy:.2f}%")
        
        # Calculate combined accuracy for agent delegations vs regular tool calls
        if sum(combined_results['is_delegation']) > 0:
            delegation_combined_accuracy = (combined_results[combined_results['is_delegation']]['thought_tool_score'] & 
                                        combined_results[combined_results['is_delegation']]['query_thought_score']).mean() * 100
            print(f"Combined accuracy for Agent Delegations: {delegation_combined_accuracy:.2f}%")
        
        if sum(~combined_results['is_delegation']) > 0:
            tool_combined_accuracy = (combined_results[~combined_results['is_delegation']]['thought_tool_score'] & 
                                    combined_results[~combined_results['is_delegation']]['query_thought_score']).mean() * 100
            print(f"Combined accuracy for Tool Calls: {tool_combined_accuracy:.2f}%")
        
    except Exception as e:
        print("Error creating combined results: " + str(e))
    
    # Add specialized metrics for agent delegation workflows
    delegation_sequences = []
    for _, group in eval_df.groupby('question_text'):
        calls = group['tool_call'].tolist()
        if any('delegate_to_' in call for call in calls):
            delegation_sequences.append(calls)
    
    if delegation_sequences:
        print("\n=== AGENT DELEGATION WORKFLOW ANALYSIS ===")
        print(f"Found {len(delegation_sequences)} workflows with agent delegations")
        
        for i, sequence in enumerate(delegation_sequences):
            print(f"\nWorkflow {i+1}:")
            for j, call in enumerate(sequence):
                is_delegation = "delegate_to_" in call
                call_type = "AGENT DELEGATION" if is_delegation else "Tool Call"
                print(f"  Step {j+1}: {call_type} - {call}")
        
        # Count how many workflows have multiple delegations
        multi_delegation_count = sum(1 for seq in delegation_sequences if sum(1 for call in seq if 'delegate_to_' in call) > 1)
        print(f"\nWorkflows with multiple agent delegations: {multi_delegation_count} ({multi_delegation_count/len(delegation_sequences)*100:.2f}%)")
        
        # Analyze common delegation patterns
        delegation_patterns = {}
        for seq in delegation_sequences:
            delegations = [call for call in seq if 'delegate_to_' in call]
            if len(delegations) > 1:
                pattern = " -> ".join(delegations)
                if pattern in delegation_patterns:
                    delegation_patterns[pattern] += 1
                else:
                    delegation_patterns[pattern] = 1
        
        if delegation_patterns:
            print("\nCommon agent delegation patterns:")
            for pattern, count in sorted(delegation_patterns.items(), key=lambda x: x[1], reverse=True):
                print(f"  {pattern}: {count} occurrences")
    
    print("\n=== EVALUATION COMPLETE ===")
    print(f"Project: {project_name}")
    print(f"Total tool calls and agent delegations evaluated: {len(eval_df)}")
    
    # Print summary metrics
    print("\nSummary Metrics:")
    print(f"  Thought-to-Tool/Agent accuracy: {tt_accuracy:.2f}%")
    print(f"  Query-to-Thought accuracy: {qt_accuracy:.2f}%")
    print(f"  Sequence Optimality accuracy: {seq_accuracy:.2f}%")
    print(f"  Combined accuracy (both correct): {combined_accuracy:.2f}%")
    
    agent_delegations = sum(1 for call in eval_df['tool_call'] if 'delegate_to_' in call)
    tool_calls = len(eval_df) - agent_delegations
    print(f"\nAgent Delegations: {agent_delegations} ({agent_delegations/len(eval_df)*100:.2f}%)")
    print(f"Tool Calls: {tool_calls} ({tool_calls/len(eval_df)*100:.2f}%)")