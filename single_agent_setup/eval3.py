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
project_name = "default"
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

# Function to extract the thought, tool call, and action input from the output message content
def extract_react_components(content):
    """Parse ReAct format to extract thought, tool call, and action input"""
    if not content:
        return {"thought": "", "tool_call": "No tool used", "action_input": ""}
    
    components = {
        "thought": "",
        "tool_call": "No tool used",
        "action_input": ""
    }
    
    try:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            # Extract thought
            if line.startswith('Thought:'):
                thought_lines = []
                j = i
                while j < len(lines) and not (lines[j].strip().startswith('Action:') or lines[j].strip() == ""):
                    if j > i:  # Skip the "Thought:" prefix line
                        thought_lines.append(lines[j].strip())
                    j += 1
                components["thought"] = ' '.join(thought_lines)
            
            # Extract tool call
            elif line.startswith('Action:'):
                components["tool_call"] = line.replace('Action:', '').strip()
            
            # Extract action input
            elif line.startswith('Action Input:'):
                action_input_lines = []
                j = i
                while j < len(lines) and j+1 < len(lines) and not lines[j+1].strip().startswith('Observation:'):
                    if j > i:  # Skip the "Action Input:" prefix line
                        action_input_lines.append(lines[j].strip())
                    j += 1
                components["action_input"] = ' '.join(action_input_lines)
    except Exception as e:
        print(f"Error extracting React components: {e}")
    
    return components

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

# Filter out rows with no tool used and drop duplicates
eval_df = trace_df[trace_df['tool_call'] != "No tool used"].drop_duplicates(subset=['thought', 'tool_call'])
print(f"Filtered to {len(eval_df)} valid tool call examples")

if len(eval_df) == 0:
    print("No valid tool calls found in the spans. Make sure your agent has run with tracing enabled.")
else:
    # Import tool definitions
    from github_tools import (
        list_my_repos, read_file, commit_file, list_issues, open_issue, 
        list_prs, create_pr, list_commits, search_repos, search_issues,
        create_repo, close_issue, list_repo_files
    )

    # Define all tools
    tools = [
        list_my_repos, read_file, commit_file, list_issues, open_issue, 
        list_prs, create_pr, list_commits, search_repos, search_issues,
        create_repo, close_issue, list_repo_files
    ]

    # Create tool definitions
    tool_definitions = ""
    for tool in tools:
        tool_name = tool.__name__
        tool_desc = tool.__doc__ or "No description available"
        tool_definitions += f"\n{tool_name}: {tool_desc}\n"

    print("Tool definitions created")
    
    # Create custom evaluation template that uses the thought instead of the original question
    TOOL_CALLING_WITH_THOUGHT_TEMPLATE = """
    You are an evaluation assistant evaluating thoughts, tool calls, and action inputs to
    determine whether the tool called would address the task described in the thought.
    The tool calls have been generated by a separate agent, and chosen from the list of
    tools provided below. It is your job to decide whether that agent chose the
    right tool to call based on what it was trying to accomplish.

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

    Your response must be a single word, either "correct" or "incorrect".
    "incorrect" means that the chosen tool would not accomplish the task described in the thought,
    the tool includes information that is not presented in the thought,
    or that the tool signature includes parameter values that don't match
    the formats specified in the tool signatures below.

    "correct" means the correct tool call was chosen based on what the agent was trying to do
    in its thought, the correct parameters were extracted, the tool call generated is runnable
    and correct, and that no outside information not present in the thought was used
    in the generated tool call.

    [Tool Definitions]: {tool_definitions}

    Please read the original question, thought, tool call, and action input carefully, then write out
    in a step by step manner an EXPLANATION to show how to determine if the
    answer is "correct" or "incorrect". Avoid simply stating the correct answer
    at the outset. Your response LABEL must be a single word, either "correct"
    or "incorrect", and should not contain any text or characters aside from
    that word.

    Example response:
    ************
    EXPLANATION: An explanation of your reasoning for why the label is "correct" or "incorrect"
    LABEL: "correct" or "incorrect"
    ************

    EXPLANATION:
    """
    
    # Create evaluation model
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
    
    # Get rails for classification
    rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())

    # Run evaluation
    print("Running evaluation...")
    from phoenix.evals import ClassificationTemplate
    
    # Create a custom template with our thought-based prompt
    tool_thought_template = ClassificationTemplate(
        rails=rails,
        template=TOOL_CALLING_WITH_THOUGHT_TEMPLATE,
        explanation_template=TOOL_CALLING_WITH_THOUGHT_TEMPLATE,
        scores=[1, 0]
    )
    
    response_classifications = llm_classify(
        data=input_data,
        template=tool_thought_template,
        model=eval_model,
        rails=rails,
        provide_explanation=True,
    )
    
    # Add score column
    response_classifications["score"] = response_classifications.apply(
        lambda x: 1 if x["label"] == "correct" else 0, axis=1
    )
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(response_classifications)
    
    # Calculate overall accuracy
    accuracy = response_classifications["score"].mean() * 100
    print(f"\nOverall accuracy: {accuracy:.2f}%")
    
    # Log the evaluations to Phoenix
    px.Client().log_evaluations(
        SpanEvaluations(eval_name="GitHub Tool Calling Eval (with Thought)", dataframe=response_classifications),
    )
    print("\nEvaluation results logged to Phoenix under 'GitHub Tool Calling Eval (with Thought)'")
    
    # Display tool usage breakdown
    tool_usage = eval_df['tool_call'].value_counts()
    print("\nTool Usage Breakdown:")
    for tool, count in tool_usage.items():
        print(f"  {tool}: {count} uses")