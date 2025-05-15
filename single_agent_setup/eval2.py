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
    TOOL_CALLING_PROMPT_TEMPLATE,
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

# Function to extract tool calls from the output message content
def extract_tool_from_content(content):
    """Parse ReAct format tool call from content string"""
    if not content:
        return "No tool used"
    
    # For ReAct format - extract tool after "Action:" keyword
    if 'Action:' in content:
        try:
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith('Action:'):
                    tool_name = line.replace('Action:', '').strip()
                    # Remove any markdown backticks
                    tool_name = tool_name.replace('```', '').strip()
                    return tool_name
        except:
            pass
    
    return "No tool used"

# Function to safely extract tool calls from the LLM output messages
def get_tool_call(row):
    try:
        outputs = row['outputs']
        
        # Handle structure where it's a list of message objects
        if isinstance(outputs, list) and len(outputs) > 0:
            message = outputs[0].get('message', {})
            content = message.get('content', '')
            return extract_tool_from_content(content)
            
        # Handle structure where it's a string that needs parsing
        elif isinstance(outputs, str):
            try:
                parsed_outputs = json.loads(outputs)
                if isinstance(parsed_outputs, list) and len(parsed_outputs) > 0:
                    message = parsed_outputs[0].get('message', {})
                    content = message.get('content', '')
                    return extract_tool_from_content(content)
            except:
                pass
        
        # Check the output field if exists
        if 'output' in row:
            output = row['output']
            if isinstance(output, dict) and 'value' in output:
                content = output.get('value', '')
                return extract_tool_from_content(content)
                
        return "No tool used"
    except Exception as e:
        print(f"Error extracting tool call: {str(e)}")
        return "No tool used"

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
trace_df['tool_call'] = trace_df.apply(get_tool_call, axis=1)

# Filter out rows with no tool used and drop duplicates
eval_df = trace_df[trace_df['tool_call'] != "No tool used"].drop_duplicates(subset=['question_text', 'tool_call'])
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
    
    # Create evaluation model
    eval_model = OpenAIModel(model="gpt-4o")
    print("Using GPT-4o for evaluation")

    # Prepare the data for llm_classify with correct variable names
    input_data = pd.DataFrame({
        'question': eval_df['question_text'],  # Use 'question' instead of 'prompt'
        'tool_call': eval_df['tool_call'],
        'tool_definitions': tool_definitions
    })
    
    # Get rails for classification
    rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())

    # Run evaluation
    print("Running evaluation...")
    response_classifications = llm_classify(
        data=input_data,  # Use 'data' parameter name instead of 'dataframe'
        template=TOOL_CALLING_PROMPT_TEMPLATE,
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
        SpanEvaluations(eval_name="GitHub Tool Calling Eval", dataframe=response_classifications),
    )
    print("\nEvaluation results logged to Phoenix under 'GitHub Tool Calling Eval'")
    
    # Display tool usage breakdown
    tool_usage = eval_df['tool_call'].value_counts()
    print("\nTool Usage Breakdown:")
    for tool, count in tool_usage.items():
        print(f"  {tool}: {count} uses")