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

# Save the raw data to CSV for debugging
eval_df.to_csv('raw_tool_calls.csv', index=False)
print("Raw tool call data saved to 'raw_tool_calls.csv'")

if len(eval_df) == 0:
    print("No valid tool calls found in the spans. Make sure your agent has run with tracing enabled.")
else:
    # Import tool definitions
    from github_tools import (
        list_my_repos, read_file, commit_file, list_issues, open_issue, 
        list_prs, create_pr, list_commits, search_repos, search_issues,
        create_repo, close_issue, list_repo_files 
    )

    # Define all tools - MAKE SURE TO ADD update_file_content
    tools = [
        list_my_repos, read_file, commit_file, list_issues, open_issue, 
        list_prs, create_pr, list_commits, search_repos, search_issues,
        create_repo, close_issue, list_repo_files 
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

    print("Tool definitions created")
    
    # Create an improved evaluation template that's more tolerant of tool choice reasoning
    IMPROVED_TOOL_CALLING_TEMPLATE = """
    You are an evaluation assistant looking at a conversational GitHub agent. You're analyzing whether the agent chose the right GitHub tool for what it was trying to accomplish in a specific moment.
    
    The agent follows a ReAct format with "Thought" followed by "Action" and "Action Input". Your job is to determine if the tool choice (Action) makes sense given what the agent was thinking about doing.
    
    NOTE: The agent works on ONE SPECIFIC SUBTASK at a time. DO NOT evaluate whether the tool addresses the ENTIRE original question, only whether it addresses the SPECIFIC SUBTASK described in the thought.

        [BEGIN DATA]
        ************
        [Original Question]: {original_question}
        ************
        [Agent's Current Thought]: {thought}
        ************
        [Tool Called]: {tool_call}
        ************
        [Tool Action Input]: {action_input}
        [END DATA]

    First, identify the specific subtask the agent is focusing on in its current thought. Then decide if the chosen tool is appropriate for that specific subtask (not the entire original question).

    Your response must be a single word, either "correct" or "incorrect".
    
    "correct" means the tool call would accomplish the specific subtask the agent is focused on in its current thought.
    
    "incorrect" means the tool would not accomplish the current subtask, or the parameters don't make sense for what the agent is trying to do in this moment.

    [Available GitHub Tools]: {tool_definitions}

    Please read the agent's current thought carefully, then write out in a step by step manner an EXPLANATION focusing only on whether the tool call matches what the agent was trying to accomplish in this specific step. Your response LABEL must be a single word, either "correct" or "incorrect".

    Example response:
    ************
    EXPLANATION: First, I'll identify what subtask the agent is working on right now based on its thought. Then I'll check if the tool call is appropriate for that subtask.
    
    In this case, the agent is currently thinking about [describe specific subtask from thought]. The agent chose [tool name], which [is/isn't] appropriate for this subtask because [reasoning].
    
    Therefore, the tool call is...
    
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
    
    # Save the input data to CSV for debugging
    input_data.to_csv('eval_input_data.csv', index=False)
    print("Evaluation input data saved to 'eval_input_data.csv'")
    
    # Get rails for classification
    rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())

    # Run evaluation
    print("Running evaluation...")
    from phoenix.evals import ClassificationTemplate
    
    # Create a custom template with our improved prompt
    improved_template = ClassificationTemplate(
        rails=rails,
        template=IMPROVED_TOOL_CALLING_TEMPLATE,
        explanation_template=IMPROVED_TOOL_CALLING_TEMPLATE,
        scores=[1, 0]
    )
    
    response_classifications = llm_classify(
        data=input_data,
        template=improved_template,
        model=eval_model,
        rails=rails,
        provide_explanation=True,
    )
    
    # Add score column
    response_classifications["score"] = response_classifications.apply(
        lambda x: 1 if x["label"] == "correct" else 0, axis=1
    )
    
    # Save the evaluation results to CSV
    response_classifications.to_csv('tool_call_evaluations.csv', index=False)
    print("Evaluation results saved to 'tool_call_evaluations.csv'")
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(response_classifications)
    
    # Calculate overall accuracy
    accuracy = response_classifications["score"].mean() * 100
    print(f"\nOverall accuracy: {accuracy:.2f}%")
    
    # Display simplified evaluation results
    print("\nDetailed Evaluation Results:")
    for i in range(len(response_classifications)):
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
        if i < len(response_classifications):
            label = response_classifications.iloc[i]['label']
            score = response_classifications.iloc[i]['score']
            if 'explanation' in response_classifications.columns:
                explanation = response_classifications.iloc[i]['explanation']
                if isinstance(explanation, str) and len(explanation) > 150:
                    explanation = explanation[:150] + "..."
        
        print("  Thought: " + str(thought))
        print("  Tool: " + str(tool_call))
        print("  Evaluation: " + str(label) + " (Score: " + str(score) + ")")
        print("  Explanation: " + str(explanation))
        print()
    
    # Log the evaluations to Phoenix
    px.Client().log_evaluations(
        SpanEvaluations(eval_name="GitHub Tool Calling Eval (Subtask Focus)", dataframe=response_classifications),
    )
    print("\nEvaluation results logged to Phoenix under 'GitHub Tool Calling Eval (Subtask Focus)'")
    
    # Display tool usage breakdown
    tool_usage = eval_df['tool_call'].value_counts()
    print("\nTool Usage Breakdown:")
    for tool, count in tool_usage.items():
        print("  " + str(tool) + ": " + str(count) + " uses")
    
    # Save a combined results file with everything for easier analysis
    try:
        # Create a more detailed evaluation results dataframe
        combined_results = pd.DataFrame({
            'original_question': input_data['original_question'],
            'thought': input_data['thought'],
            'tool_call': input_data['tool_call'],
            'action_input': input_data['action_input'],
            'evaluation': response_classifications['label'],
            'score': response_classifications['score'],
            'explanation': response_classifications['explanation']
        })
        
        # Save to CSV
        combined_results.to_csv('combined_evaluation_results.csv', index=False)
        print("\nCombined detailed results saved to 'combined_evaluation_results.csv'")
    except Exception as e:
        print("Error creating combined results: " + str(e))