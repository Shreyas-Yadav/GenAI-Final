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

# Save the raw data to CSV for debugging
eval_df.to_csv('raw_tool_calls.csv', index=False)
print("Raw tool call data saved to 'raw_tool_calls.csv'")

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

    print("Tool definitions created")
    
    # Check if update_file_content appears in tool calls
    if any('update_file_content' in str(tool_call) for tool_call in eval_df['tool_call']):
        print("\nNOTE: 'update_file_content' appears in tool calls")
        print("Using enhanced definition for this tool in the evaluation")
    
    # Create an improved evaluation template
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

    "correct" means:
    - The tool choice directly addresses what the agent said it wanted to do in its thought
    - The parameters in the action input are appropriate for that specific task

    "incorrect" means:
    - The tool doesn't match what the agent said it wanted to do in its thought
    - The parameters are missing important information or contain incorrect values

    [Available GitHub Tools]: {tool_definitions}

    Please provide a clear EXPLANATION of your reasoning first, then end with the LABEL. Focus only on whether the tool choice makes sense given the agent's thought.

    Example response:
    ************
    EXPLANATION: Looking at the agent's thought, it wants to [what the agent is trying to do]. The tool it selected is [tool name], which [analysis of why this tool is right or wrong for this specific purpose]. The parameters it provided [analysis of whether parameters match the intent].

    Based on this analysis, the tool call is...

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
    
    # Print sample evaluation input for debugging
    print("\nSample evaluation input:")
    for i, row in input_data.head().iterrows():
        print(f"Row {i}:")
        print(f"  Original question: {row['original_question'][:50]}...")
        print(f"  Thought: {row['thought'][:100]}...")
        print(f"  Tool: {row['tool_call']}")
        print(f"  Action Input: {row['action_input'][:50]}...")
        print()
    
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
        SpanEvaluations(eval_name="GitHub Tool Calling Eval (Improved)", dataframe=response_classifications),
    )
    print("\nEvaluation results logged to Phoenix under 'GitHub Tool Calling Eval (Improved)'")
    
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