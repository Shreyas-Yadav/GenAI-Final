import os
import logging
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Ensure we have the necessary API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")

# Phoenix imports
import phoenix as px
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from phoenix.otel import register
from phoenix.evals import (
    TOOL_CALLING_PROMPT_RAILS_MAP,
    TOOL_CALLING_PROMPT_TEMPLATE,
    OpenAIModel,
    llm_classify,
)

# Import the tool definitions from github_tools
from github_tools import (
    list_my_repos, read_file, commit_file, list_issues, open_issue, 
    list_prs, create_pr, list_commits, search_repos, search_issues, 
    create_repo, close_issue, tools
)

# Set up Phoenix project
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
project_name = "default"
os.environ["PHOENIX_PROJECT_NAME"] = project_name

# Register tracer provider
tracer_provider = register(auto_instrument=True, project_name=project_name)

def get_tool_definitions():
    """Generate tool definitions string for the evaluator"""
    tool_definitions = ""
    for tool in tools:
        tool_name = tool.__name__
        tool_desc = tool.__doc__ or "No description available"
        tool_definitions += f"\n{tool_name}: {tool_desc}\n"
    return tool_definitions

def get_tool_call(outputs):
    """Extract the tool call name from the LLM output"""
    try:
        # Check for OpenAI function call format
        if isinstance(outputs, list) and outputs and "function_call" in outputs[0].get("message", {}):
            tool_name = outputs[0]["message"]["function_call"]["name"]
            return tool_name
        
        # Check for OpenAI tool calls format
        elif isinstance(outputs, list) and outputs and "tool_calls" in outputs[0].get("message", {}):
            tool_name = outputs[0]["message"]["tool_calls"][0]["function"]["name"]
            return tool_name
        
        # Check for GitHub agent format - look for "Action:" in the content
        elif isinstance(outputs, list) and outputs and "content" in outputs[0].get("message", {}):
            content = outputs[0]["message"]["content"]
            if isinstance(content, str) and "Action:" in content:
                # Extract the tool name after "Action:"
                action_line = [line for line in content.split('\n') if "Action:" in line]
                if action_line:
                    tool_name = action_line[0].split("Action:")[1].strip()
                    return tool_name
        
        # Check for tool_code format (used in the GitHub agent)
        elif isinstance(outputs, list) and outputs and "content" in outputs[0].get("message", {}):
            content = outputs[0]["message"]["content"]
            if isinstance(content, str) and "```tool_code" in content:
                # Extract the tool name after "Action:"
                action_lines = content.split("```tool_code")[1].split("```")[0].strip().split("\n")
                for line in action_lines:
                    if line.startswith("Action:"):
                        tool_name = line.split("Action:")[1].strip()
                        return tool_name
        
        # Default case
        return "No tool used"
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error extracting tool call: {e}")
        return "No tool used"

def extract_question_text(input_messages):
    """Extract the plain text question from the input messages"""
    try:
        # Handle GitHub agent specific format - look for "Step input:" which contains the actual question
        if isinstance(input_messages, str) and "Step input:" in input_messages:
            # Extract the text after "Step input:"
            step_input_parts = input_messages.split("Step input:")
            if len(step_input_parts) > 1:
                # Get the text after "Step input:" and before the next section
                question = step_input_parts[1].split("\n")[0].strip()
                if question == "None":
                    # Look for previous step input if this one is None
                    for part in step_input_parts[:-1]:
                        lines = part.split("\n")
                        for line in reversed(lines):
                            if line.strip() and not line.strip().startswith(">"):
                                return line.strip()
                return question
        
        # Standard OpenAI format handling
        if isinstance(input_messages, list) and input_messages:
            # Get the last user message content
            for message in reversed(input_messages):
                if message.get("role", "") == "user" and "content" in message:
                    return message["content"]
            
            # Fallback to the first message content if no user role found
            if "content" in input_messages[0].get("message", {}):
                content = input_messages[0]["message"]["content"]
                return content
            
            # Another fallback if we have a dict with 'message' key
            if isinstance(input_messages[0], dict) and "message" in input_messages[0]:
                content = input_messages[0]["message"].get("content", str(input_messages[0]))
                return content
        
        # If all else fails, convert the whole thing to a string
        return str(input_messages)
    except Exception as e:
        logger.error(f"Error extracting question text: {e}")
        return str(input_messages)

def evaluate_tool_calling():
    """
    Evaluate the tool calling performance of the GitHub agent
    
    This function:
    1. Queries traces directly from Phoenix
    2. Extracts questions and tool calls from the traces
    3. Evaluates whether the correct tool was called
    4. Logs evaluation results back to Phoenix
    """
    # Make project_name accessible within this function
    global project_name
    
    logger.info("Starting tool calling evaluation...")
    
    # Query traces from Phoenix
    try:
        # Define the query to get LLM spans
        query = (
            SpanQuery()
            .where(
                # Filter for the `LLM` span kind
                "span_kind == 'LLM'"
            )
            .select(
                # Extract and rename the following span attributes
                question="llm.input_messages",
                outputs="llm.output_messages"
            )
        )
        
        # Execute the query
        client = px.Client()
        trace_df = client.query_spans(query, project_name=project_name)
        
        if trace_df.empty:
            logger.warning("No traces found in Phoenix. Make sure you've run the GitHub agent with tracing enabled.")
            return
        
        logger.info(f"Found {len(trace_df)} traces to evaluate.")
        
        # Extract tool calls from outputs
        trace_df["tool_call"] = trace_df["outputs"].apply(get_tool_call)
        
        # Extract questions from inputs
        trace_df["question"] = trace_df["question"].apply(extract_question_text)
        
    except Exception as e:
        logger.error(f"Error querying traces from Phoenix: {e}")
        logger.error("Make sure Phoenix is running and has collected traces from the GitHub agent.")
        return
    
    # Add tool definitions to each row
    tool_definitions = get_tool_definitions()
    trace_df["tool_definitions"] = tool_definitions
    
    # Create a clean dataframe for evaluation with the correct format
    eval_df = pd.DataFrame({
        'question': trace_df['question'],
        'tool_call': trace_df['tool_call'],
        'tool_definitions': trace_df['tool_definitions']
    })
    
    # Set up evaluator model
    eval_model = OpenAIModel(model="gpt-4o", temperature=0.0, max_tokens=1024)
    rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())
    
    logger.info("Running tool calling evaluation...")
    # Run evaluation
    try:
        eval_results = llm_classify(
            data=eval_df,  # Using 'data' parameter (not deprecated 'dataframe')
            template=TOOL_CALLING_PROMPT_TEMPLATE,
            model=eval_model,
            rails=rails,
            provide_explanation=True,
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        # Create empty results if evaluation fails
        eval_results = pd.DataFrame(columns=["label", "explanation", "score"])
    
    # Convert classification results to scores
    eval_results["score"] = eval_results.apply(
        lambda x: 1 if x["label"] == "correct" else 0, axis=1
    )
    
    # Calculate overall score
    if not eval_results.empty:
        total_score = eval_results["score"].mean() * 100
        logger.info(f"Overall tool calling accuracy: {total_score:.2f}%")
    else:
        logger.warning("No evaluation results to calculate score.")
    
    # Log evaluation results to Phoenix
    try:
        # Check if we have valid evaluation results
        if eval_results.empty:
            logger.warning("No evaluation results to log to Phoenix. Skipping.")
            return eval_results
            
        # Create Phoenix client
        client = px.Client()
        
        # Verify Phoenix client configuration
        phoenix_endpoint = os.environ.get('PHOENIX_COLLECTOR_ENDPOINT')
        if not phoenix_endpoint:
            logger.warning("PHOENIX_COLLECTOR_ENDPOINT environment variable not set!")
        
        # Get project name from environment variable
        project_name = os.environ.get('PHOENIX_PROJECT_NAME', 'default')
        
        # Fix the index for Phoenix
        # Phoenix requires the index to be 'context.span_id'
        phoenix_results = eval_results.copy()
        phoenix_results.index = [f"span-{i}" for i in range(len(phoenix_results))]
        phoenix_results.index.name = "context.span_id"
        
        # Log evaluation results to Phoenix
        # Use a more distinctive evaluation name
        eval_name = "GitHub-Agent-Tool-Calling-Eval"
        
        print(phoenix_results)
        try:
            response = client.log_evaluations(
                SpanEvaluations(eval_name=eval_name, dataframe=phoenix_results),
            )
        except Exception as e:
            logger.error(f"Exception during log_evaluations call: {e}")
            raise
    except Exception as e:
        logger.error(f"Error logging evaluation results to Phoenix: {e}")
    
    # Provide guidance on finding evaluations in the Phoenix UI
    logger.info("Evaluation complete! To view results in the Phoenix UI: http://localhost:6006")
    
    return eval_results

if __name__ == "__main__":
    evaluate_tool_calling()