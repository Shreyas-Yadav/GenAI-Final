# -*- coding: utf-8 -*-
"""
evaluate_github_agent.py

Evaluation script for the GitHub ReActAgent and tools, adapted from evaluate_tool_calling.py.
"""

import os
import nest_asyncio
import pandas as pd
from dotenv import load_dotenv

# Phoenix and LlamaIndex imports
from phoenix.evals import (
    TOOL_CALLING_PROMPT_RAILS_MAP,
    TOOL_CALLING_PROMPT_TEMPLATE,
    OpenAIModel,
    llm_classify,
)
import phoenix as px
from phoenix.otel import register

# Import the agent from master.py
from master import agent

nest_asyncio.apply()

# Load environment variables from .env and set up Phoenix for local usage
load_dotenv()
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
os.environ["PHOENIX_PROJECT_NAME"] = "GitHub Agent Tool Calling Eval"

tracer_provider = register(auto_instrument=True, project_name="GitHub Agent Tool Calling Eval")

# 1. Generate or load evaluation questions

# GITHUB_GEN_TEMPLATE = """
# You are an assistant that generates complex GitHub usage questions. Every question must specifically reference the repository 'puranjaigarg783/project-mobley' (use this exact repository name in each question). The questions should require the use of GitHub API functions such as listing issues, reading files, creating issues, listing pull requests, searching within the repository, etc.
#
# Include:
# - Multi-step requests (e.g., "Find the most recent commit in puranjaigarg783/project-mobley and open an issue about it")
# - Vague or ambiguous queries (e.g., "Show me what's new in puranjaigarg783/project-mobley")
# - Mixed intentions (e.g., "Can you create a file and also list open issues in puranjaigarg783/project-mobley?")
# - Indirect language (e.g., "I was wondering if you could help me with the puranjaigarg783/project-mobley repository?")
# - Some straightforward tool uses
#
# Respond with a list, one question per line. Do not include any numbering at the beginning of each line. Do not include any category headings.
# Generate 20 questions.
# """

# eval_model = OpenAIModel(model="gpt-4o", max_tokens=1300)
# resp = eval_model(GITHUB_GEN_TEMPLATE)
# split_response = resp.strip().split("\n")
# questions_df = pd.DataFrame(split_response, columns=["questions"])
# print(questions_df)

# Hardcoded questions for fast testing
hardcoded_questions = [
    "List all open issues in puranjaigarg783/project-mobley.",
    "Create a new issue in puranjaigarg783/project-mobley titled 'Test Issue' with the body 'This is a test.'",
    "Show me the most recent commit SHA in puranjaigarg783/project-mobley.",
    "List all files in the main branch of puranjaigarg783/project-mobley.",
    "Can you search for issues mentioning 'bug' in puranjaigarg783/project-mobley?"
]
questions_df = pd.DataFrame(hardcoded_questions, columns=["questions"])
print(questions_df)

# Define eval_model for use in evaluation
eval_model = OpenAIModel(model="gpt-4o", max_tokens=1300)

# 2. Run the agent on each question and collect responses
def run_agent_on_question(question):
    try:
        return agent.chat(question)
    except Exception as e:
        return f"Error: {e}"

questions_df["response"] = questions_df["questions"].apply(run_agent_on_question)

# 3. Collect traces from Phoenix
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery

query = (
    SpanQuery()
    .where("span_kind == 'LLM'")
    .select(
        question="llm.input_messages",
        outputs="llm.output_messages",
    )
)
trace_df = px.Client().query_spans(query, project_name="GitHub Agent Tool Calling Eval")

# Debug: Print trace_df columns and head to diagnose missing template variables
print("trace_df columns:", trace_df.columns)
print("trace_df head:\n", trace_df.head())

# 4. Extract tool call information
def get_tool_call(outputs):
    try:
        if outputs[0].get("message").get("tool_calls"):
            return (
                outputs[0]
                .get("message")
                .get("tool_calls")[0]
                .get("tool_call")
                .get("function")
                .get("name")
            )
        else:
            return "No tool used"
    except Exception:
        return "No tool used"

trace_df["tool_call"] = trace_df["outputs"].apply(get_tool_call)

# 5. Build tool definitions string for the evaluator
from master import tools as github_tools_list

tool_definitions = ""
for tool in github_tools_list:
    tool_definitions += f"""
    {tool.fn.__name__}: {tool.fn.__doc__}
    """

trace_df["tool_definitions"] = tool_definitions

# 6. Evaluate tool calls
rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())

print("Rows with missing question:", trace_df['question'].isnull().sum())
print(trace_df[trace_df['question'].isnull()])
response_classifications = llm_classify(
    dataframe=trace_df,
    template=TOOL_CALLING_PROMPT_TEMPLATE,
    model=eval_model,
    rails=rails,
    provide_explanation=True,
)
response_classifications["score"] = response_classifications.apply(
    lambda x: 1 if x["label"] == "correct" else 0, axis=1
)

print(response_classifications)

# 7. Log evaluations to Phoenix
px.Client().log_evaluations(
    SpanEvaluations(eval_name="GitHub Agent Tool Calling Eval", dataframe=response_classifications),
)

print("Evaluation complete. Results logged to Phoenix.")