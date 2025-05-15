

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qq "arize-phoenix>=8.8.0" "arize-phoenix-otel>=0.8.0" llama-index-llms-openai openai gcsfs nest_asyncio langchain langchain-openai openinference-instrumentation-langchain

import os
from getpass import getpass

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

"""### Connect to Phoenix

We'll also enable tracing with Phoenix using our Langchain auto-instrumentor to capture telemetry that we can later evaluate.

This code will connect you to an online version of Phoenix, at app.phoenix.arize.com. If you're self-hosting Phoenix, be sure to change your Collector Endpoint below, and remove the API Key.
"""
# Set up Phoenix project
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
project_name = "default"
os.environ["PHOENIX_PROJECT_NAME"] = project_name

tracer_provider = register(auto_instrument=True, project_name="default")


"""# Evaluate Tool Calls

Now that we have some example runs of our agent to analyze, we can start the evaluation process. We'll start by exporting all of those spans from Phoenix
"""

from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery

"""Since we'll only be evaluating the inputs, outputs, and function call columns, let's extract those into an easier to use df. Helpfully, Phoenix provides a method to query your span data and directly export only the values you care about."""

query = (
    SpanQuery()
    .where(
        # Filter for the `LLM` span kind.
        # The filter condition is a string of valid Python boolean expression.
        "span_kind == 'LLM'",
    )
    .select(
        # Extract and rename the following span attributes
        question="llm.input_messages",
        outputs="llm.output_messages",
    )
)


trace_df = px.Client().query_spans(query, project_name="default")
# trace_df["tool_call"] = trace_df["tool_call"].fillna("No tool used")

def get_tool_call(outputs):
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





trace_df["tool_call"] = trace_df["outputs"].apply(get_tool_call)

"""We'll also need to pass in our tool definitions to the evaluator:"""

tool_definitions = ""

from github_tools import (
    list_my_repos, read_file, commit_file, list_issues, open_issue, 
    list_prs, create_pr, list_commits, search_repos, search_issues
)


tools = [
    list_my_repos, read_file, commit_file, list_issues, open_issue,
    list_prs, create_pr, list_commits, search_repos, search_issues
]


tool_definitions = ""
for tool in tools:
    tool_name = tool.__name__
    tool_desc = tool.__doc__ or "No description available"
    tool_definitions += f"\n{tool_name}: {tool_desc}\n"

print(tool_definitions)

"""Next, we define the evaluator model to use"""

trace_df["tool_definitions"] = tool_definitions

eval_model = OpenAIModel(model="gpt-4o")

"""And we're ready to call our evaluator! The method below takes in the dataframe of traces to evaluate, our built in evaluation prompt, the eval model to use, and a rails object to snap responses from our model to a set of binary classification responses.

We'll also instruct our model to provide explanations for its responses.
"""

rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())

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

response_classifications

"""Finally, we'll export these responses back into Phoenix to view them in the UI."""

px.Client().log_evaluations(
    SpanEvaluations(eval_name="default", dataframe=response_classifications),
)

"""From here, you could iterate on different agent logic and prompts to improve performance, or you could further decompose the evaluation into individual steps looking first at Routing, then Parameter Extraction, then Code Generation to determine where to focus.

![Tool Calling Evaluation Results](https://storage.googleapis.com/arize-phoenix-assets/assets/images/tool-calling-nb-result.png)

Happy building!
"""