import os
import asyncio
from single_agent_setup.master import agent
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openrouter import OpenRouter
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
# from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
# from phoenix.otel import register



# os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
# tracer_provider = register()
# LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


async def interactive_chat():
    print("GitHub‑Agent ready. Type ‘exit’ to quit.")
    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() in ("exit","quit"):
            break
        # send as ChatMessage so OpenRouter knows it’s a user turn
        response = agent.chat(
           user_input
        )
        print("🤖 Agent:", response)

if __name__ == "__main__":
    asyncio.run(interactive_chat())
