import asyncio
from master import agent

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
