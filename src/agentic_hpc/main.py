# External modules
from openai import OpenAI
from agent import Agent

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

my_agent = Agent(client)

answer = my_agent.run_agent(
    "Tell me which CPU this machine uses, how many cores it has and details on cache memory."
)

print(answer)