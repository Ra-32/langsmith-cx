from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_classic.agents import AgentExecutor,create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import requests
from langchain_classic import hub
import langchainhub
load_dotenv()

import os
groq_api_key=os.getenv("GROQ_API_KEY")
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
os.environ['LANGCHAIN_PROJECT']="Agent-rk Project"

model=ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-genration",
    huggingfacehub_api_token=hf_token
)
hf_model=ChatHuggingFace(llm=llm)


@tool
def weather_tool(city :str)->str:
    """
    These Function give the waether data of the city
    """
    url=f"https://api.weatherstack.com/current?access_key=ec29ba055b3212c8dfe073390e8ad541&query={city}"

    response = requests.get(url)
    return str(response.json())




search_tool=DuckDuckGoSearchRun()



prompt=hub.pull("hwchase17/react")

agent=create_react_agent(
    llm=hf_model,
    tools=[search_tool,weather_tool],
    prompt=prompt
)


agent_executor=AgentExecutor(
    agent=agent,    
    tools=[search_tool,weather_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=7

    
    
)

result=agent_executor.invoke({"input":"What is the waether condition  of Mumbai? and give me the information about Rahul dravid as well"})

print(result['output'])



