from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ['LANGCHAIN_PROJECT']="rahul-langsmith"

model=ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=groq_api_key)

prompt=PromptTemplate.from_template("{question}")
parser=StrOutputParser()

chain=prompt | model | parser
CONFIG={
    'run_name': 'simple-llm',
    'tags': ['simple', 'llm'],
    'metadata': {'model': 'llama-3.1-8b-instant'}
}

result=chain.invoke({"question":"what is recipe of biryani "}, config=CONFIG)

print(result)