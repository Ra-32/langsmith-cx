from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(model="llama-3.1-8b-instant",temperature=0.2,api_key=groq_api_key,max_tokens=200)

prompt1=PromptTemplate(
    template="Genrate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Genrate a 5 point summery on detailed report {reports}",
    input_variables=['reports']
)
parser=StrOutputParser()

CONFIG={
    'run_name':'sequential_chain',
    'tags':['sequential-chain','report genration'],
    'metadata':{
        'model':'llama-3.1-8b-instant',
        'temperature':0.2
    }
}

chain=prompt1 | model | parser | prompt2 | model | parser

response=chain.invoke({'topic':'india vs pakistan cricket match'},config=CONFIG)

print(response)