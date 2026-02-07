from langchain_core.prompts import PromptTemplate ,ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import os
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")



PDF_PATH="C:\\Users\\rahul\\Downloads\\langsmith cx\\langsmith-rk\\islr.pdf"
model_groq=ChatGroq(model="llama-3.1-8b-instant",temperature=0.2,api_key=groq_api_key,max_tokens=200)
llm_embedding = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token,
)
print(PDF_PATH)

# doc loading
loader=PyPDFLoader(PDF_PATH)
docs=loader.load()
print("length of docs",len(docs))

# textsplitting and chunking 

splitters=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitters.split_documents(docs)

print("length of chunks",len(chunks))
# Create FAISS only ONCE
faiss = FAISS.from_documents(chunks, llm_embedding)

# Save to disk
faiss.save_local("faiss_index_islr")

FAISS_INDEX_PATH = "C:\\Users\\rahul\\Downloads\\langsmith cx\\langsmith-rk\\faiss_index_islr"
faiss = FAISS.load_local(
    'faiss_index_islr',
    llm_embedding,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded from disk")


# vector store created

retrivers=faiss.as_retriever(search_type="similarity",search_kwargs={"k":4})

print("retrivers created successfully")

# prompt creation
prompts=ChatPromptTemplate.from_messages([
    ("system","Answer only from the provided context. If not found say you don't know."),
    ("human","Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel_chain=RunnableParallel({
    'context': retrivers | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})
parser=StrOutputParser()

chain=parallel_chain | prompts | model_groq | parser

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)



