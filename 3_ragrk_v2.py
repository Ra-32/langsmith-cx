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
from langsmith import traceable  # <-- key import
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


print("groq",groq_api_key)
print("hf-",hf_token)
PDF_PATH="C:\\Users\\rahul\\Downloads\\langsmith cx\\langsmith-rk\\islr.pdf"
model_groq=ChatGroq(model="llama-3.1-8b-instant",temperature=0.2,api_key=groq_api_key,max_tokens=200)
llm_embedding = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token,
)
print(PDF_PATH)

# doc loading
@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # list[Document]

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emb = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token,
)
    # FAISS.from_documents internally calls the embedding model:
    vs = FAISS.from_documents(splits, emb)
    return vs

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs


vectorstore = setup_pipeline(PDF_PATH)
# textsplitting and chunking  Create FAISS only ONCE
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# Save to disk
print("retrivers created successfully")

# prompt creation
prompts=ChatPromptTemplate.from_messages([
    ("system","Answer only from the provided context. If not found say you don't know."),
    ("human","Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel_chain=RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})
parser=StrOutputParser()

chain=parallel_chain | prompts | model_groq | parser

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
config = {
    "run_name": "pdf_rag_query"
}
ans = chain.invoke(q.strip(), config=config)
print("\nA:", ans)



