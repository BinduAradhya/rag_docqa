from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import mlflow
import os

# -- Step 0: Load .env file and OpenAI API Key --
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")

# -- Step 1: Load and split PDF with metadata (page number) --
pdf_path = input("📄 Enter the name of the PDF file to use (e.g., my_resume.pdf): ")
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

# Add page numbers as metadata
for i, doc in enumerate(docs):
    doc.metadata["page"] = i + 1

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"✅ Loaded {len(chunks)} chunks")

# -- Step 2: Build FAISS Vector DB --
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding_model)

# -- Step 3: Load OpenAI GPT-3.5 --
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_key
)

# -- Step 4: Define Prompt Template (to prevent hallucination) --
prompt_template = """You are a helpful assistant. Only use the content from the retrieved documents to answer the question. Do NOT make up any information. If the answer is not in the document, say "The document does not contain this information."

Context: {context}
Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -- Step 5: Build RAG QA chain --
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# -- Step 6: Accept query from user --
query = input("💬 Enter your question: ")
result = qa_chain.invoke({"query": query})
response = result["result"]
sources = result.get("source_documents", [])

print("📌 Answer:", response)

# Optional: Show source page numbers
if sources:
    pages = sorted(set(doc.metadata.get("page") for doc in sources if "page" in doc.metadata))
    print(f"📄 Answer derived from PDF page(s): {pages}")

# -- Step 7: MLflow logging --
mlflow.set_experiment("rag-qa-pipeline")
with mlflow.start_run():
    mlflow.log_param("chunk_size", 500)
    mlflow.log_param("retrieval_k", 5)
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("llm", "gpt-3.5-turbo")
    mlflow.log_param("query", query)
    mlflow.log_text(response, 'output_answer.txt')
