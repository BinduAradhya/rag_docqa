# 🧠 RAG-based Document QA Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for answering questions from any uploaded PDF using **LangChain**, **OpenAI GPT-3.5**, and **FAISS** for vector-based semantic search. The goal is to make large documents searchable with natural language queries.

---

## 🚀 Features

- 🔍 **PDF Ingestion**: Upload any PDF and automatically split it into meaningful text chunks.
- 🧬 **Vector Search (FAISS)**: Uses HuggingFace embeddings + FAISS to index chunks for fast, semantic retrieval.
- 🤖 **OpenAI GPT-3.5 Integration**: Generates accurate and fluent answers using retrieved chunks.
- 🧠 **Metadata Filtering**: Reduces hallucination by grounding answers only in the retrieved context.
- 📈 **Experiment Tracking**: Integrated with **MLflow** to log parameters and results.
- 🐳 **Dockerized**: Fully containerized pipeline for easy reproducibility.

---

## 🛠️ Tech Stack

- `LangChain`
- `OpenAI GPT-3.5`
- `FAISS`
- `HuggingFace Embeddings`
- `MLflow`
- `Docker`
- `PyMuPDF` (PDF Loader)

---

## 📦 Setup Instructions

### 🔧 1. Clone the repo

```bash
git clone https://github.com/BinduAradhya/rag_docqa.git
cd rag_docqa
```

### 🐳 2. Build Docker Image

```bash
docker build -t rag-qa .
```

### 📂 3. Run the Pipeline

```bash
docker run -it -v "%cd%":/app rag-qa
```

> You'll be prompted to:
> - Enter the name of the PDF file (must be in the same folder)
> - Ask any question related to the content

---

## 📄 Example

```bash
📄 Enter the name of the PDF file to use (e.g., my_resume.pdf): Shivam_DA_QuestionBank.pdf
✅ Loaded 17 chunks
💬 Enter your question: What are the main steps in data pre-processing?
📌 Answer: ...
📄 Answer derived from PDF page(s): [3, 4]
```

---

## 🧪 MLflow Logs

- All runs are logged to the `mlruns/` folder.
- You can view results using:

```bash
mlflow ui
```

---

## ❗ .env Configuration

Create a `.env` file with your OpenAI key:

```
OPENAI_API_KEY=your_key_here
```

> This file is excluded from version control using `.gitignore`.

---

## 🧠 Future Improvements

- Support for multiple documents
- Web UI with file upload
- Open-source LLM alternatives (e.g., Mistral, LLaMA-3)


---

## 🙋‍♀️ Author

**Bindu Aradhya**  
[LinkedIn](https://www.linkedin.com/in/binduaradhya)
