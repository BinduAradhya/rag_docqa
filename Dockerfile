# Use official slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Optional: install additional packages for langchain-community if needed
# RUN pip install --no-cache-dir langchain-openai langchain-huggingface

# Copy rest of the codebase
COPY . .

# Optional: ensure UTF-8 encoding (good for LLM input/output)
ENV PYTHONIOENCODING=utf-8

# Run the script
CMD ["python", "rag_pipeline_mlflow.py"]
