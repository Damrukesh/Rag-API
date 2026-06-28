FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY app.py knowledge_base.py profile.txt ./
COPY chroma_db ./chroma_db
RUN pip install fastapi uvicorn chromadb ollama pypdf
RUN python knowledge_base.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
