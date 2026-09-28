FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd -m app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN chown app /app
USER app
# bake tokenizer, NLTK data and BM25 params into the image so startup needs no downloads
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); from pinecone_text.sparse import BM25Encoder; BM25Encoder.default().dump('/app/bm25_params.json'); from pythainlp.tokenize import word_tokenize; word_tokenize('ทดสอบ')"
COPY --chown=app app/ .
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
