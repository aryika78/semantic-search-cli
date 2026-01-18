# 🧮 Semantic Search CLI – Embeddings & Vector Similarity

A **production‑grade Semantic Search Command Line Interface** that demonstrates how modern AI systems retrieve relevant information using **text embeddings and vector similarity**.

This project represents the **retrieval layer of a RAG (Retrieval‑Augmented Generation) system**:

```
Text → Embeddings → Similarity → Ranking → Retrieval
```

## 🎯 Project Objectives

* Convert text into numerical embeddings
* Compare semantic similarity between texts
* Perform semantic search over a corpus
* Build and query a persistent vector index
* Benchmark embedding models
* Demonstrate performance characteristics

---

## 🏗️ Project Structure

```
semantic-search/
│
├── semantic_search/
│   ├── __init__.py
│   ├── embeddings.py      # Embedding generation
│   ├── similarity.py      # Vector similarity math
│   ├── compare.py         # Text comparison logic
│   ├── search.py          # Non-indexed semantic search
│   └── index.py           # Vector index (build/search)
│
├── cli/
│   ├── __init__.py
│   └── main.py            # CLI entry point
│
├── tests/
|   ├── conftest.py
│   ├── test_embeddings.py
│   ├── test_similarity.py
│   ├── test_compare.py
│   ├── test_search.py
│   └── test_index.py
│
├── data/
│   └── corpus.txt         # Sample corpus
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Verify Setup

```bash
python -c "from fastembed import TextEmbedding; print('FastEmbed OK')"
```

---

## 🧮 CLI Commands & Usage

All commands are executed from the project root:

```bash
python -m cli.main <command> [options]
```

---

## 1️⃣ Embed – Generate Text Embeddings

### Single Text (with timing)

```bash
python -m cli.main embed "Machine learning is powerful" --model BAAI/bge-small-en-v1.5
```

**Output:**

* Embedding dimensions
* First 10 values
* Execution time

---

### Batch from File (with progress & timing)

```bash
python -m cli.main embed --file data/corpus.txt --model BAAI/bge-small-en-v1.5
```

**Output:**

* Number of texts
* Embedding matrix shape
* Progress bar
* Execution time

---

## 2️⃣ Compare – Compare Two Texts

```bash
python -m cli.main compare "I love Python" "I enjoy programming"
```

**Metrics Returned:**

* Cosine similarity
* Euclidean distance
* Dot product
* Human‑readable interpretation

---

## 3️⃣ Semantic Search (Non‑Indexed)

```bash
python -m cli.main search "heart disease symptoms" data/corpus.txt
```

### Optional Threshold

```bash
python -m cli.main search "heart disease symptoms" data/corpus.txt --threshold 0.6
```

**Output:**

* Ranked results
* Similarity scores
* Search time

---

## 4️⃣ Vector Index – Build & Search

### Build Index (Persistent)

```bash
python -m cli.main index build data/corpus.txt --model BAAI/bge-small-en-v1.5
```

**Output:**

* Progress bar
* Document count
* Build time
* Saved index file (`index.npz`)

---

### Search Index

```bash
python -m cli.main index search "heart disease symptoms" --model BAAI/bge-small-en-v1.5
```

### With Threshold

```bash
python -m cli.main index search "heart disease symptoms" --threshold 0.6
```

**Output:**

* Ranked results
* Search time

> ⚠️ Index enforces **model consistency**. Searching with a different model than the one used to build the index will raise a clear error.

---

## ⚡ 5️⃣ Benchmark – Model Comparison

```bash
python -m cli.main benchmark "I love Python" "I enjoy programming" --models "BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5"
```

**Output:**

* Cosine / Euclidean / Dot scores
* Execution time per model

---

## 🧪 Testing

Run all tests:

```bash
pytest -v
```

Tests cover:

* Embedding correctness
* Similarity math
* Search ranking
* Index persistence
* Interpretation logic

---

---

## 🐳 Docker Usage (Run Anywhere)

This CLI is fully dockerized so it can run without installing Python locally.

### 📦 Build Image

```bash
docker build -t semantic-search-cli .
```

---

### ⚡ Create Cache Folder (One Time on Host)

```bash
mkdir fastembed_cache
```

---

## 🐳 Docker Compose (Recommended)

Docker Compose lets you run the CLI without typing long Docker commands. It automatically handles image build, volume mounts, and execution.

### 📄 docker-compose.yml

Create this file in project root:

```yaml
version: "3.9"

services:
  app:
    build: .
    container_name: semantic-search-cli
    volumes:
      - ./fastembed_cache:/root/.cache/fastembed
      - ./fastembed_cache:/root/.cache/huggingface
      - ./data:/app/data
    entrypoint: ["python", "-m", "cli.main"]
```

---

### ▶️ Build using Compose

```bash
docker compose build
```

---

### 🧮 Embed

```bash
docker compose run --rm app embed "Machine learning is powerful"
```

---

### 🔍 Compare

```bash
docker compose run --rm app compare "I love Python" "I enjoy programming"
```

---

### 📚 Semantic Search

```bash
docker compose run --rm app search "machine learning models" data/corpus.txt
```

---

### 📦 Index Build

```bash
docker compose run --rm app index build data/corpus.txt --output data/index.npz
```

---

### 🔎 Index Search

```bash
docker compose run --rm app index search "machine learning models" --index data/index.npz
```

---

### ⚡ Benchmark

```bash
docker compose run --rm app benchmark "I love Python" "I enjoy programming"
```

---

## 👩‍💻 Author

**Aryika Patni**
