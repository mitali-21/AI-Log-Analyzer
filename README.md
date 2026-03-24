# AI Log Analyzer

An **AI-powered log analysis system** that enables engineers to query application logs using natural language and receive **precise, context-aware answers with source attribution**.

Built using a **Retrieval-Augmented Generation (RAG)** architecture, this tool mimics how an experienced SRE debugs production issues — by correlating relevant log signals and explaining root causes.

**Stack:** Python · LangChain · FAISS · Groq · Streamlit

---

## The RAG Pipeline

```
Log File
   │
   ▼
Chunking (split into 20 lines + overlap)
   │
   ▼
Embedding (text-embedding-3-small)
   │
   ▼
FAISS Vector Store
   │
   ▼
User Query → Embedding
   │
   ▼
Top-K Retrieval (K=5)
   │
   ▼
LLM (GPT-4o-mini)
   │
   ▼
Answer + Source Citations
```

---

## Project Structure

```
ai-log-analyzer/
├── app.py                  ← Streamlit web UI (main interface)
├── main.py                 ← CLI (optional)
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py         ← All config: model,chunk size,top-K
│
├── analyzer/
│   └── log_reader.py       ← File I/O + log line parsing + stats
│
├── rag/                    ← The RAG pipeline
│   ├── chunker.py          ← Split log text into overlapping chunks
│   ├── embedder.py         ← OpenAI embeddings → FAISS index
│   ├── retriever.py        ← Similarity search on FAISS
│   └── generator.py        ← GPT answer generation (streaming + non-streaming)
│
└── sample_logs/
    └── app.log             ← Realistic test log (DB outage,disk,Stripe errors)
```

---

## 🔑 Key Design Decisions

### 1. Line-Based Chunking (vs Character-Based)

- Logs are **line-structured** (timestamp + level + message)
- Character chunking can break lines & stack traces  
- Ensures **complete log events** and better context

---

### 2. Overlapping Windows

- **20 lines per chunk + 5-line overlap**
- Prevents context loss across chunks  
- Helps capture multi-line errors

---

### 3. FAISS (In-Memory Vector Store)

- Logs → embeddings → stored in FAISS  
- Query → embedding → similarity search → top matches  
- Fast, simple, no setup (best for local use)

---

### 4. Top-K Retrieval (K = 5)

- Retrieve **top 5 most relevant chunks**
- Too small → miss context  
- Too large → adds noise  
- K=5 = **balanced signal vs noise**

---

### 5. Temperature = 0

- Controls LLM randomness  
- 0 = **deterministic, factual output**  
- Avoids hallucinations in debugging

---

### 6. Source Attribution

- Returns **answer + exact log lines used**  
- Builds trust and enables verification

---

### 7. Streaming Output

- Responses shown **in real-time (token by token)**  
- Improves UX and responsiveness

**How:** LLM streaming API + Streamlit (`st.write_stream`)  
**Why Streamlit:** Simple, fast UI for real-time apps

---

## 💻 Setup

```bash
1. Clone and create virtual environment:
git clone <repo-url>
cd ai-log-analyzer
python3 -m venv venv
source venv/bin/activate       
# Windows: venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Add your Groq API key
cp .env.example .env
#Get a key:  https://console.groq.com/
#Edit .env:  GROQ_API_KEY=gsk-...
```

---

## ▶️ Run

### Streamlit UI

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### CLI

```bash
python3 main.py sample_logs/app.log "What caused the database outage?"
python3 main.py sample_logs/app.log "Show all critical errors"
```

---

## 📊 Example Output

**Query:**

```
What caused the database outage?
```

**Answer:**

- Connection pool exhausted
- High DB latency
- Retry amplification  


**Sources:**

```
[ERROR] DB connection timeout
[WARN] Retry attempt 3
[ERROR] Connection pool exhausted
```



## 💲 Cost Estimate

Analyzing the 47-line sample log file costs approximately:


| Step                  | Model                  | Cost                      |
| --------------------- | ---------------------- | ------------------------- |
| Embedding (3 chunks)  | text-embedding-3-small | ~$0.00001                 |
| LLM Answer generation | gpt-4o-mini            | ~$0.0002                  |
| **Total / Query**     |                        | **~$0.0002 per question** |


---

## ⚖️ Current Trade-offs

- **In-Memory FAISS**
  - Fast and simple, but not persistent or scalable

- **Batch Processing (Single File)**
  - Easy to use, but no real-time log ingestion

- **Top-K Context (K=5)**
  - Efficient, but may miss long-range log correlations

- **No Cross-Service Correlation**
  - Works on isolated logs, not distributed systems

---

## 🚀 Future Enhancements

### 1. Caching Layer
- Use Redis for query & embedding caching  
- Reduces cost and improves latency

---

### 2. Real-Time Log Ingestion
- Integrate Kafka / Kinesis  
- Pipeline: Logs → Stream → Chunk → Embed → Store  
- Enables continuous monitoring

---

### 3. Multi-Service Correlation
- Use Trace IDs + OpenTelemetry  
- Helps with end-to-end root cause analysis

---

### 4. Persistent Vector DB
- Use Pinecone / Weaviate / Milvus  
- Enables durability, multi-user access, and scaling

---

### 5. Hybrid Retrieval (Keyword + Vector)
- Combine BM25 + embeddings  
- Improves accuracy for exact + semantic search

---

## 🧠 Key Takeaway

> Lightweight RAG system for learning and local use  
> ➝ Can evolve into a scalable, real-time log intelligence platform

---

