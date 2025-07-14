# LlamaScholar 🦙📚

**LlamaScholar** is an end-to-end research assistant that couples a fine-tuned **Llama-3 8B** model with a modern retrieval pipeline (DuckDuckGo, arXiv, and PDF-to-Chroma RAG) and serves real-time answers via **FastAPI** + SSE.  
It runs anywhere Docker runs and persists chat history in **Redis Stack**.

---

## ✨ Features

| Layer | What it gives you |
|-------|-------------------|
| **LLM** | Cloudflare Workers-AI wrapper by default, or your own HF model (`@hf/…`). |
| **Fine-tuning** | LoRA/QLoRA scripts (`scripts/finetune_lora.py`, `merge_lora.py`) for instruction tuning on custom datasets. |
| **RAG** | `ingest_pdf.py` → splits PDFs, stores chunks in **Chroma**; queried via `vector_qa` LangChain tool. |
| **Agent** | Zero-Shot-ReAct built with **LangGraph**; tools: DuckDuckGo, arXiv, vector QA. |
| **Memory** | **RedisSaver** (Redis Stack) checkpoint – chat survives restarts & scales horizontally. |
| **API** | `/ask` streams tokens (Server-Sent Events), `/health` for probes, Swagger docs `/docs`. |
| **Web UI** | `static/index.html` – 1-page client that streams answers live. |
| **Dev & Deploy** | Dockerfile, `docker-compose.yml`, `.env.example`; CI stub in `.github/workflows/`. |

---

## 🚀 Quick start (local)

```bash
git clone https://github.com/<you>/llamascholar.git
cd llamascholar
cp .env.example .env          # fill CF_ACCOUNT_ID / CF_API_TOKEN / REDIS_URL
docker compose up --build     # brings up API + Redis-Stack
# ➜ http://localhost:8000/docs  – test POST /ask
