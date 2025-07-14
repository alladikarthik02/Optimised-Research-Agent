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
```
#🛠️ Fine-tune the model (optional)
```bash
# 1️⃣  TSV → JSONL  (instruction \t answer)
poetry run python scripts/prepare_dataset.py data/pairs.tsv data/train.jsonl

# 2️⃣  LoRA fine-tune on single GPU (24 GB VRAM is enough)
poetry run python scripts/finetune_lora.py     # outputs lora-out/

# 3️⃣  (Optional) merge LoRA → full weights & push to HF
poetry run python scripts/merge_lora.py \
    meta-llama/Meta-Llama-3-8B-Instruct lora-out merged-out
huggingface-cli upload merged-out/* \
    --repo yourname/llamascholar-8b-finetuned
Then set in .env:

LLM_MODEL=@hf/yourname/llamascholar-8b-finetuned
```

🔌 Environment variables (.env)

Key	Example	Required
CF_ACCOUNT_ID	abcd1234…	✅ for default Cloudflare model
CF_API_TOKEN	cf-live-token…	✅
REDIS_URL	redis://localhost:6379/0/llamascholar-chat	✅ (use Redis Stack)
HF_HUB_TOKEN	hf_…	only for pushing merged weights
📦 Project structure

llamascholar/
  api.py                FastAPI server
  graph_runner.py       LangGraph agent builder
  memory.py             Redis / in-memory saver
  rag_tool.py           vector_qa LangChain tool
  tool_registry.py      DuckDuckGo & arXiv tools
scripts/
  prepare_dataset.py    TSV → JSONL
  finetune_lora.py      LoRA fine-tuning
  merge_lora.py         Merge LoRA into base weights
static/
  index.html            minimal streaming client

🤝 Contributing

Pull requests are welcome—please lint with ruff and run the minimal
pytest suite (make test) before opening a PR.

🪪 License

MIT — see LICENSE.
