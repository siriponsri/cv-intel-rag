---
title: CV Intel RAG
emoji: 🫀
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Cardiovascular + Diabetes + CKD regulatory intelligence RAG agent (Thai + English)
---

# 🫀 CV Intel RAG

**AI-powered regulatory & research intelligence agent for cardiovascular, diabetes, and chronic kidney disease (CV + DM + CKD).**

ตอบคำถามจาก PubMed, ClinicalTrials.gov, openFDA, FDA MedWatch, EMA, ESC, AHA — ภาษาไทย/อังกฤษ — ด้วย [Typhoon LLM](https://opentyphoon.ai) จาก SCB 10X และ [BGE-M3](https://huggingface.co/BAAI/bge-m3) embeddings

[![CI](https://github.com/YOUR_USERNAME/cv-intel-rag/actions/workflows/ci.yml/badge.svg)](../../actions)
[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/cv-intel-rag)

---

## 🎯 Use Case

องค์กรสาธารณสุข / บริษัทยา / นักวิจัยที่ต้องติดตาม
- 📄 Regulatory updates (FDA recalls, EMA alerts, safety signals)
- 🧪 Clinical trials (new phase 2/3 ในโรคกลุ่ม cardio-metabolic-renal)
- 📚 Research publications (PubMed — SGLT2, GLP-1, statins, etc.)
- 🏥 Clinical guidelines (ESC, AHA)

**แทนที่การ manual tracking** ด้วย AI agent ที่ ingest อัตโนมัติ → answer questions via chat + dashboard

## 🏗️ Architecture

```
Sources ──► Connectors ──► SQLite ──► Chunker ──► BGE-M3 ──► ChromaDB
(PubMed,     (7 modular                             ↓
 CT.gov,      connectors,                       HybridRetriever
 openFDA,     CV regex filter)                   (dense + BM25)
 RSS×4)                                               ↓
                                              Typhoon LLM + [S#] citations
                                                      ↓
                                     FastAPI (chat SSE + dashboard)
```

## 🧰 Tech Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Typhoon v2.5 (SCB 10X) | Thai-native, free tier 5 req/s, OpenAI-compatible |
| Embeddings | BGE-M3 (`BAAI/bge-m3`) | 1024-dim, 8192 ctx, Thai + English |
| Vector DB | ChromaDB (embedded) | Zero-setup, file-backed |
| Retrieval | Hybrid (dense + BM25 α=0.6) | Best-of-both for rare medical terms |
| Backend | FastAPI + SSE | Streaming chat responses |
| Frontend | Vanilla JS (chat.html + dashboard.html) | No build step, fast iteration |
| DB | SQLite + ChromaDB | Single-machine friendly, Postgres-compat schema |

## 🚀 Quick Start

### Option A — Try the live HF Space

👉 [huggingface.co/spaces/YOUR_USERNAME/cv-intel-rag](https://huggingface.co/spaces/YOUR_USERNAME/cv-intel-rag)

### Option B — Run locally

```bash
git clone https://github.com/YOUR_USERNAME/cv-intel-rag.git
cd cv-intel-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Get a free Typhoon API key: https://playground.opentyphoon.ai/settings/api-key
cp .env.example .env
# Edit .env and set TYPHOON_API_KEY

python scripts/init_db.py
python scripts/run_ingestion.py         # fetch + index (first run ~10 min)
uvicorn src.main:app --reload --port 8000
```

Open http://localhost:8000 → chat UI
Open http://localhost:8000/dashboard → records + stats

### Option C — Run on Google Colab (free T4 GPU)

See [`notebooks/`](./notebooks/) — two notebooks work together via a shared Google Drive folder:

1. **`01_ingest_and_index.ipynb`** — run once (~10 min on T4) → saves index to `Drive/cv-intel-rag/data/`
2. **`02_demo_visualization.ipynb`** — run anytime (~30 sec) → demo + visualizations for sales presentations

Full walkthrough: [`docs/COLAB_GUIDE_TH.md`](./docs/COLAB_GUIDE_TH.md)

## 📦 Deploy to HF Spaces

See [`docs/DEPLOY_GUIDE_TH.md`](./docs/DEPLOY_GUIDE_TH.md) for step-by-step. TL;DR:

1. Run notebook 1 on Colab → get `data/` with pre-built index
2. Create a new HF Space (Docker SDK)
3. Add `TYPHOON_API_KEY` in Space Settings → Secrets
4. Push this repo to the Space (or use `sync-to-hf.yml` GH Action)

## 🧪 Tests

```bash
pytest tests/ -v
```

- `test_chunker.py` — chunker logic (fast, no ML deps)
- `test_connectors.py` — PubMed/CT.gov/openFDA/RSS parsing (HTTP mocked, connector logic real)
- `test_rag_integration.py` — real embed + real ChromaDB + real retrieval (uses tiny MiniLM for CI)

## 📚 Docs (ภาษาไทย)

- [`docs/GUIDE_TH.md`](./docs/GUIDE_TH.md) — คู่มือ dev ทั่วไป + VS Code + Claude Code workflow
- [`docs/COLAB_GUIDE_TH.md`](./docs/COLAB_GUIDE_TH.md) — รัน Colab ทั้ง 2 notebooks
- [`docs/DEPLOY_GUIDE_TH.md`](./docs/DEPLOY_GUIDE_TH.md) — deploy ขึ้น HF Spaces
- [`CLAUDE.md`](./CLAUDE.md) — hints สำหรับ Claude Code (repo map, commands, conventions)

## 🔒 Security

- `.env` และ `data/` อยู่ใน `.gitignore` — **ห้าม commit API key**
- ใช้ HF Space Secrets (ไม่ใช่ hardcoded) สำหรับ production
- ถ้า repo เป็น public: ใช้ [gitleaks](https://github.com/gitleaks/gitleaks) pre-commit hook

## 📄 License

MIT

## 🙏 Credits

- [Typhoon LLM](https://opentyphoon.ai) by SCB 10X
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) by BAAI
- Sources: [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [ClinicalTrials.gov](https://clinicaltrials.gov/), [openFDA](https://open.fda.gov/), [FDA MedWatch](https://www.fda.gov/safety/medwatch-fda-safety-information-and-adverse-event-reporting-program), [EMA](https://www.ema.europa.eu/), [ESC](https://www.escardio.org/), [AHA](https://www.heart.org/)
