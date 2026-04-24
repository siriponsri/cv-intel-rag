# 🛠️ Developer Guide (ภาษาไทย)

คู่มือสำหรับคนที่ pull repo มาแล้วอยากเริ่ม develop ต่อ ไม่ว่าจะใน VS Code + Claude Code หรือแก้โค้ดเอง

---

## 1. ติดตั้ง local

```bash
git clone https://github.com/YOUR_USERNAME/cv-intel-rag.git
cd cv-intel-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# แก้ .env แล้วใส่ TYPHOON_API_KEY (ถ้ายังไม่มี → https://playground.opentyphoon.ai/settings/api-key)
```

> 💡 **ถ้าไม่มี GPU:** ตั้ง `EMBED_DEVICE=cpu` ใน `.env` แล้วลด `EMBED_BATCH_SIZE=4` — embed จะช้ากว่า GPU ~10 เท่า แต่รัน develop ได้

---

## 2. Init DB + ingest ชุดเล็กสำหรับ develop

```bash
python scripts/init_db.py                           # สร้าง SQLite schema
python scripts/run_ingestion.py --limit 5           # ดึงแค่ 5 records/source (~1 นาที)
uvicorn src.main:app --reload --port 8000
```

เปิด browser:
- http://localhost:8000 — Chat UI (SSE streaming)
- http://localhost:8000/dashboard — Records + stats
- http://localhost:8000/docs — Swagger auto-gen

---

## 3. โครงสร้างโปรเจกต์

```
src/
├── config/       — settings (pydantic-settings), domain terms (MeSH, RSS feeds, CV regex)
├── connectors/   — 7 ตัว: pubmed, clinicaltrials, openfda, rss_medwatch/ema/esc/aha
├── db/           — SQLAlchemy ORM + schema.sql + RecordRepository
├── models/       — Pydantic Record + enums
├── rag/          — chunker, embedder (BGE-M3), vectorstore (ChromaDB), retriever (hybrid), indexer
├── llm/          — OpenAICompatClient (Typhoon/vLLM/any) + NullLLMClient fallback + prompts
├── agent/        — RAGAgent.answer() + answer_stream()
├── static/       — chat.html + dashboard.html (vanilla JS, ไม่ต้อง build)
└── main.py       — FastAPI endpoints

scripts/          — init_db, run_ingestion, rebuild_index, build_notebooks
tests/            — chunker, connectors (HTTP mocked), RAG integration (real embed + real Chroma)
notebooks/        — 01_ingest_and_index.ipynb, 02_demo_visualization.ipynb
```

---

## 4. VS Code + Claude Code workflow

### ติดตั้ง

1. ติดตั้ง [Claude Code](https://docs.claude.com/en/docs/agents-and-tools/claude-code/overview) — terminal tool
2. เปิด VS Code → เปิด folder `cv-intel-rag/` → เปิด integrated terminal
3. `claude` → login → ready

### ใช้งาน

`CLAUDE.md` ที่ root ของ repo มี hints พื้นฐานอยู่แล้ว Claude Code จะอ่านอัตโนมัติ

ตัวอย่างคำสั่งที่ work ดี:

```
Add a new connector for WHO drug safety alerts. Follow the pattern in src/connectors/rss.py
```

```
The dashboard doesn't show records from the last 7 days — help me debug
```

```
Run pytest and fix any failures
```

---

## 5. รันเทสต์

```bash
pytest tests/ -v                                    # all tests
pytest tests/test_chunker.py -v                     # เร็ว (<1s), ไม่ต้อง ML deps
pytest tests/test_connectors.py -v                  # HTTP mocked, connector logic real
pytest tests/test_rag_integration.py -v             # ต้องมี sentence-transformers + chromadb
```

`test_rag_integration.py` ใช้ MiniLM (22MB) แทน BGE-M3 (2.3GB) เพื่อให้ CI เร็ว — embedder กับ ChromaDB ของจริง (ไม่ mock) แค่ตัว model เล็กกว่า production

---

## 6. เพิ่ม Connector ใหม่

1. สร้าง `src/connectors/<new>.py` ที่ inherit `BaseConnector`
2. Implement `fetch()` ให้ yield `Record` objects
3. ลงทะเบียนใน `src/connectors/registry.py`
4. เพิ่ม test ใน `tests/test_connectors.py` (mock HTTP, ตรวจว่า parse record ถูกต้อง)

อ้างอิง pattern: `src/connectors/pubmed.py` (API), `src/connectors/rss.py` (RSS with CV regex filter)

---

## 7. ปัญหาที่เจอบ่อย

| ปัญหา | วิธีแก้ |
|---|---|
| `tiktoken unavailable (403)` | **ปกติ ไม่ต้องแก้** — `_CharEstimator` fallback ทำงานเอง (ใน sandboxed env / corporate proxy tiktoken โหลด BPE ไม่ได้) |
| BGE-M3 โหลดช้ามาก | ~2.3GB ครั้งแรก — ตั้ง `HF_HOME` ให้ cache ไว้ที่อื่น หรือใช้ Colab |
| ChromaDB error "collection not found" | รัน `python scripts/rebuild_index.py` |
| ingestion 0 records | ตรวจ `.env` → `APP_ENV` กับ network — PubMed/CT.gov ต้องเข้า internet ได้ |
| test_rag_integration.py skip | ติดตั้ง `pip install sentence-transformers chromadb` (ครั้งแรก ~700MB) |

---

## 8. Deploy

ดู [`docs/DEPLOY_GUIDE_TH.md`](./DEPLOY_GUIDE_TH.md) — deploy ไปที่ Hugging Face Spaces (ฟรี, CPU, 16GB RAM)

---

## 9. Tips

- **แก้ prompt LLM:** `src/llm/prompts.py` — `build_rag_messages()` คือ prompt หลัก
- **เปลี่ยน chunk size:** `.env` → `CHUNK_SIZE_TOKENS=500` (default) ค่าน้อยลง = chunk เยอะขึ้น = เจาะจงขึ้น แต่ context แตกง่าย
- **Hybrid alpha:** `.env` → `RETRIEVE_ALPHA=0.6` (0=BM25 only, 1=dense only) ถ้า domain มีศัพท์เฉพาะเยอะ (ชื่อยา) ลดลงเป็น 0.4
- **Add category filter:** ใช้ query param `?category=safety` กับ `/search` endpoint
