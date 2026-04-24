# คู่มือนักพัฒนา — CV Intel RAG

**ระดับ:** มือใหม่ถึงระดับกลาง | **เวลาโดยประมาณ:** 30–60 นาที (ไม่รวมเวลาดาวน์โหลด dependencies)

คู่มือฉบับนี้อธิบายขั้นตอนการติดตั้ง พัฒนา ทดสอบ และขยายระบบ CV Intel RAG บนเครื่องของคุณอย่างละเอียดทีละขั้นตอน ตั้งแต่การ clone repository จนถึงการรัน server และแก้ไขโค้ด

---

## สารบัญ

1. [สิ่งที่ต้องเตรียมก่อนเริ่ม](#1-สิ่งที่ต้องเตรียมก่อนเริ่ม)
2. [ขั้นตอนการติดตั้งบนเครื่อง Local](#2-ขั้นตอนการติดตั้งบนเครื่อง-local)
3. [การตั้งค่า Environment Variables](#3-การตั้งค่า-environment-variables)
4. [การเริ่มต้นใช้งานระบบ](#4-การเริ่มต้นใช้งานระบบ)
5. [โครงสร้างโปรเจกต์](#5-โครงสร้างโปรเจกต์)
6. [การรันชุดทดสอบ (Tests)](#6-การรันชุดทดสอบ-tests)
7. [การใช้งานร่วมกับ VS Code และ Claude Code](#7-การใช้งานร่วมกับ-vs-code-และ-claude-code)
8. [การเพิ่ม Connector ใหม่](#8-การเพิ่ม-connector-ใหม่)
9. [การปรับแต่ง RAG Pipeline](#9-การปรับแต่ง-rag-pipeline)
10. [คำสั่งที่ใช้บ่อย](#10-คำสั่งที่ใช้บ่อย)
11. [ปัญหาที่พบบ่อยและวิธีแก้ไข](#11-ปัญหาที่พบบ่อยและวิธีแก้ไข)

---

## 1. สิ่งที่ต้องเตรียมก่อนเริ่ม

ตรวจสอบให้แน่ใจว่าเครื่องของคุณมีสิ่งต่อไปนี้ก่อนดำเนินการ:

| รายการ | เวอร์ชันขั้นต่ำ | วิธีตรวจสอบ | ลิงก์ดาวน์โหลด |
|--------|----------------|-------------|----------------|
| Python | 3.11 | `python --version` | [python.org](https://www.python.org/downloads/) |
| Git | 2.x | `git --version` | [git-scm.com](https://git-scm.com/) |
| Typhoon API Key | — | — | [playground.opentyphoon.ai](https://playground.opentyphoon.ai/settings/api-key) |

> **หมายเหตุเกี่ยวกับ GPU:** ระบบนี้รันได้บน CPU ล้วน (ไม่ต้องมี GPU) แต่ขั้นตอน embedding จะช้ากว่า GPU ประมาณ 10 เท่า สำหรับการพัฒนาและทดสอบนั้น CPU เพียงพอ

### 1.1 การขอ Typhoon API Key (ฟรี)

1. เปิดเบราว์เซอร์ไปที่ [playground.opentyphoon.ai/settings/api-key](https://playground.opentyphoon.ai/settings/api-key)
2. สมัครหรือล็อกอินด้วย Google Account
3. กดปุ่ม **Create new secret key**
4. คัดลอก key ที่ขึ้นต้นด้วย `sk-` และเก็บไว้ในที่ปลอดภัย (key นี้จะแสดงครั้งเดียว)

---

## 2. ขั้นตอนการติดตั้งบนเครื่อง Local

ดำเนินการตามลำดับขั้นตอนต่อไปนี้ทีละขั้นตอน ห้ามข้ามขั้นตอนใด

### ขั้นตอนที่ 2.1 — Clone Repository

```bash
git clone https://github.com/siriponsri/cv-intel-rag.git
cd cv-intel-rag
```

**ผลลัพธ์ที่ควรได้:** ไดเรกทอรี `cv-intel-rag/` ถูกสร้างขึ้นพร้อมไฟล์โปรเจกต์ทั้งหมด

### ขั้นตอนที่ 2.2 — สร้าง Virtual Environment

Virtual Environment คือ "กล่อง" แยกสำหรับ Python packages ของโปรเจกต์นี้โดยเฉพาะ ป้องกันไม่ให้ packages ของโปรเจกต์อื่นบนเครื่องชนกัน

```bash
# สร้าง virtual environment ชื่อ .venv
python -m venv .venv
```

จากนั้น **เปิดใช้งาน** virtual environment:

```bash
# Linux หรือ macOS:
source .venv/bin/activate

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

**ตรวจสอบ:** หลังจาก activate สำเร็จ ชื่อ `(.venv)` จะปรากฏที่หน้า prompt เช่น:
```
(.venv) C:\Users\YourName\cv-intel-rag>
```

### ขั้นตอนที่ 2.3 — ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

> **หมายเหตุ:** การติดตั้งครั้งแรกจะใช้เวลาประมาณ 5–15 นาที ขึ้นอยู่กับความเร็วอินเทอร์เน็ต เนื่องจากมี packages ขนาดใหญ่เช่น PyTorch (~700 MB) และ sentence-transformers

**ตรวจสอบการติดตั้ง:**
```bash
python -c "import fastapi, chromadb, sentence_transformers; print('ติดตั้งสำเร็จ')"
```
ควรพิมพ์ `ติดตั้งสำเร็จ` โดยไม่มี error

---

## 3. การตั้งค่า Environment Variables

ระบบอ่านการตั้งค่าทั้งหมดจากไฟล์ `.env` ในไดเรกทอรีหลักของโปรเจกต์

### ขั้นตอนที่ 3.1 — คัดลอกไฟล์ตัวอย่าง

```bash
# Linux / macOS:
cp .env.example .env

# Windows (PowerShell):
Copy-Item .env.example .env
```

### ขั้นตอนที่ 3.2 — แก้ไขไฟล์ .env

เปิดไฟล์ `.env` ด้วย text editor และแก้ไขค่าต่อไปนี้เป็นอย่างน้อย:

```dotenv
# จำเป็น: ใส่ Typhoon API key ที่ได้จากขั้นตอน 1.1
TYPHOON_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# สำหรับการพัฒนาบนเครื่องที่ไม่มี GPU ให้คงค่าเหล่านี้ไว้
EMBED_DEVICE=cpu
EMBED_BATCH_SIZE=16
DEFAULT_LLM_PROVIDER=typhoon_api
```

ค่าที่เหลือสามารถคงไว้เป็นค่าเริ่มต้น (default) ได้สำหรับการพัฒนา

> **ข้อควรระวัง:** ห้ามนำไฟล์ `.env` ขึ้น Git เด็ดขาด เนื่องจากมี API key ที่เป็นความลับ ไฟล์ `.gitignore` ของโปรเจกต์นี้กำหนดให้ `.env` ถูกละเว้นโดยอัตโนมัติแล้ว

---

## 4. การเริ่มต้นใช้งานระบบ

### ขั้นตอนที่ 4.1 — สร้างฐานข้อมูล

คำสั่งนี้สร้าง SQLite database ที่ `data/cv_intel.db` จาก schema ที่กำหนดไว้ใน `src/db/schema.sql`

```bash
python scripts/init_db.py
```

**ผลลัพธ์ที่ควรได้:**
```
INFO Initialising DB at sqlite:///./data/cv_intel.db
INFO ✓ DB initialised with 22 statements
```

### ขั้นตอนที่ 4.2 — ดึงข้อมูลชุดเล็กเพื่อทดสอบ

คำสั่งนี้ดึงข้อมูลจากทุกแหล่ง (PubMed, ClinicalTrials.gov, openFDA, RSS feeds) แหล่งละ 5 รายการ เพื่อให้มีข้อมูลพอทดสอบระบบ RAG ได้โดยไม่ต้องรอนาน

```bash
python scripts/run_ingestion.py --limit 5
```

**ผลลัพธ์ที่ควรได้:** log แสดงจำนวน records ที่ดึงได้จากแต่ละแหล่ง เช่น:
```
INFO [pubmed] inserted 5, updated 0
INFO [clinicaltrials] inserted 5, updated 0
INFO [openfda] inserted 3, updated 0
...
```

> **หมายเหตุ:** ถ้าต้องการ corpus ขนาดใหญ่สำหรับการทดสอบที่สมจริงมากขึ้น ใช้ `--limit 50` หรือไม่ระบุ `--limit` เพื่อดึงข้อมูลทั้งหมด (อาจใช้เวลา 10–20 นาที)

### ขั้นตอนที่ 4.3 — เริ่ม Development Server

```bash
uvicorn src.main:app --reload --port 8000
```

Flag `--reload` ทำให้ server restart อัตโนมัติทุกครั้งที่คุณแก้ไขไฟล์ Python

**ผลลัพธ์ที่ควรได้:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### ขั้นตอนที่ 4.4 — เปิดใช้งานผ่าน Browser

เปิด browser และไปที่ URL ต่อไปนี้:

| URL | คำอธิบาย |
|-----|----------|
| `http://localhost:8000` | Chat UI — ถามคำถามและรับคำตอบพร้อม citations แบบ real-time streaming |
| `http://localhost:8000/dashboard` | Dashboard — ดูและกรอง records ที่ ingested ตาม source และ category |
| `http://localhost:8000/docs` | Swagger UI — เอกสาร API แบบ interactive ที่ generate อัตโนมัติ |
| `http://localhost:8000/health` | Health check — ตรวจสอบว่า server ทำงานปกติ |

---

## 5. โครงสร้างโปรเจกต์

การทำความเข้าใจโครงสร้างช่วยให้ทราบว่าต้องแก้ไขไฟล์ใดเมื่อต้องการเปลี่ยนแปลงพฤติกรรมของระบบ

```
cv-intel-rag/
│
├── src/                          ← Source code หลักทั้งหมด
│   ├── main.py                   ← FastAPI app และ HTTP endpoints ทั้งหมด
│   │
│   ├── config/
│   │   ├── settings.py           ← การตั้งค่าทั้งหมด โหลดจาก .env
│   │   └── domain.py             ← MeSH queries, RSS feeds, CV_TERM_REGEX
│   │
│   ├── connectors/               ← ส่วนที่ดึงข้อมูลจากแหล่งภายนอก
│   │   ├── base.py               ← BaseConnector (abstract class ที่ทุกตัว inherit)
│   │   ├── pubmed.py             ← NCBI E-utilities API (PubMed)
│   │   ├── clinical_trials.py   ← ClinicalTrials.gov v2 API
│   │   ├── openfda.py            ← openFDA drug events + enforcement
│   │   ├── rss.py                ← RSS feeds สำหรับ FDA/EMA/ESC/AHA
│   │   └── registry.py           ← รายการ connectors ทั้งหมด
│   │
│   ├── db/                       ← ฐานข้อมูล SQLite
│   │   ├── models.py             ← SQLAlchemy ORM tables
│   │   ├── repository.py         ← การ query และ upsert records
│   │   ├── schema.sql            ← SQL DDL สำหรับสร้าง tables
│   │   └── session.py            ← การเชื่อมต่อฐานข้อมูล
│   │
│   ├── rag/                      ← RAG pipeline
│   │   ├── chunker.py            ← แบ่งข้อความเป็น chunks (500 tokens/chunk)
│   │   ├── embedder.py           ← แปลง text เป็น vectors ด้วย BGE-M3
│   │   ├── vectorstore.py        ← จัดการ ChromaDB vector store
│   │   ├── retriever.py          ← ค้นหาด้วย Hybrid (dense + BM25)
│   │   └── indexer.py            ← ประสาน chunk → embed → store
│   │
│   ├── llm/
│   │   ├── client.py             ← OpenAI-compatible client (Typhoon/vLLM/null)
│   │   └── prompts.py            ← System prompt template พร้อม [S#] citation
│   │
│   ├── agent/
│   │   └── rag_agent.py          ← ประสาน retrieve → format → generate
│   │
│   └── static/
│       ├── chat.html             ← Chat UI (vanilla JS, SSE streaming)
│       └── dashboard.html        ← Records browser (vanilla JS)
│
├── scripts/
│   ├── init_db.py                ← สร้าง SQLite schema
│   ├── run_ingestion.py          ← รัน connectors และบันทึกลง DB
│   ├── rebuild_index.py          ← สร้าง ChromaDB index ใหม่จาก DB
│   └── build_notebooks.py        ← สร้าง Colab notebooks
│
├── tests/
│   ├── conftest.py               ← Shared fixtures สำหรับ pytest
│   ├── test_chunker.py           ← Unit tests สำหรับ chunker (ไม่ต้อง ML)
│   ├── test_connectors.py        ← Integration tests (HTTP mocked)
│   └── test_rag_integration.py   ← End-to-end RAG test (ต้อง MiniLM + Chroma)
│
├── .env.example                  ← Template สำหรับ .env
├── .gitignore                    ← ไม่รวม .env, data/, .venv/ ใน Git
├── Dockerfile                    ← สำหรับ deploy บน HF Spaces
├── requirements.txt              ← Python dependencies ทั้งหมด
└── CLAUDE.md                     ← Hints สำหรับ Claude Code AI agent
```

---

## 6. การรันชุดทดสอบ (Tests)

โปรเจกต์นี้มีการทดสอบแบ่งเป็น 3 ระดับ สามารถรันแยกตามความต้องการได้

### ระดับที่ 1 — Unit Tests (เร็ว ไม่ต้อง ML)

ทดสอบการแบ่ง chunk, การนับ token, และการจัดการข้อความภาษาไทย ไม่ต้องดาวน์โหลด model ใด ๆ

```bash
python -m pytest tests/test_chunker.py -v
```

**เวลาโดยประมาณ:** น้อยกว่า 2 วินาที

**ตัวอย่างผลลัพธ์ที่ควรได้:**
```
tests/test_chunker.py::test_count_tokens_nonzero_for_nonempty PASSED
tests/test_chunker.py::test_short_text_returns_single_chunk PASSED
tests/test_chunker.py::test_long_text_splits_into_multiple_chunks PASSED
tests/test_chunker.py::test_build_chunks_attaches_metadata PASSED
tests/test_chunker.py::test_thai_text_chunks PASSED
5 passed in 0.45s
```

### ระดับที่ 2 — Connector Tests (ไม่ต้องเชื่อมต่ออินเทอร์เน็ต)

ทดสอบการ parse XML จาก PubMed, JSON จาก ClinicalTrials.gov, และการกรอง keyword โดย HTTP requests ถูก mock ทั้งหมด

```bash
python -m pytest tests/test_connectors.py -v
```

**เวลาโดยประมาณ:** 5–10 วินาที

### ระดับที่ 3 — RAG Integration Tests (ต้อง ML dependencies)

ทดสอบ pipeline ทั้งระบบตั้งแต่ embed จนถึง retrieve โดยใช้ MiniLM model ขนาดเล็ก (22 MB) แทน BGE-M3 เพื่อความเร็ว

```bash
python -m pytest tests/test_rag_integration.py -v
```

**เวลาโดยประมาณ:** 30–60 วินาที (ดาวน์โหลด MiniLM ครั้งแรกอาจนานกว่านี้)

### รันทุก Tests พร้อมกัน

```bash
python -m pytest tests/ -v
```

---

## 7. การใช้งานร่วมกับ VS Code และ Claude Code

Claude Code คือ AI coding assistant ที่รัน command-line สามารถอ่าน codebase และช่วยแก้ไขโค้ดได้อัตโนมัติ ไฟล์ `CLAUDE.md` ที่ root ของ repo มี hints ที่ Claude Code จะอ่านโดยอัตโนมัติทุก session

### ขั้นตอนการติดตั้ง Claude Code

1. ติดตั้ง [Node.js](https://nodejs.org/) หากยังไม่มี
2. รันคำสั่ง:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```
3. เปิด terminal ในไดเรกทอรีโปรเจกต์และรัน:
   ```bash
   claude
   ```
4. ล็อกอินด้วย Anthropic account และทำตาม prompt

### ตัวอย่างคำสั่งที่ใช้ได้ดีกับโปรเจกต์นี้

```
อ่าน CLAUDE.md และสรุปโครงสร้างโปรเจกต์ให้ฉัน
```

```
เพิ่ม connector ใหม่สำหรับ WHO Drug Information โดยใช้ pattern เดียวกับ src/connectors/rss.py
```

```
Dashboard ไม่แสดง records จาก 7 วันที่ผ่านมา ช่วย debug หน่อย
```

```
รัน pytest และแก้ไข failures ทั้งหมดให้ฉัน
```

---

## 8. การเพิ่ม Connector ใหม่

Connector คือ module ที่รับผิดชอบการดึงข้อมูลจากแหล่งข้อมูลหนึ่ง ๆ และแปลงเป็น `Record` objects

### ขั้นตอนที่ 8.1 — สร้างไฟล์ Connector

สร้างไฟล์ `src/connectors/<ชื่อแหล่งข้อมูล>.py` โดยมีโครงสร้างดังนี้:

```python
from __future__ import annotations
from datetime import datetime
from typing import Iterable, Optional
import logging

from .base import BaseConnector
from ..models.record import Record, SourceType, CategoryType

log = logging.getLogger(__name__)


class MyNewConnector(BaseConnector):
    """Connector สำหรับ <ชื่อแหล่งข้อมูล>"""

    source_name = "my_source"    # ต้องไม่ซ้ำกับ connector อื่น
    base_url = "https://api.example.com"

    def fetch(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> Iterable[Record]:
        """ดึงข้อมูลและ yield Record ทีละรายการ"""
        # TODO: implement HTTP request + parsing
        # ตัวอย่าง:
        response = self.session.get(f"{self.base_url}/endpoint", timeout=30)
        response.raise_for_status()
        for item in response.json()["results"][:limit]:
            yield Record(
                source_id=item["id"],
                source=SourceType.OTHER,
                title=item["title"],
                abstract=item.get("description", ""),
                published_date=datetime.fromisoformat(item["date"]),
                url=item["url"],
                category=CategoryType.GENERAL,
            )
```

### ขั้นตอนที่ 8.2 — ลงทะเบียนใน Registry

เปิดไฟล์ `src/connectors/registry.py` และเพิ่ม connector ใหม่:

```python
from .my_new_connector import MyNewConnector   # เพิ่มบรรทัดนี้

CONNECTORS: dict[str, type[BaseConnector]] = {
    "pubmed": PubMedConnector,
    "clinicaltrials": ClinicalTrialsConnector,
    "openfda": OpenFDAConnector,
    "rss_medwatch": RSSMedWatchConnector,
    "rss_ema": RSSEMAConnector,
    "rss_esc": RSSESCConnector,
    "rss_aha": RSSAHAConnector,
    "my_source": MyNewConnector,               # เพิ่มบรรทัดนี้
}
```

### ขั้นตอนที่ 8.3 — เขียน Test

เปิดไฟล์ `tests/test_connectors.py` และเพิ่ม test ที่ mock HTTP request:

```python
def test_my_new_connector_parses_correctly(httpx_mock):
    httpx_mock.add_response(
        url="https://api.example.com/endpoint",
        json={"results": [{"id": "1", "title": "Test", "date": "2026-01-01", "url": "https://example.com"}]},
    )
    connector = MyNewConnector()
    records = list(connector.fetch(limit=1))
    assert len(records) == 1
    assert records[0].title == "Test"
```

### ขั้นตอนที่ 8.4 — ทดสอบ

```bash
# รัน test เฉพาะของ connector ใหม่
python -m pytest tests/test_connectors.py::test_my_new_connector_parses_correctly -v

# ทดสอบ ingestion จริง
python scripts/run_ingestion.py --connector my_source --limit 3
```

---

## 9. การปรับแต่ง RAG Pipeline

### 9.1 การปรับแต่ง System Prompt

แก้ไขไฟล์ `src/llm/prompts.py` ฟังก์ชัน `build_rag_messages()` เพื่อเปลี่ยนพฤติกรรมการตอบของ LLM เช่น:
- เพิ่มข้อกำหนดให้ตอบเป็นภาษาไทยเสมอ
- กำหนด format ของคำตอบ
- เพิ่มข้อมูล context เกี่ยวกับองค์กร

### 9.2 การปรับ Chunk Size

แก้ไขใน `.env`:
```dotenv
CHUNK_SIZE_TOKENS=500     # ค่าน้อยลง = chunks เจาะจงมากขึ้น แต่ context อาจขาดช่วง
CHUNK_OVERLAP_TOKENS=80   # overlap มากขึ้น = รักษา context ข้ามขอบ chunk ได้ดีขึ้น
```

หลังจากเปลี่ยนค่า ต้อง rebuild index:
```bash
python scripts/rebuild_index.py
```

### 9.3 การปรับ Hybrid Retrieval Alpha

```dotenv
RETRIEVE_ALPHA=0.6   # 0.0 = BM25 only, 1.0 = dense only, 0.6 = default
```

- **เพิ่ม alpha** (เข้าใกล้ 1.0): เหมาะเมื่อต้องการค้นหาความหมาย (semantic search) ให้ผลดีขึ้น
- **ลด alpha** (เข้าใกล้ 0.0): เหมาะเมื่อ query มีชื่อยาหรือ abbreviation เฉพาะทาง เช่น "SGLT2i HFrEF"

### 9.4 การเปลี่ยน LLM Provider

แก้ไขใน `.env`:

```dotenv
# ใช้ Typhoon API (default)
DEFAULT_LLM_PROVIDER=typhoon_api
TYPHOON_API_KEY=sk-xxxx

# ใช้ vLLM self-hosted
DEFAULT_LLM_PROVIDER=vllm_local
VLLM_BASE_URL=http://localhost:8000/v1

# ไม่ใช้ LLM (offline fallback — คืนค่า chunks โดยไม่มีการสรุป)
DEFAULT_LLM_PROVIDER=null
```

---

## 10. คำสั่งที่ใช้บ่อย

```bash
# เริ่ม development server
uvicorn src.main:app --reload --port 8000

# Ingest ข้อมูลใหม่ทั้งหมด
python scripts/run_ingestion.py

# Ingest เฉพาะแหล่งข้อมูลที่ระบุ
python scripts/run_ingestion.py --connector pubmed --limit 20

# Rebuild vector index หลังจากเปลี่ยน chunk size หรือ model
python scripts/rebuild_index.py

# รัน tests ทั้งหมด
python -m pytest tests/ -v

# ตรวจสอบ code style
ruff check src/ tests/

# จัดรูปแบบโค้ด
black src/ tests/

# ดูสถิติข้อมูลใน database
python -c "
from src.db.session import SessionLocal
from src.db.models import RecordORM
from sqlalchemy import func
with SessionLocal() as s:
    total = s.query(func.count(RecordORM.id)).scalar()
    print(f'Total records: {total}')
"
```

---

## 11. ปัญหาที่พบบ่อยและวิธีแก้ไข

| อาการ | สาเหตุที่เป็นไปได้ | วิธีแก้ไข |
|-------|-------------------|-----------|
| `ModuleNotFoundError: No module named 'fastapi'` | Virtual environment ไม่ได้ activate | รัน `source .venv/bin/activate` (Linux/macOS) หรือ `.venv\Scripts\Activate.ps1` (Windows) |
| `tiktoken unavailable (403)` | tiktoken ไม่สามารถดาวน์โหลด BPE file ได้ใน environment นี้ | **ไม่ต้องแก้ไข** — ระบบจะใช้ `_CharEstimator` fallback โดยอัตโนมัติ |
| BGE-M3 ดาวน์โหลดช้าหรือค้าง | ไฟล์ขนาด 2.3 GB ใช้เวลานานในครั้งแรก | รอจนดาวน์โหลดเสร็จ หรือตั้ง `HF_HOME=/path/to/large/drive` เพื่อเปลี่ยนที่เก็บ cache |
| `ChromaDB: collection not found` | Vector index ยังไม่ถูกสร้าง หรือถูกลบ | รัน `python scripts/rebuild_index.py` |
| Ingestion ได้ 0 records | API key ไม่ถูกต้อง หรือไม่มีการเชื่อมต่ออินเทอร์เน็ต | ตรวจสอบ `.env` และการเชื่อมต่อเครือข่าย |
| `test_rag_integration.py` ถูก skip | ไม่มี sentence-transformers หรือ chromadb | รัน `pip install sentence-transformers chromadb` |
| Server รัน แต่ทุก query ตอบว่า "insufficient context" | Database ว่างเปล่า ยังไม่ได้ ingest | รัน `python scripts/run_ingestion.py --limit 10` |
| `TYPHOON_API_KEY not set` | ไม่ได้ใส่ key ใน `.env` หรือ environment variable | แก้ไข `.env` และตรวจสอบว่า file อยู่ที่ root ของโปรเจกต์ |

---

## ขั้นตอนถัดไป

- **ทดลองรัน Colab:** ดูคู่มือ [`COLAB_GUIDE_TH.md`](./COLAB_GUIDE_TH.md) สำหรับการรันบน Google Colab ฟรี
- **Deploy ขึ้น production:** ดูคู่มือ [`DEPLOY_GUIDE_TH.md`](./DEPLOY_GUIDE_TH.md) สำหรับการ deploy บน Hugging Face Spaces
