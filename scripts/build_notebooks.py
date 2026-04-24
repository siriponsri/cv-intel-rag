"""
Build the two Colab notebooks for the sales/demo workflow:

  01_ingest_and_index.ipynb   — runs once, populates Drive with the index
  02_demo_visualization.ipynb — runs anytime, shows the RAG pipeline step-by-step

Both share a Google Drive folder:   /content/drive/MyDrive/cv-intel-rag/
                                      ├── data/
                                      │   ├── cv_intel.db
                                      │   ├── chroma/
                                      │   └── cv-intel-rag-data.tar.gz  (for HF upload)
                                      └── logs/

Run:   python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def cell_md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in "\n".join(lines).split("\n")][:-1] or [""],
    }


def cell_code(*lines: str) -> dict:
    src = "\n".join(lines)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [ln + "\n" for ln in src.split("\n")][:-1] or [""],
    }


def save_nb(cells: list[dict], path: Path) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"✓ wrote {path}  ({len(cells)} cells)")


# ═════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — INGEST + INDEX  (run once, ~10 min on T4)
# ═════════════════════════════════════════════════════════════════════════

def build_ingest_notebook() -> list[dict]:
    cells = []

    cells.append(cell_md(
        "# 🫀 CV Intel RAG — Notebook 1: Ingest & Index",
        "",
        "**Run this once.** It fetches CV/DM/CKD data from 7 sources, embeds with",
        "BGE-M3 on the free T4 GPU, and saves the index to your Google Drive so",
        "Notebook 2 can start demos in seconds.",
        "",
        "**Time:** ~10 min total (3 min fetch + 7 min embed)",
        "",
        "### Pipeline",
        "",
        "```",
        "PubMed ───┐",
        "CT.gov ───┤",
        "openFDA ──┼──► SQLite ──► chunker ──► BGE-M3 (T4) ──► ChromaDB ──► Drive",
        "4 RSS ────┘                                                          │",
        "                                                                 Notebook 2",
        "```",
        "",
        "### Shared Drive layout",
        "",
        "```",
        "MyDrive/cv-intel-rag/",
        "├── data/",
        "│   ├── cv_intel.db        ← SQLite records",
        "│   ├── chroma/            ← ChromaDB persistent files",
        "│   └── cv-intel-rag-data.tar.gz  ← to upload to HF Space",
        "└── logs/",
        "```",
    ))

    cells.append(cell_md("## Step 1 — Install dependencies"))
    cells.append(cell_code(
        "!pip install -q \\",
        "  fastapi uvicorn pydantic pydantic-settings sqlalchemy \\",
        "  httpx beautifulsoup4 feedparser openai \\",
        "  chromadb sentence-transformers rank-bm25 tiktoken \\",
        "  pythainlp pdfplumber pymupdf python-dotenv tqdm matplotlib",
    ))

    cells.append(cell_md(
        "## Step 2 — Mount Google Drive",
        "",
        "This creates the shared project folder that Notebook 2 will read from.",
    ))
    cells.append(cell_code(
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "",
        "from pathlib import Path",
        "PROJECT_DIR = Path('/content/drive/MyDrive/cv-intel-rag')",
        "(PROJECT_DIR / 'data').mkdir(parents=True, exist_ok=True)",
        "(PROJECT_DIR / 'data' / 'raw').mkdir(exist_ok=True)",
        "(PROJECT_DIR / 'logs').mkdir(exist_ok=True)",
        "print('✓ Drive folder:', PROJECT_DIR)",
    ))

    cells.append(cell_md(
        "## Step 3 — Typhoon API key (Colab Secrets)",
        "",
        "1. Go to left sidebar → 🔑 **Secrets** → **+ Add new secret**",
        "2. Name: `TYPHOON_API_KEY` · Value: your key from",
        "   [playground.opentyphoon.ai/settings/api-key](https://playground.opentyphoon.ai/settings/api-key)",
        "3. Toggle **Notebook access** ON",
    ))
    cells.append(cell_code(
        "import os",
        "from google.colab import userdata",
        "",
        "try:",
        "    os.environ['TYPHOON_API_KEY'] = userdata.get('TYPHOON_API_KEY')",
        "    print('✓ Typhoon key loaded (len=%d)' % len(os.environ['TYPHOON_API_KEY']))",
        "except Exception as e:",
        "    print('⚠ TYPHOON_API_KEY not set — LLM calls will fail. (%s)' % e)",
    ))

    cells.append(cell_md(
        "## Step 4 — Clone the repo",
        "",
        "Replace `YOUR_USERNAME` with your GitHub username.",
    ))
    cells.append(cell_code(
        "%cd /content",
        "![ -d cv-intel-rag ] && rm -rf cv-intel-rag",
        "!git clone https://github.com/YOUR_USERNAME/cv-intel-rag.git",
        "%cd /content/cv-intel-rag",
        "!ls",
    ))

    cells.append(cell_md(
        "## Step 5 — Configure environment",
        "",
        "Point the DB and ChromaDB to the Drive folder so results persist.",
    ))
    cells.append(cell_code(
        "DB_PATH     = PROJECT_DIR / 'data' / 'cv_intel.db'",
        "CHROMA_PATH = PROJECT_DIR / 'data' / 'chroma'",
        "",
        "env = f'''APP_ENV=colab",
        "DATABASE_URL=sqlite:///{DB_PATH}",
        "CHROMA_PATH={CHROMA_PATH}",
        "DEFAULT_LLM_PROVIDER=typhoon_api",
        "TYPHOON_BASE_URL=https://api.opentyphoon.ai/v1",
        "TYPHOON_MODEL=typhoon-v2.5-30b-a3b-instruct",
        "EMBED_MODEL_NAME=BAAI/bge-m3",
        "EMBED_DEVICE=cuda",
        "EMBED_BATCH_SIZE=32",
        "'''",
        "open('.env', 'w').write(env)",
        "print(env)",
    ))

    cells.append(cell_md(
        "## Step 6 — Initialize DB + run ingestion",
        "",
        "Fetches ~30 records from each of 7 connectors (~200 records total).",
        "Adjust `LIMIT_PER_SOURCE` for smaller/larger corpora.",
    ))
    cells.append(cell_code(
        "LIMIT_PER_SOURCE = 30",
        "",
        "!python scripts/init_db.py",
        "",
        "import sys, time, subprocess",
        "t0 = time.time()",
        "result = subprocess.run(",
        "    ['python', 'scripts/run_ingestion.py', '--limit', str(LIMIT_PER_SOURCE)],",
        "    capture_output=True, text=True,",
        ")",
        "print(result.stdout[-4000:])",
        "if result.returncode != 0:",
        "    print('STDERR:', result.stderr[-2000:])",
        "print(f'\\n⏱ ingest took {time.time()-t0:.0f}s')",
    ))

    cells.append(cell_md(
        "### 📊 Ingestion summary chart",
        "",
        "Quick visual so you know every source pulled something (and none failed silently).",
    ))
    cells.append(cell_code(
        "import sqlite3, matplotlib.pyplot as plt",
        "",
        "conn = sqlite3.connect(DB_PATH)",
        "rows = conn.execute('SELECT source_name, COUNT(*) FROM records GROUP BY source_name ORDER BY 2 DESC').fetchall()",
        "conn.close()",
        "",
        "sources = [r[0] for r in rows]",
        "counts  = [r[1] for r in rows]",
        "",
        "fig, ax = plt.subplots(figsize=(9, 4))",
        "bars = ax.barh(sources, counts, color='#c0392b')",
        "ax.set_xlabel('records')",
        "ax.set_title(f'Ingested records by source  (total={sum(counts)})')",
        "for b, c in zip(bars, counts):",
        "    ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2, str(c), va='center')",
        "ax.invert_yaxis(); plt.tight_layout(); plt.show()",
    ))

    cells.append(cell_md(
        "## Step 7 — Build the vector index with BGE-M3",
        "",
        "Chunks each record's raw_text, embeds with BGE-M3 on the T4 GPU, stores",
        "in ChromaDB. First run downloads ~2.3 GB of model weights.",
    ))
    cells.append(cell_code(
        "import time",
        "t0 = time.time()",
        "!python scripts/rebuild_index.py",
        "print(f'\\n⏱ embedding took {time.time()-t0:.0f}s')",
    ))

    cells.append(cell_md(
        "### 📊 Index stats",
    ))
    cells.append(cell_code(
        "import chromadb",
        "client = chromadb.PersistentClient(path=str(CHROMA_PATH))",
        "coll = client.get_or_create_collection(name='cv_intel_chunks')",
        "print(f'✓ ChromaDB contains {coll.count()} chunks')",
        "",
        "# category distribution",
        "import sqlite3",
        "conn = sqlite3.connect(DB_PATH)",
        "cat_rows = conn.execute('SELECT category, COUNT(*) FROM records GROUP BY category').fetchall()",
        "conn.close()",
        "",
        "fig, ax = plt.subplots(figsize=(7, 4))",
        "cats = [r[0] for r in cat_rows]",
        "nums = [r[1] for r in cat_rows]",
        "ax.pie(nums, labels=cats, autopct='%1.0f%%', colors=plt.cm.Reds_r(range(40, 40+len(cats)*30, 30)))",
        "ax.set_title('Records by category'); plt.tight_layout(); plt.show()",
    ))

    cells.append(cell_md(
        "## Step 8 — Sanity check: one retrieval query",
        "",
        "Confirm the index actually retrieves sensible results before we package it.",
    ))
    cells.append(cell_code(
        "from src.rag.retriever import HybridRetriever",
        "",
        "retriever = HybridRetriever(top_k=5)",
        "query = 'SGLT2 inhibitors for heart failure with preserved ejection fraction'",
        "hits = retriever.retrieve(query)",
        "",
        "print(f'Query: {query}\\n')",
        "for i, h in enumerate(hits, 1):",
        "    print(f'[{i}] score={h.score:.3f}  source={h.metadata.get(\"source_name\")}')",
        "    print(f'    title: {h.metadata.get(\"title\",\"\")[:100]}')",
        "    print(f'    text:  {h.text[:160]}...')",
        "    print()",
    ))

    cells.append(cell_md(
        "## Step 9 — Package data/ for Hugging Face Space upload",
        "",
        "Creates a tar.gz archive of the index. You'll commit this to your HF Space",
        "repo (via git-lfs) so the deployed API starts with the same data you built here.",
    ))
    cells.append(cell_code(
        "import tarfile",
        "",
        "TAR_PATH = PROJECT_DIR / 'data' / 'cv-intel-rag-data.tar.gz'",
        "with tarfile.open(TAR_PATH, 'w:gz') as tar:",
        "    tar.add(DB_PATH, arcname='data/cv_intel.db')",
        "    tar.add(CHROMA_PATH, arcname='data/chroma')",
        "",
        "size_mb = TAR_PATH.stat().st_size / 1024 / 1024",
        "print(f'✓ archive: {TAR_PATH}  ({size_mb:.1f} MB)')",
        "print('\\nNext: see docs/DEPLOY_GUIDE_TH.md for how to commit this to HF Space')",
    ))

    cells.append(cell_md(
        "## 🎉 Done — now open `02_demo_visualization.ipynb`",
        "",
        "The demo notebook loads from this same Drive folder and shows the full",
        "RAG pipeline with visualizations — perfect for a 5-minute sales demo.",
    ))

    return cells


# ═════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — DEMO + VISUALIZATION  (fast, run anytime)
# ═════════════════════════════════════════════════════════════════════════

def build_demo_notebook() -> list[dict]:
    cells = []

    cells.append(cell_md(
        "# 🫀 CV Intel RAG — Notebook 2: Demo & Visualization",
        "",
        "**For sales demos.** Loads the pre-built index from Google Drive and walks",
        "through the RAG pipeline step-by-step with charts and timing breakdowns.",
        "",
        "**Time to run all:** ~60 seconds (depends on Typhoon API latency)",
        "",
        "> Prerequisite: you must have run `01_ingest_and_index.ipynb` first.",
        "",
        "### What this notebook shows",
        "",
        "1. 📈 Corpus overview — sources, categories, dates",
        "2. 🔎 Live retrieval — query → embedding → top-k chunks with scores",
        "3. 💬 5 demo questions — full RAG answers with [S#] citations",
        "4. ⏱ Latency breakdown — embed / retrieve / LLM",
        "5. 🌍 Coverage heatmap — source × category",
    ))

    cells.append(cell_md("## Step 1 — Setup (loads from shared Drive folder)"))
    cells.append(cell_code(
        "!pip install -q \\",
        "  pydantic pydantic-settings sqlalchemy httpx openai \\",
        "  chromadb sentence-transformers rank-bm25 tiktoken \\",
        "  python-dotenv matplotlib seaborn",
    ))
    cells.append(cell_code(
        "from google.colab import drive, userdata",
        "import os",
        "drive.mount('/content/drive')",
        "os.environ['TYPHOON_API_KEY'] = userdata.get('TYPHOON_API_KEY')",
        "",
        "from pathlib import Path",
        "PROJECT_DIR = Path('/content/drive/MyDrive/cv-intel-rag')",
        "DB_PATH     = PROJECT_DIR / 'data' / 'cv_intel.db'",
        "CHROMA_PATH = PROJECT_DIR / 'data' / 'chroma'",
        "assert DB_PATH.exists(),    f'❌ DB not found at {DB_PATH}  — run notebook 1 first'",
        "assert CHROMA_PATH.exists(), f'❌ Chroma not found at {CHROMA_PATH}'",
        "print('✓ Index loaded from Drive')",
    ))
    cells.append(cell_code(
        "%cd /content",
        "![ -d cv-intel-rag ] && rm -rf cv-intel-rag",
        "!git clone -q https://github.com/YOUR_USERNAME/cv-intel-rag.git",
        "%cd /content/cv-intel-rag",
        "",
        "env = f'''DATABASE_URL=sqlite:///{DB_PATH}",
        "CHROMA_PATH={CHROMA_PATH}",
        "DEFAULT_LLM_PROVIDER=typhoon_api",
        "TYPHOON_BASE_URL=https://api.opentyphoon.ai/v1",
        "TYPHOON_MODEL=typhoon-v2.5-30b-a3b-instruct",
        "EMBED_MODEL_NAME=BAAI/bge-m3",
        "EMBED_DEVICE=cuda",
        "'''",
        "open('.env', 'w').write(env)",
    ))

    cells.append(cell_md(
        "## Step 2 — 📈 Corpus overview",
        "",
        "What's actually in the index? (This chart goes in the sales deck.)",
    ))
    cells.append(cell_code(
        "import sqlite3, matplotlib.pyplot as plt, seaborn as sns",
        "sns.set_style('whitegrid')",
        "",
        "conn = sqlite3.connect(DB_PATH)",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4))",
        "",
        "# By source",
        "rows = conn.execute('SELECT source_name, COUNT(*) c FROM records GROUP BY 1 ORDER BY c DESC').fetchall()",
        "axes[0].barh([r[0] for r in rows], [r[1] for r in rows], color='#c0392b')",
        "axes[0].set_title(f'Records by source  (Σ={sum(r[1] for r in rows)})')",
        "axes[0].invert_yaxis()",
        "",
        "# By category",
        "rows = conn.execute('SELECT category, COUNT(*) c FROM records GROUP BY 1').fetchall()",
        "axes[1].pie([r[1] for r in rows], labels=[r[0] for r in rows], autopct='%1.0f%%',",
        "            colors=plt.cm.Reds_r([0.2, 0.4, 0.55, 0.7, 0.85][:len(rows)]))",
        "axes[1].set_title('Records by category')",
        "",
        "# Over time",
        "rows = conn.execute(\"\"\"SELECT strftime('%Y-%m', published_date) m, COUNT(*) c",
        "                        FROM records WHERE published_date IS NOT NULL",
        "                        GROUP BY 1 ORDER BY 1\"\"\").fetchall()",
        "axes[2].plot([r[0] for r in rows], [r[1] for r in rows], marker='o', color='#c0392b')",
        "axes[2].set_title('Records by month'); axes[2].tick_params(axis='x', rotation=45)",
        "",
        "conn.close(); plt.tight_layout(); plt.show()",
    ))

    cells.append(cell_md(
        "## Step 3 — 🔎 Retrieval visualization",
        "",
        "We'll follow ONE query through the pipeline and show exactly what happens",
        "at each stage. This is the demo moment to pause and explain to the customer.",
    ))
    cells.append(cell_code(
        "QUERY = 'ยา SGLT2 ลดความเสี่ยงหัวใจล้มเหลวได้อย่างไร'   # Thai query",
        "# QUERY = 'What are recent FDA safety alerts for statins?'",
        "",
        "from src.rag.embedder import get_embedder",
        "from src.rag.retriever import HybridRetriever",
        "import time",
        "",
        "embedder = get_embedder()",
        "",
        "# ─ Stage 1: embed the query ────────────────────────",
        "t0 = time.time()",
        "vec = embedder.encode_single(QUERY)",
        "t_embed = time.time() - t0",
        "",
        "print(f'Query:     {QUERY}')",
        "print(f'Embedding: dim={len(vec)}  time={t_embed*1000:.0f} ms')",
        "print(f'           first 8 dims: {[round(v,3) for v in vec[:8]]}...')",
    ))
    cells.append(cell_code(
        "# ─ Stage 2: hybrid retrieve ──────────────────────────",
        "retriever = HybridRetriever(top_k=5)",
        "t0 = time.time()",
        "hits = retriever.retrieve(QUERY)",
        "t_retrieve = time.time() - t0",
        "",
        "print(f'Retrieved {len(hits)} chunks in {t_retrieve*1000:.0f} ms\\n')",
        "",
        "# score breakdown chart",
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots(figsize=(9, 3))",
        "titles = [(h.metadata.get('title','')[:55]+'…') if len(h.metadata.get('title',''))>55 else h.metadata.get('title','(no title)') for h in hits]",
        "scores = [h.score for h in hits]",
        "ax.barh(titles, scores, color='#c0392b')",
        "ax.invert_yaxis(); ax.set_xlabel('hybrid score (dense 0.6 + BM25 0.4)')",
        "ax.set_title(f'Top-{len(hits)} retrieved chunks for this query')",
        "plt.tight_layout(); plt.show()",
        "",
        "for i, h in enumerate(hits, 1):",
        "    print(f'[S{i}] {h.metadata.get(\"source_name\",\"?\")}  score={h.score:.3f}')",
        "    print(f'     {h.text[:200]}...\\n')",
    ))

    cells.append(cell_md(
        "## Step 4 — 💬 Full RAG answers (5 demo questions)",
        "",
        "This is the money shot. Each question goes through: embed → retrieve → Typhoon → [S#] citations.",
    ))
    cells.append(cell_code(
        "from src.agent.rag_agent import RAGAgent",
        "",
        "agent = RAGAgent()",
        "",
        "DEMO_QUERIES = [",
        "    'ยา SGLT2 ลดความเสี่ยงหัวใจล้มเหลวในผู้ป่วยเบาหวานได้อย่างไร',",
        "    'What are the latest FDA safety alerts for cardiovascular drugs?',",
        "    'เปรียบเทียบ empagliflozin กับ dapagliflozin ในผู้ป่วย CKD',",
        "    'What phase 3 trials are running for GLP-1 agonists in heart failure?',",
        "    'ผลข้างเคียงที่พบบ่อยของ statins ตามรายงานล่าสุดคืออะไร',",
        "]",
        "",
        "results = []",
        "for q in DEMO_QUERIES:",
        "    print('═'*90); print(f'❓ {q}\\n')",
        "    t0 = time.time()",
        "    resp = agent.answer(q)",
        "    dt = time.time() - t0",
        "    print(f'🤖 {resp.answer}\\n')",
        "    print(f'📎 Citations: {len(resp.citations)} sources · ⏱ {dt:.1f}s\\n')",
        "    results.append({'query': q, 'latency': dt, 'n_citations': len(resp.citations)})",
    ))

    cells.append(cell_md(
        "## Step 5 — ⏱ Latency breakdown",
        "",
        "Customers always ask about speed. Here's honest numbers.",
    ))
    cells.append(cell_code(
        "import matplotlib.pyplot as plt",
        "",
        "fig, ax = plt.subplots(figsize=(10, 3.5))",
        "labels = [f'Q{i+1}' for i in range(len(results))]",
        "latencies = [r['latency'] for r in results]",
        "bars = ax.bar(labels, latencies, color='#c0392b')",
        "ax.axhline(sum(latencies)/len(latencies), ls='--', color='gray', label=f'mean={sum(latencies)/len(latencies):.1f}s')",
        "ax.set_ylabel('seconds'); ax.set_title('End-to-end RAG latency per query'); ax.legend()",
        "for b, v in zip(bars, latencies):",
        "    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{v:.1f}s', ha='center')",
        "plt.tight_layout(); plt.show()",
    ))

    cells.append(cell_md(
        "## Step 6 — 🌍 Coverage heatmap",
        "",
        "Which sources cover which categories? Useful to show the customer that",
        "the system isn't just \"PubMed only\" — it triangulates across 7 sources.",
    ))
    cells.append(cell_code(
        "import pandas as pd, seaborn as sns, sqlite3",
        "",
        "conn = sqlite3.connect(DB_PATH)",
        "df = pd.read_sql('SELECT source_name, category FROM records', conn)",
        "conn.close()",
        "",
        "pivot = df.groupby(['source_name','category']).size().unstack(fill_value=0)",
        "",
        "fig, ax = plt.subplots(figsize=(10, 4.5))",
        "sns.heatmap(pivot, annot=True, fmt='d', cmap='Reds', cbar_kws={'label':'records'}, ax=ax)",
        "ax.set_title('Source × Category coverage')",
        "plt.tight_layout(); plt.show()",
    ))

    cells.append(cell_md(
        "## 🎉 Done",
        "",
        "You've just shown the full RAG pipeline. Next steps for the deal:",
        "",
        "- Show the live [HF Space demo](https://huggingface.co/spaces/YOUR_USERNAME/cv-intel-rag)",
        "- Share this notebook (outputs visible) as a PDF leave-behind",
        "- Discuss: custom domain? daily auto-refresh? internal sources (EDC, EMR)?",
    ))

    return cells


# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    nb_dir = root / "notebooks"
    save_nb(build_ingest_notebook(), nb_dir / "01_ingest_and_index.ipynb")
    save_nb(build_demo_notebook(),   nb_dir / "02_demo_visualization.ipynb")
