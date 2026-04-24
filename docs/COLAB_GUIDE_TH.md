# 📓 Colab Guide (ภาษาไทย)

เดินตามขั้นตอนนี้เพื่อรัน CV Intel RAG บน Google Colab ฟรี (T4 GPU)

**แนวคิด:** 2 notebooks แยกกัน แต่ใช้ **Google Drive folder เดียวกัน** เป็นที่เก็บ index

```
MyDrive/cv-intel-rag/            ← shared folder
├── data/
│   ├── cv_intel.db              ← SQLite (ingest output)
│   ├── chroma/                  ← ChromaDB (embed output)
│   └── cv-intel-rag-data.tar.gz ← สำหรับ upload ขึ้น HF Space
└── logs/

01_ingest_and_index.ipynb  ─► เขียนลง folder นี้ (รัน 1 ครั้ง ~10 นาที)
02_demo_visualization.ipynb ─► อ่านจาก folder นี้ (รันเมื่อไหร่ก็ได้ ~1 นาที)
```

---

## 🔑 Step 0 — เตรียม Typhoon API key

1. ไปที่ [playground.opentyphoon.ai/settings/api-key](https://playground.opentyphoon.ai/settings/api-key) → สร้าง API key
2. **จำไว้** — จะใช้ใน Colab Secrets ทั้ง 2 notebooks

---

## 📘 Notebook 1 — Ingest & Index (รัน 1 ครั้งพอ)

### 1.1 เปิด notebook

1. Upload `notebooks/01_ingest_and_index.ipynb` เข้า Google Colab
2. **Runtime → Change runtime type → T4 GPU** (สำคัญมาก ถ้าไม่เปิด GPU ช้ากว่า 10 เท่า)

### 1.2 ตั้ง Colab Secret

1. คลิกไอคอน 🔑 Secrets ที่แถบซ้าย
2. **+ Add new secret**
3. Name: `TYPHOON_API_KEY` · Value: key ที่ได้จาก Step 0
4. Toggle **Notebook access** ON

### 1.3 แก้ repo URL

ในเซลล์ **Step 4** เปลี่ยน `YOUR_USERNAME` เป็น GitHub username ของคุณ:

```python
!git clone https://github.com/YOUR_USERNAME/cv-intel-rag.git
```

### 1.4 รันทั้งหมด

**Runtime → Run all**

จะเห็น:
- ⏱ Step 6 (ingestion) ~3 นาที — bar chart แสดง records ต่อ source
- ⏱ Step 7 (embedding) ~7 นาที — ครั้งแรกจะดาวน์โหลด BGE-M3 (2.3 GB)
- 📦 Step 9 สร้าง `data/cv-intel-rag-data.tar.gz` ใน Drive

### 1.5 ยืนยัน

เปิด Google Drive → `MyDrive/cv-intel-rag/data/` ต้องเห็น:
- `cv_intel.db` (~5 MB)
- `chroma/` folder
- `cv-intel-rag-data.tar.gz` (~20-50 MB ขึ้นกับจำนวน records)

---

## 📗 Notebook 2 — Demo & Visualization (รันเมื่อไหร่ก็ได้)

**เป้าหมาย:** นำเสนอลูกค้า — โหลดจาก Drive → show pipeline step-by-step ไม่ต้อง re-embed

### 2.1 เปิด notebook

1. Upload `notebooks/02_demo_visualization.ipynb` เข้า Colab ใหม่ (แยกจาก notebook 1 ได้)
2. **Runtime → T4 GPU** (หรือ CPU ก็ได้ถ้าไม่รีบ)
3. เพิ่ม `TYPHOON_API_KEY` ใน Secrets เหมือนเดิม
4. แก้ repo URL เหมือนเดิม

### 2.2 รันทั้งหมด

Runtime → Run all — ใช้เวลา ~60 วินาที (latency ขึ้นกับ Typhoon API)

### 2.3 สิ่งที่ลูกค้าจะเห็น

| Step | หน้าจอ | Sales angle |
|---|---|---|
| 2 | Bar + Pie + Timeline charts | "นี่คือ corpus ที่เรารวบรวมไว้ — 7 sources, ครอบคลุม X records ถึง..." |
| 3 | Query → embedding → top-k chunks bar chart | "พอลูกค้าถาม ระบบ embed คำถามเป็น vector 1024 มิติ แล้วหา chunks คล้ายกัน top 5" |
| 4 | 5 คำถามจริง (ไทย+อังกฤษ) + [S#] citations | "ดูไหม คำตอบอ้างอิง 3 sources พร้อมลิงก์กลับไปดูต้นฉบับ" |
| 5 | Latency bar chart | "เฉลี่ย ~3-5 วินาที/query — เร็วพอสำหรับ interactive use" |
| 6 | Source × Category heatmap | "ไม่ได้พึ่ง PubMed อย่างเดียว — triangulate ทุก source" |

### 2.4 Tips ขายงาน

- **Save as PDF** ตอนจบ (File → Print → Save as PDF) — ใช้เป็น leave-behind หลังประชุม
- **ใส่คำถามของลูกค้าเอง** ในเซลล์ Step 3/4 ก่อนรัน — ลูกค้าจะเห็นว่าระบบตอบโจทย์ **ของเขา** ได้จริง
- **สลับ query ไทย/อังกฤษ** — BGE-M3 ทำงานทั้งคู่โดยไม่ต้อง config

---

## 💾 ใช้ data ซ้ำใน session ใหม่

Colab ปิด runtime → data ใน `/content/` หาย แต่ Drive ยังอยู่

- Notebook 1: ไม่ต้องรันซ้ำ ถ้า corpus เดิมใช้ได้ (อยากรีเฟรชค่อยรันใหม่)
- Notebook 2: เปิด demo ใหม่ → mount Drive → ใช้ต่อได้ทันที

---

## 🔄 Refresh corpus รายสัปดาห์

รัน notebook 1 ซ้ำทุกสัปดาห์ → `run_ingestion.py` upsert ได้ (ไม่ duplicate) → index อัพเดตอัตโนมัติ

ถ้าอยากให้รันอัตโนมัติ:
- Cron-style ใน HF Space (Dockerfile CMD + cron job)
- หรือ GitHub Actions scheduled workflow

---

## ❗ Troubleshooting

| ปัญหา | วิธีแก้ |
|---|---|
| "TYPHOON_API_KEY not set" | ตรวจ Secrets ว่า toggle **Notebook access** ON แล้วหรือยัง |
| Notebook 2 error "DB not found" | รัน notebook 1 จนจบก่อน + ตรวจ path `MyDrive/cv-intel-rag/data/` |
| BGE-M3 download hang | ยกเลิก → Runtime → Disconnect and delete → Connect ใหม่ → รันใหม่ |
| GPU quota exceeded | Colab free tier จำกัด GPU ต่อวัน รอ 12-24 ชม. หรือซื้อ Colab Pro |
| Drive mount ไม่ได้ | Runtime → Disconnect → Connect → Run cell `drive.mount(...)` ใหม่ |

---

## ขั้นต่อไป

พร้อม deploy ขึ้น production แล้ว? → ดู [`DEPLOY_GUIDE_TH.md`](./DEPLOY_GUIDE_TH.md)
