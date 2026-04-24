# คู่มือการใช้งาน Google Colab — CV Intel RAG

**ระดับ:** มือใหม่ | **เวลาโดยประมาณ:** 15–20 นาที (ไม่รวมเวลา run notebook)

คู่มือฉบับนี้อธิบายขั้นตอนการรัน CV Intel RAG บน Google Colab แบบฟรี (T4 GPU) ทีละขั้นตอน เหมาะสำหรับผู้ที่ไม่มีเครื่องที่มี GPU หรือต้องการทดสอบระบบโดยไม่ติดตั้งบนเครื่องตัวเอง

---

## สารบัญ

1. [ภาพรวมและแนวคิด](#1-ภาพรวมและแนวคิด)
2. [สิ่งที่ต้องเตรียมก่อนเริ่ม](#2-สิ่งที่ต้องเตรียมก่อนเริ่ม)
3. [Notebook 1 — Ingest and Index](#3-notebook-1--ingest-and-index)
4. [Notebook 2 — Demo and Visualization](#4-notebook-2--demo-and-visualization)
5. [การใช้ข้อมูลซ้ำใน Session ใหม่](#5-การใช้ข้อมูลซ้ำใน-session-ใหม่)
6. [การอัปเดต Corpus รายสัปดาห์](#6-การอัปเดต-corpus-รายสัปดาห์)
7. [ปัญหาที่พบบ่อยและวิธีแก้ไข](#7-ปัญหาที่พบบ่อยและวิธีแก้ไข)

---

## 1. ภาพรวมและแนวคิด

### สถาปัตยกรรมของ 2 Notebooks

ระบบนี้แบ่งออกเป็น 2 notebooks ที่ทำงานร่วมกันผ่าน **Google Drive folder เดียวกัน**

```
Google Drive: MyDrive/cv-intel-rag/
├── data/
│   ├── cv_intel.db                  ← SQLite database (ผลลัพธ์จาก Notebook 1)
│   ├── chroma/                      ← ChromaDB vector store (ผลลัพธ์จาก Notebook 1)
│   └── cv-intel-rag-data.tar.gz     ← Archive สำหรับ deploy ขึ้น HF Spaces
└── logs/

Notebook 1: 01_ingest_and_index.ipynb
→ ดึงข้อมูลจาก 7 แหล่ง
→ แบ่ง text เป็น chunks
→ สร้าง BGE-M3 embeddings
→ บันทึกลง Drive
→ รัน 1 ครั้ง (ใช้เวลา ~10 นาที บน T4 GPU)

Notebook 2: 02_demo_visualization.ipynb
→ โหลดข้อมูลจาก Drive
→ แสดง visualizations
→ ทดสอบคำถามจริง
→ รันเมื่อไหรก็ได้ (ใช้เวลา ~1 นาที)
```

**เหตุผลที่แยกเป็น 2 notebooks:** การสร้าง embeddings ใช้เวลานานและต้อง GPU ทำเพียงครั้งเดียวก็เพียงพอ ส่วน notebook 2 โหลดข้อมูลที่สร้างแล้วจาก Drive ทำให้ demo รวดเร็ว

---

## 2. สิ่งที่ต้องเตรียมก่อนเริ่ม

### 2.1 บัญชีที่ต้องมี

| บัญชี | ฟรี | ลิงก์สมัคร |
|-------|-----|-----------|
| Google Account (สำหรับ Colab + Drive) | ✅ | [accounts.google.com](https://accounts.google.com) |
| Typhoon API Key | ✅ | [playground.opentyphoon.ai](https://playground.opentyphoon.ai/settings/api-key) |
| GitHub Account (สำหรับ clone repo) | ✅ | [github.com/join](https://github.com/join) |

### 2.2 การขอ Typhoon API Key

1. เปิดเบราว์เซอร์ไปที่ [playground.opentyphoon.ai/settings/api-key](https://playground.opentyphoon.ai/settings/api-key)
2. ล็อกอินด้วย Google Account
3. กดปุ่ม **Create new secret key**
4. ตั้งชื่อ key เช่น `colab-cv-intel`
5. คัดลอก key ที่ขึ้นต้นด้วย `sk-` และเก็บไว้ (จะนำไปใส่ใน Colab Secrets)

> **สำคัญ:** Key นี้จะแสดงเพียงครั้งเดียว หากปิดหน้าต่างโดยไม่คัดลอก ต้องสร้างใหม่

---

## 3. Notebook 1 — Ingest and Index

Notebook นี้ทำงานหลัก 3 อย่าง: ดึงข้อมูล → แบ่ง chunks → สร้าง embeddings ใช้เวลาประมาณ 10 นาทีบน T4 GPU และต้องรันเพียงครั้งเดียว

### ขั้นตอนที่ 3.1 — เปิด Notebook บน Colab

1. ไปที่ [colab.research.google.com](https://colab.research.google.com)
2. คลิกแถบ **GitHub** ในหน้าต่าง "Open notebook"
3. ใส่ URL: `https://github.com/siriponsri/cv-intel-rag`
4. เลือก `notebooks/01_ingest_and_index.ipynb`

**หรือ** upload ไฟล์ตรงจากเครื่อง:
1. คลิก **Upload** ในหน้าต่าง "Open notebook"
2. เลือกไฟล์ `notebooks/01_ingest_and_index.ipynb` จากโฟลเดอร์โปรเจกต์

### ขั้นตอนที่ 3.2 — เปิดใช้งาน T4 GPU

> **สำคัญมาก:** หากไม่เปิด GPU ขั้นตอน embedding จะช้ากว่าปกติประมาณ 10 เท่า

1. คลิกเมนู **Runtime** (แถบเมนูด้านบน)
2. เลือก **Change runtime type**
3. ในหัวข้อ **Hardware accelerator** เลือก **T4 GPU**
4. กดปุ่ม **Save**

**ตรวจสอบ:** มุมบนขวาของหน้าจะแสดง `T4` แทน `CPU`

### ขั้นตอนที่ 3.3 — ตั้งค่า Colab Secrets

Colab Secrets คือวิธีที่ปลอดภัยในการเก็บ API keys โดยไม่ให้ปรากฏใน notebook code

1. คลิกไอคอนรูป **กุญแจ** (🔑 Secrets) ที่แถบด้านซ้ายของ Colab
2. คลิก **+ Add new secret**
3. กรอกข้อมูล:
   - **Name:** `TYPHOON_API_KEY`  *(ต้องสะกดตรงทุกตัวอักษร)*
   - **Value:** API key ที่คัดลอกไว้จากขั้นตอน 2.2
4. กด toggle **Notebook access** ให้เป็นสีน้ำเงิน (เปิด)
5. กดปุ่ม **Save**

### ขั้นตอนที่ 3.4 — แก้ไข Repository URL (ถ้าจำเป็น)

ค้นหา cell ที่มีคำสั่ง `git clone` และตรวจสอบว่า URL ถูกต้อง:

```python
!git clone https://github.com/siriponsri/cv-intel-rag.git
```

หากคุณ fork repo มา ให้เปลี่ยน `siriponsri` เป็น GitHub username ของคุณ

### ขั้นตอนที่ 3.5 — รัน Notebook ทั้งหมด

1. คลิกเมนู **Runtime**
2. เลือก **Run all** (หรือกด `Ctrl+F9`)
3. ถ้า Colab แสดง popup ถามว่า "This notebook was not authored by Google" ให้กด **Run anyway**

**สิ่งที่จะเห็นระหว่างรัน:**

| ขั้นตอน | สิ่งที่จะเห็น | เวลาโดยประมาณ |
|---------|--------------|--------------|
| ติดตั้ง packages | แถบความคืบหน้าและ log ของ pip | 2–3 นาที |
| Mount Google Drive | popup ขอสิทธิ์ — กด **Connect to Google Drive** | 30 วินาที |
| Ingestion | log แสดงจำนวน records ต่อแหล่ง | 3–4 นาที |
| ดาวน์โหลด BGE-M3 | log แสดงการดาวน์โหลด 2.3 GB (ครั้งแรกเท่านั้น) | 3–5 นาที |
| Embedding | แถบ progress ต่อ batch | 2–3 นาที |
| บันทึกลง Drive | log `Saved to drive` | 30 วินาที |

### ขั้นตอนที่ 3.6 — ตรวจสอบผลลัพธ์

หลังจาก notebook รันเสร็จสมบูรณ์ ตรวจสอบใน Google Drive:

1. เปิด [drive.google.com](https://drive.google.com)
2. ไปที่โฟลเดอร์ `MyDrive/cv-intel-rag/data/`
3. ตรวจสอบว่ามีไฟล์และโฟลเดอร์ต่อไปนี้:
   - `cv_intel.db` (ขนาดประมาณ 3–10 MB)
   - โฟลเดอร์ `chroma/` (ขนาดประมาณ 20–100 MB)
   - `cv-intel-rag-data.tar.gz` (ไฟล์รวมสำหรับ deploy)

หากไฟล์เหล่านี้ปรากฏครบ Notebook 1 เสร็จสมบูรณ์แล้ว

---

## 4. Notebook 2 — Demo and Visualization

Notebook นี้โหลดข้อมูลจาก Drive และแสดงตัวอย่างการใช้งานระบบ ใช้เวลาเพียง 1–2 นาที

### ขั้นตอนที่ 4.1 — เปิด Notebook

ทำขั้นตอนเดียวกับ Notebook 1 แต่เลือกไฟล์ `notebooks/02_demo_visualization.ipynb`

> Notebook 2 สามารถเปิดใน Colab session ใหม่แยกจาก Notebook 1 ได้ทั้งหมด ข้อมูลจะโหลดจาก Drive

### ขั้นตอนที่ 4.2 — ตั้งค่า Runtime และ Secrets

ทำซ้ำขั้นตอน 3.2 และ 3.3 ทั้งหมด (T4 GPU + TYPHOON_API_KEY secret)

### ขั้นตอนที่ 4.3 — รัน Notebook ทั้งหมด

รัน **Runtime → Run all** เช่นเดิม

### ขั้นตอนที่ 4.4 — ผลลัพธ์ที่จะเห็นในแต่ละ Section

**Section 1 — สถิติ Corpus:**
แสดง bar chart จำนวน records ต่อแหล่งข้อมูล, pie chart สัดส่วน category และ timeline การ publish ข้อมูล

**Section 2 — RAG Pipeline Visualization:**
แสดง query ตัวอย่าง → embedding vector → bar chart ของ top-k chunks ที่ค้นพบพร้อมค่า similarity score

**Section 3 — ตัวอย่างคำถาม-คำตอบ:**
ทดสอบคำถาม 5 ข้อทั้งภาษาไทยและอังกฤษ คำตอบแต่ละข้อจะมี citation `[S1]`, `[S2]` พร้อมชื่อแหล่งอ้างอิง

**Section 4 — Latency Analysis:**
แสดง bar chart เวลาตอบของแต่ละ query เปรียบเทียบ retrieval time vs. LLM generation time

**Section 5 — Source × Category Heatmap:**
แสดง heatmap แสดงความครอบคลุมข้อมูลว่าแต่ละแหล่งมีข้อมูลประเภทไหนบ้าง

### ขั้นตอนที่ 4.5 — การปรับแต่ง Query สำหรับการนำเสนอ

หากต้องการเปลี่ยนคำถามใน Section 3 ให้แก้ไข cell ที่มี:

```python
DEMO_QUERIES = [
    "What are the cardiovascular benefits of SGLT2 inhibitors?",
    "ยา GLP-1 receptor agonist มีผลต่อไตอย่างไร",
    # เพิ่มคำถามของคุณที่นี่
]
```

> **เทคนิค:** ใส่คำถามที่เกี่ยวข้องกับงานของผู้รับฟังโดยตรง เพื่อให้เห็นว่าระบบตอบโจทย์ที่แท้จริงได้

---

## 5. การใช้ข้อมูลซ้ำใน Session ใหม่

Google Colab จะลบข้อมูลใน `/content/` ทุกครั้งที่ runtime ถูกปิด แต่ข้อมูลใน Google Drive ยังคงอยู่

**การเปิด Demo ครั้งต่อไป:**
1. เปิด Notebook 2 ใน Colab session ใหม่
2. ตั้งค่า Runtime type และ Secret ตามปกติ
3. รัน **Run all** — ระบบจะ mount Drive และโหลดข้อมูลที่มีอยู่แล้ว
4. **ไม่ต้องรัน Notebook 1 ซ้ำ** (เว้นแต่ต้องการ refresh corpus)

---

## 6. การอัปเดต Corpus รายสัปดาห์

หากต้องการให้ข้อมูลเป็นปัจจุบัน ควรรัน Notebook 1 ซ้ำสัปดาห์ละครั้ง

ระบบออกแบบให้รองรับการ upsert — ถ้ามี record ที่ source ID ซ้ำกัน ระบบจะ update แทนที่จะ insert ซ้ำ ดังนั้นจึงไม่มีข้อมูล duplicate

**ขั้นตอน:**
1. เปิด Notebook 1 ใน Colab
2. รัน Run all ตามปกติ
3. ระบบจะดึงเฉพาะข้อมูลใหม่ที่ publish หลังจากการ ingest ครั้งก่อน

**ตัวเลือกการทำ Automation:**
- **GitHub Actions Scheduled Workflow:** trigger ทุกวันจันทร์ผ่าน `schedule: cron`
- **Hugging Face Space:** ใส่ cron job ใน Dockerfile CMD สำหรับการ ingest อัตโนมัติ

---

## 7. ปัญหาที่พบบ่อยและวิธีแก้ไข

| อาการ | สาเหตุที่เป็นไปได้ | วิธีแก้ไข |
|-------|-------------------|-----------|
| `KeyError: 'TYPHOON_API_KEY'` | Secret ยังไม่ได้ตั้งค่า หรือ toggle Notebook access ยังปิดอยู่ | ไปที่ไอคอน 🔑 Secrets และตรวจสอบว่า toggle เปิดแล้ว |
| Notebook 2 error `DB file not found` | Notebook 1 ยังไม่รันจนเสร็จ หรือ path Drive ไม่ถูกต้อง | ตรวจสอบว่ามีไฟล์ `cv_intel.db` ใน `MyDrive/cv-intel-rag/data/` |
| BGE-M3 download หยุดค้าง | การเชื่อมต่อหลุดระหว่างดาวน์โหลด 2.3 GB | คลิก **Runtime → Disconnect and delete runtime** แล้วเชื่อมต่อใหม่และรันใหม่ |
| `You've used all your GPU quota` | Colab free tier จำกัด GPU usage ต่อวัน | รอ 12–24 ชั่วโมง หรือพิจารณา [Colab Pro](https://colab.research.google.com/signup) |
| Google Drive mount ล้มเหลว | Session หมดอายุ หรือ popup ถูก block | Disconnect runtime แล้ว Connect ใหม่ จากนั้นรัน cell `drive.mount(...)` แยก |
| Ingestion ได้ 0 records จาก PubMed | IP ของ Colab อาจถูก rate limit ชั่วคราว | รออีก 5 นาทีแล้วรัน cell ingestion ซ้ำ |
| คำตอบใน Notebook 2 ว่างเปล่า | Typhoon API timeout | ตรวจสอบ API key และลองรัน cell นั้นซ้ำด้วย `Ctrl+Enter` |

---

## ขั้นตอนถัดไป

พร้อม deploy ระบบขึ้น production แล้วหรือยัง? ดูคู่มือ [`DEPLOY_GUIDE_TH.md`](./DEPLOY_GUIDE_TH.md) สำหรับขั้นตอนการ deploy บน Hugging Face Spaces (ฟรี, มี public URL)
