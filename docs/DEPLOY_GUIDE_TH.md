# คู่มือการ Deploy — Hugging Face Spaces

**ระดับ:** มือใหม่ถึงระดับกลาง | **เวลาโดยประมาณ:** 30–45 นาที

คู่มือฉบับนี้อธิบายขั้นตอนการ deploy ระบบ CV Intel RAG ขึ้น Hugging Face Spaces อย่างละเอียดทีละขั้นตอน หลังจากทำตามคู่มือนี้จะได้ public URL ที่ใช้งานได้จริง

---

## สารบัญ

1. [ภาพรวมและสถาปัตยกรรม](#1-ภาพรวมและสถาปัตยกรรม)
2. [สิ่งที่ต้องเตรียมก่อนเริ่ม](#2-สิ่งที่ต้องเตรียมก่อนเริ่ม)
3. [ขั้นตอนที่ 1 — เตรียม Pre-built Index](#3-ขั้นตอนที่-1--เตรียม-pre-built-index)
4. [ขั้นตอนที่ 2 — สร้าง Hugging Face Space](#4-ขั้นตอนที่-2--สร้าง-hugging-face-space)
5. [ขั้นตอนที่ 3 — ตั้งค่า Secrets](#5-ขั้นตอนที่-3--ตั้งค่า-secrets)
6. [ขั้นตอนที่ 4 — Push โค้ดและ Index ขึ้น Space](#6-ขั้นตอนที่-4--push-โค้ดและ-index-ขึ้น-space)
7. [ขั้นตอนที่ 5 — ติดตาม Build Log](#7-ขั้นตอนที่-5--ติดตาม-build-log)
8. [ขั้นตอนที่ 6 — ทดสอบการทำงาน](#8-ขั้นตอนที่-6--ทดสอบการทำงาน)
9. [การตั้งค่า Auto-Deploy จาก GitHub](#9-การตั้งค่า-auto-deploy-จาก-github)
10. [ข้อจำกัดของ HF Spaces Free Tier](#10-ข้อจำกัดของ-hf-spaces-free-tier)
11. [การอัปเดต Index ภายหลัง](#11-การอัปเดต-index-ภายหลัง)
12. [ปัญหาที่พบบ่อยและวิธีแก้ไข](#12-ปัญหาที่พบบ่อยและวิธีแก้ไข)

---

## 1. ภาพรวมและสถาปัตยกรรม

### Flow การ Deploy

```
[Google Colab]                 [GitHub]                [HF Spaces]
Notebook 1 รัน     →    git push โค้ด + data    →    Docker build
  ↓                                                       ↓
สร้าง data/                                         uvicorn start
  cv_intel.db                                            ↓
  chroma/                                         พร้อมให้ใช้งาน
  *.tar.gz                               https://huggingface.co/spaces/
                                          <username>/cv-intel-rag
```

### สิ่งที่เกิดขึ้นเมื่อ HF Space Build

1. HF clone repository ของคุณ
2. รัน `docker build` จาก `Dockerfile` ที่ root
3. Dockerfile ติดตั้ง Python packages ทั้งหมด
4. ถ้ามีไฟล์ `data/` อยู่ใน repo — ระบบจะโหลด pre-built index
5. รัน `uvicorn src.main:app --host 0.0.0.0 --port 7860`
6. Space พร้อมให้เข้าใช้งานผ่าน public URL

---

## 2. สิ่งที่ต้องเตรียมก่อนเริ่ม

### บัญชีและ Access ที่ต้องมี

| รายการ | ฟรี | วิธีสมัคร/ขอ |
|--------|-----|-------------|
| GitHub Account | ✅ | [github.com/join](https://github.com/join) |
| Hugging Face Account | ✅ | [huggingface.co/join](https://huggingface.co/join) |
| HF Access Token (Write) | ✅ | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Typhoon API Key | ✅ | [playground.opentyphoon.ai](https://playground.opentyphoon.ai/settings/api-key) |

### การสร้าง Hugging Face Access Token

1. ล็อกอินที่ [huggingface.co](https://huggingface.co)
2. คลิกรูปโปรไฟล์มุมบนขวา → **Settings**
3. เลือกเมนู **Access Tokens** ด้านซ้าย
4. กดปุ่ม **New token**
5. ตั้งชื่อ เช่น `cv-intel-deploy`
6. เลือก **Role: Write**
7. กด **Generate a token**
8. คัดลอก token ที่ขึ้นต้นด้วย `hf_` และเก็บไว้

### ข้อกำหนดเบื้องต้นอื่น ๆ

- ต้องรัน Notebook 1 บน Colab จนเสร็จสมบูรณ์แล้ว (มีไฟล์ `data/cv-intel-rag-data.tar.gz` ใน Google Drive)
- ต้องมี Git ติดตั้งในเครื่อง

---

## 3. ขั้นตอนที่ 1 — เตรียม Pre-built Index

### ขั้นตอนที่ 3.1 — ดาวน์โหลด Index จาก Google Drive

1. เปิด [drive.google.com](https://drive.google.com)
2. ไปที่ `MyDrive/cv-intel-rag/data/`
3. คลิกขวาที่ไฟล์ `cv-intel-rag-data.tar.gz`
4. เลือก **Download**
5. บันทึกไฟล์ไว้ที่ไดเรกทอรีโปรเจกต์ (`cv-intel-rag/`)

### ขั้นตอนที่ 3.2 — แตก Archive และตรวจสอบขนาด

```bash
# ไปที่ไดเรกทอรีโปรเจกต์
cd cv-intel-rag

# แตกไฟล์
tar -xzf cv-intel-rag-data.tar.gz

# ตรวจสอบว่าได้ไฟล์ถูกต้อง
ls -la data/
```

**ผลลัพธ์ที่ควรเห็น:**
```
data/
├── cv_intel.db      (ประมาณ 3–10 MB)
└── chroma/          (ประมาณ 20–200 MB ขึ้นอยู่กับจำนวน records)
```

### ขั้นตอนที่ 3.3 — ตรวจสอบขนาดรวม

```bash
du -sh data/
```

- **น้อยกว่า 100 MB:** สามารถ commit ลง Git ได้โดยตรง → ดำเนินการต่อที่ขั้นตอน 4
- **มากกว่า 100 MB:** ต้องใช้ Git LFS → ดูขั้นตอน 3.4

### ขั้นตอนที่ 3.4 — (เฉพาะถ้าขนาดใหญ่กว่า 100 MB) ติดตั้ง Git LFS

Git Large File Storage (LFS) ออกแบบมาสำหรับไฟล์ขนาดใหญ่ที่ Git ปกติจัดการได้ไม่ดี

```bash
# macOS (ต้องมี Homebrew):
brew install git-lfs

# Ubuntu/Debian:
sudo apt-get install git-lfs

# Windows:
# ดาวน์โหลดจาก https://git-lfs.com/ แล้ว run installer

# เริ่มต้นใช้งาน Git LFS ใน repo
git lfs install

# กำหนดไฟล์ที่จะใช้ LFS
git lfs track "data/cv_intel.db"
git lfs track "data/chroma/**"

# บันทึกการตั้งค่า
git add .gitattributes
git commit -m "chore: configure git-lfs for data files"
```

**ตรวจสอบ:**
```bash
git lfs ls-files
# ควรเห็นไฟล์ data/ ใน list
```

---

## 4. ขั้นตอนที่ 2 — สร้าง Hugging Face Space

### ขั้นตอนที่ 4.1 — สร้าง Space ใหม่

1. ไปที่ [huggingface.co/new-space](https://huggingface.co/new-space)
2. กรอกข้อมูลในแต่ละช่อง:

   | ช่อง | ค่าที่ต้องกรอก | หมายเหตุ |
   |------|--------------|---------|
   | **Owner** | username ของคุณ | เลือกจาก dropdown |
   | **Space name** | `cv-intel-rag` | ใช้ตัวพิมพ์เล็กและ `-` แทนช่องว่าง |
   | **License** | `MIT` | เลือกจาก dropdown |
   | **SDK** | `Docker` | **สำคัญ:** อย่าเลือก Gradio หรือ Streamlit |
   | **Hardware** | `CPU basic - FREE` | 2 vCPU, 16 GB RAM |
   | **Visibility** | `Public` | หรือ `Private` ถ้าไม่ต้องการให้คนอื่นเห็น |

3. กดปุ่ม **Create Space**

**ผลลัพธ์:** HF จะสร้าง empty repository ให้และแสดงหน้า Space ของคุณที่ยังว่างอยู่

---

## 5. ขั้นตอนที่ 3 — ตั้งค่า Secrets

Secrets คือวิธีปลอดภัยในการเก็บ API keys บน HF Spaces โดย key จะถูก inject เป็น environment variable เมื่อ container ทำงาน และไม่ปรากฏใน source code หรือ log

### ขั้นตอนที่ 5.1 — เปิดหน้า Settings ของ Space

1. ไปที่ Space ของคุณ: `https://huggingface.co/spaces/<username>/cv-intel-rag`
2. คลิกแถบ **Settings**
3. เลื่อนลงหาหัวข้อ **Variables and secrets**

### ขั้นตอนที่ 5.2 — เพิ่ม Secret หลัก

กดปุ่ม **New secret** แล้วเพิ่มทีละ secret:

**Secret ที่ 1 (จำเป็น):**
- **Name:** `TYPHOON_API_KEY`
- **Value:** Typhoon API key ของคุณ (ขึ้นต้นด้วย `sk-`)

**Secret ที่ 2 (แนะนำ — เพิ่ม rate limit PubMed):**
- **Name:** `NCBI_API_KEY`
- **Value:** NCBI API key (ขอฟรีที่ [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/))

**Secret ที่ 3 (แนะนำ — ไม่มี daily cap สำหรับ openFDA):**
- **Name:** `OPENFDA_API_KEY`
- **Value:** openFDA API key (ขอฟรีที่ [open.fda.gov/apis/authentication](https://open.fda.gov/apis/authentication/))

### ขั้นตอนที่ 5.3 — ตรวจสอบ

หลังจากเพิ่ม secrets แล้ว จะเห็น list รายชื่อ secrets แต่ **ไม่** เห็นค่า (ซ่อนอยู่เพื่อความปลอดภัย) นี่คือพฤติกรรมที่ถูกต้อง

---

## 6. ขั้นตอนที่ 4 — Push โค้ดและ Index ขึ้น Space

มี 2 วิธี เลือกตามความสะดวก:

### วิธี A — Push ตรงไปที่ HF Space (ง่ายที่สุด)

```bash
# ไปที่ไดเรกทอรีโปรเจกต์
cd cv-intel-rag

# เพิ่ม HF Space เป็น remote
git remote add space https://huggingface.co/spaces/<HF_USERNAME>/cv-intel-rag

# Stage ไฟล์ data/ (ที่ extract มาจาก tar.gz)
git add data/
git add -A
git commit -m "deploy: add pre-built index for initial deployment"

# Push ขึ้น HF Space
# ระบบจะขอ username และ password (ใส่ HF token แทน password)
git push space main
```

**หมายเหตุ:** เมื่อระบบถามรหัสผ่าน ให้ใส่ HF Access Token (ขึ้นต้นด้วย `hf_`) ไม่ใช่ password ของ HF account

**ถ้าเจอ error "main branch not found":**
```bash
git push space main:main --force
```

### วิธี B — ผ่าน GitHub Remote (ใช้เมื่อมี GitHub repo อยู่แล้ว)

```bash
# ตรวจสอบว่ามี origin remote อยู่แล้ว
git remote -v

# ถ้ายังไม่มี เพิ่ม GitHub remote
git remote add origin https://github.com/<GITHUB_USERNAME>/cv-intel-rag.git

# Push ไปที่ GitHub ก่อน
git add data/
git add -A
git commit -m "deploy: add pre-built index"
git push -u origin main
```

จากนั้นตั้งค่า Auto-Deploy ตามขั้นตอนที่ 9 เพื่อให้ GitHub sync ไป HF อัตโนมัติ

---

## 7. ขั้นตอนที่ 5 — ติดตาม Build Log

### ขั้นตอนที่ 7.1 — เปิดดู Build Log

1. ไปที่ Space page ของคุณ
2. คลิกแถบ **Logs** (หรือ **Build** ในบางเวอร์ชัน)
3. สังเกตขั้นตอนการ build ที่จะเห็น:

```
Step 1/12: FROM python:3.11-slim
Step 2/12: RUN apt-get update...
...
Step 8/12: RUN pip install -r requirements.txt
   → ขั้นตอนนี้ใช้เวลานานที่สุด (5–10 นาที)
Step 12/12: CMD uvicorn src.main:app...

Successfully built <image_id>
```

### ขั้นตอนที่ 7.2 — ระยะเวลาโดยประมาณ

| รอบ Build | เวลาโดยประมาณ | สาเหตุ |
|-----------|--------------|--------|
| ครั้งแรก | 10–15 นาที | ดาวน์โหลด dependencies และ BGE-M3 model (2.3 GB) |
| รอบต่อไป | 3–5 นาที | Docker layer cache ทำงาน, skip ส่วนที่ไม่เปลี่ยน |

### ขั้นตอนที่ 7.3 — สัญญาณว่า Build สำเร็จ

ใน Logs จะเห็น:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7860
```

และใน Space page แถบสถานะจะเปลี่ยนจาก 🟡 **Building** เป็น 🟢 **Running**

---

## 8. ขั้นตอนที่ 6 — ทดสอบการทำงาน

### ขั้นตอนที่ 8.1 — เปิด App

1. ไปที่ Space page
2. คลิกแถบ **App**
3. ควรเห็น Chat UI แสดงขึ้น

**Public URL ของ Space:**
```
https://<HF_USERNAME>-cv-intel-rag.hf.space
```

### ขั้นตอนที่ 8.2 — ทดสอบผ่าน API

```bash
# ทดสอบ health check
curl https://<HF_USERNAME>-cv-intel-rag.hf.space/health

# คาดหวัง: {"status": "ok"}

# ทดสอบ stats
curl https://<HF_USERNAME>-cv-intel-rag.hf.space/stats

# คาดหวัง: JSON ที่มี total_records, breakdown_by_source

# ทดสอบ chat
curl -X POST https://<HF_USERNAME>-cv-intel-rag.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest SGLT2 inhibitor guidelines for heart failure?"}'

# คาดหวัง: JSON ที่มี answer และ sources พร้อม citations
```

### ขั้นตอนที่ 8.3 — ถ้าได้ 500 Internal Server Error

ไปดู **Logs tab** ใน Space page เพื่อดู error message ที่แท้จริง ปัญหาที่พบบ่อยคือ:
- `TYPHOON_API_KEY not set` → กลับไปตรวจสอบ Secrets ตามขั้นตอน 5
- `data/chroma: collection not found` → ข้อมูล index ไม่ได้ถูก push ขึ้นมา

---

## 9. การตั้งค่า Auto-Deploy จาก GitHub

วิธีนี้เหมาะสำหรับการพัฒนาอย่างต่อเนื่อง ทุกครั้งที่ push โค้ดไป GitHub จะ deploy ไป HF Spaces โดยอัตโนมัติ

### ขั้นตอนที่ 9.1 — เพิ่ม Secrets ใน GitHub

1. ไปที่ GitHub repository ของคุณ
2. คลิก **Settings** → **Secrets and variables** → **Actions**
3. กดปุ่ม **New repository secret** และเพิ่ม 3 secrets:

   | Secret Name | Value |
   |-------------|-------|
   | `HF_TOKEN` | HF Access Token (ขึ้นต้นด้วย `hf_`) |
   | `HF_USERNAME` | Hugging Face username ของคุณ |
   | `HF_SPACE` | ชื่อ Space เช่น `cv-intel-rag` |

### ขั้นตอนที่ 9.2 — ตรวจสอบ Workflow File

ไฟล์ `.github/workflows/sync-to-hf.yml` มีอยู่แล้วใน repository ตรวจสอบเนื้อหาว่าถูกต้อง:

```yaml
name: Sync to Hugging Face Space

on:
  push:
    branches: [main]

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to HF Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add space https://HuggingFace:$HF_TOKEN@huggingface.co/spaces/${{ secrets.HF_USERNAME }}/${{ secrets.HF_SPACE }}
          git push space main --force
```

### ขั้นตอนที่ 9.3 — ทดสอบ Workflow

1. แก้ไขไฟล์ใดก็ได้เล็กน้อย เช่น แก้ `README.md`
2. Commit และ push:
   ```bash
   git add .
   git commit -m "test: trigger auto-deploy"
   git push origin main
   ```
3. ไปที่ GitHub repository → แถบ **Actions** → ดูว่า workflow รันสำเร็จหรือไม่
4. ไปที่ HF Space → Logs → ดูว่า build เริ่มต้นขึ้นมาโดยอัตโนมัติ

---

## 10. ข้อจำกัดของ HF Spaces Free Tier

| ทรัพยากร | แผนฟรี | แผน Pro ($9/เดือน) |
|----------|--------|-------------------|
| CPU | 2 vCPU | 2 vCPU |
| RAM | 16 GB | 16 GB |
| พื้นที่จัดเก็บ | 50 GB | 50 GB |
| Persistent Storage | ❌ (ข้อมูลหายเมื่อ restart) | ✅ 100 GB |
| GPU | ❌ | A10G, T4, etc. (ราคาแยก) |
| หยุดทำงานหลัง idle | 48 ชั่วโมง | ไม่หยุด |
| Custom Domain | ❌ | ✅ |
| Private Spaces | จำกัด | ไม่จำกัด |

> **สำคัญเกี่ยวกับ Sleep Mode:** Free tier จะหยุดทำงานหลังจาก idle 48 ชั่วโมง และใช้เวลา ~30 วินาทีในการ wake up ครั้งแรกหลังจากนั้น หากต้องการ demo ให้เปิด Space ล่วงหน้า 1–2 นาทีก่อนการนำเสนอ

---

## 11. การอัปเดต Index ภายหลัง

### 11.1 อัปเดต Corpus (ข้อมูลใหม่)

1. รัน Notebook 1 บน Colab ใหม่ เพื่อดึงข้อมูลล่าสุด
2. ดาวน์โหลด `cv-intel-rag-data.tar.gz` ใหม่จาก Drive
3. Extract และ replace โฟลเดอร์ `data/`
4. Commit และ push:
   ```bash
   git add data/
   git commit -m "data: refresh corpus $(date +%Y-%m-%d)"
   git push origin main   # auto-deploy ไป HF
   ```

### 11.2 อัปเดตโค้ด (bug fix หรือ feature ใหม่)

```bash
# แก้ไขโค้ดตามต้องการ
git add -A
git commit -m "fix: your message here"
git push origin main   # auto-deploy ไป HF ถ้าตั้งค่า Workflow แล้ว
```

---

## 12. ปัญหาที่พบบ่อยและวิธีแก้ไข

| อาการ | สาเหตุที่เป็นไปได้ | วิธีแก้ไข |
|-------|-------------------|-----------|
| Build fail: `dockerfile not found` | `Dockerfile` ไม่ได้อยู่ที่ root ของ repository | ตรวจสอบว่าไฟล์ `Dockerfile` อยู่ที่ระดับเดียวกับ `README.md` |
| Build fail: `out of memory during pip install` | Container หน่วยความจำไม่พอระหว่าง build | เพิ่ม `ENV EMBED_BATCH_SIZE=4` ใน Dockerfile |
| App ขึ้นแต่ตอบ "insufficient context" ทุกครั้ง | โฟลเดอร์ `data/chroma/` ว่างเปล่าหรือไม่ได้ push | รัน `git lfs ls-files` ตรวจสอบว่า data files ถูก track และ push แล้ว |
| `TYPHOON_API_KEY not set` | Secret ยังไม่ได้ตั้งค่า หรือ restart Space หลังเพิ่ม Secret | ไปที่ Space Settings → Secrets ตรวจสอบ; restart Space จาก Settings |
| 504 Gateway Timeout | Typhoon API ใช้เวลานานเกินไป | ลด `RETRIEVE_TOP_K` เป็น 5 ใน Secrets/Variables หรือ `.env` |
| Port 7860 ไม่ตอบสนอง | `app_port` ใน README.md frontmatter ไม่ตรงกับ Dockerfile `EXPOSE` | ตรวจสอบทั้ง README.md (บรรทัด `app_port: 7860`) และ Dockerfile (`EXPOSE 7860`) |
| GitHub Actions workflow ล้มเหลว | Secrets ยังไม่ครบหรือ HF Token หมดอายุ/ไม่มีสิทธิ์ Write | ตรวจสอบ GitHub Secrets ทั้ง 3 ค่า; สร้าง HF Token ใหม่ถ้าจำเป็น |

---

## ขั้นตอนถัดไปหลัง Deploy สำเร็จ

- **สำรอง Public URL:** `https://<HF_USERNAME>-cv-intel-rag.hf.space` พร้อมแชร์ได้ทันที
- **ตั้งค่า Custom Domain (Pro):** Space Settings → Space custom domain → ตั้ง CNAME ใน DNS
- **ขยาย Infrastructure:** สำหรับ production workload ที่ต้องการ SLA สูงขึ้น พิจารณาย้ายไป Google Cloud Run หรือ AWS ECS โดยใช้ `Dockerfile` เดิมได้เลย เพียงเปลี่ยน port และ storage strategy
