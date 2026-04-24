# 🚀 Deploy Guide — Hugging Face Spaces (ภาษาไทย)

Deploy CV Intel RAG ขึ้น cloud ฟรี → ได้ public URL professional ใช้ขายลูกค้าได้ทันที

**Target:** Hugging Face Spaces (Docker SDK) — ฟรี, 16GB RAM, 2 vCPU, persistent index

---

## 🗺️ แผนภาพ flow

```
Colab Notebook 1                 HF Spaces (Docker)              Public URL
──────────────────               ───────────────────              ──────────
ingest + embed                   [build]                          https://huggingface.co/
       ↓                         clone repo + data                 spaces/you/cv-intel-rag
creates data/*.tar.gz     ──►    pip install                      (ready in 10-15 min)
       ↓                         pre-cache BGE-M3
commit to HF repo                [run]
(via git-lfs)                    uvicorn on :7860
                                 TYPHOON_API_KEY from Secrets
```

---

## Step 1 — เตรียมของ

คุณต้องมี:
- ✅ GitHub account
- ✅ Hugging Face account ([huggingface.co/join](https://huggingface.co/join))
- ✅ Typhoon API key
- ✅ `data/cv-intel-rag-data.tar.gz` จาก notebook 1 (อยู่ใน Google Drive)

---

## Step 2 — ดึง pre-built index ออกมา

บนเครื่องคุณ:

```bash
# ดาวน์โหลด tar.gz จาก Google Drive มาไว้ในโปรเจกต์
cd cv-intel-rag
# วาง cv-intel-rag-data.tar.gz ใน root ของ repo
tar -xzf cv-intel-rag-data.tar.gz
# จะได้ data/cv_intel.db + data/chroma/
ls data/
```

ตรวจขนาด:

```bash
du -sh data/
# ถ้า < 100 MB  → commit ตรงๆ ได้
# ถ้า > 100 MB → ต้องใช้ git-lfs (ดู Step 2.1)
```

### Step 2.1 — (ถ้าจำเป็น) ติดตั้ง git-lfs

```bash
# macOS:  brew install git-lfs
# Ubuntu: sudo apt install git-lfs
git lfs install
git lfs track "data/cv_intel.db"
git lfs track "data/chroma/**"
git add .gitattributes
```

---

## Step 3 — สร้าง HF Space

1. ไปที่ [huggingface.co/new-space](https://huggingface.co/new-space)
2. กรอก:
   - **Space name:** `cv-intel-rag`
   - **License:** MIT
   - **SDK:** **Docker** (สำคัญ — อย่าเลือก Gradio/Streamlit)
   - **Hardware:** CPU basic (ฟรี) — 16 GB RAM, 2 vCPU
   - **Visibility:** Public (หรือ Private ถ้าไม่อยากให้คนอื่นเห็น)
3. กด Create Space — จะได้ empty repo

---

## Step 4 — ตั้ง Secret

1. ไปที่ Space ที่เพิ่งสร้าง → **Settings** tab
2. เลื่อนลง **Variables and secrets**
3. **New secret**:
   - Name: `TYPHOON_API_KEY`
   - Value: key ของคุณ
4. Save

> 🔒 Secret ถูก inject เป็น env var ตอน container รัน — **ไม่ expose ใน source code**

---

## Step 5 — Push โค้ด + index ไปที่ Space

### วิธี A: Push ตรง (ง่ายสุด)

```bash
cd cv-intel-rag

# เพิ่ม HF Space เป็น remote
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/cv-intel-rag

# commit ทั้งหมด (รวม data/)
git add -A
git commit -m "Initial deploy with pre-built index"

# push — ครั้งแรกจะ prompt ถาม username + token (ใช้ HF token ที่มี Write access)
git push space main

# ถ้าเจอ "main branch not found" → lung ใช้:
git push space main:main --force
```

สร้าง HF token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token** → Write access

### วิธี B: Auto-sync จาก GitHub (แนะนำสำหรับ solo entrepreneur)

ถ้าใช้ GitHub เป็น source of truth:

1. Push repo ไป GitHub ก่อน:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/cv-intel-rag.git
   git push -u origin main
   ```

2. ตั้ง GitHub Secrets (Settings → Secrets and variables → Actions):
   - `HF_TOKEN` — HF token (Write)
   - `HF_USERNAME` — HF username
   - `HF_SPACE` — `cv-intel-rag`

3. `.github/workflows/sync-to-hf.yml` มีอยู่แล้ว → ทุกครั้งที่ push ไป `main` → auto-deploy

---

## Step 6 — ดูผล

1. ไปที่ Space page → tab **App** หรือ **Logs**
2. รอ build ~10-15 นาที (ครั้งแรก — ดาวน์โหลด BGE-M3 2.3 GB)
3. รอบต่อไป ~3-5 นาที (cache ทำงาน)

พอเห็น `Uvicorn running on http://0.0.0.0:7860` → เปิด tab **App** → ใช้ได้ละ

**Public URL:**
```
https://huggingface.co/spaces/YOUR_HF_USERNAME/cv-intel-rag
```

มี iframe embedded preview และปุ่ม "Open in full page" สำหรับโชว์ลูกค้า

---

## Step 7 — Custom domain (optional, professional look)

ถ้าอยากได้ URL เช่น `demo.yourcompany.com` แทน `*.hf.space`:

1. Space Settings → **Space custom domain**
2. พิมพ์ domain ของคุณ
3. ตั้ง CNAME ใน DNS ของคุณชี้ไปที่ `*.hf.space`

ฟีเจอร์นี้ **ฟรี** แต่ต้อง HF Pro ($9/mo) — คุ้มมากถ้าคิดจะขายจริง

---

## Step 8 — ตรวจสอบการทำงานจริง

```bash
curl https://YOUR_USERNAME-cv-intel-rag.hf.space/stats
# ควรได้ JSON: {"total_records": ..., "total_chunks": ...}

curl -X POST https://YOUR_USERNAME-cv-intel-rag.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "SGLT2 heart failure"}'
# ควรได้คำตอบ + citations
```

ถ้า 404/500 → ไปดู Logs tab ใน HF Space

---

## 📊 Limits ของ HF Spaces ฟรี

| Resource | Free | Pro ($9/mo) |
|---|---|---|
| CPU | 2 vCPU | 2 vCPU |
| RAM | 16 GB | 16 GB |
| Storage (image) | 50 GB | 50 GB |
| Persistent storage | ❌ | ✅ (100 GB) |
| GPU | ❌ | ZeroGPU / paid GPUs |
| Sleep after idle | 48 ชม. | ไม่ sleep |
| Build time limit | 2 ชม. | 2 ชม. |
| Custom domain | ❌ | ✅ |

> ⚠️ **Free tier sleep หลัง idle 48 ชม.** — ตื่นใหม่ใช้เวลา ~30 วิ ถ้า live demo ตอนลูกค้ามา แนะนำให้กดเปิดก่อน 5 นาที

---

## 🔄 Update index ภายหลัง

- **แก้โค้ด:** push to GitHub → auto-sync ไป HF → rebuild (~5 นาที)
- **อัพเดต corpus:** รัน Colab notebook 1 ใหม่ → ได้ tar.gz ใหม่ → replace `data/` ใน repo → push → HF rebuild

---

## ❗ Troubleshooting

| ปัญหา | วิธีแก้ |
|---|---|
| Build fail — "dockerfile not found" | ตรวจว่า `Dockerfile` อยู่ที่ **root** ของ repo (ไม่ใช่ใน `deploy/`) |
| Build fail — "out of memory" | ลด `EMBED_BATCH_SIZE` ใน Dockerfile env เป็น 4 |
| App ขึ้น แต่ตอบทุกคำถามด้วย "insufficient context" | `data/chroma/` ว่าง — ตรวจว่า commit เข้าไปจริง (`git lfs ls-files`) |
| "TYPHOON_API_KEY not set" | ตรวจ Space Settings → Variables and secrets |
| 504 Gateway Timeout | Typhoon API timeout — ลด `RETRIEVE_TOP_K` ใน `.env` |
| Port 7860 ใช้ไม่ได้ | ตรวจ `app_port: 7860` ใน README.md YAML frontmatter ตรงกับ Dockerfile EXPOSE |

---

## 🎯 ถัดไป

- ขายลูกค้า — ใช้ public URL ของ Space + notebook 2 เป็น leave-behind
- ถ้าได้ deal → อัพเกรด Pro → เปิด custom domain → deploy production
- สำหรับ enterprise — พิจารณาย้ายไป Cloud Run / AWS ECS (โค้ดเดิมใช้ได้ แค่เปลี่ยน Dockerfile port + storage strategy)
