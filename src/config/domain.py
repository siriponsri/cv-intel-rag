"""
Domain scope: Cardiovascular + Diabetes + Chronic Kidney Disease.

One central module that every connector imports to stay in scope.
Update this file to expand/narrow the domain.
"""
from __future__ import annotations

# ─── PubMed MeSH terms ───────────────────────────────────────────────────
# Use MeSH hierarchy for precision (tree numbers avoid false hits)
PUBMED_MESH_QUERIES = [
    # Cardiovascular Diseases (C14)
    "Cardiovascular Diseases[MeSH Terms]",
    "Heart Failure[MeSH Terms]",
    "Hypertension[MeSH Terms]",
    "Atrial Fibrillation[MeSH Terms]",
    "Myocardial Infarction[MeSH Terms]",
    # Diabetes (C19)
    "Diabetes Mellitus[MeSH Terms]",
    "Diabetes Mellitus, Type 2[MeSH Terms]",
    # CKD (C12 / C13)
    "Renal Insufficiency, Chronic[MeSH Terms]",
    "Diabetic Nephropathies[MeSH Terms]",
    # Cardiorenal / cardiometabolic drug classes
    "SGLT2 Inhibitors[Pharmacological Action]",
    "GLP-1 Receptor Agonists[MeSH Terms]",
    "Angiotensin Receptor Antagonists[MeSH Terms]",
]

# ─── ClinicalTrials.gov query terms ──────────────────────────────────────
CTGOV_CONDITIONS = [
    "Cardiovascular Disease",
    "Heart Failure",
    "Atrial Fibrillation",
    "Hypertension",
    "Type 2 Diabetes",
    "Chronic Kidney Disease",
    "Diabetic Nephropathy",
]

# ─── openFDA filter keywords ────────────────────────────────────────────
# Used against pharm_class_epc / product_description / reason_for_recall
OPENFDA_DRUG_CLASSES = [
    "cardiovascular",
    "antihypertensive",
    "anticoagulant",
    "beta blocker",
    "ace inhibitor",
    "arb",
    "statin",
    "antidiabetic",
    "sglt2",
    "glp-1",
    "insulin",
    "sartan",
    "diuretic",
]

# ─── RSS feed catalog — guideline + regulatory news ─────────────────────
# Curated for CV/DM/CKD relevance
CV_RSS_FEEDS = {
    "fda_medwatch": {
        "url": "https://www.fda.gov/AboutFDA/ContactFDA/StayInformed/RSSFeeds/MedWatch/rss.xml",
        "source_name": "FDA_MedWatch",
        "region": "US",
    },
    "ema_whats_new": {
        "url": "https://www.ema.europa.eu/en/rss.xml",
        "source_name": "EMA_WhatsNew",
        "region": "EU",
    },
    "escardio_news": {
        "url": "https://www.escardio.org/Guidelines/rss",
        "source_name": "ESC_Guidelines",
        "region": "EU",
    },
    "ahajournals": {
        "url": "https://www.ahajournals.org/action/showFeed?type=etoc&feed=rss&jc=circ",
        "source_name": "AHA_Circulation",
        "region": "US",
    },
}

# ─── Title-level regex (cheap filter before embedding) ──────────────────
# If a record's title/summary doesn't match, we tag it as out-of-scope.
#
# Note: `\b` only goes at the START of prefix terms (cardio, sglt, ...) so
# they match longer compounds (Cardiovascular, SGLT2). Short codes like
# `bp`, `dm`, `ckd` keep `\b` on BOTH sides so we don't match e.g. "bpm".
# Thai characters don't respect ASCII `\b`, so Thai terms stand alone.
CV_TERM_REGEX = (
    r"(?i)(?:"
    # Short codes — whole-word only
    r"\b(?:bp|dm|ckd|hbp|hf|cad|mi|pad|afib|acs|egfr)\b"
    r"|"
    # Prefixes — match SGLT2, cardiovascular, diabetes, etc.
    r"\b(?:cardio|cardiac|heart|coronary|atrial|ventricular|vascular|"
    r"hypertens|hypotens|"
    r"diabet|glycaem|glycemic|insulin|sglt|glp|tirzepatid|semaglutid|liraglutid|"
    r"kidney|renal|nephro|dialys|"
    r"stroke|infarct|thrombo|anticoag|antiplatelet|statin|ezetimibe)"
    r"|"
    # Multi-word phrase
    r"\bblood pressure\b"
    r"|"
    # Thai terms — no \b for Thai chars
    r"(?:หัวใจ|หลอดเลือด|ความดัน|เบาหวาน|ไต)"
    r")"
)
