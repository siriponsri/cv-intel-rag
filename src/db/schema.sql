-- =========================================================================
--  Pharma Regulatory Intelligence Platform — Database Schema
--  Layer 1 (CSV-exportable structured records) + audit tables
--  Compatible: PostgreSQL 14+ / SQLite 3.35+ (minor syntax notes inline)
-- =========================================================================

-- ──────────────────────────────────────────────────────────────────────────
-- 1.  records  —  the unified Layer 1 record table
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS records (
    record_id           TEXT PRIMARY KEY,            -- UUID v4
    title               TEXT NOT NULL,
    source_name         TEXT NOT NULL,               -- e.g. 'PubMed', 'openFDA', 'THFDA_Catalog'
    source_type         TEXT NOT NULL CHECK (        -- how the data was obtained
        source_type IN ('api', 'bulk_dataset', 'rss', 'web_scrape')
    ),
    layer               INTEGER NOT NULL DEFAULT 1 CHECK (layer = 1),
    country_or_region   TEXT NOT NULL,               -- 'Thailand', 'US', 'EU', 'Global'
    published_date      DATE,
    ingested_at         TIMESTAMP NOT NULL,
    url                 TEXT,
    category            TEXT NOT NULL CHECK (
        category IN (
            'drug_approval', 'safety', 'guideline',
            'patent', 'research', 'clinical_trial',
            'regulator', 'other'
        )
    ),
    summary_short       TEXT,                        -- 3–5 sentence AI summary
    confidence          TEXT CHECK (confidence IN ('high', 'medium', 'low')),
    review_status       TEXT NOT NULL DEFAULT 'auto' CHECK (
        review_status IN ('auto', 'reviewed', 'flagged')
    ),
    exportable_to_csv   BOOLEAN NOT NULL DEFAULT TRUE,

    -- denormalised unique key from source (e.g. PMID, NCTID, openFDA safety report id)
    external_id         TEXT,
    external_version    TEXT,

    -- optional structured fields common enough to warrant columns
    language            TEXT DEFAULT 'en',

    -- housekeeping
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (source_name, external_id, external_version)
);

CREATE INDEX IF NOT EXISTS idx_records_source          ON records (source_name);
CREATE INDEX IF NOT EXISTS idx_records_category        ON records (category);
CREATE INDEX IF NOT EXISTS idx_records_region          ON records (country_or_region);
CREATE INDEX IF NOT EXISTS idx_records_published_date  ON records (published_date);
CREATE INDEX IF NOT EXISTS idx_records_ingested_at     ON records (ingested_at);
CREATE INDEX IF NOT EXISTS idx_records_review_status   ON records (review_status);

-- ──────────────────────────────────────────────────────────────────────────
-- 2.  raw_payloads  —  verbatim source content, for audit & reprocessing
--     (kept separate so `records` stays skinny and fast to scan)
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_payloads (
    payload_id      TEXT PRIMARY KEY,                -- UUID v4
    record_id       TEXT NOT NULL REFERENCES records(record_id) ON DELETE CASCADE,
    content_type    TEXT NOT NULL,                   -- 'application/json', 'text/html', 'application/pdf', ...
    raw_text        TEXT,                            -- JSON / HTML / extracted text
    blob_path       TEXT,                            -- optional S3/local path for binary originals
    fetched_at      TIMESTAMP NOT NULL,
    checksum        TEXT,                            -- SHA-256 for dedup / change detection

    UNIQUE (record_id, checksum)
);
CREATE INDEX IF NOT EXISTS idx_raw_payloads_record ON raw_payloads (record_id);

-- ──────────────────────────────────────────────────────────────────────────
-- 3.  tags  —  topic_tags and entity_tags stored relationally
--     (easier to query than JSON arrays, still CSV-exportable via JOIN)
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tags (
    tag_id      INTEGER PRIMARY KEY AUTOINCREMENT,   -- PostgreSQL: use SERIAL or IDENTITY
    tag_type    TEXT NOT NULL CHECK (tag_type IN ('topic', 'entity')),
    tag_value   TEXT NOT NULL,
    normalized  TEXT,                                 -- lowercased / stemmed version for matching
    UNIQUE (tag_type, tag_value)
);
CREATE INDEX IF NOT EXISTS idx_tags_normalized ON tags (normalized);

CREATE TABLE IF NOT EXISTS record_tags (
    record_id   TEXT NOT NULL REFERENCES records(record_id) ON DELETE CASCADE,
    tag_id      INTEGER NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
    PRIMARY KEY (record_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_record_tags_tag ON record_tags (tag_id);

-- ──────────────────────────────────────────────────────────────────────────
-- 4.  ingestion_runs  —  audit log of every connector execution
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id          TEXT PRIMARY KEY,                 -- UUID v4
    source_name     TEXT NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    status          TEXT NOT NULL CHECK (
        status IN ('running', 'success', 'partial', 'failed')
    ),
    records_fetched INTEGER DEFAULT 0,
    records_new     INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    error_message   TEXT,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_source ON ingestion_runs (source_name);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_start  ON ingestion_runs (started_at);

-- ──────────────────────────────────────────────────────────────────────────
-- 5.  wiki_refs  —  link Layer 1 records to wiki pages they derived/updated
--     (keeps provenance so we can trace every wiki claim back to L1 if any)
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wiki_refs (
    ref_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id       TEXT REFERENCES records(record_id) ON DELETE SET NULL,
    wiki_path       TEXT NOT NULL,                    -- e.g. 'official/regulators/thai-fda.md'
    wiki_layer      INTEGER NOT NULL CHECK (wiki_layer IN (2, 3)),
    relation        TEXT NOT NULL CHECK (
        relation IN ('source', 'mentioned', 'contradicts', 'updates')
    ),
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_wiki_refs_record ON wiki_refs (record_id);
CREATE INDEX IF NOT EXISTS idx_wiki_refs_path   ON wiki_refs (wiki_path);

-- ──────────────────────────────────────────────────────────────────────────
-- 6.  review_queue  —  items flagged for human review (governance)
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS review_queue (
    queue_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id       TEXT REFERENCES records(record_id) ON DELETE CASCADE,
    reason          TEXT NOT NULL,                    -- why flagged
    priority        INTEGER DEFAULT 5,                -- 1 (highest) – 10 (lowest)
    assigned_to     TEXT,
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'in_review', 'approved', 'rejected')
    ),
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at     TIMESTAMP,
    reviewer_notes  TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue (status);

-- ──────────────────────────────────────────────────────────────────────────
-- 7.  convenience view: csv_export
--     What gets dumped to weekly CSV — joins records + tags
-- ──────────────────────────────────────────────────────────────────────────
CREATE VIEW IF NOT EXISTS csv_export AS
SELECT
    r.record_id,
    r.title,
    r.source_name,
    r.source_type,
    r.country_or_region,
    r.published_date,
    r.ingested_at,
    r.url,
    r.category,
    r.summary_short,
    r.confidence,
    r.review_status,
    r.language,
    -- aggregate tags as comma-separated strings (portable across SQLite/PG)
    (SELECT GROUP_CONCAT(t.tag_value, '|')
       FROM record_tags rt JOIN tags t ON t.tag_id = rt.tag_id
      WHERE rt.record_id = r.record_id AND t.tag_type = 'topic')    AS topic_tags,
    (SELECT GROUP_CONCAT(t.tag_value, '|')
       FROM record_tags rt JOIN tags t ON t.tag_id = rt.tag_id
      WHERE rt.record_id = r.record_id AND t.tag_type = 'entity')   AS entity_tags
FROM records r
WHERE r.exportable_to_csv = TRUE;
