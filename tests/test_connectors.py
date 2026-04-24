"""
Connector tests — no network calls. We mock httpx / feedparser and verify
the connector's parsing logic produces correct Record objects.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from src.connectors.clinical_trials import ClinicalTrialsConnector
from src.connectors.openfda import OpenFDAConnector
from src.connectors.pubmed import PubMedConnector
from src.connectors.registry import CONNECTORS, get_connector
from src.connectors.rss import RSSConnector
from src.models.record import Category, SourceType


# ─── registry ────────────────────────────────────────────────────────────

def test_registry_instantiates_all():
    for key in CONNECTORS:
        instance = get_connector(key)
        assert instance.cfg.source_name
        assert instance.cfg.country_or_region


# ─── PubMed ──────────────────────────────────────────────────────────────

_PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">88888001</PMID>
      <Article>
        <Journal>
          <Title>J Heart Failure</Title>
          <JournalIssue>
            <PubDate><Year>2026</Year><Month>Apr</Month><Day>10</Day></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>SGLT2 inhibitors and cardiovascular outcomes</ArticleTitle>
        <Abstract><AbstractText>Large RCT showed benefit.</AbstractText></Abstract>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Sodium-Glucose Transporter 2 Inhibitors</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName>Heart Failure</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_pubmed_parses_efetch_xml():
    connector = PubMedConnector(queries=["test"], days_back=7)

    with patch("src.connectors.pubmed.httpx.Client") as mock_client:
        ctx = MagicMock()
        mock_client.return_value.__enter__.return_value = ctx
        ctx.get.side_effect = [
            MagicMock(json=lambda: {"esearchresult": {"idlist": ["88888001"]}},
                      raise_for_status=lambda: None),
            MagicMock(text=_PUBMED_XML, raise_for_status=lambda: None),
        ]
        records = list(connector.fetch())

    assert len(records) == 1
    r = records[0]
    assert r.source_name == "PubMed"
    assert r.category == Category.RESEARCH.value
    assert r.external_id == "88888001"
    assert r.published_date == date(2026, 4, 10)
    assert "Heart Failure" in r.topic_tags
    assert r.url == "https://pubmed.ncbi.nlm.nih.gov/88888001/"


# ─── ClinicalTrials.gov ─────────────────────────────────────────────────

_CTGOV = {
    "studies": [{
        "protocolSection": {
            "identificationModule": {"nctId": "NCT88888888", "briefTitle": "Empagliflozin CKD study"},
            "statusModule": {"lastUpdatePostDateStruct": {"date": "2026-04-15"}},
            "conditionsModule": {"conditions": ["CKD", "Heart Failure"]},
            "designModule": {"phases": ["PHASE3"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "ACME"}},
            "descriptionModule": {"briefSummary": "RCT of empagliflozin."},
        }
    }]
}


def test_clinicaltrials_parses_json():
    connector = ClinicalTrialsConnector(conditions=["heart failure"])

    with patch("src.connectors.clinical_trials.httpx.Client") as mock_client:
        ctx = MagicMock()
        mock_client.return_value.__enter__.return_value = ctx
        ctx.get.return_value = MagicMock(json=lambda: _CTGOV, raise_for_status=lambda: None)
        records = list(connector.fetch())

    assert len(records) == 1
    r = records[0]
    assert r.external_id == "NCT88888888"
    assert r.category == Category.CLINICAL_TRIAL.value
    assert r.published_date == date(2026, 4, 15)
    assert "CKD" in r.topic_tags


# ─── openFDA (with CV filter) ───────────────────────────────────────────

_OPENFDA_ENFORCEMENT = {
    "results": [
        {   # CV-relevant — should pass filter
            "recall_number": "D-CV-0001-2026",
            "product_description": "Atenolol tablets, 50mg (beta blocker)",
            "reason_for_recall": "Impurity above limit",
            "recalling_firm": "ACME",
            "classification": "Class II",
            "report_date": "20260401",
        },
        {   # NOT CV-relevant — should be filtered out
            "recall_number": "D-OTH-0001-2026",
            "product_description": "Acetaminophen 500mg tablets",
            "reason_for_recall": "Labeling error",
            "recalling_firm": "Other",
            "classification": "Class III",
            "report_date": "20260402",
        },
    ]
}


def test_openfda_filters_non_cv_drugs():
    connector = OpenFDAConnector()

    call_count = {"n": 0}
    def mock_get(url, params=None):
        call_count["n"] += 1
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = lambda: None
        if call_count["n"] == 1:
            resp.json = lambda: _OPENFDA_ENFORCEMENT
        else:
            resp.json = lambda: {"results": []}
        return resp

    with patch("src.connectors.openfda.httpx.Client") as mock_client:
        ctx = MagicMock()
        mock_client.return_value.__enter__.return_value = ctx
        ctx.get.side_effect = mock_get
        records = list(connector.fetch(limit=10))

    cv_records = [r for r in records if "atenolol" in r.title.lower() or "beta" in r.title.lower()]
    non_cv = [r for r in records if "acetaminophen" in r.title.lower()]
    assert len(cv_records) == 1, "CV drug must pass filter"
    assert len(non_cv) == 0, "Non-CV drug must be filtered out"


# ─── RSS ────────────────────────────────────────────────────────────────

def test_rss_filters_by_cv_keyword():
    import time as _time

    fake = MagicMock()
    fake.bozo = False
    fake.entries = [
        # CV-relevant
        MagicMock(
            title="FDA safety alert: SGLT2 inhibitor side effect",
            link="https://fda.gov/a", summary="Cardiovascular concern.",
            id="cv-1",
            published_parsed=_time.struct_time((2026, 4, 15, 0, 0, 0, 0, 105, 0)),
            tags=[],
        ),
        # NOT CV-relevant — should be filtered out
        MagicMock(
            title="FDA approves new dermatology treatment",
            link="https://fda.gov/b", summary="Skin condition drug.",
            id="derm-1",
            published_parsed=_time.struct_time((2026, 4, 14, 0, 0, 0, 0, 104, 0)),
            tags=[],
        ),
    ]

    connector = RSSConnector(feed_key="fda_medwatch", category=Category.SAFETY)
    with patch("src.connectors.rss.feedparser.parse", return_value=fake):
        records = list(connector.fetch())

    assert len(records) == 1, f"CV filter must drop non-CV entries, got {len(records)}"
    assert records[0].external_id == "cv-1"
    assert records[0].source_type == SourceType.RSS.value
