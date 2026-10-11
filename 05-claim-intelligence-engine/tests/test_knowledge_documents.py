import pytest
import pandas as pd
from pathlib import Path
from src.knowledge_documents import (
    chunk_policy_text,
    build_case_records,
    build_policy_records,
)

def sample_claim(claim_id="CASE001", claim_amount=1500.0):
    return {
        "claim_id": claim_id,
        "incident_date": "2026-07-15",
        "driver_age": 34,
        "vehicle_age": 5,
        "annual_mileage": 12000,
        "vehicle_value": 28000,
        "repair_estimate": 700,
        "prior_claims": 1,
        "incident_type": "collision",
        "vehicle_type": "sedan",
        "damage_severity": "minor",
        "weather": "clear",
        "region": "north",
        "police_reported": "yes",
        "claim_amount": claim_amount,
    }

def test_policy_chunking():
    text = " ".join(f"word{i}" for i in range(12))

    chunks = chunk_policy_text(
        text,
        chunk_words=5,
        overlap_words=2,
    )

    assert len(chunks) > 1
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]

def test_empty_policy_text():
    assert chunk_policy_text("") == []

def test_training_cases_indexed():
    claims = pd.DataFrame([
        sample_claim("CASE001", 1500),
        sample_claim("CASE002", 2500),
        sample_claim("CASE003", 3500),
    ])

    splits = pd.DataFrame({
        "claim_id": ["CASE001", "CASE002", "CASE003"],
        "split": ["train", "validation", "test"],
    })

    records = build_case_records(claims, splits)

    assert len(records) == 1
    assert records[0]["metadata"]["claim_id"] == "CASE001"
    assert records[0]["metadata"]["split"] == "train"

def test_settlement_separation():
    claims = pd.DataFrame([
        sample_claim("CASE001", 1500.0)
    ])

    splits = pd.DataFrame({
        "claim_id": ["CASE001"],
        "split": ["train"],
    })

    records = build_case_records(claims, splits)

    assert len(records) == 1
    assert records[0]["metadata"]["claim_amount"] == 1500.0
    assert "claim_amount" not in records[0]["document"]
    assert "1500.0" not in records[0]["document"]

def test_policy_source_metadata(tmp_path):
    policy = tmp_path / "sample_policy.md"
    policy.write_text(
        "# Sample Policy\n\nClaims require documentation.",
        encoding="utf-8",
    )
    records = build_policy_records(tmp_path)
    assert len(records) >= 1
    assert records[0]["metadata"]["source_file"] == "sample_policy.md"
    assert records[0]["metadata"]["chunk_index"] == 0
    assert records[0]["metadata"]["source_type"] == "synthetic_policy"

def test_ignore_macos_metadata(tmp_path):
    valid = tmp_path / "collision_policy.md"
    invalid = tmp_path / "._collision_policy.md"
    valid.write_text(
        "# Collision Policy\n\nExample content.",
        encoding="utf-8",
    )
    invalid.write_bytes(b"\x00\xff\x00\xfe")
    records = build_policy_records(tmp_path)
    assert len(records) == 1
    assert records[0]["metadata"]["source_file"] == "collision_policy.md"

def test_duplicate_claim_ids():
    claims = pd.DataFrame({
        "claim_id": ["CASE001", "CASE001"],
        "claim_amount": [1000, 1200],
        "description": ["Duplicate entry test 1", "Duplicate entry test 2"]
    })
    splits = pd.DataFrame({
        "claim_id": ["CASE001", "CASE001"],
        "split": ["train", "train"]
    })

    with pytest.raises(ValueError):
        build_case_records(claims, splits)