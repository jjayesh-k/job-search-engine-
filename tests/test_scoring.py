"""
test_fetch_jobs.py — Unit tests for fetch_jobs.py

Tests the helper functions (ID generation, deduplication, filtering)
without making real API calls — so these run fast and free.

Run with:
    pytest tests/test_fetch_jobs.py -v
"""

import pytest
from src.fetch_jobs import _make_job_id, deduplicate, filter_valid


# ── _make_job_id ───────────────────────────────────────────────────────────────

def test_job_id_is_stable():
    """Same inputs always produce the same ID."""
    id1 = _make_job_id("Data Scientist", "Google", "https://example.com/job/1")
    id2 = _make_job_id("Data Scientist", "Google", "https://example.com/job/1")
    assert id1 == id2


def test_job_id_is_case_insensitive():
    """Title and company casing doesn't affect the ID."""
    id1 = _make_job_id("data scientist", "google", "https://example.com/job/1")
    id2 = _make_job_id("Data Scientist", "Google", "https://example.com/job/1")
    assert id1 == id2


def test_job_id_differs_for_different_jobs():
    """Different jobs produce different IDs."""
    id1 = _make_job_id("Data Scientist", "Google", "https://example.com/job/1")
    id2 = _make_job_id("ML Engineer",    "Meta",   "https://example.com/job/2")
    assert id1 != id2


# ── deduplicate ────────────────────────────────────────────────────────────────

def _make_job(title, company, url, source="adzuna", desc="A" * 100):
    return {
        "id":          _make_job_id(title, company, url),
        "title":       title,
        "company":     company,
        "location":    "India",
        "description": desc,
        "url":         url,
        "salary_min":  None,
        "salary_max":  None,
        "posted_date": "2025-01-01",
        "source":      source,
    }


def test_deduplicate_removes_exact_duplicates():
    jobs = [
        _make_job("Data Scientist", "Google", "https://g.com/1", "adzuna"),
        _make_job("Data Scientist", "Google", "https://g.com/1", "adzuna"),
    ]
    result = deduplicate(jobs)
    assert len(result) == 1


def test_deduplicate_keeps_adzuna_over_serpapi():
    """When the same job exists in both sources, Adzuna version is kept."""
    adzuna_job  = _make_job("Data Scientist", "Google", "https://g.com/1", "adzuna")
    serpapi_job = _make_job("Data Scientist", "Google", "https://g.com/1", "serpapi")
    result = deduplicate([adzuna_job, serpapi_job])
    assert len(result) == 1
    assert result[0]["source"] == "adzuna"


def test_deduplicate_keeps_different_jobs():
    jobs = [
        _make_job("Data Scientist", "Google", "https://g.com/1"),
        _make_job("ML Engineer",    "Meta",   "https://m.com/2"),
    ]
    result = deduplicate(jobs)
    assert len(result) == 2


# ── filter_valid ───────────────────────────────────────────────────────────────

def test_filter_removes_short_description():
    jobs = [_make_job("Data Scientist", "Google", "https://g.com/1", desc="Too short")]
    result = filter_valid(jobs)
    assert len(result) == 0


def test_filter_keeps_valid_job():
    jobs = [_make_job("Data Scientist", "Google", "https://g.com/1", desc="X" * 200)]
    result = filter_valid(jobs)
    assert len(result) == 1


def test_filter_removes_off_target_roles():
    jobs = [_make_job("Sales Manager", "Corp", "https://c.com/1", desc="X" * 200)]
    result = filter_valid(jobs)
    assert len(result) == 0


def test_filter_keeps_analyst_roles():
    jobs = [_make_job("Data Analyst", "Flipkart", "https://f.com/1", desc="X" * 200)]
    result = filter_valid(jobs)
    assert len(result) == 1
    
# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    print("\n=== Running Unit Tests for fetch_jobs.py ===\n")
    
    # This programmatically invokes pytest on this specific file.
    # The "-v" flag ensures it prints each test name and its pass/fail status.
    # sys.exit ensures the script returns the correct success/failure code to the terminal.
    sys.exit(pytest.main(["-v", __file__]))