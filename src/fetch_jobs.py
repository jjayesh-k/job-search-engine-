"""
fetch_jobs.py — Dual-source job fetcher (Adzuna + Serpapi).

Flow:
  1. fetch_adzuna()  → pulls jobs from Adzuna REST API
  2. fetch_serpapi() → pulls jobs from Google Jobs via Serpapi
  3. fetch_all_jobs() → runs both, merges, deduplicates, returns clean list

Each job is normalized into a standard dict:
  {
    "id":          str,   # unique hash for deduplication
    "title":       str,
    "company":     str,
    "location":    str,
    "description": str,
    "url":         str,
    "salary_min":  float | None,
    "salary_max":  float | None,
    "posted_date": str,
    "source":      str,   # "adzuna" | "serpapi"
  }
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import Optional

import requests

from src.config import (
    ADZUNA_APP_ID,
    ADZUNA_APP_KEY,
    ADZUNA_COUNTRY_MAP,
    ADZUNA_RESULTS_PER_ROLE,
    SERPAPI_KEY,
    SERPAPI_RESULTS_PER_ROLE,
    TARGET_COUNTRY,
    TARGET_ROLES,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_job_id(title: str, company: str, url: str) -> str:
    """
    Create a stable unique ID for a job posting.
    Used to deduplicate across Adzuna and Serpapi results.
    """
    raw = f"{title.lower().strip()}|{company.lower().strip()}|{url.strip()}"
    return hashlib.md5(raw.encode()).hexdigest()


def _safe_get(d: dict, *keys, default=None):
    """Safely traverse nested dicts without KeyError."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


# ── Adzuna Fetcher ─────────────────────────────────────────────────────────────

def fetch_adzuna(
    role: str,
    country: str = TARGET_COUNTRY,
    results: int = ADZUNA_RESULTS_PER_ROLE,
) -> list[dict]:
    """
    Fetch jobs for a single role from the Adzuna API.

    Adzuna API docs: https://developer.adzuna.com/activedocs
    Free tier: 250 calls/month

    Args:
        role:    Job title to search (e.g. "Data Scientist")
        country: 2-letter country code (e.g. "in")
        results: Number of results to request (max 50 per call)

    Returns:
        List of normalized job dicts.
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        log.warning("Adzuna credentials missing — skipping Adzuna fetch.")
        return []

    country_code = ADZUNA_COUNTRY_MAP.get(country, "in")
    url = (
        f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
    )

    params = {
        "app_id":          ADZUNA_APP_ID,
        "app_key":         ADZUNA_APP_KEY,
        "what":            role,
        "where":           "India",
        "results_per_page": results,
        # "content-type":    "application/json",
        "sort_by":         "date",           # freshest jobs first
        "full_description": 1,               # get full JD text for RAG
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        log.error(f"Adzuna HTTP error for '{role}': {e}")
        return []
    except requests.exceptions.RequestException as e:
        log.error(f"Adzuna request failed for '{role}': {e}")
        return []
    except ValueError:
        log.error(f"Adzuna returned non-JSON for '{role}'")
        return []

    jobs = []
    for item in data.get("results", []):
        title   = item.get("title", "").strip()
        company = _safe_get(item, "company", "display_name", default="Unknown")
        loc     = _safe_get(item, "location", "display_name", default="India")
        desc    = item.get("description", "")
        job_url = item.get("redirect_url", "")

        # Salary (Adzuna gives min/max in local currency)
        sal_min = item.get("salary_min")
        sal_max = item.get("salary_max")

        # Posted date — Adzuna gives ISO 8601
        created = item.get("created", "")
        try:
            posted = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            posted = ""

        job_id = _make_job_id(title, company, job_url)

        jobs.append({
            "id":          job_id,
            "title":       title,
            "company":     company,
            "location":    loc,
            "description": desc,
            "url":         job_url,
            "salary_min":  sal_min,
            "salary_max":  sal_max,
            "posted_date": posted,
            "source":      "adzuna",
        })

    log.info(f"  Adzuna → '{role}': {len(jobs)} jobs fetched")
    return jobs


# ── Serpapi Fetcher ────────────────────────────────────────────────────────────

def fetch_serpapi(
    role: str,
    country: str = TARGET_COUNTRY,
    results: int = SERPAPI_RESULTS_PER_ROLE,
) -> list[dict]:
    """
    Fetch jobs for a single role from Google Jobs via Serpapi.

    Serpapi docs: https://serpapi.com/google-jobs-api
    Free tier: 100 searches/month

    Args:
        role:    Job title to search
        country: 2-letter country code
        results: Number of results (Google Jobs returns ~10 per page)

    Returns:
        List of normalized job dicts.
    """
    if not SERPAPI_KEY:
        log.warning("Serpapi key missing — skipping Serpapi fetch.")
        return []

    url = "https://serpapi.com/search.json"

    params = {
        "engine":   "google_jobs",
        "q":        f"{role} India",
        "hl":       "en",
        "gl":       country,
        "api_key":  SERPAPI_KEY,
        "num":      results,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        log.error(f"Serpapi HTTP error for '{role}': {e}")
        return []
    except requests.exceptions.RequestException as e:
        log.error(f"Serpapi request failed for '{role}': {e}")
        return []
    except ValueError:
        log.error(f"Serpapi returned non-JSON for '{role}'")
        return []

    jobs = []
    for item in data.get("jobs_results", []):
        title   = item.get("title", "").strip()
        company = item.get("company_name", "Unknown").strip()
        loc     = item.get("location", "India").strip()

        # Serpapi nests the full description under "description"
        desc = item.get("description", "")

        # Serpapi gives a "via" field and a "share_link" — we want the apply link
        # Primary: related_links[0], fallback: share_link
        related = item.get("related_links", [])
        job_url = related[0].get("link", "") if related else item.get("share_link", "")

        # Salary — Serpapi surfaces it inconsistently; grab if present
        extensions = item.get("detected_extensions", {})
        sal_min = None
        sal_max = None
        salary_raw = extensions.get("salary", "")
        # We'll parse salary properly in Step 5; store raw string for now
        salary_note = salary_raw if salary_raw else None

        # Posted date — Serpapi gives relative strings like "3 days ago"
        posted_raw = extensions.get("posted_at", "")
        posted = posted_raw  # We store as-is; normalize in output step

        job_id = _make_job_id(title, company, job_url)

        jobs.append({
            "id":          job_id,
            "title":       title,
            "company":     company,
            "location":    loc,
            "description": desc,
            "url":         job_url,
            "salary_min":  sal_min,
            "salary_max":  sal_max,
            "salary_note": salary_note,   # extra field for Serpapi salary string
            "posted_date": posted,
            "source":      "serpapi",
        })

    log.info(f"  Serpapi → '{role}': {len(jobs)} jobs fetched")
    return jobs


# ── Deduplicator ───────────────────────────────────────────────────────────────

def deduplicate(jobs: list[dict]) -> list[dict]:
    """
    Remove duplicate jobs across sources using the stable job ID.
    When the same job appears in both Adzuna and Serpapi,
    we keep the Adzuna version (richer description).
    """
    seen: set[str] = set()
    unique: list[dict] = []

    # Sort so Adzuna jobs are processed first (kept over Serpapi dupes)
    sorted_jobs = sorted(jobs, key=lambda j: 0 if j["source"] == "adzuna" else 1)

    for job in sorted_jobs:
        if job["id"] not in seen:
            seen.add(job["id"])
            unique.append(job)

    removed = len(jobs) - len(unique)
    if removed:
        log.info(f"  Deduplication removed {removed} duplicate(s) → {len(unique)} unique jobs")

    return unique


# ── Filter: remove jobs with empty descriptions ────────────────────────────────

def filter_valid(jobs: list[dict]) -> list[dict]:
    """
    Drop jobs that have no description — they can't be RAG-scored.
    Also drop jobs with titles that are clearly off-target.
    """
    off_target_keywords = [
        "sales", "marketing", "hr ", "human resources",
        "finance manager", "accountant", "lawyer",
    ]

    valid = []
    for job in jobs:
        # Must have a description of at least 80 characters for meaningful scoring
        if len(job.get("description", "")) < 80:
            continue

        # Reject off-target titles
        title_lower = job["title"].lower()
        if any(kw in title_lower for kw in off_target_keywords):
            continue

        valid.append(job)

    removed = len(jobs) - len(valid)
    if removed:
        log.info(f"  Filtered out {removed} invalid/off-target job(s)")

    return valid


# ── Main Entry Point ───────────────────────────────────────────────────────────

def fetch_all_jobs(
    roles: list[str] = TARGET_ROLES,
    delay_seconds: float = 1.0,
) -> list[dict]:
    """
    Fetch jobs for all target roles from both Adzuna and Serpapi.
    Merges, deduplicates, and returns a clean list ready for RAG scoring.

    Args:
        roles:         List of job titles to search.
        delay_seconds: Pause between API calls to respect rate limits.

    Returns:
        List of unique, valid, normalized job dicts.
    """
    log.info(f"Starting job fetch for {len(roles)} roles across 2 sources...")
    all_jobs: list[dict] = []

    for role in roles:
        log.info(f"Fetching: '{role}'")

        # Adzuna
        adzuna_jobs = fetch_adzuna(role)
        all_jobs.extend(adzuna_jobs)
        time.sleep(delay_seconds)

        # Serpapi
        serpapi_jobs = fetch_serpapi(role)
        all_jobs.extend(serpapi_jobs)
        time.sleep(delay_seconds)

    log.info(f"Raw total across all roles: {len(all_jobs)} jobs")

    # Deduplicate across sources and roles
    all_jobs = deduplicate(all_jobs)

    # Filter out jobs with no useful description
    all_jobs = filter_valid(all_jobs)

    log.info(f"Final job pool ready for scoring: {len(all_jobs)} jobs")
    return all_jobs


# ── Quick standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with just one role to save API quota during development
    test_jobs = fetch_all_jobs(roles=["Data Scientist"], delay_seconds=0.5)

    print(f"\n{'='*60}")
    print(f"Sample output — first 2 jobs:")
    print(f"{'='*60}")

    for job in test_jobs[:2]:
        print(f"\nTitle:       {job['title']}")
        print(f"Company:     {job['company']}")
        print(f"Location:    {job['location']}")
        print(f"Source:      {job['source']}")
        print(f"URL:         {job['url']}")
        print(f"Posted:      {job['posted_date']}")
        print(f"Desc length: {len(job['description'])} chars")
        print(f"ID:          {job['id']}")
        
# ── Enhanced Standalone Test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    
    print("\n=== Starting Job Fetcher Test ===")
    
    # We use just one role for testing to avoid burning through your free SerpApi quota
    test_roles = ["Data Scientist"]
    
    try:
        # Run the fetcher
        test_jobs = fetch_all_jobs(roles=test_roles, delay_seconds=1.0)
        
        print(f"\n{'='*60}")
        print(f"Test complete! Successfully processed {len(test_jobs)} valid jobs.")
        print(f"{'='*60}")

        if test_jobs:
            print("\nPreview of the first job:")
            first_job = test_jobs[0]
            for key, value in first_job.items():
                # Truncate the description so it doesn't flood your console
                if key == "description" and value:
                    print(f"  {key.ljust(12)}: {value[:100]}... [Truncated]")
                else:
                    print(f"  {key.ljust(12)}: {value}")
            
            # Save the raw list of dicts to a JSON file for easy inspection
            output_file = "test_jobs_output.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(test_jobs, f, indent=2, ensure_ascii=False)
                
            print(f"\n[✔] Full scraped results saved to '{output_file}'")
            print("    Open this file to verify the descriptions look good for your RAG prompt!")
            
        else:
            print("\n[!] No jobs returned. Double-check your .env file to ensure API keys are loaded.")

    except Exception as e:
        print(f"\n[ERROR] The test crashed: {e}")