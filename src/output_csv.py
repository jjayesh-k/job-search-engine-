"""
output_csv.py — Writes the daily job shortlist to a clean CSV.

Produces two files every run:
  outputs/shortlist_YYYY-MM-DD.csv  — dated archive copy
  outputs/shortlist_latest.csv      — always-overwritten, for quick access

Columns (in order):
  rank, fit_score, fit_label, title, company, location,
  matched_skills, missing_skills, salary, posted_date,
  source, url, llm_summary
"""

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")

CSV_COLUMNS = [
    "rank",
    "fit_score",
    "fit_label",
    "title",
    "company",
    "location",
    "matched_skills",
    "missing_skills",
    "salary",
    "posted_date",
    "source",
    "url",
    "llm_summary",
]


# ── Salary formatter ───────────────────────────────────────────────────────────

def format_salary(job) -> str:
    """
    Build a human-readable salary string from whatever fields are available.
    Adzuna gives salary_min / salary_max as floats.
    Serpapi gives salary_note as a raw string like '₹8L – ₹12L PA'.
    """
    if job.salary_min and job.salary_max:
        return f"Rs.{int(job.salary_min):,} - Rs.{int(job.salary_max):,}"
    if job.salary_min:
        return f"Rs.{int(job.salary_min):,}+"
    if getattr(job, "salary_note", None):
        return str(job.salary_note)
    return ""


# ── Row builder ────────────────────────────────────────────────────────────────

def _build_row(rank: int, job) -> dict:
    """Convert a ScoredJob into a flat CSV row dict."""
    return {
        "rank":           rank,
        "fit_score":      f"{job.fit_score:.3f}",
        "fit_label":      job.fit_label(),
        "title":          job.title,
        "company":        job.company,
        "location":       job.location,
        "matched_skills": ", ".join(job.matched_skills),
        "missing_skills": ", ".join(job.missing_skills),
        "salary":         format_salary(job),
        "posted_date":    job.posted_date,
        "source":         job.source,
        "url":            job.url,
        "llm_summary":    job.llm_summary,
    }


# ── Main writer ────────────────────────────────────────────────────────────────

def write_shortlist_csv(
    scored_jobs: list,
    min_score: Optional[float] = None,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, list]:
    """
    Write all jobs above min_score to a dated CSV and shortlist_latest.csv.

    Args:
        scored_jobs: Sorted list of ScoredJob (highest fit first).
        min_score:   Minimum fit_score to include. Reads from config if None.
        output_dir:  Directory to write into (created if missing).

    Returns:
        (dated_csv_path, shortlist) — the path written and the jobs included.
    """
    if min_score is None:
        from src.config import MIN_FIT_SCORE
        min_score = MIN_FIT_SCORE

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter — keep all jobs above threshold, sorted by score (already sorted)
    shortlist = [j for j in scored_jobs if j.fit_score >= min_score]

    if not shortlist:
        log.warning(
            f"No jobs above threshold {min_score}. "
            "Try lowering MIN_FIT_SCORE in .env."
        )

    rows = [_build_row(i + 1, job) for i, job in enumerate(shortlist)]

    today = date.today().isoformat()
    dated_path  = output_dir / f"shortlist_{today}.csv"
    latest_path = output_dir / "shortlist_latest.csv"

    for path in [dated_path, latest_path]:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            # utf-8-sig adds BOM so Excel opens it correctly without
            # showing garbled characters for Indian company names / locations
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    log.info(f"CSV written  : {dated_path}  ({len(shortlist)} jobs)")
    log.info(f"Latest CSV   : {latest_path}")

    return dated_path, shortlist


# ── Summary printer ────────────────────────────────────────────────────────────

def print_shortlist_summary(scored_jobs: list, min_score: Optional[float] = None) -> None:
    """Print a formatted summary table of the shortlist to stdout."""
    if min_score is None:
        from src.config import MIN_FIT_SCORE
        min_score = MIN_FIT_SCORE

    shortlist = [j for j in scored_jobs if j.fit_score >= min_score]

    print(f"\n  {'Score':<8} {'Label':<12} {'Title':<35} {'Company':<22} Missing skills")
    print(f"  {'-'*8} {'-'*12} {'-'*35} {'-'*22} {'-'*25}")
    for job in shortlist:
        missing = ", ".join(job.missing_skills[:3]) or "none"
        print(
            f"  {job.fit_score:.3f}    {job.fit_label():<12} "
            f"{job.title[:34]:<35} {job.company[:21]:<22} {missing}"
        )
    print(f"\n  Total in shortlist: {len(shortlist)}")


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    scored_path = Path("outputs/scored_jobs.json")
    if not scored_path.exists():
        print("Run Step 3 first: python main.py --score-only")
        raise SystemExit(1)

    with open(scored_path, encoding="utf-8") as f:
        data = json.load(f)

    from src.score_jobs import ScoredJob
    jobs = [
        ScoredJob(**{k: v for k, v in d.items() if k != "fit_label"})
        for d in data
    ]

    print_shortlist_summary(jobs)
    csv_path, shortlist = write_shortlist_csv(jobs)
    print(f"\nWrote {len(shortlist)} jobs to {csv_path}")