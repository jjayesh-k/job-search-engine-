"""
output_csv.py — Writes the daily job shortlist to a clean CSV file.

Produces two files:
  outputs/shortlist_YYYY-MM-DD.csv  — today's top matches above MIN_FIT_SCORE
  outputs/shortlist_latest.csv      — always-overwritten for easy access

CSV columns:
  rank, fit_score, fit_label, title, company, location,
  matched_skills, missing_skills, salary, posted_date,
  source, url, llm_summary
"""

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from src.config import MIN_FIT_SCORE
from src.score_jobs import ScoredJob

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")

CSV_COLUMNS = [
    "rank", "fit_score", "fit_label", "title", "company", "location",
    "matched_skills", "missing_skills", "salary", "posted_date",
    "source", "url", "llm_summary",
]


def _format_salary(job: ScoredJob) -> str:
    if job.salary_min and job.salary_max:
        return f"Rs.{job.salary_min:,.0f} - Rs.{job.salary_max:,.0f}"
    if job.salary_min:
        return f"Rs.{job.salary_min:,.0f}+"
    if job.salary_note:
        return str(job.salary_note)
    return ""


def write_shortlist_csv(
    scored_jobs: list,
    min_score: float = MIN_FIT_SCORE,
    top_n: Optional[int] = None,
    output_dir: Path = OUTPUT_DIR,
) -> tuple:
    output_dir.mkdir(parents=True, exist_ok=True)

    shortlist = [j for j in scored_jobs if j.fit_score >= min_score]
    if top_n:
        shortlist = shortlist[:top_n]

    if not shortlist:
        log.warning(f"No jobs above threshold {min_score}.")

    today = date.today().isoformat()
    csv_path = output_dir / f"shortlist_{today}.csv"
    latest_path = output_dir / "shortlist_latest.csv"

    rows = []
    for rank, job in enumerate(shortlist, start=1):
        rows.append({
            "rank":           rank,
            "fit_score":      f"{job.fit_score:.3f}",
            "fit_label":      job.fit_label(),
            "title":          job.title,
            "company":        job.company,
            "location":       job.location,
            "matched_skills": ", ".join(job.matched_skills),
            "missing_skills": ", ".join(job.missing_skills),
            "salary":         _format_salary(job),
            "posted_date":    job.posted_date,
            "source":         job.source,
            "url":            job.url,
            "llm_summary":    job.llm_summary,
        })

    for path in [csv_path, latest_path]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    log.info(f"CSV written: {csv_path} ({len(shortlist)} jobs)")
    return csv_path, shortlist