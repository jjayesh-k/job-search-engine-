"""
main.py — Entry point for the job search pipeline.

Flags:
  --fetch-only        Step 2: fetch jobs from Adzuna + Serpapi
  --score-only        Step 3: score from cached outputs/raw_jobs.json
  --notify-only       Step 4/5: write CSV + XLSX + send Gmail
  --fetch-and-score   Steps 2 + 3
  --no-llm            Skip Ollama summaries (faster, used in CI)
  (no flag)           Full pipeline: Steps 2 → 3 → CSV → XLSX → Email

Usage examples:
  python main.py --fetch-only
  python main.py --score-only --no-llm
  python main.py --notify-only
  python main.py --fetch-and-score --no-llm
  python main.py --no-llm                    # full run, no Ollama
  python main.py                             # full run with Ollama
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)


# ── Helper: load ScoredJob list from outputs/scored_jobs.json ─────────────────

def _load_scored_jobs() -> list:
    """Deserialise ScoredJob objects from the cached JSON produced by Step 3."""
    from src.score_jobs import ScoredJob

    scored_path = Path("outputs") / "scored_jobs.json"
    if not scored_path.exists():
        print("ERROR: outputs/scored_jobs.json not found.")
        print("Run Step 3 first:  python main.py --score-only")
        raise SystemExit(1)

    with open(scored_path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        ScoredJob(
            id=d.get("id", ""),
            title=d.get("title", ""),
            company=d.get("company", ""),
            location=d.get("location", ""),
            description=d.get("description", ""),
            url=d.get("url", ""),
            salary_min=d.get("salary_min"),
            salary_max=d.get("salary_max"),
            salary_note=d.get("salary_note"),
            posted_date=d.get("posted_date", ""),
            source=d.get("source", ""),
            fit_score=d.get("fit_score", 0.0),
            semantic_score=d.get("semantic_score", 0.0),
            skill_score=d.get("skill_score", 0.0),
            matched_skills=d.get("matched_skills", []),
            missing_skills=d.get("missing_skills", []),
            bonus_skills=d.get("bonus_skills", []),
            llm_summary=d.get("llm_summary", ""),
        )
        for d in data
    ]


# ── Step 2: Fetch ──────────────────────────────────────────────────────────────

def run_fetch_only() -> list[dict]:
    """Fetch fresh jobs from Adzuna + Serpapi → outputs/raw_jobs.json."""
    from src.fetch_jobs import fetch_all_jobs
    from src.config import TARGET_ROLES

    print("\n" + "=" * 60)
    print("  STEP 2 — Job API Fetch")
    print("=" * 60 + "\n")

    jobs = fetch_all_jobs(roles=TARGET_ROLES)

    output_path = Path("outputs") / "raw_jobs.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Fetch Summary")
    print(f"{'=' * 60}")
    print(f"  Total unique jobs  : {len(jobs)}")
    print(f"  From Adzuna        : {sum(1 for j in jobs if j['source'] == 'adzuna')}")
    print(f"  From Serpapi       : {sum(1 for j in jobs if j['source'] == 'serpapi')}")
    print(f"\n  Top roles found:")
    for role, count in Counter(j["title"] for j in jobs).most_common(5):
        print(f"    {role:<40} {count} jobs")
    print(f"\n  Saved to: {output_path}")
    return jobs


# ── Step 3: Score ──────────────────────────────────────────────────────────────

def run_score_only(use_llm: bool = True) -> list:
    """Score cached raw_jobs.json against resume → outputs/scored_jobs.json."""
    from src.resume_loader import load_resume
    from src.score_jobs import score_jobs
    from src.config import MIN_FIT_SCORE

    print("\n" + "=" * 60)
    print("  STEP 3 — RAG Scoring Engine")
    print("=" * 60 + "\n")

    # PDF preferred, fall back to .txt if present
    resume_path = os.getenv("RESUME_PATH", "data/resume.pdf")
    if not Path(resume_path).exists() and Path("data/resume.txt").exists():
        resume_path = "data/resume.txt"

    print(f"Loading resume from: {resume_path}")
    try:
        resume = load_resume(resume_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1)

    print(f"  Skills detected : {', '.join(resume.skills[:10])}")
    print(f"  Word count      : {resume.word_count()}\n")

    raw_path = Path("outputs") / "raw_jobs.json"
    if not raw_path.exists():
        print("ERROR: outputs/raw_jobs.json not found.")
        print("Run Step 2 first:  python main.py --fetch-only")
        raise SystemExit(1)

    with open(raw_path, encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"Loaded {len(jobs)} jobs from {raw_path}\n")

    scored = score_jobs(jobs, resume, use_llm=use_llm)

    scored_path = Path("outputs") / "scored_jobs.json"
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump([j.to_dict() for j in scored], f, indent=2, ensure_ascii=False)

    shortlist = [j for j in scored if j.fit_score >= MIN_FIT_SCORE]
    print(f"\n{'=' * 60}")
    print(f"  Scoring Summary")
    print(f"{'=' * 60}")
    print(f"  Total scored     : {len(scored)}")
    print(f"  Above threshold  : {len(shortlist)} (fit >= {MIN_FIT_SCORE})")
    if scored:
        avg = sum(j.fit_score for j in scored) / len(scored)
        print(f"  Avg fit score    : {avg:.3f}")
    print(f"\n  Top 10 Matches:")
    print(f"  {'Score':<8} {'Label':<12} {'Title':<35} Company")
    print(f"  {'-'*8} {'-'*12} {'-'*35} {'-'*20}")
    for job in scored[:10]:
        print(
            f"  {job.fit_score:.3f}    {job.fit_label():<12} "
            f"{job.title[:34]:<35} {job.company[:20]}"
        )
    print(f"\n  Full results saved to: {scored_path}")
    return scored


# ── Step 4/5: CSV + XLSX + Email ───────────────────────────────────────────────

def run_notify_only() -> bool:
    """
    Load scored_jobs.json → write CSV → write XLSX → send Gmail.
    Safe to run standalone after --score-only.
    Returns True if email was sent successfully.
    """
    from dotenv import load_dotenv
    from src.output_csv import write_shortlist_csv, print_shortlist_summary
    from src.output_xlsx import write_shortlist_xlsx
    from src.notify import send_email
    from src.config import MIN_FIT_SCORE

    load_dotenv()

    print("\n" + "=" * 60)
    print("  STEP 4/5 — CSV + XLSX + Gmail Notification")
    print("=" * 60 + "\n")

    all_jobs = _load_scored_jobs()
    shortlist = [j for j in all_jobs if j.fit_score >= MIN_FIT_SCORE]

    print(f"  Loaded         : {len(all_jobs)} scored jobs")
    print(f"  Above threshold: {len(shortlist)} (fit >= {MIN_FIT_SCORE})\n")

    # Print terminal summary
    print_shortlist_summary(all_jobs)

    # Write CSV (dated + latest)
    csv_path, csv_written = write_shortlist_csv(all_jobs)
    print(f"\n  CSV  → {csv_path}  ({len(csv_written)} jobs)")

    # Write XLSX (dated + latest)
    xlsx_path, xlsx_written = write_shortlist_xlsx(all_jobs)
    print(f"  XLSX → {xlsx_path}  ({len(xlsx_written)} jobs)")

    # Send Gmail with CSV attached
    ok = send_email(
        shortlist=shortlist,
        total_scored=len(all_jobs),
        csv_path=csv_path,
    )
    print(f"\n  Email → {'SENT ✓' if ok else 'FAILED — check logs above'}")
    return ok


# ── Composed runners ───────────────────────────────────────────────────────────

def run_fetch_and_score(use_llm: bool = True) -> list:
    """Steps 2 + 3: fresh fetch then score."""
    run_fetch_only()
    return run_score_only(use_llm=use_llm)


def run_full_pipeline(use_llm: bool = True) -> None:
    """
    Steps 2 → 3 → CSV → XLSX → Email.
    Mirrors exactly what GitHub Actions runs daily.
    """
    run_fetch_only()
    run_score_only(use_llm=use_llm)
    run_notify_only()
    print("\n✓ Pipeline complete. Check your inbox and outputs/ folder.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Job Search Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --fetch-only                # Step 2 only
  python main.py --score-only --no-llm       # Step 3, no Ollama (fast)
  python main.py --score-only                # Step 3 with Ollama summaries
  python main.py --notify-only               # CSV + XLSX + email (cached scores)
  python main.py --fetch-and-score --no-llm  # Steps 2+3, no Ollama
  python main.py --no-llm                    # Full pipeline, no Ollama
  python main.py                             # Full pipeline with Ollama
        """,
    )
    parser.add_argument(
        "--fetch-only", action="store_true",
        help="Step 2: fetch jobs from Adzuna + Serpapi",
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Step 3: score cached raw_jobs.json against resume",
    )
    parser.add_argument(
        "--notify-only", action="store_true",
        help="Step 4/5: write CSV + XLSX + send Gmail from cached scored_jobs.json",
    )
    parser.add_argument(
        "--fetch-and-score", action="store_true",
        help="Steps 2+3: fresh fetch then score",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip Ollama LLM summaries (faster — used automatically in CI)",
    )

    args = parser.parse_args()
    use_llm = not args.no_llm

    if args.fetch_only:
        run_fetch_only()
    elif args.score_only:
        run_score_only(use_llm=use_llm)
    elif args.notify_only:
        run_notify_only()
    elif args.fetch_and_score:
        run_fetch_and_score(use_llm=use_llm)
    else:
        run_full_pipeline(use_llm=use_llm)