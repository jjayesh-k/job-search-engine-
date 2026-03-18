"""
main.py — Entry point for the job search pipeline.

Steps:
  Step 2: Fetch jobs from Adzuna + Serpapi   (--fetch-only)
  Step 3: Score jobs via RAG + Mistral       (--score-only)
  Step 2+3: Fetch then score                (--fetch-and-score)
  Full pipeline (Steps 2-5):                (no flag)

Usage:
    python main.py --fetch-only        # Step 2: fetch raw jobs
    python main.py --score-only        # Step 3: score from cached raw_jobs.json
    python main.py --fetch-and-score   # Steps 2+3: fetch fresh + score
    python main.py --no-llm            # Skip Ollama summaries (faster testing)
    python main.py                     # Full pipeline (Steps 2-5)
"""

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run_fetch_only() -> list[dict]:
    """Step 2: Fetch jobs from Adzuna + Serpapi."""
    from src.fetch_jobs import fetch_all_jobs
    from src.config import TARGET_ROLES
    from collections import Counter

    print("\n" + "=" * 60)
    print("  STEP 2 - Job API Fetch")
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


def run_score_only(use_llm: bool = True) -> list:
    """Step 3: Load cached raw jobs, score them, save scored output."""
    import os
    from src.resume_loader import load_resume
    from src.score_jobs import score_jobs
    from src.config import MIN_FIT_SCORE

    print("\n" + "=" * 60)
    print("  STEP 3 - RAG Scoring Engine")
    print("=" * 60 + "\n")

    # Load resume
    resume_path = os.getenv("RESUME_PATH", "data/resume.pdf")
    print(f"Loading resume from: {resume_path}")
    try:
        resume = load_resume(resume_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1)

    print(f"  Skills detected : {', '.join(resume.skills[:10])}")
    print(f"  Word count      : {resume.word_count()}\n")

    # Load raw jobs from Step 2
    raw_path = Path("outputs") / "raw_jobs.json"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found.")
        print("Run Step 2 first:  python main.py --fetch-only")
        raise SystemExit(1)

    with open(raw_path, encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"Loaded {len(jobs)} jobs from {raw_path}\n")

    # Score all jobs
    scored = score_jobs(jobs, resume, use_llm=use_llm)

    # Save full scored output as JSON
    scored_path = Path("outputs") / "scored_jobs.json"
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump([j.to_dict() for j in scored], f, indent=2, ensure_ascii=False)

    # Print summary
    shortlist = [j for j in scored if j.fit_score >= MIN_FIT_SCORE]

    print(f"\n{'=' * 60}")
    print(f"  Scoring Summary")
    print(f"{'=' * 60}")
    print(f"  Total scored        : {len(scored)}")
    print(f"  Above threshold     : {len(shortlist)} (fit >= {MIN_FIT_SCORE})")
    if scored:
        avg = sum(j.fit_score for j in scored) / len(scored)
        print(f"  Average fit score   : {avg:.3f}")

    print(f"\n  Top 10 Matches:")
    print(f"  {'Score':<8} {'Label':<12} {'Title':<35} Company")
    print(f"  {'-'*8} {'-'*12} {'-'*35} {'-'*20}")
    for job in scored[:10]:
        print(f"  {job.fit_score:.3f}    {job.fit_label():<12} {job.title[:34]:<35} {job.company[:20]}")

    print(f"\n  Full results saved to: {scored_path}")
    print(f"  (Step 5 will convert this to a clean daily CSV shortlist)")

    return scored


def run_fetch_and_score(use_llm: bool = True) -> list:
    """Steps 2 + 3: Fresh fetch then immediate scoring."""
    run_fetch_only()
    return run_score_only(use_llm=use_llm)


def run_full_pipeline(use_llm: bool = True):
    """All steps in sequence. Expands as Steps 4 and 5 are built."""
    scored = run_fetch_and_score(use_llm=use_llm)
    print("\n[Step 4: GitHub Actions automation - configured next]")
    print("[Step 5: CSV shortlist output - coming after Step 4]")
    return scored


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Job Search Pipeline")
    parser.add_argument("--fetch-only",      action="store_true", help="Step 2 only: fetch jobs")
    parser.add_argument("--score-only",      action="store_true", help="Step 3 only: score cached jobs")
    parser.add_argument("--fetch-and-score", action="store_true", help="Steps 2+3: fetch then score")
    parser.add_argument("--no-llm",          action="store_true", help="Skip Ollama LLM summaries (faster)")
    args = parser.parse_args()

    use_llm = not args.no_llm

    if args.fetch_only:
        run_fetch_only()
    elif args.score_only:
        run_score_only(use_llm=use_llm)
    elif args.fetch_and_score:
        run_fetch_and_score(use_llm=use_llm)
    else:
        run_full_pipeline(use_llm=use_llm)