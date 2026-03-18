"""
main.py — Entry point for the job search pipeline.

Run this file to execute the full pipeline.
During development, each step is gated so you can test incrementally.

Usage:
    python main.py              # runs full pipeline
    python main.py --fetch-only # runs Step 2 only (job fetching)
"""

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run_fetch_only():
    """Step 2: Fetch jobs from Adzuna + Serpapi and print a summary."""
    from src.fetch_jobs import fetch_all_jobs
    from src.config import TARGET_ROLES

    print("\n" + "=" * 60)
    print("  STEP 2 — Job API Integration Test")
    print("=" * 60 + "\n")

    jobs = fetch_all_jobs(roles=TARGET_ROLES)

    # Save raw results to outputs/ for inspection
    output_path = Path("outputs") / "raw_jobs.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Results Summary")
    print(f"{'=' * 60}")
    print(f"  Total unique jobs fetched : {len(jobs)}")

    # Breakdown by source
    adzuna_count  = sum(1 for j in jobs if j["source"] == "adzuna")
    serpapi_count = sum(1 for j in jobs if j["source"] == "serpapi")
    print(f"  From Adzuna               : {adzuna_count}")
    print(f"  From Serpapi              : {serpapi_count}")

    # Breakdown by role (top 5)
    from collections import Counter
    role_counts = Counter(j["title"] for j in jobs)
    print(f"\n  Top roles found:")
    for role, count in role_counts.most_common(5):
        print(f"    {role:<40} {count} jobs")

    print(f"\n  Raw output saved to: {output_path}")
    print(f"\n  Sample job:")
    if jobs:
        sample = jobs[0]
        print(f"    Title:   {sample['title']}")
        print(f"    Company: {sample['company']}")
        print(f"    Loc:     {sample['location']}")
        print(f"    Source:  {sample['source']}")
        print(f"    Desc:    {sample['description'][:120]}...")

    return jobs


def run_full_pipeline():
    """Runs all steps in sequence. Expands as we build each step."""
    jobs = run_fetch_only()
    # Step 3 (RAG scoring) will be called here next
    # Step 4 (GitHub Actions) is config, not runtime
    # Step 5 (CSV output) will be called here after scoring
    print("\n[Step 3, 4, 5 coming soon — run step by step for now]")
    return jobs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Job Search Pipeline")
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Run Step 2 only: fetch jobs and save raw JSON",
    )
    args = parser.parse_args()

    if args.fetch_only:
        run_fetch_only()
    else:
        run_full_pipeline()
        
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