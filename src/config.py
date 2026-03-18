"""
config.py — Central configuration for the job search pipeline.
All settings are loaded from .env — nothing is hardcoded.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── API Credentials ────────────────────────────────────────────────────────────
ADZUNA_APP_ID  = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY", "")

# ── Search Parameters ──────────────────────────────────────────────────────────
TARGET_COUNTRY = os.getenv("TARGET_COUNTRY", "in")

# All roles we want to search for — covers DS, ML, AI, Analyst, MLOps
TARGET_ROLES = [
    "Data Scientist",
    "Machine Learning Engineer",
    "AI Engineer",
    "Data Analyst",
    "Business Analyst",
    "Analytics Engineer",
    "MLOps Engineer",
    "ML Researcher",
    "Computer Vision Engineer",
    "NLP Engineer",
]

# Adzuna: results per role per API call (max 50)
ADZUNA_RESULTS_PER_ROLE = 20

# Serpapi: results per role (Google Jobs returns ~10 per page)
SERPAPI_RESULTS_PER_ROLE = 10

# Minimum fit score to appear in the final shortlist (set in .env or default)
MIN_FIT_SCORE = float(os.getenv("MIN_FIT_SCORE", "0.65"))

# ── Adzuna Country Codes ───────────────────────────────────────────────────────
# Reference: https://developer.adzuna.com/overview
ADZUNA_COUNTRY_MAP = {
    "in": "in",
    "us": "us",
    "gb": "gb",
    "au": "au",
    "ca": "ca",
}

# ... (Keep all your existing code above this line) ...

if __name__ == "__main__":
    print("\n=== Pipeline Configuration Test ===")
    
    # Helper to mask keys for safe console output
    def mask_key(key):
        if not key:
            return "[MISSING OR EMPTY]"
        if len(key) <= 6:
            return "***"
        return f"{key[:3]}...{key[-3:]}"

    print(f"ADZUNA_APP_ID:           {mask_key(ADZUNA_APP_ID)}")
    print(f"ADZUNA_APP_KEY:          {mask_key(ADZUNA_APP_KEY)}")
    print(f"SERPAPI_KEY:             {mask_key(SERPAPI_KEY)}")
    
    print("\n--- Search Parameters ---")
    print(f"Target Country:          {TARGET_COUNTRY}")
    # Verify the country code maps correctly
    adzuna_mapped_country = ADZUNA_COUNTRY_MAP.get(TARGET_COUNTRY, "INVALID CODE")
    print(f"Adzuna Mapped Country:   {adzuna_mapped_country}")
    
    print(f"Min Fit Score:           {MIN_FIT_SCORE} (Type: {type(MIN_FIT_SCORE).__name__})")
    print(f"Adzuna Results/Role:     {ADZUNA_RESULTS_PER_ROLE}")
    print(f"Serpapi Results/Role:    {SERPAPI_RESULTS_PER_ROLE}")
    
    print("\n--- Target Roles ---")
    print(f"Loaded {len(TARGET_ROLES)} roles:")
    for role in TARGET_ROLES:
        print(f"  - {role}")
        
    print("===================================\n")