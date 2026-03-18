"""
score_jobs.py — RAG-based semantic fit scorer.

How it works:
  1. Embed your resume using a local sentence-transformer model
  2. Embed each job description
  3. Compute cosine similarity → base "semantic score"
  4. Run a skill-gap analysis via Mistral (Ollama) → "skill score"
  5. Combine both into a final weighted "fit score"
  6. Return jobs sorted by fit score, each with a skill-gap breakdown

Scoring formula:
  fit_score = (semantic_score * 0.55) + (skill_score * 0.45)

Why this split:
  - Semantic similarity catches domain fit, seniority match, terminology overlap
  - Skill scoring catches exact technical requirements (Python, SQL, PyTorch etc.)
  - 55/45 weighting slightly favours semantic — avoids false negatives where
    your resume uses different but equivalent terms
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
import numpy as np

from src.config import MIN_FIT_SCORE
from src.resume_loader import Resume

log = logging.getLogger(__name__)

# ── Embedding model (local, no API cost) ──────────────────────────────────────
# all-MiniLM-L6-v2: fast, 384-dim, excellent for semantic similarity
# Downloads ~90MB on first run, cached permanently after
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"

# ── Scoring weights ────────────────────────────────────────────────────────────
SEMANTIC_WEIGHT = 0.55
SKILL_WEIGHT    = 0.45

# ── Key skills we always check for (your priority skills) ─────────────────────
PRIORITY_SKILLS = [
    "Python", "SQL", "Machine Learning", "Deep Learning",
    "NLP", "LLMs", "Data Pipelines", "ETL",
    "Tableau", "Power BI", "Pandas", "NumPy",
    "Scikit-learn", "TensorFlow", "PyTorch",
]


# ── Data Model ─────────────────────────────────────────────────────────────────

@dataclass
class ScoredJob:
    """A job posting enriched with fit scores and skill analysis."""

    # Original job fields
    id:           str = ""
    title:        str = ""
    company:      str = ""
    location:     str = ""
    description:  str = ""
    url:          str = ""
    salary_min:   Optional[float] = None
    salary_max:   Optional[float] = None
    salary_note:  Optional[str]   = None
    posted_date:  str = ""
    source:       str = ""

    # Scores
    semantic_score: float = 0.0   # cosine similarity (0–1)
    skill_score:    float = 0.0   # LLM skill match (0–1)
    fit_score:      float = 0.0   # weighted final score (0–1)

    # Skill gap analysis
    matched_skills:  list[str] = field(default_factory=list)  # skills you have that JD wants
    missing_skills:  list[str] = field(default_factory=list)  # skills JD wants but you lack
    bonus_skills:    list[str] = field(default_factory=list)  # skills you have beyond JD

    # LLM reasoning
    llm_summary: str = ""   # 1-sentence LLM explanation of the fit score

    def fit_label(self) -> str:
        """Human-readable fit tier."""
        if self.fit_score >= 0.82:  return "Excellent"
        if self.fit_score >= 0.70:  return "Strong"
        if self.fit_score >= 0.58:  return "Good"
        if self.fit_score >= 0.45:  return "Fair"
        return "Weak"

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "title":          self.title,
            "company":        self.company,
            "location":       self.location,
            "url":            self.url,
            "salary_min":     self.salary_min,
            "salary_max":     self.salary_max,
            "salary_note":    self.salary_note,
            "posted_date":    self.posted_date,
            "source":         self.source,
            "fit_score":      round(self.fit_score, 4),
            "fit_label":      self.fit_label(),
            "semantic_score": round(self.semantic_score, 4),
            "skill_score":    round(self.skill_score, 4),
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "bonus_skills":   self.bonus_skills,
            "llm_summary":    self.llm_summary,
        }


# ── Embedding Engine ───────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps sentence-transformers for local embedding.
    Model is loaded once and reused for all jobs.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            log.info(f"Loading embedding model: {self.model_name}")
            log.info("(First run downloads ~90MB — cached permanently after)")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            log.info("Embedding model ready.")
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Return a normalized embedding vector for a single text."""
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        """Embed a list of texts efficiently in batches."""
        model = self._load()
        log.info(f"Embedding {len(texts)} job descriptions in batches of {batch_size}...")
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return list(vecs)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two normalized vectors.
    Since we use normalize_embeddings=True, this simplifies to dot product.
    """
    return float(np.dot(a, b))


# ── Skill Gap Analyzer ─────────────────────────────────────────────────────────

def _extract_jd_skills(jd_text: str) -> list[str]:
    """
    Extract skill keywords from a job description using the same
    alias map as the resume loader — ensures comparable skill sets.
    """
    import re
    from src.resume_loader import SKILL_ALIASES

    text_lower = jd_text.lower()
    found: set[str] = set()

    sorted_aliases = sorted(SKILL_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text_lower):
            found.add(SKILL_ALIASES[alias])

    return sorted(found)


def _compute_skill_score(
    resume: Resume,
    jd_skills: list[str],
) -> tuple[float, list[str], list[str], list[str]]:
    """
    Compare resume skills vs job description skills.

    Returns:
        (skill_score, matched_skills, missing_skills, bonus_skills)

    Scoring:
        - Start at 0.0
        - Each matched priority skill adds more weight than generic skills
        - Penalise missing priority skills
        - Bonus skills don't affect score (they're a selling point, not a gap)
    """
    resume_skills = set(resume.skills)
    jd_skills_set = set(jd_skills)

    if not jd_skills_set:
        # No skills detected in JD — can't penalise, give neutral score
        return 0.6, [], [], list(resume_skills)

    matched = sorted(resume_skills & jd_skills_set)
    missing = sorted(jd_skills_set - resume_skills)
    bonus   = sorted(resume_skills - jd_skills_set)

    # Priority skills carry 2x weight
    priority_set = set(PRIORITY_SKILLS)

    matched_score = sum(2.0 if s in priority_set else 1.0 for s in matched)
    total_score   = sum(2.0 if s in priority_set else 1.0 for s in jd_skills_set)

    skill_score = matched_score / total_score if total_score > 0 else 0.5
    skill_score = min(1.0, max(0.0, skill_score))

    return round(skill_score, 4), matched, missing, bonus


# ── LLM Reasoning via Ollama ───────────────────────────────────────────────────

def _ask_ollama(prompt: str, timeout: int = 45) -> str:
    """
    Send a prompt to local Ollama (Mistral) and return the response text.
    Falls back to empty string on any failure — scoring continues without LLM.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp for consistent scoring output
            "num_predict": 120,   # limit response length
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        log.warning("Ollama not running — skipping LLM summaries. Start with: ollama serve")
        return ""
    except requests.exceptions.Timeout:
        log.warning("Ollama timed out — skipping LLM summary for this job.")
        return ""
    except Exception as e:
        log.warning(f"Ollama error: {e}")
        return ""


def _generate_llm_summary(
    resume: Resume,
    job: dict,
    semantic_score: float,
    matched_skills: list[str],
    missing_skills: list[str],
) -> str:
    """
    Ask Mistral to generate a 1-sentence fit summary for the job.
    This is the "why" behind the score — shown in the CSV output.
    """
    prompt = f"""You are a career advisor. Given the following, write exactly ONE sentence (max 25 words) explaining why this candidate is or isn't a good fit for this job.

Candidate skills: {', '.join(resume.skills[:15])}
Job title: {job.get('title', '')} at {job.get('company', '')}
Job requires: {', '.join(matched_skills + missing_skills)[:200]}
Semantic match score: {semantic_score:.2f}/1.0
Skills matched: {', '.join(matched_skills[:5])}
Skills missing: {', '.join(missing_skills[:5])}

Write ONE sentence only. Be specific. No preamble."""

    return _ask_ollama(prompt)


# ── Main Scorer ────────────────────────────────────────────────────────────────

class JobScorer:
    """
    Scores a list of job dicts against a resume using RAG + skill analysis.

    Usage:
        scorer = JobScorer(resume)
        scored_jobs = scorer.score(jobs)
    """

    def __init__(self, resume: Resume):
        self.resume  = resume
        self.embedder = EmbeddingEngine()
        self._resume_vec: Optional[np.ndarray] = None

    def _get_resume_vector(self) -> np.ndarray:
        """Embed the resume once and cache it."""
        if self._resume_vec is None:
            log.info("Embedding resume...")
            # Use full text + skills section for richest signal
            resume_text = self.resume.full_text
            if self.resume.skills:
                resume_text += "\n\nKey skills: " + ", ".join(self.resume.skills)
            self._resume_vec = self.embedder.embed(resume_text)
            log.info("Resume embedded.")
        return self._resume_vec

    def _prepare_jd_text(self, job: dict) -> str:
        """
        Combine title + company + description for JD embedding.
        Including the title helps semantic matching for seniority / domain.
        """
        parts = [
            f"Job title: {job.get('title', '')}",
            f"Company: {job.get('company', '')}",
            f"Location: {job.get('location', '')}",
            job.get("description", ""),
        ]
        return "\n".join(p for p in parts if p.strip())

    def score_one(self, job: dict, use_llm: bool = True) -> ScoredJob:
        """
        Score a single job against the resume.

        Args:
            job:     Normalized job dict from fetch_jobs.py
            use_llm: Whether to call Ollama for a summary sentence

        Returns:
            ScoredJob with all scores and skill analysis populated
        """
        resume_vec = self._get_resume_vector()

        # 1. Semantic score via embedding cosine similarity
        jd_text  = self._prepare_jd_text(job)
        jd_vec   = self.embedder.embed(jd_text)
        sem_score = cosine_similarity(resume_vec, jd_vec)

        # Cosine similarity of normalized vectors is in [-1, 1].
        # For resume vs JD it's always positive — normalize to [0, 1]
        sem_score = (sem_score + 1) / 2

        # 2. Skill gap analysis
        jd_skills = _extract_jd_skills(jd_text)
        skill_score, matched, missing, bonus = _compute_skill_score(self.resume, jd_skills)

        # 3. Weighted fit score
        fit_score = (sem_score * SEMANTIC_WEIGHT) + (skill_score * SKILL_WEIGHT)

        # 4. LLM summary (optional — only for jobs above threshold to save time)
        llm_summary = ""
        if use_llm and fit_score >= MIN_FIT_SCORE:
            llm_summary = _generate_llm_summary(
                self.resume, job, sem_score, matched, missing
            )

        return ScoredJob(
            id=job.get("id", ""),
            title=job.get("title", ""),
            company=job.get("company", ""),
            location=job.get("location", ""),
            description=job.get("description", ""),
            url=job.get("url", ""),
            salary_min=job.get("salary_min"),
            salary_max=job.get("salary_max"),
            salary_note=job.get("salary_note"),
            posted_date=job.get("posted_date", ""),
            source=job.get("source", ""),
            semantic_score=round(sem_score, 4),
            skill_score=round(skill_score, 4),
            fit_score=round(fit_score, 4),
            matched_skills=matched,
            missing_skills=missing,
            bonus_skills=bonus,
            llm_summary=llm_summary,
        )

    def score(
        self,
        jobs: list[dict],
        use_llm: bool = True,
        llm_delay: float = 0.5,
    ) -> list[ScoredJob]:
        """
        Score all jobs. Returns sorted list (highest fit first).

        Args:
            jobs:      List of normalized job dicts
            use_llm:   Whether to generate LLM summaries (slower but richer output)
            llm_delay: Seconds to wait between Ollama calls (prevent overload)

        Returns:
            List of ScoredJob sorted by fit_score descending
        """
        if not jobs:
            log.warning("No jobs to score.")
            return []

        log.info(f"Scoring {len(jobs)} jobs...")

        # Pre-embed resume once
        self._get_resume_vector()

        # Batch embed all JDs at once (much faster than one-by-one)
        log.info("Embedding all job descriptions...")
        jd_texts = [self._prepare_jd_text(job) for job in jobs]
        jd_vecs  = self.embedder.embed_batch(jd_texts)

        resume_vec = self._get_resume_vector()
        scored: list[ScoredJob] = []

        for i, (job, jd_vec) in enumerate(zip(jobs, jd_vecs)):
            # Semantic score from pre-computed vectors
            sem_score  = (cosine_similarity(resume_vec, jd_vec) + 1) / 2

            # Skill analysis
            jd_text    = jd_texts[i]
            jd_skills  = _extract_jd_skills(jd_text)
            skill_score, matched, missing, bonus = _compute_skill_score(self.resume, jd_skills)

            # Weighted fit score
            fit_score = (sem_score * SEMANTIC_WEIGHT) + (skill_score * SKILL_WEIGHT)

            # LLM summary only for jobs above the threshold
            llm_summary = ""
            if use_llm and fit_score >= MIN_FIT_SCORE:
                llm_summary = _generate_llm_summary(
                    self.resume, job, sem_score, matched, missing
                )
                time.sleep(llm_delay)

            scored.append(ScoredJob(
                id=job.get("id", ""),
                title=job.get("title", ""),
                company=job.get("company", ""),
                location=job.get("location", ""),
                description=job.get("description", ""),
                url=job.get("url", ""),
                salary_min=job.get("salary_min"),
                salary_max=job.get("salary_max"),
                salary_note=job.get("salary_note"),
                posted_date=job.get("posted_date", ""),
                source=job.get("source", ""),
                semantic_score=round(sem_score, 4),
                skill_score=round(skill_score, 4),
                fit_score=round(fit_score, 4),
                matched_skills=matched,
                missing_skills=missing,
                bonus_skills=bonus,
                llm_summary=llm_summary,
            ))

            if (i + 1) % 10 == 0:
                log.info(f"  Scored {i+1}/{len(jobs)} jobs...")

        # Sort by fit score descending
        scored.sort(key=lambda j: j.fit_score, reverse=True)

        # Summary stats
        above_threshold = [j for j in scored if j.fit_score >= MIN_FIT_SCORE]
        log.info(f"Scoring complete.")
        log.info(f"  Total scored      : {len(scored)}")
        log.info(f"  Above threshold   : {len(above_threshold)} (fit_score >= {MIN_FIT_SCORE})")
        if scored:
            log.info(f"  Top score         : {scored[0].fit_score:.3f} — {scored[0].title} @ {scored[0].company}")
            log.info(f"  Average fit score : {sum(j.fit_score for j in scored)/len(scored):.3f}")

        return scored


# ── Convenience function ───────────────────────────────────────────────────────

def score_jobs(
    jobs: list[dict],
    resume: Resume,
    use_llm: bool = True,
) -> list[ScoredJob]:
    """
    Top-level convenience function.
    Creates a JobScorer and runs the full pipeline.

    Args:
        jobs:    List of normalized job dicts from fetch_jobs.py
        resume:  Parsed Resume from resume_loader.py
        use_llm: Generate LLM summaries for top matches

    Returns:
        Sorted list of ScoredJob (highest fit first)
    """
    scorer = JobScorer(resume)
    return scorer.score(jobs, use_llm=use_llm)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from pathlib import Path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load resume
    from src.resume_loader import load_resume
    try:
        resume = load_resume("data/resume.pdf")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)

    # Load raw jobs from Step 2 output
    raw_path = Path("outputs/raw_jobs.json")
    if not raw_path.exists():
        print("ERROR: outputs/raw_jobs.json not found. Run Step 2 first:")
        print("  python main.py --fetch-only")
        raise SystemExit(1)

    with open(raw_path, encoding="utf-8") as f:
        jobs = json.load(f)

    print(f"\nLoaded {len(jobs)} jobs from raw_jobs.json")
    print("Running RAG scoring (this takes 2-5 minutes)...\n")

    # Score — disable LLM for quick test, enable for full run
    scored = score_jobs(jobs, resume, use_llm=False)

    print(f"\n{'='*60}")
    print(f"Top 5 Matches")
    print(f"{'='*60}")
    for job in scored[:5]:
        print(f"\n{job.fit_label():10} [{job.fit_score:.3f}] {job.title} @ {job.company}")
        print(f"  Semantic: {job.semantic_score:.3f}  |  Skill: {job.skill_score:.3f}")
        print(f"  Matched : {', '.join(job.matched_skills[:5])}")
        print(f"  Missing : {', '.join(job.missing_skills[:5])}")
        print(f"  URL     : {job.url}")