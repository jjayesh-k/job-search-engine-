"""
resume_loader.py — Loads, extracts, and structures your resume from PDF.

Flow:
  1. Extract raw text from PDF using pdfminer.six (most reliable for text PDFs)
  2. Clean and normalize whitespace / encoding artifacts
  3. Parse into structured sections (Skills, Experience, Education, etc.)
  4. Return both the full text (for embedding) and structured sections (for scoring)

Usage:
    from src.resume_loader import load_resume
    resume = load_resume("data/resume.pdf")
    print(resume.full_text)
    print(resume.skills)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# ── Data Model ─────────────────────────────────────────────────────────────────

@dataclass
class Resume:
    """Structured representation of the parsed resume."""

    full_text:   str             = ""   # Complete cleaned text — used for embedding
    skills:      list[str]       = field(default_factory=list)   # Extracted skill tokens
    experience:  list[str]       = field(default_factory=list)   # Experience bullet points
    education:   list[str]       = field(default_factory=list)   # Education entries
    summary:     str             = ""   # Profile / summary section if present
    raw_sections: dict[str, str] = field(default_factory=dict)   # Raw section text by heading

    def is_empty(self) -> bool:
        return len(self.full_text.strip()) < 50

    def word_count(self) -> int:
        return len(self.full_text.split())


# ── PDF Text Extraction ────────────────────────────────────────────────────────

def _extract_pdf_text(path: Path) -> str:
    """
    Extract raw text from a PDF using pdfminer.six.
    Handles multi-column layouts better than PyPDF2.

    Falls back to pypdf if pdfminer is unavailable.
    """
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(str(path))
        log.info(f"PDF extracted via pdfminer.six — {len(text)} chars")
        return text
    except ImportError:
        log.warning("pdfminer.six not found, falling back to pypdf")

    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        log.info(f"PDF extracted via pypdf — {len(text)} chars")
        return text
    except ImportError:
        raise ImportError(
            "No PDF library found. Install one:\n"
            "  pip install pdfminer.six\n"
            "  OR: pip install pypdf"
        )


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """
    Normalize text extracted from PDF.
    PDFs often have ligature artifacts, excessive whitespace,
    and broken hyphenation across lines.
    """
    # Fix common PDF ligature artifacts
    replacements = {
        "\ufb01": "fi",   # ﬁ ligature
        "\ufb02": "fl",   # ﬂ ligature
        "\ufb00": "ff",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2022": "-",    # bullet → dash
        "\u2013": "-",    # en dash
        "\u2014": "-",    # em dash
        "\u2019": "'",    # curly apostrophe
        "\u201c": '"',
        "\u201d": '"',
    }
    for bad, good in replacements.items():
        raw = raw.replace(bad, good)

    # Fix hyphenated line breaks ("com-\npany" → "company")
    raw = re.sub(r"-\n(\w)", r"\1", raw)

    # Collapse multiple blank lines to max 2
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    # Normalize spaces (tabs, multiple spaces → single space per line)
    lines = []
    for line in raw.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    raw = "\n".join(lines)

    return raw.strip()


# ── Section Parser ─────────────────────────────────────────────────────────────

# Common resume section headings and what category they map to
SECTION_PATTERNS = {
    "summary":    r"(summary|profile|objective|about\s*me)",
    "skills":     r"(skills|technical\s*skills|core\s*competencies|technologies|tools)",
    "experience": r"(experience|work\s*experience|employment|career|projects?)",
    "education":  r"(education|academic|qualifications?|degrees?|certifications?|courses?)",
}


def _parse_sections(text: str) -> dict[str, str]:
    """
    Split resume text into named sections based on heading detection.

    Strategy: find lines that look like headings (short, often uppercase,
    match our known patterns), then grab everything until the next heading.
    """
    lines = text.split("\n")
    sections: dict[str, list[str]] = {k: [] for k in SECTION_PATTERNS}
    sections["other"] = []

    current_section = "other"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            sections[current_section].append("")
            continue

        # Check if this line is a section heading
        matched = False
        for section_name, pattern in SECTION_PATTERNS.items():
            # Heading: short line (< 50 chars) that matches our pattern
            if len(stripped) < 50 and re.search(pattern, stripped, re.IGNORECASE):
                current_section = section_name
                matched = True
                break

        if not matched:
            sections[current_section].append(stripped)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


# ── Skills Extractor ───────────────────────────────────────────────────────────

# Canonical skills we track — maps variations to a canonical name
SKILL_ALIASES = {
    # Python ecosystem
    "python": "Python", "py": "Python",
    "pandas": "Pandas", "numpy": "NumPy", "matplotlib": "Matplotlib",
    "scikit-learn": "Scikit-learn", "sklearn": "Scikit-learn",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "keras": "Keras", "xgboost": "XGBoost", "lightgbm": "LightGBM",

    # SQL / Databases
    "sql": "SQL", "mysql": "MySQL", "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL", "sqlite": "SQLite",
    "mongodb": "MongoDB", "nosql": "NoSQL",

    # ML / AI
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "nlp": "NLP", "natural language processing": "NLP",
    "llm": "LLMs", "llms": "LLMs", "large language model": "LLMs",
    "computer vision": "Computer Vision", "cv": "Computer Vision",
    "reinforcement learning": "Reinforcement Learning", "rl": "Reinforcement Learning",

    # Data pipelines / ETL
    "etl": "ETL", "airflow": "Airflow", "dbt": "dbt",
    "spark": "Apache Spark", "apache spark": "Apache Spark",
    "kafka": "Kafka", "hadoop": "Hadoop",
    "data pipeline": "Data Pipelines", "pipeline": "Data Pipelines",

    # Visualization
    "tableau": "Tableau", "power bi": "Power BI", "powerbi": "Power BI",
    "matplotlib": "Matplotlib", "seaborn": "Seaborn", "plotly": "Plotly",
    "looker": "Looker",

    # Cloud
    "aws": "AWS", "amazon web services": "AWS",
    "gcp": "GCP", "google cloud": "GCP",
    "azure": "Azure", "microsoft azure": "Azure",

    # MLOps
    "mlops": "MLOps", "mlflow": "MLflow", "kubeflow": "Kubeflow",
    "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "git": "Git", "github": "GitHub", "ci/cd": "CI/CD",

    # General
    "statistics": "Statistics", "stats": "Statistics",
    "a/b testing": "A/B Testing", "ab testing": "A/B Testing",
    "excel": "Excel", "r": "R",
}


def _extract_skills(text: str) -> list[str]:
    """
    Scan resume text for known skill keywords and return canonical names.
    Deduplicates automatically.
    """
    text_lower = text.lower()
    found: set[str] = set()

    # Sort by length descending so "machine learning" matches before "learning"
    sorted_aliases = sorted(SKILL_ALIASES.keys(), key=len, reverse=True)

    for alias in sorted_aliases:
        # Match whole word / phrase only
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text_lower):
            found.add(SKILL_ALIASES[alias])

    return sorted(found)


def _extract_experience_bullets(experience_text: str) -> list[str]:
    """
    Extract individual bullet points / responsibilities from experience section.
    Filters out short lines (dates, company names) and keeps the meaty ones.
    """
    bullets = []
    for line in experience_text.split("\n"):
        line = line.strip().lstrip("-•*▪›").strip()
        # Keep lines that look like achievements (verb-led, 30+ chars)
        if len(line) >= 30:
            bullets.append(line)
    return bullets[:30]  # cap at 30 bullets for embedding efficiency


# ── Main Public Function ───────────────────────────────────────────────────────

def load_resume(path: str | Path = "data/resume.pdf") -> Resume:
    """
    Load and parse a resume from a PDF file.

    Args:
        path: Path to the PDF file. Defaults to data/resume.pdf.

    Returns:
        Resume dataclass with full_text, skills, experience, etc.

    Raises:
        FileNotFoundError: If the PDF doesn't exist at the given path.
        ImportError: If no PDF extraction library is installed.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Resume not found at '{path}'.\n"
            f"Place your PDF at: {path.resolve()}\n"
            f"Or update the path in your .env: RESUME_PATH=path/to/your/resume.pdf"
        )

    log.info(f"Loading resume from: {path}")

    # Step 1: Extract raw text
    raw_text = _extract_pdf_text(path)

    if len(raw_text.strip()) < 100:
        raise ValueError(
            f"Extracted text is too short ({len(raw_text)} chars). "
            "Your PDF may be image-based (scanned). "
            "Convert it to a text-based PDF first, or export as .txt and use that."
        )

    # Step 2: Clean text
    clean = _clean_text(raw_text)

    # Step 3: Parse sections
    sections = _parse_sections(clean)

    # Step 4: Extract structured fields
    skills = _extract_skills(clean)
    experience_bullets = _extract_experience_bullets(sections.get("experience", ""))

    resume = Resume(
        full_text=clean,
        skills=skills,
        experience=experience_bullets,
        education=[
            line.strip() for line in sections.get("education", "").split("\n")
            if len(line.strip()) > 10
        ],
        summary=sections.get("summary", ""),
        raw_sections=sections,
    )

    log.info(
        f"Resume loaded — {resume.word_count()} words, "
        f"{len(resume.skills)} skills detected, "
        f"{len(resume.experience)} experience bullets"
    )
    log.info(f"Skills found: {', '.join(resume.skills[:10])}{'...' if len(resume.skills) > 10 else ''}")

    return resume


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/resume.pdf"

    try:
        resume = load_resume(pdf_path)
        print(f"\n{'='*60}")
        print(f"Resume Summary")
        print(f"{'='*60}")
        print(f"Word count  : {resume.word_count()}")
        print(f"Skills ({len(resume.skills)}): {', '.join(resume.skills)}")
        print(f"Exp bullets : {len(resume.experience)}")
        print(f"\nFirst 500 chars of full text:")
        print(resume.full_text[:500])
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
        
