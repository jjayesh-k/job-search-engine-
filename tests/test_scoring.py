"""
test_scoring.py — Unit tests for resume_loader.py and score_jobs.py

Tests all logic that doesn't require a real PDF or live Ollama.
Fast — runs in under 3 seconds.

Run with:
    pytest tests/test_scoring.py -v
"""

import pytest
import numpy as np


# ── resume_loader tests ────────────────────────────────────────────────────────

class TestTextCleaning:
    def test_fixes_ligatures(self):
        from src.resume_loader import _clean_text
        result = _clean_text("pro\ufb01le experience")  # ﬁ ligature
        assert "fi" in result
        assert "\ufb01" not in result

    def test_fixes_hyphenated_line_breaks(self):
        from src.resume_loader import _clean_text
        result = _clean_text("pro-\ncess")
        assert "process" in result

    def test_collapses_excess_blank_lines(self):
        from src.resume_loader import _clean_text
        result = _clean_text("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in result


class TestSkillExtraction:
    def test_extracts_python(self):
        from src.resume_loader import _extract_skills
        skills = _extract_skills("Proficient in Python and SQL")
        assert "Python" in skills
        assert "SQL" in skills

    def test_extracts_ml_variants(self):
        from src.resume_loader import _extract_skills
        skills = _extract_skills("Experience with sklearn and deep learning")
        assert "Scikit-learn" in skills
        assert "Deep Learning" in skills

    def test_extracts_nlp(self):
        from src.resume_loader import _extract_skills
        skills = _extract_skills("Built NLP pipelines using LLMs")
        assert "NLP" in skills
        assert "LLMs" in skills

    def test_extracts_visualization(self):
        from src.resume_loader import _extract_skills
        skills = _extract_skills("Created dashboards in Tableau and Power BI")
        assert "Tableau" in skills
        assert "Power BI" in skills

    def test_case_insensitive(self):
        from src.resume_loader import _extract_skills
        skills = _extract_skills("PYTHON and PYTORCH experience")
        assert "Python" in skills
        assert "PyTorch" in skills

    def test_deduplicates_skills(self):
        from src.resume_loader import _extract_skills
        skills = _extract_skills("Python python PYTHON")
        assert skills.count("Python") == 1


class TestSectionParser:
    def test_finds_skills_section(self):
        from src.resume_loader import _parse_sections
        text = "John Doe\n\nSkills\nPython, SQL, TensorFlow\n\nExperience\nWorked at Google"
        sections = _parse_sections(text)
        assert "Python" in sections.get("skills", "")

    def test_finds_experience_section(self):
        from src.resume_loader import _parse_sections
        text = "Name\n\nWork Experience\nData Scientist at Infosys\n\nEducation\nBTech"
        sections = _parse_sections(text)
        assert "Infosys" in sections.get("experience", "")


class TestResumeDatclass:
    def test_is_empty_on_short_text(self):
        from src.resume_loader import Resume
        r = Resume(full_text="short")
        assert r.is_empty()

    def test_not_empty_on_real_text(self):
        from src.resume_loader import Resume
        r = Resume(full_text="X" * 200)
        assert not r.is_empty()

    def test_word_count(self):
        from src.resume_loader import Resume
        r = Resume(full_text="one two three four five")
        assert r.word_count() == 5


# ── score_jobs tests ───────────────────────────────────────────────────────────

class TestSkillScoring:
    def test_perfect_match(self):
        from src.score_jobs import _compute_skill_score
        from src.resume_loader import Resume
        resume = Resume(skills=["Python", "SQL", "Machine Learning"])
        score, matched, missing, bonus = _compute_skill_score(
            resume, ["Python", "SQL", "Machine Learning"]
        )
        assert score == 1.0
        assert set(matched) == {"Python", "SQL", "Machine Learning"}
        assert missing == []

    def test_zero_match(self):
        from src.score_jobs import _compute_skill_score
        from src.resume_loader import Resume
        resume = Resume(skills=["Python", "SQL"])
        score, matched, missing, bonus = _compute_skill_score(
            resume, ["Java", "Kubernetes", "Scala"]
        )
        assert score == 0.0
        assert matched == []
        assert set(missing) == {"Java", "Kubernetes", "Scala"}

    def test_partial_match(self):
        from src.score_jobs import _compute_skill_score
        from src.resume_loader import Resume
        resume = Resume(skills=["Python", "SQL", "Tableau"])
        score, matched, missing, bonus = _compute_skill_score(
            resume, ["Python", "SQL", "Spark", "Kafka"]
        )
        assert 0 < score < 1
        assert "Python" in matched
        assert "Spark" in missing

    def test_empty_jd_skills_gives_neutral(self):
        from src.score_jobs import _compute_skill_score
        from src.resume_loader import Resume
        resume = Resume(skills=["Python"])
        score, *_ = _compute_skill_score(resume, [])
        assert score == 0.6   # neutral fallback

    def test_bonus_skills_dont_inflate_score(self):
        from src.score_jobs import _compute_skill_score
        from src.resume_loader import Resume
        # Resume has many extra skills beyond JD
        resume = Resume(skills=["Python", "SQL", "Tableau", "PyTorch", "Keras", "AWS"])
        score_basic, _, _, _ = _compute_skill_score(resume, ["Python", "SQL"])
        # Score should be 1.0 (100% of JD requirements met) not >1.0
        assert score_basic <= 1.0


class TestCosimeSimilarity:
    def test_identical_vectors(self):
        from src.score_jobs import cosine_similarity
        v = np.array([0.5, 0.5, 0.5, 0.5])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from src.score_jobs import cosine_similarity
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)


class TestScoredJobDataclass:
    def _make_scored_job(self, fit_score):
        from src.score_jobs import ScoredJob
        return ScoredJob(
            title="Data Scientist",
            company="Infosys",
            fit_score=fit_score,
            semantic_score=fit_score,
            skill_score=fit_score,
        )

    def test_fit_label_excellent(self):
        j = self._make_scored_job(0.85)
        assert j.fit_label() == "Excellent"

    def test_fit_label_strong(self):
        j = self._make_scored_job(0.73)
        assert j.fit_label() == "Strong"

    def test_fit_label_weak(self):
        j = self._make_scored_job(0.30)
        assert j.fit_label() == "Weak"

    def test_to_dict_has_required_keys(self):
        j = self._make_scored_job(0.75)
        d = j.to_dict()
        for key in ["fit_score", "fit_label", "semantic_score", "skill_score",
                    "matched_skills", "missing_skills", "llm_summary"]:
            assert key in d

    def test_to_dict_rounds_scores(self):
        from src.score_jobs import ScoredJob
        j = ScoredJob(fit_score=0.123456789)
        d = j.to_dict()
        assert d["fit_score"] == 0.1235


class TestJDSkillExtraction:
    def test_extracts_from_jd(self):
        from src.score_jobs import _extract_jd_skills
        jd = "We need Python, SQL, and experience with deep learning and NLP."
        skills = _extract_jd_skills(jd)
        assert "Python" in skills
        assert "SQL" in skills
        assert "Deep Learning" in skills
        assert "NLP" in skills