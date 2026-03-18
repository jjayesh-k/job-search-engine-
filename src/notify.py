"""
notify.py — Gmail notifier for the daily job shortlist.

Sends a rich HTML email with:
  - Summary stats (total scored, jobs above threshold, top score)
  - Top 10 job cards with fit score, label, skills matched/missing
  - Direct apply links
  - LLM fit summary per job

Setup required (one-time):
  1. Enable 2FA on your Gmail account
  2. Generate an App Password:
     Google Account -> Security -> 2-Step Verification -> App passwords
     App name: "Job Search Pipeline" -> Generate
  3. Add to .env:
     GMAIL_SENDER=your.email@gmail.com
     GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   (16-char app password)
     GMAIL_RECIPIENT=your.email@gmail.com      (can be same as sender)
"""

import logging
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from src.score_jobs import ScoredJob
from src.config import MIN_FIT_SCORE

log = logging.getLogger(__name__)

# ── Colour map for fit labels ──────────────────────────────────────────────────
LABEL_COLORS = {
    "Excellent": ("#0F6E56", "#E1F5EE"),   # teal text, teal bg
    "Strong":    ("#185FA5", "#E6F1FB"),   # blue text, blue bg
    "Good":      ("#BA7517", "#FAEEDA"),   # amber text, amber bg
    "Fair":      ("#5F5E5A", "#F1EFE8"),   # gray text, gray bg
    "Weak":      ("#A32D2D", "#FCEBEB"),   # red text, red bg
}


# ── HTML builder ───────────────────────────────────────────────────────────────

def _score_bar(score: float) -> str:
    """Build a small inline SVG score bar (0–100%)."""
    pct = int(score * 100)
    color = (
        "#1D9E75" if score >= 0.82 else
        "#378ADD" if score >= 0.70 else
        "#BA7517" if score >= 0.58 else
        "#888780"
    )
    return (
        f'<div style="background:#F1EFE8;border-radius:4px;height:6px;width:120px;display:inline-block;vertical-align:middle;">'
        f'<div style="background:{color};border-radius:4px;height:6px;width:{pct}%;"></div>'
        f'</div> <span style="font-size:12px;color:#5F5E5A;">{score:.3f}</span>'
    )


def _skill_chips(skills: list[str], color: str, bg: str) -> str:
    """Render a list of skills as inline pill chips."""
    if not skills:
        return '<span style="color:#888780;font-size:11px;">none</span>'
    chips = "".join(
        f'<span style="display:inline-block;background:{bg};color:{color};'
        f'border-radius:20px;padding:2px 8px;font-size:11px;margin:2px;">{s}</span>'
        for s in skills
    )
    return chips


def _job_card(rank: int, job: ScoredJob) -> str:
    """Render a single job as an HTML card block."""
    label_color, label_bg = LABEL_COLORS.get(job.fit_label(), ("#5F5E5A", "#F1EFE8"))
    score_bar = _score_bar(job.fit_score)

    matched_chips = _skill_chips(job.matched_skills[:6], "#0F6E56", "#E1F5EE")
    missing_chips = _skill_chips(job.missing_skills[:5], "#A32D2D", "#FCEBEB")

    salary_html = ""
    if job.salary_min or job.salary_note:
        from src.output_csv import _format_salary
        sal = _format_salary(job)
        if sal:
            salary_html = f'<span style="color:#5F5E5A;font-size:12px;margin-left:8px;">· {sal}</span>'

    llm_html = ""
    if job.llm_summary:
        llm_html = (
            f'<div style="background:#F1EFE8;border-left:3px solid #D3D1C7;'
            f'padding:8px 12px;margin-top:8px;border-radius:0 6px 6px 0;'
            f'font-size:12px;color:#444441;font-style:italic;">'
            f'{job.llm_summary}</div>'
        )

    return f"""
<div style="border:1px solid #D3D1C7;border-radius:10px;padding:16px 20px;
            margin-bottom:12px;background:#ffffff;font-family:Arial,sans-serif;">

  <div style="display:flex;align-items:flex-start;justify-content:space-between;
              flex-wrap:wrap;gap:8px;margin-bottom:10px;">
    <div>
      <span style="font-size:13px;font-weight:600;color:#2C2C2A;">
        #{rank} &nbsp;{job.title}
      </span>
      <span style="color:#5F5E5A;font-size:13px;"> at {job.company}</span>
      {salary_html}
      <br>
      <span style="color:#888780;font-size:12px;">{job.location} · {job.source} · {job.posted_date}</span>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <span style="background:{label_bg};color:{label_color};border-radius:20px;
                   padding:3px 10px;font-size:11px;font-weight:600;">
        {job.fit_label()}
      </span>
      <br>
      <div style="margin-top:6px;">{score_bar}</div>
    </div>
  </div>

  <div style="margin-bottom:6px;">
    <span style="font-size:11px;color:#5F5E5A;text-transform:uppercase;
                 letter-spacing:0.05em;margin-right:6px;">Matched</span>
    {matched_chips}
  </div>

  <div style="margin-bottom:10px;">
    <span style="font-size:11px;color:#5F5E5A;text-transform:uppercase;
                 letter-spacing:0.05em;margin-right:6px;">Missing</span>
    {missing_chips}
  </div>

  {llm_html}

  <div style="margin-top:12px;">
    <a href="{job.url}"
       style="background:#2C2C2A;color:#ffffff;text-decoration:none;
              padding:7px 16px;border-radius:6px;font-size:12px;font-weight:600;">
      View &amp; Apply →
    </a>
  </div>

</div>"""


def build_email_html(
    shortlist: list[ScoredJob],
    total_scored: int,
    run_date: Optional[str] = None,
) -> str:
    """
    Build the full HTML email body.

    Args:
        shortlist:    Jobs above threshold, sorted by fit_score.
        total_scored: Total jobs scored today (for the summary line).
        run_date:     ISO date string. Defaults to today.
    """
    run_date = run_date or date.today().isoformat()
    top_n = min(10, len(shortlist))
    above = len(shortlist)

    top_score_str = f"{shortlist[0].fit_score:.3f}" if shortlist else "—"
    top_role_str  = f"{shortlist[0].title} @ {shortlist[0].company}" if shortlist else "—"

    job_cards_html = "".join(
        _job_card(i + 1, job) for i, job in enumerate(shortlist[:10])
    )

    no_results_html = ""
    if not shortlist:
        no_results_html = """
        <div style="text-align:center;padding:40px;color:#888780;font-size:14px;">
          No jobs above the fit threshold today.<br>
          Try lowering MIN_FIT_SCORE in your .env file.
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#F1EFE8;font-family:Arial,sans-serif;">

<div style="max-width:640px;margin:0 auto;padding:24px 16px;">

  <!-- Header -->
  <div style="background:#2C2C2A;border-radius:12px;padding:24px 28px;margin-bottom:20px;">
    <div style="color:#D3D1C7;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
                margin-bottom:6px;">Daily Job Shortlist</div>
    <div style="color:#ffffff;font-size:22px;font-weight:700;margin-bottom:4px;">
      {top_n} matches found
    </div>
    <div style="color:#888780;font-size:13px;">{run_date} · {total_scored} jobs scored today</div>
  </div>

  <!-- Stats row -->
  <div style="display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;">

    <div style="flex:1;min-width:140px;background:#ffffff;border:1px solid #D3D1C7;
                border-radius:10px;padding:14px 16px;">
      <div style="color:#888780;font-size:11px;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:4px;">Above threshold</div>
      <div style="font-size:24px;font-weight:700;color:#2C2C2A;">{above}</div>
      <div style="font-size:11px;color:#888780;">of {total_scored} scored</div>
    </div>

    <div style="flex:1;min-width:140px;background:#ffffff;border:1px solid #D3D1C7;
                border-radius:10px;padding:14px 16px;">
      <div style="color:#888780;font-size:11px;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:4px;">Top score</div>
      <div style="font-size:24px;font-weight:700;color:#2C2C2A;">{top_score_str}</div>
      <div style="font-size:11px;color:#888780;overflow:hidden;
                  text-overflow:ellipsis;white-space:nowrap;">{top_role_str}</div>
    </div>

    <div style="flex:1;min-width:140px;background:#ffffff;border:1px solid #D3D1C7;
                border-radius:10px;padding:14px 16px;">
      <div style="color:#888780;font-size:11px;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:4px;">Showing</div>
      <div style="font-size:24px;font-weight:700;color:#2C2C2A;">Top {top_n}</div>
      <div style="font-size:11px;color:#888780;">sorted by fit score</div>
    </div>

  </div>

  <!-- Job cards -->
  <div style="margin-bottom:20px;">
    {job_cards_html}
    {no_results_html}
  </div>

  <!-- Footer -->
  <div style="text-align:center;color:#888780;font-size:11px;padding-top:12px;
              border-top:1px solid #D3D1C7;">
    Generated by your Job Search Pipeline &nbsp;·&nbsp;
    Scores: semantic similarity + skill-gap analysis
  </div>

</div>
</body>
</html>"""


# ── Email sender ───────────────────────────────────────────────────────────────

def send_email(
    shortlist: list[ScoredJob],
    total_scored: int,
    csv_path: Optional[Path] = None,
    run_date: Optional[str] = None,
) -> bool:
    """
    Send the daily shortlist via Gmail SMTP.

    Args:
        shortlist:    Top scored jobs to include in email.
        total_scored: Total jobs scored (for stats).
        csv_path:     Optional — attach the CSV file to the email.
        run_date:     ISO date string. Defaults to today.

    Returns:
        True if sent successfully, False on failure.
    """
    sender    = os.getenv("GMAIL_SENDER", "")
    password  = os.getenv("GMAIL_APP_PASSWORD", "")
    recipient = os.getenv("GMAIL_RECIPIENT", sender)

    if not sender or not password:
        log.error(
            "Gmail credentials not set. Add to .env:\n"
            "  GMAIL_SENDER=your@gmail.com\n"
            "  GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx\n"
            "  GMAIL_RECIPIENT=your@gmail.com"
        )
        return False

    run_date = run_date or date.today().isoformat()
    subject  = f"Job Shortlist {run_date} — {len(shortlist)} matches found"

    # Build message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Job Search Pipeline <{sender}>"
    msg["To"]      = recipient

    # Plain text fallback
    plain_lines = [f"Job Shortlist — {run_date}", f"{len(shortlist)} jobs above threshold\n"]
    for i, job in enumerate(shortlist[:10], 1):
        plain_lines.append(
            f"{i}. [{job.fit_score:.3f} {job.fit_label()}] "
            f"{job.title} @ {job.company} — {job.url}"
        )
    plain_text = "\n".join(plain_lines)

    # HTML body
    html_body = build_email_html(shortlist, total_scored, run_date)

    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    # Attach CSV if provided
    if csv_path and csv_path.exists():
        from email.mime.base import MIMEBase
        from email import encoders
        with open(csv_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={csv_path.name}",
        )
        msg.attach(part)
        log.info(f"CSV attached: {csv_path.name}")

    # Send via Gmail SMTP
    try:
        log.info(f"Sending email to {recipient}...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password.replace(" ", ""))
            server.sendmail(sender, recipient, msg.as_string())
        log.info(f"Email sent successfully to {recipient}")
        return True

    except smtplib.SMTPAuthenticationError:
        log.error(
            "Gmail authentication failed.\n"
            "Make sure you're using an App Password (not your Gmail password).\n"
            "Generate one at: Google Account > Security > 2-Step Verification > App passwords"
        )
        return False
    except smtplib.SMTPException as e:
        log.error(f"SMTP error: {e}")
        return False
    except Exception as e:
        log.error(f"Email send failed: {e}")
        return False


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    scored_path = Path("outputs/scored_jobs.json")
    if not scored_path.exists():
        print("Run Step 3 first: python main.py --score-only")
        raise SystemExit(1)

    with open(scored_path, encoding="utf-8") as f:
        data = json.load(f)

    from src.score_jobs import ScoredJob
    all_jobs = [ScoredJob(**{k: v for k, v in d.items() if k != "fit_label"}) for d in data]
    shortlist = [j for j in all_jobs if j.fit_score >= MIN_FIT_SCORE]

    print(f"Sending test email with {len(shortlist)} jobs...")
    ok = send_email(shortlist, total_scored=len(all_jobs))
    print("SUCCESS" if ok else "FAILED — check logs above")