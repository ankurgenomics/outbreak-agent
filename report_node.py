# outbreak-agent
# Copyright 2026 Ankur Sharma, PhD
# Licensed under the Apache License, Version 2.0

"""
report_node.py -- Generates a risk dashboard PNG and a PDF triage report.

Outputs saved to reports/:
    <case_id>-risk-dashboard-<date>.png   -- 3-panel matplotlib figure
    <case_id>-triage-report-<date>.pdf    -- structured triage report

Usage (standalone):
    python report_node.py

Usage (from agent):
    from report_node import generate_dashboard, generate_pdf
    png_path = generate_dashboard(state)
    pdf_path = generate_pdf(state)
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

from models import OutbreakState

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

TIER_COLOURS = {
    "CRITICAL": "#C0392B",
    "HIGH":     "#E67E22",
    "MEDIUM":   "#F1C40F",
    "LOW":      "#27AE60",
    "UNKNOWN":  "#95A5A6",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Panel 1 -- Risk score bar chart
# ─────────────────────────────────────────────────────────────────────────────
def _panel_risk_scores(ax: plt.Axes) -> None:
    cases   = ["HONDIUS\n(ANDV cruise)", "HIGH RISK\n(ANDV land)",
               "LOW RISK\n(SNV)", "INCOMPLETE\n(unknown clade)"]
    scores  = [98, 85, 22, 60]
    tiers   = ["CRITICAL", "HIGH", "LOW", "MEDIUM"]
    colours = [TIER_COLOURS[t] for t in tiers]

    bars = ax.barh(cases, scores, color=colours, edgecolor="white",
                   linewidth=0.8, height=0.55)

    for bar, score, tier in zip(bars, scores, tiers):
        label_x = score - 2 if score > 30 else score + 2
        ha      = "right"   if score > 30 else "left"
        colour  = "white"   if score > 30 else "#333333"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"{score}/100  {tier}",
                va="center", ha=ha, color=colour,
                fontsize=8.5, fontweight="bold")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Risk Score (0-100)", fontsize=9)
    ax.set_title("Case Risk Scores", fontsize=10, fontweight="bold", pad=8)
    ax.axvline(x=75, color="#C0392B", linestyle="--", lw=0.9, alpha=0.6,
               label="CRITICAL threshold (75)")
    ax.axvline(x=50, color="#E67E22", linestyle="--", lw=0.9, alpha=0.6,
               label="HIGH threshold (50)")
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()


# ─────────────────────────────────────────────────────────────────────────────
#  Panel 2 -- Event timeline (fixed: staggered labels, no overlap)
# ─────────────────────────────────────────────────────────────────────────────
def _panel_timeline(ax: plt.Axes, symptom_onset_days: int = 12) -> None:
    d = symptom_onset_days

    # Events: (day, short_label, colour, marker, y_offset for label)
    # Alternate labels above/below the line to prevent overlap
    events = [
        (0,     "Excursion\n(exposure)",      "#27AE60", "^",  0.28, "bottom"),
        (d,     "Symptom\nonset",             "#E67E22", "o", -0.28, "top"),
        (d + 2, "Ship\ndocks",               "#2980B9", "s",  0.28, "bottom"),
        (d + 3, "Agent\ntriage",             "#8E44AD", "D", -0.28, "top"),
        (d + 4, "MoH\nnotification",         "#C0392B", "*",  0.28, "bottom"),
    ]

    # Incubation shading
    ax.axvspan(0, 8,  alpha=0.10, color="#27AE60")
    ax.axvspan(8, 45, alpha=0.04, color="#E67E22")

    # Horizontal spine
    ax.axhline(0.0, color="#BDC3C7", linewidth=1.8, zorder=1)

    for day, label, colour, marker, y_off, va in events:
        ax.plot(day, 0.0, marker=marker, color=colour,
                markersize=11, zorder=5, clip_on=False)
        # Vertical tick from spine to label
        ax.plot([day, day], [0.0, y_off * 0.85],
                color=colour, lw=0.9, zorder=3)
        ax.text(day, y_off, label, ha="center", va=va,
                fontsize=7.5, color=colour, fontweight="bold",
                linespacing=1.3)

    # Annotation box: human vs agent
    ax.annotate(
        "Human analyst: 48-72 h\noutbreak-agent: < 2 h",
        xy=(d + 3, 0.0),
        xytext=(d + 5.5, -0.55),
        fontsize=7.2, color="#8E44AD",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F3E8FF", ec="#8E44AD", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=0.8),
    )

    legend_patches = [
        mpatches.Patch(color="#27AE60", alpha=0.4, label="Min incubation (8 days)"),
        mpatches.Patch(color="#E67E22", alpha=0.3, label="Max incubation (45 days)"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper left",
              framealpha=0.7)

    ax.set_xlim(-1, d + 10)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel("Days since exposure", fontsize=9)
    ax.set_title("MV Hondius -- Outbreak Event Timeline", fontsize=10,
                 fontweight="bold", pad=8)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
#  Panel 3 -- Contact cluster growth
# ─────────────────────────────────────────────────────────────────────────────
def _panel_cluster_growth(ax: plt.Axes, cluster_size: int = 5) -> None:
    days       = np.arange(0, 16)
    index_case = np.ones(len(days))
    confirmed  = np.clip(np.floor(days / 3).astype(int), 0, cluster_size - 1)
    at_risk    = np.clip(np.floor(days * 1.8).astype(int), 0, 28)

    ax.fill_between(days, 0, index_case,
                    alpha=0.90, color=TIER_COLOURS["CRITICAL"],
                    label="Index case (CRITICAL)")
    ax.fill_between(days, index_case, index_case + confirmed,
                    alpha=0.75, color=TIER_COLOURS["HIGH"],
                    label="Confirmed contacts")
    ax.fill_between(days, index_case + confirmed,
                    index_case + confirmed + at_risk,
                    alpha=0.45, color=TIER_COLOURS["MEDIUM"],
                    label="At-risk (HVAC zones)")

    ax.axvline(x=15, color="#8E44AD", linestyle="--", lw=1.2,
               label="Agent triage complete (day 15)")

    ax.set_xlabel("Days since exposure", fontsize=9)
    ax.set_ylabel("Number of individuals", fontsize=9)
    ax.set_title("Contact Cluster Growth -- MV Hondius Scenario",
                 fontsize=10, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 32)


# ─────────────────────────────────────────────────────────────────────────────
#  PNG dashboard
# ─────────────────────────────────────────────────────────────────────────────
def generate_dashboard(state: Optional[OutbreakState] = None) -> str:
    s = state or {}
    case_id            = s.get("case_id",            "ANDV-2026-001")
    symptom_onset_days = s.get("symptom_onset_days", 12)
    cluster_size       = s.get("cluster_size",       5)
    risk_score         = s.get("risk_score",         98.0)
    risk_tier          = s.get("risk_tier",          "CRITICAL")
    clade              = s.get("clade",              "ANDV-S-clade-2026")
    transmission       = s.get("transmission_mode",  "aerosol-human-to-human")

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#F8F9FA")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.38,
                           top=0.88, bottom=0.09, left=0.07, right=0.96)

    ax_scores   = fig.add_subplot(gs[0, 0])
    ax_timeline = fig.add_subplot(gs[0, 1])
    ax_cluster  = fig.add_subplot(gs[1, :])

    _panel_risk_scores(ax_scores)
    _panel_timeline(ax_timeline, symptom_onset_days=int(symptom_onset_days))
    _panel_cluster_growth(ax_cluster, cluster_size=int(cluster_size or 5))

    tier_colour = TIER_COLOURS.get(risk_tier, "#95A5A6")
    fig.suptitle(
        f"outbreak-agent  |  Case {case_id}  |  {risk_tier}  ({risk_score:.0f}/100)",
        fontsize=13, fontweight="bold", color=tier_colour, y=0.96,
    )
    fig.text(
        0.5, 0.925,
        f"Clade: {clade}   |   Transmission: {transmission}   |   {date.today().isoformat()}",
        ha="center", fontsize=8.5, color="#555555",
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = f"{case_id}-risk-dashboard-{date.today().isoformat()}.png"
    out_path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Dashboard PNG : {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  PDF triage report
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(state: Optional[OutbreakState] = None,
                 png_path: Optional[str] = None) -> str:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable,
                                    Image, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    s = state or {}
    case_id     = s.get("case_id",            "ANDV-2026-001")
    risk_tier   = s.get("risk_tier",          "CRITICAL")
    risk_score  = s.get("risk_score",         98.0)
    clade       = s.get("clade",              "ANDV-S-clade-2026")
    mutations   = s.get("mutation_flags",     ["N-end-truncation-14aa", "G2-glycoprotein-shift"])
    completeness = s.get("genome_completeness", 87.0)
    transmission = s.get("transmission_mode", "aerosol-human-to-human")
    cluster_size = s.get("cluster_size",      5)
    vessel      = s.get("vessel",             "MV Hondius")
    action      = s.get("recommended_action", "Immediate isolation. Notify MoH within 2 hours.")
    critic_flags = s.get("critic_flags",      [])
    loops       = s.get("_loop_count",        1)
    onset_days  = s.get("symptom_onset_days", 12)
    pcr_ct      = s.get("pcr_ct_value",       21.3)
    location    = s.get("exposure_location",  "Andes foothills, Argentina")

    # Tier colour mapping for ReportLab
    tier_hex = {
        "CRITICAL": colors.HexColor("#C0392B"),
        "HIGH":     colors.HexColor("#E67E22"),
        "MEDIUM":   colors.HexColor("#F1C40F"),
        "LOW":      colors.HexColor("#27AE60"),
    }
    tier_colour = tier_hex.get(risk_tier, colors.grey)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = f"{case_id}-triage-report-{date.today().isoformat()}.pdf"
    out_path = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Styles ────────────────────────────────────────────────────────────────
    style_h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                               fontSize=18, textColor=tier_colour,
                               spaceAfter=4, alignment=TA_CENTER)
    style_h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                               fontSize=11, textColor=colors.HexColor("#2C3E50"),
                               spaceBefore=14, spaceAfter=4)
    style_meta = ParagraphStyle("Meta", parent=styles["Normal"],
                                fontSize=8.5, textColor=colors.HexColor("#7F8C8D"),
                                alignment=TA_CENTER, spaceAfter=2)
    style_body = ParagraphStyle("Body", parent=styles["Normal"],
                                fontSize=9.5, leading=14, spaceAfter=4)
    style_action = ParagraphStyle("Action", parent=styles["Normal"],
                                  fontSize=10, leading=14,
                                  textColor=tier_colour, fontName="Helvetica-Bold")
    style_footer = ParagraphStyle("Footer", parent=styles["Normal"],
                                  fontSize=7.5, textColor=colors.HexColor("#95A5A6"),
                                  alignment=TA_CENTER)

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("OUTBREAK TRIAGE REPORT", style_h1))
    story.append(Paragraph(
        f"Generated by outbreak-agent v1.0  |  {date.today().strftime('%d %B %Y')}",
        style_meta))
    story.append(Paragraph(
        "Ankur Sharma, PhD  |  ankurs103@gmail.com  |  Apache 2.0 (public nodes)",
        style_meta))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=tier_colour, spaceAfter=12))

    # ── Risk banner ───────────────────────────────────────────────────────────
    banner_data = [[
        Paragraph(f"<b>CASE ID</b><br/>{case_id}", styles["Normal"]),
        Paragraph(f"<b>RISK TIER</b><br/><font color='{tier_colour.hexval()}'><b>{risk_tier}</b></font>",
                  styles["Normal"]),
        Paragraph(f"<b>RISK SCORE</b><br/>{risk_score:.0f} / 100", styles["Normal"]),
        Paragraph(f"<b>VESSEL</b><br/>{vessel or 'N/A'}", styles["Normal"]),
    ]]
    banner_table = Table(banner_data, colWidths=[4*cm, 4*cm, 4*cm, 4.5*cm])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#F2F3F4")),
        ("BOX",          (0, 0), (-1, -1), 1, colors.HexColor("#BDC3C7")),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 14))

    # ── Risk dashboard chart (embedded PNG) ───────────────────────────────────
    # Page usable width: A4 (595pt) minus margins (2cm each side) = ~511pt
    usable_w = A4[0] - 4 * cm   # 4cm total margins
    if png_path and os.path.exists(png_path):
        style_chart_caption = ParagraphStyle(
            "ChartCaption", parent=styles["Normal"],
            fontSize=8, textColor=colors.HexColor("#7F8C8D"),
            alignment=TA_CENTER, spaceAfter=4, spaceBefore=4,
        )
        # Scale image to fill full page width; height auto-calculated
        img = Image(png_path, width=usable_w, height=usable_w * 10 / 15)
        story.append(img)
        story.append(Paragraph(
            "Figure 1. Risk dashboard generated by outbreak-agent. "
            "Left: risk scores across all mock scenarios. "
            "Centre: event timeline from exposure to MoH notification. "
            "Bottom: contact cluster growth over time.",
            style_chart_caption,
        ))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#BDC3C7"),
                                spaceBefore=6, spaceAfter=6))

    # ── Recommended action ────────────────────────────────────────────────────
    story.append(Paragraph("Recommended Action", style_h2))
    story.append(Paragraph(action, style_action))
    story.append(Spacer(1, 8))

    # ── Genomic profile ───────────────────────────────────────────────────────
    story.append(Paragraph("1. Genomic Profile", style_h2))
    mutation_str = ", ".join(mutations) if mutations else "None detected"
    genomic_data = [
        ["Field", "Value"],
        ["Clade",               clade or "UNKNOWN"],
        ["Mutation flags",      mutation_str],
        ["Genome completeness", f"{completeness:.1f}%" if completeness else "N/A"],
        ["PCR Ct value",        f"{pcr_ct}" if pcr_ct else "N/A"],
        ["Exposure location",   location or "Unknown"],
        ["Symptom onset",       f"Day {onset_days} post-exposure"],
    ]
    story.append(_make_table(genomic_data))
    story.append(Spacer(1, 10))

    # ── Epidemiological linkage ───────────────────────────────────────────────
    story.append(Paragraph("2. Epidemiological Linkage", style_h2))
    epi_data = [
        ["Field", "Value"],
        ["Transmission mode",  transmission or "Unknown"],
        ["Cluster size",       str(cluster_size) if cluster_size else "N/A"],
        ["Vessel / setting",   vessel or "Community (no vessel)"],
    ]
    story.append(_make_table(epi_data))
    story.append(Spacer(1, 10))

    # ── Critic audit log ──────────────────────────────────────────────────────
    story.append(Paragraph("3. Agent Quality Gate (Critic Audit)", style_h2))
    approved_str = "APPROVED" if not critic_flags else "FLAGGED -- REVIEW REQUIRED"
    flags_str    = "No flags raised -- all consistency checks passed." \
                   if not critic_flags else "\n".join(f"  - {f}" for f in critic_flags)
    audit_data = [
        ["Field", "Value"],
        ["Outcome",       approved_str],
        ["Re-evaluations", f"{(loops or 1) - 1} loops (max allowed: 3)"],
        ["Flags",         flags_str],
    ]
    story.append(_make_table(audit_data))
    story.append(Spacer(1, 10))

    # ── Interpretation notes ──────────────────────────────────────────────────
    story.append(Paragraph("4. Interpretation Notes", style_h2))
    notes = (
        "This report was generated by the outbreak-agent heuristic triage pipeline. "
        "All genomic classification and risk scoring in the public release uses "
        "location-based heuristics and published epidemiological thresholds. "
        "The private clinical scoring module (not included in the public repository) "
        "applies validated thresholds derived from clinical deployment data. "
        "This output is intended for research and demonstration purposes. "
        "Clinical decisions must be confirmed by a qualified public health professional."
    )
    story.append(Paragraph(notes, style_body))
    story.append(Spacer(1, 20))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#BDC3C7"), spaceAfter=6))
    story.append(Paragraph(
        f"outbreak-agent v1.0  |  Apache 2.0  |  github.com/ankurgenomics/outbreak-agent  |  "
        f"Copyright 2026 Ankur Sharma, PhD",
        style_footer))

    doc.build(story)
    print(f"  Triage PDF    : {out_path}")
    return out_path


def _make_table(data: list):
    """Helper: styled two-column table with header row."""
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    col_widths = [5*cm, 11.5*cm]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("BACKGROUND",    (0, 1), (0, -1), colors.HexColor("#F2F3F4")),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.white, colors.HexColor("#FAFAFA")]),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#BDC3C7")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    png = generate_dashboard()
    pdf = generate_pdf(png_path=png)
    print(f"\nDone.\n  PNG: {png}\n  PDF: {pdf}")
