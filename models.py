# outbreak-agent
# Copyright 2026 Ankur Sharma, PhD
# Licensed under the Apache License, Version 2.0

"""
models.py — Shared state schema for the outbreak triage agent.

OutbreakState flows through every node in the LangGraph graph.
Each node reads what it needs and writes its outputs back to state.
"""

from typing import TypedDict, List, Optional


class OutbreakState(TypedDict):
    """
    Canonical state passed between all LangGraph nodes.

    Fields written by each node:
        genomic_node  → clade, mutation_flags, genome_completeness
        linkage_node  → contact_cluster, transmission_mode, cluster_size
        risk_node     → risk_score, risk_tier, recommended_action
        critic_node   → critic_flags, approved, final_report
    """

    # ── Input (provided by caller) ────────────────────────────────────────────
    case_id:            str
    patient_age:        int
    exposure_location:  str
    vessel:             Optional[str]          # cruise ship, hospital, etc.
    cabin:              Optional[str]
    contacts:           List[str]             # raw contact IDs
    symptom_onset_days: int                   # days since first symptoms
    pcr_ct_value:       Optional[float]       # lower = higher viral load
    genome_sequence:    Optional[str]         # FASTA string or path; None = mock

    # ── genomic_node outputs ──────────────────────────────────────────────────
    clade:              Optional[str]         # e.g. "ANDV-S-clade-2026"
    mutation_flags:     Optional[List[str]]   # e.g. ["N-end-truncation", "G2-shift"]
    genome_completeness: Optional[float]      # 0.0–1.0

    # ── linkage_node outputs ──────────────────────────────────────────────────
    contact_cluster:    Optional[List[str]]   # resolved contact IDs
    transmission_mode:  Optional[str]         # "aerosol" | "fomite" | "unknown"
    cluster_size:       Optional[int]

    # ── risk_node outputs ─────────────────────────────────────────────────────
    risk_score:         Optional[float]       # 0.0–100.0
    risk_tier:          Optional[str]         # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    recommended_action: Optional[str]

    # ── critic_node outputs ───────────────────────────────────────────────────
    critic_flags:       Optional[List[str]]   # empty list = approved
    approved:           Optional[bool]
    final_report:       Optional[str]         # human-readable markdown summary
