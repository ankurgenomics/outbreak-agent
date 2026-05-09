# outbreak-agent
# Copyright 2026 Ankur Sharma, PhD
# Licensed under the Apache License, Version 2.0

"""
nodes.py — The four LangGraph node functions for outbreak triage.

Each function receives the full OutbreakState dict and returns a dict
containing ONLY the keys it modifies. LangGraph merges updates back
into the shared state automatically.

Node pipeline:
    genomic_node → linkage_node → risk_node → critic_node
                        ↑__________________________|
                        (re-entry if critic_flags not empty)

Note: In mock mode (genome_sequence is None), nodes use heuristic logic
only. The private genomics_tools/ module handles real FASTA/VCF parsing.
"""

from __future__ import annotations
import re
from typing import Any, Dict

from models import OutbreakState


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 1 — Genomic Analysis
# ─────────────────────────────────────────────────────────────────────────────

def genomic_node(state: OutbreakState) -> Dict[str, Any]:
    """
    Analyse genome sequence to identify clade, mutations, and completeness.

    In mock mode (no genome_sequence), infers clade from exposure_location.
    Real implementation lives in private genomics_tools.fasta_parser module.

    Returns: clade, mutation_flags, genome_completeness
    """
    genome = state.get("genome_sequence")
    location = state.get("exposure_location", "").lower()

    if genome:
        # ── Real mode (genome_sequence present) ──────────────────────────────
        # NOTE: Production path calls genomics_tools.parse_fasta() here.
        # That module is kept private — contact author for access.
        completeness = _estimate_completeness(genome)
        clade = _classify_clade(genome, location)
        mutations = _scan_mutations(genome)
    else:
        # ── Mock / heuristic mode ─────────────────────────────────────────────
        completeness = 0.87   # typical nanopore run quality
        if any(kw in location for kw in ["andes", "argentina", "patagonia", "chile"]):
            clade = "ANDV-S-clade-2026"
            mutations = ["N-end-truncation-14aa", "G2-glycoprotein-shift"]
        elif any(kw in location for kw in ["sin nombre", "arizona", "new mexico"]):
            clade = "SNV-NA-clade"
            mutations = []
        elif any(kw in location for kw in ["seoul", "europe", "puumala"]):
            clade = "PUUV-EU-clade"
            mutations = ["S-segment-reassortment"]
        else:
            clade = "UNKNOWN-clade"
            mutations = ["UNCLASSIFIED"]

    return {
        "clade": clade,
        "mutation_flags": mutations,
        "genome_completeness": round(completeness, 3),
    }


def _estimate_completeness(genome: str) -> float:
    """Fraction of non-N bases in the sequence."""
    if not genome:
        return 0.0
    n_count = genome.upper().count("N")
    return 1.0 - (n_count / max(len(genome), 1))


def _classify_clade(genome: str, location: str) -> str:
    """Stub — replaced by real phylogenetic placement in private module."""
    if "andv" in location or "andes" in location:
        return "ANDV-S-clade-2026"
    return "UNKNOWN-clade"


def _scan_mutations(genome: str) -> list:
    """Stub — replaced by real MUSCLE alignment scan in private module."""
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 2 — Contact Linkage
# ─────────────────────────────────────────────────────────────────────────────

def linkage_node(state: OutbreakState) -> Dict[str, Any]:
    """
    Resolve contact list into a confirmed cluster and infer transmission mode.

    Uses vessel/cabin metadata for cruise-ship scenarios (MV Hondius pattern).
    Returns: contact_cluster, transmission_mode, cluster_size
    """
    contacts  = state.get("contacts", [])
    vessel    = state.get("vessel")
    cabin     = state.get("cabin")
    clade     = state.get("clade", "")
    mutations = state.get("mutation_flags", [])

    # Filter to epidemiologically relevant contacts
    cluster = _resolve_cluster(contacts, vessel, cabin)

    # Infer transmission mode
    if vessel:
        # Enclosed-space, recirculated air → aerosol risk elevated
        # ANDV is the only hantavirus with confirmed human-to-human transmission
        if "ANDV" in clade:
            mode = "aerosol-human-to-human"
        else:
            mode = "aerosol-rodent-excreta"
    elif any("G2-glycoprotein-shift" in m for m in mutations):
        # Glycoprotein shift may alter cell tropism → aerosol more likely
        mode = "aerosol-suspected"
    else:
        mode = "fomite-or-rodent-excreta"

    return {
        "contact_cluster": cluster,
        "transmission_mode": mode,
        "cluster_size": len(cluster),
    }


def _resolve_cluster(contacts: list, vessel: str, cabin: str) -> list:
    """
    Expand contacts using cabin proximity heuristic for vessel scenarios.
    In a real system this queries the hospital/ship manifest database.
    """
    resolved = list(contacts)  # start with declared contacts

    if vessel and cabin:
        # Add adjacent cabins as high-risk contacts (cruise-ship pattern)
        cabin_num = re.search(r"\d+", cabin or "")
        if cabin_num:
            n = int(cabin_num.group())
            for adj in [n - 1, n + 1]:
                synthetic = f"adjacent_cabin_{adj}_occupant"
                if synthetic not in resolved:
                    resolved.append(synthetic)

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 3 — Risk Scoring
# ─────────────────────────────────────────────────────────────────────────────

# Weight table for risk factors
_RISK_WEIGHTS = {
    "ANDV-S-clade-2026":            30,   # highest HPS mortality clade
    "N-end-truncation-14aa":        10,   # associated with severe pulmonary oedema
    "G2-glycoprotein-shift":        15,   # altered receptor binding
    "S-segment-reassortment":        8,
    "aerosol-human-to-human":       20,
    "aerosol-suspected":            12,
    "aerosol-rodent-excreta":        5,
    "fomite-or-rodent-excreta":      3,
    "low_ct":                       10,   # PCR Ct < 25
    "elderly":                       8,   # age > 65
    "large_cluster":                 7,   # cluster_size > 5
    "incomplete_genome":             5,   # genome_completeness < 0.80
}

def risk_node(state: OutbreakState) -> Dict[str, Any]:
    """
    Compute a 0–100 composite risk score and assign a tier + action.

    Returns: risk_score, risk_tier, recommended_action
    """
    score = 0.0
    reasons = []

    clade     = state.get("clade", "")
    mutations = state.get("mutation_flags", []) or []
    t_mode    = state.get("transmission_mode", "")
    ct        = state.get("pcr_ct_value")
    age       = state.get("patient_age", 0)
    cluster   = state.get("cluster_size", 0)
    completeness = state.get("genome_completeness", 1.0)

    # Clade weight
    if clade in _RISK_WEIGHTS:
        score += _RISK_WEIGHTS[clade]
        reasons.append(f"clade={clade}")

    # Mutation weights
    for mut in mutations:
        if mut in _RISK_WEIGHTS:
            score += _RISK_WEIGHTS[mut]
            reasons.append(f"mutation={mut}")

    # Transmission mode
    if t_mode in _RISK_WEIGHTS:
        score += _RISK_WEIGHTS[t_mode]

    # PCR viral load
    if ct is not None and ct < 25:
        score += _RISK_WEIGHTS["low_ct"]
        reasons.append(f"high_viral_load(Ct={ct})")

    # Age
    if age > 65:
        score += _RISK_WEIGHTS["elderly"]
        reasons.append(f"elderly_patient(age={age})")

    # Cluster size
    if cluster > 5:
        score += _RISK_WEIGHTS["large_cluster"]
        reasons.append(f"large_cluster(n={cluster})")

    # Genome quality penalty
    if completeness < 0.80:
        score += _RISK_WEIGHTS["incomplete_genome"]

    score = min(score, 100.0)

    # Tier assignment
    if score >= 75:
        tier   = "CRITICAL"
        action = ("Immediate isolation. Notify MoH within 2 hours. "
                  "Activate IPC team. Contact trace all vessel passengers.")
    elif score >= 50:
        tier   = "HIGH"
        action = ("Hospitalise with airborne precautions. "
                  "Notify public health authority within 24 hours. "
                  "Monitor all declared contacts for 45 days.")
    elif score >= 25:
        tier   = "MEDIUM"
        action = ("Outpatient monitoring with daily check-ins. "
                  "PCR retest at day 7. Self-isolate.")
    else:
        tier   = "LOW"
        action = "Standard surveillance. No immediate escalation required."

    return {
        "risk_score": round(score, 1),
        "risk_tier":  tier,
        "recommended_action": action,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 4 — Critic / Quality Gate
# ─────────────────────────────────────────────────────────────────────────────

def critic_node(state: OutbreakState) -> Dict[str, Any]:
    """
    Audit the outputs of all previous nodes for logical consistency.

    Flags issues that should trigger re-evaluation (loop back to genomic_node).
    If critic_flags is empty → approved=True and final_report is generated.

    Returns: critic_flags, approved, final_report
    """
    flags = []

    # ── Consistency checks ────────────────────────────────────────────────────
    clade        = state.get("clade", "")
    mutations    = state.get("mutation_flags", []) or []
    risk_score   = state.get("risk_score", 0)
    risk_tier    = state.get("risk_tier", "")
    t_mode       = state.get("transmission_mode", "")
    completeness = state.get("genome_completeness", 1.0)
    cluster      = state.get("cluster_size", 0)

    # Rule 1: ANDV + aerosol mode but risk tier not HIGH/CRITICAL is suspicious
    if "ANDV" in clade and "aerosol" in t_mode and risk_tier in ("LOW", "MEDIUM"):
        flags.append(
            "ANDV aerosol transmission reported but risk tier is only "
            f"{risk_tier} (score={risk_score}) — possible under-scoring"
        )

    # Rule 2: Incomplete genome with confident clade call needs re-evaluation
    if completeness < 0.70 and clade != "UNKNOWN-clade":
        flags.append(
            f"Genome completeness {completeness:.0%} is too low for confident "
            f"clade assignment ({clade}) — re-sequence recommended"
        )

    # Rule 3: Large cluster with no vessel/location anchor
    if cluster > 8 and not state.get("vessel") and not state.get("exposure_location"):
        flags.append(
            f"Cluster size={cluster} but no exposure location anchor — "
            "contact linkage may be unreliable"
        )

    # Rule 4: Unknown clade with CRITICAL tier — flag for expert review
    if clade == "UNKNOWN-clade" and risk_tier == "CRITICAL":
        flags.append(
            "CRITICAL tier assigned for UNKNOWN-clade — requires expert "
            "phylogenetic review before public health escalation"
        )

    approved = len(flags) == 0
    report   = _build_report(state, flags) if approved else None

    return {
        "critic_flags": flags,
        "approved":     approved,
        "final_report": report,
    }


def _build_report(state: OutbreakState, flags: list) -> str:
    """Generate a human-readable markdown triage summary."""
    return f"""# Outbreak Triage Report
**Case ID:** {state.get('case_id', 'N/A')}
**Generated:** outbreak-agent v1.0

## Genomic Profile
- **Clade:** {state.get('clade')}
- **Mutations:** {', '.join(state.get('mutation_flags') or ['None'])}
- **Genome completeness:** {state.get('genome_completeness', 0):.1%}

## Epidemiological Linkage
- **Transmission mode:** {state.get('transmission_mode')}
- **Cluster size:** {state.get('cluster_size')} contacts
- **Vessel:** {state.get('vessel') or 'N/A'}

## Risk Assessment
- **Score:** {state.get('risk_score')} / 100
- **Tier:** {state.get('risk_tier')}
- **Action:** {state.get('recommended_action')}

## Quality Gate
Approved -- no critic flags raised.

---
Ankur Sharma, PhD -- Apache 2.0 (public nodes)
Clinical scoring module: private -- contact ankurs103@gmail.com
"""
