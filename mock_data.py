# outbreak-agent
# Copyright 2026 Ankur Sharma, PhD
# Licensed under the Apache License, Version 2.0

"""
mock_data.py — Pre-built test cases for outbreak-agent.

Three scenarios covering the full risk spectrum:
  - MOCK_CASE_HONDIUS   : MV Hondius cruise ship (ANDV, CRITICAL)
  - MOCK_CASE_HIGH_RISK : Land-based ANDV cluster (HIGH)
  - MOCK_CASE_LOW_RISK  : Rural rodent exposure, SNV (LOW)

Used by demo.py and all test layers.
"""

# ── CRITICAL: MV Hondius cruise ship outbreak (Apr 2026) ─────────────────────
# Based on the publicly reported Andes virus human-to-human transmission event.
# Patient demographics anonymised for demonstration purposes.
MOCK_CASE_HONDIUS = {
    "case_id":            "ANDV-2026-001",
    "patient_age":        68,
    "exposure_location":  "Andes foothills, Argentina",
    "vessel":             "MV Hondius",
    "cabin":              "Deck-4-14B",
    "contacts":           ["spouse", "dining_table_C", "excursion_group_2"],
    "symptom_onset_days": 12,
    "pcr_ct_value":       21.3,              # high viral load
    "genome_sequence":    None,              # mock mode
}

# ── HIGH: Land-based family cluster, confirmed ANDV ──────────────────────────
MOCK_CASE_HIGH_RISK = {
    "case_id":            "ANDV-2026-002",
    "patient_age":        45,
    "exposure_location":  "Patagonia, Chile",
    "vessel":             None,
    "cabin":              None,
    "contacts":           ["sibling_1", "sibling_2", "household_contact_3",
                           "coworker_A", "coworker_B", "neighbor_1"],
    "symptom_onset_days": 8,
    "pcr_ct_value":       26.1,
    "genome_sequence":    None,
}

# ── LOW: Rural rodent-excreta exposure, Sin Nombre virus ─────────────────────
MOCK_CASE_LOW_RISK = {
    "case_id":            "SNV-2026-001",
    "patient_age":        34,
    "exposure_location":  "New Mexico, USA",
    "vessel":             None,
    "cabin":              None,
    "contacts":           ["household_contact_1"],
    "symptom_onset_days": 4,
    "pcr_ct_value":       33.5,              # low viral load
    "genome_sequence":    None,
}

# ── EDGE CASE: Unknown clade, incomplete genome ───────────────────────────────
# Tests critic_node flag: genome completeness < 0.70 with confident clade call
MOCK_CASE_INCOMPLETE_GENOME = {
    "case_id":            "UNK-2026-001",
    "patient_age":        52,
    "exposure_location":  "Seoul, South Korea",
    "vessel":             None,
    "cabin":              None,
    "contacts":           ["contact_1", "contact_2"],
    "symptom_onset_days": 6,
    "pcr_ct_value":       28.0,
    # Simulate a heavily degraded sequence — mostly N bases
    "genome_sequence":    ("ATGCNN" * 100) + ("N" * 1400),
}

ALL_CASES = [
    MOCK_CASE_HONDIUS,
    MOCK_CASE_HIGH_RISK,
    MOCK_CASE_LOW_RISK,
    MOCK_CASE_INCOMPLETE_GENOME,
]
