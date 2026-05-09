# outbreak-agent — tests/test_nodes.py
# Copyright 2026 Ankur Sharma, PhD  |  Apache 2.0
#
# Layer 1: Pure unit tests — no LLM, no graph, zero cost.
# Run: pytest tests/test_nodes.py -v

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from nodes import genomic_node, linkage_node, risk_node, critic_node
from mock_data import (
    MOCK_CASE_HONDIUS,
    MOCK_CASE_HIGH_RISK,
    MOCK_CASE_LOW_RISK,
    MOCK_CASE_INCOMPLETE_GENOME,
)


# ─────────────────────────────────────────────────────────────────────────────
#  genomic_node
# ─────────────────────────────────────────────────────────────────────────────

class TestGenomicNode:

    def test_andv_clade_detected_from_location(self):
        result = genomic_node(MOCK_CASE_HONDIUS)
        assert result["clade"] == "ANDV-S-clade-2026"

    def test_andv_mutations_present(self):
        result = genomic_node(MOCK_CASE_HONDIUS)
        assert len(result["mutation_flags"]) > 0

    def test_snv_clade_detected(self):
        result = genomic_node(MOCK_CASE_LOW_RISK)
        assert result["clade"] == "SNV-NA-clade"

    def test_snv_no_mutations(self):
        result = genomic_node(MOCK_CASE_LOW_RISK)
        assert result["mutation_flags"] == []

    def test_genome_completeness_is_float_in_range(self):
        result = genomic_node(MOCK_CASE_HONDIUS)
        c = result["genome_completeness"]
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0

    def test_incomplete_genome_reduces_completeness(self):
        result = genomic_node(MOCK_CASE_INCOMPLETE_GENOME)
        # The sequence is ~70% N bases — completeness should be low
        assert result["genome_completeness"] < 0.50

    def test_unknown_location_returns_unknown_clade(self):
        case = {**MOCK_CASE_HONDIUS, "exposure_location": "Mars Colony 7", "genome_sequence": None}
        result = genomic_node(case)
        assert result["clade"] == "UNKNOWN-clade"


# ─────────────────────────────────────────────────────────────────────────────
#  linkage_node
# ─────────────────────────────────────────────────────────────────────────────

class TestLinkageNode:

    def _state_with_genomic(self, case, extra=None):
        """Build a minimal state that includes genomic_node outputs."""
        base = {**case}
        genomic_out = genomic_node(case)
        base.update(genomic_out)
        if extra:
            base.update(extra)
        return base

    def test_vessel_adds_adjacent_cabin_contacts(self):
        state = self._state_with_genomic(MOCK_CASE_HONDIUS)
        result = linkage_node(state)
        cluster = result["contact_cluster"]
        # Deck-4-14B → adjacent_cabin_13 and adjacent_cabin_15 should be added
        assert any("adjacent_cabin" in c for c in cluster)

    def test_andv_vessel_mode_is_aerosol_h2h(self):
        state = self._state_with_genomic(MOCK_CASE_HONDIUS)
        result = linkage_node(state)
        assert result["transmission_mode"] == "aerosol-human-to-human"

    def test_snv_no_vessel_mode_is_fomite(self):
        state = self._state_with_genomic(MOCK_CASE_LOW_RISK)
        result = linkage_node(state)
        assert "fomite" in result["transmission_mode"]

    def test_cluster_size_matches_contact_list_length(self):
        state = self._state_with_genomic(MOCK_CASE_LOW_RISK)
        result = linkage_node(state)
        assert result["cluster_size"] == len(result["contact_cluster"])

    def test_large_cluster_high_risk(self):
        state = self._state_with_genomic(MOCK_CASE_HIGH_RISK)
        result = linkage_node(state)
        assert result["cluster_size"] >= 6


# ─────────────────────────────────────────────────────────────────────────────
#  risk_node
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskNode:

    def _full_state(self, case):
        """Run genomic + linkage before risk."""
        state = {**case}
        state.update(genomic_node(case))
        state.update(linkage_node(state))
        return state

    def test_hondius_is_critical(self):
        state = self._full_state(MOCK_CASE_HONDIUS)
        result = risk_node(state)
        assert result["risk_tier"] == "CRITICAL"

    def test_hondius_score_above_75(self):
        state = self._full_state(MOCK_CASE_HONDIUS)
        result = risk_node(state)
        assert result["risk_score"] >= 75

    def test_low_risk_tier_is_low_or_medium(self):
        state = self._full_state(MOCK_CASE_LOW_RISK)
        result = risk_node(state)
        assert result["risk_tier"] in ("LOW", "MEDIUM")

    def test_score_is_capped_at_100(self):
        # Build a worst-case state manually
        state = {**MOCK_CASE_HONDIUS,
                 "clade": "ANDV-S-clade-2026",
                 "mutation_flags": ["N-end-truncation-14aa", "G2-glycoprotein-shift"],
                 "transmission_mode": "aerosol-human-to-human",
                 "pcr_ct_value": 10.0,
                 "patient_age": 80,
                 "cluster_size": 20,
                 "genome_completeness": 0.5}
        result = risk_node(state)
        assert result["risk_score"] <= 100.0

    def test_recommended_action_is_string(self):
        state = self._full_state(MOCK_CASE_HONDIUS)
        result = risk_node(state)
        assert isinstance(result["recommended_action"], str)
        assert len(result["recommended_action"]) > 10


# ─────────────────────────────────────────────────────────────────────────────
#  critic_node
# ─────────────────────────────────────────────────────────────────────────────

class TestCriticNode:

    def _full_state(self, case):
        state = {**case}
        state.update(genomic_node(case))
        state.update(linkage_node(state))
        state.update(risk_node(state))
        return state

    def test_hondius_approved(self):
        state = self._full_state(MOCK_CASE_HONDIUS)
        result = critic_node(state)
        assert result["approved"] is True
        assert result["critic_flags"] == []

    def test_hondius_has_final_report(self):
        state = self._full_state(MOCK_CASE_HONDIUS)
        result = critic_node(state)
        assert result["final_report"] is not None
        assert "ANDV-2026-001" in result["final_report"]

    def test_low_risk_approved(self):
        state = self._full_state(MOCK_CASE_LOW_RISK)
        result = critic_node(state)
        assert result["approved"] is True

    def test_low_completeness_with_confident_clade_raises_flag(self):
        state = self._full_state(MOCK_CASE_HONDIUS)
        # Force low completeness with a non-unknown clade
        state["genome_completeness"] = 0.55
        state["clade"] = "ANDV-S-clade-2026"
        result = critic_node(state)
        assert any("completeness" in f.lower() for f in result["critic_flags"])

    def test_unknown_clade_critical_tier_raises_flag(self):
        state = self._full_state(MOCK_CASE_HONDIUS)
        state["clade"] = "UNKNOWN-clade"
        state["risk_tier"] = "CRITICAL"
        result = critic_node(state)
        assert any("unknown-clade" in f.lower() or "UNKNOWN" in f for f in result["critic_flags"])

    def test_no_report_when_flags_present(self):
        state = self._full_state(MOCK_CASE_HONDIUS)
        state["genome_completeness"] = 0.55
        result = critic_node(state)
        # When flags exist, approved=False and final_report=None
        if not result["approved"]:
            assert result["final_report"] is None
