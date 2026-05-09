# outbreak-agent — tests/test_graph.py
# Copyright 2026 Ankur Sharma, PhD  |  Apache 2.0
#
# Layer 2: Full graph integration tests using FakeListLLM.
# No real API key needed. Zero cost.
# Run: pytest tests/test_graph.py -v

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from agent import run_graph
from mock_data import (
    MOCK_CASE_HONDIUS,
    MOCK_CASE_HIGH_RISK,
    MOCK_CASE_LOW_RISK,
    MOCK_CASE_INCOMPLETE_GENOME,
    ALL_CASES,
)


class TestGraphIntegration:
    """
    Tests that run the full LangGraph state machine (all 4 nodes) end-to-end.
    No LLM is used — nodes are deterministic heuristic functions.
    """

    def test_hondius_graph_returns_critical(self):
        state = run_graph(MOCK_CASE_HONDIUS)
        assert state["risk_tier"] == "CRITICAL"

    def test_hondius_graph_approved(self):
        state = run_graph(MOCK_CASE_HONDIUS)
        assert state["approved"] is True

    def test_hondius_final_report_contains_case_id(self):
        state = run_graph(MOCK_CASE_HONDIUS)
        assert state["final_report"] is not None
        assert MOCK_CASE_HONDIUS["case_id"] in state["final_report"]

    def test_low_risk_graph_returns_low_or_medium(self):
        state = run_graph(MOCK_CASE_LOW_RISK)
        assert state["risk_tier"] in ("LOW", "MEDIUM")

    def test_all_required_keys_populated(self):
        """Every node must have written its outputs into final state."""
        required = [
            "clade", "mutation_flags", "genome_completeness",
            "contact_cluster", "transmission_mode", "cluster_size",
            "risk_score", "risk_tier", "recommended_action",
            "critic_flags", "approved",
        ]
        state = run_graph(MOCK_CASE_HONDIUS)
        for key in required:
            assert key in state, f"Missing key: {key}"
            assert state[key] is not None, f"Key is None: {key}"

    def test_loop_count_does_not_exceed_max(self):
        """Graph safety valve: _loop_count must be <= MAX_LOOPS (3)."""
        from agent import MAX_LOOPS
        for case in ALL_CASES:
            state = run_graph(case)
            assert state.get("_loop_count", 0) <= MAX_LOOPS, \
                f"Loop count exceeded MAX_LOOPS for {case['case_id']}"

    def test_all_cases_run_without_exception(self):
        """Smoke: every mock case completes without raising."""
        for case in ALL_CASES:
            state = run_graph(case)
            assert "risk_tier" in state

    def test_high_risk_score_higher_than_low_risk(self):
        high_state = run_graph(MOCK_CASE_HIGH_RISK)
        low_state  = run_graph(MOCK_CASE_LOW_RISK)
        assert high_state["risk_score"] > low_state["risk_score"]

    def test_hondius_cluster_larger_than_input_contacts(self):
        """linkage_node should add adjacent cabin contacts."""
        state = run_graph(MOCK_CASE_HONDIUS)
        input_contacts = len(MOCK_CASE_HONDIUS["contacts"])
        assert state["cluster_size"] > input_contacts

    def test_incomplete_genome_case_completes(self):
        """Even a degraded-genome case should reach a final state."""
        state = run_graph(MOCK_CASE_INCOMPLETE_GENOME)
        assert state["risk_tier"] is not None
