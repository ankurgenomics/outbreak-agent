# outbreak-agent -- tests/test_smoke.py
# Ankur Sharma, PhD  |  Apache 2.0
#
# Layer 3: Live smoke test -- requires a real language model API key.
# Cost: ~$0.01 per run (1 model call). Run MANUALLY only.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   pytest tests/test_smoke.py -v -s --smoke
#
# Skipped in CI by default (no --smoke flag).

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from mock_data import MOCK_CASE_HONDIUS


def pytest_addoption(parser):
    """Register --smoke flag so pytest does not error when it is absent."""
    try:
        parser.addoption("--smoke", action="store_true", default=False,
                         help="Run live LLM smoke tests (costs money)")
    except ValueError:
        pass  # already registered by conftest


@pytest.fixture
def smoke(request):
    if not request.config.getoption("--smoke", default=False):
        pytest.skip("Pass --smoke to run live LLM tests")


class TestSmokeWithRealModel:
    """
    Minimal smoke tests to confirm the graph wiring works with a real language model.
    These tests are intentionally thin -- full coverage is in test_nodes.py
    and test_graph.py which are free to run.
    """

    def test_hondius_smoke_returns_critical(self, smoke):
        """
        Run the full graph against the MV Hondius case with a real language model.
        Expects CRITICAL tier -- nodes use heuristics so the model does not affect
        risk_tier; this test mainly checks that no API errors are thrown.
        """
        from agent import run_graph
        state = run_graph(MOCK_CASE_HONDIUS)
        assert state["risk_tier"] == "CRITICAL", (
            f"Expected CRITICAL, got {state['risk_tier']}. "
            f"Score: {state['risk_score']}"
        )
        print(f"\nSmoke test passed | tier={state['risk_tier']} "
              f"score={state['risk_score']} loops={state.get('_loop_count')}")

    def test_hondius_smoke_final_report_non_empty(self, smoke):
        from agent import run_graph
        state = run_graph(MOCK_CASE_HONDIUS)
        assert state.get("final_report"), "final_report should not be empty"
        assert len(state["final_report"]) > 100, "final_report seems too short"
        print(f"\nReport preview:\n{state['final_report'][:300]}...")
