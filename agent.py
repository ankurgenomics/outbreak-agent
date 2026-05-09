# outbreak-agent
# Copyright 2026 Ankur Sharma, PhD
# Licensed under the Apache License, Version 2.0

"""
agent.py — LangGraph StateGraph wiring for the outbreak triage agent.

Graph topology:
    START → genomic_node → linkage_node → risk_node → critic_node
                ↑_____________________________________________|
                         (only if critic_flags not empty)

Usage:
    from agent import run_graph
    report = run_graph(case_dict)
"""

from langgraph.graph import StateGraph, END

from models import OutbreakState
from nodes import genomic_node, linkage_node, risk_node, critic_node

# Maximum re-evaluation loops before we force-exit (safety valve)
MAX_LOOPS = 3


def _route_critic(state: OutbreakState) -> str:
    """
    Conditional edge function called after critic_node.

    Returns "genomic_node" to loop back for re-evaluation,
    or END to terminate the graph.
    """
    flags     = state.get("critic_flags") or []
    approved  = state.get("approved", False)
    loop_count = state.get("_loop_count", 0)   # internal counter

    if approved or loop_count >= MAX_LOOPS:
        return END
    return "genomic_node"


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""

    def genomic_with_counter(state: OutbreakState) -> dict:
        """Wrap genomic_node to track re-evaluation loop count."""
        result = genomic_node(state)
        result["_loop_count"] = state.get("_loop_count", 0) + 1
        return result

    g = StateGraph(OutbreakState)

    g.add_node("genomic_node",  genomic_with_counter)
    g.add_node("linkage_node",  linkage_node)
    g.add_node("risk_node",     risk_node)
    g.add_node("critic_node",   critic_node)

    g.set_entry_point("genomic_node")
    g.add_edge("genomic_node", "linkage_node")
    g.add_edge("linkage_node", "risk_node")
    g.add_edge("risk_node",    "critic_node")

    g.add_conditional_edges(
        "critic_node",
        _route_critic,
        {
            "genomic_node": "genomic_node",
            END:             END,
        },
    )

    return g.compile()


# Module-level compiled graph (singleton — build once, reuse)
_graph = None

def run_graph(case: dict) -> OutbreakState:
    """
    Run the full triage pipeline on a case dictionary.

    Args:
        case: dict matching OutbreakState fields (at minimum: case_id,
              patient_age, exposure_location, contacts, symptom_onset_days)

    Returns:
        Final OutbreakState after all nodes have run and critic has approved
        (or MAX_LOOPS exhausted).
    """
    global _graph
    if _graph is None:
        _graph = build_graph()

    # Inject internal counter and default Optional fields
    initial_state: OutbreakState = {
        "case_id":            case.get("case_id", "UNKNOWN"),
        "patient_age":        case.get("patient_age", 0),
        "exposure_location":  case.get("exposure_location", ""),
        "vessel":             case.get("vessel"),
        "cabin":              case.get("cabin"),
        "contacts":           case.get("contacts", []),
        "symptom_onset_days": case.get("symptom_onset_days", 0),
        "pcr_ct_value":       case.get("pcr_ct_value"),
        "genome_sequence":    case.get("genome_sequence"),
        # Node outputs — initialised to None
        "clade":              None,
        "mutation_flags":     None,
        "genome_completeness": None,
        "contact_cluster":    None,
        "transmission_mode":  None,
        "cluster_size":       None,
        "risk_score":         None,
        "risk_tier":          None,
        "recommended_action": None,
        "critic_flags":       None,
        "approved":           None,
        "final_report":       None,
        # Internal
        "_loop_count":        0,
    }

    final_state = _graph.invoke(initial_state)
    return final_state
