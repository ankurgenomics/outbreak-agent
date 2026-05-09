# outbreak-agent
# Copyright 2026 Ankur Sharma, PhD
# Licensed under the Apache License, Version 2.0

"""
demo.py — CLI runner for outbreak-agent.

Runs all mock cases through the full LangGraph pipeline and prints
a colour-coded summary to the terminal.

Usage:
    python demo.py                     # run all mock cases
    python demo.py --case hondius      # run one case (hondius | high | low | incomplete)
"""

import sys
import argparse
from agent import run_graph
from report_node import generate_dashboard, generate_pdf
from mock_data import (
    MOCK_CASE_HONDIUS,
    MOCK_CASE_HIGH_RISK,
    MOCK_CASE_LOW_RISK,
    MOCK_CASE_INCOMPLETE_GENOME,
)

# ANSI colours
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

TIER_COLOUR = {
    "CRITICAL": RED,
    "HIGH":     YELLOW,
    "MEDIUM":   BLUE,
    "LOW":      GREEN,
}

CASES = {
    "hondius":    MOCK_CASE_HONDIUS,
    "high":       MOCK_CASE_HIGH_RISK,
    "low":        MOCK_CASE_LOW_RISK,
    "incomplete": MOCK_CASE_INCOMPLETE_GENOME,
}


def print_banner():
    print(f"\n{'='*65}")
    print(f"{BOLD}  outbreak-agent  |  Agentic Outbreak Triage  |  v1.0{RESET}")
    print(f"  Ankur Sharma, PhD  |  Apache 2.0")
    print(f"  https://github.com/ankurgenomics/outbreak-agent")
    print(f"{BOLD}{'='*65}{RESET}\n")


def print_case_result(state: dict):
    case_id  = state.get("case_id", "N/A")
    tier     = state.get("risk_tier", "N/A")
    score    = state.get("risk_score", 0)
    clade    = state.get("clade", "N/A")
    t_mode   = state.get("transmission_mode", "N/A")
    cluster  = state.get("cluster_size", 0)
    flags    = state.get("critic_flags") or []
    approved = state.get("approved", False)
    loops    = state.get("_loop_count", 1)

    colour = TIER_COLOUR.get(tier, RESET)

    print(f"{BOLD}▶ Case: {case_id}{RESET}")
    print(f"  Clade          : {clade}")
    print(f"  Mutations      : {', '.join(state.get('mutation_flags') or ['None'])}")
    print(f"  Genome quality : {state.get('genome_completeness', 0):.1%}")
    print(f"  Transmission   : {t_mode}")
    print(f"  Cluster size   : {cluster} contacts")
    print(f"  Risk score     : {colour}{BOLD}{score}/100{RESET}")
    print(f"  Risk tier      : {colour}{BOLD}{tier}{RESET}")
    print(f"  Action         : {state.get('recommended_action', '')}")

    if flags:
        print(f"\n  {YELLOW}  Critic flags ({loops} loop(s)):{RESET}")
        for f in flags:
            print(f"     - {f}")
    else:
        print(f"\n  {GREEN}  Approved by critic (loops: {loops}){RESET}")

    print(f"\n{'─'*65}\n")


def run_case(name: str):
    case = CASES.get(name.lower())
    if not case:
        print(f"Unknown case '{name}'. Choose from: {', '.join(CASES)}")
        sys.exit(1)
    print(f"Running case: {BOLD}{name}{RESET}\n")
    state = run_graph(case)
    print_case_result(state)
    if state.get("final_report"):
        print(state["final_report"])
    # Generate risk dashboard PNG and PDF triage report
    png_path = generate_dashboard(state)
    pdf_path = generate_pdf(state, png_path=png_path)
    print(f"  Dashboard : {png_path}")
    print(f"  PDF Report: {pdf_path}\n")


def run_all():
    for name, case in CASES.items():
        print(f"Running: {BOLD}{name.upper()}{RESET}")
        state = run_graph(case)
        print_case_result(state)


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="outbreak-agent demo runner")
    parser.add_argument(
        "--case", type=str, default=None,
        help="Case to run: hondius | high | low | incomplete (default: all)"
    )
    args = parser.parse_args()

    if args.case:
        run_case(args.case)
    else:
        run_all()


if __name__ == "__main__":
    main()
