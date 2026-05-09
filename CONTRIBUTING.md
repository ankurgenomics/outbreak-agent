# Contributing to outbreak-agent

Thank you for your interest! Contributions are welcome under the Apache 2.0 license.

## What You Can Contribute

- Bug fixes and performance improvements
- Additional mock outbreak scenarios (`mock_data.py`)
- New LangGraph nodes (e.g., `alert_node`, `report_node`)
- Better test coverage
- Documentation improvements

## What is NOT in Scope (Private Modules)

The following are kept in a separate private repository and are **not** accepting
public contributions at this time:

- `genomics_tools/` — real FASTA/VCF parsers and clinical scoring logic
- `clinical_triage/` — proprietary risk stratification models
- Any production patient data or validated clinical thresholds

If you need access to the clinical module for legitimate research purposes,
contact Ankur Sharma at ankurs103@gmail.com with your institutional affiliation.

## How to Contribute

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Make your changes and add/update tests
4. Verify all tests pass: `pytest tests/`
5. Open a Pull Request with a clear description

## Code Style

- Python 3.9+
- Type hints on all public functions
- Docstrings on all node functions
- `pytest` for all tests — no test, no merge

## Attribution

By submitting a contribution, you agree that your work will be licensed under
Apache 2.0 and that Ankur Sharma, PhD retains the right to use it in derivative
works including the clinical module.

---

*Ankur Sharma, PhD -- outbreak-agent is a public research demonstration.*
*Blog: https://ankurgenomics.github.io | GitHub: https://github.com/ankurgenomics*
