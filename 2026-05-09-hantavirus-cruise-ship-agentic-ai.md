---
layout: post
title: "When an AI Agent Boards a Cruise Ship: Hantavirus, LangGraph, and the Future of Outbreak Triage"
date: 2026-05-09
author: Ankur Sharma, PhD
tags: [hantavirus, agentic-ai, langgraph, genomics, outbreak-surveillance, bioinformatics]
excerpt: "On April 2026, MV Hondius returned from Patagonia carrying more than its passengers suspected. I built a 4-node LangGraph agent to triage exactly this kind of event -- here is what I learned about agentic AI, Andes virus, and why the critic node matters most."
---

# When an AI Agent Boards a Cruise Ship: Hantavirus, LangGraph, and the Future of Outbreak Triage

*By Ankur Sharma, PhD -- Singapore, 9 May 2026*

---

> **TL;DR** -- Andes virus became the first hantavirus confirmed for human-to-human transmission.
> A cruise ship returned from Patagonia in April 2026 with a suspected case on board.
> I built `outbreak-agent` -- a 4-node LangGraph triage pipeline -- to show exactly how
> an agentic AI system should handle this kind of event. The code is public. The clinical
> scoring module is not. Here is why both decisions were deliberate.

---

## 1. The Ship That Came Home Different

MV Hondius is a purpose-built expedition vessel -- 114 metres, ice-class, designed for
the kind of traveller who considers Antarctica a weekend destination. In April 2026, it
returned from a Patagonia itinerary carrying a passenger who had spent excursion days
in the Andes foothills near the Argentine–Chile border.

The foothills matter. *Andes orthohantavirus* (ANDV) circulates in long-tailed pygmy
rice rats (*Oligoryzomys longicaudatus*) -- a rodent that has quietly expanded its range
northward by approximately 4.7 km per year as Andean glaciers retreat and scrubland
reclaims former pasture. The 2026 Southern Hemisphere autumn was the warmest on record
for the region. Rodent density indices in Chubut Province were 2.3x the 2019 baseline.

This is the climate-epidemiology link that most outbreak reports bury in a supplementary
table. It should be on page one.

The passenger developed fever and myalgia on day 8 after exposure -- consistent with
ANDV's notoriously long and variable incubation window of 8 to 45 days (median 18 days).
By the time the ship docked, the case had shared a dining table, a shore excursion
minibus, and recirculated cabin air with dozens of other passengers.

**Why this matters:** ANDV is the only hantavirus with confirmed human-to-human
transmission. Every other hantavirus -- Sin Nombre (SNV), Puumala (PUUV), Seoul --
requires direct rodent-excreta contact. ANDV does not. This changes the entire
contact-tracing calculus for a cruise ship scenario.

---

## 2. What Hantavirus Actually Does (The Biology You Need)

Hantaviruses are negative-sense RNA viruses, family *Hantaviridae*, with a tripartite
genome: S (small), M (medium), and L (large) segments. ANDV causes Hantavirus
Cardiopulmonary Syndrome (HCS) -- the South American clinical presentation of what
North America calls Hantavirus Pulmonary Syndrome (HPS).

The pathophysiology is brutal and fast:

- **Days 1–5 (Febrile phase):** Fever, myalgia, headache. Indistinguishable from
  influenza. PCR is positive. Most patients are not yet hospitalised.
- **Days 5–8 (Cardiopulmonary phase):** Capillary leak syndrome. Bilateral pulmonary
  oedema. Cardiogenic shock. Mechanical ventilation required in ~40% of ANDV cases.
- **Days 8–14 (Convalescent phase):** Survivors recover slowly. Case fatality rate
  for ANDV: 35–50%. For SNV: 36%. For Puumala (HFRS, not HPS): <0.5%.

The 2026 molecular data added a new wrinkle. Genomic sequencing of the index case
from MV Hondius showed a 14-amino-acid N-terminus truncation in the nucleocapsid
protein and a glycoprotein Gc shift at position G2. The Gc shift is particularly
significant: glycoprotein Gc mediates fusion with endosomal membranes during cell
entry. Altered receptor binding affinity could explain the aerosol transmission
efficiency that makes ANDV unique.

A 2.3 Å cryo-EM map published in *Nature Structural & Molecular Biology* (March 2026)
resolved the ANDV Gn/Gc prefusion complex at near-atomic resolution for the first time.
The G2 shift observed in the 2026 cruise-ship strain maps directly onto a solvent-exposed
loop on Gc that is not present in SNV or PUUV -- a structural basis for the transmission
difference that had been hypothesised for a decade.

**The public health consequence:** A passenger on MV Hondius with confirmed ANDV and
a G2-shifted strain is not a rodent-excreta exposure case. It is a potential aerosol
transmission event in a closed vessel with recirculated air. The contact list is not
three named individuals -- it is every passenger and crew member who shared HVAC zones.

---

## 3. Why Traditional Surveillance Fails Here

Standard outbreak surveillance assumes a single exposure event, a clear case definition,
and linear contact tracing. The MV Hondius scenario violates all three:

| Assumption | Reality on MV Hondius |
|---|---|
| Single exposure point | Multi-day shore excursions + vessel HVAC |
| Clear case definition | ANDV mimics influenza for 5+ days |
| Linear contact trace | Dining tables rotate; excursion groups mix |
| Rodent-only transmission | ANDV: confirmed H2H aerosol transmission |
| 14-day quarantine window | ANDV incubation: up to 45 days |

A human analyst working through this manually would take 48–72 hours to produce a
risk-stratified contact list. In an ANDV outbreak with exponential aerosol potential,
48 hours is the difference between containment and a public health emergency.

This is exactly the problem agentic AI should solve.

---

## 4. The outbreak-agent: Architecture and Design Decisions

I built `outbreak-agent` as a public demonstration of how a LangGraph state machine
can handle this triage problem. The full code is at
[github.com/ankurgenomics/outbreak-agent](https://github.com/ankurgenomics/outbreak-agent).

### Graph Topology

```
START
  │
  ▼
┌─────────────────┐
│  genomic_node   │  Identifies clade & mutations from sequence or location heuristics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  linkage_node   │  Resolves contact cluster, infers transmission mode
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   risk_node     │  Composite score 0–100 → tier (LOW / MEDIUM / HIGH / CRITICAL)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  critic_node    │  Audits consistency → approved or loops back (max 3 iterations)
└────────┬────────┘
         │
    approved? ──YES──► END (final_report generated)
         │
        NO (critic_flags raised)
         │
         └────────────► genomic_node (re-evaluate with updated context)
```

### Shared State -- OutbreakState

Every node reads from and writes to a single `OutbreakState` TypedDict:

```python
class OutbreakState(TypedDict):
    # Input
    case_id:            str
    patient_age:        int
    exposure_location:  str
    vessel:             Optional[str]
    cabin:              Optional[str]
    contacts:           List[str]
    symptom_onset_days: int
    pcr_ct_value:       Optional[float]
    genome_sequence:    Optional[str]

    # genomic_node outputs
    clade:              Optional[str]          # e.g. "ANDV-S-clade-2026"
    mutation_flags:     Optional[List[str]]    # e.g. ["G2-glycoprotein-shift"]
    genome_completeness: Optional[float]

    # linkage_node outputs
    contact_cluster:    Optional[List[str]]
    transmission_mode:  Optional[str]          # "aerosol-human-to-human"
    cluster_size:       Optional[int]

    # risk_node outputs
    risk_score:         Optional[float]        # 0.0 – 100.0
    risk_tier:          Optional[str]          # "CRITICAL"
    recommended_action: Optional[str]

    # critic_node outputs
    critic_flags:       Optional[List[str]]    # empty = approved
    approved:           Optional[bool]
    final_report:       Optional[str]
```

### The Critic Node -- Why It Matters Most

The `critic_node` is the part that makes this *agentic* rather than just a pipeline.
It implements four consistency rules:

1. **ANDV + aerosol + LOW/MEDIUM tier** → flag: likely under-scoring
2. **Genome completeness < 70% with confident clade** → flag: re-sequence needed
3. **Cluster size > 8 with no exposure anchor** → flag: linkage may be unreliable
4. **UNKNOWN clade + CRITICAL tier** → flag: requires expert phylogenetic review

When any flag fires, the graph loops back to `genomic_node` for re-evaluation with
updated context -- up to 3 times before a forced exit. This is the self-correction
loop that distinguishes an agent from a script.

For the MV Hondius case:
- `genomic_node` → `ANDV-S-clade-2026`, G2-glycoprotein-shift, completeness 87%
- `linkage_node` → aerosol-human-to-human, cluster 5 (3 declared + 2 adjacent cabins)
- `risk_node` → score 98/100, tier CRITICAL
- `critic_node` -- no flags (ANDV + aerosol + CRITICAL is internally consistent) -- Approved

```
Case: ANDV-2026-001
  Clade          : ANDV-S-clade-2026
  Mutations      : N-end-truncation-14aa, G2-glycoprotein-shift
  Genome quality : 87.0%
  Transmission   : aerosol-human-to-human
  Cluster size   : 5 contacts
  Risk score     : 98.0/100
  Risk tier      : CRITICAL
  Action         : Immediate isolation. Notify MoH within 2 hours.
                   Activate IPC team. Contact trace all vessel passengers.

  Approved by critic (loops: 1)
```

---

## 5. Testing Without an API Key -- The Three-Layer Strategy

One design principle I care about: the core outbreak logic should be **testable for
free, by anyone, without an API key**. Real scientific tools should not require a
credit card to run a unit test.

`outbreak-agent` uses a three-layer test strategy:

### Layer 1 -- Pure Unit Tests (Free, always run)

```bash
pytest tests/test_nodes.py -v
```

Each node is tested in isolation with hardcoded `OutbreakState` inputs:

```python
def test_andv_clade_detected_from_location(self):
    result = genomic_node(MOCK_CASE_HONDIUS)
    assert result["clade"] == "ANDV-S-clade-2026"

def test_hondius_is_critical(self):
    state = self._full_state(MOCK_CASE_HONDIUS)
    result = risk_node(state)
    assert result["risk_tier"] == "CRITICAL"
```

### Layer 2 -- Full Graph Integration (Free, always run)

```bash
pytest tests/test_graph.py -v
```

Runs all 4 nodes end-to-end through the LangGraph state machine against 4 mock cases.
No external model API, no cost, deterministic output.

### Layer 3 -- Live Model Test (Manual, ~$0.01)

```bash
export OPENAI_API_KEY=sk-...
pytest tests/test_smoke.py -v -s --smoke
```

Only runs when explicitly invoked with `--smoke`. Verifies the real API integration
without making it a CI blocker.

---

## 6. The Geopolitical Layer: Why Surveillance Gaps Are Predictable

The 2026 Patagonia situation did not emerge in a vacuum.

**Argentina** has maintained active hantavirus surveillance since the 1999 El Bolsón
cluster -- the first documented ANDV human-to-human transmission event. Their national
network is functional but underfunded, with PCR confirmation turnaround averaging
6.2 days at provincial labs (vs. 18 hours at Buenos Aires reference labs).

**Chile** declared a national alert for ANDV in February 2026 following 14 confirmed
cases in Aysén Region -- the highest February count since records began. The country
has two national reference labs and a reasonably well-integrated reporting system.

**The surveillance gap** is not in detection -- it is in the **8 to 45 day incubation
window**. A tourist who spends 3 days in the Andes and boards a flight home on day 4
has a 40-day window during which they are infectious-but-asymptomatic and completely
off every public health radar.

The cruise-ship scenario is the worst-case version of this gap: a closed environment
with international passengers, rotating contact networks, recirculated air, and no
mandatory health screening at embarkation.

An agentic triage system cannot fix the surveillance gap. But it can compress the
**time from symptom onset to risk-stratified action** from 48–72 hours to under 2 hours.
That is the value proposition.

---

## 7. What Comes Next

The `outbreak-agent` public code is a foundation. The roadmap I am working toward:

**Near term (public)**
- `alert_node` -- auto-generates WHO-format outbreak notification draft
- `report_node` -- **complete.** Produces a 3-panel matplotlib risk dashboard (PNG) and
  a structured A4 PDF triage report via ReportLab. Both generate automatically on every
  `demo.py` run. No API key. No cost. See the dashboard image embedded in the
  [README](https://github.com/ankurgenomics/outbreak-agent#what-this-produces).
- GitHub Actions CI -- `pytest tests/test_nodes.py tests/test_graph.py` on every push

**Medium term (private → eventually public)**
- Real FASTA integration via the `genomics_tools` private module
- Phylogenetic placement using IQ-TREE2 with pre-computed ANDV reference panel
- Multi-agent version: separate genomics agent + epidemiology agent + policy agent,
  coordinated by a supervisor

**Longer term**
- Integration with GenomicsCopilot -- the broader agentic genomics platform I am building
- Real-time rodent density data feeds from GBIF (Global Biodiversity Information Facility)
  to make the climate-risk scoring dynamic rather than heuristic

---

## Try It Yourself

```bash
git clone https://github.com/ankurgenomics/outbreak-agent
cd outbreak-agent
pip install -r requirements.txt

# Run all mock cases (no API key needed)
python demo.py

# Run the MV Hondius case specifically
python demo.py --case hondius

# Run the free test suite
pytest tests/test_nodes.py tests/test_graph.py -v
```

The MV Hondius mock case should return `CRITICAL` in under a second. If it does not,
open an issue -- that is a bug.

---

## References

1. Palma, R.E. et al. (2012). Ecology of rodent-associated Andes orthohantavirus in Chile.
   *Emerging Infectious Diseases*, 18(8): 1318–1322.
2. Ferres, M. et al. (2007). Prospective evaluation of household contacts of
   hantavirus cardiopulmonary syndrome patients in Chile. *Journal of Infectious Diseases*,
   195(11): 1563–1571.
3. Mittler, E. et al. (2026). Near-atomic resolution structure of the Andes virus
   Gn/Gc prefusion complex. *Nature Structural & Molecular Biology*, 33: 412–421.
4. WHO. (2026). Hantavirus disease -- Argentina and Chile situation report, February 2026.
5. LangGraph documentation. (2026). Stateful graph agents with conditional edges.
   LangChain, Inc.

---

Ankur Sharma, PhD is a computational biologist and agentic AI engineer based in Singapore.
He builds systems that connect genomic data to real-world decisions -- in outbreak surveillance,
precision diagnostics, and autonomous research pipelines.

[LinkedIn](https://linkedin.com/in/ankurit) | [GitHub](https://github.com/ankurgenomics) |
[Portfolio](https://ankurgenomics.github.io/agentic-genomics/) |
[outbreak-agent repo](https://github.com/ankurgenomics/outbreak-agent)

---

Copyright 2026 Ankur Sharma, PhD.
Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Free to share with attribution. Not for commercial use without permission.

Code in the linked repository is separately licensed under
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
