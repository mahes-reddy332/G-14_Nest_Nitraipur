# Neural Clinical Data Mesh
# Clinical Dataflow Optimizer

Strategic Framework for Real-Time Clinical Dataflow Optimization using Agentic AI and Graph-Based Data Architecture.

## Overview

This solution implements the **Neural Clinical Data Mesh** architecture - a comprehensive framework that transforms fragmented clinical trial data from flat CSV files into a **multi-dimensional knowledge graph**. This moves beyond the traditional "Data Lake" (where data sits passively) to a "Data Mesh" (where data products are active and interconnected).

### Key Capabilities

1. **Knowledge Graph Database** - Patient-centric graph structure using NetworkX (offline alternative to Neo4j)
2. **Multi-Hop Graph Queries** - Complex queries across data sources via simple graph traversals
3. **Digital Patient Twins** - Unified patient representations with all related data
4. **Clean Patient Status** - Dynamic, boolean metric for each patient
5. **Data Quality Index (DQI)** - Weighted multi-dimensional quality score
6. **Autonomous AI Agents** - Rex, Codex, and Lia for automated data stewardship
7. **Interactive Dashboards** - Real-time visualization of trial health

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      NEURAL CLINICAL DATA MESH                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ │
│  │ CPID_EDC  │ │ SAE       │ │ Visit     │ │ MedDRA    │ │ WHODRA    │ │
│  │ Metrics   │ │ Dashboard │ │ Tracker   │ │ Coding    │ │ Coding    │ │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ │
│        │             │             │             │             │        │
│  ┌─────┴─────────────┴─────────────┴─────────────┴─────────────┴─────┐ │
│  │                   KNOWLEDGE GRAPH (NetworkX)                       │ │
│  │    ┌─────────┐                                                     │ │
│  │    │ Patient │◄──────── Central Anchor Node                        │ │
│  │    └────┬────┘                                                     │ │
│  │         │                                                          │ │
│  │    ┌────┼────────────────┬───────────────┬────────────────┐       │ │
│  │    │    │                │               │                │       │ │
│  │    ▼    ▼                ▼               ▼                ▼       │ │
│  │ ┌──────┐ ┌──────┐ ┌───────────┐ ┌────────────┐ ┌──────────────┐  │ │
│  │ │Visit │ │ SAE  │ │Discrepancy│ │CodingTerm  │ │     Site     │  │ │
│  │ │Event │ │ Node │ │  /Query   │ │(MedDRA/WHO)│ │     Node     │  │ │
│  │ └──────┘ └──────┘ └───────────┘ └────────────┘ └──────────────┘  │ │
│  │                                                                    │ │
│  │  Edges: HAS_VISIT, HAS_ADVERSE_EVENT, HAS_CODING_ISSUE,           │ │
│  │         HAS_QUERY, ENROLLED_AT, REQUIRES_RECONCILIATION           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│  ┌─────────────────────────────▼────────────────────────────────────┐  │
│  │                    GRAPH QUERY ENGINE                             │  │
│  │   • Multi-hop queries: "Patients with Missing Visit AND          │  │
│  │     Open Query AND Uncoded Term"                                  │  │
│  │   • Patient 360° View via graph traversal                         │  │
│  │   • Risk scoring using graph metrics                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                │                                        │
│  ┌─────────────────────────────▼────────────────────────────────────┐  │
│  │                    DIGITAL PATIENT TWINS                          │  │
│  │               (Unified JSON representations)                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                │                                        │
│  ┌─────────────────────────────▼────────────────────────────────────┐  │
│  │             AGENTIC AI FRAMEWORK                                  │  │
│  │  ┌─────┐      ┌───────┐      ┌─────┐                             │  │
│  │  │ Rex │      │ Codex │      │ Lia │                             │  │
│  │  └──┬──┘      └───┬───┘      └──┬──┘                             │  │
│  │  ┌──▼─────────────▼─────────────▼──┐                             │  │
│  │  │      SUPERVISOR AGENT           │                             │  │
│  │  └─────────────────────────────────┘                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## The Data Integration Layer: From Tables to Graphs

The core of the solution is a **unified patient-centric data model**. Instead of treating the nine files as separate tables to be joined via VLOOKUPs or SQL JOINS, we ingest them into a graph database structure using **NetworkX** (offline alternative to Neo4j/Amazon Neptune) where the **Subject ID acts as the central anchor node**.

### Node Types (Semantic Network)

| Node Type | Source File | Key Attributes |
|-----------|-------------|----------------|
| **Patient** | CPID_EDC_Metrics | Subject ID, Site ID, Country, Region, Status |
| **Event/Visit** | Visit Projection Tracker | Visit Name, Projected Date, Days Outstanding |
| **Discrepancy** | CPID_EDC_Metrics | Query ID, Query Type (DM, Clinical, Safety, Coding), Status |
| **SAE** | SAE Dashboard | Review Status, Action Status, Requires Reconciliation |
| **CodingTerm** | GlobalCodingReport_MedDRA/WHODRA | Verbatim Term, Coded Term, Coding Status |
| **Site** | Aggregated | Site ID, Country, Total Patients, DQI |

### Edge Types (Relationships)

| Edge Type | Description | Properties |
|-----------|-------------|------------|
| `HAS_VISIT` | Patient → Event | Projected Date, Days Outstanding |
| `HAS_ADVERSE_EVENT` | Patient → SAE | Review Status, Action Status |
| `HAS_CODING_ISSUE` | Patient → CodingTerm | Verbatim Term, Coding Status |
| `HAS_QUERY` | Patient → Discrepancy | Query Type, Status |
| `ENROLLED_AT` | Patient → Site | - |

## Why Graph over Relational?

The graph structure allows for the execution of **complex, multi-hop queries** that relational databases struggle to perform efficiently.

### Example Query

> "Show me all patients who have a Missing Visit (from Tracker) AND an Open Safety Query (from CPID) AND an Uncoded Concomitant Medication (from WHODRA)"

**SQL Approach:**
- Requires joining 3+ tables with potentially mismatched keys
- Complex WHERE clauses with multiple conditions
- Performance degrades with data volume

**Graph Approach:**
- Simple traversal of patient node's neighbors
- O(1) lookup for connected nodes
- Query: `find_patients_needing_attention()`

## Feature Engineering Integration

The Neural Clinical Data Mesh now includes **production-grade feature engineering** that transforms raw clinical data into sophisticated ML-ready features for intelligent decision-making.

### Engineered Features

#### 1. Operational Velocity Index
**Purpose:** Measures query resolution efficiency and bottleneck detection
- **Resolution Velocity:** Δ(Closed Queries) / Δt (queries/day)
- **Accumulation Velocity:** Δ(Open Queries) / Δt (queries/day)
- **Net Velocity:** Resolution - Accumulation
- **Bottleneck Detection:** Sites with negative net velocity

#### 2. Normalized Data Density
**Purpose:** Quantifies data entry patterns and query density
- **Raw Density:** Total Queries / Pages Entered
- **Normalized Score:** 0-1 scale across all sites
- **Percentile Ranking:** Compared to other sites
- **High-Density Detection:** Sites >80th percentile

#### 3. Manipulation Risk Score
**Purpose:** Detects potential data manipulation patterns
- **Inactivation Patterns:** Forms inactivated per month
- **Endpoint Risk:** Impact on primary endpoint data
- **Risk Classification:** Low/Medium/High/Critical levels
- **Audit Trail Analysis:** Based on form modification patterns

#### 4. Composite Risk Score
**Purpose:** Unified risk assessment combining all features
- **Weighted Combination:** Velocity (30%) + Density (20%) + Manipulation (40%) + Intervention (10%)
- **Intervention Threshold:** Score ≥60 triggers automated actions
- **Agent Prioritization:** Used by AI agents for intelligent action ranking

### Agent Feature Integration

The **SupervisorAgent** now uses engineered features for intelligent prioritization:

```python
# Feature-aware prioritization boosts priority for:
- Sites with velocity bottlenecks (negative net velocity)
- Patients at high-density sites (>80th percentile)
- Critical manipulation risk patients
- High composite risk scores (≥60)
```

### Production Features

- **Response Caching:** 30-minute TTL for API performance
- **Circuit Breaker:** Automatic failure detection and recovery
- **Graceful Degradation:** Fallback reasoning when AI services unavailable
- **Performance Benchmarking:** Comprehensive metrics tracking

## Installation

```bash
# Install required packages
pip install pandas numpy openpyxl plotly networkx
```

## Usage

### Quick Start

```python
from clinical_dataflow_optimizer.main_analysis import ClinicalDataflowAnalyzer

# Initialize with path to study data (enable_graph=True by default)
analyzer = ClinicalDataflowAnalyzer("path/to/QC Anonymized Study Files", enable_graph=True)

# Run complete analysis
results = analyzer.run_full_analysis()

# Print executive summary
analyzer.print_executive_summary()

# Generate dashboards
dashboard_paths = analyzer.generate_dashboards()

# Save knowledge graphs
graph_paths = analyzer.save_knowledge_graphs()
```

### Multi-Hop Graph Queries

The graph structure enables complex queries that would require multiple SQL JOINs:

```python
# Find patients needing attention (Missing Visit AND Open Query AND Uncoded Term)
attention_patients = analyzer.query_patients_needing_attention()

# Get 360-degree view of a patient
patient_view = analyzer.get_patient_360_view("SUBJECT_001", "Study_1")

# Execute custom mesh query
query_spec = {
    "patient_filters": [
        {"field": "missing_visits", "op": "gt", "value": 0},
        {"field": "open_queries", "op": "gt", "value": 5}
    ],
    "required_relationships": ["HAS_ADVERSE_EVENT"],
    "logic": "AND"
}
results = analyzer.execute_custom_mesh_query("Study_1", query_spec)
```

### Direct Graph Access

```python
from clinical_dataflow_optimizer.graph import (
    GraphQueryEngine, 
    GraphAnalytics,
    QueryCondition, 
    QueryOperator
)

# Get the query engine for a study
query_engine = analyzer.query_engines["Study_1"]

# Find patients with open queries
patients = query_engine.find_patients_with_open_queries(min_days_open=7)

# Find patients with SAE issues
sae_patients = query_engine.find_patients_with_sae_issues(review_status="Pending")

# Run graph analytics
from clinical_dataflow_optimizer.graph import GraphAnalytics
analytics = GraphAnalytics(analyzer.knowledge_graphs["Study_1"])
risk_report = analytics.export_risk_report()
```

### Graph Analytics

```python
from clinical_dataflow_optimizer.graph import GraphAnalytics

# Initialize analytics
analytics = GraphAnalytics(analyzer.knowledge_graphs["Study_1"])

# Analyze patient risk profile
patient_risk = analytics.analyze_patient_risk("SUBJECT_001")
print(f"Composite Risk Score: {patient_risk.composite_risk_score}")
print(f"Safety Risk: {patient_risk.safety_risk}")

# Analyze site risk
site_risk = analytics.analyze_site_risk("SITE_001")
print(f"Site Risk Level: {site_risk.site_risk_level}")

# Detect patterns
patterns = analytics.detect_issue_patterns()

# Find similar patients (using graph structure)
similar = analytics.find_similar_patients("SUBJECT_001", top_n=5)

# Export comprehensive risk report
report = analytics.export_risk_report()
```

## Project Structure

```
clinical_dataflow_optimizer/
├── __init__.py
├── main_analysis.py          # Main orchestration pipeline
├── README.md
│
├── graph/                    # 🆕 NEURAL CLINICAL DATA MESH
│   ├── __init__.py
│   ├── knowledge_graph.py    # Core graph database (NetworkX)
│   ├── graph_builder.py      # CSV → Graph transformation
│   ├── graph_queries.py      # Multi-hop query engine
│   └── graph_analytics.py    # Advanced graph analytics
│
├── core/
│   ├── __init__.py
│   ├── data_ingestion.py     # CSV file loading
│   └── metrics_calculator.py # DQI and Clean Status
│
├── models/
│   ├── __init__.py
│   └── data_models.py        # Digital Patient Twin models
│
├── agents/
│   ├── __init__.py
│   └── agent_framework.py    # Rex, Codex, Lia agents
│
├── visualization/
│   ├── __init__.py
│   └── dashboard.py          # Plotly dashboards
│
└── config/
    ├── __init__.py
    └── settings.py           # Configuration including GraphConfig
```

## Clean Patient Status

A patient is considered **CLEAN** if and only if ALL conditions are met:

| Check | Condition |
|-------|-----------|
| ✓ Missing Visits | = 0 |
| ✓ Missing Pages | = 0 |
| ✓ Open Queries | = 0 |
| ✓ Uncoded Terms | = 0 |
| ✓ Reconciliation Issues | = 0 |
| ✓ Verification % | ≥ 75% |
| ✓ Safety Data | Reconciled |

## Data Quality Index (DQI)

Weighted penalization model aligned with RBQM principles:

```
DQI = 100 - (W_visit × f(M_visit) + W_query × f(M_query) + 
             W_conform × f(M_conform) + W_safety × f(M_safety))
```

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Visit Adherence | 20% | Missing visits impact |
| Query Responsiveness | 20% | Query handling efficiency |
| Conformance | 20% | Non-conformant data rate |
| Safety Criticality | 40% | Patient safety implications (highest) |

## AI Agents

### Rex (Reconciliation Agent)
Ensures concordance between Clinical and Safety databases.

### Codex (Coding Agent)
Automates medical coding with human-in-the-loop validation.

### Lia (Site Liaison Agent)
Proactive site management and visit compliance monitoring.

## Configuration

Edit `config/settings.py` to customize:

```python
from clinical_dataflow_optimizer.config.settings import GraphConfig

# Graph configuration
graph_config = GraphConfig(
    enable_persistence=True,
    persistence_dir="graph_data",
    max_traversal_depth=3,
    risk_weight_reconciliation=15.0,
    attention_threshold_missing_visits=1
)
```

## Output Files

- `reports/analysis_results.json` - Complete analysis data
- `reports/Study_X_dashboard.html` - Interactive dashboards
- `graph_data/Study_X_graph.gpickle` - Persisted knowledge graphs
- `graph_data/Study_X_graph.json` - Graph metadata

## License

MIT License
