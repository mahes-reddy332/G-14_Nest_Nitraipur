"""
Demo: Data Integration Layer - From Tables to Graphs
Demonstrates the multi-hop query capabilities of the Clinical Data Mesh

This script shows how graph-based queries outperform SQL for complex
clinical trial data analysis:

Example Query: "Show me all patients who have:
- A Missing Visit (from Visit Tracker) AND
- An Open Safety Query (from CPID) AND  
- An Uncoded Concomitant Medication (from WHODRA)"

SQL: Requires 3+ table JOINs with potentially mismatched keys
Graph: Simple traversal of patient node neighbors
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_integration import ClinicalDataMesh, build_clinical_data_mesh
import json


def demo_table_to_graph_transformation(study_folder: str = "Study 1_CPID_Input Files - Anonymization"):
    """Demonstrate the table-to-graph transformation"""
    
    print("=" * 80)
    print("DATA INTEGRATION LAYER: FROM TABLES TO GRAPHS")
    print("=" * 80)
    
    # Setup paths
    base_path = Path(__file__).parent.parent / "QC Anonymized Study Files"
    study_path = base_path / study_folder
    
    if not study_path.exists():
        print(f"Error: Study folder not found at {study_path}")
        return
    
    print("\n" + "=" * 80)
    print("PHASE 1: BUILDING THE KNOWLEDGE GRAPH")
    print("=" * 80)
    print(f"\nSource: {study_path}")
    print("\nTransforming flat CSV/Excel files into semantic network...")
    print("  - Patient nodes (central anchor) - from CPID_EDC_Metrics")
    print("  - Site nodes - aggregated from patient data")
    print("  - Event/Visit nodes - from Visit Projection Tracker")
    print("  - Discrepancy nodes - from CPID_EDC_Metrics (queries)")
    print("  - SAE nodes - from SAE Dashboard")
    print("  - CodingTerm nodes - from GlobalCodingReport_MedDRA/WHODRA")
    print()
    
    # Build the data mesh
    mesh = ClinicalDataMesh()
    mesh.build_from_study_folder(study_path)
    
    # Display statistics
    stats = mesh.get_statistics()
    print("\n--- GRAPH STATISTICS ---")
    print(f"Study ID: {stats.study_id}")
    print(f"Total Nodes: {stats.total_nodes:,}")
    print(f"Total Edges: {stats.total_edges:,}")
    print(f"\nNode Breakdown:")
    for node_type, count in stats.node_counts.items():
        print(f"  - {node_type}: {count:,}")
    print(f"\nEdge Breakdown:")
    for edge_type, count in stats.edge_counts.items():
        print(f"  - {edge_type}: {count:,}")
    print(f"\nAverage connections per patient: {stats.average_patient_connections:.1f}")
    print(f"Max connections per patient: {stats.max_patient_connections}")
    
    print("\n" + "=" * 80)
    print("PHASE 2: MULTI-HOP QUERY DEMONSTRATIONS")
    print("=" * 80)
    
    # Query 1: The flagship multi-hop query
    print("\n" + "-" * 60)
    print("QUERY 1: Patients Needing Immediate Attention")
    print("-" * 60)
    print("""
This is the query that demonstrates why graph databases excel:
Find patients with:
  - Missing Visit (from Visit Tracker) AND
  - Open Query (from CPID) AND
  - Uncoded Term (from GlobalCodingReport)

SQL Equivalent:
  SELECT p.* 
  FROM patients p
  INNER JOIN visit_tracker v ON p.subject_id = v.subject_id 
    AND v.is_missing = TRUE
  INNER JOIN cpid_queries q ON p.subject_id = q.subject_id 
    AND q.status = 'Open'
  INNER JOIN coding_report c ON p.subject_id = c.subject_id 
    AND c.status = 'Uncoded'

Graph: Simple neighbor traversal - O(connections per patient)
""")
    
    result1 = mesh.query_patients_needing_attention()
    print(f"Results: {result1.patient_count} patients found")
    print(f"Execution time: {result1.execution_time_ms:.2f}ms")
    if result1.patients:
        print("\nTop 5 highest-risk patients:")
        for p in result1.patients[:5]:
            print(f"  * {p.subject_id} - Risk Score: {p.risk_score:.1f}")
            print(f"    Matched: {', '.join(p.matched_conditions[:3])}")
    
    # Query 2: Visit and Query Issues
    print("\n" + "-" * 60)
    print("QUERY 2: Overdue Visits with Open Queries")
    print("-" * 60)
    print("""
Find patients with visits overdue >30 days AND open queries.

Graph traversal:
  Patient -[HAS_VISIT]-> Event (days_outstanding > 30)
  Patient -[HAS_QUERY]-> Discrepancy (status = 'Open')
""")
    
    result2 = mesh.query_patients_with_visit_and_query_issues(
        min_days_outstanding=30,
        min_open_queries=1
    )
    print(f"Results: {result2.patient_count} patients found")
    print(f"Execution time: {result2.execution_time_ms:.2f}ms")
    
    # Query 3: SAE and Coding Issues
    print("\n" + "-" * 60)
    print("QUERY 3: SAE Pending Review + Uncoded Terms")
    print("-" * 60)
    print("""
Critical for safety reporting - patients with incomplete SAE review
AND uncoded medical terms.

Graph traversal:
  Patient -[HAS_ADVERSE_EVENT]-> SAE (review_status contains 'Pending')
  Patient -[HAS_CODING_ISSUE]-> CodingTerm (status = 'Uncoded')
""")
    
    result3 = mesh.query_patients_with_sae_and_coding_issues(review_status="Pending")
    print(f"Results: {result3.patient_count} patients found")
    print(f"Execution time: {result3.execution_time_ms:.2f}ms")
    
    # Query 4: Flexible multi-criteria
    print("\n" + "-" * 60)
    print("QUERY 4: Flexible Multi-Criteria (OR Logic)")
    print("-" * 60)
    print("""
Find patients with ANY of:
  - Missing Visit OR
  - Open Query OR  
  - Uncoded Term
  
Demonstrates flexibility of graph queries vs rigid SQL JOINs.
""")
    
    result4 = mesh.query_patients_by_multi_criteria(
        has_missing_visit=True,
        has_open_query=True,
        has_uncoded_term=True,
        logic="OR"
    )
    print(f"Results: {result4.patient_count} patients found (with any issue)")
    print(f"Execution time: {result4.execution_time_ms:.2f}ms")
    
    # Query 5: Strict multi-criteria (AND)
    print("\n" + "-" * 60)
    print("QUERY 5: Strict Multi-Criteria (AND Logic)")
    print("-" * 60)
    
    result5 = mesh.query_patients_by_multi_criteria(
        has_missing_visit=True,
        has_open_query=True,
        has_uncoded_term=True,
        logic="AND"
    )
    print(f"Results: {result5.patient_count} patients found (with ALL issues)")
    print(f"Execution time: {result5.execution_time_ms:.2f}ms")
    
    print("\n" + "=" * 80)
    print("PHASE 3: GRAPH ADVANTAGES SUMMARY")
    print("=" * 80)
    print("""
Why Graph > SQL for Clinical Trial Data:

1. MULTI-HOP QUERIES
   SQL: Requires complex JOINs across 3-5 tables
   Graph: Simple neighbor traversal from patient node

2. SCHEMA FLEXIBILITY  
   SQL: Rigid schema, costly to add relationships
   Graph: Add new edge types without schema changes

3. QUERY PERFORMANCE
   SQL: O(n*m*k) for multi-table JOINs
   Graph: O(edges per patient) for traversals

4. SEMANTIC CLARITY
   SQL: Relationships hidden in foreign keys
   Graph: Relationships are first-class citizens (edges)

5. PATTERN DETECTION
   SQL: Complex subqueries for pattern matching
   Graph: Native pattern matching with graph algorithms

Example: "Find patients who share a site with another patient 
         who has an SAE, where both patients have uncoded terms"
   
   SQL: 4+ self-joins with complex conditions
   Graph: Two-hop traversal with edge filtering
""")
    
    # Export results
    print("\n" + "=" * 80)
    print("PHASE 4: EXPORT")
    print("=" * 80)
    
    reports_path = Path(__file__).parent.parent / "reports"
    reports_path.mkdir(exist_ok=True)
    
    # Export graph JSON
    graph_json_path = reports_path / f"{mesh.study_id}_knowledge_graph.json"
    mesh.export_graph_json(graph_json_path)
    print(f"✓ Knowledge graph exported: {graph_json_path}")
    
    # Export patient network CSV
    network_csv_path = reports_path / f"{mesh.study_id}_patient_network.csv"
    mesh.export_patient_network_csv(network_csv_path)
    print(f"✓ Patient network exported: {network_csv_path}")
    
    # Export query results
    query_results_path = reports_path / f"{mesh.study_id}_multi_hop_queries.json"
    with open(query_results_path, 'w') as f:
        json.dump({
            'attention_query': result1.to_dict(),
            'visit_query': result2.to_dict(),
            'sae_coding_query': result3.to_dict(),
            'multi_or': result4.to_dict(),
            'multi_and': result5.to_dict()
        }, f, indent=2, default=str)
    print(f"✓ Query results exported: {query_results_path}")
    
    print("\n" + "=" * 80)
    print("DATA INTEGRATION LAYER DEMO COMPLETE")
    print("=" * 80)
    
    return mesh


def compare_sql_vs_graph_query():
    """Side-by-side comparison of SQL vs Graph query approaches"""
    
    print("\n" + "=" * 80)
    print("SQL vs GRAPH: SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    
    print("""
+-----------------------------------------------------------------------------+
| QUERY: Find patients with Missing Visit AND Open Query AND Uncoded Term    |
+-----------------------------------------------------------------------------+
|                                                                             |
|  SQL APPROACH (Traditional)                                                 |
|  -------------------------                                                  |
|  SELECT DISTINCT p.subject_id, p.site_id, p.status                         |
|  FROM cpid_edc_metrics p                                                    |
|  INNER JOIN (                                                               |
|      SELECT DISTINCT subject_id                                             |
|      FROM visit_projection_tracker                                          |
|      WHERE is_missing = TRUE                                                |
|  ) v ON p.subject_id = v.subject_id                                         |
|  INNER JOIN (                                                               |
|      SELECT DISTINCT subject_id                                             |
|      FROM cpid_queries                                                      |
|      WHERE status = 'Open'                                                  |
|  ) q ON p.subject_id = q.subject_id                                         |
|  INNER JOIN (                                                               |
|      SELECT DISTINCT subject_id                                             |
|      FROM global_coding_report                                              |
|      WHERE coding_status = 'UnCoded'                                        |
|  ) c ON p.subject_id = c.subject_id                                         |
|  WHERE p.study_id = 'STUDY_001';                                            |
|                                                                             |
|  Complexity: O(n x m x k x l) where n,m,k,l = table row counts              |
|  Requires: 4 full table scans + 3 hash joins                                |
|  Index needed: subject_id on ALL tables                                     |
|                                                                             |
+-----------------------------------------------------------------------------+
|                                                                             |
|  GRAPH APPROACH (Knowledge Graph)                                           |
|  --------------------------------                                           |
|                                                                             |
|  // Cypher-style query                                                      |
|  MATCH (p:Patient)-[:HAS_VISIT]->(v:Event {is_missing: true})               |
|  MATCH (p)-[:HAS_QUERY]->(q:Discrepancy {status: 'Open'})                   |
|  MATCH (p)-[:HAS_CODING_ISSUE]->(c:CodingTerm {coding_status: 'UnCoded'})   |
|  RETURN p                                                                   |
|                                                                             |
|  // Python API                                                              |
|  mesh.query_patients_by_multi_criteria(                                     |
|      has_missing_visit=True,                                                |
|      has_open_query=True,                                                   |
|      has_uncoded_term=True,                                                 |
|      logic="AND"                                                            |
|  )                                                                          |
|                                                                             |
|  Complexity: O(p x avg_connections) where p = patient count                 |
|  Requires: Traverse each patient's immediate neighbors only                 |
|  No joins needed: Relationships are first-class edges                       |
|                                                                             |
+-----------------------------------------------------------------------------+

Performance Comparison:
-----------------------
For a study with:
  * 10,000 patients
  * 50,000 visit records
  * 100,000 query records
  * 25,000 coding records

SQL:  ~10,000 x 50,000 x 100,000 x 25,000 comparisons (worst case)
      Even with indexes: 4 index scans + 3 hash joins

Graph: ~10,000 x 15 comparisons (15 avg connections per patient)
       Direct edge traversal, no key matching needed
""")


if __name__ == "__main__":
    # Run the main demo
    mesh = demo_table_to_graph_transformation()
    
    # Show SQL vs Graph comparison
    compare_sql_vs_graph_query()
