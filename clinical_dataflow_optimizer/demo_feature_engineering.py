"""
Feature Engineering Demo Script
Demonstrates the three engineered features for AI/ML model training

Features:
1. Operational Velocity Index (V_res = ΔClosed Queries / Δt)
2. Normalized Data Density (D = Total Queries / Pages Entered)
3. Manipulation Risk Score (based on inactivation patterns)
"""

import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.feature_engineering import (
    SiteFeatureEngineer,
    engineer_study_features,
    OperationalVelocityIndex,
    NormalizedDataDensity,
    ManipulationRiskScore
)
from core.data_ingestion import ClinicalDataIngester
import json


def demo_feature_engineering(study_folder: str = "Study 1_CPID_Input Files - Anonymization"):
    """
    Demonstrate feature engineering for a single study
    """
    print("=" * 70)
    print("FEATURE ENGINEERING FOR CLINICAL TRIAL AI MODELS")
    print("=" * 70)
    
    # Setup paths
    base_path = Path(__file__).parent.parent / "QC Anonymized Study Files"
    study_path = base_path / study_folder
    
    if not study_path.exists():
        print(f"Error: Study folder not found at {study_path}")
        return
    
    # Initialize data ingester
    ingester = ClinicalDataIngester(base_path)
    
    # Load required data files
    print("\n[1] LOADING DATA FILES")
    print("-" * 40)
    
    # CPID EDC Metrics (required)
    cpid_files = list(study_path.glob("*CPID_EDC_Metrics*.xlsx"))
    if not cpid_files:
        print("Error: No CPID EDC Metrics file found")
        return
    
    cpid_df = ingester.load_cpid_metrics(cpid_files[0])
    print(f"✓ CPID Metrics: {len(cpid_df)} patient records")
    
    # Inactivated Forms (optional but important for manipulation risk)
    inact_files = list(study_path.glob("*Inactivated*.xlsx"))
    inact_df = None
    if inact_files:
        inact_df = ingester.load_inactivated_forms(inact_files[0])
        print(f"✓ Inactivated Forms: {len(inact_df) if inact_df is not None else 0} records")
    else:
        print("⚠ No Inactivated Forms file found (manipulation risk will be limited)")
    
    # Engineer features
    print("\n[2] ENGINEERING FEATURES")
    print("-" * 40)
    
    study_id = study_folder.split("_")[0].replace(" ", "_")
    engineer, feature_matrix = engineer_study_features(
        study_id=study_id,
        cpid_metrics=cpid_df,
        inactivated_forms=inact_df
    )
    
    print(f"✓ Engineered features for {len(feature_matrix)} sites")
    
    # Display results
    print("\n[3] FEATURE SUMMARY BY SITE")
    print("-" * 40)
    
    for site_id, features in engineer.features_by_site.items():
        v = features['velocity_index']
        d = features['data_density']
        m = features['manipulation_risk']
        c = features['composite_risk_score']
        
        print(f"\n■ {site_id}")
        print(f"  ┌─ Velocity Index")
        print(f"  │  Net Velocity: {v.net_velocity:.3f} queries/day")
        print(f"  │  Trend: {v.velocity_trend.value}")
        print(f"  │  Bottleneck: {'YES' if v.is_bottleneck else 'No'}")
        print(f"  │")
        print(f"  ├─ Data Density")
        print(f"  │  Query Density: {d.query_density_percentage:.2f}%")
        print(f"  │  Risk Level: {d.density_risk_level.value}")
        print(f"  │")
        print(f"  ├─ Manipulation Risk")
        print(f"  │  Score: {m.total_risk_score:.1f}/100")
        print(f"  │  Level: {m.risk_level.value}")
        if m.detected_patterns:
            print(f"  │  Patterns: {', '.join(m.detected_patterns[:2])}")
        print(f"  │")
        print(f"  └─ Composite Risk: {c.get('composite_score', 0):.1f} ({c.get('risk_level', 'Unknown')})")
    
    # Feature Matrix for ML
    print("\n[4] FEATURE MATRIX FOR ML MODELS")
    print("-" * 40)
    print(f"Shape: {feature_matrix.shape}")
    print(f"Features: {len(feature_matrix.columns)}")
    print("\nColumns:")
    for col in feature_matrix.columns:
        print(f"  • {col}")
    
    # High-risk sites
    print("\n[5] SITES REQUIRING ATTENTION")
    print("-" * 40)
    
    high_risk_sites = [
        (site_id, f['composite_risk_score'])
        for site_id, f in engineer.features_by_site.items()
        if f['composite_risk_score'].get('composite_score', 0) >= 30
    ]
    
    if high_risk_sites:
        for site_id, risk in sorted(high_risk_sites, key=lambda x: x[1].get('composite_score', 0), reverse=True):
            print(f"  ⚠ {site_id}: Score {risk.get('composite_score', 0):.1f} ({risk.get('risk_level', 'Unknown')})")
    else:
        print("  ✓ No high-risk sites detected")
    
    # Export
    print("\n[6] EXPORT")
    print("-" * 40)
    
    output_path = Path(__file__).parent.parent / "reports" / f"{study_id}_features.json"
    with open(output_path, 'w') as f:
        json.dump(engineer.to_dict(), f, indent=2, default=str)
    print(f"✓ Features exported to: {output_path}")
    
    csv_path = Path(__file__).parent.parent / "reports" / f"{study_id}_feature_matrix.csv"
    feature_matrix.to_csv(csv_path, index=False)
    print(f"✓ Feature matrix exported to: {csv_path}")
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    
    return engineer, feature_matrix


if __name__ == "__main__":
    demo_feature_engineering()
