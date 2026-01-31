
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

"""
KRI Calculator Service
Computes Key Risk Indicators based on derived metrics.
"""

def calculate_site_kris(site_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Site-level KRIs and Risk Scores.
    Input: DataFrame with columns [study_id, site_id, open_query_count, query_aging_index, missing_visit_ratio]
    """
    df = site_features.copy()
    
    # KRI 1: High Query Aging Risk
    # Threshold: Index > 15 days is High Risk
    df['kri_query_aging_score'] = np.where(df['query_aging_index'] > 15, 10, 
                                          np.where(df['query_aging_index'] > 7, 5, 0))
                                          
    # KRI 2: High Open Query Volume
    # Threshold: > 50 open queries
    df['kri_open_query_score'] = np.where(df['open_query_count'] > 50, 10,
                                         np.where(df['open_query_count'] > 20, 5, 0))
    
    # Composite Site Risk Score
    # Simple weighted sum for MVP
    df['site_risk_score'] = (
        (df['kri_query_aging_score'] * 0.6) + 
        (df['kri_open_query_score'] * 0.4)
    )
    
    # Risk Categories
    df['risk_level'] = pd.cut(df['site_risk_score'], 
                              bins=[-1, 3, 7, 100], 
                              labels=['Low', 'Medium', 'High'])
                              
    return df

def calculate_subject_kris(subject_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Subject-level KRIs (Readiness).
    Input: DataFrame with columns [study_id, subject_id, missing_visit_ratio]
    """
    df = subject_features.copy()
    
    # KRI: Missing Visit Ratio
    # > 10% missing is critical
    df['kri_missing_visit_score'] = np.where(df['missing_visit_ratio'] > 0.10, 10,
                                            np.where(df['missing_visit_ratio'] > 0.05, 5, 0))
                                            
    df['readiness_score'] = 100 - (df['kri_missing_visit_score'] * 10)
    
    return df
