"""
Feature Engineering Module for AI/ML Model Training
Transforms raw clinical trial data into high-value features for predictive analytics

This module implements three core engineered features:
1. Operational Velocity Index (Query Resolution Velocity)
2. Normalized Data Density (Queries per Page)
3. Manipulation Risk Score (based on inactivation patterns)

These features feed the Risk-Based Monitoring (RBM) engine and AI models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.data_models import RiskLevel, SiteMetrics
from config.settings import DEFAULT_CLEAN_THRESHOLDS

logger = logging.getLogger(__name__)


class ManipulationRiskLevel(Enum):
    """Classification levels for manipulation risk"""
    CRITICAL = "Critical"      # Score >= 80: Immediate investigation required
    HIGH = "High"              # Score 60-79: Enhanced monitoring
    ELEVATED = "Elevated"      # Score 40-59: Targeted review
    MODERATE = "Moderate"      # Score 20-39: Routine monitoring
    LOW = "Low"                # Score < 20: Standard operations


class VelocityTrend(Enum):
    """Trend classification for velocity metrics"""
    ACCELERATING_POSITIVE = "Accelerating Positive"   # Improving rapidly
    POSITIVE = "Positive"                              # Steady improvement
    STABLE = "Stable"                                  # No significant change
    NEGATIVE = "Negative"                              # Declining
    BOTTLENECK = "Bottleneck"                          # Critical backlog


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering thresholds"""
    # Velocity Index Configuration
    velocity_window_days: int = 7           # Rolling window for velocity calculation
    critical_velocity_threshold: float = -5.0  # Negative = queries accumulating
    positive_velocity_threshold: float = 2.0   # Positive = queries being resolved
    
    # Data Density Configuration
    high_density_threshold: float = 0.10     # 10% - queries per page is concerning
    critical_density_threshold: float = 0.15  # 15% - queries per page is critical
    low_density_threshold: float = 0.02       # 2% - acceptable threshold
    
    # Manipulation Risk Configuration
    primary_endpoint_forms: List[str] = field(default_factory=lambda: [
        'Efficacy', 'Primary', 'Endpoint', 'Tumor', 'Response', 
        'Survival', 'PFS', 'OS', 'ORR', 'DOR', 'Assessment'
    ])
    high_risk_audit_actions: List[str] = field(default_factory=lambda: [
        'Inactivated', 'Deleted', 'Removed', 'Cleared', 'Reset',
        'Unverified', 'Unsigned', 'Unlocked'
    ])
    inactivation_frequency_threshold: int = 5  # Inactivations per month
    
    # Risk score weights
    weight_inactivation_frequency: float = 0.30
    weight_endpoint_data_risk: float = 0.35
    weight_temporal_pattern: float = 0.20
    weight_audit_trail_anomaly: float = 0.15


DEFAULT_FEATURE_CONFIG = FeatureEngineeringConfig()


@dataclass 
class OperationalVelocityIndex:
    """
    Feature 1: Operational Velocity Index
    
    Measures the rate at which queries are being resolved vs. opened.
    V_res = Δ(# Closed Queries) / Δt
    
    A negative velocity indicates bottleneck - queries accumulating faster
    than they are being resolved.
    """
    site_id: str
    study_id: str
    
    # Core velocity metrics
    resolution_velocity: float = 0.0      # Queries closed per day
    accumulation_velocity: float = 0.0    # Queries opened per day
    net_velocity: float = 0.0             # Resolution - Accumulation
    
    # Trend analysis
    velocity_trend: VelocityTrend = VelocityTrend.STABLE
    days_to_clear_backlog: Optional[float] = None
    
    # Historical data
    queries_opened_period: int = 0
    queries_closed_period: int = 0
    current_open_queries: int = 0
    period_days: int = 7
    
    # Risk indicators
    is_bottleneck: bool = False
    bottleneck_severity: float = 0.0
    
    def calculate(
        self, 
        current_open: int,
        current_total: int,
        previous_open: Optional[int] = None,
        previous_total: Optional[int] = None,
        period_days: int = 7
    ):
        """
        Calculate velocity index from query metrics
        
        Args:
            current_open: Current count of open queries
            current_total: Current total query count
            previous_open: Open queries at start of period
            previous_total: Total queries at start of period
            period_days: Number of days in measurement period
        """
        self.current_open_queries = current_open
        self.period_days = period_days
        
        if previous_open is not None and previous_total is not None:
            # Calculate queries opened in period
            self.queries_opened_period = max(0, current_total - previous_total)
            
            # Calculate queries closed in period
            # Closed = Opened - Change in Open
            change_in_open = current_open - previous_open
            self.queries_closed_period = max(0, self.queries_opened_period - change_in_open)
            
            # Calculate velocities (per day)
            self.accumulation_velocity = self.queries_opened_period / period_days
            self.resolution_velocity = self.queries_closed_period / period_days
            self.net_velocity = self.resolution_velocity - self.accumulation_velocity
            
        else:
            # Estimate from current state (less accurate)
            # Assume closed queries = total - open
            closed_queries = current_total - current_open
            # Rough estimate: assume queries were generated over study duration
            self.resolution_velocity = closed_queries / period_days if closed_queries > 0 else 0
            self.accumulation_velocity = current_total / period_days if current_total > 0 else 0
            self.net_velocity = self.resolution_velocity - self.accumulation_velocity
        
        # Determine trend
        self._classify_trend()
        
        # Calculate backlog clearance time
        self._calculate_backlog_clearance()
        
        return self
    
    def _classify_trend(self):
        """Classify the velocity trend"""
        config = DEFAULT_FEATURE_CONFIG
        
        if self.net_velocity <= config.critical_velocity_threshold:
            self.velocity_trend = VelocityTrend.BOTTLENECK
            self.is_bottleneck = True
            self.bottleneck_severity = abs(self.net_velocity / config.critical_velocity_threshold)
        elif self.net_velocity < 0:
            self.velocity_trend = VelocityTrend.NEGATIVE
            self.is_bottleneck = True
            self.bottleneck_severity = abs(self.net_velocity) / abs(config.critical_velocity_threshold)
        elif self.net_velocity < config.positive_velocity_threshold:
            self.velocity_trend = VelocityTrend.STABLE
            self.is_bottleneck = False
            self.bottleneck_severity = 0.0
        elif self.net_velocity < config.positive_velocity_threshold * 2:
            self.velocity_trend = VelocityTrend.POSITIVE
            self.is_bottleneck = False
        else:
            self.velocity_trend = VelocityTrend.ACCELERATING_POSITIVE
            self.is_bottleneck = False
    
    def _calculate_backlog_clearance(self):
        """Calculate estimated days to clear current backlog"""
        if self.net_velocity > 0 and self.current_open_queries > 0:
            self.days_to_clear_backlog = self.current_open_queries / self.net_velocity
        elif self.net_velocity <= 0 and self.current_open_queries > 0:
            # Backlog is growing, estimate is infinite
            self.days_to_clear_backlog = float('inf')
        else:
            self.days_to_clear_backlog = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'study_id': self.study_id,
            'resolution_velocity': round(self.resolution_velocity, 3),
            'accumulation_velocity': round(self.accumulation_velocity, 3),
            'net_velocity': round(self.net_velocity, 3),
            'velocity_trend': self.velocity_trend.value,
            'days_to_clear_backlog': round(self.days_to_clear_backlog, 1) if self.days_to_clear_backlog != float('inf') else 'Infinite',
            'is_bottleneck': self.is_bottleneck,
            'bottleneck_severity': round(self.bottleneck_severity, 2),
            'period_metrics': {
                'queries_opened': self.queries_opened_period,
                'queries_closed': self.queries_closed_period,
                'current_open': self.current_open_queries,
                'period_days': self.period_days
            }
        }


@dataclass
class NormalizedDataDensity:
    """
    Feature 2: Normalized Data Density
    
    D_density = Total Queries / # Pages Entered
    
    This 'Queries per Page' metric isolates site quality from site volume.
    A site with 100 queries on 10,000 pages (1% density) performs better
    than a site with 10 queries on 100 pages (10% density).
    """
    site_id: str
    study_id: str
    
    # Core density metrics
    total_queries: int = 0
    total_pages_entered: int = 0
    query_density: float = 0.0              # Queries per page
    query_density_percentage: float = 0.0   # Density as percentage
    
    # Breakdown by query type
    open_query_density: float = 0.0
    closed_query_density: float = 0.0
    
    # Comparative metrics
    density_percentile: Optional[float] = None  # Compared to other sites
    density_z_score: Optional[float] = None     # Standard deviations from mean
    
    # Risk classification
    density_risk_level: RiskLevel = RiskLevel.LOW
    
    # Additional context
    verification_rate: float = 0.0
    non_conformant_density: float = 0.0
    
    def calculate(
        self,
        total_queries: int,
        pages_entered: int,
        open_queries: int = 0,
        non_conformant_pages: int = 0,
        forms_verified: int = 0
    ):
        """
        Calculate normalized data density
        
        Args:
            total_queries: Total number of queries generated
            pages_entered: Total CRF pages entered
            open_queries: Currently open queries
            non_conformant_pages: Pages with non-conformant data
            forms_verified: Number of forms that have been verified
        """
        self.total_queries = total_queries
        self.total_pages_entered = pages_entered
        
        if pages_entered > 0:
            # Primary density metric
            self.query_density = total_queries / pages_entered
            self.query_density_percentage = self.query_density * 100
            
            # Breakdown by query status
            self.open_query_density = open_queries / pages_entered
            closed_queries = total_queries - open_queries
            self.closed_query_density = closed_queries / pages_entered
            
            # Non-conformant data density
            self.non_conformant_density = non_conformant_pages / pages_entered
            
            # Verification rate
            self.verification_rate = (forms_verified / pages_entered * 100) if pages_entered > 0 else 0
        else:
            self.query_density = 0.0
            self.query_density_percentage = 0.0
            
        # Classify risk level
        self._classify_risk()
        
        return self
    
    def _classify_risk(self):
        """Classify risk level based on density thresholds"""
        config = DEFAULT_FEATURE_CONFIG
        
        if self.query_density >= config.critical_density_threshold:
            self.density_risk_level = RiskLevel.CRITICAL
        elif self.query_density >= config.high_density_threshold:
            self.density_risk_level = RiskLevel.HIGH
        elif self.query_density >= config.low_density_threshold:
            self.density_risk_level = RiskLevel.MEDIUM
        else:
            self.density_risk_level = RiskLevel.LOW
    
    def calculate_comparative_metrics(
        self, 
        all_site_densities: List[float]
    ):
        """
        Calculate comparative metrics against other sites
        
        Args:
            all_site_densities: List of density values from all sites
        """
        if not all_site_densities or len(all_site_densities) < 2:
            return
            
        # Calculate percentile
        sorted_densities = sorted(all_site_densities)
        rank = sorted_densities.index(self.query_density) if self.query_density in sorted_densities else 0
        self.density_percentile = (rank / len(sorted_densities)) * 100
        
        # Calculate z-score
        mean_density = np.mean(all_site_densities)
        std_density = np.std(all_site_densities)
        if std_density > 0:
            self.density_z_score = (self.query_density - mean_density) / std_density
        else:
            self.density_z_score = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'study_id': self.study_id,
            'query_density': round(self.query_density, 4),
            'query_density_percentage': round(self.query_density_percentage, 2),
            'density_risk_level': self.density_risk_level.value,
            'metrics': {
                'total_queries': self.total_queries,
                'total_pages_entered': self.total_pages_entered,
                'open_query_density': round(self.open_query_density, 4),
                'closed_query_density': round(self.closed_query_density, 4),
                'non_conformant_density': round(self.non_conformant_density, 4),
                'verification_rate': round(self.verification_rate, 2)
            },
            'comparative': {
                'density_percentile': round(self.density_percentile, 1) if self.density_percentile else None,
                'density_z_score': round(self.density_z_score, 2) if self.density_z_score else None
            }
        }


@dataclass
class ManipulationRiskScore:
    """
    Feature 3: Manipulation Risk Score
    
    Generated from Inactivated forms/folders and Audit Actions.
    Frequent inactivation of forms, especially those containing primary 
    endpoint data, is a non-random pattern that may indicate:
    - Severe site training issues (entering data in wrong place)
    - Potential misconduct or 'gaming' of the system
    - Attempts to hide protocol deviations
    
    This score feeds directly into the Risk-Based Monitoring (RBM) engine.
    """
    site_id: str
    study_id: str
    
    # Core risk score (0-100)
    total_risk_score: float = 0.0
    risk_level: ManipulationRiskLevel = ManipulationRiskLevel.LOW
    
    # Component scores (0-100 each)
    inactivation_frequency_score: float = 0.0    # Based on count/frequency
    endpoint_data_risk_score: float = 0.0        # Based on endpoint form impact
    temporal_pattern_score: float = 0.0          # Based on timing patterns
    audit_trail_anomaly_score: float = 0.0       # Based on audit action types
    
    # Raw metrics
    total_inactivations: int = 0
    endpoint_form_inactivations: int = 0
    inactivation_rate_per_month: float = 0.0
    
    # Pattern analysis
    detected_patterns: List[str] = field(default_factory=list)
    high_risk_forms: List[str] = field(default_factory=list)
    high_risk_actions: List[str] = field(default_factory=list)
    
    # Temporal patterns
    inactivations_by_user: Dict[str, int] = field(default_factory=dict)
    inactivations_by_period: Dict[str, int] = field(default_factory=dict)
    
    # Recommended actions
    recommended_actions: List[str] = field(default_factory=list)
    requires_investigation: bool = False
    
    def calculate(
        self,
        inactivated_forms_df: Optional[pd.DataFrame] = None,
        cpid_metrics_df: Optional[pd.DataFrame] = None,
        study_duration_days: int = 365,
        config: FeatureEngineeringConfig = None
    ):
        """
        Calculate manipulation risk score from inactivation data
        
        Args:
            inactivated_forms_df: DataFrame with inactivated forms/folders data
            cpid_metrics_df: DataFrame with CPID metrics (for context)
            study_duration_days: Duration of study for rate calculation
            config: Feature engineering configuration
        """
        config = config or DEFAULT_FEATURE_CONFIG
        
        if inactivated_forms_df is None or len(inactivated_forms_df) == 0:
            self.total_risk_score = 0.0
            self.risk_level = ManipulationRiskLevel.LOW
            return self
        
        # Filter to this site if site_id column exists
        site_df = inactivated_forms_df
        if 'site_id' in inactivated_forms_df.columns:
            site_df = inactivated_forms_df[
                inactivated_forms_df['site_id'].astype(str) == str(self.site_id)
            ]
        
        if len(site_df) == 0:
            self.total_risk_score = 0.0
            self.risk_level = ManipulationRiskLevel.LOW
            return self
        
        # Calculate component scores
        self._calculate_frequency_score(site_df, study_duration_days, config)
        self._calculate_endpoint_risk_score(site_df, config)
        self._calculate_temporal_pattern_score(site_df, config)
        self._calculate_audit_anomaly_score(site_df, config)
        
        # Calculate weighted total score
        self.total_risk_score = (
            config.weight_inactivation_frequency * self.inactivation_frequency_score +
            config.weight_endpoint_data_risk * self.endpoint_data_risk_score +
            config.weight_temporal_pattern * self.temporal_pattern_score +
            config.weight_audit_trail_anomaly * self.audit_trail_anomaly_score
        )
        
        # Classify risk level
        self._classify_risk()
        
        # Generate recommended actions
        self._generate_recommendations()
        
        return self
    
    def _calculate_frequency_score(
        self, 
        df: pd.DataFrame, 
        study_duration_days: int,
        config: FeatureEngineeringConfig
    ):
        """Calculate score based on inactivation frequency"""
        self.total_inactivations = len(df)
        
        # Calculate monthly rate
        months = max(study_duration_days / 30, 1)
        self.inactivation_rate_per_month = self.total_inactivations / months
        
        # Score based on rate compared to threshold
        threshold = config.inactivation_frequency_threshold
        if self.inactivation_rate_per_month >= threshold * 3:
            self.inactivation_frequency_score = 100.0
            self.detected_patterns.append("Extreme inactivation frequency")
        elif self.inactivation_rate_per_month >= threshold * 2:
            self.inactivation_frequency_score = 75.0
            self.detected_patterns.append("High inactivation frequency")
        elif self.inactivation_rate_per_month >= threshold:
            self.inactivation_frequency_score = 50.0
            self.detected_patterns.append("Elevated inactivation frequency")
        else:
            # Linear scale below threshold
            self.inactivation_frequency_score = (
                self.inactivation_rate_per_month / threshold
            ) * 50
    
    def _calculate_endpoint_risk_score(
        self, 
        df: pd.DataFrame,
        config: FeatureEngineeringConfig
    ):
        """Calculate score based on impact to primary endpoint data"""
        # Check form names for endpoint-related terms
        form_col = None
        for col in ['form_name', 'Form Name', 'FormName', 'form']:
            if col in df.columns:
                form_col = col
                break
        
        if form_col is None:
            self.endpoint_data_risk_score = 0.0
            return
        
        endpoint_count = 0
        for idx, row in df.iterrows():
            form_name = str(row.get(form_col, '')).lower()
            for keyword in config.primary_endpoint_forms:
                if keyword.lower() in form_name:
                    endpoint_count += 1
                    if form_name not in self.high_risk_forms:
                        self.high_risk_forms.append(str(row.get(form_col, '')))
                    break
        
        self.endpoint_form_inactivations = endpoint_count
        
        if self.total_inactivations > 0:
            endpoint_ratio = endpoint_count / self.total_inactivations
            # Score increases non-linearly with endpoint ratio
            self.endpoint_data_risk_score = min(100, endpoint_ratio * 200)
            
            if endpoint_ratio > 0.3:
                self.detected_patterns.append(
                    f"High ratio of endpoint form inactivations ({endpoint_ratio:.1%})"
                )
    
    def _calculate_temporal_pattern_score(
        self, 
        df: pd.DataFrame,
        config: FeatureEngineeringConfig
    ):
        """Analyze temporal patterns in inactivations"""
        date_col = None
        for col in ['inactivation_date', 'date', 'Action Date', 'Date']:
            if col in df.columns:
                date_col = col
                break
        
        user_col = None
        for col in ['user', 'User', 'Modified By', 'User Name']:
            if col in df.columns:
                user_col = col
                break
        
        score = 0.0
        
        # Check for user concentration
        if user_col:
            user_counts = df[user_col].value_counts()
            self.inactivations_by_user = user_counts.to_dict()
            
            if len(user_counts) > 0:
                top_user_ratio = user_counts.iloc[0] / len(df)
                if top_user_ratio > 0.8:
                    score += 40
                    self.detected_patterns.append(
                        f"Single user responsible for {top_user_ratio:.0%} of inactivations"
                    )
                elif top_user_ratio > 0.6:
                    score += 25
        
        # Check for temporal clustering
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce')
                dates = dates.dropna()
                
                if len(dates) > 3:
                    # Check for clustering (many events in short period)
                    date_diffs = dates.sort_values().diff().dt.days
                    avg_gap = date_diffs.mean()
                    
                    if avg_gap < 2:  # Less than 2 days average gap
                        score += 30
                        self.detected_patterns.append(
                            "Temporal clustering of inactivations detected"
                        )
                    
                    # Group by month/week
                    self.inactivations_by_period = dates.dt.to_period('M').value_counts().to_dict()
            except Exception as e:
                logger.warning(f"Error analyzing temporal patterns: {e}")
        
        self.temporal_pattern_score = min(100, score)
    
    def _calculate_audit_anomaly_score(
        self, 
        df: pd.DataFrame,
        config: FeatureEngineeringConfig
    ):
        """Analyze audit actions for anomalies"""
        action_col = None
        for col in ['audit_action', 'Audit Action', 'Action', 'Reason']:
            if col in df.columns:
                action_col = col
                break
        
        if action_col is None:
            self.audit_trail_anomaly_score = 0.0
            return
        
        high_risk_count = 0
        for idx, row in df.iterrows():
            action = str(row.get(action_col, '')).lower()
            for keyword in config.high_risk_audit_actions:
                if keyword.lower() in action:
                    high_risk_count += 1
                    if action not in [a.lower() for a in self.high_risk_actions]:
                        self.high_risk_actions.append(str(row.get(action_col, '')))
                    break
        
        if self.total_inactivations > 0:
            risk_ratio = high_risk_count / self.total_inactivations
            self.audit_trail_anomaly_score = min(100, risk_ratio * 150)
            
            if risk_ratio > 0.5:
                self.detected_patterns.append(
                    f"High proportion of risky audit actions ({risk_ratio:.0%})"
                )
    
    def _classify_risk(self):
        """Classify overall risk level"""
        if self.total_risk_score >= 80:
            self.risk_level = ManipulationRiskLevel.CRITICAL
            self.requires_investigation = True
        elif self.total_risk_score >= 60:
            self.risk_level = ManipulationRiskLevel.HIGH
            self.requires_investigation = True
        elif self.total_risk_score >= 40:
            self.risk_level = ManipulationRiskLevel.ELEVATED
        elif self.total_risk_score >= 20:
            self.risk_level = ManipulationRiskLevel.MODERATE
        else:
            self.risk_level = ManipulationRiskLevel.LOW
    
    def _generate_recommendations(self):
        """Generate recommended actions based on risk analysis"""
        self.recommended_actions = []
        
        if self.risk_level == ManipulationRiskLevel.CRITICAL:
            self.recommended_actions.extend([
                "Immediate site audit required",
                "Review all inactivated primary endpoint data",
                "Contact Quality Assurance for investigation",
                "Freeze site for new enrollments pending review"
            ])
        elif self.risk_level == ManipulationRiskLevel.HIGH:
            self.recommended_actions.extend([
                "Schedule targeted monitoring visit",
                "Review user access patterns",
                "Enhanced source document verification",
                "Training assessment for site staff"
            ])
        elif self.risk_level == ManipulationRiskLevel.ELEVATED:
            self.recommended_actions.extend([
                "Increase remote monitoring frequency",
                "Focus on endpoint data verification",
                "Review training compliance records"
            ])
        elif self.risk_level == ManipulationRiskLevel.MODERATE:
            self.recommended_actions.extend([
                "Continue routine monitoring",
                "Note patterns for trend analysis"
            ])
        
        # Add specific recommendations based on patterns
        if "endpoint form inactivations" in ' '.join(self.detected_patterns).lower():
            self.recommended_actions.append(
                "Priority review of all endpoint form data entries"
            )
        
        if "single user" in ' '.join(self.detected_patterns).lower():
            self.recommended_actions.append(
                "Review user's access permissions and training status"
            )
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'study_id': self.study_id,
            'total_risk_score': round(self.total_risk_score, 1),
            'risk_level': self.risk_level.value,
            'requires_investigation': self.requires_investigation,
            'component_scores': {
                'inactivation_frequency': round(self.inactivation_frequency_score, 1),
                'endpoint_data_risk': round(self.endpoint_data_risk_score, 1),
                'temporal_pattern': round(self.temporal_pattern_score, 1),
                'audit_trail_anomaly': round(self.audit_trail_anomaly_score, 1)
            },
            'metrics': {
                'total_inactivations': self.total_inactivations,
                'endpoint_form_inactivations': self.endpoint_form_inactivations,
                'inactivation_rate_per_month': round(self.inactivation_rate_per_month, 2)
            },
            'detected_patterns': self.detected_patterns,
            'high_risk_forms': self.high_risk_forms[:5],  # Limit output
            'high_risk_actions': self.high_risk_actions[:5],
            'recommended_actions': self.recommended_actions
        }


class SiteFeatureEngineer:
    """
    Main feature engineering class that computes all three features
    for sites within a study
    """
    
    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or DEFAULT_FEATURE_CONFIG
        self.features_by_site: Dict[str, Dict] = {}
    
    def engineer_features(
        self,
        study_id: str,
        cpid_metrics: pd.DataFrame,
        inactivated_forms: Optional[pd.DataFrame] = None,
        historical_metrics: Optional[Dict[str, pd.DataFrame]] = None,
        study_duration_days: int = 365
    ) -> Dict[str, Dict]:
        """
        Engineer all three features for all sites in a study
        
        Args:
            study_id: Study identifier
            cpid_metrics: CPID EDC Metrics DataFrame
            inactivated_forms: Inactivated Forms DataFrame
            historical_metrics: Historical data for velocity calculation
            study_duration_days: Duration of study
            
        Returns:
            Dictionary of engineered features keyed by site_id
        """
        # Aggregate metrics by site
        site_aggregates = self._aggregate_by_site(cpid_metrics)
        
        # Calculate features for each site
        features_by_site = {}
        all_densities = []
        
        for site_id, site_data in site_aggregates.items():
            # Feature 1: Operational Velocity Index
            velocity = OperationalVelocityIndex(site_id=site_id, study_id=study_id)
            
            # Use historical data if available
            prev_open = None
            prev_total = None
            if historical_metrics and site_id in historical_metrics:
                hist = historical_metrics[site_id]
                prev_open = hist.get('open_queries', None)
                prev_total = hist.get('total_queries', None)
            
            velocity.calculate(
                current_open=site_data['open_queries'],
                current_total=site_data['total_queries'],
                previous_open=prev_open,
                previous_total=prev_total,
                period_days=self.config.velocity_window_days
            )
            
            # Feature 2: Normalized Data Density
            density = NormalizedDataDensity(site_id=site_id, study_id=study_id)
            density.calculate(
                total_queries=site_data['total_queries'],
                pages_entered=site_data['pages_entered'],
                open_queries=site_data['open_queries'],
                non_conformant_pages=site_data.get('non_conformant', 0),
                forms_verified=site_data.get('forms_verified', 0)
            )
            all_densities.append(density.query_density)
            
            # Feature 3: Manipulation Risk Score
            manipulation = ManipulationRiskScore(site_id=site_id, study_id=study_id)
            manipulation.calculate(
                inactivated_forms_df=inactivated_forms,
                cpid_metrics_df=cpid_metrics,
                study_duration_days=study_duration_days,
                config=self.config
            )
            
            features_by_site[site_id] = {
                'site_id': site_id,
                'study_id': study_id,
                'velocity_index': velocity,
                'data_density': density,
                'manipulation_risk': manipulation,
                'composite_risk_score': self._calculate_composite_score(
                    velocity, density, manipulation
                )
            }
        
        # Update comparative metrics for density
        for site_id, features in features_by_site.items():
            features['data_density'].calculate_comparative_metrics(all_densities)
        
        self.features_by_site = features_by_site
        return features_by_site
    
    def _aggregate_by_site(self, cpid_metrics: pd.DataFrame) -> Dict[str, Dict]:
        """Aggregate CPID metrics by site"""
        site_col = 'site_id' if 'site_id' in cpid_metrics.columns else 'Site ID'
        
        if site_col not in cpid_metrics.columns:
            logger.warning("No site_id column found in CPID metrics")
            return {}
        
        aggregates = {}
        
        for site_id in cpid_metrics[site_col].unique():
            if pd.isna(site_id):
                continue
                
            site_df = cpid_metrics[cpid_metrics[site_col] == site_id]
            
            def safe_sum(col):
                if col in site_df.columns:
                    return site_df[col].sum()
                return 0
            
            aggregates[str(site_id)] = {
                'open_queries': int(safe_sum('open_queries')),
                'total_queries': int(safe_sum('total_queries')),
                'pages_entered': int(safe_sum('pages_entered')),
                'missing_pages': int(safe_sum('missing_pages')),
                'missing_visits': int(safe_sum('missing_visits')),
                'non_conformant': int(safe_sum('non_conformant')),
                'forms_verified': int(safe_sum('forms_verified')),
                'patient_count': len(site_df)
            }
        
        return aggregates
    
    def _calculate_composite_score(
        self,
        velocity: OperationalVelocityIndex,
        density: NormalizedDataDensity,
        manipulation: ManipulationRiskScore
    ) -> Dict:
        """
        Calculate a composite risk score from all three features
        
        This score represents overall site health and feeds the RBM engine.
        """
        # Normalize scores to 0-100 scale
        velocity_score = 0
        if velocity.is_bottleneck:
            velocity_score = min(100, velocity.bottleneck_severity * 50)
        
        density_score = 0
        if density.density_risk_level == RiskLevel.CRITICAL:
            density_score = 100
        elif density.density_risk_level == RiskLevel.HIGH:
            density_score = 75
        elif density.density_risk_level == RiskLevel.MEDIUM:
            density_score = 50
        else:
            density_score = density.query_density_percentage * 5  # Scale up
        
        manipulation_score = manipulation.total_risk_score
        
        # Weighted composite (manipulation has highest weight as it's most critical)
        composite = (
            0.25 * velocity_score +
            0.25 * density_score +
            0.50 * manipulation_score
        )
        
        # Determine composite risk level
        if composite >= 70:
            risk_level = "Critical"
        elif composite >= 50:
            risk_level = "High"
        elif composite >= 30:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'composite_score': round(composite, 1),
            'risk_level': risk_level,
            'component_contributions': {
                'velocity': round(0.25 * velocity_score, 1),
                'density': round(0.25 * density_score, 1),
                'manipulation': round(0.50 * manipulation_score, 1)
            },
            'requires_intervention': composite >= 50
        }
    
    def get_feature_matrix(self) -> pd.DataFrame:
        """
        Generate a feature matrix suitable for ML model training
        
        Returns:
            DataFrame with engineered features for all sites
        """
        rows = []
        
        for site_id, features in self.features_by_site.items():
            velocity = features['velocity_index']
            density = features['data_density']
            manipulation = features['manipulation_risk']
            composite = features['composite_risk_score']
            
            row = {
                'site_id': site_id,
                'study_id': features['study_id'],
                
                # Velocity features
                'resolution_velocity': velocity.resolution_velocity,
                'accumulation_velocity': velocity.accumulation_velocity,
                'net_velocity': velocity.net_velocity,
                'is_bottleneck': int(velocity.is_bottleneck),
                'bottleneck_severity': velocity.bottleneck_severity,
                'days_to_clear_backlog': velocity.days_to_clear_backlog if velocity.days_to_clear_backlog != float('inf') else -1,
                
                # Density features
                'query_density': density.query_density,
                'query_density_pct': density.query_density_percentage,
                'open_query_density': density.open_query_density,
                'non_conformant_density': density.non_conformant_density,
                'verification_rate': density.verification_rate,
                'density_z_score': density.density_z_score or 0,
                
                # Manipulation features
                'manipulation_risk_score': manipulation.total_risk_score,
                'inactivation_frequency_score': manipulation.inactivation_frequency_score,
                'endpoint_data_risk_score': manipulation.endpoint_data_risk_score,
                'temporal_pattern_score': manipulation.temporal_pattern_score,
                'audit_anomaly_score': manipulation.audit_trail_anomaly_score,
                'total_inactivations': manipulation.total_inactivations,
                'inactivation_rate_monthly': manipulation.inactivation_rate_per_month,
                
                # Composite scores
                'composite_risk_score': composite['composite_score'],
                'requires_intervention': int(composite['requires_intervention'])
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def to_dict(self) -> Dict:
        """Export all features as dictionary"""
        result = {}
        for site_id, features in self.features_by_site.items():
            result[site_id] = {
                'velocity_index': features['velocity_index'].to_dict(),
                'data_density': features['data_density'].to_dict(),
                'manipulation_risk': features['manipulation_risk'].to_dict(),
                'composite_risk': features['composite_risk_score']
            }
        return result


def engineer_study_features(
    study_id: str,
    cpid_metrics: pd.DataFrame,
    inactivated_forms: Optional[pd.DataFrame] = None,
    config: FeatureEngineeringConfig = None
) -> Tuple[SiteFeatureEngineer, pd.DataFrame]:
    """
    Convenience function to engineer features for a study
    
    Args:
        study_id: Study identifier
        cpid_metrics: CPID metrics DataFrame
        inactivated_forms: Inactivated forms DataFrame
        config: Feature engineering configuration
        
    Returns:
        Tuple of (SiteFeatureEngineer instance, Feature matrix DataFrame)
    """
    engineer = SiteFeatureEngineer(config=config)
    engineer.engineer_features(
        study_id=study_id,
        cpid_metrics=cpid_metrics,
        inactivated_forms=inactivated_forms
    )
    
    feature_matrix = engineer.get_feature_matrix()
    
    return engineer, feature_matrix
