"""
RBM Report Generator - Risk-Based Monitoring Report Automation
===============================================================

The system automates the administrative burden of the CRA by analyzing
CPID_EDC_Metrics and drafting CRA Visit Letters and Monitoring Reports.

Analysis Areas:
- # PDs Confirmed (Protocol Deviations)
- CRFs overdue for signs
- Broken Signatures
- Missing Pages
- Query aging
- Data Quality Index (DQI)

Draft Output Example:
"Dear Investigator, During the upcoming monitoring visit, please prioritize:
1) There are 5 confirmed Protocol Deviations related to inclusion criteria 
   that require CAPA documentation.
2) Note that 3 CRFs are overdue for signature beyond 90 days, which is a 
   critical compliance risk.
3) Please address the 'Broken Signatures' on the Informed Consent forms 
   for Subject 002."

Impact:
- Moves CRA from "detective" (finding issues) to "solver" (addressing them)
- Reduces monitoring report preparation from hours to minutes
- Standardizes report quality across CRAs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import re

logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    """Risk categories for RBM"""
    CRITICAL = "critical"          # Immediate regulatory/safety risk
    HIGH = "high"                  # Significant compliance risk
    MODERATE = "moderate"          # Notable issues requiring attention
    LOW = "low"                    # Minor issues for routine follow-up
    INFORMATIONAL = "informational"  # For awareness only


class IssueType(Enum):
    """Types of issues identified in RBM"""
    PROTOCOL_DEVIATION = "Protocol Deviation"
    OVERDUE_SIGNATURE = "Overdue Signature"
    BROKEN_SIGNATURE = "Broken Signature"
    MISSING_DATA = "Missing Data"
    QUERY_AGING = "Query Aging"
    CONSENT_ISSUE = "Consent Issue"
    SAFETY_REPORTING = "Safety Reporting"
    DATA_QUALITY = "Data Quality"
    VISIT_COMPLIANCE = "Visit Compliance"
    SOURCE_VERIFICATION = "Source Verification"


@dataclass
class ComplianceIssue:
    """A compliance issue identified for a site"""
    issue_type: IssueType
    risk_category: RiskCategory
    description: str
    subject_ids: List[str] = field(default_factory=list)
    count: int = 0
    metric_value: Optional[float] = None
    threshold_breached: Optional[str] = None
    recommended_action: str = ""
    capa_required: bool = False
    regulatory_impact: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'issue_type': self.issue_type.value,
            'risk_category': self.risk_category.value,
            'description': self.description,
            'subject_ids': self.subject_ids,
            'count': self.count,
            'metric_value': self.metric_value,
            'threshold_breached': self.threshold_breached,
            'recommended_action': self.recommended_action,
            'capa_required': self.capa_required,
            'regulatory_impact': self.regulatory_impact
        }
    
    def get_priority_score(self) -> int:
        """Calculate priority score for sorting"""
        risk_scores = {
            RiskCategory.CRITICAL: 100,
            RiskCategory.HIGH: 75,
            RiskCategory.MODERATE: 50,
            RiskCategory.LOW: 25,
            RiskCategory.INFORMATIONAL: 10
        }
        base_score = risk_scores.get(self.risk_category, 10)
        
        # Boost for CAPA requirement
        if self.capa_required:
            base_score += 20
        
        # Boost for regulatory impact
        if self.regulatory_impact:
            base_score += 30
        
        return base_score


@dataclass
class SiteRiskProfile:
    """Risk profile for a clinical site"""
    site_id: str
    site_name: Optional[str] = None
    country: Optional[str] = None
    investigator: Optional[str] = None
    total_subjects: int = 0
    active_subjects: int = 0
    
    # Metrics
    protocol_deviations: int = 0
    crfs_overdue_signature: int = 0
    broken_signatures: int = 0
    missing_pages: int = 0
    open_queries: int = 0
    queries_over_14_days: int = 0
    queries_over_30_days: int = 0
    dqi_score: Optional[float] = None
    ssm_score: Optional[float] = None
    
    # Risk classification
    overall_risk: RiskCategory = RiskCategory.LOW
    issues: List[ComplianceIssue] = field(default_factory=list)
    
    # Visit info
    last_monitoring_visit: Optional[datetime] = None
    next_scheduled_visit: Optional[datetime] = None
    days_since_last_visit: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'site_name': self.site_name,
            'country': self.country,
            'investigator': self.investigator,
            'total_subjects': self.total_subjects,
            'active_subjects': self.active_subjects,
            'protocol_deviations': self.protocol_deviations,
            'crfs_overdue_signature': self.crfs_overdue_signature,
            'broken_signatures': self.broken_signatures,
            'missing_pages': self.missing_pages,
            'open_queries': self.open_queries,
            'queries_over_14_days': self.queries_over_14_days,
            'queries_over_30_days': self.queries_over_30_days,
            'dqi_score': self.dqi_score,
            'ssm_score': self.ssm_score,
            'overall_risk': self.overall_risk.value,
            'issues': [i.to_dict() for i in self.issues],
            'days_since_last_visit': self.days_since_last_visit
        }


@dataclass
class CRAVisitLetter:
    """CRA Visit Letter / Pre-Visit Summary"""
    site_id: str
    site_name: Optional[str] = None
    investigator_name: Optional[str] = None
    visit_date: Optional[datetime] = None
    generated_at: datetime = field(default_factory=datetime.now)
    
    # Content
    greeting: str = ""
    introduction: str = ""
    priority_items: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    subject_specific_items: Dict[str, List[str]] = field(default_factory=dict)
    closing: str = ""
    
    # Metadata
    risk_profile: Optional[SiteRiskProfile] = None
    total_issues: int = 0
    critical_issues: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'site_name': self.site_name,
            'investigator_name': self.investigator_name,
            'visit_date': self.visit_date.isoformat() if self.visit_date else None,
            'generated_at': self.generated_at.isoformat(),
            'priority_items': self.priority_items,
            'action_items': self.action_items,
            'subject_specific_items': self.subject_specific_items,
            'total_issues': self.total_issues,
            'critical_issues': self.critical_issues
        }
    
    def to_letter_format(self) -> str:
        """Generate formatted letter"""
        lines = [
            f"**Site:** {self.site_id}" + (f" - {self.site_name}" if self.site_name else ""),
            f"**Date:** {self.generated_at.strftime('%B %d, %Y')}",
            "",
            "---",
            "",
            self.greeting,
            "",
            self.introduction,
            ""
        ]
        
        if self.priority_items:
            lines.append("**Priority Items for This Visit:**")
            lines.append("")
            for i, item in enumerate(self.priority_items, 1):
                lines.append(f"{i}) {item}")
            lines.append("")
        
        if self.subject_specific_items:
            lines.append("**Subject-Specific Items:**")
            lines.append("")
            for subject, items in self.subject_specific_items.items():
                lines.append(f"- **{subject}:**")
                for item in items:
                    lines.append(f"  - {item}")
            lines.append("")
        
        if self.action_items:
            lines.append("**Additional Action Items:**")
            lines.append("")
            for item in self.action_items:
                lines.append(f"- {item}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append(self.closing)
        
        return "\n".join(lines)


@dataclass
class MonitoringReport:
    """Full Monitoring Visit Report"""
    report_id: str
    site_id: str
    visit_date: datetime
    generated_at: datetime = field(default_factory=datetime.now)
    
    # Site information
    site_profile: Optional[SiteRiskProfile] = None
    
    # Report sections
    executive_summary: str = ""
    compliance_findings: List[ComplianceIssue] = field(default_factory=list)
    subject_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    follow_up_items: List[str] = field(default_factory=list)
    
    # Metrics summary
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Risk assessment
    overall_risk_rating: RiskCategory = RiskCategory.LOW
    risk_trend: str = "stable"  # improving, stable, deteriorating
    
    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'site_id': self.site_id,
            'visit_date': self.visit_date.isoformat(),
            'generated_at': self.generated_at.isoformat(),
            'executive_summary': self.executive_summary,
            'compliance_findings': [f.to_dict() for f in self.compliance_findings],
            'recommendations': self.recommendations,
            'follow_up_items': self.follow_up_items,
            'metrics_summary': self.metrics_summary,
            'overall_risk_rating': self.overall_risk_rating.value,
            'risk_trend': self.risk_trend
        }
    
    def to_markdown(self) -> str:
        """Generate markdown-formatted report"""
        risk_icons = {
            RiskCategory.CRITICAL: "🔴",
            RiskCategory.HIGH: "🟠",
            RiskCategory.MODERATE: "🟡",
            RiskCategory.LOW: "🟢",
            RiskCategory.INFORMATIONAL: "ℹ️"
        }
        
        lines = [
            f"# Monitoring Visit Report",
            f"**Report ID:** {self.report_id}",
            f"**Site:** {self.site_id}",
            f"**Visit Date:** {self.visit_date.strftime('%B %d, %Y')}",
            f"**Risk Rating:** {risk_icons[self.overall_risk_rating]} {self.overall_risk_rating.value.upper()} ({self.risk_trend})",
            "",
            "---",
            "",
            "## Executive Summary",
            self.executive_summary,
            ""
        ]
        
        if self.compliance_findings:
            lines.append("## Compliance Findings")
            lines.append("")
            
            # Group by risk category
            for risk in [RiskCategory.CRITICAL, RiskCategory.HIGH, RiskCategory.MODERATE, RiskCategory.LOW]:
                findings = [f for f in self.compliance_findings if f.risk_category == risk]
                if findings:
                    lines.append(f"### {risk_icons[risk]} {risk.value.title()} Risk Items")
                    for finding in findings:
                        lines.append(f"- **{finding.issue_type.value}**: {finding.description}")
                        if finding.recommended_action:
                            lines.append(f"  - *Action:* {finding.recommended_action}")
                    lines.append("")
        
        if self.recommendations:
            lines.append("## Recommendations")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        if self.follow_up_items:
            lines.append("## Follow-Up Items")
            for item in self.follow_up_items:
                lines.append(f"- [ ] {item}")
            lines.append("")
        
        if self.metrics_summary:
            lines.append("## Metrics Summary")
            lines.append("")
            lines.append("| Metric | Value | Status |")
            lines.append("|--------|-------|--------|")
            for metric, value in self.metrics_summary.items():
                if isinstance(value, dict):
                    val = value.get('value', 'N/A')
                    status = value.get('status', '✓')
                else:
                    val = value
                    status = '✓'
                lines.append(f"| {metric} | {val} | {status} |")
            lines.append("")
        
        lines.append("---")
        lines.append(f"*Report generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(lines)


class RBMReportGenerator:
    """
    Risk-Based Monitoring Report Generator
    
    Automates the generation of CRA Visit Letters and Monitoring Reports
    by analyzing CPID_EDC_Metrics and related data sources.
    """
    
    # Thresholds for risk classification
    THRESHOLDS = {
        'protocol_deviations': {'warning': 3, 'critical': 5},
        'crfs_overdue_90_days': {'warning': 1, 'critical': 3},
        'broken_signatures': {'warning': 1, 'critical': 2},
        'missing_pages': {'warning': 5, 'critical': 10},
        'queries_over_14_days': {'warning': 5, 'critical': 10},
        'queries_over_30_days': {'warning': 2, 'critical': 5},
        'dqi_score': {'warning': 0.7, 'critical': 0.5},  # Below threshold is bad
        'ssm_score': {'warning': 70, 'critical': 50}     # Below threshold is bad
    }
    
    # Column mappings
    COLUMN_MAPPINGS = {
        'site': ['Site ID', 'Site', 'SiteID'],
        'subject': ['Subject ID', 'Subject', 'SubjectID'],
        'country': ['Country', 'CountryCode'],
        'pds': ['# PDs Confirmed', 'Protocol Deviations', 'PD Count', 'PDs'],
        'overdue_signs': ['CRFs overdue for signs', 'Overdue Signatures', 'CRFs overdue for signs beyond 90 days'],
        'broken_signs': ['Broken Signatures', '# Broken Signatures'],
        'missing_pages': ['# Missing Pages', 'Missing Pages'],
        'open_queries': ['# Open Queries', 'Open Queries'],
        'dqi': ['DQI', 'Data Quality Index'],
        'ssm': ['SSM', 'Site Status Metric']
    }
    
    def __init__(self):
        """Initialize the RBM Report Generator"""
        self.data_sources: Dict[str, pd.DataFrame] = {}
        self._letter_templates = self._build_letter_templates()
        logger.info("RBMReportGenerator initialized")
    
    def _build_letter_templates(self) -> Dict[str, str]:
        """Build letter templates"""
        return {
            'greeting': "Dear Investigator,",
            'introduction': "During the upcoming monitoring visit, please prioritize the following items that have been identified for review:",
            'introduction_urgent': "During the upcoming monitoring visit, there are **critical compliance items** that require immediate attention:",
            'closing': "Please ensure all relevant documentation is available for review. We look forward to a productive visit.\n\nBest regards,\nClinical Research Associate",
            'closing_urgent': "These items require **immediate attention** prior to the monitoring visit. Please contact the study team if you have any questions.\n\nBest regards,\nClinical Research Associate"
        }
    
    def load_data(self, data_sources: Dict[str, pd.DataFrame]):
        """Load data sources for report generation"""
        self.data_sources = data_sources
        logger.info(f"Loaded {len(data_sources)} data sources for RBM reporting")
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find a column from a list of candidates"""
        for col in candidates:
            if col in df.columns:
                return col
        
        # Try case-insensitive match
        lower_cols = {c.lower(): c for c in df.columns}
        for col in candidates:
            if col.lower() in lower_cols:
                return lower_cols[col.lower()]
        
        # Try partial match
        for col in candidates:
            for df_col in df.columns:
                if col.lower() in df_col.lower():
                    return df_col
        
        return None
    
    def analyze_site_risk(self, site_id: str) -> SiteRiskProfile:
        """
        Analyze risk profile for a specific site
        
        Args:
            site_id: The site identifier
            
        Returns:
            SiteRiskProfile with comprehensive risk assessment
        """
        cpid_df = self.data_sources.get('cpid')
        if cpid_df is None or (isinstance(cpid_df, pd.DataFrame) and cpid_df.empty):
            cpid_df = self.data_sources.get('CPID_EDC_Metrics')
        
        profile = SiteRiskProfile(site_id=site_id)
        
        if cpid_df is None or (isinstance(cpid_df, pd.DataFrame) and cpid_df.empty):
            return profile
        
        # Find site column
        site_col = self._find_column(cpid_df, self.COLUMN_MAPPINGS['site'])
        if not site_col:
            return profile
        
        # Extract site number for matching
        site_num = re.search(r'\d+', str(site_id))
        site_num = site_num.group() if site_num else site_id
        
        # Filter for site
        site_data = cpid_df[cpid_df[site_col].astype(str).str.contains(str(site_num), na=False)]
        
        if len(site_data) == 0:
            return profile
        
        # Count subjects
        subject_col = self._find_column(cpid_df, self.COLUMN_MAPPINGS['subject'])
        if subject_col:
            profile.total_subjects = site_data[subject_col].nunique()
        
        # Get country
        country_col = self._find_column(cpid_df, self.COLUMN_MAPPINGS['country'])
        if country_col:
            profile.country = str(site_data[country_col].iloc[0])
        
        # Extract metrics (aggregate across subjects)
        metrics_to_extract = {
            'pds': 'protocol_deviations',
            'overdue_signs': 'crfs_overdue_signature',
            'broken_signs': 'broken_signatures',
            'missing_pages': 'missing_pages',
            'open_queries': 'open_queries'
        }
        
        for mapping_key, profile_attr in metrics_to_extract.items():
            col = self._find_column(cpid_df, self.COLUMN_MAPPINGS[mapping_key])
            if col:
                try:
                    value = pd.to_numeric(site_data[col], errors='coerce').sum()
                    setattr(profile, profile_attr, int(value) if pd.notna(value) else 0)
                except:
                    pass
        
        # Get DQI and SSM (average across site)
        dqi_col = self._find_column(cpid_df, self.COLUMN_MAPPINGS['dqi'])
        if dqi_col:
            try:
                profile.dqi_score = pd.to_numeric(site_data[dqi_col], errors='coerce').mean()
            except:
                pass
        
        ssm_col = self._find_column(cpid_df, self.COLUMN_MAPPINGS['ssm'])
        if ssm_col:
            try:
                profile.ssm_score = pd.to_numeric(site_data[ssm_col], errors='coerce').mean()
            except:
                pass
        
        # Identify compliance issues
        profile.issues = self._identify_issues(profile, site_data, site_col, subject_col)
        
        # Determine overall risk
        profile.overall_risk = self._calculate_overall_risk(profile)
        
        return profile
    
    def _identify_issues(self, profile: SiteRiskProfile, site_data: pd.DataFrame,
                        site_col: str, subject_col: Optional[str]) -> List[ComplianceIssue]:
        """Identify compliance issues for a site"""
        issues = []
        
        # Protocol Deviations
        if profile.protocol_deviations > 0:
            risk = RiskCategory.CRITICAL if profile.protocol_deviations >= self.THRESHOLDS['protocol_deviations']['critical'] else \
                   RiskCategory.HIGH if profile.protocol_deviations >= self.THRESHOLDS['protocol_deviations']['warning'] else \
                   RiskCategory.MODERATE
            
            issues.append(ComplianceIssue(
                issue_type=IssueType.PROTOCOL_DEVIATION,
                risk_category=risk,
                description=f"There are {profile.protocol_deviations} confirmed Protocol Deviations that require CAPA documentation",
                count=profile.protocol_deviations,
                metric_value=profile.protocol_deviations,
                threshold_breached=f">{self.THRESHOLDS['protocol_deviations']['warning']}",
                recommended_action="Review PD documentation and ensure CAPA plans are in place",
                capa_required=True,
                regulatory_impact=True
            ))
        
        # CRFs Overdue for Signature
        if profile.crfs_overdue_signature > 0:
            risk = RiskCategory.CRITICAL if profile.crfs_overdue_signature >= self.THRESHOLDS['crfs_overdue_90_days']['critical'] else \
                   RiskCategory.HIGH if profile.crfs_overdue_signature >= self.THRESHOLDS['crfs_overdue_90_days']['warning'] else \
                   RiskCategory.MODERATE
            
            issues.append(ComplianceIssue(
                issue_type=IssueType.OVERDUE_SIGNATURE,
                risk_category=risk,
                description=f"Note that {profile.crfs_overdue_signature} CRFs are overdue for signature beyond 90 days, which is a critical compliance risk",
                count=profile.crfs_overdue_signature,
                metric_value=profile.crfs_overdue_signature,
                threshold_breached=">90 days",
                recommended_action="Obtain signatures immediately or document reason for delay",
                capa_required=risk == RiskCategory.CRITICAL,
                regulatory_impact=True
            ))
        
        # Broken Signatures
        if profile.broken_signatures > 0:
            # Try to get subject IDs with broken signatures
            subjects_with_broken = []
            if subject_col:
                broken_col = self._find_column(site_data, self.COLUMN_MAPPINGS['broken_signs'])
                if broken_col:
                    broken_data = site_data[pd.to_numeric(site_data[broken_col], errors='coerce') > 0]
                    subjects_with_broken = broken_data[subject_col].dropna().unique().tolist()[:5]
            
            risk = RiskCategory.HIGH if profile.broken_signatures >= self.THRESHOLDS['broken_signatures']['critical'] else \
                   RiskCategory.MODERATE
            
            subject_text = f" for {', '.join(map(str, subjects_with_broken))}" if subjects_with_broken else ""
            
            issues.append(ComplianceIssue(
                issue_type=IssueType.BROKEN_SIGNATURE,
                risk_category=risk,
                description=f"Please address the {profile.broken_signatures} 'Broken Signatures' on forms{subject_text}",
                subject_ids=[str(s) for s in subjects_with_broken],
                count=profile.broken_signatures,
                metric_value=profile.broken_signatures,
                recommended_action="Re-sign affected forms to restore data integrity",
                regulatory_impact=True
            ))
        
        # Missing Pages
        if profile.missing_pages > 0:
            risk = RiskCategory.HIGH if profile.missing_pages >= self.THRESHOLDS['missing_pages']['critical'] else \
                   RiskCategory.MODERATE if profile.missing_pages >= self.THRESHOLDS['missing_pages']['warning'] else \
                   RiskCategory.LOW
            
            issues.append(ComplianceIssue(
                issue_type=IssueType.MISSING_DATA,
                risk_category=risk,
                description=f"{profile.missing_pages} pages are missing and require data entry",
                count=profile.missing_pages,
                metric_value=profile.missing_pages,
                recommended_action="Complete data entry for missing pages",
            ))
        
        # Open Queries
        if profile.open_queries > 0:
            risk = RiskCategory.HIGH if profile.open_queries >= 20 else \
                   RiskCategory.MODERATE if profile.open_queries >= 10 else \
                   RiskCategory.LOW
            
            issues.append(ComplianceIssue(
                issue_type=IssueType.QUERY_AGING,
                risk_category=risk,
                description=f"{profile.open_queries} open queries require site response",
                count=profile.open_queries,
                metric_value=profile.open_queries,
                recommended_action="Review and respond to all open queries",
            ))
        
        # DQI Score below threshold
        if profile.dqi_score is not None and profile.dqi_score < self.THRESHOLDS['dqi_score']['warning']:
            risk = RiskCategory.CRITICAL if profile.dqi_score < self.THRESHOLDS['dqi_score']['critical'] else RiskCategory.HIGH
            
            issues.append(ComplianceIssue(
                issue_type=IssueType.DATA_QUALITY,
                risk_category=risk,
                description=f"Data Quality Index ({profile.dqi_score:.1%}) is below acceptable threshold",
                metric_value=profile.dqi_score,
                threshold_breached=f"<{self.THRESHOLDS['dqi_score']['warning']:.0%}",
                recommended_action="Review data entry practices and implement quality improvements",
            ))
        
        # SSM Score below threshold
        if profile.ssm_score is not None and profile.ssm_score < self.THRESHOLDS['ssm_score']['warning']:
            risk = RiskCategory.CRITICAL if profile.ssm_score < self.THRESHOLDS['ssm_score']['critical'] else RiskCategory.HIGH
            
            issues.append(ComplianceIssue(
                issue_type=IssueType.DATA_QUALITY,
                risk_category=risk,
                description=f"Site Status Metric ({profile.ssm_score:.0f}) indicates performance concerns",
                metric_value=profile.ssm_score,
                threshold_breached=f"<{self.THRESHOLDS['ssm_score']['warning']}",
                recommended_action="Schedule performance improvement discussion with site",
            ))
        
        # Sort by priority
        issues.sort(key=lambda x: x.get_priority_score(), reverse=True)
        
        return issues
    
    def _calculate_overall_risk(self, profile: SiteRiskProfile) -> RiskCategory:
        """Calculate overall risk category for site"""
        if not profile.issues:
            return RiskCategory.LOW
        
        # Count issues by category
        critical_count = len([i for i in profile.issues if i.risk_category == RiskCategory.CRITICAL])
        high_count = len([i for i in profile.issues if i.risk_category == RiskCategory.HIGH])
        
        if critical_count > 0:
            return RiskCategory.CRITICAL
        elif high_count >= 2:
            return RiskCategory.HIGH
        elif high_count == 1:
            return RiskCategory.MODERATE
        else:
            return RiskCategory.LOW
    
    def generate_visit_letter(self, site_id: str, visit_date: datetime = None) -> CRAVisitLetter:
        """
        Generate a CRA Visit Letter for a site
        
        Args:
            site_id: The site identifier
            visit_date: Optional scheduled visit date
            
        Returns:
            CRAVisitLetter with prioritized items
        """
        # Get risk profile
        profile = self.analyze_site_risk(site_id)
        
        # Create letter
        letter = CRAVisitLetter(
            site_id=site_id,
            site_name=profile.site_name,
            visit_date=visit_date,
            risk_profile=profile,
            total_issues=len(profile.issues),
            critical_issues=len([i for i in profile.issues if i.risk_category == RiskCategory.CRITICAL])
        )
        
        # Set greeting
        letter.greeting = self._letter_templates['greeting']
        
        # Set introduction based on severity
        has_critical = letter.critical_issues > 0
        letter.introduction = self._letter_templates['introduction_urgent'] if has_critical else self._letter_templates['introduction']
        
        # Build priority items from issues
        for i, issue in enumerate(profile.issues[:5], 1):
            letter.priority_items.append(f"{issue.description}")
        
        # Build subject-specific items
        for issue in profile.issues:
            if issue.subject_ids:
                for subject in issue.subject_ids[:3]:
                    if subject not in letter.subject_specific_items:
                        letter.subject_specific_items[subject] = []
                    letter.subject_specific_items[subject].append(
                        f"{issue.issue_type.value}: {issue.recommended_action}"
                    )
        
        # Build action items from recommendations
        for issue in profile.issues:
            if issue.recommended_action and issue.capa_required:
                letter.action_items.append(f"CAPA Required: {issue.recommended_action}")
        
        # Set closing
        letter.closing = self._letter_templates['closing_urgent'] if has_critical else self._letter_templates['closing']
        
        return letter
    
    def generate_monitoring_report(self, site_id: str, visit_date: datetime = None) -> MonitoringReport:
        """
        Generate a full Monitoring Visit Report
        
        Args:
            site_id: The site identifier
            visit_date: The visit date (defaults to today)
            
        Returns:
            MonitoringReport with comprehensive findings
        """
        if visit_date is None:
            visit_date = datetime.now()
        
        # Get risk profile
        profile = self.analyze_site_risk(site_id)
        
        # Create report
        report = MonitoringReport(
            report_id=f"MVR-{site_id}-{visit_date.strftime('%Y%m%d')}",
            site_id=site_id,
            visit_date=visit_date,
            site_profile=profile,
            compliance_findings=profile.issues,
            overall_risk_rating=profile.overall_risk
        )
        
        # Generate executive summary
        report.executive_summary = self._generate_executive_summary(profile)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(profile)
        
        # Generate follow-up items
        report.follow_up_items = self._generate_follow_up_items(profile)
        
        # Build metrics summary
        report.metrics_summary = {
            'Total Subjects': {'value': profile.total_subjects, 'status': '✓'},
            'Protocol Deviations': {
                'value': profile.protocol_deviations,
                'status': '⚠️' if profile.protocol_deviations > 0 else '✓'
            },
            'CRFs Overdue (>90 days)': {
                'value': profile.crfs_overdue_signature,
                'status': '🔴' if profile.crfs_overdue_signature >= 3 else '⚠️' if profile.crfs_overdue_signature > 0 else '✓'
            },
            'Broken Signatures': {
                'value': profile.broken_signatures,
                'status': '⚠️' if profile.broken_signatures > 0 else '✓'
            },
            'Missing Pages': {
                'value': profile.missing_pages,
                'status': '⚠️' if profile.missing_pages > 5 else '✓'
            },
            'Open Queries': {
                'value': profile.open_queries,
                'status': '⚠️' if profile.open_queries > 10 else '✓'
            }
        }
        
        if profile.dqi_score is not None:
            report.metrics_summary['Data Quality Index'] = {
                'value': f"{profile.dqi_score:.1%}",
                'status': '🔴' if profile.dqi_score < 0.5 else '⚠️' if profile.dqi_score < 0.7 else '✓'
            }
        
        if profile.ssm_score is not None:
            report.metrics_summary['Site Status Metric'] = {
                'value': f"{profile.ssm_score:.0f}",
                'status': '🔴' if profile.ssm_score < 50 else '⚠️' if profile.ssm_score < 70 else '✓'
            }
        
        return report
    
    def _generate_executive_summary(self, profile: SiteRiskProfile) -> str:
        """Generate executive summary for the report"""
        risk_descriptions = {
            RiskCategory.CRITICAL: "This site requires **immediate attention** due to critical compliance issues.",
            RiskCategory.HIGH: "This site has **significant compliance concerns** that should be prioritized.",
            RiskCategory.MODERATE: "This site has **notable issues** that require follow-up but are manageable.",
            RiskCategory.LOW: "This site is performing **within acceptable parameters** with minor items for review."
        }
        
        summary_parts = [
            f"Site {profile.site_id} has **{profile.total_subjects} subjects** enrolled.",
            risk_descriptions.get(profile.overall_risk, ""),
        ]
        
        # Add key metrics
        if profile.protocol_deviations > 0:
            summary_parts.append(f"There are **{profile.protocol_deviations} Protocol Deviations** requiring documentation.")
        
        if profile.crfs_overdue_signature > 0:
            summary_parts.append(f"**{profile.crfs_overdue_signature} CRFs** are overdue for signature beyond 90 days.")
        
        if profile.broken_signatures > 0:
            summary_parts.append(f"**{profile.broken_signatures} broken signatures** need to be addressed.")
        
        return " ".join(summary_parts)
    
    def _generate_recommendations(self, profile: SiteRiskProfile) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        # CAPA recommendations
        capa_issues = [i for i in profile.issues if i.capa_required]
        if capa_issues:
            recommendations.append(f"Initiate CAPA process for {len(capa_issues)} identified compliance issue(s)")
        
        # Regulatory impact items
        reg_issues = [i for i in profile.issues if i.regulatory_impact]
        if reg_issues:
            recommendations.append("Document all regulatory-impacting issues in the site file")
        
        # Training recommendations
        if profile.protocol_deviations >= 3:
            recommendations.append("Schedule protocol re-training session with site staff")
        
        # Data quality recommendations
        if profile.dqi_score is not None and profile.dqi_score < 0.7:
            recommendations.append("Implement data quality improvement plan with site")
        
        # Signature process
        if profile.crfs_overdue_signature > 0 or profile.broken_signatures > 0:
            recommendations.append("Review signature workflow and implement timely completion process")
        
        if not recommendations:
            recommendations.append("Continue routine monitoring activities")
        
        return recommendations
    
    def _generate_follow_up_items(self, profile: SiteRiskProfile) -> List[str]:
        """Generate follow-up action items"""
        follow_ups = []
        
        for issue in profile.issues:
            if issue.risk_category in [RiskCategory.CRITICAL, RiskCategory.HIGH]:
                follow_ups.append(f"Resolve {issue.issue_type.value}: {issue.recommended_action}")
        
        # Add standard follow-ups
        if profile.protocol_deviations > 0:
            follow_ups.append("Obtain signed CAPA documentation for all Protocol Deviations")
        
        if profile.open_queries > 0:
            follow_ups.append(f"Close {profile.open_queries} open queries within 14 days")
        
        return follow_ups[:10]  # Limit to 10 items
    
    def generate_batch_reports(self, site_list: List[str] = None) -> Dict[str, MonitoringReport]:
        """
        Generate reports for multiple sites
        
        Args:
            site_list: List of site IDs. If None, generates for all sites.
            
        Returns:
            Dictionary of site_id -> MonitoringReport
        """
        if site_list is None:
            # Get all sites from CPID
            cpid_df = self.data_sources.get('cpid')
            if cpid_df is None or (isinstance(cpid_df, pd.DataFrame) and cpid_df.empty):
                cpid_df = self.data_sources.get('CPID_EDC_Metrics')
            if cpid_df is not None and not cpid_df.empty:
                site_col = self._find_column(cpid_df, self.COLUMN_MAPPINGS['site'])
                if site_col:
                    site_list = cpid_df[site_col].dropna().unique().tolist()
        
        if not site_list:
            return {}
        
        reports = {}
        for site_id in site_list:
            try:
                report = self.generate_monitoring_report(str(site_id))
                reports[str(site_id)] = report
            except Exception as e:
                logger.error(f"Failed to generate report for {site_id}: {e}")
        
        return reports
    
    def get_portfolio_summary(self, reports: Dict[str, MonitoringReport] = None) -> str:
        """Generate a portfolio-level summary across all sites"""
        if reports is None:
            reports = self.generate_batch_reports()
        
        if not reports:
            return "No reports available."
        
        # Count by risk
        risk_counts = {}
        for report in reports.values():
            risk = report.overall_risk_rating
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        lines = [
            "# RBM Portfolio Summary",
            f"**Total Sites Analyzed:** {len(reports)}",
            f"**Report Date:** {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "## Risk Distribution",
        ]
        
        for risk in [RiskCategory.CRITICAL, RiskCategory.HIGH, RiskCategory.MODERATE, RiskCategory.LOW]:
            count = risk_counts.get(risk, 0)
            if count > 0:
                icon = {'critical': '🔴', 'high': '🟠', 'moderate': '🟡', 'low': '🟢'}[risk.value]
                lines.append(f"- {icon} **{risk.value.upper()}:** {count} site(s)")
        
        # List critical sites
        critical_sites = [r for r in reports.values() if r.overall_risk_rating == RiskCategory.CRITICAL]
        if critical_sites:
            lines.extend(["", "## ⚠️ Sites Requiring Immediate Attention"])
            for report in critical_sites:
                lines.append(f"- **{report.site_id}**: {len(report.compliance_findings)} issues identified")
        
        return "\n".join(lines)
