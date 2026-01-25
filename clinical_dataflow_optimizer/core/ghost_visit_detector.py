"""
Ghost Visit Detector - Scenario C: The "Ghost" Visit

This module implements the detection and handling of missing visit data
where patients were scheduled for visits but no data has been entered.

Problem: A patient is scheduled for a visit (e.g., "Cycle 3" on Jan 1st).
It is now Jan 10th, and no data has been entered. The site has not flagged
a "Missed Visit."

Agentic Solution Pipeline:
1. Data Check: Ingest Visit Projection Tracker, filter for Projected Date < Current Date
2. Calculation: Calculate Days Outstanding
3. Context Check: Check Inactivated forms for valid reasons (e.g., "Patient Withdrew")
4. Action: 
   - If valid inactivation: No query needed
   - If site is "Green" (good standing): Send standard reminder
   - If site is "Red" (non-compliance history): Escalate to CRA for phone call

Author: Clinical Dataflow Optimizer AI Agent
Version: 1.0.0
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class VisitStatus(Enum):
    """Status classification for ghost visits"""
    GHOST_CONFIRMED = auto()     # True missing visit - no data, no valid reason
    VALIDLY_INACTIVATED = auto() # Missing but validly inactivated (withdrawal, etc.)
    PARTIAL_DATA = auto()        # Some data entered but incomplete
    PENDING_REVIEW = auto()      # Needs manual review
    RECENTLY_ENTERED = auto()    # Data entered since last check


class SiteStanding(Enum):
    """Site compliance standing for escalation decisions"""
    GREEN = auto()    # Good standing - standard reminder
    YELLOW = auto()   # Some issues - enhanced reminder
    RED = auto()      # Non-compliance history - escalate to CRA


class EscalationLevel(Enum):
    """Escalation level for ghost visit actions"""
    STANDARD_REMINDER = auto()     # Email reminder to site
    ENHANCED_REMINDER = auto()     # Priority email with deadline
    CRA_ESCALATION = auto()        # Direct phone call from CRA
    SPONSOR_ESCALATION = auto()    # Sponsor notification for critical


class InactivationReason(Enum):
    """Valid reasons for inactivated visits"""
    PATIENT_WITHDREW = auto()
    PATIENT_DISCONTINUED = auto()
    VISIT_NOT_APPLICABLE = auto()
    VISIT_NOT_DONE = auto()
    PROTOCOL_DEVIATION = auto()
    PATIENT_LOST_TO_FOLLOWUP = auto()
    VISIT_WINDOW_MISSED = auto()
    DATA_ENTRY_ERROR = auto()
    OTHER = auto()


@dataclass
class GhostVisit:
    """Represents a ghost visit requiring attention"""
    subject_id: str
    site_id: str
    country: str
    visit_name: str
    projected_date: datetime
    days_outstanding: int
    visit_status: VisitStatus
    site_standing: SiteStanding
    escalation_level: EscalationLevel
    inactivation_check: Dict[str, Any] = field(default_factory=dict)
    auto_reminder: str = ""
    detection_timestamp: datetime = field(default_factory=datetime.now)
    risk_score: float = 0.0
    data_entry_deadline: Optional[datetime] = None


@dataclass
class GhostVisitConfig:
    """Configuration for ghost visit detection"""
    # Days thresholds
    days_warning_threshold: int = 7          # Days before warning
    days_critical_threshold: int = 14        # Days before critical escalation
    days_sponsor_escalation: int = 21        # Days before sponsor notification
    
    # Site standing thresholds (based on metrics)
    site_red_threshold: int = 5              # Open issues for Red status
    site_yellow_threshold: int = 3           # Open issues for Yellow status
    
    # Visit patterns indicating scheduled visits
    scheduled_visit_patterns: List[str] = field(default_factory=lambda: [
        r'Cycle\s*\d+',           # Cycle 1, Cycle 2, etc.
        r'Visit\s*\d+',           # Visit 1, Visit 2, etc.
        r'Week\s*\d+',            # Week 1, Week 2, etc.
        r'W\d+D\d+',              # W2D7 (Week 2 Day 7)
        r'Day\s*\d+',             # Day 1, Day 15, etc.
        r'V\d+',                  # V1, V2, etc.
        r'Screening',             # Screening visit
        r'Baseline',              # Baseline visit
        r'End\s*of\s*Treatment',  # End of Treatment
        r'EOT',                   # EOT
        r'Follow-?up',            # Follow-up visits
        r'Unscheduled'            # Unscheduled visits
    ])
    
    # Valid inactivation reasons (no query needed)
    valid_inactivation_keywords: List[str] = field(default_factory=lambda: [
        'withdrew', 'withdrawal', 'discontinued', 'lost to follow',
        'not applicable', 'not done', 'deceased', 'death',
        'protocol deviation', 'screen failure', 'not eligible',
        'consent withdrawn', 'patient request', 'early termination'
    ])
    
    # Reminder templates
    reminder_templates: Dict[str, str] = field(default_factory=lambda: {
        'standard': (
            "Reminder: Visit '{visit}' for Subject {subject_id} was projected for "
            "{projected_date}. It is now {days_outstanding} days past the scheduled date. "
            "Please enter the visit data or mark as 'Missed Visit' if applicable."
        ),
        'enhanced': (
            "PRIORITY: Visit '{visit}' for Subject {subject_id} is {days_outstanding} days overdue "
            "(projected {projected_date}). Data entry required within 3 business days. "
            "If visit was not conducted, please document in EDC with appropriate reason code."
        ),
        'cra_escalation': (
            "ESCALATION REQUIRED: Site {site_id} - Subject {subject_id}\n"
            "Visit '{visit}' is {days_outstanding} days overdue (projected {projected_date}).\n"
            "Site has history of data entry delays. Recommend immediate phone call to Site Coordinator.\n"
            "Reference: ICH E6 R2 Section 4.9.5 - Source document verification"
        ),
        'sponsor_alert': (
            "SPONSOR ALERT: Critical data gap at Site {site_id}\n"
            "Subject {subject_id} - Visit '{visit}' is {days_outstanding} days overdue.\n"
            "No response to previous reminders. May impact database lock timeline.\n"
            "Recommend site audit consideration."
        )
    })
    
    # Data entry deadline buffer (days from detection)
    data_entry_deadline_days: int = 5
    
    # Enable risk scoring
    enable_risk_scoring: bool = True


# Default configuration
DEFAULT_GHOST_VISIT_CONFIG = GhostVisitConfig()


class GhostVisitDetector:
    """
    Detects and handles ghost visits - scheduled visits with no data entry.
    
    Implements the 4-step agentic solution:
    1. Data Check: Scan Visit Projection Tracker for overdue visits
    2. Calculation: Calculate Days Outstanding
    3. Context Check: Verify against Inactivated forms
    4. Action: Send reminder or escalate based on site standing
    """
    
    def __init__(self, config: GhostVisitConfig = None):
        self.config = config or DEFAULT_GHOST_VISIT_CONFIG
        self.ghost_visits: List[GhostVisit] = []
        self._site_metrics: Dict[str, Dict[str, Any]] = {}
        self._inactivation_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._detection_stats = {
            'total_overdue': 0,
            'ghost_confirmed': 0,
            'validly_inactivated': 0,
            'by_escalation': {},
            'by_site': {}
        }
        
        logger.info("GhostVisitDetector initialized")
    
    def detect(
        self,
        visit_tracker_data: pd.DataFrame,
        inactivated_forms_data: Optional[pd.DataFrame] = None,
        cpid_data: Optional[pd.DataFrame] = None,
        current_date: Optional[datetime] = None,
        study_id: str = ""
    ) -> List[GhostVisit]:
        """
        Main detection pipeline for ghost visits.
        
        Args:
            visit_tracker_data: Visit Projection Tracker DataFrame
            inactivated_forms_data: Inactivated Forms Report DataFrame
            cpid_data: CPID_EDC_Metrics for site standing
            current_date: Current date for calculations (defaults to now)
            study_id: Study identifier
            
        Returns:
            List of GhostVisit objects requiring attention
        """
        self.ghost_visits = []
        self._reset_stats()
        
        if current_date is None:
            current_date = datetime.now()
        
        logger.info(f"Starting ghost visit detection for study {study_id}")
        
        # Step 1: Data Check - Find overdue visits
        overdue_visits = self._scan_for_overdue_visits(visit_tracker_data, current_date)
        self._detection_stats['total_overdue'] = len(overdue_visits)
        
        logger.info(f"Found {len(overdue_visits)} overdue visits to analyze")
        
        # Build context caches
        if inactivated_forms_data is not None:
            self._build_inactivation_cache(inactivated_forms_data)
        
        if cpid_data is not None:
            self._build_site_metrics(cpid_data)
        
        # Process each overdue visit
        for _, row in overdue_visits.iterrows():
            ghost = self._analyze_visit(row, current_date, study_id)
            if ghost:
                self.ghost_visits.append(ghost)
                self._update_stats(ghost)
        
        logger.info(f"Detection complete: {self._detection_stats}")
        return self.ghost_visits
    
    def _reset_stats(self):
        """Reset detection statistics"""
        self._detection_stats = {
            'total_overdue': 0,
            'ghost_confirmed': 0,
            'validly_inactivated': 0,
            'by_escalation': {},
            'by_site': {}
        }
    
    def _scan_for_overdue_visits(
        self,
        visit_tracker: pd.DataFrame,
        current_date: datetime
    ) -> pd.DataFrame:
        """
        Step 1: Data Check - Scan Visit Projection Tracker for overdue visits.
        Filter for Projected Date < Current Date
        """
        # Handle duplicate columns
        if visit_tracker.columns.duplicated().any():
            visit_tracker = visit_tracker.loc[:, ~visit_tracker.columns.duplicated()]
        
        # Find projected date column
        date_col = None
        for col in ['Projected Date', 'projected_date', 'ProjectedDate', 'Visit Date']:
            if col in visit_tracker.columns:
                date_col = col
                break
        
        if date_col is None:
            logger.warning("No Projected Date column found in visit tracker")
            return pd.DataFrame()
        
        # Convert dates
        def parse_date(val):
            if pd.isna(val):
                return None
            if isinstance(val, datetime):
                return val
            try:
                # Handle various date formats
                for fmt in ['%d%b%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%d-%b-%Y']:
                    try:
                        return datetime.strptime(str(val).strip().upper(), fmt)
                    except ValueError:
                        continue
                return pd.to_datetime(val)
            except:
                return None
        
        visit_tracker_copy = visit_tracker.copy()
        visit_tracker_copy['_parsed_date'] = visit_tracker_copy[date_col].apply(parse_date)
        
        # Filter for overdue (Projected Date < Current Date)
        # We don't check for days_warning_threshold here - we want ALL overdue visits
        overdue_mask = visit_tracker_copy['_parsed_date'].notna() & \
                       (visit_tracker_copy['_parsed_date'] < current_date)
        
        overdue = visit_tracker_copy[overdue_mask].copy()
        
        # Add parsed date for later use
        return overdue
    
    def _build_inactivation_cache(self, inactivated_data: pd.DataFrame):
        """
        Build cache of inactivated visits for context checking.
        Maps subject_id -> list of inactivated entries
        """
        self._inactivation_cache = {}
        
        # Handle duplicate columns
        if inactivated_data.columns.duplicated().any():
            inactivated_data = inactivated_data.loc[:, ~inactivated_data.columns.duplicated()]
        
        # Find subject column
        subject_col = None
        for col in ['Subject', 'subject_id', 'Subject ID']:
            if col in inactivated_data.columns:
                subject_col = col
                break
        
        # Find folder/visit column
        folder_col = None
        for col in ['Folder', 'folder', 'Visit', 'visit']:
            if col in inactivated_data.columns:
                folder_col = col
                break
        
        # Find audit action column
        action_col = None
        for col in ['Audit Action', 'audit_action', 'AuditAction']:
            if col in inactivated_data.columns:
                action_col = col
                break
        
        if not subject_col or not action_col:
            logger.warning("Required columns not found in inactivated forms data")
            return
        
        for _, row in inactivated_data.iterrows():
            subject_id = str(row.get(subject_col, '')).strip()
            if not subject_id:
                continue
            
            if subject_id not in self._inactivation_cache:
                self._inactivation_cache[subject_id] = []
            
            self._inactivation_cache[subject_id].append({
                'folder': str(row.get(folder_col, '')) if folder_col else '',
                'form': str(row.get('Form ', row.get('Form', ''))),
                'action': str(row.get(action_col, '')),
                'site': str(row.get('Study Site Number', row.get('Site', '')))
            })
        
        logger.info(f"Built inactivation cache for {len(self._inactivation_cache)} subjects")
    
    def _build_site_metrics(self, cpid_data: pd.DataFrame):
        """
        Build site metrics from CPID for site standing determination.
        Uses query status, page status, and other metrics to determine standing.
        """
        self._site_metrics = {}
        
        try:
            # Handle multi-level headers
            if isinstance(cpid_data.columns, pd.MultiIndex):
                cpid_flat = cpid_data.copy()
                cpid_flat.columns = [
                    str(c[0]) if 'Unnamed' in str(c[1]) else f"{c[0]}_{c[1]}"
                    for c in cpid_data.columns
                ]
            else:
                cpid_flat = cpid_data.copy()
            
            # Find site column
            site_col = None
            for col in cpid_flat.columns:
                col_lower = str(col).lower()
                if 'site' in col_lower and 'id' in col_lower:
                    site_col = col
                    break
            
            if not site_col:
                # Try positional fallback
                if len(cpid_flat.columns) > 3:
                    site_col = cpid_flat.columns[3]
            
            if not site_col:
                logger.warning("No site column found in CPID data")
                return
            
            # Find open queries column
            queries_col = None
            for col in cpid_flat.columns:
                col_str = str(col).lower()
                if 'quer' in col_str and 'open' in col_str:
                    queries_col = col
                    break
            
            # Aggregate metrics by site
            for _, row in cpid_flat.iterrows():
                site_id = str(row.get(site_col, '')).strip()
                if not site_id or 'nan' in site_id.lower():
                    continue
                
                if site_id not in self._site_metrics:
                    self._site_metrics[site_id] = {
                        'open_queries': 0,
                        'subjects_count': 0,
                        'total_issues': 0
                    }
                
                self._site_metrics[site_id]['subjects_count'] += 1
                
                # Sum up open queries if column exists
                if queries_col and pd.notna(row.get(queries_col)):
                    try:
                        self._site_metrics[site_id]['open_queries'] += int(row[queries_col])
                    except (ValueError, TypeError):
                        pass
            
            logger.info(f"Built site metrics for {len(self._site_metrics)} sites")
            
        except Exception as e:
            logger.warning(f"Error building site metrics: {e}")
    
    def _analyze_visit(
        self,
        row: pd.Series,
        current_date: datetime,
        study_id: str
    ) -> Optional[GhostVisit]:
        """
        Steps 2-4: Analyze a single overdue visit.
        
        2. Calculation: Calculate Days Outstanding
        3. Context Check: Check inactivation status
        4. Action: Determine escalation level
        """
        # Extract row data
        subject_id = str(row.get('Subject', '')).strip()
        site_id = str(row.get('Site', '')).strip()
        country = str(row.get('Country', '')).strip()
        visit_name = str(row.get('Visit', '')).strip()
        
        # Get projected date
        projected_date = row.get('_parsed_date')
        if projected_date is None:
            return None
        
        # Step 2: Calculation - Days Outstanding
        days_outstanding_col = None
        for col in ['# Days Outstanding', 'Days Outstanding', 'days_outstanding']:
            if col in row.index:
                days_outstanding_col = col
                break
        
        if days_outstanding_col and pd.notna(row.get(days_outstanding_col)):
            days_outstanding = int(row[days_outstanding_col])
        else:
            # Calculate manually
            days_outstanding = (current_date - projected_date).days
        
        # Skip if not past warning threshold
        if days_outstanding < self.config.days_warning_threshold:
            return None
        
        # Step 3: Context Check - Check inactivation status
        inactivation_check = self._check_inactivation_status(subject_id, visit_name)
        
        # Determine visit status
        if inactivation_check.get('is_valid_inactivation'):
            visit_status = VisitStatus.VALIDLY_INACTIVATED
        else:
            visit_status = VisitStatus.GHOST_CONFIRMED
        
        # Get site standing
        site_standing = self._get_site_standing(site_id)
        
        # Step 4: Action - Determine escalation level and generate reminder
        escalation_level, auto_reminder = self._determine_action(
            visit_status, site_standing, days_outstanding,
            subject_id, site_id, visit_name, projected_date
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            days_outstanding, site_standing, visit_status
        )
        
        # Set data entry deadline
        deadline = current_date + timedelta(days=self.config.data_entry_deadline_days)
        
        return GhostVisit(
            subject_id=subject_id,
            site_id=site_id,
            country=country,
            visit_name=visit_name,
            projected_date=projected_date,
            days_outstanding=days_outstanding,
            visit_status=visit_status,
            site_standing=site_standing,
            escalation_level=escalation_level,
            inactivation_check=inactivation_check,
            auto_reminder=auto_reminder,
            risk_score=risk_score,
            data_entry_deadline=deadline
        )
    
    def _check_inactivation_status(
        self,
        subject_id: str,
        visit_name: str
    ) -> Dict[str, Any]:
        """
        Step 3: Context Check - Check if visit was validly inactivated.
        
        Checks Inactivated forms for:
        - Did site enter and then inactivate the visit?
        - Is there a valid reason (Patient Withdrew, Not Applicable, etc.)?
        """
        result = {
            'is_valid_inactivation': False,
            'inactivation_reason': None,
            'inactivation_action': None,
            'requires_query': True
        }
        
        if subject_id not in self._inactivation_cache:
            return result
        
        subject_inactivations = self._inactivation_cache[subject_id]
        
        # Look for matching visit/folder
        for entry in subject_inactivations:
            folder = entry.get('folder', '').lower()
            action = entry.get('action', '').lower()
            
            # Check if folder matches visit name
            visit_lower = visit_name.lower()
            if visit_lower in folder or self._visit_matches(visit_lower, folder):
                # Check if it's a valid inactivation reason
                for keyword in self.config.valid_inactivation_keywords:
                    if keyword in action:
                        result['is_valid_inactivation'] = True
                        result['inactivation_reason'] = self._classify_inactivation(action)
                        result['inactivation_action'] = entry.get('action')
                        result['requires_query'] = False
                        return result
        
        # Check for subject-level inactivation (withdrawal, discontinuation)
        for entry in subject_inactivations:
            action = entry.get('action', '').lower()
            
            # Check for subject discontinuation
            if any(kw in action for kw in ['withdrew', 'discontinued', 'lost to follow', 'deceased']):
                result['is_valid_inactivation'] = True
                result['inactivation_reason'] = self._classify_inactivation(action)
                result['inactivation_action'] = entry.get('action')
                result['requires_query'] = False
                return result
        
        return result
    
    def _visit_matches(self, visit_name: str, folder_name: str) -> bool:
        """Check if visit name matches folder name (handles different formats)"""
        # Extract numeric parts
        visit_nums = re.findall(r'\d+', visit_name)
        folder_nums = re.findall(r'\d+', folder_name)
        
        if visit_nums and folder_nums and visit_nums[0] == folder_nums[0]:
            return True
        
        # Check for common patterns
        patterns = [
            (r'w(\d+)d(\d+)', r'week\s*\1.*day\s*\2'),  # W2D7 -> Week 2 Day 7
            (r'cycle\s*(\d+)', r'cycle\s*\1'),          # Cycle matching
            (r'visit\s*(\d+)', r'visit\s*\1'),          # Visit matching
        ]
        
        for pattern1, pattern2 in patterns:
            match1 = re.search(pattern1, visit_name, re.IGNORECASE)
            match2 = re.search(pattern2, folder_name, re.IGNORECASE)
            if match1 and match2:
                return True
        
        return False
    
    def _classify_inactivation(self, action_text: str) -> InactivationReason:
        """Classify the inactivation reason from audit action text"""
        action_lower = action_text.lower()
        
        if 'withdrew' in action_lower or 'withdrawal' in action_lower:
            return InactivationReason.PATIENT_WITHDREW
        elif 'discontinued' in action_lower:
            return InactivationReason.PATIENT_DISCONTINUED
        elif 'not applicable' in action_lower:
            return InactivationReason.VISIT_NOT_APPLICABLE
        elif 'not done' in action_lower:
            return InactivationReason.VISIT_NOT_DONE
        elif 'lost to follow' in action_lower:
            return InactivationReason.PATIENT_LOST_TO_FOLLOWUP
        elif 'protocol deviation' in action_lower:
            return InactivationReason.PROTOCOL_DEVIATION
        elif 'window' in action_lower and 'miss' in action_lower:
            return InactivationReason.VISIT_WINDOW_MISSED
        elif 'error' in action_lower:
            return InactivationReason.DATA_ENTRY_ERROR
        else:
            return InactivationReason.OTHER
    
    def _get_site_standing(self, site_id: str) -> SiteStanding:
        """
        Determine site standing from CPID metrics.
        
        Green: Good standing - few open queries
        Yellow: Some issues - moderate queries
        Red: Non-compliance history - many open queries
        """
        if site_id not in self._site_metrics:
            # Default to Yellow if no data
            return SiteStanding.YELLOW
        
        metrics = self._site_metrics[site_id]
        open_queries = metrics.get('open_queries', 0)
        subjects = metrics.get('subjects_count', 1)
        
        # Calculate queries per subject
        queries_per_subject = open_queries / max(subjects, 1)
        
        if queries_per_subject >= self.config.site_red_threshold:
            return SiteStanding.RED
        elif queries_per_subject >= self.config.site_yellow_threshold:
            return SiteStanding.YELLOW
        else:
            return SiteStanding.GREEN
    
    def _determine_action(
        self,
        visit_status: VisitStatus,
        site_standing: SiteStanding,
        days_outstanding: int,
        subject_id: str,
        site_id: str,
        visit_name: str,
        projected_date: datetime
    ) -> Tuple[EscalationLevel, str]:
        """
        Step 4: Action - Determine escalation level and generate reminder.
        
        Decision matrix:
        - Validly inactivated: No action needed
        - Green site + < critical days: Standard reminder
        - Yellow site OR > warning days: Enhanced reminder
        - Red site OR > critical days: CRA escalation
        - > sponsor days: Sponsor escalation
        """
        if visit_status == VisitStatus.VALIDLY_INACTIVATED:
            return (EscalationLevel.STANDARD_REMINDER, "")  # No action needed
        
        # Format date for templates
        date_str = projected_date.strftime('%d-%b-%Y')
        
        # Determine escalation level
        if days_outstanding >= self.config.days_sponsor_escalation:
            level = EscalationLevel.SPONSOR_ESCALATION
            template = self.config.reminder_templates['sponsor_alert']
            
        elif days_outstanding >= self.config.days_critical_threshold or \
             site_standing == SiteStanding.RED:
            level = EscalationLevel.CRA_ESCALATION
            template = self.config.reminder_templates['cra_escalation']
            
        elif days_outstanding >= self.config.days_warning_threshold or \
             site_standing == SiteStanding.YELLOW:
            level = EscalationLevel.ENHANCED_REMINDER
            template = self.config.reminder_templates['enhanced']
            
        else:
            level = EscalationLevel.STANDARD_REMINDER
            template = self.config.reminder_templates['standard']
        
        # Generate reminder text
        reminder = template.format(
            visit=visit_name,
            subject_id=subject_id,
            site_id=site_id,
            projected_date=date_str,
            days_outstanding=days_outstanding
        )
        
        return (level, reminder)
    
    def _calculate_risk_score(
        self,
        days_outstanding: int,
        site_standing: SiteStanding,
        visit_status: VisitStatus
    ) -> float:
        """Calculate risk score (0-1) for prioritization"""
        if not self.config.enable_risk_scoring:
            return 0.0
        
        # Base score from days outstanding
        if days_outstanding >= self.config.days_sponsor_escalation:
            days_score = 1.0
        elif days_outstanding >= self.config.days_critical_threshold:
            days_score = 0.8
        elif days_outstanding >= self.config.days_warning_threshold:
            days_score = 0.5
        else:
            days_score = 0.2
        
        # Site standing factor
        standing_factor = {
            SiteStanding.RED: 1.0,
            SiteStanding.YELLOW: 0.7,
            SiteStanding.GREEN: 0.4
        }.get(site_standing, 0.5)
        
        # Status factor
        status_factor = {
            VisitStatus.GHOST_CONFIRMED: 1.0,
            VisitStatus.PENDING_REVIEW: 0.7,
            VisitStatus.PARTIAL_DATA: 0.5,
            VisitStatus.VALIDLY_INACTIVATED: 0.1
        }.get(visit_status, 0.5)
        
        # Weighted combination
        risk_score = (0.4 * days_score + 0.3 * standing_factor + 0.3 * status_factor)
        
        return round(min(1.0, risk_score), 2)
    
    def _update_stats(self, ghost: GhostVisit):
        """Update detection statistics"""
        if ghost.visit_status == VisitStatus.GHOST_CONFIRMED:
            self._detection_stats['ghost_confirmed'] += 1
        elif ghost.visit_status == VisitStatus.VALIDLY_INACTIVATED:
            self._detection_stats['validly_inactivated'] += 1
        
        # By escalation
        level_name = ghost.escalation_level.name
        self._detection_stats['by_escalation'][level_name] = \
            self._detection_stats['by_escalation'].get(level_name, 0) + 1
        
        # By site
        self._detection_stats['by_site'][ghost.site_id] = \
            self._detection_stats['by_site'].get(ghost.site_id, 0) + 1
    
    def get_ghost_visits(self) -> List[GhostVisit]:
        """Get all confirmed ghost visits"""
        return [g for g in self.ghost_visits if g.visit_status == VisitStatus.GHOST_CONFIRMED]
    
    def get_cra_escalations(self) -> List[Dict[str, Any]]:
        """Get visits requiring CRA escalation (phone call)"""
        escalations = []
        for ghost in self.ghost_visits:
            if ghost.escalation_level in [EscalationLevel.CRA_ESCALATION, 
                                          EscalationLevel.SPONSOR_ESCALATION]:
                escalations.append({
                    'subject_id': ghost.subject_id,
                    'site_id': ghost.site_id,
                    'country': ghost.country,
                    'visit_name': ghost.visit_name,
                    'days_outstanding': ghost.days_outstanding,
                    'projected_date': ghost.projected_date.strftime('%Y-%m-%d'),
                    'site_standing': ghost.site_standing.name,
                    'escalation_level': ghost.escalation_level.name,
                    'risk_score': ghost.risk_score,
                    'auto_reminder': ghost.auto_reminder,
                    'data_entry_deadline': ghost.data_entry_deadline.strftime('%Y-%m-%d') if ghost.data_entry_deadline else None
                })
        return escalations
    
    def get_standard_reminders(self) -> List[Dict[str, Any]]:
        """Get visits requiring standard reminders"""
        reminders = []
        for ghost in self.ghost_visits:
            if ghost.escalation_level in [EscalationLevel.STANDARD_REMINDER,
                                          EscalationLevel.ENHANCED_REMINDER] and \
               ghost.visit_status == VisitStatus.GHOST_CONFIRMED:
                reminders.append({
                    'subject_id': ghost.subject_id,
                    'site_id': ghost.site_id,
                    'visit_name': ghost.visit_name,
                    'days_outstanding': ghost.days_outstanding,
                    'projected_date': ghost.projected_date.strftime('%Y-%m-%d'),
                    'escalation_level': ghost.escalation_level.name,
                    'reminder_text': ghost.auto_reminder
                })
        return reminders
    
    def get_valid_inactivations(self) -> List[Dict[str, Any]]:
        """Get visits that were validly inactivated (no action needed)"""
        valid = []
        for ghost in self.ghost_visits:
            if ghost.visit_status == VisitStatus.VALIDLY_INACTIVATED:
                valid.append({
                    'subject_id': ghost.subject_id,
                    'site_id': ghost.site_id,
                    'visit_name': ghost.visit_name,
                    'inactivation_reason': ghost.inactivation_check.get('inactivation_reason', '').name 
                        if ghost.inactivation_check.get('inactivation_reason') else 'Unknown',
                    'audit_action': ghost.inactivation_check.get('inactivation_action', '')
                })
        return valid
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of ghost visit detection"""
        return {
            'detection_timestamp': datetime.now().isoformat(),
            'statistics': self._detection_stats,
            'breakdown': {
                'ghost_confirmed': self._detection_stats['ghost_confirmed'],
                'validly_inactivated': self._detection_stats['validly_inactivated'],
                'total_analyzed': len(self.ghost_visits)
            },
            'by_escalation': self._detection_stats.get('by_escalation', {}),
            'by_site': self._detection_stats.get('by_site', {}),
            'high_risk_visits': len([g for g in self.ghost_visits if g.risk_score >= 0.7]),
            'requires_cra_action': len(self.get_cra_escalations()),
            'requires_standard_reminder': len(self.get_standard_reminders())
        }
    
    def get_site_standing_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of site standings"""
        summary = {}
        for ghost in self.ghost_visits:
            if ghost.site_id not in summary:
                summary[ghost.site_id] = {
                    'standing': ghost.site_standing.name,
                    'ghost_visits': 0,
                    'valid_inactivations': 0,
                    'total_overdue': 0,
                    'avg_days_outstanding': 0,
                    'days_list': []
                }
            
            summary[ghost.site_id]['total_overdue'] += 1
            summary[ghost.site_id]['days_list'].append(ghost.days_outstanding)
            
            if ghost.visit_status == VisitStatus.GHOST_CONFIRMED:
                summary[ghost.site_id]['ghost_visits'] += 1
            elif ghost.visit_status == VisitStatus.VALIDLY_INACTIVATED:
                summary[ghost.site_id]['valid_inactivations'] += 1
        
        # Calculate averages
        for site_id in summary:
            days_list = summary[site_id]['days_list']
            summary[site_id]['avg_days_outstanding'] = sum(days_list) / len(days_list) if days_list else 0
            del summary[site_id]['days_list']
        
        return summary
