"""
Query Parser - Natural Language Understanding for Clinical Data Queries
=======================================================================

Parses natural language queries to extract:
- Intent: What type of analysis is requested (trend, comparison, aggregation, etc.)
- Entities: What data entities are involved (sites, subjects, countries, visits)
- Metrics: What metrics to analyze (missing visits, open queries, DQI, etc.)
- Filters: What constraints to apply (country, time period, visit name, etc.)
- Temporal: Time-based constraints (last 3 snapshots, this month, etc.)
"""

import re
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of analytical intents that can be extracted from queries"""
    TREND_ANALYSIS = auto()      # "trending up", "over time", "last N snapshots"
    COMPARISON = auto()          # "compare", "vs", "difference between"
    AGGREGATION = auto()         # "total", "sum", "count", "average"
    FILTER_LIST = auto()         # "show me all", "list", "which sites"
    TOP_N = auto()               # "top 5", "highest", "worst performing"
    BOTTOM_N = auto()            # "bottom 5", "lowest", "best performing"
    ANOMALY_DETECTION = auto()   # "unusual", "outliers", "anomalies"
    CORRELATION = auto()         # "correlates with", "related to", "caused by"
    DRILL_DOWN = auto()          # "details for", "breakdown of", "specifics"
    SAFETY_CHECK = auto()        # "SAE", "safety", "serious adverse"
    COMPLIANCE_CHECK = auto()    # "compliance", "protocol deviation", "overdue"
    FORECAST = auto()            # "predict", "forecast", "estimate"
    NARRATIVE_GENERATION = auto() # "generate narrative", "patient safety narrative", "safety report"
    RBM_REPORT = auto()          # "RBM report", "site monitoring report", "CRA report"
    UNKNOWN = auto()


class EntityType(Enum):
    """Types of entities that can be extracted from queries"""
    SITE = auto()
    SUBJECT = auto()
    COUNTRY = auto()
    REGION = auto()
    VISIT = auto()
    FORM = auto()
    STUDY = auto()
    CRA = auto()
    INVESTIGATOR = auto()


class MetricType(Enum):
    """Types of metrics that can be referenced in queries"""
    MISSING_VISITS = "missing_visits"
    MISSING_PAGES = "missing_pages"
    OPEN_QUERIES = "open_queries"
    UNCODED_TERMS = "uncoded_terms"
    DQI = "data_quality_index"
    SSM = "site_status_metric"
    CLEAN_PATIENT_RATE = "clean_patient_rate"
    SAE_COUNT = "sae_count"
    PROTOCOL_DEVIATIONS = "protocol_deviations"
    INACTIVATED_FORMS = "inactivated_forms"
    QUERY_AGING = "query_aging"
    VISIT_COMPLIANCE = "visit_compliance"
    FROZEN_CRFS = "frozen_crfs"
    LOCKED_CRFS = "locked_crfs"
    DAYS_OUTSTANDING = "days_outstanding"


@dataclass
class TimeConstraint:
    """Temporal constraint extracted from query"""
    type: str  # 'snapshot', 'date_range', 'relative'
    value: Any
    unit: Optional[str] = None  # 'days', 'weeks', 'months', 'snapshots'
    
    def to_dict(self) -> Dict:
        return {
            'type': self.type,
            'value': self.value,
            'unit': self.unit
        }


@dataclass
class EntityFilter:
    """Filter constraint for an entity"""
    entity_type: EntityType
    operator: str  # 'equals', 'in', 'contains', 'starts_with', 'greater_than', etc.
    values: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'entity_type': self.entity_type.name,
            'operator': self.operator,
            'values': self.values
        }


@dataclass
class MetricFilter:
    """Filter constraint for a metric"""
    metric: MetricType
    operator: str  # 'greater_than', 'less_than', 'equals', 'between', 'trending_up', 'trending_down'
    value: Any
    threshold: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'metric': self.metric.value,
            'operator': self.operator,
            'value': self.value,
            'threshold': self.threshold
        }


@dataclass
class ParsedQuery:
    """Result of parsing a natural language query"""
    original_query: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    primary_metric: Optional[MetricType] = None
    secondary_metrics: List[MetricType] = field(default_factory=list)
    entity_filters: List[EntityFilter] = field(default_factory=list)
    metric_filters: List[MetricFilter] = field(default_factory=list)
    time_constraint: Optional[TimeConstraint] = None
    top_n: Optional[int] = None
    group_by: List[EntityType] = field(default_factory=list)
    sort_order: str = "descending"
    data_sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    parse_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'original_query': self.original_query,
            'intent': self.intent.name,
            'primary_metric': self.primary_metric.value if self.primary_metric else None,
            'secondary_metrics': [m.value for m in self.secondary_metrics],
            'entity_filters': [f.to_dict() for f in self.entity_filters],
            'metric_filters': [f.to_dict() for f in self.metric_filters],
            'time_constraint': self.time_constraint.to_dict() if self.time_constraint else None,
            'top_n': self.top_n,
            'group_by': [e.name for e in self.group_by],
            'sort_order': self.sort_order,
            'data_sources': self.data_sources,
            'confidence': self.confidence,
            'parse_notes': self.parse_notes
        }


class QueryParser:
    """
    Natural Language Query Parser for Clinical Data
    
    Uses pattern matching and keyword extraction to understand user queries
    and convert them into structured query representations.
    """
    
    # Intent detection patterns
    INTENT_PATTERNS = {
        QueryIntent.TREND_ANALYSIS: [
            r'\btrend(ing|s)?\b', r'\bover time\b', r'\blast\s+\d+\s+(snapshot|week|month|day)s?\b',
            r'\bincreas(e|ing)\b', r'\bdecreas(e|ing)\b', r'\bchanges?\b', r'\bhistor(y|ical)\b'
        ],
        QueryIntent.COMPARISON: [
            r'\bcompar(e|ison|ing)\b', r'\bvs\.?\b', r'\bversus\b', r'\bdifference\b',
            r'\bbetween\b.*\band\b', r'\bagainst\b'
        ],
        QueryIntent.AGGREGATION: [
            r'\btotal\b', r'\bsum\b', r'\bcount\b', r'\baverage\b', r'\bmean\b',
            r'\baggregate\b', r'\boverall\b'
        ],
        QueryIntent.FILTER_LIST: [
            r'\bshow\s+me\b', r'\blist\b', r'\bwhich\b', r'\bwhat\b', r'\bfind\b',
            r'\ball\s+\w+\s+where\b', r'\bget\b'
        ],
        QueryIntent.TOP_N: [
            r'\btop\s+\d+\b', r'\bhighest\b', r'\bworst\b', r'\bmost\b', r'\blargest\b',
            r'\bmaximum\b', r'\bpeak\b'
        ],
        QueryIntent.BOTTOM_N: [
            r'\bbottom\s+\d+\b', r'\blowest\b', r'\bbest\b', r'\bleast\b', r'\bsmallest\b',
            r'\bminimum\b', r'\bfewest\b'
        ],
        QueryIntent.ANOMALY_DETECTION: [
            r'\bunusual\b', r'\boutlier\b', r'\banomaly\b', r'\babnormal\b',
            r'\bunexpected\b', r'\bsuspicious\b', r'\bflagged\b'
        ],
        QueryIntent.CORRELATION: [
            r'\bcorrelat(e|ion|es)\b', r'\brelat(ed|ion)\b', r'\bcaus(e|ed|ing)\b',
            r'\bassociat(e|ed|ion)\b', r'\bconnect(ed|ion)\b', r'\blink(ed)?\b'
        ],
        QueryIntent.DRILL_DOWN: [
            r'\bdetails?\b', r'\bbreakdown\b', r'\bspecific\b', r'\bdrill\b',
            r'\bdeeper\b', r'\bmore\s+info\b'
        ],
        QueryIntent.NARRATIVE_GENERATION: [
            r'\b(generate|create)\s+(a\s+)?narrative\b', r'\bpatient\s+safety\s+narrative\b',
            r'\bsafety\s+narrative\s+report\b', r'\bpatient\s+story\b', r'\bsafety\s+summary\s+narrative\b',
            r'\bclinical\s+narrative\b', r'\bnarrative\s+for\s+patient\b', r'\bgenerate\s+narrative\s+for\b'
        ],
        QueryIntent.SAFETY_CHECK: [
            r'\bsae\b', r'\bsafety\b', r'\bserious\s+adverse\b', r'\bdeath\b',
            r'\bhospitali[sz]ation\b', r'\blife.?threatening\b'
        ],
        QueryIntent.COMPLIANCE_CHECK: [
            r'\bcompliance\b', r'\bprotocol\s+deviation\b', r'\boverdue\b',
            r'\bviolation\b', r'\bnon.?conform\b'
        ],
        QueryIntent.FORECAST: [
            r'\bpredict\b', r'\bforecast\b', r'\bestimat(e|ion)\b', r'\bproject(ed|ion)?\b',
            r'\bexpect(ed)?\b', r'\bwill\s+be\b'
        ],
        QueryIntent.NARRATIVE_GENERATION: [
            r'\b(generate|create)\s+(a\s+)?narrative\b', r'\bpatient\s+safety\s+narrative\b',
            r'\bsafety\s+narrative\s+report\b', r'\bpatient\s+story\b', r'\bsafety\s+summary\s+narrative\b',
            r'\bclinical\s+narrative\b', r'\bnarrative\s+for\s+patient\b', r'\bgenerate\s+narrative\s+for\b'
        ],
        QueryIntent.RBM_REPORT: [
            r'\brbm\s+report\b', r'\bsite\s+monitoring\s+report\b', r'\bcra\s+report\b',
            r'\brisk.?based\s+monitoring\s+report\b', r'\bmonitoring\s+plan\s+report\b', 
            r'\bsite\s+visit\s+report\b', r'\b(generate|create)\s+cra\s+report\b',
            r'\b(generate|create)\s+rbm\s+report\b', r'\bcra\s+visit\s+report\b',
            r'\brbm\s+monitoring\s+report\b'
        ]
    }
    
    # Entity extraction patterns
    ENTITY_PATTERNS = {
        EntityType.SITE: [
            r'\bsites?\b', r'\bsite\s*(\d+|[A-Z]+\d*)\b', r'\bcenter\b', r'\blocation\b'
        ],
        EntityType.SUBJECT: [
            r'\bsubjects?\b', r'\bpatients?\b', r'\bparticipants?\b', r'\bsubject\s*(\d+)\b'
        ],
        EntityType.COUNTRY: [
            r'\bcountr(y|ies)\b', r'\b(US|USA|UK|FRA|GER|JPN|CAN|AUS|IND|CHN)\b',
            r'\b(United\s+States|France|Germany|Japan|Canada|Australia|India|China)\b'
        ],
        EntityType.REGION: [
            r'\bregions?\b', r'\b(EMEA|APAC|LATAM|NA|Americas?)\b'
        ],
        EntityType.VISIT: [
            r'\bvisits?\b', r'\bcycles?\b', r'\b(cycle|visit)\s*\d+\b', r'\bscreening\b',
            r'\bbaseline\b', r'\bfollow.?up\b', r'\bW\d+D\d+\b'
        ],
        EntityType.FORM: [
            r'\bforms?\b', r'\bcrfs?\b', r'\bpages?\b', r'\b[A-Z]{2,3}\d{3}\b'
        ],
        EntityType.STUDY: [
            r'\bstud(y|ies)\b', r'\btrial\b', r'\bprotocol\b'
        ]
    }
    
    # Metric extraction patterns (maps patterns to MetricType)
    METRIC_PATTERNS = {
        MetricType.MISSING_VISITS: [
            r'\bmissing\s+visits?\b', r'\bvisits?\s+missing\b', r'\bmissed\s+visits?\b'
        ],
        MetricType.MISSING_PAGES: [
            r'\bmissing\s+pages?\b', r'\bpages?\s+missing\b', r'\bmissing\s+crfs?\b'
        ],
        MetricType.OPEN_QUERIES: [
            r'\bopen\s+quer(y|ies)\b', r'\bquer(y|ies)\s+open\b', r'\boutstanding\s+quer\b',
            r'\bunresolved\s+quer\b'
        ],
        MetricType.UNCODED_TERMS: [
            r'\buncoded\b', r'\bcoding\s+pending\b', r'\bnot\s+coded\b'
        ],
        MetricType.DQI: [
            r'\bdqi\b', r'\bdata\s+quality\s+index\b', r'\bquality\s+score\b'
        ],
        MetricType.SSM: [
            r'\bssm\b', r'\bsite\s+status\b', r'\bsite\s+metric\b', r'\bred\s+sites?\b'
        ],
        MetricType.CLEAN_PATIENT_RATE: [
            r'\bclean\s+patient\b', r'\bclean\s+rate\b', r'\bclean\s+subject\b'
        ],
        MetricType.SAE_COUNT: [
            r'\bsae\s+count\b', r'\bsaes?\b', r'\bserious\s+adverse\b', r'\bsafety\s+events?\b'
        ],
        MetricType.PROTOCOL_DEVIATIONS: [
            r'\bprotocol\s+deviation\b', r'\bpds?\b', r'\bdeviation\b'
        ],
        MetricType.INACTIVATED_FORMS: [
            r'\binactivat(e|ed)\s+forms?\b', r'\binactivat(e|ed)\s+folders?\b',
            r'\binactivat(e|ed)\s+records?\b', r'\binactivation\b'
        ],
        MetricType.QUERY_AGING: [
            r'\bquery\s+ag(e|ing)\b', r'\bold\s+quer(y|ies)\b', r'\baged\s+quer\b'
        ],
        MetricType.VISIT_COMPLIANCE: [
            r'\bvisit\s+compliance\b', r'\bschedule\s+adherence\b'
        ],
        MetricType.FROZEN_CRFS: [
            r'\bfrozen\s+(crfs?|pages?|forms?)\b', r'\bcrfs?\s+frozen\b'
        ],
        MetricType.LOCKED_CRFS: [
            r'\blocked\s+(crfs?|pages?|forms?)\b', r'\bcrfs?\s+locked\b'
        ],
        MetricType.DAYS_OUTSTANDING: [
            r'\bdays?\s+outstanding\b', r'\boverdue\s+days?\b', r'\bdays?\s+overdue\b'
        ]
    }
    
    # Data source mapping
    METRIC_TO_SOURCE = {
        MetricType.MISSING_VISITS: ['CPID_EDC_Metrics', 'Visit_Projection_Tracker'],
        MetricType.MISSING_PAGES: ['CPID_EDC_Metrics', 'Missing_Pages_Report'],
        MetricType.OPEN_QUERIES: ['CPID_EDC_Metrics'],
        MetricType.UNCODED_TERMS: ['CPID_EDC_Metrics', 'GlobalCodingReport_MedDRA', 'GlobalCodingReport_WHODD'],
        MetricType.DQI: ['CPID_EDC_Metrics'],
        MetricType.SSM: ['CPID_EDC_Metrics'],
        MetricType.CLEAN_PATIENT_RATE: ['CPID_EDC_Metrics'],
        MetricType.SAE_COUNT: ['eSAE_Dashboard', 'Compiled_EDRR'],
        MetricType.PROTOCOL_DEVIATIONS: ['CPID_EDC_Metrics'],
        MetricType.INACTIVATED_FORMS: ['Inactivated_Forms_Report'],
        MetricType.QUERY_AGING: ['CPID_EDC_Metrics'],
        MetricType.VISIT_COMPLIANCE: ['Visit_Projection_Tracker'],
        MetricType.FROZEN_CRFS: ['CPID_EDC_Metrics'],
        MetricType.LOCKED_CRFS: ['CPID_EDC_Metrics'],
        MetricType.DAYS_OUTSTANDING: ['Visit_Projection_Tracker']
    }
    
    # Country codes
    COUNTRY_CODES = {
        'US': 'United States', 'USA': 'United States', 'UK': 'United Kingdom',
        'FRA': 'France', 'GER': 'Germany', 'JPN': 'Japan', 'CAN': 'Canada',
        'AUS': 'Australia', 'IND': 'India', 'CHN': 'China', 'ESP': 'Spain',
        'ITA': 'Italy', 'BRA': 'Brazil', 'MEX': 'Mexico', 'KOR': 'South Korea'
    }
    
    # Operator patterns
    OPERATOR_PATTERNS = {
        'greater_than': [r'\bgreater\s+than\b', r'\babove\b', r'\bover\b', r'\bexceeds?\b', r'\b>\b', r'\bmore\s+than\b'],
        'less_than': [r'\bless\s+than\b', r'\bbelow\b', r'\bunder\b', r'\b<\b', r'\bfewer\s+than\b'],
        'equals': [r'\bequals?\b', r'\bis\b', r'\b=\b', r'\bexactly\b'],
        'between': [r'\bbetween\b', r'\bfrom\s+\d+\s+to\s+\d+\b', r'\brange\b'],
        'trending_up': [r'\btrending\s+up\b', r'\bincreasing\b', r'\brising\b', r'\bgrowing\b'],
        'trending_down': [r'\btrending\s+down\b', r'\bdecreasing\b', r'\bfalling\b', r'\bdeclining\b']
    }
    
    def __init__(self):
        """Initialize the query parser"""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self._intent_compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        
        self._entity_compiled = {
            entity: [re.compile(p, re.IGNORECASE) for p in patterns]
            for entity, patterns in self.ENTITY_PATTERNS.items()
        }
        
        self._metric_compiled = {
            metric: [re.compile(p, re.IGNORECASE) for p in patterns]
            for metric, patterns in self.METRIC_PATTERNS.items()
        }
        
        self._operator_compiled = {
            op: [re.compile(p, re.IGNORECASE) for p in patterns]
            for op, patterns in self.OPERATOR_PATTERNS.items()
        }
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into a structured representation
        
        Args:
            query: Natural language query string
            
        Returns:
            ParsedQuery object with extracted components
        """
        parsed = ParsedQuery(original_query=query)
        
        # Normalize query
        normalized = self._normalize_query(query)
        
        # Extract intent
        parsed.intent, intent_confidence = self._extract_intent(normalized)
        parsed.parse_notes.append(f"Detected intent: {parsed.intent.name}")
        
        # Extract metrics
        metrics = self._extract_metrics(normalized)
        if metrics:
            parsed.primary_metric = metrics[0]
            parsed.secondary_metrics = metrics[1:]
            parsed.parse_notes.append(f"Primary metric: {parsed.primary_metric.value}")
        
        # Extract entity filters
        parsed.entity_filters = self._extract_entity_filters(normalized)
        for ef in parsed.entity_filters:
            parsed.parse_notes.append(f"Entity filter: {ef.entity_type.name} {ef.operator} {ef.values}")
        
        # Extract metric filters
        parsed.metric_filters = self._extract_metric_filters(normalized, metrics)
        
        # Extract time constraint
        parsed.time_constraint = self._extract_time_constraint(normalized)
        if parsed.time_constraint:
            parsed.parse_notes.append(f"Time constraint: {parsed.time_constraint.type} = {parsed.time_constraint.value}")
        
        # Extract top/bottom N
        parsed.top_n = self._extract_top_n(normalized)
        if parsed.top_n:
            parsed.parse_notes.append(f"Top N: {parsed.top_n}")
        
        # Determine group by
        parsed.group_by = self._determine_group_by(normalized, parsed.entity_filters)
        
        # Determine sort order
        parsed.sort_order = self._determine_sort_order(normalized, parsed.intent)
        
        # Map data sources
        parsed.data_sources = self._map_data_sources(metrics, parsed.entity_filters)
        parsed.parse_notes.append(f"Data sources: {parsed.data_sources}")
        
        # Calculate confidence
        parsed.confidence = self._calculate_confidence(parsed, intent_confidence)
        
        return parsed
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent parsing"""
        # Convert to lowercase for matching
        normalized = query.lower()
        
        # Standardize common variations
        replacements = [
            (r'\s+', ' '),  # Multiple spaces to single
            (r"'", "'"),    # Smart quotes to standard
            (r'"', '"'),
            (r'cycle\s+(\d+)', r'cycle \1'),  # Normalize cycle references
            (r'site\s+(\d+)', r'site \1'),     # Normalize site references
        ]
        
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized.strip()
    
    def _extract_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Extract the primary intent from the query"""
        intent_scores = {}
        
        for intent, patterns in self._intent_compiled.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(query)
                score += len(matches)
            intent_scores[intent] = score
        
        # Find highest scoring intent
        if not intent_scores or max(intent_scores.values()) == 0:
            return QueryIntent.FILTER_LIST, 0.3  # Default to list
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence = min(1.0, best_intent[1] / 3)  # Normalize score to confidence
        
        return best_intent[0], confidence
    
    def _extract_metrics(self, query: str) -> List[MetricType]:
        """Extract metrics mentioned in the query"""
        metrics = []
        
        for metric, patterns in self._metric_compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    if metric not in metrics:
                        metrics.append(metric)
                    break
        
        return metrics
    
    def _extract_entity_filters(self, query: str) -> List[EntityFilter]:
        """Extract entity filters from the query"""
        filters = []
        
        # Extract country filters
        country_pattern = re.compile(r'\b(in|from|for)\s+(?:the\s+)?([A-Z]{2,3}|United\s+States|France|Germany|Japan)\b', re.IGNORECASE)
        country_matches = country_pattern.findall(query)
        if country_matches:
            countries = []
            for _, country in country_matches:
                normalized = country.upper()
                if normalized in self.COUNTRY_CODES:
                    countries.append(normalized)
                else:
                    countries.append(country)
            if countries:
                filters.append(EntityFilter(
                    entity_type=EntityType.COUNTRY,
                    operator='in',
                    values=countries
                ))
        
        # Extract site filters
        site_pattern = re.compile(r'\bsite\s*(\d+)\b', re.IGNORECASE)
        site_matches = site_pattern.findall(query)
        if site_matches:
            filters.append(EntityFilter(
                entity_type=EntityType.SITE,
                operator='in',
                values=[f"Site {s}" for s in site_matches]
            ))
        
        # Extract visit/cycle filters
        cycle_pattern = re.compile(r'\b(?:cycle|visit)\s*(\d+|W\d+D\d+)\b', re.IGNORECASE)
        cycle_matches = cycle_pattern.findall(query)
        if cycle_matches:
            filters.append(EntityFilter(
                entity_type=EntityType.VISIT,
                operator='in',
                values=cycle_matches
            ))
        
        # Extract subject filters
        subject_pattern = re.compile(r'\bsubject\s*(\d+)\b', re.IGNORECASE)
        subject_matches = subject_pattern.findall(query)
        if subject_matches:
            filters.append(EntityFilter(
                entity_type=EntityType.SUBJECT,
                operator='in',
                values=[f"Subject {s}" for s in subject_matches]
            ))
        
        # Extract region filters
        region_pattern = re.compile(r'\b(EMEA|APAC|LATAM|NA|Americas)\b', re.IGNORECASE)
        region_matches = region_pattern.findall(query)
        if region_matches:
            filters.append(EntityFilter(
                entity_type=EntityType.REGION,
                operator='in',
                values=[r.upper() for r in region_matches]
            ))
        
        return filters
    
    def _extract_metric_filters(self, query: str, metrics: List[MetricType]) -> List[MetricFilter]:
        """Extract metric-based filters (e.g., 'rate > 10%')"""
        filters = []
        
        # Detect trending direction
        for op, patterns in self._operator_compiled.items():
            if op in ['trending_up', 'trending_down']:
                for pattern in patterns:
                    if pattern.search(query):
                        for metric in metrics:
                            filters.append(MetricFilter(
                                metric=metric,
                                operator=op,
                                value=None
                            ))
        
        # Detect numeric thresholds
        threshold_pattern = re.compile(r'(greater|less|more|fewer|over|under|above|below)\s+than\s+(\d+(?:\.\d+)?)\s*%?', re.IGNORECASE)
        threshold_matches = threshold_pattern.findall(query)
        for op_word, value in threshold_matches:
            op = 'greater_than' if op_word.lower() in ['greater', 'more', 'over', 'above'] else 'less_than'
            for metric in metrics:
                filters.append(MetricFilter(
                    metric=metric,
                    operator=op,
                    value=float(value)
                ))
        
        return filters
    
    def _extract_time_constraint(self, query: str) -> Optional[TimeConstraint]:
        """Extract temporal constraints from the query"""
        # Pattern: last N snapshots/days/weeks/months
        last_n_pattern = re.compile(r'\blast\s+(\d+)\s+(snapshot|day|week|month)s?\b', re.IGNORECASE)
        match = last_n_pattern.search(query)
        if match:
            return TimeConstraint(
                type='relative',
                value=int(match.group(1)),
                unit=match.group(2).lower() + 's'
            )
        
        # Pattern: since <date>
        since_pattern = re.compile(r'\bsince\s+(\w+\s+\d+|\d{1,2}/\d{1,2}/\d{2,4})\b', re.IGNORECASE)
        match = since_pattern.search(query)
        if match:
            return TimeConstraint(
                type='since',
                value=match.group(1)
            )
        
        # Pattern: this week/month/year
        this_period = re.compile(r'\bthis\s+(week|month|year)\b', re.IGNORECASE)
        match = this_period.search(query)
        if match:
            return TimeConstraint(
                type='current_period',
                value=match.group(1).lower()
            )
        
        return None
    
    def _extract_top_n(self, query: str) -> Optional[int]:
        """Extract top/bottom N value"""
        top_pattern = re.compile(r'\b(top|bottom|first|last)\s+(\d+)\b', re.IGNORECASE)
        match = top_pattern.search(query)
        if match:
            return int(match.group(2))
        
        # Check for implicit top (highest, worst, etc.)
        if re.search(r'\bhighest\b|\bworst\b|\bbest\b|\blowest\b', query, re.IGNORECASE):
            return 5  # Default to top 5
        
        return None
    
    def _determine_group_by(self, query: str, entity_filters: List[EntityFilter]) -> List[EntityType]:
        """Determine what entities to group results by"""
        group_by = []
        
        # If asking about sites, group by site
        if re.search(r'\bsites?\b', query, re.IGNORECASE):
            group_by.append(EntityType.SITE)
        
        # If asking about countries, group by country
        if re.search(r'\bcountr(y|ies)\b', query, re.IGNORECASE):
            group_by.append(EntityType.COUNTRY)
        
        # If asking about subjects, group by subject
        if re.search(r'\bsubjects?\b|\bpatients?\b', query, re.IGNORECASE):
            group_by.append(EntityType.SUBJECT)
        
        # Default to site if no grouping detected
        if not group_by:
            group_by.append(EntityType.SITE)
        
        return group_by
    
    def _determine_sort_order(self, query: str, intent: QueryIntent) -> str:
        """Determine sort order for results"""
        if intent == QueryIntent.BOTTOM_N:
            return 'ascending'
        
        if re.search(r'\blowest\b|\bbest\b|\bfewest\b|\bsmallest\b', query, re.IGNORECASE):
            return 'ascending'
        
        return 'descending'
    
    def _map_data_sources(self, metrics: List[MetricType], entity_filters: List[EntityFilter]) -> List[str]:
        """Map metrics and entities to required data sources"""
        sources = set()
        
        for metric in metrics:
            if metric in self.METRIC_TO_SOURCE:
                sources.update(self.METRIC_TO_SOURCE[metric])
        
        # Add default sources if none detected
        if not sources:
            sources.add('CPID_EDC_Metrics')
        
        # Add visit tracker if visit filter present
        for ef in entity_filters:
            if ef.entity_type == EntityType.VISIT:
                sources.add('Visit_Projection_Tracker')
        
        return list(sources)
    
    def _calculate_confidence(self, parsed: ParsedQuery, intent_confidence: float) -> float:
        """Calculate overall parsing confidence"""
        confidence = intent_confidence
        
        # Boost for each extracted component
        if parsed.primary_metric:
            confidence += 0.2
        if parsed.entity_filters:
            confidence += 0.15
        if parsed.time_constraint:
            confidence += 0.1
        if parsed.top_n:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def explain_parse(self, parsed: ParsedQuery) -> str:
        """Generate human-readable explanation of the parse"""
        explanation = []
        explanation.append(f"Query: \"{parsed.original_query}\"")
        explanation.append(f"\nUnderstanding:")
        explanation.append(f"  Intent: {parsed.intent.name.replace('_', ' ').title()}")
        
        if parsed.primary_metric:
            explanation.append(f"  Primary Metric: {parsed.primary_metric.value.replace('_', ' ').title()}")
        
        if parsed.entity_filters:
            explanation.append(f"  Filters:")
            for ef in parsed.entity_filters:
                explanation.append(f"    - {ef.entity_type.name}: {ef.operator} {ef.values}")
        
        if parsed.time_constraint:
            explanation.append(f"  Time: {parsed.time_constraint.type} = {parsed.time_constraint.value} {parsed.time_constraint.unit or ''}")
        
        if parsed.data_sources:
            explanation.append(f"  Data Sources: {', '.join(parsed.data_sources)}")
        
        explanation.append(f"\n  Confidence: {parsed.confidence:.1%}")
        
        return "\n".join(explanation)
