"""
Ambiguous Coding Detector - Scenario B: The "Ambiguous" Concomitant Medication

This module implements the detection and handling of ambiguous verbatim terms
in GlobalCodingReport_WHODRA that are too vague for WHO Drug coding.

Problem: Sites enter vague medication names like "Pain killer" which cannot
be coded in WHO Drug dictionary. Human coders manually create queries and
wait weeks for site clarification.

Agentic Solution Pipeline:
1. Data Check: Scan GlobalCodingReport_WHODRA for Coding Status = "UnCoded Term"
2. LLM Query: Assess if term is specific enough for WHODRA coding
3. Reasoning: Evaluate confidence (Low/Medium/High)
4. Action: 
   - High Confidence (>95%): Auto-apply code
   - Medium Confidence (80-95%): Propose code for approval
   - Low Confidence (<80%): Trigger Clarification Workflow
5. Learning: Track site clarifications for probability weight updates

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


class AmbiguityLevel(Enum):
    """Classification of term ambiguity level"""
    SPECIFIC = auto()      # Clear trade name or generic name
    MODERATE = auto()      # Recognizable but may need verification
    AMBIGUOUS = auto()     # Drug class or vague descriptor
    ILLEGIBLE = auto()     # Non-standard or unreadable
    UNKNOWN = auto()       # Not found in any reference


class CodingConfidence(Enum):
    """Confidence levels for coding decisions (Human-in-the-Loop Protocol)"""
    HIGH = auto()          # >95% - Auto-apply code
    MEDIUM = auto()        # 80-95% - Propose for single-click approval
    LOW = auto()           # <80% - Request site clarification


class ClarificationReason(Enum):
    """Reasons for requesting site clarification"""
    DRUG_CLASS_NOT_SPECIFIC = auto()      # e.g., "pain killer", "antibiotic"
    BRAND_REGIONAL_UNKNOWN = auto()        # Regional brand not in dictionary
    ABBREVIATION_UNCLEAR = auto()          # e.g., "ASA", "MVI"
    MISSPELLING_POSSIBLE = auto()          # Possible typo
    INCOMPLETE_ENTRY = auto()              # Partial term
    ILLEGIBLE_TEXT = auto()                # Cannot be read
    MULTIPLE_MATCHES = auto()              # Could map to several drugs
    NO_MATCH_FOUND = auto()                # Not in WHO Drug dictionary


@dataclass
class AmbiguousTerm:
    """Represents an ambiguous term requiring clarification"""
    subject_id: str
    site_id: str
    form_oid: str
    field_oid: str
    logline: int
    verbatim_term: str
    ambiguity_level: AmbiguityLevel
    confidence: CodingConfidence
    confidence_score: float
    reason: ClarificationReason
    suggested_matches: List[Dict[str, Any]] = field(default_factory=list)
    llm_assessment: str = ""
    auto_query: str = ""
    detection_timestamp: datetime = field(default_factory=datetime.now)
    days_pending: int = 0
    clarification_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AmbiguousCodingConfig:
    """Configuration for ambiguous coding detection"""
    # Confidence thresholds
    high_confidence_threshold: float = 0.95
    medium_confidence_threshold: float = 0.80
    
    # Ambiguous term patterns (drug classes, vague descriptors)
    ambiguous_drug_classes: List[str] = field(default_factory=lambda: [
        'pain killer', 'painkiller', 'pain medication', 'pain med',
        'antibiotic', 'antibiotics', 'abx',
        'blood pressure medication', 'bp medication', 'bp med',
        'blood thinner', 'anticoagulant',
        'diabetes medication', 'diabetes med', 'sugar pill',
        'heart medication', 'heart med', 'cardiac medication',
        'sleeping pill', 'sleep aid', 'sleep medication',
        'anxiety medication', 'anti-anxiety', 'anxiety med',
        'depression medication', 'antidepressant',
        'allergy medication', 'antihistamine', 'allergy pill',
        'cough medicine', 'cough syrup', 'cold medicine',
        'stomach medication', 'antacid', 'acid reflux med',
        'steroid', 'corticosteroid',
        'supplement', 'vitamin', 'herbal', 'natural remedy',
        'traditional medicine', 'home remedy', 'homeopathic',
        'over the counter', 'otc', 'otc medication',
        'prescription', 'rx', 'prescribed medication',
        'eye drops', 'ear drops', 'nasal spray',
        'cream', 'ointment', 'lotion', 'topical',
        'injection', 'injectable', 'shot'
    ])
    
    # Patterns indicating incomplete or illegible entries
    illegible_patterns: List[str] = field(default_factory=lambda: [
        r'^[a-z]{1,2}$',           # 1-2 letters only
        r'^\d+$',                   # Numbers only
        r'^[^a-zA-Z]+$',           # No letters
        r'.*\?\?.*',               # Contains ??
        r'.*illegible.*',
        r'.*unclear.*',
        r'.*unreadable.*',
        r'.*unknown.*',
        r'^n/?a$',
        r'^none$',
        r'^nil$',
        r'^\-+$',                   # Dashes only
        r'^\s*$'                    # Whitespace only
    ])
    
    # Common abbreviations that need clarification
    ambiguous_abbreviations: Dict[str, List[str]] = field(default_factory=lambda: {
        'asa': ['Aspirin', 'Aminosalicylic acid'],
        'mvi': ['Multivitamin', 'Multiple vitamins injection'],
        'pn': ['Penicillin', 'Pain', 'Pro re nata'],
        'abx': ['Amoxicillin', 'Antibiotics general'],
        'bp': ['Blood pressure medication', 'Bisphosphonate'],
        'nsaid': ['Ibuprofen', 'Naproxen', 'Aspirin', 'Celecoxib'],
        'ppi': ['Omeprazole', 'Pantoprazole', 'Esomeprazole'],
        'ssri': ['Sertraline', 'Fluoxetine', 'Citalopram'],
        'ace': ['Lisinopril', 'Enalapril', 'Ramipril'],
        'arb': ['Losartan', 'Valsartan', 'Irbesartan'],
        'bb': ['Metoprolol', 'Atenolol', 'Bisoprolol'],
        'ccb': ['Amlodipine', 'Diltiazem', 'Nifedipine']
    })
    
    # Query templates for site clarification
    query_templates: Dict[str, str] = field(default_factory=lambda: {
        'drug_class': (
            "Term '{verbatim}' is a drug class, not a specific medication. "
            "Please provide the specific Trade Name or Generic Name "
            "(e.g., for 'pain killer': Paracetamol, Ibuprofen, Aspirin). "
            "[Reference: WHO Drug Dictionary coding requirement]"
        ),
        'abbreviation': (
            "Abbreviation '{verbatim}' could refer to multiple medications: {options}. "
            "Please clarify the exact Trade Name or Generic Name for Subject {subject_id}."
        ),
        'incomplete': (
            "Term '{verbatim}' appears incomplete. "
            "Please provide the full medication name for accurate WHODRA coding."
        ),
        'illegible': (
            "Term '{verbatim}' in the Concomitant Medication form is illegible or unclear. "
            "Please clarify the intended medication name for Subject {subject_id}."
        ),
        'no_match': (
            "Term '{verbatim}' not found in WHO Drug Dictionary. "
            "Please verify spelling or provide the standard Trade Name/Generic Name."
        ),
        'multiple_match': (
            "Term '{verbatim}' matches multiple entries in WHO Drug Dictionary: {options}. "
            "Please confirm which medication was administered to Subject {subject_id}."
        )
    })
    
    # Days thresholds for escalation
    days_warning_threshold: int = 7
    days_critical_threshold: int = 14
    
    # Learning - track term clarifications
    enable_learning: bool = True


# Default configuration
DEFAULT_AMBIGUOUS_CODING_CONFIG = AmbiguousCodingConfig()


class AmbiguousCodingDetector:
    """
    Detects and handles ambiguous concomitant medication terms in WHODRA coding.
    
    Implements the 5-step agentic solution:
    1. Data Check: Scan for Coding Status = "UnCoded Term"
    2. LLM Query: Assess term specificity
    3. Reasoning: Classify confidence level
    4. Action: Auto-code, propose, or request clarification
    5. Learning: Update probability weights based on resolutions
    """
    
    def __init__(self, config: AmbiguousCodingConfig = None):
        self.config = config or DEFAULT_AMBIGUOUS_CODING_CONFIG
        self.ambiguous_terms: List[AmbiguousTerm] = []
        self._learning_cache: Dict[str, Dict[str, Any]] = {}  # verbatim -> resolution
        self._detection_stats = {
            'total_uncoded': 0,
            'auto_codable': 0,
            'proposed': 0,
            'clarification_needed': 0,
            'by_reason': {}
        }
        
        # WHO Drug reference dictionary (simulated LLM knowledge)
        self._whodrug_reference = self._build_whodrug_reference()
        
        logger.info(f"AmbiguousCodingDetector initialized with config")
    
    def _build_whodrug_reference(self) -> Dict[str, Dict[str, Any]]:
        """Build WHO Drug reference dictionary for term matching"""
        return {
            # Common pain medications
            'paracetamol': {'trade_names': ['Tylenol', 'Panadol', 'Calpol'], 'atc': 'N02BE01', 'generic': 'Paracetamol'},
            'acetaminophen': {'trade_names': ['Tylenol', 'Panadol'], 'atc': 'N02BE01', 'generic': 'Paracetamol'},
            'tylenol': {'trade_names': ['Tylenol'], 'atc': 'N02BE01', 'generic': 'Paracetamol'},
            'ibuprofen': {'trade_names': ['Advil', 'Motrin', 'Brufen'], 'atc': 'M01AE01', 'generic': 'Ibuprofen'},
            'advil': {'trade_names': ['Advil'], 'atc': 'M01AE01', 'generic': 'Ibuprofen'},
            'aspirin': {'trade_names': ['Aspirin', 'Bayer'], 'atc': 'B01AC06', 'generic': 'Acetylsalicylic acid'},
            'naproxen': {'trade_names': ['Aleve', 'Naprosyn'], 'atc': 'M01AE02', 'generic': 'Naproxen'},
            
            # Blood pressure medications
            'lisinopril': {'trade_names': ['Zestril', 'Prinivil'], 'atc': 'C09AA03', 'generic': 'Lisinopril'},
            'amlodipine': {'trade_names': ['Norvasc'], 'atc': 'C08CA01', 'generic': 'Amlodipine'},
            'losartan': {'trade_names': ['Cozaar'], 'atc': 'C09CA01', 'generic': 'Losartan'},
            'metoprolol': {'trade_names': ['Lopressor', 'Toprol'], 'atc': 'C07AB02', 'generic': 'Metoprolol'},
            
            # Diabetes medications
            'metformin': {'trade_names': ['Glucophage'], 'atc': 'A10BA02', 'generic': 'Metformin'},
            'glipizide': {'trade_names': ['Glucotrol'], 'atc': 'A10BB07', 'generic': 'Glipizide'},
            
            # Cholesterol medications
            'atorvastatin': {'trade_names': ['Lipitor'], 'atc': 'C10AA05', 'generic': 'Atorvastatin'},
            'simvastatin': {'trade_names': ['Zocor'], 'atc': 'C10AA01', 'generic': 'Simvastatin'},
            
            # GI medications
            'omeprazole': {'trade_names': ['Prilosec', 'Losec'], 'atc': 'A02BC01', 'generic': 'Omeprazole'},
            'pantoprazole': {'trade_names': ['Protonix'], 'atc': 'A02BC02', 'generic': 'Pantoprazole'},
            
            # Antibiotics
            'amoxicillin': {'trade_names': ['Amoxil'], 'atc': 'J01CA04', 'generic': 'Amoxicillin'},
            'azithromycin': {'trade_names': ['Zithromax', 'Z-pack'], 'atc': 'J01FA10', 'generic': 'Azithromycin'},
            'ciprofloxacin': {'trade_names': ['Cipro'], 'atc': 'J01MA02', 'generic': 'Ciprofloxacin'},
            
            # Mental health medications
            'sertraline': {'trade_names': ['Zoloft'], 'atc': 'N06AB06', 'generic': 'Sertraline'},
            'fluoxetine': {'trade_names': ['Prozac'], 'atc': 'N06AB03', 'generic': 'Fluoxetine'},
            'alprazolam': {'trade_names': ['Xanax'], 'atc': 'N05BA12', 'generic': 'Alprazolam'},
            
            # Common supplements
            'multivitamin': {'trade_names': ['Centrum', 'One-A-Day'], 'atc': 'A11AA03', 'generic': 'Multivitamins'},
            'vitamin d': {'trade_names': ['D3', 'Cholecalciferol'], 'atc': 'A11CC05', 'generic': 'Colecalciferol'},
            'fish oil': {'trade_names': ['Lovaza'], 'atc': 'C10AX06', 'generic': 'Omega-3-triglycerides'},
        }
    
    def detect(
        self,
        whodra_data: pd.DataFrame,
        cpid_data: Optional[pd.DataFrame] = None,
        study_id: str = ""
    ) -> List[AmbiguousTerm]:
        """
        Main detection pipeline for ambiguous WHODRA terms.
        
        Args:
            whodra_data: GlobalCodingReport_WHODRA DataFrame
            cpid_data: CPID_EDC_Metrics DataFrame for context
            study_id: Study identifier
            
        Returns:
            List of AmbiguousTerm objects requiring attention
        """
        self.ambiguous_terms = []
        self._reset_stats()
        
        logger.info(f"Starting ambiguous coding detection for study {study_id}")
        
        # Step 1: Data Check - Find uncoded terms
        uncoded_rows = self._scan_for_uncoded_terms(whodra_data)
        self._detection_stats['total_uncoded'] = len(uncoded_rows)
        
        logger.info(f"Found {len(uncoded_rows)} uncoded terms to analyze")
        
        # Build site context from CPID if available
        site_context = self._build_site_context(cpid_data) if cpid_data is not None else {}
        
        # Process each uncoded term
        for _, row in uncoded_rows.iterrows():
            term = self._analyze_term(row, site_context, study_id)
            if term:
                self.ambiguous_terms.append(term)
                self._update_stats(term)
        
        logger.info(f"Detection complete: {self._detection_stats}")
        return self.ambiguous_terms
    
    def _reset_stats(self):
        """Reset detection statistics"""
        self._detection_stats = {
            'total_uncoded': 0,
            'auto_codable': 0,
            'proposed': 0,
            'clarification_needed': 0,
            'by_reason': {}
        }
    
    def _scan_for_uncoded_terms(self, whodra_data: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Data Check - Scan GlobalCodingReport_WHODRA for uncoded terms.
        Filter for rows where Coding Status = "UnCoded Term"
        """
        # Handle duplicate columns
        if whodra_data.columns.duplicated().any():
            whodra_data = whodra_data.loc[:, ~whodra_data.columns.duplicated()]
        
        # Find coding status column
        status_col = None
        for col in ['Coding Status', 'coding_status', 'CodingStatus']:
            if col in whodra_data.columns:
                status_col = col
                break
        
        if status_col is None:
            logger.warning("No Coding Status column found in WHODRA data")
            return pd.DataFrame()
        
        # Filter for uncoded terms
        uncoded_mask = whodra_data[status_col].astype(str).str.lower().str.contains(
            'uncoded|un-coded|not coded', na=False
        )
        
        return whodra_data[uncoded_mask].copy()
    
    def _build_site_context(self, cpid_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Build site context from CPID for subject lookup"""
        context = {}
        
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
            
            # Find subject and site columns
            subject_col = None
            site_col = None
            for col in cpid_flat.columns:
                col_lower = str(col).lower()
                if 'subject' in col_lower and 'id' in col_lower:
                    subject_col = col
                elif 'site' in col_lower and 'id' in col_lower:
                    site_col = col
            
            # Build context dictionary
            if subject_col and site_col:
                for _, row in cpid_flat.iterrows():
                    subj = str(row.get(subject_col, ''))
                    site = str(row.get(site_col, ''))
                    if subj and site:
                        context[subj] = {'site_id': site}
        except Exception as e:
            logger.warning(f"Error building site context: {e}")
        
        return context
    
    def _analyze_term(
        self,
        row: pd.Series,
        site_context: Dict[str, Dict[str, Any]],
        study_id: str
    ) -> Optional[AmbiguousTerm]:
        """
        Steps 2-4: Analyze a single uncoded term.
        
        2. LLM Query: Assess term specificity
        3. Reasoning: Classify confidence level
        4. Action: Determine appropriate response
        """
        # Extract row data
        subject_id = str(row.get('Subject', ''))
        form_oid = str(row.get('Form OID', ''))
        field_oid = str(row.get('Field OID', ''))
        logline = int(row.get('Logline', 0)) if pd.notna(row.get('Logline')) else 0
        
        # Get site ID from context
        site_id = site_context.get(subject_id, {}).get('site_id', 'Unknown')
        
        # For real data, we need to extract verbatim term from a different source
        # In this simulated scenario, we generate a placeholder
        verbatim = self._extract_verbatim_term(row, field_oid)
        
        if not verbatim:
            return None
        
        # Step 2: LLM Query - Assess term specificity
        assessment = self._assess_term_specificity(verbatim)
        
        # Step 3: Reasoning - Classify confidence and ambiguity
        ambiguity_level, confidence, confidence_score, reason = self._classify_term(
            verbatim, assessment
        )
        
        # Step 4: Action - Generate auto query if clarification needed
        auto_query = ""
        if confidence == CodingConfidence.LOW:
            auto_query = self._generate_clarification_query(
                verbatim, subject_id, site_id, reason, assessment.get('suggestions', [])
            )
        
        return AmbiguousTerm(
            subject_id=subject_id,
            site_id=site_id,
            form_oid=form_oid,
            field_oid=field_oid,
            logline=logline,
            verbatim_term=verbatim,
            ambiguity_level=ambiguity_level,
            confidence=confidence,
            confidence_score=confidence_score,
            reason=reason,
            suggested_matches=assessment.get('suggestions', []),
            llm_assessment=assessment.get('assessment', ''),
            auto_query=auto_query
        )
    
    def _extract_verbatim_term(self, row: pd.Series, field_oid: str) -> str:
        """
        Extract verbatim term from coding report row.
        In real implementation, this would link to EDC data.
        For simulation, we use field patterns or generate test terms.
        """
        # Check if there's a verbatim column
        for col in ['Verbatim Term', 'verbatim_term', 'VerbatimTerm', 'CMTRT']:
            if col in row.index and pd.notna(row.get(col)):
                return str(row[col]).strip()
        
        # Simulate verbatim terms for testing (based on field OID pattern)
        # In production, this would be fetched from EDC
        simulated_terms = [
            'pain killer', 'antibiotic', 'blood pressure pill', 'vitamin',
            'sleeping pill', 'heart medication', 'stomach medicine',
            'steroid', 'herbal supplement', 'anxiety medication'
        ]
        
        import hashlib
        # Use row data to create deterministic but varied terms
        hash_input = f"{row.get('Subject', '')}{row.get('Logline', '')}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        return simulated_terms[hash_val % len(simulated_terms)]
    
    def _assess_term_specificity(self, verbatim: str) -> Dict[str, Any]:
        """
        Step 2: LLM Query - Simulate LLM assessment of term specificity.
        
        Returns assessment with:
        - is_specific: Boolean indicating if term is codable
        - confidence: Float confidence score
        - assessment: String explanation
        - suggestions: List of possible matches
        """
        verbatim_lower = verbatim.lower().strip()
        
        # Check learning cache first
        if verbatim_lower in self._learning_cache:
            cached = self._learning_cache[verbatim_lower]
            return {
                'is_specific': True,
                'confidence': 0.90,
                'assessment': f"Previously resolved to: {cached.get('resolved_term', verbatim)}",
                'suggestions': [cached]
            }
        
        # Check against WHO Drug reference
        if verbatim_lower in self._whodrug_reference:
            ref = self._whodrug_reference[verbatim_lower]
            return {
                'is_specific': True,
                'confidence': 0.98,
                'assessment': f"Exact match in WHO Drug Dictionary: {ref['generic']}",
                'suggestions': [{
                    'generic': ref['generic'],
                    'trade_names': ref['trade_names'],
                    'atc': ref['atc'],
                    'confidence': 0.98
                }]
            }
        
        # Check for partial matches
        partial_matches = []
        for term, ref in self._whodrug_reference.items():
            if verbatim_lower in term or term in verbatim_lower:
                partial_matches.append({
                    'generic': ref['generic'],
                    'trade_names': ref['trade_names'],
                    'atc': ref['atc'],
                    'confidence': 0.85,
                    'matched_on': term
                })
        
        if partial_matches:
            if len(partial_matches) == 1:
                return {
                    'is_specific': True,
                    'confidence': 0.85,
                    'assessment': f"Partial match found: {partial_matches[0]['generic']}",
                    'suggestions': partial_matches
                }
            else:
                return {
                    'is_specific': False,
                    'confidence': 0.60,
                    'assessment': f"Multiple partial matches ({len(partial_matches)}). Clarification needed.",
                    'suggestions': partial_matches
                }
        
        # Check for ambiguous drug classes
        for drug_class in self.config.ambiguous_drug_classes:
            if drug_class in verbatim_lower or verbatim_lower in drug_class:
                return {
                    'is_specific': False,
                    'confidence': 0.30,
                    'assessment': f"Term is a drug class, not specific medication. Low confidence.",
                    'suggestions': self._get_class_examples(drug_class)
                }
        
        # Check for abbreviations
        for abbrev, options in self.config.ambiguous_abbreviations.items():
            if verbatim_lower == abbrev:
                return {
                    'is_specific': False,
                    'confidence': 0.40,
                    'assessment': f"Abbreviation '{abbrev}' could refer to: {', '.join(options)}",
                    'suggestions': [{'generic': opt, 'confidence': 0.40} for opt in options]
                }
        
        # Check for illegible patterns
        for pattern in self.config.illegible_patterns:
            if re.match(pattern, verbatim_lower, re.IGNORECASE):
                return {
                    'is_specific': False,
                    'confidence': 0.0,
                    'assessment': "Term is illegible or incomplete.",
                    'suggestions': []
                }
        
        # Unknown term - not in dictionary
        return {
            'is_specific': False,
            'confidence': 0.50,
            'assessment': f"Term '{verbatim}' not found in WHO Drug Dictionary. May be regional brand or misspelling.",
            'suggestions': []
        }
    
    def _get_class_examples(self, drug_class: str) -> List[Dict[str, Any]]:
        """Get example medications for a drug class"""
        class_examples = {
            'pain': ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Naproxen'],
            'antibiotic': ['Amoxicillin', 'Azithromycin', 'Ciprofloxacin'],
            'blood pressure': ['Lisinopril', 'Amlodipine', 'Losartan', 'Metoprolol'],
            'diabetes': ['Metformin', 'Glipizide', 'Insulin'],
            'sleeping': ['Zolpidem', 'Eszopiclone', 'Melatonin'],
            'anxiety': ['Alprazolam', 'Lorazepam', 'Diazepam'],
            'stomach': ['Omeprazole', 'Pantoprazole', 'Famotidine'],
            'heart': ['Aspirin', 'Metoprolol', 'Lisinopril', 'Atorvastatin'],
            'steroid': ['Prednisone', 'Dexamethasone', 'Hydrocortisone'],
            'vitamin': ['Multivitamin', 'Vitamin D', 'Vitamin B12', 'Folic acid']
        }
        
        for key, examples in class_examples.items():
            if key in drug_class.lower():
                return [{'generic': ex, 'confidence': 0.30} for ex in examples]
        
        return [{'generic': 'Unknown', 'confidence': 0.0}]
    
    def _classify_term(
        self,
        verbatim: str,
        assessment: Dict[str, Any]
    ) -> Tuple[AmbiguityLevel, CodingConfidence, float, ClarificationReason]:
        """
        Step 3: Reasoning - Classify term ambiguity and confidence.
        
        Returns:
            - AmbiguityLevel: Specific, Moderate, Ambiguous, Illegible, Unknown
            - CodingConfidence: High (>95%), Medium (80-95%), Low (<80%)
            - confidence_score: Float 0-1
            - ClarificationReason: Why clarification is needed
        """
        verbatim_lower = verbatim.lower().strip()
        confidence_score = assessment.get('confidence', 0.0)
        
        # Check for illegible patterns
        for pattern in self.config.illegible_patterns:
            if re.match(pattern, verbatim_lower, re.IGNORECASE):
                return (
                    AmbiguityLevel.ILLEGIBLE,
                    CodingConfidence.LOW,
                    0.0,
                    ClarificationReason.ILLEGIBLE_TEXT
                )
        
        # Determine ambiguity level based on assessment
        if assessment.get('is_specific') and confidence_score >= self.config.high_confidence_threshold:
            return (
                AmbiguityLevel.SPECIFIC,
                CodingConfidence.HIGH,
                confidence_score,
                ClarificationReason.NO_MATCH_FOUND  # Not really needed for high confidence
            )
        
        elif assessment.get('is_specific') and confidence_score >= self.config.medium_confidence_threshold:
            return (
                AmbiguityLevel.MODERATE,
                CodingConfidence.MEDIUM,
                confidence_score,
                ClarificationReason.MULTIPLE_MATCHES if len(assessment.get('suggestions', [])) > 1 
                else ClarificationReason.MISSPELLING_POSSIBLE
            )
        
        else:
            # Low confidence - determine specific reason
            suggestions = assessment.get('suggestions', [])
            
            # Check if it's a drug class
            for drug_class in self.config.ambiguous_drug_classes:
                if drug_class in verbatim_lower or verbatim_lower in drug_class:
                    return (
                        AmbiguityLevel.AMBIGUOUS,
                        CodingConfidence.LOW,
                        confidence_score,
                        ClarificationReason.DRUG_CLASS_NOT_SPECIFIC
                    )
            
            # Check if it's an abbreviation
            if verbatim_lower in self.config.ambiguous_abbreviations:
                return (
                    AmbiguityLevel.AMBIGUOUS,
                    CodingConfidence.LOW,
                    confidence_score,
                    ClarificationReason.ABBREVIATION_UNCLEAR
                )
            
            # Multiple matches
            if len(suggestions) > 1:
                return (
                    AmbiguityLevel.AMBIGUOUS,
                    CodingConfidence.LOW,
                    confidence_score,
                    ClarificationReason.MULTIPLE_MATCHES
                )
            
            # No match found
            if not suggestions:
                return (
                    AmbiguityLevel.UNKNOWN,
                    CodingConfidence.LOW,
                    confidence_score,
                    ClarificationReason.NO_MATCH_FOUND
                )
            
            # Default to incomplete
            return (
                AmbiguityLevel.AMBIGUOUS,
                CodingConfidence.LOW,
                confidence_score,
                ClarificationReason.INCOMPLETE_ENTRY
            )
    
    def _generate_clarification_query(
        self,
        verbatim: str,
        subject_id: str,
        site_id: str,
        reason: ClarificationReason,
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """
        Step 4: Action - Generate auto-drafted query for site clarification.
        """
        # Select appropriate template based on reason
        if reason == ClarificationReason.DRUG_CLASS_NOT_SPECIFIC:
            template = self.config.query_templates['drug_class']
            examples = ', '.join([s.get('generic', '') for s in suggestions[:3]]) if suggestions else 'Specific Trade/Generic Name'
            return template.format(verbatim=verbatim).replace('(e.g., for \'pain killer\': Paracetamol, Ibuprofen, Aspirin)', f'(e.g., {examples})')
        
        elif reason == ClarificationReason.ABBREVIATION_UNCLEAR:
            template = self.config.query_templates['abbreviation']
            options = ', '.join([s.get('generic', '') for s in suggestions[:5]]) if suggestions else 'multiple medications'
            return template.format(verbatim=verbatim, options=options, subject_id=subject_id)
        
        elif reason == ClarificationReason.ILLEGIBLE_TEXT:
            template = self.config.query_templates['illegible']
            return template.format(verbatim=verbatim, subject_id=subject_id)
        
        elif reason == ClarificationReason.NO_MATCH_FOUND:
            template = self.config.query_templates['no_match']
            return template.format(verbatim=verbatim)
        
        elif reason == ClarificationReason.MULTIPLE_MATCHES:
            template = self.config.query_templates['multiple_match']
            options = ', '.join([s.get('generic', '') for s in suggestions[:5]]) if suggestions else 'multiple entries'
            return template.format(verbatim=verbatim, options=options, subject_id=subject_id)
        
        else:
            template = self.config.query_templates['incomplete']
            return template.format(verbatim=verbatim)
    
    def _update_stats(self, term: AmbiguousTerm):
        """Update detection statistics"""
        if term.confidence == CodingConfidence.HIGH:
            self._detection_stats['auto_codable'] += 1
        elif term.confidence == CodingConfidence.MEDIUM:
            self._detection_stats['proposed'] += 1
        else:
            self._detection_stats['clarification_needed'] += 1
        
        reason_name = term.reason.name
        self._detection_stats['by_reason'][reason_name] = \
            self._detection_stats['by_reason'].get(reason_name, 0) + 1
    
    def learn_from_resolution(
        self,
        verbatim: str,
        resolved_term: str,
        generic_name: str,
        trade_name: str = "",
        atc_code: str = ""
    ):
        """
        Step 5: Learning - Update probability weights from site clarification.
        
        When a site responds with clarification (e.g., "Pain killer" -> "Advil"),
        store this for future reference.
        """
        if not self.config.enable_learning:
            return
        
        verbatim_lower = verbatim.lower().strip()
        self._learning_cache[verbatim_lower] = {
            'resolved_term': resolved_term,
            'generic_name': generic_name,
            'trade_name': trade_name,
            'atc_code': atc_code,
            'resolution_date': datetime.now().isoformat(),
            'times_seen': self._learning_cache.get(verbatim_lower, {}).get('times_seen', 0) + 1
        }
        
        logger.info(f"Learning: '{verbatim}' -> '{resolved_term}' (generic: {generic_name})")
    
    def get_clarification_queries(self) -> List[Dict[str, Any]]:
        """Get all auto-generated clarification queries"""
        queries = []
        for term in self.ambiguous_terms:
            if term.confidence == CodingConfidence.LOW and term.auto_query:
                queries.append({
                    'subject_id': term.subject_id,
                    'site_id': term.site_id,
                    'form_oid': term.form_oid,
                    'field_oid': term.field_oid,
                    'logline': term.logline,
                    'verbatim_term': term.verbatim_term,
                    'reason': term.reason.name,
                    'query_text': term.auto_query,
                    'confidence_score': term.confidence_score,
                    'suggested_matches': term.suggested_matches,
                    'detection_timestamp': term.detection_timestamp.isoformat()
                })
        return queries
    
    def get_proposed_codes(self) -> List[Dict[str, Any]]:
        """Get terms with medium confidence for single-click approval"""
        proposed = []
        for term in self.ambiguous_terms:
            if term.confidence == CodingConfidence.MEDIUM:
                proposed.append({
                    'subject_id': term.subject_id,
                    'site_id': term.site_id,
                    'form_oid': term.form_oid,
                    'field_oid': term.field_oid,
                    'logline': term.logline,
                    'verbatim_term': term.verbatim_term,
                    'confidence_score': term.confidence_score,
                    'suggested_code': term.suggested_matches[0] if term.suggested_matches else {},
                    'llm_assessment': term.llm_assessment
                })
        return proposed
    
    def get_auto_codable(self) -> List[Dict[str, Any]]:
        """Get terms with high confidence for automatic coding"""
        auto_code = []
        for term in self.ambiguous_terms:
            if term.confidence == CodingConfidence.HIGH:
                auto_code.append({
                    'subject_id': term.subject_id,
                    'site_id': term.site_id,
                    'form_oid': term.form_oid,
                    'field_oid': term.field_oid,
                    'logline': term.logline,
                    'verbatim_term': term.verbatim_term,
                    'confidence_score': term.confidence_score,
                    'code': term.suggested_matches[0] if term.suggested_matches else {},
                    'llm_assessment': term.llm_assessment
                })
        return auto_code
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of ambiguous coding detection"""
        return {
            'detection_timestamp': datetime.now().isoformat(),
            'statistics': self._detection_stats,
            'breakdown': {
                'high_confidence_auto_code': len(self.get_auto_codable()),
                'medium_confidence_proposed': len(self.get_proposed_codes()),
                'low_confidence_clarification': len(self.get_clarification_queries())
            },
            'by_ambiguity_level': {
                level.name: len([t for t in self.ambiguous_terms if t.ambiguity_level == level])
                for level in AmbiguityLevel
            },
            'by_site': self._group_by_site(),
            'by_reason': self._detection_stats.get('by_reason', {}),
            'learning_cache_size': len(self._learning_cache)
        }
    
    def _group_by_site(self) -> Dict[str, int]:
        """Group ambiguous terms by site"""
        by_site = {}
        for term in self.ambiguous_terms:
            site = term.site_id
            by_site[site] = by_site.get(site, 0) + 1
        return by_site
