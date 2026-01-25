"""
AI-Powered Generative Narrative Engine
======================================

Advanced narrative generation using LLM for dynamic, contextual clinical reports.
Replaces static templates with AI-generated, regulatory-safe content.

Features:
- LLM-powered narrative synthesis
- Multi-source data integration
- Regulatory compliance validation
- Customizable report types
- Real-time generation with streaming support
"""

import json
import logging
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import LLM integration
try:
    from agents.llm_integration import (
        AgentReasoningEngine,
        get_reasoning_engine,
        LLMClientFactory,
        LLMConfig,
        LLMProvider,
        PromptTemplates
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM integration not available for narrative generation")


class NarrativeType(Enum):
    """Types of narratives that can be generated"""
    PATIENT_SAFETY = "patient_safety"      # For Medical Monitors
    SITE_PERFORMANCE = "site_performance"  # For CRAs
    STUDY_SUMMARY = "study_summary"        # For Study Managers
    REGULATORY_REPORT = "regulatory"       # For regulatory submissions
    DATA_QUALITY = "data_quality"          # For Data Management
    EXECUTIVE_SUMMARY = "executive"        # For leadership


class NarrativeSeverity(Enum):
    """Severity levels for narratives"""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INFORMATIONAL = "info"


@dataclass
class NarrativeContext:
    """Context for narrative generation"""
    subject_id: Optional[str] = None
    site_id: Optional[str] = None
    study_id: Optional[str] = None
    narrative_type: NarrativeType = NarrativeType.PATIENT_SAFETY
    time_period: Optional[str] = None
    audience: str = "Medical Monitor"
    include_recommendations: bool = True
    include_data_sources: bool = True
    max_length: int = 2000
    regulatory_safe: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'site_id': self.site_id,
            'study_id': self.study_id,
            'narrative_type': self.narrative_type.value,
            'time_period': self.time_period,
            'audience': self.audience,
            'include_recommendations': self.include_recommendations,
            'include_data_sources': self.include_data_sources,
            'max_length': self.max_length,
            'regulatory_safe': self.regulatory_safe
        }


@dataclass
class GeneratedNarrative:
    """Result of narrative generation"""
    narrative_id: str
    narrative_type: NarrativeType
    context: NarrativeContext
    title: str
    content: str
    severity: NarrativeSeverity
    key_findings: List[str]
    recommendations: List[str]
    data_sources: List[str]
    generated_at: datetime
    generation_time_ms: float
    model_used: str
    confidence: float
    regulatory_validated: bool
    
    def to_dict(self) -> Dict:
        return {
            'narrative_id': self.narrative_id,
            'narrative_type': self.narrative_type.value,
            'context': self.context.to_dict(),
            'title': self.title,
            'content': self.content,
            'severity': self.severity.value,
            'key_findings': self.key_findings,
            'recommendations': self.recommendations,
            'data_sources': self.data_sources,
            'generated_at': self.generated_at.isoformat(),
            'generation_time_ms': self.generation_time_ms,
            'model_used': self.model_used,
            'confidence': self.confidence,
            'regulatory_validated': self.regulatory_validated
        }
    
    def to_markdown(self) -> str:
        """Convert narrative to markdown format"""
        severity_icons = {
            NarrativeSeverity.CRITICAL: "🔴",
            NarrativeSeverity.HIGH: "🟠",
            NarrativeSeverity.MODERATE: "🟡",
            NarrativeSeverity.LOW: "🟢",
            NarrativeSeverity.INFORMATIONAL: "ℹ️"
        }
        
        lines = [
            f"# {self.title}",
            f"**Type:** {self.narrative_type.value.replace('_', ' ').title()}",
            f"**Severity:** {severity_icons.get(self.severity, '')} {self.severity.value.upper()}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Summary",
            self.content,
            ""
        ]
        
        if self.key_findings:
            lines.append("## Key Findings")
            for finding in self.key_findings:
                lines.append(f"- {finding}")
            lines.append("")
        
        if self.recommendations:
            lines.append("## Recommended Actions")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        if self.data_sources:
            lines.append("---")
            lines.append(f"*Data Sources: {', '.join(self.data_sources)}*")
            lines.append(f"*Confidence: {self.confidence:.1%} | Model: {self.model_used}*")
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Convert narrative to HTML format"""
        severity_colors = {
            NarrativeSeverity.CRITICAL: '#dc3545',
            NarrativeSeverity.HIGH: '#fd7e14',
            NarrativeSeverity.MODERATE: '#ffc107',
            NarrativeSeverity.LOW: '#28a745',
            NarrativeSeverity.INFORMATIONAL: '#17a2b8'
        }
        
        html = f"""
        <div class="narrative-container">
            <h2>{self.title}</h2>
            <div class="metadata">
                <span class="badge" style="background-color: {severity_colors.get(self.severity, '#6c757d')}">
                    {self.severity.value.upper()}
                </span>
                <span class="date">{self.generated_at.strftime('%Y-%m-%d %H:%M')}</span>
            </div>
            <div class="content">
                <p>{self.content}</p>
            </div>
        """
        
        if self.key_findings:
            html += "<div class='findings'><h3>Key Findings</h3><ul>"
            for finding in self.key_findings:
                html += f"<li>{finding}</li>"
            html += "</ul></div>"
        
        if self.recommendations:
            html += "<div class='recommendations'><h3>Recommended Actions</h3><ol>"
            for rec in self.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ol></div>"
        
        html += "</div>"
        return html


class GenerativeNarrativeEngine:
    """
    AI-Powered Narrative Generation Engine
    
    Generates dynamic, contextual clinical narratives using LLM
    with regulatory compliance validation.
    """
    
    # Prompt templates for different narrative types
    NARRATIVE_PROMPTS = {
        NarrativeType.PATIENT_SAFETY: """You are a clinical data analyst generating a patient safety narrative for Medical Monitor review.

Subject Information:
{subject_data}

Adverse Events:
{adverse_events}

Concomitant Medications:
{medications}

Lab Issues:
{lab_issues}

Generate a professional clinical narrative that:
1. Summarizes the patient's current safety status
2. Highlights any serious adverse events chronologically
3. Notes any potential drug interactions or concerns
4. Identifies outstanding data issues requiring attention
5. Uses formal medical writing style appropriate for regulatory documentation

Keep the narrative concise (max {max_length} characters) and factual.
Do not make assumptions beyond the provided data.
""",

        NarrativeType.SITE_PERFORMANCE: """You are a clinical operations analyst generating a site performance summary for CRA review.

Site Information:
{site_data}

Key Metrics:
{metrics}

Issues Identified:
{issues}

Generate a professional performance narrative that:
1. Summarizes the site's current operational status
2. Highlights key performance indicators (enrollment, data quality, query resolution)
3. Identifies areas of concern requiring CRA attention
4. Provides specific, actionable recommendations
5. Maintains objective, constructive tone

Keep the narrative concise (max {max_length} characters) and evidence-based.
""",

        NarrativeType.STUDY_SUMMARY: """You are a clinical study analyst generating an executive study summary.

Study Information:
{study_data}

Overall Metrics:
{metrics}

Site Performance Summary:
{site_summary}

Safety Overview:
{safety_summary}

Generate an executive summary that:
1. Provides high-level study status overview
2. Highlights key achievements and concerns
3. Summarizes enrollment and data quality metrics
4. Notes any safety signals or trends
5. Recommends strategic actions

Keep concise (max {max_length} characters) and suitable for senior leadership.
""",

        NarrativeType.DATA_QUALITY: """You are a data quality analyst generating a data quality report.

Study/Site: {entity_id}

Quality Metrics:
{quality_metrics}

Issues Detected:
{issues}

Trends:
{trends}

Generate a data quality narrative that:
1. Summarizes current data quality status (DQI score, clean patient rate)
2. Identifies top data quality issues by category
3. Highlights trends (improving/declining)
4. Provides specific remediation recommendations
5. Prioritizes actions by impact

Keep concise (max {max_length} characters) and actionable.
"""
    }
    
    # Regulatory terms to avoid in narratives
    REGULATORY_RESTRICTED_TERMS = [
        'definitely caused by', 'certainly related to', 'proven to',
        'will definitely', 'guaranteed', 'always results in',
        'never occurs', 'impossible', 'perfect'
    ]
    
    # Safe replacement suggestions
    REGULATORY_SAFE_ALTERNATIVES = {
        'definitely caused by': 'potentially related to',
        'certainly related to': 'possibly associated with',
        'proven to': 'observed to',
        'will definitely': 'may',
        'guaranteed': 'expected',
        'always results in': 'commonly associated with',
        'never occurs': 'rarely occurs',
        'impossible': 'unlikely',
        'perfect': 'complete'
    }
    
    def __init__(
        self,
        use_llm: bool = True,
        llm_config: Optional[LLMConfig] = None
    ):
        """Initialize the generative narrative engine"""
        self.use_llm = use_llm and LLM_AVAILABLE
        
        if self.use_llm:
            if llm_config:
                client = LLMClientFactory.create(llm_config)
                self.reasoning_engine = AgentReasoningEngine(client)
            else:
                self.reasoning_engine = get_reasoning_engine()
        else:
            self.reasoning_engine = None
        
        # Data sources
        self.data_sources: Dict[str, pd.DataFrame] = {}
        
        # Generation statistics
        self._stats = {
            'narratives_generated': 0,
            'total_generation_time_ms': 0,
            'regulatory_validations': 0,
            'validation_failures': 0
        }
        
        logger.info(f"GenerativeNarrativeEngine initialized (LLM: {self.use_llm})")
    
    def load_data(self, data_sources: Dict[str, pd.DataFrame]):
        """Load data sources for narrative generation"""
        self.data_sources = data_sources
        logger.info(f"Loaded {len(data_sources)} data sources")
    
    def generate(
        self,
        context: NarrativeContext,
        custom_data: Optional[Dict] = None
    ) -> GeneratedNarrative:
        """
        Generate a narrative based on context and data.
        
        Args:
            context: NarrativeContext specifying what to generate
            custom_data: Optional custom data to include
            
        Returns:
            GeneratedNarrative with AI-generated content
        """
        import time
        import uuid
        
        start_time = time.time()
        narrative_id = f"narr_{uuid.uuid4().hex[:12]}"
        
        # Gather data for the narrative
        data = self._gather_data(context, custom_data)
        
        # Select appropriate prompt template
        prompt_template = self.NARRATIVE_PROMPTS.get(
            context.narrative_type,
            self.NARRATIVE_PROMPTS[NarrativeType.PATIENT_SAFETY]
        )
        
        # Build prompt with data
        prompt = self._build_prompt(prompt_template, data, context.max_length)
        
        # Generate narrative content
        if self.use_llm and self.reasoning_engine:
            content, model_used = self._generate_with_llm(prompt)
        else:
            content, model_used = self._generate_fallback(context, data)
        
        # Extract key findings and recommendations
        key_findings = self._extract_findings(content)
        recommendations = self._extract_recommendations(content) if context.include_recommendations else []
        
        # Determine severity
        severity = self._determine_severity(data, content)
        
        # Regulatory validation
        regulatory_validated = True
        if context.regulatory_safe:
            content, regulatory_validated = self._validate_regulatory(content)
        
        # Calculate generation time
        generation_time_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self._stats['narratives_generated'] += 1
        self._stats['total_generation_time_ms'] += generation_time_ms
        
        # Create title
        title = self._generate_title(context)
        
        return GeneratedNarrative(
            narrative_id=narrative_id,
            narrative_type=context.narrative_type,
            context=context,
            title=title,
            content=content,
            severity=severity,
            key_findings=key_findings,
            recommendations=recommendations,
            data_sources=list(self.data_sources.keys()) if context.include_data_sources else [],
            generated_at=datetime.now(),
            generation_time_ms=generation_time_ms,
            model_used=model_used,
            confidence=0.85 if self.use_llm else 0.6,
            regulatory_validated=regulatory_validated
        )
    
    def generate_stream(
        self,
        context: NarrativeContext,
        custom_data: Optional[Dict] = None
    ) -> Generator[str, None, None]:
        """
        Generate narrative with streaming output.
        Yields chunks of text as they're generated.
        """
        # For now, generate complete and yield in chunks
        # In production, would use streaming LLM API
        narrative = self.generate(context, custom_data)
        
        # Simulate streaming by yielding chunks
        chunk_size = 100
        content = narrative.content
        
        for i in range(0, len(content), chunk_size):
            yield content[i:i + chunk_size]
    
    def _gather_data(
        self,
        context: NarrativeContext,
        custom_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Gather all relevant data for narrative generation"""
        data = {
            'subject_data': {},
            'site_data': {},
            'study_data': {},
            'adverse_events': [],
            'medications': [],
            'lab_issues': [],
            'metrics': {},
            'issues': [],
            'trends': {}
        }
        
        # Add custom data if provided
        if custom_data:
            data.update(custom_data)
        
        # Extract from data sources based on context
        cpid_df = self.data_sources.get('cpid') or self.data_sources.get('CPID_EDC_Metrics')
        sae_df = self.data_sources.get('sae') or self.data_sources.get('esae_dashboard')
        coding_df = self.data_sources.get('coding') or self.data_sources.get('GlobalCodingReport')
        
        if context.subject_id and cpid_df is not None:
            data['subject_data'] = self._extract_subject_data(cpid_df, context.subject_id)
            
        if context.site_id and cpid_df is not None:
            data['site_data'] = self._extract_site_data(cpid_df, context.site_id)
            data['metrics'] = self._extract_site_metrics(cpid_df, context.site_id)
        
        if context.subject_id and sae_df is not None:
            data['adverse_events'] = self._extract_adverse_events(sae_df, context.subject_id)
        
        if context.subject_id and coding_df is not None:
            data['medications'] = self._extract_medications(coding_df, context.subject_id)
        
        return data
    
    def _extract_subject_data(self, df: pd.DataFrame, subject_id: str) -> Dict:
        """Extract subject data from CPID"""
        subject_col = self._find_column(df, ['Subject ID', 'Subject', 'subject_id'])
        if not subject_col:
            return {}
        
        subject_rows = df[df[subject_col].astype(str) == str(subject_id)]
        if subject_rows.empty:
            return {}
        
        row = subject_rows.iloc[0]
        return {
            'subject_id': subject_id,
            'site_id': str(row.get('Site ID', row.get('Site', ''))),
            'country': str(row.get('Country', '')),
            'status': str(row.get('Subject Status (Source: PRIMARY Form)', 'Unknown')),
            'missing_visits': int(row.get('Missing Visits', 0)),
            'open_queries': int(row.get('Open Queries', 0)),
            'uncoded_terms': int(row.get('Uncoded Terms', 0))
        }
    
    def _extract_site_data(self, df: pd.DataFrame, site_id: str) -> Dict:
        """Extract site data from CPID"""
        site_col = self._find_column(df, ['Site ID', 'Site', 'site_id'])
        if not site_col:
            return {}
        
        site_rows = df[df[site_col].astype(str) == str(site_id)]
        if site_rows.empty:
            return {}
        
        return {
            'site_id': site_id,
            'total_subjects': len(site_rows),
            'country': str(site_rows.iloc[0].get('Country', '')),
            'region': str(site_rows.iloc[0].get('Region', ''))
        }
    
    def _extract_site_metrics(self, df: pd.DataFrame, site_id: str) -> Dict:
        """Extract site metrics from CPID"""
        site_col = self._find_column(df, ['Site ID', 'Site', 'site_id'])
        if not site_col:
            return {}
        
        site_rows = df[df[site_col].astype(str) == str(site_id)]
        if site_rows.empty:
            return {}
        
        metrics = {}
        metric_cols = ['Missing Visits', 'Open Queries', 'Uncoded Terms', 'Verification %']
        
        for col in metric_cols:
            if col in site_rows.columns:
                values = site_rows[col].dropna()
                if len(values) > 0:
                    metrics[col] = {
                        'total': float(values.sum()),
                        'mean': float(values.mean()),
                        'max': float(values.max())
                    }
        
        return metrics
    
    def _extract_adverse_events(self, df: pd.DataFrame, subject_id: str) -> List[Dict]:
        """Extract adverse events for subject"""
        subject_col = self._find_column(df, ['Patient ID', 'Subject ID', 'Subject'])
        if not subject_col:
            return []
        
        subject_rows = df[df[subject_col].astype(str) == str(subject_id)]
        
        events = []
        for _, row in subject_rows.iterrows():
            events.append({
                'form_name': str(row.get('Form Name', '')),
                'review_status': str(row.get('Review Status', '')),
                'action_status': str(row.get('Action Status', '')),
                'discrepancy_id': str(row.get('Discrepancy ID', ''))
            })
        
        return events
    
    def _extract_medications(self, df: pd.DataFrame, subject_id: str) -> List[Dict]:
        """Extract medications for subject"""
        subject_col = self._find_column(df, ['Subject_ID', 'Subject ID', 'Subject'])
        if not subject_col:
            return []
        
        subject_rows = df[df[subject_col].astype(str).str.replace('Subject ', '') == str(subject_id)]
        
        meds = []
        for _, row in subject_rows.iterrows():
            meds.append({
                'term': str(row.get('Term', '')),
                'coding_status': str(row.get('Coding_Status', '')),
                'dictionary': str(row.get('Dictionary', ''))
            })
        
        return meds
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column by possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _build_prompt(self, template: str, data: Dict, max_length: int) -> str:
        """Build prompt from template and data"""
        # Format data for prompt
        formatted = {
            'subject_data': json.dumps(data.get('subject_data', {}), indent=2),
            'site_data': json.dumps(data.get('site_data', {}), indent=2),
            'study_data': json.dumps(data.get('study_data', {}), indent=2),
            'adverse_events': json.dumps(data.get('adverse_events', []), indent=2),
            'medications': json.dumps(data.get('medications', []), indent=2),
            'lab_issues': json.dumps(data.get('lab_issues', []), indent=2),
            'metrics': json.dumps(data.get('metrics', {}), indent=2),
            'issues': json.dumps(data.get('issues', []), indent=2),
            'trends': json.dumps(data.get('trends', {}), indent=2),
            'quality_metrics': json.dumps(data.get('metrics', {}), indent=2),
            'site_summary': json.dumps(data.get('site_data', {}), indent=2),
            'safety_summary': json.dumps(data.get('adverse_events', []), indent=2),
            'entity_id': data.get('subject_data', {}).get('subject_id', data.get('site_data', {}).get('site_id', '')),
            'max_length': max_length
        }
        
        try:
            return template.format(**formatted)
        except KeyError as e:
            logger.warning(f"Missing template key: {e}")
            return template
    
    def _generate_with_llm(self, prompt: str) -> tuple:
        """Generate narrative using LLM"""
        try:
            response = self.reasoning_engine.llm.generate(prompt)
            return response.content, response.model
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_text(), "fallback"
    
    def _generate_fallback(self, context: NarrativeContext, data: Dict) -> tuple:
        """Generate fallback narrative without LLM"""
        # Build a rule-based narrative
        parts = []
        
        if context.narrative_type == NarrativeType.PATIENT_SAFETY:
            subject_data = data.get('subject_data', {})
            if subject_data:
                parts.append(f"Subject {subject_data.get('subject_id', 'Unknown')} at Site {subject_data.get('site_id', 'Unknown')}")
                parts.append(f"Current status: {subject_data.get('status', 'Unknown')}")
                
                issues = []
                if subject_data.get('missing_visits', 0) > 0:
                    issues.append(f"{subject_data.get('missing_visits')} missing visits")
                if subject_data.get('open_queries', 0) > 0:
                    issues.append(f"{subject_data.get('open_queries')} open queries")
                if subject_data.get('uncoded_terms', 0) > 0:
                    issues.append(f"{subject_data.get('uncoded_terms')} uncoded terms")
                
                if issues:
                    parts.append(f"Outstanding issues: {', '.join(issues)}")
                else:
                    parts.append("No outstanding data issues identified.")
            
            adverse_events = data.get('adverse_events', [])
            if adverse_events:
                parts.append(f"\n{len(adverse_events)} adverse event record(s) found in safety database.")
                for ae in adverse_events[:3]:  # Limit to first 3
                    parts.append(f"- {ae.get('form_name', 'Unknown')}: {ae.get('review_status', 'Unknown')} ({ae.get('action_status', 'Unknown')})")
        
        elif context.narrative_type == NarrativeType.SITE_PERFORMANCE:
            site_data = data.get('site_data', {})
            if site_data:
                parts.append(f"Site {site_data.get('site_id', 'Unknown')} Performance Summary")
                parts.append(f"Total subjects: {site_data.get('total_subjects', 0)}")
                parts.append(f"Location: {site_data.get('country', 'Unknown')}, {site_data.get('region', 'Unknown')}")
            
            metrics = data.get('metrics', {})
            if metrics:
                parts.append("\nKey Metrics:")
                for metric_name, metric_values in metrics.items():
                    if isinstance(metric_values, dict):
                        parts.append(f"- {metric_name}: Total={metric_values.get('total', 0):.0f}, Avg={metric_values.get('mean', 0):.1f}")
        
        content = "\n".join(parts) if parts else "No data available for narrative generation."
        return content, "rule-based"
    
    def _generate_fallback_text(self) -> str:
        """Generate minimal fallback text"""
        return "Unable to generate narrative. Please review the source data directly."
    
    def _extract_findings(self, content: str) -> List[str]:
        """Extract key findings from narrative content"""
        findings = []
        
        # Look for bullet points or numbered items
        bullet_pattern = r'(?:^|\n)[•\-\*]\s*(.+?)(?=\n|$)'
        bullets = re.findall(bullet_pattern, content)
        findings.extend(bullets[:5])
        
        # Look for sentences with key indicators
        key_phrases = ['identified', 'detected', 'found', 'observed', 'noted', 'shows', 'indicates']
        sentences = content.split('. ')
        
        for sentence in sentences:
            for phrase in key_phrases:
                if phrase in sentence.lower() and len(sentence) > 30:
                    finding = sentence.strip()
                    if not finding.endswith('.'):
                        finding += '.'
                    if finding not in findings:
                        findings.append(finding)
                    break
        
        return findings[:5]  # Limit to 5 findings
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from narrative content"""
        recommendations = []
        
        # Look for recommendation indicators
        rec_phrases = [
            'recommend', 'should', 'action required', 'suggest', 
            'please', 'need to', 'must', 'priority'
        ]
        
        sentences = content.split('. ')
        
        for sentence in sentences:
            for phrase in rec_phrases:
                if phrase in sentence.lower() and len(sentence) > 20:
                    rec = sentence.strip()
                    if not rec.endswith('.'):
                        rec += '.'
                    if rec not in recommendations:
                        recommendations.append(rec)
                    break
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _determine_severity(self, data: Dict, content: str) -> NarrativeSeverity:
        """Determine severity level based on data and content"""
        # Check for critical indicators
        critical_keywords = ['death', 'life-threatening', 'hospitalization', 'critical', 'urgent', 'immediate']
        high_keywords = ['serious', 'sae', 'escalate', 'overdue', 'pending > 7']
        
        content_lower = content.lower()
        
        for keyword in critical_keywords:
            if keyword in content_lower:
                return NarrativeSeverity.CRITICAL
        
        for keyword in high_keywords:
            if keyword in content_lower:
                return NarrativeSeverity.HIGH
        
        # Check data indicators
        adverse_events = data.get('adverse_events', [])
        if adverse_events:
            for ae in adverse_events:
                if 'sae' in str(ae.get('form_name', '')).lower():
                    return NarrativeSeverity.HIGH
        
        subject_data = data.get('subject_data', {})
        if subject_data:
            if subject_data.get('missing_visits', 0) > 5:
                return NarrativeSeverity.MODERATE
            if subject_data.get('open_queries', 0) > 10:
                return NarrativeSeverity.MODERATE
        
        return NarrativeSeverity.LOW
    
    def _validate_regulatory(self, content: str) -> tuple:
        """Validate and sanitize content for regulatory compliance"""
        self._stats['regulatory_validations'] += 1
        
        validated_content = content
        is_valid = True
        
        for restricted_term in self.REGULATORY_RESTRICTED_TERMS:
            if restricted_term in content.lower():
                safe_alternative = self.REGULATORY_SAFE_ALTERNATIVES.get(
                    restricted_term,
                    'may be associated with'
                )
                validated_content = re.sub(
                    restricted_term,
                    safe_alternative,
                    validated_content,
                    flags=re.IGNORECASE
                )
                is_valid = False
                self._stats['validation_failures'] += 1
        
        return validated_content, is_valid
    
    def _generate_title(self, context: NarrativeContext) -> str:
        """Generate appropriate title for narrative"""
        type_titles = {
            NarrativeType.PATIENT_SAFETY: f"Patient Safety Narrative - Subject {context.subject_id}",
            NarrativeType.SITE_PERFORMANCE: f"Site Performance Report - Site {context.site_id}",
            NarrativeType.STUDY_SUMMARY: f"Study Summary - {context.study_id}",
            NarrativeType.REGULATORY_REPORT: f"Regulatory Report - {context.study_id}",
            NarrativeType.DATA_QUALITY: f"Data Quality Report - {context.site_id or context.study_id}",
            NarrativeType.EXECUTIVE_SUMMARY: f"Executive Summary - {context.study_id}"
        }
        
        return type_titles.get(context.narrative_type, "Clinical Narrative")
    
    def get_stats(self) -> Dict:
        """Get generation statistics"""
        avg_time = 0
        if self._stats['narratives_generated'] > 0:
            avg_time = self._stats['total_generation_time_ms'] / self._stats['narratives_generated']
        
        return {
            **self._stats,
            'average_generation_time_ms': avg_time,
            'llm_enabled': self.use_llm
        }


# Convenience function for one-shot narrative generation
def generate_narrative(
    narrative_type: str,
    subject_id: Optional[str] = None,
    site_id: Optional[str] = None,
    study_id: Optional[str] = None,
    data_sources: Dict[str, pd.DataFrame] = None
) -> GeneratedNarrative:
    """
    One-shot narrative generation function.
    
    Example:
        >>> narrative = generate_narrative('patient_safety', subject_id='101-001', data_sources={'cpid': cpid_df})
        >>> print(narrative.to_markdown())
    """
    engine = GenerativeNarrativeEngine()
    
    if data_sources:
        engine.load_data(data_sources)
    
    context = NarrativeContext(
        subject_id=subject_id,
        site_id=site_id,
        study_id=study_id,
        narrative_type=NarrativeType(narrative_type)
    )
    
    return engine.generate(context)
