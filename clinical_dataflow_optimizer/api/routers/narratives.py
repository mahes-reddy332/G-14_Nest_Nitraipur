"""
Narratives API Router
Provides endpoints for automated patient safety narratives and RBM reports
"""

import sys
from pathlib import Path
# Add parent directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..services.narrative_service import AINarrativeService
from core.data_ingestion import ClinicalDataIngester
from narratives.patient_narrative_generator import PatientNarrativeGenerator
from narratives.rbm_report_generator import RBMReportGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/narratives", tags=["narratives"])

# Global narrative generators
patient_narrative_gen = None
rbm_report_gen = None
data_sources = {}

class NarrativeRequest(BaseModel):
    subject_id: str
    study_id: Optional[str] = "Study_1"

class RBMRequest(BaseModel):
    site_id: str
    study_id: Optional[str] = "Study_1"

class NarrativeResponse(BaseModel):
    subject_id: str
    narrative: str
    clinical_coherence_score: float
    data_sources_used: List[str]
    generated_at: datetime
    ai_generated: bool = True

class ClinicalInsightsRequest(BaseModel):
    study_id: str = "Study_1"
    focus_areas: List[str] = ["safety", "efficacy", "operational"]

class ClinicalInsightsResponse(BaseModel):
    study_id: str
    insights: str
    key_findings: List[str]
    recommendations: List[str]
    confidence: float
    generated_at: datetime
    ai_generated: bool = True

class RBMResponse(BaseModel):
    site_id: str
    report: str
    risk_categories: Dict[str, Any]
    prioritized_issues: List[str]
    generated_at: datetime
    ai_generated: bool = True

# Global narrative services
ai_narrative_service = None
patient_narrative_gen = None
rbm_report_gen = None
data_sources = {}

async def get_narrative_services():
    """Initialize narrative services with AI capability"""
    global ai_narrative_service, patient_narrative_gen, rbm_report_gen, data_sources

    if ai_narrative_service is None:
        ai_narrative_service = AINarrativeService()
        await ai_narrative_service.initialize()

    if patient_narrative_gen is None or rbm_report_gen is None:
        # Load data sources if not already loaded
        if not data_sources:
            try:
                from pathlib import Path
                data_path = Path(__file__).parent.parent.parent / ".." / ".." / "QC Anonymized Study Files"
                if data_path.exists():
                    ingester = ClinicalDataIngester()
                    data_sources = ingester.load_study_data(str(data_path / "Study 1_CPID_Input Files - Anonymization"))
                    logger.info(f"Loaded data sources for narratives: {list(data_sources.keys())}")
            except Exception as e:
                logger.warning(f"Could not load data sources for narratives: {e}")
                data_sources = {}

        # Initialize fallback generators
        patient_narrative_gen = PatientNarrativeGenerator()
        rbm_report_gen = RBMReportGenerator()

        # Load data into generators
        if data_sources:
            patient_narrative_gen.load_data(data_sources)
            rbm_report_gen.load_data(data_sources)

    return ai_narrative_service, patient_narrative_gen, rbm_report_gen

def get_narrative_generators():
    """Legacy function for backward compatibility"""
    import asyncio
    return asyncio.run(get_narrative_services())

@router.post("/patient-narrative", response_model=NarrativeResponse)
async def generate_patient_narrative(request: NarrativeRequest):
    """
    Generate an AI-powered patient safety narrative
    """
    try:
        ai_service, patient_gen, _ = await get_narrative_services()

        # Try AI generation first
        if ai_service._initialized and ai_service.llm:
            # Get data from existing generators for AI processing
            patient_data = patient_gen._extract_subject_profile(request.subject_id) if patient_gen else {}
            adverse_events = patient_gen._extract_adverse_events(request.subject_id) if patient_gen else []
            medications = patient_gen._extract_medications(request.subject_id) if patient_gen else []
            lab_issues = patient_gen._extract_lab_issues(request.subject_id) if patient_gen else []

            # Convert to dict format for AI service
            patient_dict = {
                'subject_id': request.subject_id,
                'study_id': request.study_id,
                'age': getattr(patient_data, 'age', None),
                'gender': getattr(patient_data, 'gender', None),
                'site_id': getattr(patient_data, 'site_id', None)
            } if patient_data else {'subject_id': request.subject_id, 'study_id': request.study_id}

            ae_list = [{'preferred_term': ae.preferred_term, 'start_date': ae.start_date, 'severity': ae.severity,
                       'serious': ae.serious, 'outcome': ae.outcome, 'reconciliation_status': ae.reconciliation_status}
                      for ae in adverse_events] if adverse_events else []

            med_list = [{'medication_name': m.medication_name, 'dose': m.dose, 'frequency': m.frequency,
                        'start_date': m.start_date, 'end_date': m.end_date}
                       for m in medications] if medications else []

            lab_list = [{'lab_name': l.lab_name, 'issue_type': l.issue_type, 'visit': l.visit,
                        'date': l.date, 'significance': l.significance}
                       for l in lab_issues] if lab_issues else []

            result = await ai_service.generate_patient_narrative(patient_dict, ae_list, med_list, lab_list)
        else:
            # Fallback to template-based generation
            _, patient_gen, _ = await get_narrative_services()
            result = patient_gen.generate_narrative(request.subject_id)

            # Convert to expected format
            result = {
                'narrative': result.narrative_text if hasattr(result, 'narrative_text') else str(result),
                'generated_by': 'template',
                'confidence': 0.6,
                'clinical_coherence_score': 0.7,
                'data_sources_used': ['patient_data', 'adverse_events', 'medications', 'lab_data'],
                'generated_at': datetime.now().isoformat(),
                'regulatory_compliant': True
            }

        return NarrativeResponse(
            subject_id=request.subject_id,
            narrative=result.get('narrative', ''),
            clinical_coherence_score=result.get('clinical_coherence_score', 0.0),
            data_sources_used=result.get('data_sources_used', []),
            generated_at=datetime.now(),
            ai_generated=result.get('generated_by') == 'ai'
        )

    except Exception as e:
        logger.error(f"Error generating patient narrative: {e}")
        raise HTTPException(status_code=500, detail=f"Narrative generation failed: {str(e)}")

@router.post("/rbm-report", response_model=RBMResponse)
async def generate_rbm_report(request: RBMRequest):
    """
    Generate an AI-powered RBM monitoring report
    """
    try:
        ai_service, _, rbm_gen = await get_narrative_services()

        # Try AI generation first
        if ai_service._initialized and ai_service.llm:
            # Get data from existing generators for AI processing
            site_data = {'site_id': request.site_id, 'study_id': request.study_id}
            metrics = rbm_gen._extract_site_metrics(request.site_id) if rbm_gen else {}
            issues = rbm_gen._identify_site_issues(request.site_id) if rbm_gen else []
            achievements = rbm_gen._identify_site_achievements(request.site_id) if rbm_gen else []

            # Convert to dict format for AI service
            metrics_dict = {k: v for k, v in metrics.items()} if metrics else {}
            issues_list = [{'description': i.get('description', ''), 'category': i.get('category', ''),
                           'severity': i.get('severity', 'medium'), 'count': i.get('count', 1),
                           'aging_days': i.get('aging_days', 0)} for i in issues] if issues else []
            achievements_list = [{'description': a.get('description', ''), 'impact': a.get('impact', '')}
                               for a in achievements] if achievements else []

            result = await ai_service.generate_rbm_report(site_data, metrics_dict, issues_list, achievements_list)
        else:
            # Fallback to template-based generation
            _, _, rbm_gen = await get_narrative_services()
            result = rbm_gen.generate_monitoring_report(request.site_id)

            # Convert to expected format
            result = {
                'report': result.report_text if hasattr(result, 'report_text') else str(result),
                'generated_by': 'template',
                'risk_categories': result.risk_categories if hasattr(result, 'risk_categories') else {},
                'prioritized_issues': result.prioritized_issues if hasattr(result, 'prioritized_issues') else [],
                'generated_at': datetime.now().isoformat(),
                'actionable': False
            }

        return RBMResponse(
            site_id=request.site_id,
            report=result.get('report', ''),
            risk_categories=result.get('risk_categories', {}),
            prioritized_issues=result.get('prioritized_issues', []),
            generated_at=datetime.now(),
            ai_generated=result.get('generated_by') == 'ai'
        )

    except Exception as e:
        logger.error(f"Error generating RBM report: {e}")
        raise HTTPException(status_code=500, detail=f"RBM report generation failed: {str(e)}")

@router.post("/clinical-insights", response_model=ClinicalInsightsResponse)
async def generate_clinical_insights(request: ClinicalInsightsRequest):
    """
    Generate AI-powered clinical insights and recommendations
    """
    try:
        ai_service, _, _ = await get_narrative_services()

        # Prepare data for AI insights generation
        study_data = {'study_id': request.study_id}
        safety_data = []  # Would be populated from actual data sources
        efficacy_data = []  # Would be populated from actual data sources
        operational_data = []  # Would be populated from actual data sources

        # Generate insights
        result = await ai_service.generate_clinical_insights(
            study_data, safety_data, efficacy_data, operational_data
        )

        return ClinicalInsightsResponse(
            study_id=request.study_id,
            insights=result.get('insights', ''),
            key_findings=result.get('key_findings', []),
            recommendations=result.get('recommendations', []),
            confidence=result.get('confidence', 0.5),
            generated_at=datetime.now(),
            ai_generated=result.get('generated_by') == 'ai'
        )

    except Exception as e:
        logger.error(f"Error generating clinical insights: {e}")
        raise HTTPException(status_code=500, detail=f"Clinical insights generation failed: {str(e)}")

@router.get("/patient-narratives/batch")
async def get_batch_narratives(
    study_id: str = Query("Study_1", description="Study ID"),
    limit: int = Query(10, description="Maximum number of narratives to generate"),
    risk_priority: str = Query("high", description="Risk priority filter: high, medium, low")
):
    """
    Generate narratives for multiple high-risk patients
    """
    try:
        narrative_gen, _ = get_narrative_generators()

        # Get high-risk patients
        high_risk_patients = narrative_gen.identify_high_risk_patients(
            study_id=study_id,
            risk_threshold=risk_priority,
            limit=limit
        )

        narratives = []
        for patient in high_risk_patients:
            try:
                narrative_result = narrative_gen.generate_narrative(
                    subject_id=patient['subject_id'],
                    study_id=study_id
                )
                if narrative_result:
                    narratives.append({
                        'subject_id': patient['subject_id'],
                        'site_id': patient['site_id'],
                        'risk_level': patient['risk_level'],
                        'narrative': narrative_result.get('narrative', ''),
                        'clinical_coherence_score': narrative_result.get('clinical_coherence_score', 0.0),
                        'generated_at': datetime.now()
                    })
            except Exception as e:
                logger.warning(f"Failed to generate narrative for {patient['subject_id']}: {e}")

        return {
            'study_id': study_id,
            'total_narratives': len(narratives),
            'risk_priority': risk_priority,
            'narratives': narratives,
            'generated_at': datetime.now()
        }

    except Exception as e:
        logger.error(f"Error generating batch narratives: {e}")
        raise HTTPException(status_code=500, detail=f"Batch narrative generation failed: {str(e)}")

@router.get("/rbm-reports/batch")
async def get_batch_rbm_reports(
    study_id: str = Query("Study_1", description="Study ID"),
    limit: int = Query(5, description="Maximum number of reports to generate")
):
    """
    Generate RBM reports for multiple high-risk sites
    """
    try:
        _, rbm_gen = get_narrative_generators()

        # Get high-risk sites
        high_risk_sites = rbm_gen.identify_high_risk_sites(
            study_id=study_id,
            limit=limit
        )

        reports = []
        for site in high_risk_sites:
            try:
                report_result = rbm_gen.generate_site_report(
                    site_id=site['site_id'],
                    study_id=study_id
                )
                if report_result:
                    reports.append({
                        'site_id': site['site_id'],
                        'country': site['country'],
                        'risk_score': site['risk_score'],
                        'report': report_result.get('report', ''),
                        'risk_categories': report_result.get('risk_categories', {}),
                        'generated_at': datetime.now()
                    })
            except Exception as e:
                logger.warning(f"Failed to generate RBM report for {site['site_id']}: {e}")

        return {
            'study_id': study_id,
            'total_reports': len(reports),
            'reports': reports,
            'generated_at': datetime.now()
        }

    except Exception as e:
        logger.error(f"Error generating batch RBM reports: {e}")
        raise HTTPException(status_code=500, detail=f"Batch RBM report generation failed: {str(e)}")

@router.get("/capabilities")
async def get_narrative_capabilities():
    """Get information about narrative generation capabilities"""
    return {
        "patient_narratives": {
            "description": "Automated patient safety narratives synthesizing SAE, CPID, and coding data",
            "data_sources": ["SAE_Dashboard", "CPID_EDC_Metrics", "GlobalCodingReport_MedDRA", "GlobalCodingReport_WHODRA"],
            "output_format": "Clinical narrative with temporal sequencing",
            "compliance": "AI-generated drafts requiring medical review"
        },
        "rbm_reports": {
            "description": "Risk-based monitoring reports for CRAs with prioritized site issues",
            "data_sources": ["CPID_EDC_Metrics", "Visit_Projection_Tracker"],
            "risk_categories": ["CRITICAL", "HIGH", "MODERATE", "LOW"],
            "output_format": "Actionable CRA visit reports"
        },
        "features": [
            "Cross-dataset synthesis",
            "Risk prioritization",
            "Clinical terminology compliance",
            "Audit trail generation",
            "Batch processing capabilities"
        ]
    }

@router.get("/health")
async def narrative_health_check():
    """Health check for narrative services"""
    try:
        narrative_gen, rbm_gen = get_narrative_generators()
        return {
            "status": "healthy",
            "patient_narrative_generator": patient_narrative_gen is not None,
            "rbm_report_generator": rbm_gen is not None,
            "data_sources_loaded": len(data_sources),
            "timestamp": datetime.now()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }