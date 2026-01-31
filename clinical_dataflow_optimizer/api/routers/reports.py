"""
Reports API Router
Endpoints for listing and retrieving generated reports
"""

from fastapi import APIRouter, HTTPException, Path, Query
from typing import List, Dict, Any
from pydantic import BaseModel
from pathlib import Path as SysPath
from datetime import datetime
import json
import logging
import csv
from io import StringIO
from uuid import uuid4

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = SysPath(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"

_study_report_cache: Dict[str, Dict[str, Any]] = {}


def _build_data_status(total_patients: int, source_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    reasons = []
    loaded_files = [f for f in source_files if f.get('status') == 'loaded']
    missing_files = [f for f in source_files if f.get('status') != 'loaded']

    if total_patients == 0:
        reasons.append({
            'code': 'NO_PATIENTS',
            'message': 'No patient records are available for this study yet.'
        })

    if not loaded_files:
        reasons.append({
            'code': 'NO_SOURCE_FILES',
            'message': 'No source files were loaded for this study.'
        })
    elif missing_files:
        reasons.append({
            'code': 'PARTIAL_SOURCE_FILES',
            'message': f"{len(missing_files)} source file types are missing."
        })

    status = 'ok' if not reasons else 'partial'
    if total_patients == 0 and not loaded_files:
        status = 'empty'

    return {
        'status': status,
        'reasons': reasons,
        'loaded_files': len(loaded_files),
        'missing_files': len(missing_files)
    }


def _add_insight(insights: List[Dict[str, Any]], category: str, severity: str, title: str,
                 what: str, why: str, evidence: Dict[str, Any]) -> None:
    insights.append({
        'insight_id': f"INS-{uuid4().hex[:8]}",
        'category': category,
        'severity': severity,
        'title': title,
        'what_happened': what,
        'why_it_matters': why,
        'evidence': evidence,
        'generated_at': datetime.utcnow().isoformat()
    })


async def _build_study_report(study_id: str) -> Dict[str, Any]:
    from api.config import get_initialized_data_service, get_initialized_metrics_service, get_initialized_alert_service, get_settings

    settings = get_settings()
    data_service = await get_initialized_data_service()
    metrics_service = await get_initialized_metrics_service()
    alert_service = await get_initialized_alert_service()

    study_detail = await data_service.get_study_detail(study_id)
    if not study_detail:
        raise HTTPException(status_code=404, detail=f"Study {study_id} not found")

    metrics = await metrics_service.get_study_metrics(study_id)
    agg = await data_service.get_cpid_aggregate(study_id)
    risk = await data_service.get_risk_assessment(study_id)
    sites = await data_service.get_sites({'study_id': study_id}, 'dqi_score', 'asc')
    alerts = await alert_service.get_alerts({'study_id': study_id}, limit=50, offset=0)
    source_files = await data_service.get_study_source_files(study_id)

    total_patients = agg.get('total_patients', 0)
    missing_visits = agg.get('missing_visits', 0)
    missing_pages = agg.get('missing_pages', 0)
    open_queries = agg.get('open_queries', 0)

    completeness_rate = round(100 - ((missing_visits + missing_pages) / max(1, total_patients) * 100), 1) if total_patients > 0 else 0.0
    visit_completion_rate = round(100 - (missing_visits / max(1, total_patients) * 100), 1) if total_patients > 0 else 0.0
    form_completion_rate = round(100 - (missing_pages / max(1, total_patients) * 100), 1) if total_patients > 0 else 0.0

    freshness_days = None
    loaded_times = [f.get('loaded_at') for f in source_files if f.get('loaded_at')]
    if loaded_times:
        latest = max(datetime.fromisoformat(ts) for ts in loaded_times)
        freshness_days = (datetime.utcnow() - latest).days

    data_status = _build_data_status(total_patients, source_files)

    insights: List[Dict[str, Any]] = []
    dqi_score = float(metrics.get('dqi_score', 0) or 0)
    cleanliness_rate = float(metrics.get('cleanliness_rate', 0) or 0)
    query_resolution_rate = float(metrics.get('query_resolution_rate', 0) or 0)
    coding_completion_rate = float(metrics.get('coding_completion_rate', 0) or 0)
    sae_reconciliation_rate = float(metrics.get('sae_reconciliation_rate', 0) or 0)

    if dqi_score < 75:
        severity = 'critical' if dqi_score < 60 else 'warning'
        _add_insight(
            insights,
            'quality',
            severity,
            'DQI below target',
            f"DQI is {dqi_score:.1f} for {study_id}.",
            'Lower data quality can delay downstream analyses and increase review effort.',
            {'dqi_score': dqi_score, 'target': 80}
        )
    elif dqi_score >= 85:
        _add_insight(
            insights,
            'quality',
            'info',
            'DQI performing well',
            f"DQI is {dqi_score:.1f}, meeting target quality thresholds.",
            'High DQI reduces the risk of rework and improves decision confidence.',
            {'dqi_score': dqi_score, 'target': 80}
        )

    if cleanliness_rate < 85:
        _add_insight(
            insights,
            'quality',
            'warning',
            'Cleanliness rate below target',
            f"Cleanliness rate is {cleanliness_rate:.1f}%.",
            'Low cleanliness increases monitoring and query resolution workload.',
            {'cleanliness_rate': cleanliness_rate, 'target': 90}
        )

    open_query_rate = open_queries / max(1, total_patients)
    if open_queries >= max(20, int(total_patients * 0.3)):
        _add_insight(
            insights,
            'operational',
            'warning',
            'Open query backlog',
            f"There are {open_queries} open queries ({open_query_rate:.2f} per patient).",
            'Query backlogs slow data cleaning and can delay database lock.',
            {'open_queries': open_queries, 'open_queries_per_patient': round(open_query_rate, 2)}
        )

    if visit_completion_rate < 90 or form_completion_rate < 90:
        _add_insight(
            insights,
            'quality',
            'warning',
            'Completion gaps detected',
            f"Visit completion is {visit_completion_rate:.1f}% and form completion is {form_completion_rate:.1f}%.",
            'Missing visits or forms can introduce bias and impact safety monitoring.',
            {'visit_completion_rate': visit_completion_rate, 'form_completion_rate': form_completion_rate}
        )

    if coding_completion_rate > 0 and coding_completion_rate < 90:
        _add_insight(
            insights,
            'compliance',
            'warning',
            'Coding backlog detected',
            f"Coding completion is {coding_completion_rate:.1f}%.",
            'Delayed coding can impact medical review timelines and reporting.',
            {'coding_completion_rate': coding_completion_rate, 'target': 95}
        )

    if sae_reconciliation_rate > 0 and sae_reconciliation_rate < 95:
        _add_insight(
            insights,
            'risk',
            'critical',
            'SAE reconciliation below target',
            f"SAE reconciliation rate is {sae_reconciliation_rate:.1f}%.",
            'Delayed reconciliation can pose safety and compliance risks.',
            {'sae_reconciliation_rate': sae_reconciliation_rate, 'target': 95}
        )

    if not insights:
        _add_insight(
            insights,
            'performance',
            'info',
            'Key indicators within expected range',
            'No critical deviations were detected in the latest metrics.',
            'Stable indicators support on-track execution and reporting confidence.',
            {'dqi_score': dqi_score, 'cleanliness_rate': cleanliness_rate, 'open_queries': open_queries}
        )

    kpis = [
        {'label': 'DQI Score', 'value': dqi_score, 'unit': '%', 'status': 'good' if dqi_score >= 80 else 'warning'},
        {'label': 'Cleanliness Rate', 'value': cleanliness_rate, 'unit': '%', 'status': 'good' if cleanliness_rate >= 90 else 'warning'},
        {'label': 'Open Queries', 'value': open_queries, 'unit': None, 'status': 'warning' if open_queries >= max(20, int(total_patients * 0.3)) else 'good'},
        {'label': 'Pending SAEs', 'value': metrics.get('sae_count', 0), 'unit': None, 'status': 'warning' if metrics.get('sae_count', 0) > 0 else 'good'},
        {'label': 'Coding Completion', 'value': coding_completion_rate, 'unit': '%', 'status': 'good' if coding_completion_rate >= 95 else 'warning'},
        {'label': 'Query Resolution Rate', 'value': query_resolution_rate, 'unit': '%', 'status': 'good' if query_resolution_rate >= 85 else 'warning'},
    ]

    report = {
        'study_id': study_id,
        'study_name': study_detail.get('study_name') or study_detail.get('study_id'),
        'generated_at': datetime.utcnow().isoformat(),
        'filters': {
            'days': 30,
            'timezone': 'UTC'
        },
        'overview': {
            'study_id': study_id,
            'study_name': study_detail.get('study_name') or study_id,
            'status': study_detail.get('status', 'active'),
            'phase': study_detail.get('phase'),
            'therapeutic_area': study_detail.get('therapeutic_area'),
            'objective': None,
            'start_date': study_detail.get('start_date'),
            'timeline': {
                'start_date': study_detail.get('start_date'),
                'end_date': None
            },
            'target_enrollment': study_detail.get('target_enrollment', total_patients),
            'current_enrollment': study_detail.get('current_enrollment', total_patients),
            'enrollment_progress': study_detail.get('enrollment_progress', 0),
            'total_sites': study_detail.get('total_sites', 0),
            'active_sites': study_detail.get('active_sites', 0),
            'last_updated': study_detail.get('last_updated')
        },
        'kpis': kpis,
        'data_quality': {
            'completeness_rate': completeness_rate,
            'visit_completion_rate': visit_completion_rate,
            'form_completion_rate': form_completion_rate,
            'freshness_days': freshness_days,
            'anomaly_indicators': [
                {'type': 'missing_visits', 'value': missing_visits},
                {'type': 'missing_pages', 'value': missing_pages},
            ],
        },
        'insights': insights,
        'trends': {
            'dqi': metrics.get('dqi_trend', []),
            'cleanliness': metrics.get('cleanliness_trend', []),
            'query_velocity': metrics.get('query_velocity', 0),
        },
        'risks_and_alerts': alerts,
        'sites_summary': [
            {
                'site_id': s.get('site_id'),
                'site_name': s.get('site_name') or s.get('site_id'),
                'country': s.get('country'),
                'region': s.get('region'),
                'total_patients': s.get('total_patients', 0),
                'dqi_score': s.get('dqi_score', 0),
                'cleanliness_rate': s.get('cleanliness_rate', 0),
                'open_queries': s.get('open_queries', 0),
                'risk_level': s.get('risk_level', 'unknown')
            }
            for s in sites
        ],
        'source_files': source_files,
        'data_status': data_status,
        'risk_assessment': risk
    }

    cache_ttl = max(60, settings.cache_timeout)
    _study_report_cache[study_id] = {
        'report': report,
        'generated_at': datetime.utcnow(),
        'expires_at': datetime.utcnow().timestamp() + cache_ttl
    }

    return report


class ReportSummary(BaseModel):
    report_id: str
    name: str
    report_type: str
    size_bytes: int
    last_modified: str


class ReportDetail(ReportSummary):
    content_type: str
    content: Any


class StudyReportSummary(BaseModel):
    study_id: str
    study_name: str
    status: str
    total_patients: int
    total_sites: int
    dqi_score: float
    cleanliness_rate: float
    open_queries: int
    pending_saes: int
    uncoded_terms: int
    last_updated: str
    data_status: Dict[str, Any]


class StudyReportInsight(BaseModel):
    insight_id: str
    category: str
    severity: str
    title: str
    what_happened: str
    why_it_matters: str
    evidence: Dict[str, Any]
    generated_at: str


class StudyReportDetail(BaseModel):
    study_id: str
    study_name: str
    generated_at: str
    filters: Dict[str, Any]
    overview: Dict[str, Any]
    kpis: List[Dict[str, Any]]
    data_quality: Dict[str, Any]
    insights: List[StudyReportInsight]
    trends: Dict[str, Any]
    risks_and_alerts: List[Dict[str, Any]]
    sites_summary: List[Dict[str, Any]]
    source_files: List[Dict[str, Any]]
    data_status: Dict[str, Any]


def _safe_report_path(report_id: str) -> SysPath:
    safe_name = SysPath(report_id).name
    return REPORTS_DIR / safe_name


# Route ordering is critical: static routes must come before parameterized routes
# Order: /, /studies, /studies/{id}/export, /studies/{id}, /{id}

@router.get("/", response_model=List[ReportSummary])
async def list_reports() -> List[Dict[str, Any]]:
    """List available reports in the reports directory"""
    if not REPORTS_DIR.exists():
        return []

    reports = []
    for file_path in REPORTS_DIR.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".json", ".html"}:
            continue
        stat = file_path.stat()
        reports.append(
            {
                "report_id": file_path.name,
                "name": file_path.stem,
                "report_type": file_path.suffix.lower().lstrip("."),
                "size_bytes": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )

    reports.sort(key=lambda r: r["last_modified"], reverse=True)
    return reports


@router.get("/studies", response_model=List[StudyReportSummary])
async def list_study_reports() -> List[Dict[str, Any]]:
    """List available study reports derived from live data"""
    try:
        from api.config import get_initialized_data_service
        data_service = await get_initialized_data_service()
        studies = await data_service.get_all_studies()
        summaries = []
        for study in studies:
            source_files = await data_service.get_study_source_files(study.get('study_id'))
            data_status = _build_data_status(study.get('total_patients', 0), source_files)
            summaries.append({
                'study_id': study.get('study_id'),
                'study_name': study.get('study_name') or study.get('study_id'),
                'status': study.get('status', 'active'),
                'total_patients': study.get('total_patients', 0),
                'total_sites': study.get('total_sites', 0),
                'dqi_score': study.get('dqi_score', 0),
                'cleanliness_rate': study.get('cleanliness_rate', 0),
                'open_queries': study.get('open_queries', 0),
                'pending_saes': study.get('pending_saes', 0),
                'uncoded_terms': study.get('uncoded_terms', 0),
                'last_updated': study.get('last_updated', datetime.utcnow().isoformat()),
                'data_status': data_status
            })
        return summaries
    except Exception as e:
        logger.error(f"Error listing study reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/studies/{study_id}/export")
async def export_study_report(study_id: str, format: str = Query("csv", pattern="^(csv)$")):
    """Export a study report in CSV format"""
    report = await _build_study_report(study_id)

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["section", "key", "value", "details"])

    for kpi in report.get('kpis', []):
        writer.writerow([
            'kpi',
            kpi.get('label'),
            kpi.get('value'),
            kpi.get('unit') or ''
        ])

    data_quality = report.get('data_quality', {})
    for key in ['completeness_rate', 'visit_completion_rate', 'form_completion_rate', 'freshness_days']:
        writer.writerow(['data_quality', key, data_quality.get(key), ''])

    for insight in report.get('insights', []):
        writer.writerow([
            'insight',
            insight.get('title'),
            insight.get('severity'),
            json.dumps(insight.get('evidence', {}))
        ])

    output.seek(0)
    filename = f"study_report_{study_id}.csv"
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"'
    }
    return StreamingResponse(iter([output.getvalue()]), media_type='text/csv', headers=headers)


@router.get("/studies/{study_id}", response_model=StudyReportDetail)
async def get_study_report(study_id: str) -> Dict[str, Any]:
    """Get a study report with insights derived from validated data"""
    cache_entry = _study_report_cache.get(study_id)
    if cache_entry and cache_entry.get('expires_at', 0) > datetime.utcnow().timestamp():
        return cache_entry['report']

    return await _build_study_report(study_id)


async def _read_report_file(report_id: str) -> Dict[str, Any]:
    """Helper function to read report file content"""
    file_path = _safe_report_path(report_id)

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")

    if file_path.suffix.lower() not in {".json", ".html"}:
        raise HTTPException(status_code=400, detail="Unsupported report type")

    stat = file_path.stat()
    report_type = file_path.suffix.lower().lstrip(".")

    try:
        if report_type == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            content_type = "application/json"
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            content_type = "text/html"

        return {
            "report_id": file_path.name,
            "name": file_path.stem,
            "report_type": report_type,
            "size_bytes": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "content_type": content_type,
            "content": content,
        }
    except Exception as e:
        logger.error(f"Failed to read report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read report")


# This route handles /artifacts/{report_id} for backwards compatibility
@router.get("/artifacts/{report_id}", response_model=ReportDetail)
async def get_report_artifact(report_id: str = Path(..., description="Report filename")) -> Dict[str, Any]:
    """Get report content by filename via /artifacts/ path"""
    return await _read_report_file(report_id)


# This route must be last - it's a catch-all for direct /{report_id} access
@router.get("/{report_id}", response_model=ReportDetail)
async def get_report(report_id: str = Path(..., description="Report filename")) -> Dict[str, Any]:
    """Get report content by filename (direct access)"""
    # Exclude paths that look like API routes to avoid conflicts
    if report_id in ("studies", "artifacts") or not (report_id.endswith(".json") or report_id.endswith(".html")):
        raise HTTPException(status_code=404, detail="Not found")
    return await _read_report_file(report_id)
