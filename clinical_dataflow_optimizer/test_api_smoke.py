from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api.main import app, DashboardSummary
from api.routers.studies import StudySummary, StudyMetrics
from api.routers.metrics import (
    KPITile,
    DQIMetrics,
    CleanlinessMetrics,
    QueryMetrics,
    SAEMetrics,
    CodingMetrics,
    OperationalVelocity,
)
from api.routers.reports import ReportSummary, ReportDetail


client = TestClient(app)


def test_dashboard_summary_schema():
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200
    DashboardSummary(**response.json())


def test_studies_schema():
    response = client.get("/api/studies/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        StudySummary(**data[0])


def test_study_metrics_schema():
    studies = client.get("/api/studies/").json()
    if not studies:
        return
    study_id = studies[0]["study_id"]
    response = client.get(f"/api/studies/{study_id}/metrics")
    assert response.status_code == 200
    StudyMetrics(**response.json())


def test_metrics_endpoints_schema():
    response = client.get("/api/metrics/kpi-tiles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        KPITile(**data[0])

    response = client.get("/api/metrics/dqi")
    assert response.status_code == 200
    DQIMetrics(**response.json())

    response = client.get("/api/metrics/cleanliness")
    assert response.status_code == 200
    CleanlinessMetrics(**response.json())

    response = client.get("/api/metrics/queries")
    assert response.status_code == 200
    QueryMetrics(**response.json())

    response = client.get("/api/metrics/saes")
    assert response.status_code == 200
    SAEMetrics(**response.json())

    response = client.get("/api/metrics/coding")
    assert response.status_code == 200
    CodingMetrics(**response.json())

    response = client.get("/api/metrics/velocity")
    assert response.status_code == 200
    OperationalVelocity(**response.json())


def test_reports_schema():
    response = client.get("/api/reports/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        ReportSummary(**data[0])
        report_id = data[0]["report_id"]
        detail = client.get(f"/api/reports/{report_id}")
        assert detail.status_code == 200
        ReportDetail(**detail.json())
