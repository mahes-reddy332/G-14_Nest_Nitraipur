"""
Reports API Router
Endpoints for listing and retrieving generated reports
"""

from fastapi import APIRouter, HTTPException, Path
from typing import List, Dict, Any
from pydantic import BaseModel
from pathlib import Path as SysPath
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = SysPath(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"


class ReportSummary(BaseModel):
    report_id: str
    name: str
    report_type: str
    size_bytes: int
    last_modified: str


class ReportDetail(ReportSummary):
    content_type: str
    content: Any


def _safe_report_path(report_id: str) -> SysPath:
    safe_name = SysPath(report_id).name
    return REPORTS_DIR / safe_name


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


@router.get("/{report_id}", response_model=ReportDetail)
async def get_report(report_id: str = Path(..., description="Report filename")) -> Dict[str, Any]:
    """Get report content by filename"""
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
