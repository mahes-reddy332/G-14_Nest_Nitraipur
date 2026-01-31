
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class LabService:
    """
    Service for handling Laboratory Data logic.
    Connects to DataService to retrieve actual domain datasets.
    """

    def __init__(self, data_service):
        self.data_service = data_service
    
    async def get_missing_lab_data(self, study_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get missing laboratory data.
        Returns a list of missing lab records from DataService.
        """
        return await self.data_service.get_missing_lab_data(study_id)

    async def get_reconciliation_summary(self, study_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get reconciliation summary metrics.
        """
        data = await self.get_missing_lab_data(study_id)
        
        total = len(data)
        missing_names = len([d for d in data if d.get('missing_element') == 'Lab Name'])
        missing_units = len([d for d in data if d.get('missing_element') == 'Unit'])
        missing_ranges = len([d for d in data if d.get('missing_element') == 'Reference Range'])
        
        # Calculate breakdowns for frontend charts
        by_lab_type = {}
        by_site = {}
        total_days = 0
        
        for d in data:
            # Lab Type count
            l_type = d.get('lab_test_name', 'Unknown')
            by_lab_type[l_type] = by_lab_type.get(l_type, 0) + 1
            
            # Site count
            site = d.get('site_id', 'Unknown')
            by_site[site] = by_site.get(site, 0) + 1
            
            # Resolution time (mock calc or real)
            total_days += d.get('days_since_collection', 0)

        return {
            "total_missing_lab_names": missing_names,
            "total_missing_reference_ranges": missing_ranges,
            "total_missing_units": missing_units,
            "average_resolution_time": round(total_days / total, 1) if total > 0 else 0,
            "by_lab_type": [{"type": k, "count": v} for k, v in by_lab_type.items()],
            "by_site": [{"site_id": k, "issues": v} for k, v in by_site.items()]
        }
