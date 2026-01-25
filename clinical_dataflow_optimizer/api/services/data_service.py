"""
Clinical Data Service
Integrates with existing data processing pipeline to serve API endpoints
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import logging
import re
import pickle
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_ingestion import ClinicalDataIngester
from core.metrics_calculator import PatientTwinBuilder
from core.data_quality_index import DataQualityIndexCalculator
from core.digital_twin import DigitalTwinFactory
from models.data_models import DigitalPatientTwin

logger = logging.getLogger(__name__)


class ClinicalDataService:
    """
    Service layer connecting FastAPI to existing clinical data processing
    Preserves existing business logic while exposing data via API
    """
    
    def __init__(self):
        self.data_path = PROJECT_ROOT.parent / "QC Anonymized Study Files"
        self.ingester = None
        self.twin_builder = None
        self.dqi_calculator = None
        self.twin_factory = None
        self._initialized = False
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        self._last_cache_time = None
        
    async def initialize(self):
        """Initialize data services - connects to existing pipeline"""
        if self._initialized:
            return
            
        try:
            # Check if data path exists, use fallback if not
            if not self.data_path.exists():
                logger.warning(f"Data path does not exist: {self.data_path}")
                # Try alternate locations
                alt_paths = [
                    PROJECT_ROOT.parent / "QC Anonymized Study Files",
                    PROJECT_ROOT / "QC Anonymized Study Files",
                    Path.cwd() / "QC Anonymized Study Files",
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        self.data_path = alt_path
                        logger.info(f"Using alternate data path: {self.data_path}")
                        break
                else:
                    logger.warning("No valid data path found. Service will use cached data only.")
            
            # Initialize components with graceful fallbacks
            if self.data_path.exists():
                self.ingester = ClinicalDataIngester(self.data_path)
            else:
                self.ingester = None
                logger.warning("ClinicalDataIngester not initialized - no data path available")
            
            self.twin_builder = PatientTwinBuilder()
            self.dqi_calculator = DataQualityIndexCalculator()
            
            # Load initial data
            await self._load_data()
            self._initialized = True
            logger.info("ClinicalDataService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ClinicalDataService: {e}")
            # Still mark as initialized to prevent repeated failures
            self._initialized = True
            self._cache['studies'] = {}
            self._cache['patients'] = {}
            self._cache['sites'] = {}
            logger.warning("ClinicalDataService initialized with empty data due to error")
    
    async def _load_data(self):
        """Load data from existing pipeline or cache"""
        # CACHE_DIR setup to prevent reloader loops
        cache_dir = PROJECT_ROOT / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "data_cache.pkl"
        
        try:
            # Determine available study folders
            study_dirs = []
            if self.data_path.exists():
                study_dirs = [
                    d for d in self.data_path.iterdir()
                    if d.is_dir() and "study" in d.name.lower()
                ]

            total_study_dirs = len(study_dirs)
            force_reload = os.getenv("FORCE_RELOAD_DATA", "false").lower() in {"1", "true", "yes"}

            # Check if cache exists and is recent
            if cache_file.exists() and not force_reload:
                logger.info("Loading data from cache...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self._cache.update(cached_data)
                    self._last_cache_time = datetime.now()

                cached_studies = len(self._cache.get('studies', {}))
                if total_study_dirs > 0 and cached_studies < total_study_dirs:
                    logger.warning(
                        f"Cache has {cached_studies} studies, but {total_study_dirs} study folders exist. Reloading..."
                    )
                else:
                    logger.info(
                        f"Loaded cached data: {cached_studies} studies, "
                        f"{len(self._cache.get('patients', {}))} patients, "
                        f"{len(self._cache.get('sites', {}))} sites"
                    )
                    # Even when loading from cache, we need to load graphs for twin generation
                    await self._load_graphs()
                    if self._cache.get('graphs'):
                        first_study = list(self._cache['graphs'].keys())[0]
                        graph = self._cache['graphs'][first_study]
                        self.twin_factory = DigitalTwinFactory(graph)
                        logger.info("DigitalTwinFactory initialized from cached graphs")
                    return
            
            # Load from CSV files
            logger.info("Loading data from CSV files...")
            all_data = {}
            
            if self.data_path.exists():
                for study_dir in study_dirs:
                    study_id = self._extract_study_id(study_dir.name)
                    try:
                        study_data = self.ingester.ingest_study(study_id, study_dir)
                        if study_data:
                            all_data[study_id] = study_data
                    except Exception as e:
                        logger.warning(f"Failed to load study {study_id}: {e}")
            
            self._cache['studies'] = all_data
            self._cache['patients'] = self._build_patient_index(all_data)
            self._cache['sites'] = self._build_site_index(all_data)
            self._last_cache_time = datetime.now()
            
            # Save to cache
            logger.info("Saving data to cache...")
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            
            # Load NetworkX graphs for Digital Twin generation
            await self._load_graphs()
            
            # Initialize Digital Twin Factory
            if self._cache.get('graphs'):
                # Use the first available graph for twin factory
                first_study = list(self._cache['graphs'].keys())[0]
                graph = self._cache['graphs'][first_study]
                self.twin_factory = DigitalTwinFactory(graph)
                logger.info("DigitalTwinFactory initialized")
            
            logger.info(f"Loaded {len(all_data)} studies, {len(self._cache['patients'])} patients, {len(self._cache['sites'])} sites")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Don't raise - allow service to start with empty data
            self._cache['studies'] = {}
            self._cache['patients'] = {}
            self._cache['sites'] = {}
    
    async def _load_graphs(self):
        """Load NetworkX graphs from graph_data directory"""
        graph_dir = PROJECT_ROOT.parent / "graph_data"
        if not graph_dir.exists():
            logger.warning(f"Graph data directory not found: {graph_dir}")
            self._cache['graphs'] = {}
            return
        
        self._cache['graphs'] = {}
        
        try:
            # Load all .gpickle files (NetworkX graphs)
            for gpickle_file in graph_dir.glob("*.gpickle"):
                try:
                    study_id = gpickle_file.stem.replace("_graph", "")
                    with open(gpickle_file, 'rb') as f:
                        graph = pickle.load(f)
                    
                    self._cache['graphs'][study_id] = graph
                    logger.info(f"Loaded graph for {study_id}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
                    
                except Exception as e:
                    logger.error(f"Error loading graph {gpickle_file}: {e}")
            
            logger.info(f"Loaded {len(self._cache['graphs'])} NetworkX graphs")
            
        except Exception as e:
            logger.error(f"Error loading graphs: {e}")
            self._cache['graphs'] = {}
    
    def _extract_study_id(self, name: str) -> str:
        """Extract study ID from folder name"""
        match = re.search(r'Study\s*(\d+)', name, re.IGNORECASE)
        return f"Study_{match.group(1)}" if match else name
    
    def _build_patient_index(self, studies_data: Dict) -> Dict:
        """Build patient index from study data"""
        patients = {}
        for study_id, data in studies_data.items():
            if 'cpid_metrics' in data and data['cpid_metrics'] is not None:
                cpid_df = data['cpid_metrics']
                patient_col = self._find_patient_column(cpid_df)
                if patient_col:
                    for _, row in cpid_df.iterrows():
                        patient_id = str(row[patient_col]).strip()
                        if patient_id and patient_id != 'nan':
                            patients[patient_id] = {
                                'study_id': study_id,
                                'data': data
                            }
        return patients
    
    def _build_site_index(self, studies_data: Dict) -> Dict:
        """Build site index from study data"""
        sites = {}
        for study_id, data in studies_data.items():
            if 'cpid_metrics' in data and data['cpid_metrics'] is not None:
                cpid_df = data['cpid_metrics']
                site_col = self._find_column(cpid_df, ['site_id', 'Site', 'SITEID', 'site'])
                if site_col:
                    for site_id in cpid_df[site_col].unique():
                        if pd.notna(site_id):
                            site_key = f"{study_id}_{site_id}"
                            sites[site_key] = {
                                'site_id': str(site_id),
                                'study_id': study_id,
                                'data': data
                            }
        return sites
    
    def _find_patient_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find patient ID column"""
        candidates = ['USUBJID', 'Subject', 'SUBJID', 'PatientID', 'subject_id']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find column from candidates"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _safe_int(self, value: Any) -> int:
        """Safely convert a value to int"""
        try:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return 0
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def _get_cpid_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Return standardized column names used in CPID metrics"""
        return {
            "patient": self._find_column(df, ["subject_id", "Subject ID", "USUBJID", "Subject"]),
            "site": self._find_column(df, ["site_id", "Site ID", "SITEID", "Site"]),
            "country": self._find_column(df, ["country", "Country", "COUNTRY"]),
            "region": self._find_column(df, ["region", "Region", "REGION"]),
            "open_queries": self._find_column(df, ["open_queries", "Open Queries", "# Open Queries"]),
            "total_queries": self._find_column(df, ["total_queries", "Total Queries", "# Total Queries"]),
            "uncoded_terms": self._find_column(df, ["uncoded_terms", "Uncoded Terms", "# Uncoded Terms"]),
            "missing_visits": self._find_column(df, ["missing_visits", "Missing Visits", "# Missing Visits"]),
            "missing_pages": self._find_column(df, ["missing_pages", "Missing Page", "# Missing Pages"]),
            "expected_visits": self._find_column(df, ["expected_visits", "Expected Visits", "# Expected Visits"]),
            "pages_entered": self._find_column(df, ["pages_entered", "Pages Entered", "# Pages Entered"]),
            "verification_pct": self._find_column(df, ["verification_pct", "% Clean", "Clean %", "Clean Entered"]),
        }

    def _calculate_cleanliness_score(
        self,
        open_queries: int,
        missing_visits: int,
        missing_pages: int,
        uncoded_terms: int
    ) -> float:
        """Calculate a deterministic cleanliness score from blocking factors"""
        penalty = (open_queries * 2) + (missing_visits * 5) + (missing_pages * 2) + (uncoded_terms * 1)
        return max(0.0, 100.0 - min(100.0, penalty))

    def _is_coded_status(self, status: Any) -> bool:
        """Determine if a coding status represents a coded term"""
        status_lower = str(status).lower()
        if "uncoded" in status_lower or "not coded" in status_lower or "pending" in status_lower:
            return False
        if "coded" in status_lower or "complete" in status_lower or "resolved" in status_lower:
            return True
        return False
    
    async def get_dashboard_summary(self, study_id: Optional[str] = None) -> Dict[str, Any]:
        """Get overall dashboard summary"""
        await self._ensure_initialized()
        
        studies = self._cache.get('studies', {})
        if study_id:
            studies = {k: v for k, v in studies.items() if k == study_id}
        
        total_patients = 0
        clean_patients = 0
        dirty_patients = 0
        total_sites = set()
        open_queries = 0
        pending_saes = 0
        uncoded_terms = 0
        dqi_scores = []
        
        for sid, data in studies.items():
            # CPID Metrics contains patient-level data
            cpid_df = data.get('cpid_metrics')
            if cpid_df is not None and isinstance(cpid_df, pd.DataFrame) and not cpid_df.empty:
                cols = self._get_cpid_columns(cpid_df)
                patient_col = cols["patient"]
                site_col = cols["site"]

                study_total_patients = 0
                study_open_queries = 0
                study_uncoded_terms = 0
                study_missing_visits = 0
                study_missing_pages = 0

                if patient_col:
                    study_total_patients = cpid_df[patient_col].nunique()
                    total_patients += study_total_patients

                if site_col:
                    for site in cpid_df[site_col].unique():
                        if pd.notna(site):
                            total_sites.add(f"{sid}_{site}")

                if cols["open_queries"]:
                    study_open_queries = self._safe_int(cpid_df[cols["open_queries"]].fillna(0).sum())
                    open_queries += study_open_queries

                if cols["uncoded_terms"]:
                    study_uncoded_terms = self._safe_int(cpid_df[cols["uncoded_terms"]].fillna(0).sum())
                    uncoded_terms += study_uncoded_terms

                if cols["missing_visits"]:
                    study_missing_visits = self._safe_int(cpid_df[cols["missing_visits"]].fillna(0).sum())

                if cols["missing_pages"]:
                    study_missing_pages = self._safe_int(cpid_df[cols["missing_pages"]].fillna(0).sum())

                # Calculate clean/dirty using score-based approach (consistent with cleanliness metrics)
                if patient_col:
                    grouped = cpid_df.groupby(patient_col)
                    for _, rows in grouped:
                        oq = self._safe_int(rows[cols["open_queries"]].fillna(0).sum()) if cols.get("open_queries") else 0
                        mv = self._safe_int(rows[cols["missing_visits"]].fillna(0).sum()) if cols.get("missing_visits") else 0
                        mp = self._safe_int(rows[cols["missing_pages"]].fillna(0).sum()) if cols.get("missing_pages") else 0
                        ut = self._safe_int(rows[cols["uncoded_terms"]].fillna(0).sum()) if cols.get("uncoded_terms") else 0
                        score = self._calculate_cleanliness_score(oq, mv, mp, ut)
                        if score >= 85:
                            clean_patients += 1
                        else:
                            dirty_patients += 1
            
            # SAE Dashboard contains SAE data
            sae_df = data.get('sae_dashboard')
            if sae_df is not None and isinstance(sae_df, pd.DataFrame) and not sae_df.empty:
                pending_saes += len(sae_df)
            
            # Calculate DQI for this study
            study_dqi = 85.0
            if study_total_patients > 0:
                query_penalty = min(20, (study_open_queries / max(1, study_total_patients)) * 10)
                missing_penalty = min(15, ((study_missing_visits + study_missing_pages) / max(1, study_total_patients)) * 5)
                uncoded_penalty = min(10, (study_uncoded_terms / max(1, study_total_patients)) * 5)
                study_dqi = max(50, 100 - query_penalty - missing_penalty - uncoded_penalty)
            dqi_scores.append(study_dqi)
        
        overall_dqi = sum(dqi_scores) / len(dqi_scores) if dqi_scores else 85.0
        
        return {
            'total_studies': len(studies),
            'total_patients': total_patients,
            'total_sites': len(total_sites),
            'clean_patients': clean_patients,
            'dirty_patients': dirty_patients,
            'overall_dqi': round(overall_dqi, 1),
            'open_queries': open_queries,
            'pending_saes': pending_saes,
            'uncoded_terms': uncoded_terms,
            'last_updated': datetime.now().isoformat()
        }

    async def get_cpid_aggregate(self, study_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate CPID metrics across studies or for a specific study"""
        await self._ensure_initialized()

        studies = self._cache.get('studies', {})
        if study_id:
            studies = {k: v for k, v in studies.items() if k == study_id}

        total_patients = 0
        open_queries = 0
        total_queries = 0
        uncoded_terms = 0
        missing_visits = 0
        missing_pages = 0
        clean_patients = 0
        dirty_patients = 0
        at_risk_patients = 0

        for _, data in studies.items():
            cpid_df = data.get('cpid_metrics')
            if cpid_df is None or not isinstance(cpid_df, pd.DataFrame) or cpid_df.empty:
                continue

            cols = self._get_cpid_columns(cpid_df)
            patient_col = cols.get("patient")
            if not patient_col:
                continue

            total_patients += cpid_df[patient_col].nunique()

            if cols.get("open_queries"):
                open_queries += self._safe_int(cpid_df[cols["open_queries"]].fillna(0).sum())
            if cols.get("total_queries"):
                total_queries += self._safe_int(cpid_df[cols["total_queries"]].fillna(0).sum())
            if cols.get("uncoded_terms"):
                uncoded_terms += self._safe_int(cpid_df[cols["uncoded_terms"]].fillna(0).sum())
            if cols.get("missing_visits"):
                missing_visits += self._safe_int(cpid_df[cols["missing_visits"]].fillna(0).sum())
            if cols.get("missing_pages"):
                missing_pages += self._safe_int(cpid_df[cols["missing_pages"]].fillna(0).sum())

            # Per-patient classification
            grouped = cpid_df.groupby(patient_col)
            for _, rows in grouped:
                oq = self._safe_int(rows[cols["open_queries"]].fillna(0).sum()) if cols.get("open_queries") else 0
                mv = self._safe_int(rows[cols["missing_visits"]].fillna(0).sum()) if cols.get("missing_visits") else 0
                mp = self._safe_int(rows[cols["missing_pages"]].fillna(0).sum()) if cols.get("missing_pages") else 0
                ut = self._safe_int(rows[cols["uncoded_terms"]].fillna(0).sum()) if cols.get("uncoded_terms") else 0
                score = self._calculate_cleanliness_score(oq, mv, mp, ut)

                if score >= 85:
                    clean_patients += 1
                elif score >= 70:
                    at_risk_patients += 1
                else:
                    dirty_patients += 1

        if total_queries == 0:
            total_queries = open_queries

        return {
            "total_patients": total_patients,
            "open_queries": open_queries,
            "total_queries": total_queries,
            "uncoded_terms": uncoded_terms,
            "missing_visits": missing_visits,
            "missing_pages": missing_pages,
            "clean_patients": clean_patients,
            "dirty_patients": dirty_patients,
            "at_risk_patients": at_risk_patients,
        }

    async def get_sae_aggregate(self, study_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate SAE metrics across studies or for a specific study"""
        await self._ensure_initialized()

        studies = self._cache.get('studies', {})
        if study_id:
            studies = {k: v for k, v in studies.items() if k == study_id}

        total_saes = 0
        reconciled = 0
        pending = 0
        by_category: Dict[str, int] = {}

        for _, data in studies.items():
            sae_df = data.get('sae_dashboard')
            if sae_df is None or not isinstance(sae_df, pd.DataFrame) or sae_df.empty:
                continue

            total_saes += len(sae_df)

            status_col = None
            for candidate in ["review_status", "action_status"]:
                if candidate in sae_df.columns:
                    status_col = candidate
                    break

            if status_col:
                status_series = sae_df[status_col].fillna("").astype(str).str.lower()
                pending_mask = status_series.str.contains("pending|open|incomplete|in progress")
                reconciled_mask = status_series.str.contains("closed|complete|reconciled|resolved")
                pending += int(pending_mask.sum())
                reconciled += int(reconciled_mask.sum())
            else:
                pending += len(sae_df)

            if "sae_type" in sae_df.columns:
                counts = sae_df["sae_type"].fillna("Unknown").astype(str).value_counts()
                for key, value in counts.items():
                    by_category[key] = by_category.get(key, 0) + int(value)

        return {
            "total_saes": total_saes,
            "reconciled": reconciled,
            "pending": pending,
            "by_category": by_category
        }

    async def get_coding_aggregate(self, study_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate coding metrics across studies or for a specific study"""
        await self._ensure_initialized()

        studies = self._cache.get('studies', {})
        if study_id:
            studies = {k: v for k, v in studies.items() if k == study_id}

        total_terms = 0
        coded_terms = 0
        pending_terms = 0
        meddra_total = 0
        meddra_coded = 0
        meddra_pending = 0
        whodrug_total = 0
        whodrug_coded = 0
        whodrug_pending = 0

        for _, data in studies.items():
            for key in ["meddra_coding", "whodra_coding"]:
                coding_df = data.get(key)
                if coding_df is None or not isinstance(coding_df, pd.DataFrame) or coding_df.empty:
                    continue

                total_terms += len(coding_df)
                if key == "meddra_coding":
                    meddra_total += len(coding_df)
                else:
                    whodrug_total += len(coding_df)

                if "coding_status" in coding_df.columns:
                    statuses = coding_df["coding_status"].fillna("").astype(str)
                    for status in statuses:
                        if self._is_coded_status(status):
                            coded_terms += 1
                            if key == "meddra_coding":
                                meddra_coded += 1
                            else:
                                whodrug_coded += 1
                        else:
                            pending_terms += 1
                            if key == "meddra_coding":
                                meddra_pending += 1
                            else:
                                whodrug_pending += 1
                else:
                    pending_terms += len(coding_df)
                    if key == "meddra_coding":
                        meddra_pending += len(coding_df)
                    else:
                        whodrug_pending += len(coding_df)

        return {
            "total_terms": total_terms,
            "coded_terms": coded_terms,
            "pending_terms": pending_terms,
            "meddra": {
                "total": meddra_total,
                "coded": meddra_coded,
                "uncoded": meddra_pending
            },
            "whodrug": {
                "total": whodrug_total,
                "coded": whodrug_coded,
                "uncoded": whodrug_pending
            }
        }
    
    async def get_all_studies(self) -> List[Dict[str, Any]]:
        """Get summary of all studies"""
        await self._ensure_initialized()
        
        studies = []
        for study_id, data in self._cache.get('studies', {}).items():
            summary = await self._get_study_summary(study_id, data)
            studies.append(summary)
        
        return studies
    
    async def _get_study_summary(self, study_id: str, data: Dict) -> Dict[str, Any]:
        """Get summary for a single study"""
        total_patients = 0
        clean_patients = 0
        total_sites = 0
        open_queries = 0
        pending_saes = 0
        uncoded_terms = 0
        missing_visits = 0
        missing_pages = 0
        
        # CPID Metrics contains patient-level data
        cpid_df = data.get('cpid_metrics')
        if cpid_df is not None and isinstance(cpid_df, pd.DataFrame) and not cpid_df.empty:
            cols = self._get_cpid_columns(cpid_df)
            patient_col = cols["patient"]
            site_col = cols["site"]

            if patient_col:
                total_patients = cpid_df[patient_col].nunique()

            if site_col:
                total_sites = cpid_df[site_col].nunique()

            if cols["open_queries"]:
                open_queries = self._safe_int(cpid_df[cols["open_queries"]].fillna(0).sum())

            if cols["uncoded_terms"]:
                uncoded_terms = self._safe_int(cpid_df[cols["uncoded_terms"]].fillna(0).sum())

            if cols["missing_visits"]:
                missing_visits = self._safe_int(cpid_df[cols["missing_visits"]].fillna(0).sum())

            if cols["missing_pages"]:
                missing_pages = self._safe_int(cpid_df[cols["missing_pages"]].fillna(0).sum())

            # Calculate clean patients based on blocking factors
            # A patient is "dirty" if they have open queries, missing visits, missing pages, or uncoded terms
            if patient_col:
                dirty_mask = pd.Series([False] * len(cpid_df), index=cpid_df.index)
                if cols["open_queries"]:
                    dirty_mask |= (cpid_df[cols["open_queries"]].fillna(0) > 0)
                if cols["missing_visits"]:
                    dirty_mask |= (cpid_df[cols["missing_visits"]].fillna(0) > 0)
                if cols["missing_pages"]:
                    dirty_mask |= (cpid_df[cols["missing_pages"]].fillna(0) > 0)
                if cols["uncoded_terms"]:
                    dirty_mask |= (cpid_df[cols["uncoded_terms"]].fillna(0) > 0)

                dirty_patients_count = cpid_df[dirty_mask][patient_col].nunique()
                clean_patients = total_patients - dirty_patients_count
        
        # SAE Dashboard contains SAE data
        sae_df = data.get('sae_dashboard')
        if sae_df is not None and isinstance(sae_df, pd.DataFrame) and not sae_df.empty:
            pending_saes = len(sae_df)
        
        dirty_patients = total_patients - clean_patients
        cleanliness_rate = (clean_patients / total_patients * 100) if total_patients > 0 else 0
        
        # Calculate DQI based on actual metrics
        dqi_score = 85.0  # Base score
        if total_patients > 0:
            # Penalize for open queries, missing data, uncoded terms
            query_penalty = min(20, (open_queries / total_patients) * 10)
            missing_penalty = min(15, ((missing_visits + missing_pages) / total_patients) * 5)
            uncoded_penalty = min(10, (uncoded_terms / total_patients) * 5)
            dqi_score = max(50, 100 - query_penalty - missing_penalty - uncoded_penalty)
        
        return {
            'study_id': study_id,
            'study_name': study_id.replace('_', ' '),
            'total_patients': total_patients,
            'total_sites': total_sites,
            'clean_patients': clean_patients,
            'dirty_patients': dirty_patients,
            'cleanliness_rate': round(cleanliness_rate, 1),
            'dqi_score': round(dqi_score, 1),
            'open_queries': open_queries,
            'pending_saes': pending_saes,
            'uncoded_terms': uncoded_terms,
            'enrollment_progress': 75.0,  # Placeholder
            'status': 'active',
            'last_updated': datetime.now().isoformat()
        }
    
    async def get_study_detail(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed study information"""
        await self._ensure_initialized()
        
        data = self._cache.get('studies', {}).get(study_id)
        if not data:
            return None
        
        summary = await self._get_study_summary(study_id, data)

        # Add derived information
        regions = await self.get_study_regions(study_id)
        countries = sorted({c for r in regions for c in r.get('countries', [])})

        summary.update({
            'protocol': None,
            'phase': None,
            'therapeutic_area': None,
            'start_date': None,
            'target_enrollment': summary['total_patients'],
            'current_enrollment': summary['total_patients'],
            'regions': [r['region'] for r in regions],
            'countries': countries,
            'active_sites': summary['total_sites'],
            'metrics': {
                'dqi': summary['dqi_score'],
                'cleanliness': summary['cleanliness_rate'],
                'query_resolution': 0.0,
                'sae_reconciliation': 0.0
            },
            'trends': {
                'dqi': [],
                'cleanliness': [],
                'enrollment': []
            },
            'risk_indicators': []
        })
        
        return summary
    
    async def get_study_regions(self, study_id: str) -> List[Dict[str, Any]]:
        """Get region breakdown for a study"""
        await self._ensure_initialized()

        data = self._cache.get('studies', {}).get(study_id)
        if not data:
            return []

        cpid_df = data.get('cpid_metrics')
        if cpid_df is None or not isinstance(cpid_df, pd.DataFrame) or cpid_df.empty:
            return []

        cols = self._get_cpid_columns(cpid_df)
        region_col = cols.get("region")
        if not region_col:
            return []

        regions: List[Dict[str, Any]] = []
        for region, group in cpid_df.groupby(region_col):
            region_name = str(region) if pd.notna(region) else "Unknown"
            patient_col = cols.get("patient")
            site_col = cols.get("site")
            country_col = cols.get("country")

            total_patients = group[patient_col].nunique() if patient_col else 0
            total_sites = group[site_col].nunique() if site_col else 0
            countries = []
            if country_col:
                countries = sorted({str(c) for c in group[country_col].dropna().unique().tolist()})

            open_queries = self._safe_int(group[cols["open_queries"]].fillna(0).sum()) if cols.get("open_queries") else 0
            uncoded_terms = self._safe_int(group[cols["uncoded_terms"]].fillna(0).sum()) if cols.get("uncoded_terms") else 0
            missing_visits = self._safe_int(group[cols["missing_visits"]].fillna(0).sum()) if cols.get("missing_visits") else 0
            missing_pages = self._safe_int(group[cols["missing_pages"]].fillna(0).sum()) if cols.get("missing_pages") else 0

            clean_patients = 0
            dirty_patients = 0
            if patient_col:
                dirty_mask = pd.Series([False] * len(group), index=group.index)
                if cols.get("open_queries"):
                    dirty_mask |= (group[cols["open_queries"]].fillna(0) > 0)
                if cols.get("missing_visits"):
                    dirty_mask |= (group[cols["missing_visits"]].fillna(0) > 0)
                if cols.get("missing_pages"):
                    dirty_mask |= (group[cols["missing_pages"]].fillna(0) > 0)
                if cols.get("uncoded_terms"):
                    dirty_mask |= (group[cols["uncoded_terms"]].fillna(0) > 0)

                dirty_patients = group[dirty_mask][patient_col].nunique()
                clean_patients = total_patients - dirty_patients

            dqi_score = 85.0
            if total_patients > 0:
                query_penalty = min(20, (open_queries / max(1, total_patients)) * 10)
                missing_penalty = min(15, ((missing_visits + missing_pages) / max(1, total_patients)) * 5)
                uncoded_penalty = min(10, (uncoded_terms / max(1, total_patients)) * 5)
                dqi_score = max(50, 100 - query_penalty - missing_penalty - uncoded_penalty)

            risk_level = 'low' if dqi_score >= 85 else 'medium' if dqi_score >= 70 else 'high'

            regions.append({
                'region': region_name,
                'countries': countries,
                'total_sites': total_sites,
                'total_patients': total_patients,
                'clean_patients': clean_patients,
                'dqi_score': round(dqi_score, 1),
                'risk_level': risk_level
            })

        return regions
    
    async def get_study_source_files(self, study_id: str) -> List[Dict[str, Any]]:
        """Get list of processed source files for a study"""
        await self._ensure_initialized()
        
        data = self._cache.get('studies', {}).get(study_id)
        if not data:
            return []
        
        # File type to display name mapping
        file_type_names = {
            'cpid_metrics': 'CPID EDC Metrics',
            'visit_tracker': 'Visit Projection Tracker',
            'sae_dashboard': 'eSAE Dashboard',
            'meddra_coding': 'MedDRA Coding Report',
            'whodra_coding': 'WHODRA Coding Report',
            'missing_pages': 'Missing Pages Report',
            'compiled_edrr': 'Compiled EDRR',
            'inactivated_forms': 'Inactivated Forms Report',
            'missing_lab': 'Missing Lab Report'
        }
        
        source_files = []
        for file_type, display_name in file_type_names.items():
            df = data.get(file_type)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                # Get record counts from the DataFrame
                record_count = len(df)
                source_files.append({
                    'file_type': file_type,
                    'display_name': display_name,
                    'status': 'loaded',
                    'record_count': record_count,
                    'loaded_at': self._last_cache_time.isoformat() if self._last_cache_time else datetime.now().isoformat()
                })
            else:
                source_files.append({
                    'file_type': file_type,
                    'display_name': display_name,
                    'status': 'not_found',
                    'record_count': 0,
                    'loaded_at': None
                })
        
        return source_files
    
    async def get_all_source_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get processed source files for all studies"""
        await self._ensure_initialized()
        
        result = {}
        for study_id in self._cache.get('studies', {}).keys():
            result[study_id] = await self.get_study_source_files(study_id)
        
        return result
    
    async def get_patients(self, filters: Dict, page: int, page_size: int,
                          sort_by: str, sort_order: str) -> Dict[str, Any]:
        """Get paginated patient list - optimized for large datasets"""
        await self._ensure_initialized()
        
        all_patients = self._cache.get('patients', {})
        total_count = len(all_patients)
        
        # Early filter by study_id to reduce candidate set (fast check on info dict)
        if filters.get('study_id'):
            study_filter = filters['study_id']
            patient_items = [(pid, info) for pid, info in all_patients.items() 
                           if info.get('study_id') == study_filter]
        else:
            patient_items = list(all_patients.items())
        
        # Calculate pagination indices - limit how many we process
        # Build summaries only for what's needed plus a buffer for filtering
        max_process = min(len(patient_items), page * page_size + 500)
        
        patients = []
        for patient_id, info in patient_items[:max_process]:
            patient = await self._build_patient_summary(patient_id, info)
            
            # Apply filters after building summary (status requires built summary)
            if filters.get('status'):
                if filters['status'] == 'clean' and not patient['clean_status']['is_clean']:
                    continue
                if filters['status'] == 'dirty' and patient['clean_status']['is_clean']:
                    continue
            
            patients.append(patient)
            
            # Early exit once we have enough for this page + next page
            if len(patients) >= page * page_size:
                break
        
        # Sort
        reverse = sort_order == 'desc'
        if sort_by == 'cleanliness_score':
            patients.sort(key=lambda x: x['clean_status']['cleanliness_score'], reverse=reverse)
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            'total': total_count if not filters else len(patients),
            'page': page,
            'page_size': page_size,
            'patients': patients[start:end],
            'filters_applied': filters
        }
    
    async def _build_patient_summary(self, patient_id: str, info: Dict) -> Dict[str, Any]:
        """Build patient summary object"""
        cpid_df = info.get('data', {}).get('cpid_metrics')
        cols = self._get_cpid_columns(cpid_df) if isinstance(cpid_df, pd.DataFrame) else {}

        open_queries = 0
        total_queries = 0
        missing_visits = 0
        missing_pages = 0
        uncoded_terms = 0
        expected_visits = 0
        pages_entered = 0
        site_id = ''
        country = 'Unknown'
        region = 'Unknown'

        if isinstance(cpid_df, pd.DataFrame) and cols.get("patient"):
            patient_col = cols["patient"]
            patient_rows = cpid_df[cpid_df[patient_col].astype(str).str.strip() == str(patient_id)]

            if not patient_rows.empty:
                if cols.get("site"):
                    site_id = str(patient_rows[cols["site"]].iloc[0])
                if cols.get("country"):
                    country = str(patient_rows[cols["country"]].iloc[0])
                if cols.get("region"):
                    region = str(patient_rows[cols["region"]].iloc[0])

                if cols.get("open_queries"):
                    open_queries = self._safe_int(patient_rows[cols["open_queries"]].fillna(0).sum())
                if cols.get("total_queries"):
                    total_queries = self._safe_int(patient_rows[cols["total_queries"]].fillna(0).sum())
                if cols.get("missing_visits"):
                    missing_visits = self._safe_int(patient_rows[cols["missing_visits"]].fillna(0).sum())
                if cols.get("missing_pages"):
                    missing_pages = self._safe_int(patient_rows[cols["missing_pages"]].fillna(0).sum())
                if cols.get("uncoded_terms"):
                    uncoded_terms = self._safe_int(patient_rows[cols["uncoded_terms"]].fillna(0).sum())
                if cols.get("expected_visits"):
                    expected_visits = self._safe_int(patient_rows[cols["expected_visits"]].fillna(0).max())
                if cols.get("pages_entered"):
                    pages_entered = self._safe_int(patient_rows[cols["pages_entered"]].fillna(0).sum())

        if total_queries == 0:
            total_queries = open_queries

        blocking_factors = []
        if open_queries > 0:
            blocking_factors.append('Open queries')
        if missing_visits > 0:
            blocking_factors.append('Missing visits')
        if missing_pages > 0:
            blocking_factors.append('Missing pages')
        if uncoded_terms > 0:
            blocking_factors.append('Uncoded terms')

        is_clean = len(blocking_factors) == 0
        cleanliness_score = self._calculate_cleanliness_score(
            open_queries, missing_visits, missing_pages, uncoded_terms
        )

        total_visits = expected_visits if expected_visits > 0 else 0
        completed_visits = max(0, total_visits - missing_visits) if total_visits > 0 else 0

        return {
            'patient_id': patient_id,
            'study_id': info['study_id'],
            'site_id': site_id,
            'site_name': f"Clinical Site {site_id}" if site_id else 'Unknown',
            'country': country,
            'region': region,
            'enrollment_date': '',
            'clean_status': {
                'is_clean': is_clean,
                'cleanliness_score': round(cleanliness_score, 1),
                'status': 'clean' if is_clean else 'dirty',
                'blocking_factors': blocking_factors,
                'missing_visits': missing_visits,
                'open_queries': open_queries,
                'uncoded_terms': uncoded_terms,
                'pending_saes': 0,
                'missing_forms': missing_pages,
                'unsigned_forms': 0,
                'last_calculated': datetime.now().isoformat()
            },
            'risk_level': 'low' if cleanliness_score >= 85 else 'medium' if cleanliness_score >= 70 else 'high',
            'last_visit_date': '',
            'next_scheduled_visit': '',
            'total_visits': total_visits,
            'completed_visits': completed_visits,
            'total_queries': total_queries,
            'pages_entered': pages_entered
        }
    
    async def get_patient_detail(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed patient information using Digital Twin"""
        await self._ensure_initialized()
        
        # Try to get real Digital Twin first
        if self.twin_factory:
            try:
                twin = self.twin_factory.create_twin(patient_id)
                if twin:
                    # Convert DigitalPatientTwin to API format
                    api_twin = await self._convert_twin_to_api_format(twin)
                    # Broadcast real-time update
                    await self.notify_twin_update(patient_id, api_twin)
                    return api_twin
            except Exception as e:
                logger.warning(f"Error creating digital twin for {patient_id}: {e}")
        
        # Fallback to cached/mock data
        info = self._cache.get('patients', {}).get(patient_id)
        if not info:
            # Return a mock patient for demo
            info = {'study_id': 'Study_1', 'data': {}}
        
        summary = await self._build_patient_summary(patient_id, info)

        # Add detailed information (empty/defaults when not available)
        summary.update({
            'randomization_date': None,
            'treatment_arm': None,
            'current_status': 'Unknown',
            'visits': [],
            'queries': [],
            'saes': [],
            'coding_status': {},
            'forms_status': {},
            'timeline': [],
            'ai_insights': []
        })
        
        return summary
    
    async def get_all_twins(self, study_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all digital twins for a study (real-time generation)"""
        await self._ensure_initialized()
        
        if not self.twin_factory:
            logger.warning("DigitalTwinFactory not available, returning mock data")
            return []
        
        try:
            # Get all patients for the study
            patients = []
            if study_id:
                # Filter patients by study
                all_patients = self._cache.get('patients', {})
                patients = [pid for pid, info in all_patients.items() if info.get('study_id') == study_id]
            else:
                # Get all patients
                patients = list(self._cache.get('patients', {}).keys())
            
            twins = []
            for patient_id in patients[:50]:  # Limit to 50 for performance
                try:
                    twin = self.twin_factory.create_twin(patient_id)
                    if twin:
                        api_twin = await self._convert_twin_to_api_format(twin)
                        twins.append(api_twin)
                except Exception as e:
                    logger.warning(f"Error creating twin for {patient_id}: {e}")
            
            logger.info(f"Generated {len(twins)} digital twins")
            return twins
            
        except Exception as e:
            logger.error(f"Error generating twins: {e}")
            return []
    
    async def notify_twin_update(self, patient_id: str, twin_data: Dict[str, Any]):
        """Notify connected clients about twin updates"""
        from api.services.realtime_service import RealtimeService
        
        # Get realtime service instance (assuming it's available globally)
        # This would need to be injected properly in a real implementation
        try:
            realtime_service = RealtimeService()
            update_message = {
                'type': 'twin_update',
                'patient_id': patient_id,
                'twin_data': twin_data,
                'timestamp': datetime.now().isoformat()
            }
            await realtime_service.broadcast_update(update_message)
        except Exception as e:
            logger.warning(f"Could not broadcast twin update: {e}")
    
    async def _convert_twin_to_api_format(self, twin: DigitalPatientTwin) -> Dict[str, Any]:
        """Convert DigitalPatientTwin to API response format"""
        # Extract basic patient info
        patient_id = twin.subject_id
        study_id = getattr(twin, 'study_id', 'Study_1')
        site_id = getattr(twin, 'site_id', None)
        
        # Build clean status from twin data
        clean_status = {
            'is_clean': not twin.clean_status,  # twin.clean_status is True when NOT clean
            'cleanliness_score': 100.0 if twin.clean_status else 50.0,  # Simplified scoring
            'status': 'clean' if not twin.clean_status else 'dirty',
            'blocking_factors': [item.description for item in twin.blocking_items] if twin.blocking_items else [],
            'missing_visits': len([item for item in twin.blocking_items if 'visit' in item.description.lower()]) if twin.blocking_items else 0,
            'open_queries': len([item for item in twin.blocking_items if 'query' in item.description.lower()]) if twin.blocking_items else 0,
            'uncoded_terms': len([item for item in twin.blocking_items if 'code' in item.description.lower()]) if twin.blocking_items else 0,
            'pending_saes': len([item for item in twin.blocking_items if 'sae' in item.description.lower()]) if twin.blocking_items else 0,
            'missing_forms': 0,  # Not available in twin
            'unsigned_forms': 0,  # Not available in twin
            'last_calculated': datetime.now().isoformat()
        }
        
        # Build risk metrics
        risk_metrics = {}
        if twin.risk_metrics:
            risk_metrics = {
                'query_aging_index': getattr(twin.risk_metrics, 'query_aging_index', 0.0),
                'protocol_deviation_count': getattr(twin.risk_metrics, 'protocol_deviation_count', 0),
                'manipulation_risk_score': getattr(twin.risk_metrics, 'manipulation_risk_score', 'Low')
            }
        
        return {
            'patient_id': patient_id,
            'study_id': study_id,
            'site_id': site_id,
            'site_name': f"Clinical Site {site_id}" if site_id else 'Unknown',
            'country': getattr(twin, 'country', None),
            'region': getattr(twin, 'region', None),
            'enrollment_date': getattr(twin, 'enrollment_date', None),
            'clean_status': clean_status,
            'risk_metrics': risk_metrics,
            'randomization_date': getattr(twin, 'randomization_date', None),
            'treatment_arm': getattr(twin, 'treatment_arm', None),
            'current_status': str(twin.status) if twin.status else 'Unknown',
            'visits': [],  # Would need to extract from graph
            'queries': [],  # Would need to extract from graph
            'saes': [],  # Would need to extract from graph
            'coding_status': {},
            'forms_status': {},
            'timeline': [],
            'ai_insights': [],
            'twin_generated': True,
            'generated_at': datetime.now().isoformat()
        }
    
    async def calculate_clean_status(self, patient_id: str) -> Dict[str, Any]:
        """Calculate clean patient status"""
        patient = await self.get_patient_detail(patient_id)
        if patient:
            return patient['clean_status']
        
        return {
            'is_clean': False,
            'cleanliness_score': 0,
            'status': 'pending',
            'blocking_factors': ['Patient not found'],
            'missing_visits': 0,
            'open_queries': 0,
            'uncoded_terms': 0,
            'pending_saes': 0,
            'missing_forms': 0,
            'unsigned_forms': 0,
            'last_calculated': datetime.now().isoformat()
        }
    
    async def get_dirty_patients(self, study_id: Optional[str], limit: int) -> List[Dict]:
        """Get patients with dirty status"""
        result = await self.get_patients(
            filters={'study_id': study_id, 'status': 'dirty'} if study_id else {'status': 'dirty'},
            page=1,
            page_size=limit,
            sort_by='cleanliness_score',
            sort_order='asc'
        )
        return result['patients']
    
    async def get_at_risk_patients(self, study_id: Optional[str], threshold: float) -> List[Dict]:
        """Get at-risk patients"""
        result = await self.get_patients(
            filters={'study_id': study_id} if study_id else {},
            page=1,
            page_size=100,
            sort_by='cleanliness_score',
            sort_order='asc'
        )
        return [p for p in result['patients'] if p['clean_status']['cleanliness_score'] < threshold * 100]
    
    async def get_status_changes(self, study_id: Optional[str], hours: int, limit: int) -> List[Dict]:
        """Get recent status changes"""
        return []
    
    async def get_sites(self, filters: Dict, sort_by: str, sort_order: str) -> List[Dict]:
        """Get sites with filters"""
        await self._ensure_initialized()
        
        sites = []
        for site_key, info in self._cache.get('sites', {}).items():
            site = await self._build_site_summary(site_key, info)
            
            if filters.get('study_id') and info['study_id'] != filters['study_id']:
                continue
            if filters.get('country') and site.get('country') != filters['country']:
                continue
            if filters.get('region') and site.get('region') != filters['region']:
                continue
            if filters.get('status') and site.get('status') != filters['status']:
                continue
            if filters.get('risk_level') and site.get('risk_level') != filters['risk_level']:
                continue
            if filters.get('min_patients') is not None and site.get('total_patients', 0) < filters['min_patients']:
                continue
            
            sites.append(site)
        
        return sites[:50]  # Limit results
    
    async def _build_site_summary(self, site_key: str, info: Dict) -> Dict[str, Any]:
        """Build site summary"""
        cpid_df = info.get('data', {}).get('cpid_metrics')
        cols = self._get_cpid_columns(cpid_df) if isinstance(cpid_df, pd.DataFrame) else {}

        total_patients = 0
        clean_patients = 0
        dirty_patients = 0
        open_queries = 0
        uncoded_terms = 0
        missing_visits = 0
        missing_pages = 0
        total_queries = 0
        pending_saes = 0
        country = 'Unknown'
        region = 'Unknown'

        if isinstance(cpid_df, pd.DataFrame) and cols.get("site"):
            site_col = cols["site"]
            site_rows = cpid_df[cpid_df[site_col].astype(str).str.strip() == str(info['site_id'])]

            if not site_rows.empty:
                if cols.get("country"):
                    country = str(site_rows[cols["country"]].iloc[0])
                if cols.get("region"):
                    region = str(site_rows[cols["region"]].iloc[0])

                if cols.get("patient"):
                    total_patients = site_rows[cols["patient"]].nunique()

                if cols.get("open_queries"):
                    open_queries = self._safe_int(site_rows[cols["open_queries"]].fillna(0).sum())

                if cols.get("total_queries"):
                    total_queries = self._safe_int(site_rows[cols["total_queries"]].fillna(0).sum())

                if cols.get("uncoded_terms"):
                    uncoded_terms = self._safe_int(site_rows[cols["uncoded_terms"]].fillna(0).sum())

                if cols.get("missing_visits"):
                    missing_visits = self._safe_int(site_rows[cols["missing_visits"]].fillna(0).sum())

                if cols.get("missing_pages"):
                    missing_pages = self._safe_int(site_rows[cols["missing_pages"]].fillna(0).sum())

                if cols.get("patient"):
                    dirty_mask = pd.Series([False] * len(site_rows), index=site_rows.index)
                    if cols.get("open_queries"):
                        dirty_mask |= (site_rows[cols["open_queries"]].fillna(0) > 0)
                    if cols.get("missing_visits"):
                        dirty_mask |= (site_rows[cols["missing_visits"]].fillna(0) > 0)
                    if cols.get("missing_pages"):
                        dirty_mask |= (site_rows[cols["missing_pages"]].fillna(0) > 0)
                    if cols.get("uncoded_terms"):
                        dirty_mask |= (site_rows[cols["uncoded_terms"]].fillna(0) > 0)

                    dirty_patients = site_rows[dirty_mask][cols["patient"]].nunique()
                    clean_patients = total_patients - dirty_patients

        if total_queries == 0:
            total_queries = open_queries

        cleanliness_rate = (clean_patients / total_patients * 100) if total_patients > 0 else 0
        query_resolution_rate = ((total_queries - open_queries) / total_queries * 100) if total_queries > 0 else 0
        dqi_score = 85.0
        if total_patients > 0:
            query_penalty = min(20, (open_queries / max(1, total_patients)) * 10)
            missing_penalty = min(15, ((missing_visits + missing_pages) / max(1, total_patients)) * 5)
            uncoded_penalty = min(10, (uncoded_terms / max(1, total_patients)) * 5)
            dqi_score = max(50, 100 - query_penalty - missing_penalty - uncoded_penalty)

        performance_score = (dqi_score + cleanliness_rate + query_resolution_rate) / 3 if total_patients > 0 else 0

        if dqi_score >= 85:
            risk_level = 'low'
        elif dqi_score >= 70:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        verification_pct = 0
        if isinstance(cpid_df, pd.DataFrame) and cols.get("verification_pct") and cols.get("site"):
            site_rows = cpid_df[cpid_df[cols["site"]].astype(str).str.strip() == str(info['site_id'])]
            if not site_rows.empty:
                verification_pct = site_rows[cols["verification_pct"]].fillna(0).mean()
                verification_pct = float(verification_pct)
                if verification_pct <= 1:
                    verification_pct *= 100

        return {
            'site_id': info['site_id'],
            'site_name': f"Clinical Site {info['site_id']}",
            'study_id': info['study_id'],
            'country': country,
            'region': region,
            'status': 'active',
            'total_patients': total_patients,
            'clean_patients': clean_patients,
            'dirty_patients': dirty_patients,
            'cleanliness_rate': round(cleanliness_rate, 1),
            'dqi_score': round(dqi_score, 1),
            'open_queries': open_queries,
            'pending_saes': pending_saes,
            'risk_level': risk_level,
            'performance': {
                'query_resolution_rate': round(query_resolution_rate, 1),
                'query_resolution_velocity': 0.0,
                'enrollment_rate': 0.0,
                'data_entry_timeliness': round(verification_pct, 1) if verification_pct else 0.0,
                'sae_reporting_timeliness': 0.0,
                'overall_score': round(performance_score, 1) if performance_score else 0.0
            }
        }
    
    async def get_site_detail(self, site_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed site information"""
        await self._ensure_initialized()
        
        for site_key, info in self._cache.get('sites', {}).items():
            if info['site_id'] == site_id:
                summary = await self._build_site_summary(site_key, info)
                summary.update({
                    'address': None,
                    'principal_investigator': None,
                    'activation_date': None,
                    'target_enrollment': summary['total_patients'],
                    'metrics': {
                        'dqi': summary['dqi_score'],
                        'cleanliness': summary['cleanliness_rate'],
                        'query_resolution_rate': summary['performance']['query_resolution_rate']
                    },
                    'trends': {
                        'dqi': [],
                        'cleanliness': []
                    },
                    'risk_indicators': [],
                    'cra_assignments': [],
                    'recent_activity': []
                })
                return summary
        
        return None
    
    async def get_high_risk_sites(self, study_id: Optional[str], limit: int) -> List[Dict]:
        """Get high risk sites"""
        sites = await self.get_sites({'study_id': study_id} if study_id else {}, 'dqi_score', 'asc')
        return [s for s in sites if s['risk_level'] == 'high'][:limit]
    
    async def get_slow_resolution_sites(self, study_id: Optional[str], threshold: float, limit: int) -> List[Dict]:
        """Get sites with slow query resolution"""
        sites = await self.get_sites({'study_id': study_id} if study_id else {}, 'dqi_score', 'asc')
        return [s for s in sites if s['performance']['query_resolution_velocity'] > threshold][:limit]
    
    async def get_site_comparison(self, study_id: str, metric: str) -> List[Dict]:
        """Get site comparison for ranking"""
        sites = await self.get_sites({'study_id': study_id}, metric, 'desc')
        return [
            {
                'site_id': s['site_id'],
                'site_name': s['site_name'],
                'dqi_score': s['dqi_score'],
                'cleanliness_rate': s['cleanliness_rate'],
                'query_resolution_rate': s['performance']['query_resolution_rate'],
                'enrollment_progress': round(s['cleanliness_rate'], 1),
                'rank': i + 1
            }
            for i, s in enumerate(sites)
        ]
    
    async def get_drill_down_data(self, level: str, parent_id: Optional[str]) -> List[Dict]:
        """Get hierarchical drill-down data"""
        if level == 'study':
            return await self.get_all_studies()
        elif level == 'region':
            return await self.get_study_regions(parent_id or 'Study_1')
        elif level == 'site':
            return await self.get_sites({'study_id': parent_id} if parent_id else {}, 'dqi_score', 'desc')
        elif level == 'subject':
            result = await self.get_patients({'site_id': parent_id} if parent_id else {}, 1, 50, 'cleanliness_score', 'desc')
            return result['patients']
        
        return []
    
    async def get_filter_options(self) -> Dict[str, List[str]]:
        """Get available filter options"""
        await self._ensure_initialized()

        countries = set()
        regions = set()

        for _, data in self._cache.get('studies', {}).items():
            cpid_df = data.get('cpid_metrics')
            if cpid_df is None or not isinstance(cpid_df, pd.DataFrame) or cpid_df.empty:
                continue
            cols = self._get_cpid_columns(cpid_df)
            if cols.get("country"):
                countries.update([str(c) for c in cpid_df[cols["country"]].dropna().unique().tolist()])
            if cols.get("region"):
                regions.update([str(r) for r in cpid_df[cols["region"]].dropna().unique().tolist()])

        return {
            'studies': list(self._cache.get('studies', {}).keys()),
            'countries': sorted(countries),
            'regions': sorted(regions),
            'risk_levels': ['low', 'medium', 'high'],
            'statuses': ['clean', 'dirty', 'pending']
        }
    
    async def get_risk_assessment(self, study_id: str) -> Dict[str, Any]:
        """Get risk assessment for study"""
        summary = await self.get_dashboard_summary(study_id)

        overall_dqi = summary.get('overall_dqi', 0)
        open_queries = summary.get('open_queries', 0)
        pending_saes = summary.get('pending_saes', 0)

        if overall_dqi >= 85:
            overall_risk = 'low'
        elif overall_dqi >= 70:
            overall_risk = 'medium'
        else:
            overall_risk = 'high'

        risk_factors = []
        recommendations = []

        if open_queries > 0:
            risk_factors.append({'factor': 'Open queries', 'severity': 'medium', 'impact': 'Potential data delays'})
            recommendations.append('Prioritize query resolution for high-volume sites')

        if pending_saes > 0:
            risk_factors.append({'factor': 'Pending SAE reconciliation', 'severity': 'high', 'impact': 'Safety oversight risk'})
            recommendations.append('Review and reconcile pending SAEs')

        risk_score = max(0, min(100, int(100 - overall_dqi)))

        return {
            'overall_risk': overall_risk,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
    
    async def get_heatmap_data(self, study_id: str, metric: str) -> Dict[str, Any]:
        """Get heatmap visualization data"""
        sites = await self.get_sites({'study_id': study_id}, metric, 'desc')

        def build_matrix_from_sites(site_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            rows = [s.get('site_name') or s.get('site_id') for s in site_rows]
            columns = [metric.upper()]
            values = [[float(s.get(metric, s.get('dqi_score', 0)))] for s in site_rows]
            return {
                'rows': rows,
                'columns': columns,
                'values': values,
                'metric': metric
            }

        if sites:
            return build_matrix_from_sites(sites)

        # Fallback: derive from reports knowledge graph
        reports_candidates = [
            PROJECT_ROOT.parent / 'reports',
            PROJECT_ROOT / 'reports'
        ]
        report_dir = next((d for d in reports_candidates if d.exists()), None)
        if not report_dir:
            return {'rows': [], 'columns': [metric.upper()], 'values': [], 'metric': metric}

        report_path = report_dir / f"{study_id}_knowledge_graph.json"
        if not report_path.exists():
            # Use first available graph as fallback
            graphs = sorted(report_dir.glob('*_knowledge_graph.json'))
            report_path = graphs[0] if graphs else None
        if not report_path or not report_path.exists():
            return {'rows': [], 'columns': [metric.upper()], 'values': [], 'metric': metric}

        try:
            import json
            with open(report_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            site_acc = {}
            for node in graph_data.get('nodes', []):
                if node.get('node_type') != 'Patient':
                    continue
                attrs = node.get('attributes', {})
                site_id = attrs.get('site_id')
                if not site_id:
                    continue
                entry = site_acc.setdefault(site_id, {
                    'dqi_sum': 0.0,
                    'clean_sum': 0.0,
                    'open_queries': 0.0,
                    'count': 0
                })
                entry['dqi_sum'] += float(attrs.get('data_quality_index', 0) or 0)
                entry['clean_sum'] += float(attrs.get('clean_percentage', 0) or 0)
                entry['open_queries'] += float(attrs.get('open_queries', 0) or 0)
                entry['count'] += 1

            derived_sites = []
            for site_id, acc in site_acc.items():
                count = max(1, acc['count'])
                dqi_avg = acc['dqi_sum'] / count
                clean_avg = acc['clean_sum'] / count
                queries_total = acc['open_queries']
                if metric in {'dqi', 'dqi_score', 'data_quality'}:
                    value = dqi_avg
                elif metric in {'cleanliness', 'clean_rate'}:
                    value = clean_avg
                elif metric in {'queries', 'open_queries'}:
                    value = queries_total
                elif metric == 'risk':
                    value = max(0.0, 100.0 - dqi_avg)
                else:
                    value = dqi_avg

                derived_sites.append({
                    'site_id': site_id,
                    'site_name': site_id,
                    metric: value,
                    'dqi_score': dqi_avg
                })

            return build_matrix_from_sites(derived_sites)
        except Exception as e:
            logger.warning(f"Failed to build heatmap from reports: {e}")
            return {'rows': [], 'columns': [metric.upper()], 'values': [], 'metric': metric}
    
    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self._initialized:
            await self.initialize()
    
    # Additional stub methods for completeness
    async def get_patient_timeline(self, patient_id: str) -> List[Dict]:
        patient = await self.get_patient_detail(patient_id)
        return patient.get('timeline', []) if patient else []
    
    async def get_blocking_factors(self, patient_id: str) -> List[Dict]:
        patient = await self.get_patient_detail(patient_id)
        if patient:
            return [{'factor': f, 'category': 'general'} for f in patient['clean_status']['blocking_factors']]
        return []
    
    async def get_lock_readiness(self, patient_id: str) -> Dict:
        status = await self.calculate_clean_status(patient_id)
        return {
            'ready_for_lock': status['is_clean'],
            'readiness_score': status['cleanliness_score'],
            'blockers': status['blocking_factors']
        }
    
    async def get_site_patients(self, site_id: str, status: Optional[str]) -> List[Dict]:
        result = await self.get_patients({'site_id': site_id, 'status': status} if status else {'site_id': site_id}, 1, 100, 'patient_id', 'asc')
        return result['patients']
    
    async def get_site_queries(self, site_id: str, status: Optional[str]) -> List[Dict]:
        return []
    
    async def get_cra_activity(self, site_id: str, days: int) -> List[Dict]:
        return []
    
    async def get_site_issues(self, site_id: str) -> List[Dict]:
        return []
