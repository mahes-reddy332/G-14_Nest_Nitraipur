import axios from 'axios'
import type {
  DashboardSummary,
  Study,
  StudyMetrics,
  SourceFile,
  Patient,
  PatientDetail,
  CleanPatientStatus,
  Site,
  SitePerformance,
  KPITile,
  DQIMetrics,
  CleanlinessMetrics,
  QueryMetrics,
  SAEMetrics,
  CodingMetrics,
  OperationalVelocity,
  Alert,
  AlertSummary,
  AgentStatus,
  AgentInsight,
  AgentRecommendation,
  HeatmapData,
  ReportSummary,
  ReportDetail,
  StudyReportSummary,
  StudyReportDetail,
} from '../types'
import { apiRequestWithRetry, RetryConfig } from '../utils/apiRetry'
import { normalizeApiResponse } from '../utils/apiResponse'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'
const DEBUG_API = import.meta.env.VITE_DEBUG_API === 'true'

const logApi = (...args: any[]) => {
  if (DEBUG_API) {
    console.debug('[api]', ...args)
  }
}

const coerceNumber = (value: unknown, field: string): number => {
  if (value === null || value === undefined) {
    throw new Error(`Missing numeric field: ${field}`)
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) {
    return Number(value)
  }
  throw new Error(`Invalid numeric field: ${field}`)
}

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Enhanced API call with retry logic
async function apiCall<T = any>(
  config: Parameters<typeof apiRequestWithRetry>[0],
  operationName: string,
  retryConfig?: RetryConfig
): Promise<T> {
  logApi('request', operationName, config)
  const response = await apiRequestWithRetry(
    { ...config, baseURL: API_BASE_URL },
    retryConfig
  )
  logApi('response', operationName, response.data)
  return response.data
}

// ==================== PERFORMANCE OPTIMIZED BUNDLED ENDPOINT ====================
// This fetches all initial dashboard data in a SINGLE request for fast initial load
export interface InitialDashboardData {
  success: boolean
  timestamp: string
  study_filter: string | null
  summary: {
    total_studies: number
    total_patients: number
    total_sites: number
    clean_patients: number
    dirty_patients: number
    overall_dqi: number
    open_queries: number
    pending_saes: number
    uncoded_terms: number
  }
  query_metrics: {
    total_queries: number
    open_queries: number
    closed_queries: number
    resolution_rate: number
    avg_resolution_time: number
    aging_distribution: Record<string, number>
    velocity_trend: number[]
  }
  cleanliness: {
    cleanliness_rate: number
    total_patients: number
    clean_patients: number
    dirty_patients: number
    at_risk_count: number
    trend: number[]
  }
  alerts: {
    active_alerts: number
    critical_count: number
    high_count: number
  }
  _cache_hit: boolean
  _response_time_ms: number
}

export const dashboardInitialLoadApi = {
  /**
   * Fetch ALL initial dashboard data in a single bundled request
   * This is the PRIMARY endpoint for dashboard initialization
   * Performance: ~100-300ms vs 3-5s with individual calls
   */
  getInitialLoad: async (studyId?: string): Promise<InitialDashboardData> => {
    const { data } = await api.get('/dashboard/initial-load', {
      params: { study_id: studyId }
    })
    logApi('initial-load', data)
    return data
  },
}

// Dashboard API
export const dashboardApi = {
  getSummary: async (): Promise<DashboardSummary> => {
    return apiCall({ method: 'GET', url: '/dashboard/summary' }, 'getDashboardSummary')
  },
}

// Reports API
export const reportsApi = {
  list: async (): Promise<ReportSummary[]> => {
    return apiCall({ method: 'GET', url: '/reports/' }, 'listReports')
  },

  getById: async (reportId: string): Promise<ReportDetail> => {
    return apiCall({ method: 'GET', url: `/reports/artifacts/${reportId}` }, `getReport-${reportId}`)
  },

  listStudyReports: async (): Promise<StudyReportSummary[]> => {
    return apiCall({ method: 'GET', url: '/reports/studies' }, 'listStudyReports')
  },

  getStudyReport: async (studyId: string): Promise<StudyReportDetail> => {
    return apiCall({ method: 'GET', url: `/reports/studies/${studyId}` }, `getStudyReport-${studyId}`)
  },

  exportStudyReportCsv: async (studyId: string): Promise<Blob> => {
    const response = await api.get(`/reports/studies/${studyId}/export`, {
      params: { format: 'csv' },
      responseType: 'blob',
    })
    return response.data
  },
}

// Studies API
export const studiesApi = {
  getAll: async (): Promise<Study[]> => {
    const data = await apiCall({ method: 'GET', url: '/studies/' }, 'getAllStudies')
    // Map backend StudySummary to frontend Study type
    const mapped = (Array.isArray(data) ? data : []).map((s: any) => ({
      study_id: s.study_id,
      name: s.study_name || s.name || s.study_id,
      phase: s.phase || 'Unknown',
      status: s.status || 'unknown',
      therapeutic_area: s.therapeutic_area || 'Unknown',
      total_patients: s.total_patients ?? 0,
      total_sites: s.total_sites ?? 0,
      clean_patients: s.clean_patients ?? 0,
      dirty_patients: s.dirty_patients ?? 0,
      dqi_score: s.dqi_score ?? s.overall_dqi ?? 0,
      last_updated: s.last_updated || new Date().toISOString(),
    }))

    return mapped.sort((a, b) => {
      const aNum = Number.parseInt(a.study_id.replace(/\D/g, ''), 10)
      const bNum = Number.parseInt(b.study_id.replace(/\D/g, ''), 10)
      if (Number.isFinite(aNum) && Number.isFinite(bNum)) {
        return aNum - bNum
      }
      return a.study_id.localeCompare(b.study_id)
    })
  },

  getById: async (studyId: string): Promise<Study> => {
    const data = await apiCall({ method: 'GET', url: `/studies/${studyId}` }, `getStudyById-${studyId}`)
    // Map backend response fields to frontend Study type
    const totalPatients = data.total_patients ?? data.current_enrollment ?? 0
    const cleanlinessRate = data.metrics?.cleanliness ?? data.cleanliness ?? 0
    const cleanPatients = data.clean_patients ?? Math.round((totalPatients * cleanlinessRate) / 100)
    const dqiScore = data.dqi_score ?? data.overall_dqi ?? data.metrics?.dqi ?? 0

    return {
      study_id: data.study_id,
      name: data.name || data.study_name || data.study_id,
      phase: data.phase || 'Unknown',
      status: data.status || 'active',
      therapeutic_area: data.therapeutic_area || 'Unknown',
      total_patients: totalPatients,
      total_sites: data.total_sites || 0,
      clean_patients: cleanPatients,
      dirty_patients: data.dirty_patients ?? (totalPatients - cleanPatients),
      dqi_score: dqiScore,
      last_updated: data.last_updated || new Date().toISOString(),
    }
  },

  getMetrics: async (studyId: string): Promise<StudyMetrics> => {
    const { data } = await api.get(`/studies/${studyId}/metrics`)
    return data
  },

  getRegions: async (studyId: string) => {
    const { data } = await api.get(`/studies/${studyId}/regions`)
    return data
  },

  getTrends: async (studyId: string, days = 30) => {
    const { data } = await api.get(`/studies/${studyId}/trends`, { params: { days } })
    return data
  },

  getRiskAssessment: async (studyId: string) => {
    const { data } = await api.get(`/studies/${studyId}/risk-assessment`)
    return data
  },

  getHeatmap: async (studyId: string, metric = 'dqi') => {
    const { data } = await api.get(`/studies/${studyId}/heatmap`, { params: { metric } })
    return data
  },

  // Get source files for a specific study
  getSourceFiles: async (studyId: string) => {
    const { data } = await api.get(`/studies/${studyId}/source-files`)
    return data
  },

  // Get sites for a specific study
  getSites: async (studyId: string): Promise<Site[]> => {
    try {
      const { data } = await api.get(`/studies/${studyId}/sites`)
      return (Array.isArray(data) ? data : []).map((s: any) => ({
        site_id: s.site_id,
        name: s.name || s.site_name || s.site_id,
        country: s.country || 'Unknown',
        region: s.region || 'Unknown',
        status: s.status || 'active',
        patient_count: s.patient_count || s.total_patients || 0,
        dqi_score: s.dqi_score || s.overall_dqi || 0,
        query_resolution_time: s.avg_query_resolution_days || 0,
        last_updated: s.last_updated || new Date().toISOString(),
      }))
    } catch (error) {
      // Fallback to all sites filtered by study
      return sitesApi.getAll({ study_id: studyId })
    }
  },
}

// Patients API
export const patientsApi = {
  getAll: async (params?: {
    study_id?: string
    site_id?: string
    status?: string
    skip?: number
    limit?: number
    page?: number
    page_size?: number
  }): Promise<{ patients: Patient[]; total: number; clean_patients?: number; dirty_patients?: number; at_risk_patients?: number }> => {
    const { data } = await api.get('/patients/', { params })
    // Backend returns PatientListResponse with nested patients array
    // Map backend PatientSummary to frontend Patient type
    const patients = (data.patients || data || []).map((p: any) => ({
      patient_id: p.patient_id,
      study_id: p.study_id,
      site_id: p.site_id,
      enrollment_date: p.enrollment_date || '',
      status: p.clean_status?.status || 'unknown',
      is_clean: p.clean_status?.is_clean ?? false,
      cleanliness_score: p.clean_status?.cleanliness_score ?? 0,
      visit_count: p.total_visits || p.completed_visits || 0,
      query_count: p.clean_status?.open_queries ?? 0,
      last_updated: p.last_visit_date || '',
    }))

    return {
      patients,
      total: typeof data?.total === 'number' ? data.total : patients.length,
      clean_patients: typeof data?.clean_patients === 'number' ? data.clean_patients : undefined,
      dirty_patients: typeof data?.dirty_patients === 'number' ? data.dirty_patients : undefined,
      at_risk_patients: typeof data?.at_risk_patients === 'number' ? data.at_risk_patients : undefined,
    }
  },

  getById: async (patientId: string): Promise<PatientDetail> => {
    const { data } = await api.get(`/patients/${patientId}`)
    return data
  },

  getCleanStatus: async (patientId: string): Promise<CleanPatientStatus> => {
    const { data } = await api.get(`/patients/${patientId}/clean-status`)
    // Map backend response to frontend CleanPatientStatus type
    return {
      patient_id: patientId,
      is_clean: data.is_clean ?? false,
      cleanliness_score: data.cleanliness_score ?? 0,
      blocking_factors: (data.blocking_factors || []).map((f: any) =>
        typeof f === 'string'
          ? { factor_type: f, description: f, severity: 'medium', domain: 'general', resolution_action: 'Review required' }
          : f
      ),
      last_checked: data.last_calculated || new Date().toISOString(),
      lock_readiness: data.is_clean ? 'Ready' : 'Not Ready',
    }
  },

  getDirtyPatients: async (studyId?: string, limit?: number): Promise<Patient[]> => {
    const { data } = await api.get('/patients/dirty', { params: { study_id: studyId, limit } })
    // Map backend response to frontend Patient type
    return (Array.isArray(data) ? data : []).map((p: any) => ({
      patient_id: p.patient_id,
      study_id: p.study_id,
      site_id: p.site_id,
      enrollment_date: p.enrollment_date || '',
      status: p.clean_status?.status || 'dirty',
      is_clean: false,
      cleanliness_score: p.clean_status?.cleanliness_score ?? 0,
      visit_count: p.total_visits || 0,
      query_count: p.clean_status?.open_queries ?? 0,
      last_updated: p.last_visit_date || '',
    }))
  },

  getAtRiskPatients: async (studyId?: string): Promise<Patient[]> => {
    const { data } = await api.get('/patients/at-risk', { params: { study_id: studyId } })
    return (Array.isArray(data) ? data : []).map((p: any) => ({
      patient_id: p.patient_id,
      study_id: p.study_id,
      site_id: p.site_id,
      enrollment_date: p.enrollment_date || '',
      status: 'at-risk',
      is_clean: false,
      cleanliness_score: p.clean_status?.cleanliness_score ?? 0,
      visit_count: p.total_visits || 0,
      query_count: p.clean_status?.open_queries ?? 0,
      last_updated: p.last_visit_date || '',
    }))
  },

  getStatusChanges: async (hours = 24) => {
    const { data } = await api.get('/patients/status-changes', { params: { hours } })
    return data
  },

  getBlockingFactors: async (studyId?: string) => {
    const { data } = await api.get('/patients/blocking-factors', { params: { study_id: studyId } })
    return data
  },

  getLockReadiness: async (studyId?: string) => {
    const { data } = await api.get('/patients/lock-readiness', { params: { study_id: studyId } })
    return data
  },
}

// Sites API
export const sitesApi = {
  getAll: async (params?: {
    study_id?: string
    region?: string
    country?: string
    skip?: number
    limit?: number
  }): Promise<Site[]> => {
    const { data } = await api.get('/sites/', { params })
    // Map backend SiteSummary to frontend Site type
    return (Array.isArray(data) ? data : []).map((s: any) => ({
      site_id: s.site_id,
      name: s.name || s.site_name || s.site_id,
      country: s.country || 'Unknown',
      region: s.region || 'Unknown',
      status: s.status || 'active',
      patient_count: s.patient_count || s.total_patients || 0,
      dqi_score: s.dqi_score || s.overall_dqi || 0,
      query_resolution_time: s.avg_query_resolution_days || s.query_resolution_time || 0,
      last_updated: s.last_updated || new Date().toISOString(),
    }))
  },

  getById: async (siteId: string): Promise<Site> => {
    const { data } = await api.get(`/sites/${siteId}`)
    return {
      site_id: data.site_id,
      name: data.name || data.site_name || data.site_id,
      country: data.country || 'Unknown',
      region: data.region || 'Unknown',
      status: data.status || 'active',
      patient_count: data.patient_count || data.total_patients || 0,
      dqi_score: data.dqi_score || data.overall_dqi || 0,
      query_resolution_time: data.avg_query_resolution_days || data.query_resolution_time || 0,
      last_updated: data.last_updated || new Date().toISOString(),
    }
  },

  getPerformance: async (siteId: string): Promise<SitePerformance> => {
    const { data } = await api.get(`/sites/${siteId}/performance`)
    return data
  },

  getHighRisk: async (studyId?: string): Promise<Site[]> => {
    const { data } = await api.get('/sites/high-risk', { params: { study_id: studyId } })
    return data
  },

  getSlowResolution: async (studyId?: string, threshold = 7): Promise<Site[]> => {
    const { data } = await api.get('/sites/slow-resolution', {
      params: { study_id: studyId, threshold },
    })
    return data
  },

  getComparison: async (studyId?: string): Promise<SitePerformance[]> => {
    const { data } = await api.get('/sites/comparison', { params: { study_id: studyId } })
    return data
  },

  getCraActivity: async (studyId?: string) => {
    const { data } = await api.get('/sites/cra-activity', { params: { study_id: studyId } })
    return data
  },
}

// Metrics API
export const metricsApi = {
  /**
   * Get bundled dashboard summary - OPTIMIZED for initial load
   * Uses the bundled endpoint when possible for better performance
   */
  getDashboardSummary: async (studyId?: string): Promise<DashboardSummary> => {
    try {
      // Try the optimized bundled endpoint first
      const initialData = await dashboardInitialLoadApi.getInitialLoad(studyId)
      const summary = initialData.summary
      if (!summary) {
        throw new Error('Missing summary payload in initial-load response')
      }
      return {
        total_studies: coerceNumber(summary.total_studies, 'summary.total_studies'),
        total_patients: coerceNumber(summary.total_patients, 'summary.total_patients'),
        total_sites: coerceNumber(summary.total_sites, 'summary.total_sites'),
        clean_patients: coerceNumber(summary.clean_patients, 'summary.clean_patients'),
        dirty_patients: coerceNumber(summary.dirty_patients, 'summary.dirty_patients'),
        overall_dqi: coerceNumber(summary.overall_dqi, 'summary.overall_dqi'),
        open_queries: coerceNumber(summary.open_queries, 'summary.open_queries'),
        pending_saes: coerceNumber(summary.pending_saes, 'summary.pending_saes'),
        uncoded_terms: coerceNumber(summary.uncoded_terms, 'summary.uncoded_terms'),
        // Include query metrics from bundled response
        _query_metrics: initialData.query_metrics,
        _cleanliness: initialData.cleanliness,
        _alerts: initialData.alerts,
        _response_time_ms: initialData._response_time_ms,
      } as DashboardSummary
    } catch {
      // Fallback to individual endpoint
      const { data } = await api.get('/dashboard/summary', { params: { study_id: studyId } })
      logApi('dashboard-summary', data)
      return {
        total_studies: coerceNumber(data.total_studies, 'summary.total_studies'),
        total_patients: coerceNumber(data.total_patients, 'summary.total_patients'),
        total_sites: coerceNumber(data.total_sites, 'summary.total_sites'),
        clean_patients: coerceNumber(data.clean_patients, 'summary.clean_patients'),
        dirty_patients: coerceNumber(data.dirty_patients, 'summary.dirty_patients'),
        overall_dqi: coerceNumber(data.overall_dqi, 'summary.overall_dqi'),
        open_queries: coerceNumber(data.open_queries, 'summary.open_queries'),
        pending_saes: coerceNumber(data.pending_saes, 'summary.pending_saes'),
        uncoded_terms: coerceNumber(data.uncoded_terms, 'summary.uncoded_terms'),
      } as DashboardSummary
    }
  },

  getKPITiles: async (studyId?: string): Promise<KPITile[]> => {
    const { data } = await api.get('/metrics/kpi-tiles', { params: { study_id: studyId } })
    return Array.isArray(data) ? data : []
  },

  getDQI: async (studyId?: string, siteId?: string, days?: number): Promise<DQIMetrics> => {
    const { data } = await api.get('/metrics/dqi', { params: { study_id: studyId, site_id: siteId, days } })
    // Map backend response to frontend DQIMetrics type
    return {
      overall_dqi: coerceNumber(data.overall_dqi, 'dqi.overall_dqi'),
      completeness: coerceNumber(data.completeness, 'dqi.completeness'),
      accuracy: coerceNumber(data.accuracy, 'dqi.accuracy'),
      consistency: coerceNumber(data.consistency, 'dqi.consistency'),
      timeliness: coerceNumber(data.timeliness, 'dqi.timeliness'),
      trend: Array.isArray(data.trend) ? data.trend : [],
      by_domain: {
        visits: coerceNumber(data.completeness, 'dqi.completeness'),
        queries: coerceNumber(data.consistency, 'dqi.consistency'),
        coding: coerceNumber(data.accuracy, 'dqi.accuracy'),
        safety: coerceNumber(data.conformity ?? data.timeliness, 'dqi.safety'),
        ...(data.by_domain ?? {}),
      },
    }
  },

  getCleanliness: async (studyId?: string, siteId?: string, days?: number): Promise<CleanlinessMetrics> => {
    const { data } = await api.get('/metrics/cleanliness', { params: { study_id: studyId, site_id: siteId, days } })
    const cleanCount = coerceNumber(data.clean_patients ?? data.clean_count, 'cleanliness.clean_patients')
    const dirtyCount = coerceNumber(data.dirty_patients ?? data.dirty_count, 'cleanliness.dirty_patients')
    const atRiskCount = coerceNumber(data.pending_patients ?? data.at_risk_count, 'cleanliness.at_risk_count')
    const total = cleanCount + dirtyCount + atRiskCount

    return {
      clean_count: cleanCount,
      dirty_count: dirtyCount,
      at_risk_count: atRiskCount,
      cleanliness_rate: total > 0 ? coerceNumber(data.overall_rate ?? data.cleanliness_rate, 'cleanliness.overall_rate') : 0,
      trend: Array.isArray(data.trend) ? data.trend : [],
      by_study: data.by_category ?? data.by_study ?? {},
    }
  },

  getQueries: async (studyId?: string, siteId?: string, days?: number): Promise<QueryMetrics> => {
    const { data } = await api.get('/metrics/queries', { params: { study_id: studyId, site_id: siteId, days } })
    // Map backend response to frontend QueryMetrics type
    return {
      total_queries: coerceNumber(data.total_queries, 'queries.total_queries'),
      open_queries: coerceNumber(data.open_queries, 'queries.open_queries'),
      closed_queries: coerceNumber(data.closed_queries, 'queries.closed_queries'),
      resolution_rate: coerceNumber(data.resolution_rate, 'queries.resolution_rate'),
      avg_resolution_time: coerceNumber(data.avg_resolution_time_days ?? data.avg_resolution_time, 'queries.avg_resolution_time'),
      aging_distribution: data.aging_distribution ?? {},
    }
  },

  getSAEs: async (studyId?: string): Promise<SAEMetrics> => {
    const { data } = await api.get('/metrics/saes', { params: { study_id: studyId } })
    return {
      total_saes: coerceNumber(data.total_saes, 'saes.total_saes'),
      reconciled: coerceNumber(data.reconciled, 'saes.reconciled'),
      pending: coerceNumber(data.pending, 'saes.pending'),
      overdue: coerceNumber(data.overdue, 'saes.overdue'),
      reconciliation_rate: coerceNumber(data.reconciliation_rate, 'saes.reconciliation_rate'),
      avg_reconciliation_days: coerceNumber(data.avg_reconciliation_days, 'saes.avg_reconciliation_days'),
      by_seriousness: data.by_seriousness ?? {},
    }
  },

  getCoding: async (studyId?: string): Promise<CodingMetrics> => {
    const { data } = await api.get('/metrics/coding', { params: { study_id: studyId } })
    return {
      total_terms: coerceNumber(data.total_terms, 'coding.total_terms'),
      coded: coerceNumber(data.coded, 'coding.coded'),
      uncoded: coerceNumber(data.uncoded, 'coding.uncoded'),
      completion_rate: coerceNumber(data.completion_rate, 'coding.completion_rate'),
      meddra_status: data.meddra_status ?? {},
      whodrug_status: data.whodrug_status ?? {},
      uncoded_breakdown: data.uncoded_breakdown ?? [],
    }
  },

  getVelocity: async (studyId?: string, days = 7, siteId?: string): Promise<OperationalVelocity> => {
    const { data } = await api.get('/metrics/velocity', { params: { study_id: studyId, site_id: siteId, days } })
    // Map backend response to frontend OperationalVelocity type
    // Generate trend data if not provided in expected format
    const trend = data.trend || []
    const mappedTrend = Array.isArray(trend) && trend.length > 0 && typeof trend[0] === 'object'
      ? trend
      : trend.map((v: number, i: number) => ({
        date: new Date(Date.now() - (trend.length - 1 - i) * 86400000).toISOString().split('T')[0],
        queries: Math.round(v),
        resolutions: 0,
      }))

    return {
      queries_per_day: coerceNumber(data.query_resolution_velocity ?? data.queries_per_day, 'velocity.queries_per_day'),
      resolutions_per_day: coerceNumber(data.resolutions_per_day, 'velocity.resolutions_per_day'),
      data_entries_per_day: coerceNumber(data.form_completion_velocity ?? data.data_entries_per_day, 'velocity.data_entries_per_day'),
      trend: mappedTrend,
    }
  },

  getHeatmap: async (studyId?: string, metric = 'dqi'): Promise<HeatmapData> => {
    const { data } = await api.get('/metrics/heatmap', { params: { study_id: studyId, metric } })
    return data
  },

  getAnomalies: async (studyId?: string) => {
    const { data } = await api.get('/metrics/anomalies', { params: { study_id: studyId } })
    return data
  },

  // DQI Breakdown for the enhanced component
  getDQIBreakdown: async (studyId?: string, siteId?: string, days?: number) => {
    try {
      const data = await metricsApi.getDQI(studyId, siteId, days)
      return {
        overall: data.overall_dqi,
        completeness: data.completeness,
        accuracy: data.accuracy,
        consistency: data.consistency,
        timeliness: data.timeliness,
        trend: Array.isArray(data.trend) ? data.trend : [],
      }
    } catch (error) {
      console.error('Failed to load DQI breakdown:', error)
      throw error
    }
  },
}

// Alerts API
export const alertsApi = {
  getAll: async (params?: {
    status?: string
    severity?: string
    category?: string
    limit?: number
  }): Promise<Alert[]> => {
    const { data } = await api.get('/alerts/', { params })
    return Array.isArray(data) ? data : []
  },

  getSummary: async (): Promise<AlertSummary> => {
    const { data } = await api.get('/alerts/summary')
    return data
  },

  getCritical: async (): Promise<Alert[]> => {
    const { data } = await api.get('/alerts/critical')
    return data
  },

  getRecent: async (hours = 24): Promise<Alert[]> => {
    const { data } = await api.get('/alerts/recent', { params: { hours } })
    return data
  },

  acknowledge: async (alertId: string, user: string, note?: string): Promise<Alert> => {
    const { data } = await api.post(`/alerts/${alertId}/acknowledge`, {
      acknowledged_by: user,
      note: note || ''
    })
    return data
  },

  resolve: async (alertId: string, user: string, notes?: string, actions?: string[]): Promise<Alert> => {
    const { data } = await api.post(`/alerts/${alertId}/resolve`, {
      resolved_by: user,
      resolution_note: notes || '',
      actions_taken: actions || [],
    })
    return data
  },

  dismiss: async (alertId: string, user: string, reason: string): Promise<Alert> => {
    const { data } = await api.post(`/alerts/${alertId}/dismiss`, null, {
      params: { dismissed_by: user, reason }
    })
    return data
  },

  getHistory: async (days = 30, category?: string): Promise<Alert[]> => {
    const { data } = await api.get('/alerts/history', { params: { days, category } })
    return data
  },

  getTrends: async (days = 7) => {
    const { data } = await api.get('/alerts/trends', { params: { days } })
    return data
  },
}

// Agents API
export const agentsApi = {
  getStatus: async (): Promise<Record<string, AgentStatus>> => {
    const { data } = await api.get('/agents/status')
    // Handle both array and object responses
    if (Array.isArray(data)) {
      const statusMap: Record<string, AgentStatus> = {}
      data.forEach((agent: any) => {
        statusMap[agent.agent_type || agent.name] = agent
      })
      return statusMap
    }
    return data || {}
  },

  getInsights: async (params?: {
    agent_type?: string
    priority?: string
    limit?: number
  }): Promise<AgentInsight[]> => {
    const { data } = await api.get('/agents/insights', { params })
    return Array.isArray(data) ? data : []
  },

  getRecommendations: async (params?: {
    category?: string
    study_id?: string
    limit?: number
  }): Promise<AgentRecommendation[]> => {
    const { data } = await api.get('/agents/recommendations', { params })
    return data
  },

  getExplainability: async (insightId: string) => {
    const { data } = await api.get(`/agents/explain/${insightId}`)
    return data
  },

  getReconciliationDiscrepancies: async (studyId?: string, severity?: string) => {
    const { data } = await api.get('/agents/reconciliation/discrepancies', {
      params: { study_id: studyId, severity },
    })
    return data
  },

  getCodingIssues: async (studyId?: string, status?: string) => {
    const { data } = await api.get('/agents/coding/issues', {
      params: { study_id: studyId, status },
    })
    return data
  },

}

export const labsApi = {
  getMissingLabData: async (studyId?: string) => {
    const { data } = await api.get('/labs/missing', { params: { study_id: studyId } })
    return data
  },
  getLabSummary: async (studyId?: string) => {
    const { data } = await api.get('/labs/summary', { params: { study_id: studyId } })
    return data
  },
}

export const safetyApi = {
  getSAEs: async (params?: { study_id?: string; site_id?: string; status?: string; severity?: string }) => {
    const { data } = await api.get('/safety/saes', { params })
    return Array.isArray(data) ? data : []
  },
  getSummary: async (studyId?: string) => {
    const { data } = await api.get('/safety/summary', { params: { study_id: studyId } })
    return data
  },
}

export const codingApi = {
  getMedDRA: async (params?: { study_id?: string; site_id?: string; status?: string }) => {
    const { data } = await api.get('/coding/meddra', { params })
    return normalizeApiResponse<any>(data)
  },
  getWHODrug: async (params?: { study_id?: string; site_id?: string; status?: string }) => {
    const { data } = await api.get('/coding/whodrug', { params })
    return normalizeApiResponse<any>(data)
  },
  getMetrics: async (studyId?: string) => {
    const { data } = await api.get('/coding/metrics', { params: { study_id: studyId } })
    return data
  },
}

export default api
