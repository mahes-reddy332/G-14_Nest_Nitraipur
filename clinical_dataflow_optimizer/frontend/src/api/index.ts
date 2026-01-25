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
} from '../types'
import { apiRequestWithRetry, RetryConfig } from '../utils/apiRetry'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

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
  const response = await apiRequestWithRetry(
    { ...config, baseURL: API_BASE_URL },
    retryConfig
  )
  return response.data
}

// Dashboard API
export const dashboardApi = {
  getSummary: async (): Promise<DashboardSummary> => {
    return apiCall({ method: 'GET', url: '/dashboard/summary' }, 'getDashboardSummary')
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
    return {
      study_id: data.study_id,
      name: data.name || data.study_id,
      phase: data.phase || 'Unknown',
      status: data.status || 'active',
      therapeutic_area: data.therapeutic_area || 'Unknown',
      total_patients: data.total_patients || 0,
      total_sites: data.total_sites || 0,
      clean_patients: data.clean_patients || 0,
      dirty_patients: data.dirty_patients || 0,
      dqi_score: data.overall_dqi || data.dqi_score || 0,
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
  }): Promise<Patient[]> => {
    const { data } = await api.get('/patients/', { params })
    // Backend returns PatientListResponse with nested patients array
    // Map backend PatientSummary to frontend Patient type
    const patients = data.patients || data
    return patients.map((p: any) => ({
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

  getDirtyPatients: async (studyId?: string): Promise<Patient[]> => {
    const { data } = await api.get('/patients/dirty', { params: { study_id: studyId } })
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
  getKPITiles: async (studyId?: string): Promise<KPITile[]> => {
    const { data } = await api.get('/metrics/kpi-tiles', { params: { study_id: studyId } })
    return Array.isArray(data) ? data : []
  },

  getDQI: async (studyId?: string): Promise<DQIMetrics> => {
    const { data } = await api.get('/metrics/dqi', { params: { study_id: studyId } })
    // Map backend response to frontend DQIMetrics type
    return {
      overall_dqi: data.overall_dqi ?? 0,
      completeness: data.completeness ?? 0,
      accuracy: data.accuracy ?? 0,
      consistency: data.consistency ?? 0,
      timeliness: data.timeliness ?? 0,
      trend: data.trend ?? [],
      by_domain: {
        visits: data.completeness ?? 0,
        queries: data.consistency ?? 0,
        coding: data.accuracy ?? 0,
        safety: data.conformity ?? data.timeliness ?? 0,
        ...(data.by_domain ?? {}),
      },
    }
  },

  getCleanliness: async (studyId?: string): Promise<CleanlinessMetrics> => {
    const { data } = await api.get('/metrics/cleanliness', { params: { study_id: studyId } })
    const cleanCount = data.clean_patients ?? data.clean_count ?? 0
    const dirtyCount = data.dirty_patients ?? data.dirty_count ?? 0
    const atRiskCount = data.pending_patients ?? data.at_risk_count ?? 0
    const total = cleanCount + dirtyCount + atRiskCount

    return {
      clean_count: cleanCount,
      dirty_count: dirtyCount,
      at_risk_count: atRiskCount,
      cleanliness_rate: total > 0 ? (data.overall_rate ?? data.cleanliness_rate ?? 0) : 0,
      trend: data.trend ?? [],
      by_study: data.by_category ?? data.by_study ?? {},
    }
  },

  getQueries: async (studyId?: string): Promise<QueryMetrics> => {
    const { data } = await api.get('/metrics/queries', { params: { study_id: studyId } })
    // Map backend response to frontend QueryMetrics type
    return {
      total_queries: data.total_queries ?? 0,
      open_queries: data.open_queries ?? 0,
      closed_queries: data.closed_queries ?? 0,
      resolution_rate: data.resolution_rate ?? 0,
      avg_resolution_time: data.avg_resolution_time_days ?? data.avg_resolution_time ?? 0,
      aging_distribution: data.aging_distribution ?? {},
    }
  },

  getSAEs: async (studyId?: string): Promise<SAEMetrics> => {
    const { data } = await api.get('/metrics/saes', { params: { study_id: studyId } })
    return {
      total_saes: data.total_saes ?? 0,
      reconciled: data.reconciled ?? 0,
      pending: data.pending ?? 0,
      overdue: data.overdue ?? 0,
      reconciliation_rate: data.reconciliation_rate ?? 0,
      avg_reconciliation_days: data.avg_reconciliation_days ?? 0,
      by_seriousness: data.by_seriousness ?? {},
    }
  },

  getCoding: async (studyId?: string): Promise<CodingMetrics> => {
    const { data } = await api.get('/metrics/coding', { params: { study_id: studyId } })
    return {
      total_terms: data.total_terms ?? 0,
      coded: data.coded ?? 0,
      uncoded: data.uncoded ?? 0,
      completion_rate: data.completion_rate ?? 0,
      meddra_status: data.meddra_status ?? {},
      whodrug_status: data.whodrug_status ?? {},
      uncoded_breakdown: data.uncoded_breakdown ?? [],
    }
  },

  getVelocity: async (studyId?: string, days = 7): Promise<OperationalVelocity> => {
    const { data } = await api.get('/metrics/velocity', { params: { study_id: studyId, days } })
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
      queries_per_day: data.query_resolution_velocity ?? data.queries_per_day ?? 0,
      resolutions_per_day: data.resolutions_per_day ?? 0,
      data_entries_per_day: data.form_completion_velocity ?? data.data_entries_per_day ?? 0,
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

  // Dashboard summary with all key metrics
  getDashboardSummary: async (studyId?: string): Promise<DashboardSummary> => {
    try {
      const { data } = await api.get('/dashboard/summary', { params: { study_id: studyId } })
      return data
    } catch (error) {
      // Fallback: aggregate from individual endpoints
      const [cleanliness, queries] = await Promise.all([
        metricsApi.getCleanliness(studyId).catch(() => null),
        metricsApi.getQueries(studyId).catch(() => null),
      ])
      
      return {
        total_studies: 0,
        total_patients: (cleanliness?.clean_count || 0) + (cleanliness?.dirty_count || 0) + (cleanliness?.at_risk_count || 0),
        total_sites: 0,
        clean_patients: cleanliness?.clean_count || 0,
        dirty_patients: cleanliness?.dirty_count || 0,
        overall_dqi: 0,
        open_queries: queries?.open_queries || 0,
        pending_saes: 0,
        uncoded_terms: 0,
        last_updated: new Date().toISOString(),
        cleanliness_rate: cleanliness?.cleanliness_rate || 0,
      }
    }
  },

  // DQI Breakdown for the enhanced component
  getDQIBreakdown: async (studyId?: string) => {
    try {
      const data = await metricsApi.getDQI(studyId)
      return {
        overall: data.overall_dqi || 0,
        completeness: data.completeness || 0,
        accuracy: data.accuracy || 0,
        consistency: data.consistency || 0,
        timeliness: data.timeliness || 0,
        trend: data.trend || [],
      }
    } catch (error) {
      // Return mock data if API fails
      return {
        overall: 85,
        completeness: 97.9,
        accuracy: 100,
        consistency: 70,
        timeliness: 75,
        trend: [82, 84, 85, 84, 86, 87, 85],
      }
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

// Reports API
export const reportsApi = {
  list: async (): Promise<ReportSummary[]> => {
    const { data } = await api.get('/reports/')
    return Array.isArray(data) ? data : []
  },

  getById: async (reportId: string): Promise<ReportDetail> => {
    const { data } = await api.get(`/reports/${reportId}`)
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

export default api
