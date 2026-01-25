// API Types for Clinical Dashboard

// Dashboard Summary
export interface DashboardSummary {
  total_studies: number
  total_patients: number
  total_sites: number
  clean_patients: number
  dirty_patients: number
  overall_dqi: number
  open_queries: number
  pending_saes: number
  uncoded_terms: number
  last_updated: string
  cleanliness_rate?: number
  active_alerts?: number
}

// Study Types
export interface Study {
  study_id: string
  name: string
  phase: string
  status: string
  therapeutic_area: string
  total_patients: number
  total_sites: number
  clean_patients: number
  dirty_patients: number
  dqi_score: number
  last_updated: string
}

export interface StudyMetrics {
  study_id: string
  dqi_score: number
  cleanliness_rate: number
  query_count: number
  query_resolution_rate: number
  query_velocity: number
  sae_count: number
  sae_reconciliation_rate: number
  coding_completion_rate: number
  visit_completion_rate: number
  form_completion_rate: number
  dqi_trend?: number[]
  cleanliness_trend?: number[]
}

// Source File Types
export interface SourceFile {
  file_type: string
  display_name: string
  status: 'loaded' | 'not_found'
  record_count: number
  loaded_at: string | null
}

// Patient Types
export interface Patient {
  patient_id: string
  study_id: string
  site_id: string
  enrollment_date: string
  status: string
  is_clean: boolean
  cleanliness_score: number
  visit_count: number
  query_count: number
  last_updated: string
}

export interface PatientDetail extends Patient {
  demographics: {
    age: number
    gender: string
    race: string
  }
  visits: Visit[]
  queries: Query[]
  adverse_events: AdverseEvent[]
  blocking_factors: BlockingFactor[]
}

export interface CleanPatientStatus {
  patient_id: string
  is_clean: boolean
  cleanliness_score: number
  blocking_factors: BlockingFactor[]
  last_checked: string
  lock_readiness: string
}

export interface BlockingFactor {
  factor_type: string
  description: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  domain: string
  resolution_action: string
}

// Site Types
export interface Site {
  site_id: string
  name: string
  country: string
  region: string
  status: string
  patient_count: number
  dqi_score: number
  query_resolution_time: number
  last_updated: string
}

export interface SitePerformance {
  site_id: string
  name: string
  dqi_score: number
  cleanliness_rate: number
  query_count: number
  query_resolution_rate: number
  avg_resolution_time: number
  enrollment_rate: number
  protocol_deviations: number
}

// Metrics Types
export interface KPITile {
  id: string
  title: string
  value: number | string
  unit?: string
  trend: 'up' | 'down' | 'stable'
  trend_value: number
  status: 'good' | 'warning' | 'critical'
  icon?: string
}

export interface DQIMetrics {
  overall_dqi: number
  completeness: number
  accuracy: number
  consistency: number
  timeliness: number
  trend: number[]
  by_domain: Record<string, number>
}

export interface CleanlinessMetrics {
  clean_count: number
  dirty_count: number
  at_risk_count: number
  cleanliness_rate: number
  trend: number[]
  by_study: Record<string, number>
}

export interface QueryMetrics {
  total_queries: number
  open_queries: number
  closed_queries: number
  resolution_rate: number
  avg_resolution_time: number
  aging_distribution: Record<string, number>
}

export interface SAEMetrics {
  total_saes: number
  reconciled: number
  pending: number
  overdue: number
  reconciliation_rate: number
  avg_reconciliation_days: number
  by_seriousness: Record<string, number>
}

export interface CodingMetrics {
  total_terms: number
  coded: number
  uncoded: number
  completion_rate: number
  meddra_status: Record<string, number>
  whodrug_status: Record<string, number>
  uncoded_breakdown: Record<string, unknown>[]
}

export interface OperationalVelocity {
  queries_per_day: number
  resolutions_per_day: number
  data_entries_per_day: number
  trend: {
    date: string
    queries: number
    resolutions: number
  }[]
}

// Alert Types
export interface Alert {
  alert_id: string
  category: 'data_quality' | 'safety' | 'operational' | 'compliance' | 'system'
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'new' | 'acknowledged' | 'in_progress' | 'resolved' | 'dismissed'
  title: string
  description: string
  source: string
  affected_entity: {
    type: string
    id: string
  }
  details: Record<string, unknown>
  created_at: string
  updated_at?: string
  acknowledged_by?: string
  acknowledged_at?: string
  resolved_at?: string
  resolved_by?: string
  resolution_notes?: string
  actions_taken: {
    action: string
    performed_by: string
    performed_at: string
  }[]
}

export interface AlertSummary {
  total_alerts: number
  active_alerts: number
  by_status: Record<string, number>
  by_severity: Record<string, number>
  by_category: Record<string, number>
  last_updated: string
}

// Agent Types
export interface AgentStatus {
  name: string
  status: 'active' | 'idle' | 'error'
  last_activity: string
  tasks_completed: number
  tasks_pending: number
  capabilities: string[]
}

export interface AgentInsight {
  insight_id: string
  agent: string
  title: string
  description: string
  category: string
  priority: 'critical' | 'high' | 'medium' | 'low'
  confidence: number
  affected_entities: {
    type: string
    count: number
    ids: string[]
  }
  recommended_action: string
  generated_at: string
  expires_at?: string
  metadata: Record<string, unknown>
}

export interface AgentRecommendation {
  recommendation_id: string
  title: string
  description: string
  category: string
  impact: 'high' | 'medium' | 'low'
  effort: 'high' | 'medium' | 'low'
  priority_score: number
  source_agent: string
  related_insights: string[]
  action_items: string[]
  estimated_completion_time: string
  generated_at: string
  status: 'pending' | 'in_progress' | 'completed' | 'dismissed'
}

// Supporting Types
export interface Visit {
  visit_id: string
  visit_name: string
  scheduled_date: string
  actual_date?: string
  status: string
}

export interface Query {
  query_id: string
  domain: string
  field: string
  query_text: string
  status: string
  opened_date: string
  closed_date?: string
  age_days: number
}

export interface AdverseEvent {
  ae_id: string
  term: string
  severity: string
  serious: boolean
  onset_date: string
  outcome: string
  coded_term?: string
  meddra_code?: string
}

// Heatmap Types
export interface HeatmapData {
  rows: string[]
  columns: string[]
  values: number[][]
  metric: string
}

// WebSocket Event Types
export interface WebSocketEvent {
  type: 'patient_status' | 'query_update' | 'metric_change' | 'new_alert'
  data: Record<string, unknown>
  timestamp: string
}

// API Response wrapper
export interface ApiResponse<T> {
  data: T
  success: boolean
  message?: string
  timestamp: string
}

// Pagination
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

// Reports
export interface ReportSummary {
  report_id: string
  name: string
  report_type: 'json' | 'html'
  size_bytes: number
  last_modified: string
}

export interface ReportDetail extends ReportSummary {
  content_type: string
  content: unknown
}
