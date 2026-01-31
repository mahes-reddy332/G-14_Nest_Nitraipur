// API Types for Clinical Dashboard - Comprehensive Frontend Specification

// ============================================================================
// WebSocket & Connection Types
// ============================================================================

export type ConnectionState =
  | 'idle'
  | 'waiting_for_backend'
  | 'connecting'
  | 'connected'
  | 'retrying'
  | 'disconnected'
  | 'failed'

export interface DateRangeFilter {
  start: string | null
  end: string | null
}

// ============================================================================
// Filter Types (Global Filtering Capability)
// ============================================================================

export interface GlobalFilters {
  regions: string[]
  countries: string[]
  sites: string[]
  subjectIds: string[]
  subjectStatus: ('Screened' | 'Enrolled' | 'Completed' | 'Withdrawn' | 'Screen Failed')[]
  dateRange: DateRangeFilter
  dataQualityStatus: 'all' | 'clean' | 'not_clean'
}

// ============================================================================
// EDC Metrics Types (CPID_EDC_Metrics)
// ============================================================================

export interface SubjectEDCMetrics {
  region: string
  country: string
  site_id: string
  site_name: string
  subject_id: string
  subject_status: string
  enrollment_date: string
  last_visit_date: string
  visits_planned: number
  visits_completed: number
  missing_visits_count: number
  missing_visits_percent: number
  missing_pages_count: number
  missing_pages_percent: number
  open_queries_total: number
  data_queries: number
  protocol_deviation_queries: number
  safety_queries: number
  non_conformant_data_count: number
  sdv_percentage: number
  frozen_forms_count: number
  locked_forms_count: number
  signed_forms_count: number
  overdue_crfs_count: number
  inactivated_folders_count: number
  is_clean_patient: boolean
  last_update_timestamp: string
}

// ============================================================================
// Visit Management Types
// ============================================================================

export interface MissingVisit {
  subject_id: string
  site_id: string
  site_name: string
  visit_name: string
  visit_number: number
  projected_visit_date: string
  days_overdue: number
  visit_type: 'Screening' | 'Baseline' | 'Follow-up' | 'End of Study'
  last_contact_date: string
  cra_assigned: string
  follow_up_status: 'Pending' | 'In Progress' | 'Contacted' | 'Resolved'
}

export interface VisitComplianceMetrics {
  average_days_overdue: number
  visit_compliance_rate: number
  sites_with_overdue_visits: { site_id: string; count: number }[]
  subjects_with_overdue_visits: { subject_id: string; count: number }[]
}

// ============================================================================
// Laboratory Data Types
// ============================================================================

export interface MissingLabData {
  id: string
  subject_id: string
  site_id: string
  site_name: string
  visit_name: string
  lab_test_name: string
  missing_element: 'Lab Name' | 'Reference Range' | 'Unit'
  collection_date: string
  received_date: string
  days_since_collection: number
  priority_level: 'Critical' | 'High' | 'Medium' | 'Low'
  assigned_to: string
  resolution_status: 'Open' | 'In Progress' | 'Resolved'
  comments: string
}

export interface LabReconciliationSummary {
  total_missing_lab_names: number
  total_missing_reference_ranges: number
  total_missing_units: number
  average_resolution_time: number
  by_lab_type: { type: string; count: number }[]
  by_site: { site_id: string; issues: number }[]
}

// ============================================================================
// Safety Monitoring Types (SAE Dashboard)
// ============================================================================

export interface SAEDataManagement {
  subject_id: string
  site_id: string
  site_name: string
  sae_description: string
  onset_date: string
  report_date: string
  discrepancy_type: string
  discrepancy_status: 'Open' | 'Under Review' | 'Resolved'
  days_open: number
  assigned_data_manager: string
  last_update_date: string
  priority: 'Expedited' | 'Standard'
  action_required: string
  comments: string[]
}

export interface SAESafetyView {
  subject_id: string
  site_id: string
  site_name: string
  sae_term: string
  onset_date: string
  severity: 'Mild' | 'Moderate' | 'Severe'
  causality_assessment: string
  expectedness: 'Expected' | 'Unexpected'
  review_status: 'Pending Initial Review' | 'Under Review' | 'Completed'
  medical_review_date: string
  safety_physician_assigned: string
  follow_up_required: boolean
  regulatory_reporting_status: string
  comments: string[]
}

export interface SAEAnalytics {
  total_saes: number
  severity_distribution: { severity: string; count: number }[]
  by_site: { site_id: string; count: number }[]
  avg_time_to_resolution: number
  reviewed_within_target: number
  outstanding_reviews: number
  trend_data: { date: string; count: number }[]
}

// ============================================================================
// Query & Data Quality Types
// ============================================================================

export interface QueryDetail {
  query_id: string
  subject_id: string
  site_id: string
  site_name: string
  visit_name: string
  form_name: string
  query_type: 'Data Query' | 'Protocol Deviation' | 'Safety Query' | 'Lab Query' | 'Other'
  query_field: string
  query_text: string
  opened_date: string
  days_open: number
  query_status: 'Open' | 'Answered' | 'Closed' | 'Cancelled'
  assigned_to: string
  response_due_date: string
  last_response_date: string
  priority_level: 'Critical' | 'High' | 'Medium' | 'Low'
}

export interface QueryAging {
  under_7_days: number
  days_7_to_14: number
  days_15_to_30: number
  over_30_days: number
}

export interface NonConformantData {
  total_count: number
  by_rule: { rule: string; count: number }[]
  by_site: { site_id: string; rate: number }[]
  trend: { date: string; count: number }[]
}

// ============================================================================
// Coding Types (MedDRA & WHO Drug)
// ============================================================================

export interface MedDRACoding {
  subject_id: string
  site_id: string
  site_name: string
  term_type: 'Adverse Event' | 'Medical History' | 'Indication'
  verbatim_term: string
  meddra_coded_term: string
  preferred_term: string
  high_level_term: string
  system_organ_class: string
  coding_status: 'Uncoded' | 'Pending Review' | 'Coded' | 'Approved'
  coder_assigned: string
  date_term_entered: string
  date_coded: string
  days_pending_coding: number
  comments: string
}

export interface WHODrugCoding {
  subject_id: string
  site_id: string
  site_name: string
  medication_type: 'Concomitant' | 'Prior' | 'Protocol'
  verbatim_drug_name: string
  who_drug_coded_term: string
  drug_code: string
  atc_classification: string
  coding_status: 'Uncoded' | 'Pending Review' | 'Coded' | 'Approved'
  coder_assigned: string
  date_medication_entered: string
  date_coded: string
  days_pending_coding: number
  comments: string
}

export interface CodingMetricsSummary {
  total_terms_requiring_coding: number
  uncoded_terms_count: number
  coded_pending_review: number
  fully_approved: number
  average_turnaround_time: number
  accuracy_rate: number
}

// ============================================================================
// Forms & Verification Types (SDV)
// ============================================================================

export interface SDVStatus {
  subject_id: string
  site_id: string
  site_name: string
  visit_name: string
  form_name: string
  total_fields: number
  fields_verified: number
  percent_verified: number
  sdv_status: 'Not Started' | 'In Progress' | 'Complete'
  cra_performing_sdv: string
  last_monitoring_visit_date: string
  next_planned_monitoring_visit: string
  days_since_last_sdv_activity: number
}

export interface FormStatus {
  subject_id: string
  site_id: string
  site_name: string
  visit_name: string
  form_name: string
  is_frozen: boolean
  is_locked: boolean
  is_signed: boolean
  frozen_date: string
  locked_date: string
  signed_date: string
  status_changed_by: string
  signature_applied_by: string
}

export interface OverdueCRF {
  subject_id: string
  site_id: string
  site_name: string
  form_name: string
  expected_completion_date: string
  days_overdue: number
  priority: 'Critical' | 'High' | 'Medium' | 'Low'
  assigned_cra: string
  follow_up_status: string
}

export interface InactivatedForm {
  subject_id: string
  site_id: string
  site_name: string
  form_name: string
  inactivation_date: string
  inactivated_by: string
  reason: 'Protocol Deviation' | 'Data Entry Error' | 'Visit Not Conducted' | 'Other'
  detailed_explanation: string
  approval_status: string
  approver_name: string
  approval_date: string
}

// ============================================================================
// CRA Activity Types
// ============================================================================

export interface CRAPerformance {
  cra_name: string
  total_sites_assigned: number
  total_monitoring_visits: number
  average_visits_per_month: number
  average_time_per_visit: number
  query_resolution_rate: number
  sdv_completion_rate: number
  avg_time_to_close_findings: number
  site_data_quality_score: number
}

export interface MonitoringVisit {
  cra_name: string
  site_id: string
  site_name: string
  visit_date: string
  visit_type: 'Initiation' | 'Routine' | 'Close-out' | 'For Cause'
  findings_count: number
  critical_findings_count: number
  follow_up_items_count: number
  report_status: 'Draft' | 'Submitted' | 'Approved'
  report_submission_date: string
  next_planned_visit_date: string
}

export interface CRAFollowUp {
  cra_name: string
  site_id: string
  site_name: string
  follow_up_description: string
  priority: 'Critical' | 'High' | 'Medium' | 'Low'
  date_identified: string
  due_date: string
  status: 'Open' | 'In Progress' | 'Resolved' | 'Overdue'
  days_since_identified: number
  last_update_date: string
  resolution_notes: string
}

// ============================================================================
// Reports & Analytics Types
// ============================================================================

export interface DataLockReadiness {
  readiness_score: number
  planned_cutoff_date: string
  forecasted_cutoff_date: string
  blocking_issues_count: number
  days_to_target: number
}

export interface Milestone {
  milestone_name: string
  planned_date: string
  forecasted_date: string
  status: 'On Track' | 'At Risk' | 'Delayed'
  completion_percentage: number
  blocking_issues: string[]
  owner: string
  last_updated: string
}

export interface DQIDetail {
  overall_score: number
  parameter_weights: {
    missing_visits: number
    missing_pages: number
    open_queries: number
    non_conformant_data: number
    unverified_forms: number
    uncoded_terms: number
    unresolved_sae: number
  }
  scores_by_region: { region: string; score: number }[]
  scores_by_country: { country: string; score: number }[]
  scores_by_site: { site_id: string; score: number }[]
  scores_by_subject: { subject_id: string; score: number }[]
  trend: { date: string; score: number }[]
}

export interface DerivedMetrics {
  missing_visits_percent: number
  missing_pages_percent: number
  clean_crfs_percent: number
  open_queries_percent: number
  non_conformant_percent: number
  verification_complete_percent: number
  clean_patients_percent: number
}

// ============================================================================
// Third-Party Data Reconciliation Types (Compiled_EDRR)
// ============================================================================

export interface ThirdPartyIssue {
  subject_id: string
  site_id: string
  site_name: string
  data_source: 'Lab' | 'ECG' | 'Imaging' | 'Other'
  issue_type: string
  issue_description: string
  priority_level: 'Critical' | 'High' | 'Medium' | 'Low'
  date_identified: string
  days_open: number
  assigned_to: string
  status: 'Open' | 'Under Review' | 'Pending Response' | 'Resolved'
  expected_resolution_date: string
  comments: string[]
}

// ============================================================================
// Collaboration & Communication Types
// ============================================================================

export interface Comment {
  comment_id: string
  author: string
  author_role: string
  content: string
  timestamp: string
  attachments: { name: string; url: string }[]
  is_resolved: boolean
  mentions: string[]
}

export interface Task {
  task_id: string
  title: string
  description: string
  status: 'To Do' | 'In Progress' | 'Completed'
  assignee: string
  due_date: string
  priority: 'Critical' | 'High' | 'Medium' | 'Low'
  created_by: string
  created_at: string
  dependencies: string[]
}

export interface UserNotification {
  notification_id: string
  type: 'alert' | 'task' | 'comment' | 'system'
  title: string
  message: string
  timestamp: string
  is_read: boolean
  link: string
}

// ============================================================================
// AI Features Types
// ============================================================================

export interface NLQueryResult {
  query: string
  response: string
  supporting_data: Record<string, unknown>
  charts: { type: string; data: unknown }[]
  drill_down_links: { label: string; url: string }[]
  follow_up_suggestions: string[]
}

export interface AIRecommendation {
  recommendation_id: string
  type: 'action' | 'insight' | 'prediction'
  title: string
  description: string
  priority: 'Critical' | 'High' | 'Medium' | 'Low'
  confidence_score: number
  affected_entities: { type: string; id: string; name: string }[]
  suggested_action: string
  estimated_resolution_time: string
  generated_at: string
}

export interface PredictiveAnalytics {
  forecasted_lock_date: string
  at_risk_sites: { site_id: string; risk_score: number; reasons: string[] }[]
  predicted_query_resolution_time: number
  trending_issues: { issue: string; trend: 'increasing' | 'decreasing' | 'stable' }[]
}

// ============================================================================
// User Management Types
// ============================================================================

export type UserRole =
  | 'Data Quality Manager'
  | 'Clinical Research Associate'
  | 'Site Coordinator'
  | 'Medical Monitor'
  | 'Statistician'
  | 'Study Manager'

export interface User {
  user_id: string
  username: string
  name: string
  email: string
  role: UserRole
  assigned_sites: string[]
  timezone: string
  language: string
  notifications_enabled: boolean
  last_login: string
}

// ============================================================================
// System Health Types
// ============================================================================

export interface SystemStatus {
  overall_status: 'operational' | 'degraded' | 'down'
  services: {
    name: string
    status: 'operational' | 'degraded' | 'down'
    last_check: string
  }[]
  last_data_refresh: string
  data_sources: {
    name: string
    connected: boolean
    last_sync: string
  }[]
}

// Dashboard Summary (Extended with bundled data from initial-load endpoint)
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
  last_updated?: string
  cleanliness_rate?: number
  active_alerts?: number
  // Bundled data from /api/dashboard/initial-load endpoint
  _query_metrics?: {
    total_queries: number
    open_queries: number
    closed_queries: number
    resolution_rate: number
    avg_resolution_time: number
    aging_distribution: Record<string, number>
    velocity_trend: number[]
  }
  _cleanliness?: {
    cleanliness_rate: number
    total_patients: number
    clean_patients: number
    dirty_patients: number
    at_risk_count: number
    trend: number[]
  }
  _alerts?: {
    active_alerts: number
    critical_count: number
    high_count: number
  }
  _response_time_ms?: number
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

export interface StudyReportSummary {
  study_id: string
  study_name: string
  status: string
  total_patients: number
  total_sites: number
  dqi_score: number
  cleanliness_rate: number
  open_queries: number
  pending_saes: number
  uncoded_terms: number
  last_updated: string
  data_status: {
    status: 'ok' | 'partial' | 'empty'
    reasons: Array<{ code: string; message: string }>
    loaded_files: number
    missing_files: number
  }
}

export interface StudyReportInsight {
  insight_id: string
  category: string
  severity: 'info' | 'warning' | 'critical'
  title: string
  what_happened: string
  why_it_matters: string
  evidence: Record<string, unknown>
  generated_at: string
}

export interface StudyReportDetail {
  study_id: string
  study_name: string
  generated_at: string
  filters: Record<string, unknown>
  overview: Record<string, unknown>
  kpis: Array<{ label: string; value: number; unit?: string | null; status: string }>
  data_quality: Record<string, unknown>
  insights: StudyReportInsight[]
  trends: Record<string, unknown>
  risks_and_alerts: Array<Record<string, unknown>>
  sites_summary: Array<Record<string, unknown>>
  source_files: Array<Record<string, unknown>>
  data_status: StudyReportSummary['data_status']
}
