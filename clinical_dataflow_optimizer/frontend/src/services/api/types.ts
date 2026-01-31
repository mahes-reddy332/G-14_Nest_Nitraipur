/**
 * API Types
 * Comprehensive type definitions for API requests and responses
 */

// Generic paginated response
export interface PaginatedResponse<T> {
  data: T[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}

// Base filter interface
export interface BaseFilters {
  page?: number
  pageSize?: number
  sortBy?: string
  sortOrder?: 'asc' | 'desc'
}

// Dashboard Types
export interface DashboardSummary {
  totalSites: number
  totalSubjects: number
  cleanPatientPercentage: number
  dataQualityIndex: number
  openQueriesCount: number
  missingSAECount: number
  overdueCRFsCount: number
  unresolvedSAEsCount: number
}

export interface RegionalPerformanceData {
  regionId: string
  regionName: string
  totalSubjects: number
  cleanPercentage: number
  dqiScore: number
  openQueries: number
}

export interface TrendData {
  date: string
  value: number
  metric: string
}

// EDC Metrics Types
export interface EDCMetricsFilters extends BaseFilters {
  region?: string[]
  country?: string[]
  siteId?: string[]
  subjectId?: string
  status?: string[]
  dateFrom?: string
  dateTo?: string
  isClean?: boolean
}

export interface SubjectMetric {
  id: string
  region: string
  country: string
  siteId: string
  siteName: string
  subjectId: string
  subjectStatus: string
  enrollmentDate: string
  lastVisitDate: string
  totalVisitsPlanned: number
  totalVisitsCompleted: number
  missingVisitsCount: number
  missingVisitsPercentage: number
  missingPagesCount: number
  missingPagesPercentage: number
  openQueriesTotal: number
  openQueriesByType: Record<string, number>
  nonConformantDataCount: number
  sdvStatusPercentage: number
  frozenFormsCount: number
  lockedFormsCount: number
  signedFormsCount: number
  overdueCRFsCount: number
  inactivatedFoldersCount: number
  isClean: boolean
  lastUpdated: string
}

export interface SubjectDetailMetric extends SubjectMetric {
  visitHistory: VisitDetail[]
  queryHistory: QuerySummary[]
  formStatus: FormStatusSummary[]
}

export interface DerivedMetrics {
  entityId: string
  entityName: string
  entityType: 'site' | 'region' | 'country'
  totalSubjects: number
  cleanSubjects: number
  cleanPercentage: number
  avgDQI: number
  totalOpenQueries: number
  avgQueryAge: number
}

// Data Quality Types
export interface QueryFilters extends BaseFilters {
  type?: string[]
  status?: string[]
  siteId?: string[]
  ageInDays?: { min: number; max: number }
  priority?: string[]
}

export interface Query {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  visitName: string
  formName: string
  fieldName: string
  queryType: string
  queryText: string
  status: 'open' | 'answered' | 'closed' | 'cancelled'
  priority: 'critical' | 'high' | 'medium' | 'low'
  openedDate: string
  daysOpen: number
  assignedTo: string
  responseDueDate: string
  lastResponseDate?: string
}

export interface QueryDetail extends Query {
  comments: Comment[]
  history: QueryHistoryItem[]
}

export interface QueryHistoryItem {
  timestamp: string
  action: string
  user: string
  details: string
}

export interface QueryMetrics {
  totalOpen: number
  byType: Record<string, number>
  byAge: Record<string, number>
  averageResolutionTime: number
  resolutionRate: number
}

export interface NonConformantData {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  ruleType: string
  ruleDescription: string
  fieldName: string
  currentValue: string
  expectedValue: string
  severity: 'critical' | 'major' | 'minor'
  detectedDate: string
}

export interface DataQualityIndex {
  entityId: string
  entityType: 'overall' | 'region' | 'country' | 'site' | 'subject'
  entityName: string
  dqiScore: number
  parameterScores: {
    missingVisits: number
    missingPages: number
    openQueries: number
    nonConformantData: number
    unverifiedForms: number
    uncodedTerms: number
    unresolvedSAEs: number
  }
  lastCalculated: string
}

// Visit Management Types
export interface VisitFilters extends BaseFilters {
  siteId?: string[]
  daysOverdue?: { min: number; max: number }
  visitType?: string[]
}

export interface MissingVisit {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  visitName: string
  visitType: string
  projectedDate: string
  daysOverdue: number
  lastContactDate?: string
  craAssigned: string
  followUpStatus: 'pending' | 'in_progress' | 'contacted' | 'resolved'
}

export interface VisitDetail {
  visitId: string
  visitName: string
  visitType: string
  plannedDate: string
  actualDate?: string
  status: string
}

export interface VisitComplianceMetrics {
  averageDaysOverdue: number
  visitComplianceRate: number
  sitesWithMultipleOverdue: { siteId: string; siteName: string; count: number }[]
  subjectsWithMultipleOverdue: { subjectId: string; count: number }[]
}

export interface CalendarHeatmapData {
  date: string
  count: number
  type: string
}

// Laboratory Data Types
export interface LabFilters extends BaseFilters {
  missingElement?: 'lab_name' | 'reference_range' | 'unit'
  priority?: string[]
  siteId?: string[]
}

export interface MissingLabData {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  visitName: string
  labType: string
  collectionDate: string
  missingElement: 'lab_name' | 'reference_range' | 'unit'
  currentValue?: string
  priority: 'high' | 'medium' | 'low'
  daysOpen: number
}

export interface LabReconciliationMetrics {
  totalMissingLabNames: number
  totalMissingRanges: number
  totalMissingUnits: number
  averageResolutionTime: number
  byLabType: Record<string, number>
}

// Safety Types
export interface SAEFilters extends BaseFilters {
  view: 'data_management' | 'safety'
  status?: string[]
  severity?: string[]
  siteId?: string[]
  daysOpen?: { min: number; max: number }
}

export interface SAE {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  saeDescription: string
  onsetDate: string
  reportDate: string
  severity: 'mild' | 'moderate' | 'severe'
  discrepancyType?: string
  discrepancyStatus?: string
  reviewStatus: 'pending' | 'under_review' | 'completed'
  daysOpen: number
  assignedDataManager?: string
  assignedSafetyPhysician?: string
  causalityAssessment?: string
  expectedness?: string
  lastUpdateDate: string
}

export interface SAEDetail extends SAE {
  comments: Comment[]
  history: SAEHistoryItem[]
  relatedDocuments: Document[]
}

export interface SAEHistoryItem {
  timestamp: string
  action: string
  user: string
  details: string
}

export interface SAEMetrics {
  totalSAEs: number
  bySeverity: Record<string, number>
  averageTimeToResolution: number
  pendingReviews: number
  expeditedReportsDue: number
}

// Coding Types
export interface CodingFilters extends BaseFilters {
  status?: string[]
  termType?: string[]
  siteId?: string[]
  daysPending?: { min: number; max: number }
}

export interface MedDRATerm {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  termType: 'AE' | 'MH' | 'Indication'
  verbatimTerm: string
  codedTerm?: string
  preferredTerm?: string
  soc?: string
  status: 'uncoded' | 'pending_review' | 'coded' | 'approved'
  coderAssigned?: string
  dateEntered: string
  dateCoded?: string
  daysPending: number
}

export interface WHODrugTerm {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  medicationType: 'concomitant' | 'prior' | 'protocol'
  verbatimDrugName: string
  codedTerm?: string
  drugCode?: string
  atcClass?: string
  status: 'uncoded' | 'pending_review' | 'coded' | 'approved'
  coderAssigned?: string
  dateEntered: string
  dateCoded?: string
  daysPending: number
}

export interface CodingMetrics {
  totalTerms: number
  uncoded: number
  pendingReview: number
  coded: number
  approved: number
  avgTurnaroundDays: number
  byCategory: Record<string, number>
}

// Forms & Verification Types
export interface SDVFilters extends BaseFilters {
  siteId?: string[]
  subjectId?: string
  verificationStatus?: string[]
}

export interface SDVStatus {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  visitName: string
  formName: string
  totalFields: number
  fieldsVerified: number
  sdvPercentage: number
  verificationStatus: 'not_started' | 'in_progress' | 'complete' | 'requires_reverification'
  lastVerifiedDate?: string
  verifiedBy?: string
  criticalFieldsPending: number
}

export interface FormStatusData {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  visitName: string
  formName: string
  status: 'incomplete' | 'complete' | 'frozen' | 'locked' | 'signed'
  dateCreated: string
  dateCompleted?: string
  dateFrozen?: string
  dateLocked?: string
  dateSigned?: string
  hasQueries: boolean
}

export interface OverdueCRF {
  id: string
  subjectId: string
  siteId: string
  siteName: string
  visitName: string
  formName: string
  expectedDate: string
  daysOverdue: number
  priority: 'high' | 'medium' | 'low'
  assignedTo: string
  reminderSent: boolean
}

// CRA Activity Types
export interface CRAFilters extends BaseFilters {
  craId?: string
  siteId?: string[]
  visitType?: string[]
  dateFrom?: string
  dateTo?: string
}

export interface CRAPerformanceMetrics {
  craId: string
  craName: string
  region: string
  assignedSites: number
  totalVisitsCompleted: number
  avgMonitoringDuration: number
  avgQueriesPerVisit: number
  sdvCompletionRate: number
  issueResolutionRate: number
  followUpAdherence: number
  performanceScore: number
  performanceTrend: 'up' | 'stable' | 'down'
  lastVisitDate: string
}

export interface MonitoringVisit {
  visitId: string
  siteId: string
  siteName: string
  craId: string
  craName: string
  visitType: 'initiation' | 'routine' | 'interim' | 'close_out' | 'for_cause'
  plannedDate: string
  actualDate?: string
  status: 'scheduled' | 'in_progress' | 'completed' | 'report_pending' | 'overdue'
  durationHours?: number
  subjectsReviewed: number
  queriesGenerated: number
  issuesIdentified: number
  reportSubmitted: boolean
}

export interface FollowUpItem {
  id: string
  siteId: string
  siteName: string
  craId: string
  craName: string
  issueCategory: string
  description: string
  priority: 'high' | 'medium' | 'low'
  status: 'open' | 'in_progress' | 'pending_response' | 'resolved'
  dateIdentified: string
  dueDate: string
  daysOpen: number
  responseReceived: boolean
}

// Reports Types
export interface Milestone {
  id: string
  name: string
  targetDate: string
  status: 'not_started' | 'in_progress' | 'completed' | 'at_risk'
  completionPercentage: number
  blockers: string[]
}

export interface DerivedMetricsSummary {
  aggregateLevel: 'site' | 'region' | 'overall'
  metrics: {
    entityId: string
    entityName: string
    cleanPatientRate: number
    queryResolutionRate: number
    sdvCompletionRate: number
    dataLockReadiness: number
  }[]
}

// Collaboration Types
export interface Comment {
  id: string
  text: string
  author: string
  timestamp: string
  attachments?: string[]
}

export interface Notification {
  id: string
  type: string
  title: string
  message: string
  read: boolean
  createdAt: string
  link?: string
}

export interface Task {
  id: string
  title: string
  description: string
  assignedTo: string
  assignedBy: string
  dueDate: string
  priority: 'high' | 'medium' | 'low'
  status: 'pending' | 'in_progress' | 'completed'
  relatedEntity?: { type: string; id: string }
  createdAt: string
}

// AI Types
export interface NLQueryResponse {
  answer: string
  supportingData: unknown[]
  suggestedActions: string[]
  relatedQueries: string[]
}

export interface AIRecommendation {
  id: string
  entityType: string
  entityId: string
  recommendation: string
  impact: 'high' | 'medium' | 'low'
  confidence: number
  suggestedAction: string
}

// Metadata Types
export interface Region {
  id: string
  name: string
  code: string
}

export interface Country {
  id: string
  name: string
  code: string
  regionId: string
}

export interface Site {
  id: string
  name: string
  countryId: string
  regionId: string
  status: 'active' | 'inactive' | 'pending'
  primaryInvestigator: string
}

export interface Document {
  id: string
  name: string
  type: string
  url: string
  uploadedAt: string
  uploadedBy: string
}

// Auth Types
export interface LoginCredentials {
  email: string
  password: string
}

export interface AuthResponse {
  accessToken: string
  refreshToken: string
  user: {
    id: string
    email: string
    name: string
    role: string
    permissions: string[]
  }
}

// Export Summary
export interface QuerySummary {
  type: string
  count: number
  avgAge: number
}

export interface FormStatusSummary {
  formName: string
  status: string
  completionPercentage: number
}
