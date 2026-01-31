/**
 * Central Hooks Export
 * Exports all custom React hooks
 */

// Auth hooks
export {
  useCurrentUser,
  useLogin,
  useLogout,
  useForgotPassword,
  useResetPassword,
  useRefreshToken,
} from './useAuth'

// Dashboard hooks
export {
  useDashboardSummary,
  useRegionalPerformance,
  useDashboardAlerts,
  useDashboardTrends,
  useDashboardOverview,
} from './useDashboard'

// EDC Metrics hooks
export {
  useEDCMetrics,
  useSubjectDetail,
  useDerivedMetrics,
  useCleanPatientMetrics,
  useSiteMetrics,
  useExportEDCMetrics,
} from './useEDCMetrics'

// Data Quality hooks
export {
  useQueries,
  useQueryDetail,
  useUpdateQuery,
  useAddQueryComment,
  useQueryMetrics,
  useDataQualityIndex,
  useNonConformantData,
  useQueryAgingDistribution,
  useQueryResolutionTrend,
} from './useDataQuality'

// Visit Management hooks
export {
  useMissingVisits,
  useVisitComplianceMetrics,
  useCalendarHeatmap,
  useVisitProjections,
  useSiteMissingVisitsSummary,
  useUpdateVisitStatus,
} from './useVisitManagement'

// Laboratory hooks
export {
  useMissingLabData,
  useLabReconciliationMetrics,
  useLabTypeDistribution,
  useSiteLabPerformance,
  useUpdateLabData,
  useBulkUpdateLabData,
} from './useLaboratory'

// Safety hooks
export {
  useSAEDashboard,
  useSAEDetail,
  useSAEMetrics,
  useSAETrend,
  useSeverityDistribution,
  useExpeditedReportsDue,
  useUpdateSAE,
  useAddSAEComment,
  useAssignSAE,
} from './useSafety'

// Coding hooks (MedDRA and WHO Drug)
export {
  useMedDRATerms,
  useMedDRAMetrics,
  useMedDRASOCDistribution,
  useMedDRATypeDistribution,
  useMedDRACodingTrend,
  useUpdateMedDRATerm,
  useBulkCodeMedDRATerms,
  useWHODrugTerms,
  useWHODrugMetrics,
  useWHODrugATCDistribution,
  useWHODrugTypeDistribution,
  useWHODrugCodingTrend,
  useUpdateWHODrugTerm,
  useBulkCodeWHODrugTerms,
} from './useCoding'

// Forms hooks
export {
  useSDVStatus,
  useFormStatus,
  useOverdueCRFs,
  useInactivatedForms,
  useSDVProgress,
  useFormStatusDistribution,
  useFormLifecycleMetrics,
  useUpdateFormStatus,
} from './useForms'

// CRA Activity hooks
export {
  useCRAPerformanceMetrics,
  useCRADetail,
  useMonitoringVisits,
  useFollowUpItems,
  useVisitSchedule,
  useCRAComparisonMetrics,
  useVisitTypeDistribution,
  useUpdateFollowUpItem,
  useSubmitVisitReport,
} from './useCRAActivity'

// Metadata hooks
export {
  useRegions,
  useCountries,
  useSites,
  useSubjects,
  useVisitTypes,
  useQueryTypes,
  useUserRoles,
  useStudies,
  useFormTypes,
  useLabTypes,
  useCRAs,
} from './useMetadata'

// WebSocket hooks
export {
  useWebSocket,
  useWebSocketSubscription,
  useQueryUpdates,
  useSAEReports,
  useVisitCompletions,
  useFormSignatures,
  useAlertNotifications,
  useCodingUpdates,
  useSubjectUpdates,
  useCRAVisitCompletions,
} from './useWebSocket'
