/**
 * Dashboard Hooks
 * React Query hooks for dashboard data
 */

import { useQuery } from '@tanstack/react-query'
import { dashboardService } from '../services/api'

export const QUERY_KEYS = {
  summary: 'dashboard-summary',
  regional: 'dashboard-regional',
  alerts: 'dashboard-alerts',
  trends: 'dashboard-trends',
  overview: 'dashboard-overview',
}

export const useDashboardSummary = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.summary],
    queryFn: () => dashboardService.getSummary(),
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: true,
  })
}

export const useRegionalPerformance = (regionId?: string, countryId?: string) => {
  return useQuery({
    queryKey: [QUERY_KEYS.regional, regionId, countryId],
    queryFn: () => dashboardService.getRegionalPerformance(regionId, countryId),
    staleTime: 10 * 60 * 1000,
  })
}

export const useDashboardAlerts = (
  severity?: 'critical' | 'warning' | 'info',
  limit?: number
) => {
  return useQuery({
    queryKey: [QUERY_KEYS.alerts, severity, limit],
    queryFn: () => dashboardService.getAlerts(severity, limit),
    staleTime: 2 * 60 * 1000, // 2 minutes for alerts
    refetchOnWindowFocus: true,
  })
}

export const useDashboardTrends = (
  metric: string,
  startDate: string,
  endDate: string
) => {
  return useQuery({
    queryKey: [QUERY_KEYS.trends, metric, startDate, endDate],
    queryFn: () => dashboardService.getTrends(metric, startDate, endDate),
    enabled: !!metric && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })
}

export const useDashboardOverview = (studyId?: string) => {
  return useQuery({
    queryKey: [QUERY_KEYS.overview, studyId],
    queryFn: () => dashboardService.getOverview(studyId),
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: true,
  })
}
