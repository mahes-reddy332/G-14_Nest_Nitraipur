/**
 * CRA Activity Hooks
 * React Query hooks for CRA activity data
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { craActivityService } from '../services/api'
import type { CRAFilters } from '../services/api/types'

export const QUERY_KEYS = {
  performance: 'cra-performance',
  detail: 'cra-detail',
  visits: 'cra-visits',
  followUps: 'cra-follow-ups',
  schedule: 'cra-schedule',
  comparison: 'cra-comparison',
  visitTypeDistribution: 'cra-visit-type-distribution',
}

export const useCRAPerformanceMetrics = (craId?: string) => {
  return useQuery({
    queryKey: [QUERY_KEYS.performance, craId],
    queryFn: () => craActivityService.getPerformanceMetrics(craId),
    staleTime: 10 * 60 * 1000,
  })
}

export const useCRADetail = (craId: string | undefined) => {
  return useQuery({
    queryKey: [QUERY_KEYS.detail, craId],
    queryFn: () => craActivityService.getCRADetail(craId!),
    enabled: !!craId,
  })
}

export const useMonitoringVisits = (filters: CRAFilters) => {
  return useQuery({
    queryKey: [QUERY_KEYS.visits, filters],
    queryFn: () => craActivityService.getMonitoringVisits(filters),
    staleTime: 5 * 60 * 1000,
  })
}

export const useFollowUpItems = (
  craId?: string,
  status?: string[],
  priority?: string[],
  page?: number,
  pageSize?: number
) => {
  return useQuery({
    queryKey: [QUERY_KEYS.followUps, craId, status, priority, page, pageSize],
    queryFn: () =>
      craActivityService.getFollowUpItems(craId, status, priority, page, pageSize),
    staleTime: 5 * 60 * 1000,
  })
}

export const useVisitSchedule = (
  craId?: string,
  siteId?: string,
  startDate?: string,
  endDate?: string
) => {
  return useQuery({
    queryKey: [QUERY_KEYS.schedule, craId, siteId, startDate, endDate],
    queryFn: () =>
      craActivityService.getVisitSchedule(craId, siteId, startDate, endDate),
    staleTime: 10 * 60 * 1000,
  })
}

export const useCRAComparisonMetrics = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.comparison],
    queryFn: () => craActivityService.getCRAComparisonMetrics(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useVisitTypeDistribution = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.visitTypeDistribution],
    queryFn: () => craActivityService.getVisitTypeDistribution(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useUpdateFollowUpItem = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      itemId,
      updates,
    }: {
      itemId: string
      updates: { status?: string; resolutionNotes?: string }
    }) => craActivityService.updateFollowUpItem(itemId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.followUps] })
    },
  })
}

export const useSubmitVisitReport = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      visitId,
      report,
    }: {
      visitId: string
      report: {
        subjectsReviewed: number
        queriesGenerated: number
        issuesIdentified: number
        notes: string
        followUpItems: { description: string; priority: string; dueDate: string }[]
      }
    }) => craActivityService.submitVisitReport(visitId, report),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.visits] })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.followUps] })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.performance] })
    },
  })
}
