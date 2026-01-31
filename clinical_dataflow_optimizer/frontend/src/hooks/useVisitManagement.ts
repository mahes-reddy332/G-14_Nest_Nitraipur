/**
 * Visit Management Hooks
 * React Query hooks for visit management data fetching
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { visitManagementService } from '../services/api'
import type { VisitFilters } from '../services/api/types'

export const QUERY_KEYS = {
  missingVisits: 'visits-missing',
  compliance: 'visits-compliance',
  calendar: 'visits-calendar',
  projections: 'visits-projections',
  siteSummary: 'visits-site-summary',
}

export const useMissingVisits = (filters: VisitFilters) => {
  return useQuery({
    queryKey: [QUERY_KEYS.missingVisits, filters],
    queryFn: () => visitManagementService.getMissingVisits(filters),
    staleTime: 5 * 60 * 1000,
  })
}

export const useVisitComplianceMetrics = (
  siteId?: string,
  dateFrom?: string,
  dateTo?: string
) => {
  return useQuery({
    queryKey: [QUERY_KEYS.compliance, siteId, dateFrom, dateTo],
    queryFn: () =>
      visitManagementService.getComplianceMetrics(siteId, dateFrom, dateTo),
    staleTime: 10 * 60 * 1000,
  })
}

export const useCalendarHeatmap = (year: number, month?: number) => {
  return useQuery({
    queryKey: [QUERY_KEYS.calendar, year, month],
    queryFn: () => visitManagementService.getCalendarHeatmap(year, month),
    staleTime: 10 * 60 * 1000,
  })
}

export const useVisitProjections = (siteId?: string, weeks?: number) => {
  return useQuery({
    queryKey: [QUERY_KEYS.projections, siteId, weeks],
    queryFn: () => visitManagementService.getVisitProjections(siteId, weeks),
    staleTime: 10 * 60 * 1000,
  })
}

export const useSiteMissingVisitsSummary = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.siteSummary],
    queryFn: () => visitManagementService.getSiteMissingVisitsSummary(),
    staleTime: 5 * 60 * 1000,
  })
}

export const useUpdateVisitStatus = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      visitId,
      status,
      notes,
    }: {
      visitId: string
      status: string
      notes?: string
    }) => visitManagementService.updateVisitStatus(visitId, status, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.missingVisits] })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.compliance] })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.siteSummary] })
    },
  })
}
