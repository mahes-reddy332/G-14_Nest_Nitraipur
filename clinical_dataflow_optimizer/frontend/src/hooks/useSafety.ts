/**
 * Safety Hooks
 * React Query hooks for safety monitoring data fetching
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { safetyService } from '../services/api'
import type { SAEFilters } from '../services/api/types'

export const QUERY_KEYS = {
  dashboard: 'safety-dashboard',
  sae: 'safety-sae',
  metrics: 'safety-metrics',
  trend: 'safety-trend',
  severity: 'safety-severity',
  expedited: 'safety-expedited',
}

export const useSAEDashboard = (filters: SAEFilters) => {
  return useQuery({
    queryKey: [QUERY_KEYS.dashboard, filters],
    queryFn: () => safetyService.getSAEDashboard(filters),
    staleTime: 5 * 60 * 1000,
  })
}

export const useSAEDetail = (saeId: string | undefined) => {
  return useQuery({
    queryKey: [QUERY_KEYS.sae, saeId],
    queryFn: () => safetyService.getSAEById(saeId!),
    enabled: !!saeId,
  })
}

export const useSAEMetrics = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.metrics],
    queryFn: () => safetyService.getSAEMetrics(),
    staleTime: 5 * 60 * 1000,
  })
}

export const useSAETrend = (startDate: string, endDate: string) => {
  return useQuery({
    queryKey: [QUERY_KEYS.trend, startDate, endDate],
    queryFn: () => safetyService.getSAETrend(startDate, endDate),
    enabled: !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })
}

export const useSeverityDistribution = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.severity],
    queryFn: () => safetyService.getSeverityDistribution(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useExpeditedReportsDue = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.expedited],
    queryFn: () => safetyService.getExpeditedReportsDue(),
    staleTime: 5 * 60 * 1000,
  })
}

export const useUpdateSAE = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      saeId,
      updates,
    }: {
      saeId: string
      updates: {
        status?: string
        reviewNotes?: string
        assignedTo?: string
        causalityAssessment?: string
      }
    }) => safetyService.updateSAE(saeId, updates),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.dashboard] })
      queryClient.invalidateQueries({
        queryKey: [QUERY_KEYS.sae, variables.saeId],
      })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.metrics] })
    },
  })
}

export const useAddSAEComment = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      saeId,
      comment,
      type,
    }: {
      saeId: string
      comment: string
      type: 'dm' | 'safety'
    }) => safetyService.addSAEComment(saeId, comment, type),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: [QUERY_KEYS.sae, variables.saeId],
      })
    },
  })
}

export const useAssignSAE = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      saeId,
      assignedTo,
      role,
    }: {
      saeId: string
      assignedTo: string
      role: 'data_manager' | 'safety_physician'
    }) => safetyService.assignSAE(saeId, assignedTo, role),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.dashboard] })
      queryClient.invalidateQueries({
        queryKey: [QUERY_KEYS.sae, variables.saeId],
      })
    },
  })
}
