/**
 * Data Quality Hooks
 * React Query hooks for data quality data fetching
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { dataQualityService } from '../services/api'
import type { QueryFilters, Query } from '../services/api/types'

export const QUERY_KEYS = {
  queries: 'data-quality-queries',
  query: 'data-quality-query',
  metrics: 'data-quality-metrics',
  dqi: 'data-quality-dqi',
  nonConformant: 'data-quality-non-conformant',
  aging: 'data-quality-aging',
  trend: 'data-quality-trend',
}

export const useQueries = (filters: QueryFilters) => {
  return useQuery({
    queryKey: [QUERY_KEYS.queries, filters],
    queryFn: () => dataQualityService.getQueries(filters),
    staleTime: 5 * 60 * 1000,
  })
}

export const useQueryDetail = (queryId: string | undefined) => {
  return useQuery({
    queryKey: [QUERY_KEYS.query, queryId],
    queryFn: () => dataQualityService.getQueryById(queryId!),
    enabled: !!queryId,
  })
}

export const useUpdateQuery = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      queryId,
      updates,
    }: {
      queryId: string
      updates: Partial<Query>
    }) => dataQualityService.updateQuery(queryId, updates),
    onSuccess: (_data, variables) => {
      // Invalidate and refetch
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.queries] })
      queryClient.invalidateQueries({
        queryKey: [QUERY_KEYS.query, variables.queryId],
      })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.metrics] })
    },
  })
}

export const useAddQueryComment = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      queryId,
      comment,
      attachments,
    }: {
      queryId: string
      comment: string
      attachments?: string[]
    }) => dataQualityService.addComment(queryId, comment, attachments),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: [QUERY_KEYS.query, variables.queryId],
      })
    },
  })
}

export const useQueryMetrics = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.metrics],
    queryFn: () => dataQualityService.getQueryMetrics(),
    staleTime: 5 * 60 * 1000,
  })
}

export const useDataQualityIndex = (
  aggregateBy: 'overall' | 'region' | 'country' | 'site' | 'subject'
) => {
  return useQuery({
    queryKey: [QUERY_KEYS.dqi, aggregateBy],
    queryFn: () => dataQualityService.getDataQualityIndex(aggregateBy),
    staleTime: 10 * 60 * 1000,
  })
}

export const useNonConformantData = (siteId?: string, ruleType?: string) => {
  return useQuery({
    queryKey: [QUERY_KEYS.nonConformant, siteId, ruleType],
    queryFn: () => dataQualityService.getNonConformantData(siteId, ruleType),
    staleTime: 5 * 60 * 1000,
  })
}

export const useQueryAgingDistribution = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.aging],
    queryFn: () => dataQualityService.getQueryAgingDistribution(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useQueryResolutionTrend = (startDate: string, endDate: string) => {
  return useQuery({
    queryKey: [QUERY_KEYS.trend, startDate, endDate],
    queryFn: () => dataQualityService.getQueryResolutionTrend(startDate, endDate),
    enabled: !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })
}
