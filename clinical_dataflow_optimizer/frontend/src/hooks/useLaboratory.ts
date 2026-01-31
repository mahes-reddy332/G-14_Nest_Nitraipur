/**
 * Laboratory Hooks
 * React Query hooks for laboratory data fetching
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { laboratoryService } from '../services/api'
import type { LabFilters } from '../services/api/types'

export const QUERY_KEYS = {
  missingData: 'lab-missing-data',
  reconciliation: 'lab-reconciliation',
  typeDistribution: 'lab-type-distribution',
  sitePerformance: 'lab-site-performance',
}

export const useMissingLabData = (filters: LabFilters) => {
  return useQuery({
    queryKey: [QUERY_KEYS.missingData, filters],
    queryFn: () => laboratoryService.getMissingData(filters),
    staleTime: 5 * 60 * 1000,
  })
}

export const useLabReconciliationMetrics = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.reconciliation],
    queryFn: () => laboratoryService.getReconciliationMetrics(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useLabTypeDistribution = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.typeDistribution],
    queryFn: () => laboratoryService.getLabTypeDistribution(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useSiteLabPerformance = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.sitePerformance],
    queryFn: () => laboratoryService.getSiteLabPerformance(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useUpdateLabData = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      labId,
      updates,
    }: {
      labId: string
      updates: { labName?: string; referenceRange?: string; unit?: string }
    }) => laboratoryService.updateLabData(labId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.missingData] })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.reconciliation] })
    },
  })
}

export const useBulkUpdateLabData = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (
      updates: { labId: string; labName?: string; referenceRange?: string; unit?: string }[]
    ) => laboratoryService.bulkUpdateLabData(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.missingData] })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.reconciliation] })
    },
  })
}
