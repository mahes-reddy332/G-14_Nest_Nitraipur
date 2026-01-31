/**
 * Forms Hooks
 * React Query hooks for forms and verification data
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { formsService } from '../services/api'
import type { SDVFilters } from '../services/api/types'

export const QUERY_KEYS = {
  sdvStatus: 'forms-sdv-status',
  formStatus: 'forms-form-status',
  overdueCRFs: 'forms-overdue-crfs',
  inactivated: 'forms-inactivated',
  sdvProgress: 'forms-sdv-progress',
  statusDistribution: 'forms-status-distribution',
  lifecycle: 'forms-lifecycle',
}

export const useSDVStatus = (filters: SDVFilters) => {
  return useQuery({
    queryKey: [QUERY_KEYS.sdvStatus, filters],
    queryFn: () => formsService.getSDVStatus(filters),
    staleTime: 5 * 60 * 1000,
  })
}

export const useFormStatus = (
  status?: ('frozen' | 'locked' | 'signed')[],
  siteId?: string[],
  page?: number,
  pageSize?: number
) => {
  return useQuery({
    queryKey: [QUERY_KEYS.formStatus, status, siteId, page, pageSize],
    queryFn: () => formsService.getFormStatus(status, siteId, page, pageSize),
    staleTime: 5 * 60 * 1000,
  })
}

export const useOverdueCRFs = (
  daysOverdue?: { min: number; max: number },
  priority?: string[],
  page?: number,
  pageSize?: number
) => {
  return useQuery({
    queryKey: [QUERY_KEYS.overdueCRFs, daysOverdue, priority, page, pageSize],
    queryFn: () => formsService.getOverdueCRFs(daysOverdue, priority, page, pageSize),
    staleTime: 5 * 60 * 1000,
  })
}

export const useInactivatedForms = (siteId?: string[], reason?: string[]) => {
  return useQuery({
    queryKey: [QUERY_KEYS.inactivated, siteId, reason],
    queryFn: () => formsService.getInactivatedForms(siteId, reason),
    staleTime: 10 * 60 * 1000,
  })
}

export const useSDVProgress = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.sdvProgress],
    queryFn: () => formsService.getSDVProgress(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useFormStatusDistribution = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.statusDistribution],
    queryFn: () => formsService.getFormStatusDistribution(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useFormLifecycleMetrics = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.lifecycle],
    queryFn: () => formsService.getFormLifecycleMetrics(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useUpdateFormStatus = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      formId,
      status,
      notes,
    }: {
      formId: string
      status: string
      notes?: string
    }) => formsService.updateFormStatus(formId, status, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.sdvStatus] })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.formStatus] })
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.statusDistribution] })
    },
  })
}
