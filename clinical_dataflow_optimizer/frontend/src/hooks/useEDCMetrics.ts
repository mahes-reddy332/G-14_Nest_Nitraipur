/**
 * EDC Metrics Hooks
 * React Query hooks for EDC metrics data fetching
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { edcMetricsService } from '../services/api'
import type { EDCMetricsFilters } from '../services/api/types'

export const QUERY_KEYS = {
  subjects: 'edc-metrics-subjects',
  subject: 'edc-metrics-subject',
  derived: 'edc-metrics-derived',
  cleanPatient: 'edc-metrics-clean-patient',
  site: 'edc-metrics-site',
}

export const useEDCMetrics = (filters: EDCMetricsFilters) => {
  return useQuery({
    queryKey: [QUERY_KEYS.subjects, filters],
    queryFn: () => edcMetricsService.getSubjects(filters),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: true,
  })
}

export const useSubjectDetail = (subjectId: string | undefined) => {
  return useQuery({
    queryKey: [QUERY_KEYS.subject, subjectId],
    queryFn: () => edcMetricsService.getSubjectById(subjectId!),
    enabled: !!subjectId,
  })
}

export const useDerivedMetrics = (aggregateBy: 'site' | 'region' | 'country') => {
  return useQuery({
    queryKey: [QUERY_KEYS.derived, aggregateBy],
    queryFn: () => edcMetricsService.getDerivedMetrics(aggregateBy),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

export const useCleanPatientMetrics = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.cleanPatient],
    queryFn: () => edcMetricsService.getCleanPatientMetrics(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useSiteMetrics = (siteId: string | undefined) => {
  return useQuery({
    queryKey: [QUERY_KEYS.site, siteId],
    queryFn: () => edcMetricsService.getSiteMetrics(siteId!),
    enabled: !!siteId,
  })
}

export const useExportEDCMetrics = () => {
  return useMutation({
    mutationFn: ({
      filters,
      format,
    }: {
      filters: EDCMetricsFilters
      format: 'excel' | 'csv' | 'pdf'
    }) => edcMetricsService.exportData(filters, format),
    onSuccess: (data) => {
      // Trigger download
      window.open(data.downloadUrl, '_blank')
    },
  })
}
