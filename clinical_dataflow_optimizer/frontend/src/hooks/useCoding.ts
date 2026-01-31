/**
 * Coding Hooks
 * React Query hooks for MedDRA and WHO Drug coding
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { meddraCodingService, whoDrugCodingService } from '../services/api'
import type { CodingFilters } from '../services/api/types'

export const MEDDRA_QUERY_KEYS = {
  terms: 'meddra-terms',
  metrics: 'meddra-metrics',
  socDistribution: 'meddra-soc-distribution',
  typeDistribution: 'meddra-type-distribution',
  trend: 'meddra-trend',
}

export const WHO_DRUG_QUERY_KEYS = {
  terms: 'whodrug-terms',
  metrics: 'whodrug-metrics',
  atcDistribution: 'whodrug-atc-distribution',
  typeDistribution: 'whodrug-type-distribution',
  trend: 'whodrug-trend',
}

// MedDRA Hooks
export const useMedDRATerms = (filters: CodingFilters) => {
  return useQuery({
    queryKey: [MEDDRA_QUERY_KEYS.terms, filters],
    queryFn: () => meddraCodingService.getTerms(filters),
    staleTime: 5 * 60 * 1000,
  })
}

export const useMedDRAMetrics = () => {
  return useQuery({
    queryKey: [MEDDRA_QUERY_KEYS.metrics],
    queryFn: () => meddraCodingService.getMetrics(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useMedDRASOCDistribution = () => {
  return useQuery({
    queryKey: [MEDDRA_QUERY_KEYS.socDistribution],
    queryFn: () => meddraCodingService.getSOCDistribution(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useMedDRATypeDistribution = () => {
  return useQuery({
    queryKey: [MEDDRA_QUERY_KEYS.typeDistribution],
    queryFn: () => meddraCodingService.getTermTypeDistribution(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useMedDRACodingTrend = (startDate: string, endDate: string) => {
  return useQuery({
    queryKey: [MEDDRA_QUERY_KEYS.trend, startDate, endDate],
    queryFn: () => meddraCodingService.getCodingTrend(startDate, endDate),
    enabled: !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })
}

export const useUpdateMedDRATerm = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      termId,
      updates,
    }: {
      termId: string
      updates: {
        codedTerm?: string
        preferredTerm?: string
        soc?: string
        status?: string
      }
    }) => meddraCodingService.updateTerm(termId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [MEDDRA_QUERY_KEYS.terms] })
      queryClient.invalidateQueries({ queryKey: [MEDDRA_QUERY_KEYS.metrics] })
    },
  })
}

export const useBulkCodeMedDRATerms = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (
      terms: { termId: string; codedTerm: string; preferredTerm: string; soc: string }[]
    ) => meddraCodingService.bulkCodeTerms(terms),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [MEDDRA_QUERY_KEYS.terms] })
      queryClient.invalidateQueries({ queryKey: [MEDDRA_QUERY_KEYS.metrics] })
    },
  })
}

// WHO Drug Hooks
export const useWHODrugTerms = (filters: CodingFilters) => {
  return useQuery({
    queryKey: [WHO_DRUG_QUERY_KEYS.terms, filters],
    queryFn: () => whoDrugCodingService.getTerms(filters),
    staleTime: 5 * 60 * 1000,
  })
}

export const useWHODrugMetrics = () => {
  return useQuery({
    queryKey: [WHO_DRUG_QUERY_KEYS.metrics],
    queryFn: () => whoDrugCodingService.getMetrics(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useWHODrugATCDistribution = () => {
  return useQuery({
    queryKey: [WHO_DRUG_QUERY_KEYS.atcDistribution],
    queryFn: () => whoDrugCodingService.getATCDistribution(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useWHODrugTypeDistribution = () => {
  return useQuery({
    queryKey: [WHO_DRUG_QUERY_KEYS.typeDistribution],
    queryFn: () => whoDrugCodingService.getMedicationTypeDistribution(),
    staleTime: 10 * 60 * 1000,
  })
}

export const useWHODrugCodingTrend = (startDate: string, endDate: string) => {
  return useQuery({
    queryKey: [WHO_DRUG_QUERY_KEYS.trend, startDate, endDate],
    queryFn: () => whoDrugCodingService.getCodingTrend(startDate, endDate),
    enabled: !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })
}

export const useUpdateWHODrugTerm = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      termId,
      updates,
    }: {
      termId: string
      updates: {
        codedTerm?: string
        drugCode?: string
        atcClass?: string
        status?: string
      }
    }) => whoDrugCodingService.updateTerm(termId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [WHO_DRUG_QUERY_KEYS.terms] })
      queryClient.invalidateQueries({ queryKey: [WHO_DRUG_QUERY_KEYS.metrics] })
    },
  })
}

export const useBulkCodeWHODrugTerms = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (
      terms: { termId: string; codedTerm: string; drugCode: string; atcClass: string }[]
    ) => whoDrugCodingService.bulkCodeTerms(terms),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [WHO_DRUG_QUERY_KEYS.terms] })
      queryClient.invalidateQueries({ queryKey: [WHO_DRUG_QUERY_KEYS.metrics] })
    },
  })
}
