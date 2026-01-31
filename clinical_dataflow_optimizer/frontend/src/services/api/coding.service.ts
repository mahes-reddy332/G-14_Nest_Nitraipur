/**
 * Coding Services
 * Handles MedDRA and WHO Drug coding
 */

import { apiClient } from './client'
import type {
  MedDRATerm,
  WHODrugTerm,
  CodingFilters,
  CodingMetrics,
  PaginatedResponse,
} from './types'

// MedDRA Coding Service
export class MedDRACodingService {
  private basePath = '/coding/meddra'

  async getTerms(
    filters: CodingFilters
  ): Promise<PaginatedResponse<MedDRATerm>> {
    const params = this.buildQueryParams(filters)
    return apiClient.get<PaginatedResponse<MedDRATerm>>(this.basePath, {
      params,
    })
  }

  async updateTerm(
    termId: string,
    updates: {
      codedTerm?: string
      preferredTerm?: string
      soc?: string
      status?: string
    }
  ): Promise<MedDRATerm> {
    return apiClient.put<MedDRATerm>(`${this.basePath}/${termId}`, updates)
  }

  async getMetrics(): Promise<CodingMetrics> {
    return apiClient.get<CodingMetrics>(`${this.basePath}/metrics`)
  }

  async getSOCDistribution(): Promise<{
    soc: string
    count: number
    percentage: number
  }[]> {
    return apiClient.get(`${this.basePath}/soc-distribution`)
  }

  async getTermTypeDistribution(): Promise<{
    termType: string
    total: number
    uncoded: number
    coded: number
  }[]> {
    return apiClient.get(`${this.basePath}/term-type-distribution`)
  }

  async getCodingTrend(
    startDate: string,
    endDate: string
  ): Promise<{
    date: string
    coded: number
    approved: number
  }[]> {
    return apiClient.get(`${this.basePath}/trend`, {
      params: { startDate, endDate },
    })
  }

  async bulkCodeTerms(
    terms: { termId: string; codedTerm: string; preferredTerm: string; soc: string }[]
  ): Promise<{ coded: number; failed: number; errors: string[] }> {
    return apiClient.post(`${this.basePath}/bulk-code`, { terms })
  }

  private buildQueryParams(filters: CodingFilters): Record<string, string | number | undefined> {
    const params: Record<string, string | number | undefined> = {}

    if (filters.status?.length) params.status = filters.status.join(',')
    if (filters.termType?.length) params.termType = filters.termType.join(',')
    if (filters.siteId?.length) params.siteId = filters.siteId.join(',')
    if (filters.daysPending) {
      params.daysPendingMin = filters.daysPending.min
      params.daysPendingMax = filters.daysPending.max
    }
    if (filters.page) params.page = filters.page
    if (filters.pageSize) params.pageSize = filters.pageSize
    if (filters.sortBy) params.sortBy = filters.sortBy
    if (filters.sortOrder) params.sortOrder = filters.sortOrder

    return params
  }
}

// WHO Drug Coding Service
export class WHODrugCodingService {
  private basePath = '/coding/whodrug'

  async getTerms(
    filters: CodingFilters
  ): Promise<PaginatedResponse<WHODrugTerm>> {
    const params = this.buildQueryParams(filters)
    return apiClient.get<PaginatedResponse<WHODrugTerm>>(this.basePath, {
      params,
    })
  }

  async updateTerm(
    termId: string,
    updates: {
      codedTerm?: string
      drugCode?: string
      atcClass?: string
      status?: string
    }
  ): Promise<WHODrugTerm> {
    return apiClient.put<WHODrugTerm>(`${this.basePath}/${termId}`, updates)
  }

  async getMetrics(): Promise<CodingMetrics> {
    return apiClient.get<CodingMetrics>(`${this.basePath}/metrics`)
  }

  async getATCDistribution(): Promise<{
    atcClass: string
    atcName: string
    count: number
    percentage: number
  }[]> {
    return apiClient.get(`${this.basePath}/atc-distribution`)
  }

  async getMedicationTypeDistribution(): Promise<{
    medicationType: string
    total: number
    uncoded: number
    coded: number
  }[]> {
    return apiClient.get(`${this.basePath}/medication-type-distribution`)
  }

  async getCodingTrend(
    startDate: string,
    endDate: string
  ): Promise<{
    date: string
    coded: number
    approved: number
  }[]> {
    return apiClient.get(`${this.basePath}/trend`, {
      params: { startDate, endDate },
    })
  }

  async bulkCodeTerms(
    terms: { termId: string; codedTerm: string; drugCode: string; atcClass: string }[]
  ): Promise<{ coded: number; failed: number; errors: string[] }> {
    return apiClient.post(`${this.basePath}/bulk-code`, { terms })
  }

  private buildQueryParams(filters: CodingFilters): Record<string, string | number | undefined> {
    const params: Record<string, string | number | undefined> = {}

    if (filters.status?.length) params.status = filters.status.join(',')
    if (filters.termType?.length) params.medicationType = filters.termType.join(',')
    if (filters.siteId?.length) params.siteId = filters.siteId.join(',')
    if (filters.daysPending) {
      params.daysPendingMin = filters.daysPending.min
      params.daysPendingMax = filters.daysPending.max
    }
    if (filters.page) params.page = filters.page
    if (filters.pageSize) params.pageSize = filters.pageSize
    if (filters.sortBy) params.sortBy = filters.sortBy
    if (filters.sortOrder) params.sortOrder = filters.sortOrder

    return params
  }
}

export const meddraCodingService = new MedDRACodingService()
export const whoDrugCodingService = new WHODrugCodingService()
