/**
 * EDC Metrics Service
 * Handles patient/subject level EDC metrics and derived metrics
 */

import { apiClient } from './client'
import type {
  SubjectMetric,
  SubjectDetailMetric,
  DerivedMetrics,
  EDCMetricsFilters,
  PaginatedResponse,
} from './types'

export class EDCMetricsService {
  private basePath = '/edc-metrics'

  async getSubjects(
    filters: EDCMetricsFilters
  ): Promise<PaginatedResponse<SubjectMetric>> {
    const params = this.buildQueryParams(filters)
    return apiClient.get<PaginatedResponse<SubjectMetric>>(
      `${this.basePath}/subjects`,
      { params }
    )
  }

  async getSubjectById(subjectId: string): Promise<SubjectDetailMetric> {
    return apiClient.get<SubjectDetailMetric>(
      `${this.basePath}/subjects/${subjectId}`
    )
  }

  async getDerivedMetrics(
    aggregateBy: 'site' | 'region' | 'country'
  ): Promise<DerivedMetrics[]> {
    return apiClient.get<DerivedMetrics[]>(`${this.basePath}/derived-metrics`, {
      params: { aggregateBy },
    })
  }

  async exportData(
    filters: EDCMetricsFilters,
    format: 'excel' | 'csv' | 'pdf'
  ): Promise<{ downloadUrl: string; expiresAt: string }> {
    return apiClient.post(`${this.basePath}/export`, { filters, format })
  }

  async getSiteMetrics(siteId: string): Promise<{
    site: { id: string; name: string }
    metrics: DerivedMetrics
    subjects: SubjectMetric[]
  }> {
    return apiClient.get(`${this.basePath}/sites/${siteId}`)
  }

  async getCleanPatientMetrics(): Promise<{
    totalPatients: number
    cleanPatients: number
    cleanPercentage: number
    byRegion: { region: string; cleanPercentage: number }[]
    bySite: { siteId: string; siteName: string; cleanPercentage: number }[]
  }> {
    return apiClient.get(`${this.basePath}/clean-patient-metrics`)
  }

  private buildQueryParams(filters: EDCMetricsFilters): Record<string, string | number | boolean | undefined> {
    const params: Record<string, string | number | boolean | undefined> = {}

    if (filters.region?.length) params.region = filters.region.join(',')
    if (filters.country?.length) params.country = filters.country.join(',')
    if (filters.siteId?.length) params.siteId = filters.siteId.join(',')
    if (filters.subjectId) params.subjectId = filters.subjectId
    if (filters.status?.length) params.status = filters.status.join(',')
    if (filters.dateFrom) params.dateFrom = filters.dateFrom
    if (filters.dateTo) params.dateTo = filters.dateTo
    if (filters.isClean !== undefined) params.isClean = filters.isClean
    if (filters.page) params.page = filters.page
    if (filters.pageSize) params.pageSize = filters.pageSize
    if (filters.sortBy) params.sortBy = filters.sortBy
    if (filters.sortOrder) params.sortOrder = filters.sortOrder

    return params
  }
}

export const edcMetricsService = new EDCMetricsService()
