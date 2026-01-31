/**
 * Laboratory Service
 * Handles missing lab data and reconciliation metrics
 */

import { apiClient } from './client'
import type {
  MissingLabData,
  LabFilters,
  LabReconciliationMetrics,
  PaginatedResponse,
} from './types'

export class LaboratoryService {
  private basePath = '/laboratory'

  async getMissingData(
    filters: LabFilters
  ): Promise<PaginatedResponse<MissingLabData>> {
    const params = this.buildQueryParams(filters)
    return apiClient.get<PaginatedResponse<MissingLabData>>(
      `${this.basePath}/missing-data`,
      { params }
    )
  }

  async getReconciliationMetrics(): Promise<LabReconciliationMetrics> {
    return apiClient.get<LabReconciliationMetrics>(
      `${this.basePath}/reconciliation-metrics`
    )
  }

  async updateLabData(
    labId: string,
    updates: {
      labName?: string
      referenceRange?: string
      unit?: string
    }
  ): Promise<MissingLabData> {
    return apiClient.put<MissingLabData>(
      `${this.basePath}/${labId}`,
      updates
    )
  }

  async getLabTypeDistribution(): Promise<{
    labType: string
    total: number
    missingName: number
    missingRange: number
    missingUnit: number
  }[]> {
    return apiClient.get(`${this.basePath}/type-distribution`)
  }

  async getSiteLabPerformance(): Promise<{
    siteId: string
    siteName: string
    totalLabs: number
    completionRate: number
    avgResolutionDays: number
  }[]> {
    return apiClient.get(`${this.basePath}/site-performance`)
  }

  async bulkUpdateLabData(
    updates: { labId: string; labName?: string; referenceRange?: string; unit?: string }[]
  ): Promise<{ updated: number; failed: number; errors: string[] }> {
    return apiClient.post(`${this.basePath}/bulk-update`, { updates })
  }

  private buildQueryParams(filters: LabFilters): Record<string, string | number | undefined> {
    const params: Record<string, string | number | undefined> = {}

    if (filters.missingElement) params.missingElement = filters.missingElement
    if (filters.priority?.length) params.priority = filters.priority.join(',')
    if (filters.siteId?.length) params.siteId = filters.siteId.join(',')
    if (filters.page) params.page = filters.page
    if (filters.pageSize) params.pageSize = filters.pageSize
    if (filters.sortBy) params.sortBy = filters.sortBy
    if (filters.sortOrder) params.sortOrder = filters.sortOrder

    return params
  }
}

export const laboratoryService = new LaboratoryService()
