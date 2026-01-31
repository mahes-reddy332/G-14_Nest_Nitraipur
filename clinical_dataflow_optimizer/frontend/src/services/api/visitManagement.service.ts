/**
 * Visit Management Service
 * Handles missing visits, compliance metrics, and calendar data
 */

import { apiClient } from './client'
import type {
  MissingVisit,
  VisitFilters,
  VisitComplianceMetrics,
  CalendarHeatmapData,
  PaginatedResponse,
} from './types'

export class VisitManagementService {
  private basePath = '/visits'

  async getMissingVisits(
    filters: VisitFilters
  ): Promise<PaginatedResponse<MissingVisit>> {
    const params = this.buildQueryParams(filters)
    return apiClient.get<PaginatedResponse<MissingVisit>>(
      `${this.basePath}/missing`,
      { params }
    )
  }

  async getComplianceMetrics(
    siteId?: string,
    dateFrom?: string,
    dateTo?: string
  ): Promise<VisitComplianceMetrics> {
    return apiClient.get<VisitComplianceMetrics>(
      `${this.basePath}/compliance-metrics`,
      { params: { siteId, dateFrom, dateTo } }
    )
  }

  async getCalendarHeatmap(
    year: number,
    month?: number
  ): Promise<CalendarHeatmapData[]> {
    return apiClient.get<CalendarHeatmapData[]>(
      `${this.basePath}/calendar-heatmap`,
      { params: { year, month } }
    )
  }

  async updateVisitStatus(
    visitId: string,
    status: string,
    notes?: string
  ): Promise<MissingVisit> {
    return apiClient.put<MissingVisit>(`${this.basePath}/${visitId}/status`, {
      status,
      notes,
    })
  }

  async getVisitProjections(
    siteId?: string,
    weeks?: number
  ): Promise<{
    week: string
    expectedVisits: number
    completedVisits: number
    projectedCompletion: number
  }[]> {
    return apiClient.get(`${this.basePath}/projections`, {
      params: { siteId, weeks },
    })
  }

  async getSiteMissingVisitsSummary(): Promise<{
    siteId: string
    siteName: string
    missingCount: number
    avgDaysOverdue: number
  }[]> {
    return apiClient.get(`${this.basePath}/site-summary`)
  }

  private buildQueryParams(filters: VisitFilters): Record<string, string | number | undefined> {
    const params: Record<string, string | number | undefined> = {}

    if (filters.siteId?.length) params.siteId = filters.siteId.join(',')
    if (filters.daysOverdue) {
      params.daysOverdueMin = filters.daysOverdue.min
      params.daysOverdueMax = filters.daysOverdue.max
    }
    if (filters.visitType?.length) params.visitType = filters.visitType.join(',')
    if (filters.page) params.page = filters.page
    if (filters.pageSize) params.pageSize = filters.pageSize
    if (filters.sortBy) params.sortBy = filters.sortBy
    if (filters.sortOrder) params.sortOrder = filters.sortOrder

    return params
  }
}

export const visitManagementService = new VisitManagementService()
