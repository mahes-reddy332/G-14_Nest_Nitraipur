/**
 * CRA Activity Service
 * Handles CRA performance metrics, monitoring visits, and follow-ups
 */

import { apiClient } from './client'
import type {
  CRAPerformanceMetrics,
  CRAFilters,
  MonitoringVisit,
  FollowUpItem,
  PaginatedResponse,
} from './types'

export class CRAActivityService {
  private basePath = '/cra'

  async getPerformanceMetrics(craId?: string): Promise<{
    totalActiveCRAs: number
    totalMonitoringVisits: number
    averageVisitsPerCRA: number
    sitesPerCRA: { craId: string; craName: string; sites: number }[]
    craMetrics: CRAPerformanceMetrics[]
  }> {
    return apiClient.get(`${this.basePath}/performance-metrics`, {
      params: { craId },
    })
  }

  async getCRADetail(craId: string): Promise<CRAPerformanceMetrics> {
    return apiClient.get(`${this.basePath}/${craId}`)
  }

  async getMonitoringVisits(
    filters: CRAFilters
  ): Promise<PaginatedResponse<MonitoringVisit>> {
    const params = this.buildQueryParams(filters)
    return apiClient.get<PaginatedResponse<MonitoringVisit>>(
      `${this.basePath}/monitoring-visits`,
      { params }
    )
  }

  async getFollowUpItems(
    craId?: string,
    status?: string[],
    priority?: string[],
    page?: number,
    pageSize?: number
  ): Promise<PaginatedResponse<FollowUpItem>> {
    return apiClient.get<PaginatedResponse<FollowUpItem>>(
      `${this.basePath}/follow-up-items`,
      {
        params: {
          craId,
          status: status?.join(','),
          priority: priority?.join(','),
          page,
          pageSize,
        },
      }
    )
  }

  async updateFollowUpItem(
    itemId: string,
    updates: {
      status?: string
      resolutionNotes?: string
    }
  ): Promise<FollowUpItem> {
    return apiClient.put<FollowUpItem>(
      `${this.basePath}/follow-up-items/${itemId}`,
      updates
    )
  }

  async getVisitSchedule(
    craId?: string,
    siteId?: string,
    startDate?: string,
    endDate?: string
  ): Promise<{
    visitId: string
    siteId: string
    siteName: string
    plannedDate: string
    visitType: string
    status: string
  }[]> {
    return apiClient.get(`${this.basePath}/visit-schedule`, {
      params: { craId, siteId, startDate, endDate },
    })
  }

  async getCRAComparisonMetrics(): Promise<{
    craId: string
    craName: string
    sdvRate: number
    resolutionRate: number
    visitCompletionRate: number
    avgVisitDuration: number
  }[]> {
    return apiClient.get(`${this.basePath}/comparison-metrics`)
  }

  async getVisitTypeDistribution(): Promise<{
    visitType: string
    count: number
    percentage: number
  }[]> {
    return apiClient.get(`${this.basePath}/visit-type-distribution`)
  }

  async submitVisitReport(
    visitId: string,
    report: {
      subjectsReviewed: number
      queriesGenerated: number
      issuesIdentified: number
      notes: string
      followUpItems: { description: string; priority: string; dueDate: string }[]
    }
  ): Promise<MonitoringVisit> {
    return apiClient.post(`${this.basePath}/visits/${visitId}/report`, report)
  }

  private buildQueryParams(filters: CRAFilters): Record<string, string | number | undefined> {
    const params: Record<string, string | number | undefined> = {}

    if (filters.craId) params.craId = filters.craId
    if (filters.siteId?.length) params.siteId = filters.siteId.join(',')
    if (filters.visitType?.length) params.visitType = filters.visitType.join(',')
    if (filters.dateFrom) params.dateFrom = filters.dateFrom
    if (filters.dateTo) params.dateTo = filters.dateTo
    if (filters.page) params.page = filters.page
    if (filters.pageSize) params.pageSize = filters.pageSize
    if (filters.sortBy) params.sortBy = filters.sortBy
    if (filters.sortOrder) params.sortOrder = filters.sortOrder

    return params
  }
}

export const craActivityService = new CRAActivityService()
