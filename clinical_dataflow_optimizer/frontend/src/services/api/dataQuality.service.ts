/**
 * Data Quality Service
 * Handles queries, non-conformant data, and DQI
 */

import { apiClient } from './client'
import type {
  Query,
  QueryDetail,
  QueryFilters,
  QueryMetrics,
  NonConformantData,
  DataQualityIndex,
  Comment,
  PaginatedResponse,
} from './types'

export class DataQualityService {
  private basePath = '/data-quality'

  async getQueries(filters: QueryFilters): Promise<PaginatedResponse<Query>> {
    const params = this.buildQueryParams(filters)
    return apiClient.get<PaginatedResponse<Query>>(`${this.basePath}/queries`, {
      params,
    })
  }

  async getQueryById(queryId: string): Promise<QueryDetail> {
    return apiClient.get<QueryDetail>(`${this.basePath}/queries/${queryId}`)
  }

  async updateQuery(queryId: string, updates: Partial<Query>): Promise<Query> {
    return apiClient.put<Query>(`${this.basePath}/queries/${queryId}`, updates)
  }

  async addComment(
    queryId: string,
    comment: string,
    attachments?: string[]
  ): Promise<Comment> {
    return apiClient.post<Comment>(
      `${this.basePath}/queries/${queryId}/comments`,
      { comment, attachments }
    )
  }

  async getQueryMetrics(): Promise<QueryMetrics> {
    return apiClient.get<QueryMetrics>(`${this.basePath}/query-metrics`)
  }

  async getNonConformantData(
    siteId?: string,
    ruleType?: string
  ): Promise<NonConformantData[]> {
    return apiClient.get<NonConformantData[]>(
      `${this.basePath}/non-conformant`,
      { params: { siteId, ruleType } }
    )
  }

  async getDataQualityIndex(
    aggregateBy: 'overall' | 'region' | 'country' | 'site' | 'subject'
  ): Promise<DataQualityIndex[]> {
    return apiClient.get<DataQualityIndex[]>(`${this.basePath}/dqi`, {
      params: { aggregateBy },
    })
  }

  async getQueryAgingDistribution(): Promise<{
    ranges: { range: string; count: number }[]
    averageAge: number
    oldestQuery: number
  }> {
    return apiClient.get(`${this.basePath}/query-aging`)
  }

  async getQueryResolutionTrend(
    startDate: string,
    endDate: string
  ): Promise<{ date: string; opened: number; closed: number }[]> {
    return apiClient.get(`${this.basePath}/query-resolution-trend`, {
      params: { startDate, endDate },
    })
  }

  private buildQueryParams(filters: QueryFilters): Record<string, string | number | undefined> {
    const params: Record<string, string | number | undefined> = {}

    if (filters.type?.length) params.type = filters.type.join(',')
    if (filters.status?.length) params.status = filters.status.join(',')
    if (filters.siteId?.length) params.siteId = filters.siteId.join(',')
    if (filters.ageInDays) {
      params.ageInDaysMin = filters.ageInDays.min
      params.ageInDaysMax = filters.ageInDays.max
    }
    if (filters.priority?.length) params.priority = filters.priority.join(',')
    if (filters.page) params.page = filters.page
    if (filters.pageSize) params.pageSize = filters.pageSize

    return params
  }
}

export const dataQualityService = new DataQualityService()
