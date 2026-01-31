/**
 * Forms Service
 * Handles SDV status, form status, and overdue CRFs
 */

import { apiClient } from './client'
import type {
  SDVStatus,
  SDVFilters,
  FormStatusData,
  OverdueCRF,
  PaginatedResponse,
} from './types'

export class FormsService {
  private basePath = '/forms'

  async getSDVStatus(
    filters: SDVFilters
  ): Promise<PaginatedResponse<SDVStatus>> {
    const params = this.buildQueryParams(filters)
    return apiClient.get<PaginatedResponse<SDVStatus>>(
      `${this.basePath}/sdv-status`,
      { params }
    )
  }

  async getFormStatus(
    status?: ('frozen' | 'locked' | 'signed')[],
    siteId?: string[],
    page?: number,
    pageSize?: number
  ): Promise<PaginatedResponse<FormStatusData>> {
    return apiClient.get<PaginatedResponse<FormStatusData>>(
      `${this.basePath}/form-status`,
      {
        params: {
          status: status?.join(','),
          siteId: siteId?.join(','),
          page,
          pageSize,
        },
      }
    )
  }

  async getOverdueCRFs(
    daysOverdue?: { min: number; max: number },
    priority?: string[],
    page?: number,
    pageSize?: number
  ): Promise<PaginatedResponse<OverdueCRF>> {
    return apiClient.get<PaginatedResponse<OverdueCRF>>(
      `${this.basePath}/overdue-crfs`,
      {
        params: {
          daysOverdueMin: daysOverdue?.min,
          daysOverdueMax: daysOverdue?.max,
          priority: priority?.join(','),
          page,
          pageSize,
        },
      }
    )
  }

  async getInactivatedForms(
    siteId?: string[],
    reason?: string[]
  ): Promise<PaginatedResponse<{
    id: string
    subjectId: string
    siteId: string
    siteName: string
    formName: string
    reason: string
    inactivatedDate: string
    inactivatedBy: string
  }>> {
    return apiClient.get(`${this.basePath}/inactivated`, {
      params: {
        siteId: siteId?.join(','),
        reason: reason?.join(','),
      },
    })
  }

  async updateFormStatus(
    formId: string,
    status: string,
    notes?: string
  ): Promise<FormStatusData> {
    return apiClient.put<FormStatusData>(
      `${this.basePath}/${formId}/status`,
      { status, notes }
    )
  }

  async getSDVProgress(): Promise<{
    totalForms: number
    verifiedForms: number
    sdvPercentage: number
    bySite: { siteId: string; siteName: string; sdvPercentage: number }[]
  }> {
    return apiClient.get(`${this.basePath}/sdv-progress`)
  }

  async getFormStatusDistribution(): Promise<{
    status: string
    count: number
    percentage: number
  }[]> {
    return apiClient.get(`${this.basePath}/status-distribution`)
  }

  async getFormLifecycleMetrics(): Promise<{
    avgTimeToComplete: number
    avgTimeToFreeze: number
    avgTimeToLock: number
    avgTimeToSign: number
  }> {
    return apiClient.get(`${this.basePath}/lifecycle-metrics`)
  }

  private buildQueryParams(filters: SDVFilters): Record<string, string | number | undefined> {
    const params: Record<string, string | number | undefined> = {}

    if (filters.siteId?.length) params.siteId = filters.siteId.join(',')
    if (filters.subjectId) params.subjectId = filters.subjectId
    if (filters.verificationStatus?.length) {
      params.verificationStatus = filters.verificationStatus.join(',')
    }
    if (filters.page) params.page = filters.page
    if (filters.pageSize) params.pageSize = filters.pageSize
    if (filters.sortBy) params.sortBy = filters.sortBy
    if (filters.sortOrder) params.sortOrder = filters.sortOrder

    return params
  }
}

export const formsService = new FormsService()
