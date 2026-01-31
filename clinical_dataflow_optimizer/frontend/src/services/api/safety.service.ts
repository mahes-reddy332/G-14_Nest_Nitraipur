/**
 * Safety Service
 * Handles SAE dashboard, metrics, and safety monitoring
 */

import { apiClient } from './client'
import type {
  SAE,
  SAEDetail,
  SAEFilters,
  SAEMetrics,
  Comment,
  PaginatedResponse,
} from './types'

export class SafetyService {
  private basePath = '/safety'

  async getSAEDashboard(
    filters: SAEFilters
  ): Promise<PaginatedResponse<SAE>> {
    const params = this.buildQueryParams(filters)
    return apiClient.get<PaginatedResponse<SAE>>(
      `${this.basePath}/sae-dashboard`,
      { params }
    )
  }

  async getSAEById(saeId: string): Promise<SAEDetail> {
    return apiClient.get<SAEDetail>(`${this.basePath}/sae/${saeId}`)
  }

  async updateSAE(
    saeId: string,
    updates: {
      status?: string
      reviewNotes?: string
      assignedTo?: string
      causalityAssessment?: string
    }
  ): Promise<SAE> {
    return apiClient.put<SAE>(`${this.basePath}/sae/${saeId}`, updates)
  }

  async getSAEMetrics(): Promise<SAEMetrics> {
    return apiClient.get<SAEMetrics>(`${this.basePath}/sae-metrics`)
  }

  async addSAEComment(
    saeId: string,
    comment: string,
    type: 'dm' | 'safety'
  ): Promise<Comment> {
    return apiClient.post<Comment>(`${this.basePath}/sae/${saeId}/comments`, {
      comment,
      type,
    })
  }

  async getSAETrend(
    startDate: string,
    endDate: string
  ): Promise<{
    date: string
    newSAEs: number
    resolvedSAEs: number
    openSAEs: number
  }[]> {
    return apiClient.get(`${this.basePath}/sae-trend`, {
      params: { startDate, endDate },
    })
  }

  async getSeverityDistribution(): Promise<{
    severity: string
    count: number
    percentage: number
  }[]> {
    return apiClient.get(`${this.basePath}/severity-distribution`)
  }

  async getExpeditedReportsDue(): Promise<{
    saeId: string
    subjectId: string
    siteId: string
    dueDate: string
    daysRemaining: number
    reportType: string
  }[]> {
    return apiClient.get(`${this.basePath}/expedited-reports-due`)
  }

  async assignSAE(
    saeId: string,
    assignedTo: string,
    role: 'data_manager' | 'safety_physician'
  ): Promise<SAE> {
    return apiClient.post(`${this.basePath}/sae/${saeId}/assign`, {
      assignedTo,
      role,
    })
  }

  private buildQueryParams(filters: SAEFilters): Record<string, string | number | undefined> {
    const params: Record<string, string | number | undefined> = {}

    params.view = filters.view
    if (filters.status?.length) params.status = filters.status.join(',')
    if (filters.severity?.length) params.severity = filters.severity.join(',')
    if (filters.siteId?.length) params.siteId = filters.siteId.join(',')
    if (filters.daysOpen) {
      params.daysOpenMin = filters.daysOpen.min
      params.daysOpenMax = filters.daysOpen.max
    }
    if (filters.page) params.page = filters.page
    if (filters.pageSize) params.pageSize = filters.pageSize
    if (filters.sortBy) params.sortBy = filters.sortBy
    if (filters.sortOrder) params.sortOrder = filters.sortOrder

    return params
  }
}

export const safetyService = new SafetyService()
