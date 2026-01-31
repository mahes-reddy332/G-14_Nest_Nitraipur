/**
 * Dashboard Service
 * Handles dashboard summary, regional performance, and alerts
 */

import { apiClient } from './client'
import type {
  DashboardSummary,
  RegionalPerformanceData,
  TrendData,
  Notification,
} from './types'

export class DashboardService {
  private basePath = '/dashboard'

  async getSummary(): Promise<DashboardSummary> {
    return apiClient.get<DashboardSummary>(`${this.basePath}/summary`)
  }

  async getRegionalPerformance(
    regionId?: string,
    countryId?: string
  ): Promise<RegionalPerformanceData[]> {
    return apiClient.get<RegionalPerformanceData[]>(
      `${this.basePath}/regional-performance`,
      { params: { regionId, countryId } }
    )
  }

  async getAlerts(
    severity?: 'critical' | 'warning' | 'info',
    limit?: number
  ): Promise<Notification[]> {
    return apiClient.get<Notification[]>(`${this.basePath}/alerts`, {
      params: { severity, limit },
    })
  }

  async getTrends(
    metric: string,
    startDate: string,
    endDate: string
  ): Promise<TrendData[]> {
    return apiClient.get<TrendData[]>(`${this.basePath}/trends`, {
      params: { metric, startDate, endDate },
    })
  }

  async getOverview(studyId?: string): Promise<{
    summary: DashboardSummary
    recentAlerts: Notification[]
    trends: TrendData[]
  }> {
    return apiClient.get(`${this.basePath}/overview`, {
      params: { studyId },
    })
  }
}

export const dashboardService = new DashboardService()
