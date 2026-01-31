import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import EnhancedKPISections from '../EnhancedKPISections'
import { metricsApi } from '../../../api'

vi.mock('../../../api', async () => {
  const actual = await vi.importActual<typeof import('../../../api')>('../../../api')
  return {
    ...actual,
    metricsApi: {
      ...actual.metricsApi,
      getDashboardSummary: vi.fn(),
    },
  }
})

const renderWithClient = (ui: React.ReactNode) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

describe('EnhancedKPISections', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders loading skeletons while data is pending', async () => {
    ;(metricsApi.getDashboardSummary as any).mockImplementation(
      () => new Promise(() => undefined)
    )

    const { container } = renderWithClient(<EnhancedKPISections />)
    expect(container.querySelectorAll('.ant-skeleton').length).toBeGreaterThan(0)
  })

  it('renders KPI values from API without placeholder trends', async () => {
    ;(metricsApi.getDashboardSummary as any).mockResolvedValue({
      total_studies: 1,
      total_patients: 123,
      total_sites: 4,
      clean_patients: 100,
      dirty_patients: 23,
      overall_dqi: 88.4,
      open_queries: 55,
      pending_saes: 2,
      uncoded_terms: 9,
      _query_metrics: {
        total_queries: 90,
        open_queries: 55,
        closed_queries: 35,
        resolution_rate: 38.9,
        avg_resolution_time: 4.2,
        aging_distribution: { '0-7 days': 10, '8-14 days': 5, '15-30 days': 3, '30+ days': 1 },
        velocity_trend: [],
      },
      _cleanliness: {
        cleanliness_rate: 81.3,
        total_patients: 123,
        clean_patients: 100,
        dirty_patients: 15,
        at_risk_count: 8,
        trend: [],
      },
      _alerts: { active_alerts: 0, critical_count: 0, high_count: 0 },
    })

    renderWithClient(<EnhancedKPISections />)

    await screen.findByText('Total Patients')
    await waitFor(() => {
      expect(screen.getByText('123')).toBeInTheDocument()
      expect(screen.getByText('88.4')).toBeInTheDocument()
    })
  })
})
