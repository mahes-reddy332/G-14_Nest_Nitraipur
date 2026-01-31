import React from 'react'
import { Row, Col, Tooltip, Skeleton, Empty, Collapse } from 'antd'
import {
  ArrowUpOutlined,
  ArrowDownOutlined,
  MinusOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  UserOutlined,
  SafetyOutlined,
  FileSearchOutlined,
  BarChartOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { metricsApi } from '../../api'
import { useStore } from '../../store'
import '../../styles/clinical-design-system.css'
import { getDaysFromRange } from '../../utils/filtering'

// Types
interface KPIMetric {
  id: string
  title: string
  value: number | string
  unit?: string
  trend: 'up' | 'down' | 'stable'
  trendValue: number
  status: 'healthy' | 'warning' | 'critical' | 'info'
  tooltip: string
  sparkline?: number[]
}

interface KPISectionProps {
  title: string
  icon: React.ReactNode
  metrics: KPIMetric[]
  loading?: boolean
}

// Status configurations
const statusConfig = {
  healthy: {
    color: 'var(--status-healthy)',
    bgColor: 'var(--status-healthy-bg)',
    icon: <CheckCircleOutlined />,
  },
  warning: {
    color: 'var(--status-attention)',
    bgColor: 'var(--status-attention-bg)',
    icon: <ExclamationCircleOutlined />,
  },
  critical: {
    color: 'var(--status-critical)',
    bgColor: 'var(--status-critical-bg)',
    icon: <CloseCircleOutlined />,
  },
  info: {
    color: 'var(--status-info)',
    bgColor: 'var(--status-info-bg)',
    icon: <InfoCircleOutlined />,
  },
}

// Single KPI Card Component
function KPICard({ metric }: { metric: KPIMetric }) {
  const config = statusConfig[metric.status]

  const trendIcon = {
    up: <ArrowUpOutlined />,
    down: <ArrowDownOutlined />,
    stable: <MinusOutlined />,
  }[metric.trend]

  // Determine trend color based on metric type
  // For most metrics, up is good. For queries/SAEs, down is good
  const isPositiveTrendGood = !['open_queries', 'pending_saes', 'dirty_patients'].includes(metric.id)
  const trendIsPositive = metric.trend === 'up' ? isPositiveTrendGood : !isPositiveTrendGood
  const trendColor = metric.trend === 'stable'
    ? 'var(--gray-500)'
    : trendIsPositive
      ? 'var(--status-healthy)'
      : 'var(--status-critical)'

  return (
    <Tooltip title={metric.tooltip} placement="top">
      <div className={`kpi-card kpi-card--${metric.status}`}>
        <div className="kpi-card__header">
          <span className="kpi-card__title">{metric.title}</span>
          <span className={`kpi-card__status-icon kpi-card__status-icon--${metric.status}`}>
            {config.icon}
          </span>
        </div>

        <div
          className="kpi-card__value"
          style={{ color: metric.status === 'info' ? 'var(--gray-800)' : config.color }}
        >
          {typeof metric.value === 'number'
            ? metric.value.toLocaleString()
            : metric.value}
          {metric.unit && (
            <span style={{ fontSize: '18px', marginLeft: '4px', fontWeight: 500 }}>
              {metric.unit}
            </span>
          )}
        </div>

        <div className="kpi-card__trend" style={{ color: trendColor }}>
          {trendIcon}
          <span>
            {metric.trendValue > 0 ? '+' : ''}{metric.trendValue}% vs last period
          </span>
        </div>

        {/* Mini Sparkline */}
        {metric.sparkline && metric.sparkline.length > 0 && (
          <div className="kpi-card__sparkline">
            <svg width="100%" height="100%" viewBox="0 0 100 32" preserveAspectRatio="none">
              <polyline
                points={metric.sparkline.map((v, i) =>
                  `${(i / (metric.sparkline!.length - 1)) * 100},${32 - (v / Math.max(...metric.sparkline!)) * 28}`
                ).join(' ')}
                fill="none"
                stroke={config.color}
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        )}
      </div>
    </Tooltip>
  )
}

// KPI Section Component (group of related KPIs)
function KPISection({ title, icon, metrics, loading }: KPISectionProps) {
  if (loading) {
    return (
      <div className="kpi-section">
        <div className="section-header">
          <span className="section-header__icon">{icon}</span>
          <span className="section-header__title">{title}</span>
        </div>
        <div className="kpi-grid">
          {[1, 2, 3].map((i) => (
            <div key={i} className="kpi-card">
              <Skeleton active paragraph={{ rows: 2 }} />
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="kpi-section animate-fade-in">
      <div className="section-header">
        <span className="section-header__icon">{icon}</span>
        <span className="section-header__title">{title}</span>
      </div>
      <div className="kpi-grid">
        {metrics.map((metric) => (
          <KPICard key={metric.id} metric={metric} />
        ))}
      </div>
    </div>
  )
}

/**
 * EnhancedKPISections Component
 * 
 * Organizes KPIs into logical sections:
 * 1. Patient & Enrollment Health
 * 2. Data Quality & Compliance
 * 3. Query & Resolution Performance
 * 
 * PERFORMANCE OPTIMIZED: Uses bundled initial-load endpoint to fetch ALL data
 * in a single request instead of 3 separate API calls.
 */
export default function EnhancedKPISections() {
  const { selectedStudyId, selectedSiteId, dateRange } = useStore()
  const days = getDaysFromRange(dateRange, 30)
  const filterActive = Boolean(selectedSiteId || (dateRange.start && dateRange.end))

  // OPTIMIZED: Single bundled query fetches summary + query metrics + cleanliness
  // The getDashboardSummary function now uses the /api/dashboard/initial-load endpoint
  // which returns all data bundled together
  const { data: summary, isLoading, isError } = useQuery({
    queryKey: ['dashboardSummary', selectedStudyId],
    queryFn: () => metricsApi.getDashboardSummary(selectedStudyId || undefined),
    refetchInterval: 60000,
    staleTime: 30000, // Consider data fresh for 30 seconds
  })

  const { data: dqiFiltered } = useQuery({
    queryKey: ['dqiFiltered', selectedStudyId, selectedSiteId, days],
    queryFn: () => metricsApi.getDQI(selectedStudyId || undefined, selectedSiteId || undefined, days),
    refetchInterval: 60000,
    enabled: filterActive,
  })

  const { data: cleanlinessFiltered } = useQuery({
    queryKey: ['cleanlinessFiltered', selectedStudyId, selectedSiteId, days],
    queryFn: () => metricsApi.getCleanliness(selectedStudyId || undefined, selectedSiteId || undefined, days),
    refetchInterval: 60000,
    enabled: filterActive,
  })

  const { data: queriesFiltered } = useQuery({
    queryKey: ['queriesFiltered', selectedStudyId, selectedSiteId, days],
    queryFn: () => metricsApi.getQueries(selectedStudyId || undefined, selectedSiteId || undefined, days),
    refetchInterval: 60000,
    enabled: filterActive,
  })

  const showLoading = isLoading || (!summary && !isError)
  const showEmpty = !isLoading && (isError || !summary)

  // Extract bundled data from the optimized response
  // The _query_metrics and _cleanliness fields are populated from the bundled endpoint
  const queryMetrics = filterActive ? (queriesFiltered || null) : (summary?._query_metrics || null)
  const cleanliness = filterActive ? (cleanlinessFiltered || null) : (summary?._cleanliness || null)
  const effectiveDQI = filterActive ? (dqiFiltered?.overall_dqi ?? summary?.overall_dqi ?? 0) : (summary?.overall_dqi || 0)

  // Calculate Patient & Enrollment metrics
  const patientMetrics: KPIMetric[] = [
    {
      id: 'total_patients',
      title: 'Total Patients',
      value: summary?.total_patients || 0,
      trend: 'stable',
      trendValue: 0,
      status: 'info',
      tooltip: 'Total enrolled patients across all active studies',
    },
    {
      id: 'clean_patients',
      title: 'Clean Patient Rate',
      value: cleanliness?.cleanliness_rate?.toFixed(1) ?? '0',
      unit: '%',
      trend: 'stable',
      trendValue: 0,
      status: (cleanliness?.cleanliness_rate ?? 0) >= 90 ? 'healthy' : (cleanliness?.cleanliness_rate ?? 0) >= 75 ? 'warning' : 'critical',
      tooltip: 'Percentage of patients with no blocking data issues. Target: ≥90%',
      sparkline: cleanliness?.trend || [],
    },
    {
      id: 'at_risk_patients',
      title: 'At-Risk Patients',
      value: cleanliness?.at_risk_count || 0,
      trend: 'stable',
      trendValue: 0,
      status: (cleanliness?.at_risk_count || 0) > 200 ? 'warning' : 'healthy',
      tooltip: 'Patients approaching dirty status threshold',
    },
  ]

  // Calculate Data Quality metrics
  const dataQualityMetrics: KPIMetric[] = [
    {
      id: 'dqi',
      title: 'Data Quality Index',
      value: effectiveDQI,
      trend: 'stable',
      trendValue: 0,
      status: effectiveDQI >= 85 ? 'healthy' : effectiveDQI >= 70 ? 'warning' : 'critical',
      tooltip: 'Composite score measuring completeness, accuracy, consistency, and timeliness',
    },
    {
      id: 'pending_saes',
      title: 'Pending SAEs',
      value: summary?.pending_saes || 0,
      trend: 'stable',
      trendValue: 0,
      status: (summary?.pending_saes || 0) > 100 ? 'critical' : (summary?.pending_saes || 0) > 50 ? 'warning' : 'healthy',
      tooltip: 'Serious Adverse Events awaiting reconciliation with safety database',
    },
    {
      id: 'uncoded_terms',
      title: 'Uncoded Terms',
      value: summary?.uncoded_terms || 0,
      trend: 'stable',
      trendValue: 0,
      status: (summary?.uncoded_terms || 0) > 50 ? 'warning' : 'healthy',
      tooltip: 'Medical terms awaiting MedDRA/WHO Drug coding',
    },
  ]

  // Calculate Query Performance metrics
  const queryPerformanceMetrics: KPIMetric[] = [
    {
      id: 'open_queries',
      title: 'Open Queries',
      value: queryMetrics?.open_queries || summary?.open_queries || 0,
      trend: 'stable',
      trendValue: 0,
      status: (queryMetrics?.open_queries || 0) > 1000 ? 'critical' : (queryMetrics?.open_queries || 0) > 500 ? 'warning' : 'healthy',
      tooltip: 'Total queries currently open and requiring resolution',
    },
    {
      id: 'resolution_rate',
      title: 'Resolution Rate',
      value: queryMetrics?.resolution_rate?.toFixed(0) || '0',
      unit: '%',
      trend: 'stable',
      trendValue: 0,
      status: (queryMetrics?.resolution_rate || 0) >= 80 ? 'healthy' : (queryMetrics?.resolution_rate || 0) >= 60 ? 'warning' : 'critical',
      tooltip: 'Percentage of queries resolved within SLA timeframe',
    },
    {
      id: 'avg_resolution',
      title: 'Avg Resolution Time',
      value: queryMetrics?.avg_resolution_time?.toFixed(1) || '0',
      unit: 'days',
      trend: 'stable',
      trendValue: 0,
      status: (queryMetrics?.avg_resolution_time || 0) <= 5 ? 'healthy' : (queryMetrics?.avg_resolution_time || 0) <= 10 ? 'warning' : 'critical',
      tooltip: 'Average time to resolve a query. Target: ≤5 days',
    },
  ]

  if (showEmpty) {
    return <Empty description="KPI summary unavailable" />
  }

  // We only render the Patient/Enrollment metrics for the top row of the dashboard
  // The Data Quality metrics are handled by other charts

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24 }}>
      {patientMetrics.map((metric) => (
        <div key={metric.id} className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
          {/* Glow effect based on status */}
          <div style={{
            position: 'absolute',
            top: -50, right: -50, width: 100, height: 100, borderRadius: '50%',
            background: metric.status === 'healthy' ? 'var(--neon-green)' : metric.status === 'warning' ? 'var(--neon-yellow)' : 'var(--neon-cyan)',
            filter: 'blur(60px)',
            opacity: 0.15,
            pointerEvents: 'none'
          }} />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
            <span style={{
              color: 'rgba(255,255,255,0.7)',
              fontSize: 14,
              textTransform: 'uppercase',
              letterSpacing: 1,
              fontWeight: 500
            }}>
              {metric.title}
            </span>
            <Tooltip title={metric.tooltip}>
              <InfoCircleOutlined style={{ color: 'rgba(255,255,255,0.3)' }} />
            </Tooltip>
          </div>

          <div style={{ fontSize: 42, fontWeight: 'bold', color: '#fff', textShadow: '0 0 15px rgba(255,255,255,0.2)' }}>
            {typeof metric.value === 'number' ? metric.value.toLocaleString() : metric.value}
            {metric.unit && <span style={{ fontSize: 20, color: 'rgba(255,255,255,0.6)', marginLeft: 4 }}>{metric.unit}</span>}
          </div>

          {/* Trend or Sparkline */}
          {metric.sparkline ? (
            <div style={{ marginTop: 'auto', height: 40 }}>
              <svg width="100%" height="100%" viewBox="0 0 100 32" preserveAspectRatio="none">
                <polyline
                  points={metric.sparkline.map((v, i) =>
                    `${(i / (metric.sparkline!.length - 1)) * 100},${32 - (v / Math.max(...metric.sparkline!)) * 28}`
                  ).join(' ')}
                  fill="none"
                  stroke={metric.status === 'healthy' ? 'var(--neon-green)' : 'var(--neon-cyan)'}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
            </div>
          ) : (
            <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'rgba(255,255,255,0.5)' }}>
              <span style={{ color: metric.status === 'healthy' ? 'var(--neon-green)' : 'var(--neon-yellow)' }}>
                {metric.status === 'healthy' ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
              </span>
              <span>Status: {metric.status.toUpperCase()}</span>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
