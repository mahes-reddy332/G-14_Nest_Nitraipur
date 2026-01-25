import React from 'react'
import { Row, Col, Tooltip, Skeleton } from 'antd'
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
 */
export default function EnhancedKPISections() {
  const { selectedStudyId } = useStore()

  // Fetch all metrics
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['dashboardSummary', selectedStudyId],
    queryFn: () => metricsApi.getDashboardSummary(selectedStudyId || undefined),
    refetchInterval: 60000,
  })

  const { data: queryMetrics, isLoading: queryLoading } = useQuery({
    queryKey: ['queryMetrics', selectedStudyId],
    queryFn: () => metricsApi.getQueries(selectedStudyId || undefined),
    refetchInterval: 60000,
  })

  const { data: cleanliness, isLoading: cleanlinessLoading } = useQuery({
    queryKey: ['cleanlinessMetrics', selectedStudyId],
    queryFn: () => metricsApi.getCleanliness(selectedStudyId || undefined),
    refetchInterval: 60000,
  })

  const isLoading = summaryLoading || queryLoading || cleanlinessLoading

  // Calculate Patient & Enrollment metrics
  const patientMetrics: KPIMetric[] = [
    {
      id: 'total_patients',
      title: 'Total Patients',
      value: summary?.total_patients || 0,
      trend: 'up',
      trendValue: 5.2,
      status: 'info',
      tooltip: 'Total enrolled patients across all active studies',
      sparkline: [85, 88, 90, 92, 95, 97, 100],
    },
    {
      id: 'clean_patients',
      title: 'Clean Patient Rate',
      value: cleanliness?.cleanliness_rate?.toFixed(1) ?? '0',
      unit: '%',
      trend: (cleanliness?.cleanliness_rate ?? 0) >= 90 ? 'stable' : 'down',
      trendValue: 2.3,
      status: (cleanliness?.cleanliness_rate ?? 0) >= 90 ? 'healthy' : (cleanliness?.cleanliness_rate ?? 0) >= 75 ? 'warning' : 'critical',
      tooltip: 'Percentage of patients with no blocking data issues. Target: ≥90%',
      sparkline: cleanliness?.trend || [88, 89, 90, 89, 91, 90, 91],
    },
    {
      id: 'at_risk_patients',
      title: 'At-Risk Patients',
      value: cleanliness?.at_risk_count || 0,
      trend: 'up',
      trendValue: 8,
      status: (cleanliness?.at_risk_count || 0) > 200 ? 'warning' : 'healthy',
      tooltip: 'Patients approaching dirty status threshold',
    },
  ]

  // Calculate Data Quality metrics
  const dataQualityMetrics: KPIMetric[] = [
    {
      id: 'dqi',
      title: 'Data Quality Index',
      value: summary?.overall_dqi || 0,
      trend: 'stable',
      trendValue: 0.5,
      status: (summary?.overall_dqi || 0) >= 85 ? 'healthy' : (summary?.overall_dqi || 0) >= 70 ? 'warning' : 'critical',
      tooltip: 'Composite score measuring completeness, accuracy, consistency, and timeliness',
      sparkline: [82, 84, 85, 84, 86, 87, 87],
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
      trend: 'down',
      trendValue: -15,
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
      trend: 'down',
      trendValue: -8.5,
      status: (queryMetrics?.open_queries || 0) > 1000 ? 'critical' : (queryMetrics?.open_queries || 0) > 500 ? 'warning' : 'healthy',
      tooltip: 'Total queries currently open and requiring resolution',
    },
    {
      id: 'resolution_rate',
      title: 'Resolution Rate',
      value: queryMetrics?.resolution_rate?.toFixed(0) || '0',
      unit: '%',
      trend: 'up',
      trendValue: 3.2,
      status: (queryMetrics?.resolution_rate || 0) >= 80 ? 'healthy' : (queryMetrics?.resolution_rate || 0) >= 60 ? 'warning' : 'critical',
      tooltip: 'Percentage of queries resolved within SLA timeframe',
    },
    {
      id: 'avg_resolution',
      title: 'Avg Resolution Time',
      value: queryMetrics?.avg_resolution_time?.toFixed(1) || '0',
      unit: 'days',
      trend: 'down',
      trendValue: -12,
      status: (queryMetrics?.avg_resolution_time || 0) <= 5 ? 'healthy' : (queryMetrics?.avg_resolution_time || 0) <= 10 ? 'warning' : 'critical',
      tooltip: 'Average time to resolve a query. Target: ≤5 days',
    },
  ]

  return (
    <div className="enhanced-kpi-sections">
      <Row gutter={[24, 0]}>
        <Col xs={24}>
          <KPISection
            title="Patient & Enrollment Health"
            icon={<UserOutlined />}
            metrics={patientMetrics}
            loading={isLoading}
          />
        </Col>
      </Row>
      
      <Row gutter={[24, 0]}>
        <Col xs={24} lg={12}>
          <KPISection
            title="Data Quality & Compliance"
            icon={<SafetyOutlined />}
            metrics={dataQualityMetrics}
            loading={isLoading}
          />
        </Col>
        <Col xs={24} lg={12}>
          <KPISection
            title="Query & Resolution Performance"
            icon={<FileSearchOutlined />}
            metrics={queryPerformanceMetrics}
            loading={isLoading}
          />
        </Col>
      </Row>
    </div>
  )
}
