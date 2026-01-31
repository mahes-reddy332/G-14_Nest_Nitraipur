import React from 'react'
import { Typography, Button, Tooltip, Space } from 'antd'
import {
  AlertOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  RightOutlined,
  FireOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  QuestionCircleOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { metricsApi, alertsApi } from '../../api'
import { useStore } from '../../store'
import '../../styles/clinical-design-system.css'

const { Text } = Typography

interface AtRiskItem {
  id: string
  label: string
  value: number
  severity: 'critical' | 'warning' | 'info'
  icon: React.ReactNode
  link: string
  tooltip: string
}

/**
 * AtRiskSummary Component
 * 
 * A prominent banner at the top of the dashboard that highlights
 * critical issues requiring immediate attention. Designed for
 * executive-level visibility in <30 seconds.
 * 
 * PERFORMANCE OPTIMIZED: Uses bundled dashboard summary which includes
 * query metrics and alert data, reducing from 3 API calls to 1.
 */
export default function AtRiskSummary() {
  const navigate = useNavigate()
  const { selectedStudyId } = useStore()

  // OPTIMIZED: Single bundled query fetches all needed data
  // The getDashboardSummary now returns _query_metrics and _alerts from bundled endpoint
  const { data: summary, isLoading } = useQuery({
    queryKey: ['dashboardSummary', selectedStudyId],
    queryFn: () => metricsApi.getDashboardSummary(selectedStudyId || undefined),
    refetchInterval: 30000,
    staleTime: 15000, // Consider data fresh for 15 seconds
  })

  if (isLoading || !summary) {
    return (
      <div className="at-risk-banner at-risk-banner--warning animate-fade-in">
        <div
          className="at-risk-banner__icon"
          style={{ background: 'var(--status-info)' }}
        >
          <ClockCircleOutlined />
        </div>
        <div className="at-risk-banner__content">
          <div className="at-risk-banner__title">Loading Clinical Data</div>
          <Text type="secondary">Initializing real-time metrics from source files.</Text>
        </div>
      </div>
    )
  }

  // Extract bundled data - no separate API calls needed!
  const queryMetrics = summary?._query_metrics || null
  const alertSummary = summary?._alerts || null

  // Calculate at-risk items
  const atRiskItems: AtRiskItem[] = []
  let overallSeverity: 'critical' | 'warning' | 'healthy' = 'healthy'

  // Check for critical alerts (using bundled data format)
  const criticalAlerts = alertSummary?.critical_count || 0
  if (criticalAlerts > 0) {
    atRiskItems.push({
      id: 'critical-alerts',
      label: 'Critical Alerts',
      value: criticalAlerts,
      severity: 'critical',
      icon: <AlertOutlined />,
      link: '/alerts?severity=critical',
      tooltip: 'Alerts requiring immediate attention',
    })
    overallSeverity = 'critical'
  }

  // Check for open queries
  const openQueries = summary?.open_queries || queryMetrics?.open_queries || 0
  if (openQueries > 0) {
    const querySeverity = openQueries > 1000 ? 'critical' : openQueries > 500 ? 'warning' : 'info'
    atRiskItems.push({
      id: 'open-queries',
      label: 'Open Queries',
      value: openQueries,
      severity: querySeverity,
      icon: <QuestionCircleOutlined />,
      link: '/queries?status=open',
      tooltip: 'Queries awaiting resolution',
    })
    if (querySeverity === 'critical' && overallSeverity !== 'critical') {
      overallSeverity = 'critical'
    } else if (querySeverity === 'warning' && overallSeverity === 'healthy') {
      overallSeverity = 'warning'
    }
  }

  // Check for pending SAEs
  const pendingSAEs = summary?.pending_saes || 0
  if (pendingSAEs > 0) {
    const saeSeverity = pendingSAEs > 100 ? 'critical' : 'warning'
    atRiskItems.push({
      id: 'pending-saes',
      label: 'Pending SAEs',
      value: pendingSAEs,
      severity: saeSeverity,
      icon: <FireOutlined />,
      link: '/safety?status=pending',
      tooltip: 'Serious Adverse Events awaiting reconciliation',
    })
    if (saeSeverity === 'critical') overallSeverity = 'critical'
    else if (overallSeverity === 'healthy') overallSeverity = 'warning'
  }

  // Check for overdue queries (SLA breach)
  const overdueQueries = queryMetrics?.aging_distribution?.['30+ days'] || 0
  if (overdueQueries > 0) {
    atRiskItems.push({
      id: 'overdue-queries',
      label: 'SLA Breach (30+ days)',
      value: overdueQueries,
      severity: 'critical',
      icon: <ClockCircleOutlined />,
      link: '/queries?aging=30plus',
      tooltip: 'Queries exceeding 30-day SLA threshold',
    })
    overallSeverity = 'critical'
  }

  // Check for zero resolutions with high queries (anomaly detection)
  const resolutionRate = queryMetrics?.resolution_rate || 0
  const estimatedResolutionsPerDay = queryMetrics ? Math.round((queryMetrics.resolution_rate / 100) * queryMetrics.total_queries / 30) : 0
  if (openQueries > 1000 && estimatedResolutionsPerDay === 0) {
    atRiskItems.push({
      id: 'resolution-anomaly',
      label: 'Resolution Anomaly',
      value: 0,
      severity: 'warning',
      icon: <ExclamationCircleOutlined />,
      link: '/queries',
      tooltip: 'High query volume with zero daily resolutions - investigate workflow',
    })
    if (overallSeverity === 'healthy') overallSeverity = 'warning'
  }

  // If no at-risk items, show healthy state
  if (atRiskItems.length === 0) {
    return (
      <div className="at-risk-banner at-risk-banner--healthy animate-fade-in">
        <div
          className="at-risk-banner__icon"
          style={{ background: 'var(--status-healthy)' }}
        >
          <CheckCircleOutlined />
        </div>
        <div className="at-risk-banner__content">
          <div className="at-risk-banner__title">
            All Systems Operational
          </div>
          <Text type="secondary">
            No critical issues detected. Data quality metrics are within acceptable thresholds.
          </Text>
        </div>
      </div>
    )
  }

  const bannerClass = overallSeverity === 'critical' 
    ? 'at-risk-banner' 
    : 'at-risk-banner at-risk-banner--warning'

  const iconClass = overallSeverity === 'critical' ? 'pulse-critical' : ''

  return (
    <div className={`${bannerClass} animate-fade-in`}>
      <div
        className={`at-risk-banner__icon ${iconClass}`}
        style={{
          background: overallSeverity === 'critical' 
            ? 'var(--status-critical)' 
            : 'var(--status-attention)'
        }}
      >
        {overallSeverity === 'critical' ? <AlertOutlined /> : <WarningOutlined />}
      </div>
      
      <div className="at-risk-banner__content">
        <div className="at-risk-banner__title">
          {overallSeverity === 'critical' 
            ? 'Critical Issues Require Attention' 
            : 'Items Requiring Review'}
        </div>
        <Text type="secondary">
          {atRiskItems.length} issue{atRiskItems.length > 1 ? 's' : ''} identified that may impact study timelines or data quality
        </Text>
        
        <div className="at-risk-banner__items">
          {atRiskItems.map((item) => (
            <Tooltip key={item.id} title={item.tooltip}>
              <div
                className={`at-risk-item at-risk-item--${item.severity}`}
                onClick={() => navigate(item.link)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    navigate(item.link)
                  }
                }}
                role="button"
                tabIndex={0}
                aria-label={`${item.label} ${item.value}`}
                style={{ cursor: 'pointer' }}
              >
                <span style={{ opacity: 0.7 }}>{item.icon}</span>
                <span className="at-risk-item__value">{item.value.toLocaleString()}</span>
                <span>{item.label}</span>
                <RightOutlined style={{ fontSize: 10, opacity: 0.5 }} />
              </div>
            </Tooltip>
          ))}
        </div>
      </div>
      
      <Space>
        <Button 
          type="primary" 
          size="small"
          onClick={() => navigate('/alerts')}
          style={{
            background: overallSeverity === 'critical' 
              ? 'var(--status-critical)' 
              : 'var(--status-attention)',
            borderColor: 'transparent',
          }}
        >
          View All Issues
        </Button>
      </Space>
    </div>
  )
}
