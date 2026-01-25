import React from 'react'
import { Row, Col, Typography, Progress, Tooltip, Badge, Space, Skeleton } from 'antd'
import {
  QuestionCircleOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { metricsApi } from '../../api'
import { useStore } from '../../store'
import '../../styles/clinical-design-system.css'

const { Text, Title } = Typography

interface AgingBucket {
  label: string
  range: string
  count: number
  percentage: number
  status: 'healthy' | 'warning' | 'critical'
  isSLABreach: boolean
}

/**
 * EnhancedQueryStatus Component
 * 
 * Improved Query Management panel with:
 * - Clear aging buckets for rapid assessment
 * - SLA breach indicators
 * - Actionable drill-down affordances
 * - Resolution velocity metrics
 */
export default function EnhancedQueryStatus() {
  const navigate = useNavigate()
  const { selectedStudyId } = useStore()

  const { data: queryMetrics, isLoading } = useQuery({
    queryKey: ['queryMetrics', selectedStudyId],
    queryFn: () => metricsApi.getQueries(selectedStudyId || undefined),
    refetchInterval: 60000,
  })

  if (isLoading) {
    return (
      <div className="clinical-card">
        <div className="clinical-card__header">
          <span className="clinical-card__title">
            <QuestionCircleOutlined className="clinical-card__title-icon" />
            Query Management
          </span>
        </div>
        <div className="clinical-card__body">
          <Skeleton active paragraph={{ rows: 6 }} />
        </div>
      </div>
    )
  }

  const openQueries = queryMetrics?.open_queries || 0
  const closedQueries = queryMetrics?.closed_queries || 0
  const totalQueries = queryMetrics?.total_queries || 1
  const resolutionRate = queryMetrics?.resolution_rate || 0
  const avgResolutionTime = queryMetrics?.avg_resolution_time || 0

  // Calculate aging buckets from the aging_distribution
  const agingDist = queryMetrics?.aging_distribution || {}
  const agingBuckets: AgingBucket[] = [
    {
      label: '0-7 days',
      range: '0_7_days',
      count: agingDist['0_7_days'] || agingDist['0-7 days'] || 0,
      percentage: 0,
      status: 'healthy',
      isSLABreach: false,
    },
    {
      label: '8-14 days',
      range: '8_14_days',
      count: agingDist['8_14_days'] || agingDist['8-14 days'] || 0,
      percentage: 0,
      status: 'warning',
      isSLABreach: false,
    },
    {
      label: '15-30 days',
      range: '15_30_days',
      count: agingDist['15_30_days'] || agingDist['15-30 days'] || 0,
      percentage: 0,
      status: 'warning',
      isSLABreach: false,
    },
    {
      label: '30+ days',
      range: '30_plus_days',
      count: agingDist['30_plus_days'] || agingDist['30+ days'] || agingDist['30+_days'] || 0,
      percentage: 0,
      status: 'critical',
      isSLABreach: true,
    },
  ]

  // Calculate percentages
  const totalAging = agingBuckets.reduce((sum, b) => sum + b.count, 0) || 1
  agingBuckets.forEach(bucket => {
    bucket.percentage = (bucket.count / totalAging) * 100
  })

  // Resolution status
  const resolutionStatus = resolutionRate >= 80 ? 'healthy' : resolutionRate >= 60 ? 'warning' : 'critical'
  const avgTimeStatus = avgResolutionTime <= 5 ? 'healthy' : avgResolutionTime <= 10 ? 'warning' : 'critical'

  // Check for anomaly: high queries but zero resolutions
  // Note: daily_resolutions is calculated from resolution_rate * total_queries / 100
  const estimatedDailyResolutions = queryMetrics ? Math.round((queryMetrics.resolution_rate / 100) * queryMetrics.total_queries / 30) : 0
  const hasResolutionAnomaly = openQueries > 1000 && estimatedDailyResolutions === 0

  return (
    <div className="clinical-card animate-fade-in">
      <div className="clinical-card__header">
        <span className="clinical-card__title">
          <QuestionCircleOutlined className="clinical-card__title-icon" />
          Query Management
        </span>
        <Space>
          <Badge 
            count={`${resolutionRate.toFixed(0)}% Resolved`}
            style={{ 
              backgroundColor: resolutionStatus === 'healthy' 
                ? 'var(--status-healthy)' 
                : resolutionStatus === 'warning'
                  ? 'var(--status-attention)'
                  : 'var(--status-critical)'
            }}
          />
        </Space>
      </div>
      
      <div className="clinical-card__body">
        {/* Anomaly Warning */}
        {hasResolutionAnomaly && (
          <div 
            style={{ 
              background: 'var(--status-attention-bg)', 
              border: '1px solid var(--status-attention-border)',
              borderRadius: 'var(--radius-md)',
              padding: 'var(--space-3)',
              marginBottom: 'var(--space-4)',
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--space-2)',
            }}
          >
            <WarningOutlined style={{ color: 'var(--status-attention)' }} />
            <Text style={{ fontSize: 12 }}>
              <strong>Resolution Anomaly:</strong> High query volume with zero daily resolutions. 
              <a onClick={() => navigate('/queries')} style={{ marginLeft: 4 }}>Investigate →</a>
            </Text>
          </div>
        )}

        {/* Key Stats Row */}
        <Row gutter={[16, 16]} style={{ marginBottom: 'var(--space-5)' }}>
          <Col span={8}>
            <div className="stat-mini">
              <div className="stat-mini__label">Open</div>
              <div 
                className="stat-mini__value" 
                style={{ color: openQueries > 1000 ? 'var(--status-critical)' : 'var(--gray-800)' }}
              >
                {openQueries.toLocaleString()}
              </div>
            </div>
          </Col>
          <Col span={8}>
            <div className="stat-mini">
              <div className="stat-mini__label">Closed</div>
              <div className="stat-mini__value" style={{ color: 'var(--status-healthy)' }}>
                {closedQueries.toLocaleString()}
              </div>
            </div>
          </Col>
          <Col span={8}>
            <div className="stat-mini">
              <div className="stat-mini__label">Avg Time</div>
              <div 
                className="stat-mini__value"
                style={{ 
                  color: avgTimeStatus === 'healthy' 
                    ? 'var(--status-healthy)' 
                    : avgTimeStatus === 'warning'
                      ? 'var(--status-attention)'
                      : 'var(--status-critical)'
                }}
              >
                {avgResolutionTime.toFixed(1)}d
              </div>
            </div>
          </Col>
        </Row>

        {/* Query Aging Buckets */}
        <div style={{ marginBottom: 'var(--space-4)' }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 'var(--space-2)', 
            marginBottom: 'var(--space-3)' 
          }}>
            <ClockCircleOutlined style={{ color: 'var(--gray-500)' }} />
            <Text strong style={{ fontSize: 12, color: 'var(--gray-600)' }}>
              OPEN QUERY AGING
            </Text>
          </div>
          
          <div className="query-aging">
            {agingBuckets.map((bucket, index) => (
              <Tooltip 
                key={bucket.label}
                title={`${bucket.count} queries (${bucket.percentage.toFixed(1)}%)${bucket.isSLABreach ? ' - SLA Breach!' : ''}`}
              >
                <div 
                  className={`query-aging__bucket ${bucket.isSLABreach ? 'query-aging__bucket--sla-breach' : ''}`}
                  onClick={() => navigate(`/queries?aging=${bucket.range}`)}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="query-aging__bucket-label">
                    {bucket.label}
                    {bucket.isSLABreach && (
                      <ThunderboltOutlined 
                        style={{ color: 'var(--status-critical)', marginLeft: 4 }} 
                      />
                    )}
                  </div>
                  <div 
                    className={`query-aging__bucket-value query-aging__bucket--${
                      index === 0 ? '0-7' : index === 1 ? '8-14' : index === 2 ? '15-30' : '30plus'
                    }`}
                  >
                    {bucket.count.toLocaleString()}
                  </div>
                </div>
              </Tooltip>
            ))}
          </div>
        </div>

        {/* Resolution Progress Bar */}
        <div style={{ marginBottom: 'var(--space-4)' }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            marginBottom: 'var(--space-2)' 
          }}>
            <Text style={{ fontSize: 12, color: 'var(--gray-600)' }}>Resolution Progress</Text>
            <Text strong style={{ fontSize: 12 }}>
              {closedQueries.toLocaleString()} / {totalQueries.toLocaleString()}
            </Text>
          </div>
          <div className="clinical-progress">
            <div 
              className={`clinical-progress__bar clinical-progress__bar--${resolutionStatus}`}
              style={{ width: `${resolutionRate}%` }}
            />
          </div>
        </div>

        {/* Quick Metrics */}
        <Row gutter={[12, 8]}>
          <Col span={12}>
            <div className="metric-row">
              <span className="metric-row__label">
                <ThunderboltOutlined style={{ marginRight: 4 }} />
                Queries/Day
              </span>
              <span className="metric-row__value">
                {queryMetrics ? Math.round(queryMetrics.total_queries / 30).toLocaleString() : '0'}
              </span>
            </div>
          </Col>
          <Col span={12}>
            <div className="metric-row">
              <span className="metric-row__label">
                <CheckCircleOutlined style={{ marginRight: 4 }} />
                Resolutions/Day
              </span>
              <span 
                className="metric-row__value"
                style={{ 
                  color: estimatedDailyResolutions === 0 
                    ? 'var(--status-critical)' 
                    : 'var(--gray-800)' 
                }}
              >
                {estimatedDailyResolutions.toLocaleString()}
              </span>
            </div>
          </Col>
        </Row>
      </div>
    </div>
  )
}
