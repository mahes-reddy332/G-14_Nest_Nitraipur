import React from 'react'
import { Row, Col, Typography, Progress, Tooltip, Badge, Space, Skeleton, Empty, Button } from 'antd'
import { ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
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
import { getDaysFromRange } from '../../utils/filtering'

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
  const { selectedStudyId, selectedSiteId, dateRange } = useStore()
  const days = getDaysFromRange(dateRange, 30)

  const { data: queryMetrics, isLoading, isError } = useQuery({
    queryKey: ['queryMetrics', selectedStudyId, selectedSiteId, days],
    queryFn: () => metricsApi.getQueries(selectedStudyId || undefined, selectedSiteId || undefined, days),
    refetchInterval: 60000,
  })

  // Loading Skeleton
  if (isLoading) {
    return (
      <div className="clinical-card" style={{ height: '100%' }}>
        <div className="clinical-card__header">
          <span className="clinical-card__title">
            <QuestionCircleOutlined className="clinical-card__title-icon" />
            Query Management
          </span>
        </div>
        <div className="clinical-card__body">
          <Skeleton active paragraph={{ rows: 3 }} />
        </div>
      </div>
    )
  }

  // Error State
  if (isError || !queryMetrics) {
    return (
      <div className="clinical-card" style={{ height: '100%' }}>
        <div className="clinical-card__header">
          <span className="clinical-card__title">
            <QuestionCircleOutlined className="clinical-card__title-icon" />
            Query Management
          </span>
        </div>
        <div className="clinical-card__body">
          <Empty description="Query metrics unavailable" />
        </div>
      </div>
    )
  }

  const openQueries = queryMetrics?.open_queries || 0
  const closedQueries = queryMetrics?.closed_queries || 0
  const totalQueries = queryMetrics?.total_queries || 1
  const resolutionRate = queryMetrics?.resolution_rate || 0
  const avgResolutionTime = queryMetrics?.avg_resolution_time || 0

  const queryStatusData = [
    { name: 'Open', value: openQueries, color: '#ff4d4f' },
    { name: 'Closed', value: closedQueries, color: '#52c41a' },
  ]

  // Calculate aging buckets
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

  // Anomaly Detection
  const estimatedDailyResolutions = queryMetrics ? Math.round((queryMetrics.resolution_rate / 100) * queryMetrics.total_queries / 30) : 0
  const hasResolutionAnomaly = openQueries > 1000 && estimatedDailyResolutions === 0

  return (
    <div className="clinical-card animate-fade-in" style={{ height: '100%' }}>
      {/* Header */}
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
        {/* Anomaly Banner */}
        {hasResolutionAnomaly && (
          <div
            style={{
              background: 'var(--status-attention-bg)',
              border: '1px solid var(--status-attention-border)',
              borderRadius: 'var(--radius-md)',
              padding: 'var(--space-3)',
              marginBottom: 'var(--space-5)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 'var(--space-2)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
              <WarningOutlined style={{ color: 'var(--status-attention)', fontSize: 20 }} />
              <div>
                <Text strong style={{ display: 'block', color: 'var(--status-attention)' }}>Resolution Anomaly Detected</Text>
                <Text style={{ fontSize: 13, color: 'var(--gray-400)' }}>
                  High query volume with zero daily resolutions detected in the selected period.
                </Text>
              </div>
            </div>
            <Button type="primary" ghost size="small" onClick={() => navigate('/queries')} danger>
              Investigate
            </Button>
          </div>
        )}

        <Row gutter={[32, 24]}>
          {/* Column 1: Query Status */}
          <Col xs={24} md={8}>
            <Title level={5} style={{ color: 'var(--gray-500)', fontSize: 12, textTransform: 'uppercase', margin: '0 0 16px 0' }}>
              Query Status
            </Title>
            <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
              <div style={{ width: 100, height: 100 }}>
                {openQueries + closedQueries > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={queryStatusData}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={35}
                        outerRadius={50}
                        paddingAngle={2}
                        stroke="none"
                      >
                        {queryStatusData.map((entry) => (
                          <Cell key={entry.name} fill={entry.color} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={null} />
                )}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ marginBottom: 12 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>Open</Text>
                  <div style={{ fontSize: 24, fontWeight: 'bold', lineHeight: 1, color: openQueries > 1000 ? 'var(--status-critical)' : '#fff' }}>
                    {openQueries.toLocaleString()}
                  </div>
                </div>
                <div>
                  <Text type="secondary" style={{ fontSize: 12 }}>Closed</Text>
                  <div style={{ fontSize: 24, fontWeight: 'bold', lineHeight: 1, color: 'var(--status-healthy)' }}>
                    {closedQueries.toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          </Col>

          {/* Column 2: Query Aging */}
          <Col xs={24} md={8} style={{ borderLeft: '1px solid var(--surface-border)', borderRight: '1px solid var(--surface-border)' }}>
            <Title level={5} style={{ color: 'var(--gray-500)', fontSize: 12, textTransform: 'uppercase', margin: '0 0 16px 0' }}>
              Open Query Aging
            </Title>
            <div className="query-aging" style={{ gap: 8 }}>
              {agingBuckets.map((bucket, index) => (
                <Tooltip
                  key={bucket.label}
                  title={`${bucket.count} queries (${bucket.percentage.toFixed(1)}%)${bucket.isSLABreach ? ' - SLA Breach!' : ''}`}
                >
                  <div
                    className={`query-aging__bucket ${bucket.isSLABreach ? 'query-aging__bucket--sla-breach' : ''}`}
                    onClick={() => navigate(`/queries?aging=${bucket.range}`)}
                    style={{ padding: '12px 4px' }}
                  >
                    <div className="query-aging__bucket-label" style={{ fontSize: 9 }}>{bucket.label}</div>
                    <div className={`query-aging__bucket-value query-aging__bucket--${index === 0 ? '0-7' : index === 1 ? '8-14' : index === 2 ? '15-30' : '30plus'
                      }`} style={{ fontSize: 16 }}>
                      {bucket.count > 1000 ? `${(bucket.count / 1000).toFixed(1)}k` : bucket.count}
                    </div>
                  </div>
                </Tooltip>
              ))}
            </div>
          </Col>

          {/* Column 3: Performance Metrics */}
          <Col xs={24} md={8}>
            <Title level={5} style={{ color: 'var(--gray-500)', fontSize: 12, textTransform: 'uppercase', margin: '0 0 16px 0' }}>
              Performance Metrics
            </Title>

            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <Text style={{ fontSize: 12, color: 'var(--gray-400)' }}>Avg Resolution Time</Text>
                <Text strong style={{
                  color: avgTimeStatus === 'healthy' ? 'var(--status-healthy)'
                    : avgTimeStatus === 'warning' ? 'var(--status-attention)'
                      : 'var(--status-critical)'
                }}>
                  {avgResolutionTime.toFixed(1)} days
                </Text>
              </div>
              <Progress
                percent={Math.min(100, Math.max(0, 100 - (avgResolutionTime * 5)))}
                showInfo={false}
                strokeColor={avgTimeStatus === 'healthy' ? 'var(--status-healthy)' : 'var(--status-critical)'}
                trailColor="rgba(255,255,255,0.05)"
                size="small"
              />
            </div>

            <Row gutter={16}>
              <Col span={12}>
                <div className="stat-mini" style={{ padding: '8px 12px' }}>
                  <div className="stat-mini__label">Queries/Day</div>
                  <div className="stat-mini__value" style={{ fontSize: 16 }}>
                    {queryMetrics ? Math.round(queryMetrics.total_queries / 30).toLocaleString() : '0'}
                  </div>
                </div>
              </Col>
              <Col span={12}>
                <div className="stat-mini" style={{ padding: '8px 12px' }}>
                  <div className="stat-mini__label">Resolutions/Day</div>
                  <div className="stat-mini__value" style={{ fontSize: 16 }}>
                    {estimatedDailyResolutions.toLocaleString()}
                  </div>
                </div>
              </Col>
            </Row>
          </Col>
        </Row>
      </div>
    </div>
  )
}
