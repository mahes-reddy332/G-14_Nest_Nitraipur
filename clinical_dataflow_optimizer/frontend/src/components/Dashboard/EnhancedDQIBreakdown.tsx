import React from 'react'
import { Typography, Tooltip, Progress, Space, Skeleton, Empty } from 'antd'
import {
  CheckCircleOutlined,
  SafetyOutlined,
  SyncOutlined,
  ClockCircleOutlined,
  InfoCircleOutlined,
  RiseOutlined,
  FallOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { metricsApi } from '../../api'
import { useStore } from '../../store'
import '../../styles/clinical-design-system.css'
import { getDaysFromRange } from '../../utils/filtering'

const { Text, Title } = Typography

interface DQIComponent {
  id: string
  label: string
  value: number
  target: number
  icon: React.ReactNode
  tooltip: string
  trend?: 'up' | 'down' | 'stable'
  trendValue?: number
}

/**
 * EnhancedDQIBreakdown Component
 * 
 * Improved Data Quality Index breakdown with:
 * - Progress bars instead of donut charts for precise values
 * - Clear thresholds and targets
 * - Trend indicators
 * - Tooltip explanations
 */
export default function EnhancedDQIBreakdown() {
  const { selectedStudyId, selectedSiteId, dateRange } = useStore()
  const days = getDaysFromRange(dateRange, 30)

  const { data: dqiMetrics, isLoading, isError } = useQuery({
    queryKey: ['dqiBreakdown', selectedStudyId, selectedSiteId, days],
    queryFn: () => metricsApi.getDQIBreakdown(selectedStudyId || undefined, selectedSiteId || undefined, days),
    refetchInterval: 60000,
  })

  if (isLoading) {
    return (
      <div className="clinical-card">
        <div className="clinical-card__header">
          <span className="clinical-card__title">
            <SafetyOutlined className="clinical-card__title-icon" />
            Data Quality Index Breakdown
          </span>
        </div>
        <div className="clinical-card__body">
          <Skeleton active paragraph={{ rows: 6 }} />
        </div>
      </div>
    )
  }

  if (isError || !dqiMetrics) {
    return (
      <div className="clinical-card">
        <div className="clinical-card__header">
          <span className="clinical-card__title">
            <SafetyOutlined className="clinical-card__title-icon" />
            Data Quality Index
          </span>
        </div>
        <div className="clinical-card__body">
          <Empty description="DQI metrics unavailable" />
        </div>
      </div>
    )
  }

  const overallDQI = dqiMetrics?.overall || 0

  // DQI Components with targets
  const dqiComponents: DQIComponent[] = [
    {
      id: 'completeness',
      label: 'Completeness',
      value: dqiMetrics?.completeness || 0,
      target: 98,
      icon: <CheckCircleOutlined />,
      tooltip: 'Percentage of required data fields that are populated. Target: ≥98%',
      trend: 'up',
      trendValue: 1.2,
    },
    {
      id: 'accuracy',
      label: 'Accuracy',
      value: dqiMetrics?.accuracy || 0,
      target: 100,
      icon: <SafetyOutlined />,
      tooltip: 'Percentage of data values that pass validation rules. Target: 100%',
      trend: 'stable',
      trendValue: 0,
    },
    {
      id: 'consistency',
      label: 'Consistency',
      value: dqiMetrics?.consistency || 0,
      target: 95,
      icon: <SyncOutlined />,
      tooltip: 'Cross-form data consistency without contradictions. Target: ≥95%',
      trend: 'down',
      trendValue: -2.1,
    },
    {
      id: 'timeliness',
      label: 'Timeliness',
      value: dqiMetrics?.timeliness || 0,
      target: 90,
      icon: <ClockCircleOutlined />,
      tooltip: 'Data entry within expected visit windows. Target: ≥90%',
      trend: 'up',
      trendValue: 3.5,
    },
  ]

  const getStatusColor = (value: number, target: number) => {
    const ratio = value / target
    if (ratio >= 0.95) return 'var(--status-healthy)'
    if (ratio >= 0.80) return 'var(--status-attention)'
    return 'var(--status-critical)'
  }

  const getStatusClass = (value: number, target: number) => {
    const ratio = value / target
    if (ratio >= 0.95) return 'healthy'
    if (ratio >= 0.80) return 'warning'
    return 'critical'
  }

  const overallStatus = getStatusClass(overallDQI, 85)

  return (
    <div className="glass-panel" style={{ height: '100%', padding: 24 }}>
      <div className="clinical-card__header">
        <span className="clinical-card__title" style={{ color: 'var(--neon-green)' }}>
          <SafetyOutlined className="clinical-card__title-icon" style={{ color: 'var(--neon-green)' }} />
          Data Quality Index
        </span>
        <Tooltip title="Composite score measuring overall data quality. Target: ≥85">
          <InfoCircleOutlined style={{ color: 'rgba(255, 255, 255, 0.3)', cursor: 'help' }} />
        </Tooltip>
      </div>

      <div className="clinical-card__body">
        {/* Overall DQI Score */}
        <div
          style={{
            textAlign: 'center',
            marginBottom: 24,
            padding: 24,
            background: 'rgba(0, 0, 0, 0.2)',
            borderRadius: 16,
            border: '1px solid rgba(255, 255, 255, 0.05)'
          }}
        >
          <div style={{ marginBottom: 8 }}>
            <Text style={{ fontSize: 12, textTransform: 'uppercase', color: 'rgba(255, 255, 255, 0.5)', letterSpacing: 1 }}>
              Overall Score
            </Text>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: 4 }}>
            <span
              style={{
                fontSize: 56,
                fontWeight: 700,
                color: overallStatus === 'healthy' ? 'var(--neon-green)' : overallStatus === 'warning' ? 'var(--neon-yellow)' : 'var(--neon-red)',
                lineHeight: 1,
                textShadow: `0 0 20px ${overallStatus === 'healthy' ? 'var(--neon-green)' : overallStatus === 'warning' ? 'var(--neon-yellow)' : 'var(--neon-red)'}`
              }}
            >
              {overallDQI.toFixed(0)}
            </span>
            <span style={{ fontSize: 20, color: 'rgba(255, 255, 255, 0.3)' }}>/100</span>
          </div>
          <div style={{ marginTop: 12 }}>
            <span
              style={{
                padding: '4px 12px',
                borderRadius: 12,
                fontSize: 12,
                fontWeight: 600,
                background: overallStatus === 'healthy' ? 'rgba(0, 255, 153, 0.1)' : overallStatus === 'warning' ? 'rgba(252, 238, 10, 0.1)' : 'rgba(255, 51, 51, 0.1)',
                color: overallStatus === 'healthy' ? 'var(--neon-green)' : overallStatus === 'warning' ? 'var(--neon-yellow)' : 'var(--neon-red)',
                border: `1px solid ${overallStatus === 'healthy' ? 'var(--neon-green)' : overallStatus === 'warning' ? 'var(--neon-yellow)' : 'var(--neon-red)'}`
              }}
            >
              {overallStatus === 'healthy' ? 'ON TARGET' : overallStatus === 'warning' ? 'NEEDS ATTENTION' : 'BELOW TARGET'}
            </span>
          </div>
        </div>

        {/* Component Breakdown */}
        <div>
          <Text
            strong
            style={{
              fontSize: 11,
              color: 'rgba(255, 255, 255, 0.4)',
              textTransform: 'uppercase',
              display: 'block',
              marginBottom: 16,
              letterSpacing: 1
            }}
          >
            Component Breakdown
          </Text>

          <Space direction="vertical" size={20} style={{ width: '100%' }}>
            {dqiComponents.map((component) => {
              const statusColor = component.value >= component.target * 0.95 ? 'var(--neon-green)' : component.value >= component.target * 0.8 ? 'var(--neon-yellow)' : 'var(--neon-red)';

              return (
                <Tooltip key={component.id} title={component.tooltip}>
                  <div style={{ cursor: 'help' }}>
                    {/* Label Row */}
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: 6,
                    }}>
                      <span style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        fontSize: 13,
                        color: 'rgba(255, 255, 255, 0.8)',
                      }}>
                        <span style={{ color: statusColor }}>{component.icon}</span>
                        {component.label}
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        {/* Trend Indicator */}
                        {component.trend && component.trend !== 'stable' && (
                          <span
                            style={{
                              fontSize: 11,
                              color: component.trend === 'up'
                                ? 'var(--neon-green)'
                                : 'var(--neon-red)',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 2,
                            }}
                          >
                            {component.trend === 'up' ? <RiseOutlined /> : <FallOutlined />}
                            {Math.abs(component.trendValue || 0)}%
                          </span>
                        )}
                        <span style={{
                          fontSize: 14,
                          fontWeight: 600,
                          color: statusColor,
                          minWidth: 45,
                          textAlign: 'right',
                          textShadow: `0 0 5px ${statusColor}`
                        }}>
                          {component.value.toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div
                      style={{
                        position: 'relative',
                        height: 6,
                        background: 'rgba(255, 255, 255, 0.1)',
                        borderRadius: 3,
                        overflow: 'hidden',
                      }}
                    >
                      {/* Target line */}
                      <div
                        style={{
                          position: 'absolute',
                          left: `${component.target}%`,
                          top: 0,
                          bottom: 0,
                          width: 2,
                          background: 'rgba(255, 255, 255, 0.3)',
                          zIndex: 2,
                        }}
                      />
                      {/* Progress bar */}
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.min(component.value, 100)}%`,
                          background: statusColor,
                          borderRadius: 3,
                          transition: 'width 0.3s ease',
                          boxShadow: `0 0 8px ${statusColor}`
                        }}
                      />
                    </div>
                  </div>
                </Tooltip>
              )
            })}
          </Space>
        </div>
      </div>
    </div>
  )
}
