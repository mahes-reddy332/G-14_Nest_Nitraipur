import React from 'react'
import { Typography, Tooltip, Progress, Space, Skeleton } from 'antd'
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
  const { selectedStudyId } = useStore()

  const { data: dqiMetrics, isLoading } = useQuery({
    queryKey: ['dqiBreakdown', selectedStudyId],
    queryFn: () => metricsApi.getDQIBreakdown(selectedStudyId || undefined),
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
    <div className="clinical-card animate-fade-in">
      <div className="clinical-card__header">
        <span className="clinical-card__title">
          <SafetyOutlined className="clinical-card__title-icon" />
          Data Quality Index
        </span>
        <Tooltip title="Composite score measuring overall data quality. Target: ≥85">
          <InfoCircleOutlined style={{ color: 'var(--gray-400)', cursor: 'help' }} />
        </Tooltip>
      </div>
      
      <div className="clinical-card__body">
        {/* Overall DQI Score */}
        <div 
          style={{ 
            textAlign: 'center', 
            marginBottom: 'var(--space-6)',
            padding: 'var(--space-4)',
            background: 'var(--gray-50)',
            borderRadius: 'var(--radius-lg)',
          }}
        >
          <div style={{ marginBottom: 'var(--space-2)' }}>
            <Text type="secondary" style={{ fontSize: 12, textTransform: 'uppercase' }}>
              Overall Score
            </Text>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: 4 }}>
            <span 
              style={{ 
                fontSize: 48, 
                fontWeight: 700, 
                color: getStatusColor(overallDQI, 85),
                lineHeight: 1,
              }}
            >
              {overallDQI.toFixed(0)}
            </span>
            <span style={{ fontSize: 20, color: 'var(--gray-500)' }}>/100</span>
          </div>
          <div style={{ marginTop: 'var(--space-2)' }}>
            <span 
              className={`metric-row__badge metric-row__badge--${overallStatus}`}
            >
              {overallStatus === 'healthy' ? 'On Target' : overallStatus === 'warning' ? 'Needs Attention' : 'Below Target'}
            </span>
          </div>
        </div>

        {/* Component Breakdown */}
        <div>
          <Text 
            strong 
            style={{ 
              fontSize: 11, 
              color: 'var(--gray-500)', 
              textTransform: 'uppercase',
              display: 'block',
              marginBottom: 'var(--space-3)',
            }}
          >
            Component Breakdown
          </Text>
          
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            {dqiComponents.map((component) => {
              const statusColor = getStatusColor(component.value, component.target)
              const statusClass = getStatusClass(component.value, component.target)
              
              return (
                <Tooltip key={component.id} title={component.tooltip}>
                  <div style={{ cursor: 'help' }}>
                    {/* Label Row */}
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      marginBottom: 'var(--space-1)',
                    }}>
                      <span style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 'var(--space-2)',
                        fontSize: 13,
                        color: 'var(--gray-700)',
                      }}>
                        <span style={{ color: statusColor }}>{component.icon}</span>
                        {component.label}
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                        {/* Trend Indicator */}
                        {component.trend && component.trend !== 'stable' && (
                          <span 
                            style={{ 
                              fontSize: 11,
                              color: component.trend === 'up' 
                                ? 'var(--status-healthy)' 
                                : 'var(--status-critical)',
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
                        }}>
                          {component.value.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    
                    {/* Progress Bar */}
                    <div 
                      style={{ 
                        position: 'relative',
                        height: 8,
                        background: 'var(--gray-100)',
                        borderRadius: 'var(--radius-sm)',
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
                          background: 'var(--gray-400)',
                          zIndex: 2,
                        }}
                      />
                      {/* Progress bar */}
                      <div 
                        style={{
                          height: '100%',
                          width: `${Math.min(component.value, 100)}%`,
                          background: statusColor,
                          borderRadius: 'var(--radius-sm)',
                          transition: 'width 0.3s ease',
                        }}
                      />
                    </div>
                    
                    {/* Target label */}
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'flex-end',
                      marginTop: 2,
                    }}>
                      <Text style={{ fontSize: 10, color: 'var(--gray-400)' }}>
                        Target: {component.target}%
                      </Text>
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
