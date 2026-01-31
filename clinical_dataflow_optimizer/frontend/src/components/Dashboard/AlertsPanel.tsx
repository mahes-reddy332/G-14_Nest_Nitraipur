import { Card, List, Tag, Typography, Progress, Tooltip } from 'antd'
import {
  AlertOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { alertsApi } from '../../api'
import type { Alert } from '../../types'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'

dayjs.extend(relativeTime)

const { Text } = Typography

const severityConfig = {
  critical: { color: '#ff4d4f', icon: <AlertOutlined /> },
  high: { color: '#fa8c16', icon: <WarningOutlined /> },
  medium: { color: '#faad14', icon: <InfoCircleOutlined /> },
  low: { color: '#52c41a', icon: <CheckCircleOutlined /> },
}

const statusColors = {
  new: 'red',
  acknowledged: 'orange',
  in_progress: 'blue',
  resolved: 'green',
  dismissed: 'default',
}

interface AlertItemProps {
  alert: Alert
  onClick: () => void
}

function AlertItem({ alert, onClick }: AlertItemProps) {
  const severityKey = alert.severity || 'low'
  const config = severityConfig[severityKey] || severityConfig.low
  const statusKey = alert.status || 'new'
  const categoryLabel = alert.category ? alert.category.replace('_', ' ') : 'general'

  return (
    <List.Item
      onClick={onClick}
      style={{ cursor: 'pointer' }}
      extra={
        <Tag color={statusColors[statusKey] || 'default'}>{statusKey.replace('_', ' ')}</Tag>
      }
    >
      <List.Item.Meta
        avatar={
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              backgroundColor: `${config.color}20`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: config.color,
            }}
          >
            {config.icon}
          </div>
        }
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Tag color={config.color} style={{ margin: 0 }}>
              {severityKey}
            </Tag>
            <Text strong ellipsis style={{ maxWidth: 200 }}>
              {alert.title || 'Untitled alert'}
            </Text>
          </div>
        }
        description={
          <Text type="secondary" style={{ fontSize: 12 }}>
            {dayjs(alert.created_at).fromNow()} • {categoryLabel}
          </Text>
        }
      />
    </List.Item>
  )
}

export default function AlertsPanel() {
  const navigate = useNavigate()

  const { data: alerts = [], isLoading } = useQuery({
    queryKey: ['recentAlerts'],
    queryFn: () => alertsApi.getRecent(24),
    refetchInterval: 30000,
  })

  const { data: summary } = useQuery({
    queryKey: ['alertSummary'],
    queryFn: alertsApi.getSummary,
    refetchInterval: 30000,
  })

  const criticalCount = summary?.by_severity?.critical || 0

  return (
    <div className="glass-panel" style={{
      height: '100%',
      padding: 0,
      display: 'flex',
      flexDirection: 'column',
      border: '1px solid var(--neon-red)',
      boxShadow: '0 0 15px rgba(255, 51, 51, 0.15), inset 0 0 20px rgba(255, 51, 51, 0.05)',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Decorative corner accents */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: 20, height: 2, background: 'var(--neon-red)', boxShadow: '0 0 10px var(--neon-red)' }} />
      <div style={{ position: 'absolute', top: 0, left: 0, width: 2, height: 20, background: 'var(--neon-red)', boxShadow: '0 0 10px var(--neon-red)' }} />
      <div style={{ position: 'absolute', bottom: 0, right: 0, width: 20, height: 2, background: 'var(--neon-red)', boxShadow: '0 0 10px var(--neon-red)' }} />
      <div style={{ position: 'absolute', bottom: 0, right: 0, width: 2, height: 20, background: 'var(--neon-red)', boxShadow: '0 0 10px var(--neon-red)' }} />

      <div style={{ padding: '24px 24px 0 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12 }}>
          <ExclamationCircleOutlined style={{ fontSize: 48, color: 'var(--neon-red)' }} />
          <div>
            <h2 style={{
              margin: 0,
              color: 'var(--neon-red)',
              fontFamily: 'var(--font-display)',
              fontSize: 24,
              textTransform: 'uppercase',
              letterSpacing: 2,
              textShadow: '0 0 10px rgba(255, 51, 51, 0.4)'
            }}>
              Priority Alert
            </h2>
            <div style={{ color: '#fff', fontSize: 13, letterSpacing: 1, textTransform: 'uppercase' }}>
              Critical Issues Require Attention
            </div>
          </div>
        </div>

        <Text style={{ display: 'block', color: 'rgba(255,255,255,0.7)', marginBottom: 20, fontSize: 16 }}>
          {criticalCount} issues identified that may impact timelines or data quality.
        </Text>
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: '0 12px' }}>
        <List
          loading={isLoading}
          dataSource={alerts.slice(0, 3)} // Show only top 3
          split={false}
          renderItem={(alert) => (
            <div
              className="glass-panel-hover"
              onClick={() => navigate(`/alerts?id=${alert.alert_id}`)}
              style={{
                margin: '8px 12px',
                padding: '12px',
                borderRadius: 8,
                cursor: 'pointer',
                border: '1px solid rgba(255, 51, 51, 0.2)',
                background: 'rgba(20, 20, 35, 0.4)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <Text strong style={{ color: '#fff' }}>{alert.title || 'Untitled Alert'}</Text>
                <Tag color="red" style={{
                  margin: 0,
                  background: 'rgba(255, 51, 51, 0.2)',
                  border: '1px solid var(--neon-red)',
                  color: '#fff',
                  boxShadow: '0 0 5px rgba(255, 51, 51, 0.2)'
                }}>
                  CRITICAL
                </Tag>
              </div>
              <Text type="secondary" style={{ fontSize: 12 }}>{dayjs(alert.created_at).fromNow()}</Text>
            </div>
          )}
        />
      </div>

      <div style={{ padding: 16, borderTop: '1px solid rgba(255,51,51,0.2)', display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
        <div style={{
          padding: '8px 16px',
          borderRadius: 20,
          background: 'rgba(255, 51, 51, 0.1)',
          border: '1px solid var(--neon-red)',
          color: '#fff',
          fontSize: 12
        }}>
          {summary?.active_alerts || 0} Total Active Alerts
        </div>
      </div>
    </div>
  )
}
