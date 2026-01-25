import { Card, List, Tag, Typography, Progress, Tooltip } from 'antd'
import {
  AlertOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
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
  const totalActive = summary?.active_alerts || 0

  return (
    <Card
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span>Alerts</span>
          {criticalCount > 0 && (
            <Tag color="red">{criticalCount} Critical</Tag>
          )}
        </div>
      }
      extra={
        <a onClick={() => navigate('/alerts')}>View All</a>
      }
    >
      {/* Alert severity distribution */}
      {summary && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
            <Tooltip title={`Critical: ${summary.by_severity?.critical || 0}`}>
              <Progress
                percent={
                  totalActive > 0
                    ? ((summary.by_severity?.critical || 0) / totalActive) * 100
                    : 0
                }
                strokeColor="#ff4d4f"
                showInfo={false}
                size="small"
                style={{ flex: 1 }}
              />
            </Tooltip>
            <Tooltip title={`High: ${summary.by_severity?.high || 0}`}>
              <Progress
                percent={
                  totalActive > 0
                    ? ((summary.by_severity?.high || 0) / totalActive) * 100
                    : 0
                }
                strokeColor="#fa8c16"
                showInfo={false}
                size="small"
                style={{ flex: 1 }}
              />
            </Tooltip>
            <Tooltip title={`Medium: ${summary.by_severity?.medium || 0}`}>
              <Progress
                percent={
                  totalActive > 0
                    ? ((summary.by_severity?.medium || 0) / totalActive) * 100
                    : 0
                }
                strokeColor="#faad14"
                showInfo={false}
                size="small"
                style={{ flex: 1 }}
              />
            </Tooltip>
            <Tooltip title={`Low: ${summary.by_severity?.low || 0}`}>
              <Progress
                percent={
                  totalActive > 0
                    ? ((summary.by_severity?.low || 0) / totalActive) * 100
                    : 0
                }
                strokeColor="#52c41a"
                showInfo={false}
                size="small"
                style={{ flex: 1 }}
              />
            </Tooltip>
          </div>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {totalActive} active alerts
          </Text>
        </div>
      )}

      {/* Alert list */}
      <List
        loading={isLoading}
        dataSource={alerts.slice(0, 5)}
        renderItem={(alert) => (
          <AlertItem
            alert={alert}
            onClick={() => navigate(`/alerts?id=${alert.alert_id}`)}
          />
        )}
        locale={{ emptyText: 'No recent alerts' }}
      />
    </Card>
  )
}
