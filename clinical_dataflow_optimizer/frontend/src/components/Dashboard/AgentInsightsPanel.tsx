import { Card, List, Tag, Typography, Avatar, Progress } from 'antd'
import { RobotOutlined, BulbOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { agentsApi } from '../../api'
import type { AgentInsight } from '../../types'

const { Text } = Typography

const priorityColors = {
  critical: 'red',
  high: 'orange',
  medium: 'gold',
  low: 'green',
}

const agentAvatarColors: Record<string, string> = {
  reconciliation: '#1890ff',
  coding: '#722ed1',
  data_quality: '#13c2c2',
  predictive: '#eb2f96',
  site_liaison: '#52c41a',
  supervisor: '#fa8c16',
}

interface InsightItemProps {
  insight: AgentInsight
  onClick: () => void
}

function InsightItem({ insight, onClick }: InsightItemProps) {
  const confidence = Number.isFinite(insight.confidence) ? insight.confidence : 0
  const priority = insight.priority || 'low'
  return (
    <List.Item onClick={onClick} style={{ cursor: 'pointer' }}>
      <List.Item.Meta
        avatar={
          <Avatar
            style={{
              backgroundColor: agentAvatarColors[insight.agent] || '#8c8c8c',
            }}
            icon={<RobotOutlined />}
          />
        }
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Tag color={priorityColors[priority] || 'green'}>{priority}</Tag>
            <Text strong ellipsis style={{ maxWidth: 250 }}>
              {insight.title || 'Untitled insight'}
            </Text>
          </div>
        }
        description={
          <div>
            <Text type="secondary" ellipsis style={{ display: 'block', fontSize: 12 }}>
              {insight.description || 'No description available'}
            </Text>
            <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Progress
                percent={Math.round(confidence * 100)}
                size="small"
                style={{ width: 60, margin: 0 }}
                showInfo={false}
              />
              <Text type="secondary" style={{ fontSize: 11 }}>
                {Math.round(confidence * 100)}% confidence
              </Text>
            </div>
          </div>
        }
      />
    </List.Item>
  )
}

export default function AgentInsightsPanel() {
  const navigate = useNavigate()

  const { data: insights = [], isLoading } = useQuery({
    queryKey: ['agentInsights'],
    queryFn: () => agentsApi.getInsights({ limit: 5 }),
    refetchInterval: 60000,
  })

  const { data: agentStatus } = useQuery({
    queryKey: ['agentStatus'],
    queryFn: agentsApi.getStatus,
    refetchInterval: 30000,
  })

  const activeAgents = agentStatus
    ? Object.values(agentStatus).filter((a) => a.status === 'active').length
    : 0
  const totalAgents = agentStatus ? Object.keys(agentStatus).length : 0

  return (
    <Card
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <BulbOutlined style={{ color: '#faad14' }} />
          <span>AI Agent Insights</span>
          {activeAgents > 0 && (
            <Tag color="blue">
              {activeAgents}/{totalAgents} Active
            </Tag>
          )}
        </div>
      }
      extra={<a onClick={() => navigate('/agents')}>View All</a>}
    >
      <List
        loading={isLoading}
        dataSource={insights}
        renderItem={(insight) => (
          <InsightItem
            insight={insight}
            onClick={() => navigate(`/agents?insight=${insight.insight_id}`)}
          />
        )}
        locale={{ emptyText: 'No insights available' }}
      />
    </Card>
  )
}
