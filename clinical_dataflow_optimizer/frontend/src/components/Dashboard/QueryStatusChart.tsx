import { Card, Row, Col, Statistic, Typography, Tag, Progress, Tooltip, Space } from 'antd'
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  Legend,
} from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { metricsApi } from '../../api'
import { useStore } from '../../store'
import {
  QuestionCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'

const { Text, Title } = Typography

const COLORS = {
  open: '#ff4d4f',
  closed: '#52c41a',
  pending: '#faad14',
  overdue: '#722ed1',
}

const agingColors = {
  '0-7 days': '#52c41a',
  '8-14 days': '#faad14',
  '15-30 days': '#fa8c16',
  '30+ days': '#ff4d4f',
}

export default function QueryStatusChart() {
  const { selectedStudyId } = useStore()

  const { data: queryMetrics, isLoading } = useQuery({
    queryKey: ['queryMetrics', selectedStudyId],
    queryFn: () => metricsApi.getQueries(selectedStudyId || undefined),
    refetchInterval: 60000,
  })

  if (isLoading || !queryMetrics) {
    return <Card title="Query Management Status" loading />
  }

  // Calculate pending queries
  const pendingQueries = queryMetrics.total_queries - queryMetrics.open_queries - queryMetrics.closed_queries

  // Prepare donut chart data
  const statusData = [
    { name: 'Open', value: queryMetrics.open_queries, color: COLORS.open },
    { name: 'Closed', value: queryMetrics.closed_queries, color: COLORS.closed },
    { name: 'Pending', value: Math.max(0, pendingQueries), color: COLORS.pending },
  ].filter(d => d.value > 0)

  // Prepare aging distribution data
  const agingData = Object.entries(queryMetrics.aging_distribution || {}).map(([key, value]) => ({
    name: key.replace('_', ' ').replace('days', ' days'),
    value: value as number,
    color: agingColors[key.replace('_', ' ').replace('days', ' days') as keyof typeof agingColors] || '#8c8c8c',
  }))

  const totalQueries = queryMetrics.total_queries || 0
  const resolutionRate = queryMetrics.resolution_rate || 0

  return (
    <Card
      style={{ height: '100%' }}
      title={
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            flexWrap: 'wrap',
            rowGap: 4,
          }}
        >
          <QuestionCircleOutlined />
          <span>Query Management Status</span>
          <Tag color={resolutionRate >= 80 ? 'success' : resolutionRate >= 60 ? 'warning' : 'error'}>
            {resolutionRate.toFixed(0)}% Resolved
          </Tag>
        </div>
      }
    >
      <Row gutter={[24, 24]}>
        {/* Donut Chart */}
        <Col xs={24} md={12}>
          <div style={{ height: 220, position: 'relative' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                  animationBegin={0}
                  animationDuration={800}
                >
                  {statusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip
                  formatter={(value: number, name: string) => [
                    `${value} (${((value / totalQueries) * 100).toFixed(1)}%)`,
                    name,
                  ]}
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #f0f0f0',
                    borderRadius: 8,
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  }}
                />
                <Legend
                  verticalAlign="bottom"
                  height={36}
                  formatter={(value, entry) => (
                    <Text style={{ fontSize: 11 }}>{value}</Text>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
            {/* Center text */}
            <div
              style={{
                position: 'absolute',
                top: '45%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center',
              }}
            >
              <Text type="secondary" style={{ fontSize: 11 }}>Total</Text>
              <div>
                <Text strong style={{ fontSize: 24 }}>{totalQueries}</Text>
              </div>
            </div>
          </div>
        </Col>

        {/* Stats and Aging */}
        <Col xs={24} md={12}>
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            {/* Key Stats */}
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic
                  title={<Text type="secondary" style={{ fontSize: 11 }}>Open</Text>}
                  value={queryMetrics.open_queries}
                  valueStyle={{ color: COLORS.open, fontSize: 20 }}
                  prefix={<ExclamationCircleOutlined />}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title={<Text type="secondary" style={{ fontSize: 11 }}>Closed</Text>}
                  value={queryMetrics.closed_queries}
                  valueStyle={{ color: COLORS.closed, fontSize: 20 }}
                  prefix={<CheckCircleOutlined />}
                />
              </Col>
            </Row>

            {/* Avg Resolution Time */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <Text type="secondary" style={{ fontSize: 11 }}>Avg Resolution Time</Text>
                <Text strong>{queryMetrics.avg_resolution_time?.toFixed(1) || 0} days</Text>
              </div>
              <Progress
                percent={Math.min(100, (queryMetrics.avg_resolution_time || 0) * 10)}
                strokeColor={queryMetrics.avg_resolution_time <= 5 ? '#52c41a' : queryMetrics.avg_resolution_time <= 10 ? '#faad14' : '#ff4d4f'}
                showInfo={false}
                size="small"
              />
            </div>

            {/* Query Aging Distribution */}
            <div>
              <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 8 }}>
                <ClockCircleOutlined style={{ marginRight: 4 }} />
                Open Query Aging
              </Text>
              {agingData.length > 0 ? (
                agingData.map(({ name, value, color }) => (
                  <div key={name} style={{ marginBottom: 6 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                      <Text style={{ fontSize: 11 }}>{name}</Text>
                      <Text strong style={{ fontSize: 11 }}>{value}</Text>
                    </div>
                    <Progress
                      percent={queryMetrics.open_queries > 0 ? (value / queryMetrics.open_queries) * 100 : 0}
                      strokeColor={color}
                      showInfo={false}
                      size="small"
                    />
                  </div>
                ))
              ) : (
                <Text type="secondary" style={{ fontSize: 11 }}>No aging data available</Text>
              )}
            </div>
          </Space>
        </Col>
      </Row>
    </Card>
  )
}
