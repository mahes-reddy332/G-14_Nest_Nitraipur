import { Card, Row, Col, Statistic, Typography } from 'antd'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { metricsApi } from '../../api'
import { useStore } from '../../store'

const { Text } = Typography

export default function QueryVelocityChart() {
  const { selectedStudyId } = useStore()

  const { data: velocity, isLoading } = useQuery({
    queryKey: ['velocity', selectedStudyId],
    queryFn: () => metricsApi.getVelocity(selectedStudyId || undefined, 7),
    refetchInterval: 60000,
  })

  if (isLoading || !velocity) {
    return <Card title="Query Velocity" loading />
  }

  const trendData = Array.isArray(velocity.trend) ? velocity.trend : []

  return (
    <Card title="Query & Resolution Velocity">
      <Row gutter={24} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Statistic
            title="Queries/Day"
            value={velocity.queries_per_day}
            valueStyle={{ color: '#1890ff' }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="Resolutions/Day"
            value={velocity.resolutions_per_day}
            valueStyle={{ color: '#52c41a' }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="Data Entries/Day"
            value={velocity.data_entries_per_day}
            valueStyle={{ color: '#722ed1' }}
          />
        </Col>
      </Row>

      <div style={{ height: 250 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={trendData}
            margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11 }}
              tickFormatter={(value) => {
                const date = new Date(value)
                return `${date.getMonth() + 1}/${date.getDate()}`
              }}
            />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #f0f0f0',
                borderRadius: 4,
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="queries"
              name="Queries"
              stroke="#1890ff"
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line
              type="monotone"
              dataKey="resolutions"
              name="Resolutions"
              stroke="#52c41a"
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
        7-day trend showing query creation and resolution activity
      </Text>
    </Card>
  )
}
