import { Card, Row, Col, Statistic, Typography, Segmented, Empty } from 'antd'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
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
import { useState } from 'react'
import { AreaChartOutlined, BarChartOutlined, LineChartOutlined } from '@ant-design/icons'
import { getDaysFromRange } from '../../utils/filtering'

const { Text } = Typography

export default function QueryVelocityChart({ embedded = false }: { embedded?: boolean }) {
  const { selectedStudyId, selectedSiteId, dateRange } = useStore()
  const [mode, setMode] = useState<'line' | 'area' | 'bar'>('line')
  const days = getDaysFromRange(dateRange, 7)

  const { data: velocity, isLoading } = useQuery({
    queryKey: ['velocity', selectedStudyId, selectedSiteId, days],
    queryFn: () => metricsApi.getVelocity(selectedStudyId || undefined, days, selectedSiteId || undefined),
    refetchInterval: 60000,
  })

  // Loading State
  if (isLoading) {
    if (embedded) return <div style={{ padding: 20, textAlign: 'center' }}>Loading...</div>
    return <Card title="Query Velocity" loading />
  }

  // Empty State
  if (!velocity || Object.keys(velocity).length === 0) {
    const emptyContent = <div style={{ padding: 20, textAlign: 'center', color: 'rgba(255,255,255,0.5)' }}>No Data Available</div>
    if (embedded) return emptyContent
    return <Card title="Query Velocity">{emptyContent}</Card>
  }
  
  const trendData = Array.isArray(velocity.trend) ? velocity.trend : []

  // Internal Content Renderer
  const renderContent = () => (
    <>
      <Row gutter={24} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Statistic
            title={<span style={{ fontSize: 12, color: 'rgba(255,255,255,0.5)' }}>Queries/Day</span>}
            value={velocity.queries_per_day}
            valueStyle={{ color: '#1890ff', fontSize: 20, fontWeight: 600 }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title={<span style={{ fontSize: 12, color: 'rgba(255,255,255,0.5)' }}>Resolutions/Day</span>}
            value={velocity.resolutions_per_day}
            valueStyle={{ color: '#52c41a', fontSize: 20, fontWeight: 600 }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title={<span style={{ fontSize: 12, color: 'rgba(255,255,255,0.5)' }}>Data Entries/Day</span>}
            value={velocity.data_entries_per_day}
            valueStyle={{ color: '#722ed1', fontSize: 20, fontWeight: 600 }}
          />
        </Col>
      </Row>

      <div style={{ height: embedded ? 200 : 250 }}>
        {trendData.length === 0 ? (
          <Empty description={<span style={{ color: 'rgba(255,255,255,0.5)' }}>No trend data</span>} image={Empty.PRESENTED_IMAGE_SIMPLE} />
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            {mode === 'line' ? (
              <LineChart
                data={trendData}
                margin={{ top: 5, right: 10, left: -20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.5)' }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => {
                    const date = new Date(value)
                    return `${date.getMonth() + 1}/${date.getDate()}`
                  }}
                />
                <YAxis 
                  tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.5)' }} 
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(20, 20, 35, 0.9)',
                    border: '1px solid rgba(0, 243, 255, 0.2)',
                    borderRadius: 8,
                    color: '#fff',
                    backdropFilter: 'blur(4px)'
                  }}
                  itemStyle={{ color: '#fff' }}
                  labelStyle={{ color: 'rgba(255,255,255,0.7)', marginBottom: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="queries"
                  stroke="#1890ff"
                  strokeWidth={2}
                  dot={{ r: 2, fill: '#1890ff', strokeWidth: 0 }}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                  isAnimationActive={true}
                />
                <Line
                  type="monotone"
                  dataKey="resolutions"
                  stroke="#52c41a"
                  strokeWidth={2}
                  dot={{ r: 2, fill: '#52c41a', strokeWidth: 0 }}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                  isAnimationActive={true}
                />
              </LineChart>
            ) : mode === 'area' ? (
              <AreaChart data={trendData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.5)' }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => {
                    const date = new Date(value)
                    return `${date.getMonth() + 1}/${date.getDate()}`
                  }}
                />
                <YAxis 
                  tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.5)' }} 
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(20, 20, 35, 0.9)',
                    border: '1px solid rgba(0, 243, 255, 0.2)',
                    borderRadius: 8,
                    color: '#fff',
                    backdropFilter: 'blur(4px)'
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="queries"
                  stroke="#1890ff"
                  fill="#1890ff"
                  fillOpacity={0.1}
                />
                <Area
                  type="monotone"
                  dataKey="resolutions"
                  stroke="#52c41a"
                  fill="#52c41a"
                  fillOpacity={0.1}
                />
              </AreaChart>
            ) : (
              <BarChart data={trendData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.5)' }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => {
                    const date = new Date(value)
                    return `${date.getMonth() + 1}/${date.getDate()}`
                  }}
                />
                <YAxis 
                  tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.5)' }} 
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(20, 20, 35, 0.9)',
                    border: '1px solid rgba(0, 243, 255, 0.2)',
                    borderRadius: 8,
                    color: '#fff',
                    backdropFilter: 'blur(4px)'
                  }}
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                />
                <Bar dataKey="queries" fill="#1890ff" radius={[2, 2, 0, 0]} />
                <Bar dataKey="resolutions" fill="#52c41a" radius={[2, 2, 0, 0]} />
              </BarChart>
            )}
          </ResponsiveContainer>
        )}
      </div>

      {!embedded && (
        <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
          {days}-day trend showing query creation and resolution activity
        </Text>
      )}
    </>
  )

  if (embedded) {
    return <div style={{ width: '100%', height: '100%' }}>{renderContent()}</div>
  }

  return (
    <Card
      title="Query & Resolution Velocity"
      extra={
        <Segmented
          size="small"
          options={[
            { value: 'line', icon: <LineChartOutlined /> },
            { value: 'area', icon: <AreaChartOutlined /> },
            { value: 'bar', icon: <BarChartOutlined /> },
          ]}
          value={mode}
          onChange={(val) => setMode(val as 'line' | 'area' | 'bar')}
        />
      }
    >
      {renderContent()}
    </Card>
  )
}
