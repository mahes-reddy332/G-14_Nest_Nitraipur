import { Card, Empty, Spin, Typography, Segmented, Badge, Tooltip } from 'antd'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { studiesApi } from '../../api'
import { FundProjectionScreenOutlined, RadarChartOutlined, BarChartOutlined } from '@ant-design/icons'

const { Text } = Typography

type ChartMode = 'bar' | 'radar'

export default function StudyDistributionChart() {
  const [chartMode, setChartMode] = useState<ChartMode>('bar')

  const { data: studies = [], isLoading } = useQuery({
    queryKey: ['allStudies'],
    queryFn: studiesApi.getAll,
    refetchInterval: 120000,
  })

  if (isLoading) {
    return <Card title="Study Performance Comparison" loading />
  }

  if (studies.length === 0) {
    return (
      <Card title="Study Performance Comparison">
        <Empty description="No studies available" />
      </Card>
    )
  }

  // Prepare chart data - take top 8 studies
  const chartData = studies.slice(0, 8).map((study) => ({
    name: study.name || study.study_id,
    shortName: (study.name || study.study_id).slice(0, 10),
    dqi: study.dqi_score || 0,
    patients: study.total_patients || 0,
    cleanRate: study.total_patients > 0 
      ? Math.round((study.clean_patients / study.total_patients) * 100) 
      : 0,
    sites: study.total_sites || 0,
  }))

  // For radar chart
  const radarData = chartData.slice(0, 6).map((study) => ({
    study: study.shortName,
    DQI: study.dqi,
    'Clean Rate': study.cleanRate,
    Patients: Math.min(study.patients, 100), // Normalize for radar
    Sites: study.sites * 10, // Scale for visibility
  }))

  const getBarColor = (value: number) => {
    if (value >= 80) return '#52c41a'
    if (value >= 60) return '#faad14'
    return '#ff4d4f'
  }

  return (
    <Card
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <FundProjectionScreenOutlined />
          <span>Study Performance Comparison</span>
          <Badge count={studies.length} style={{ backgroundColor: '#1890ff' }} />
        </div>
      }
      extra={
        <Segmented
          size="small"
          options={[
            { value: 'bar', icon: <BarChartOutlined /> },
            { value: 'radar', icon: <RadarChartOutlined /> },
          ]}
          value={chartMode}
          onChange={(val) => setChartMode(val as ChartMode)}
        />
      }
    >
      <div style={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          {chartMode === 'bar' ? (
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 0, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="shortName" 
                tick={{ fontSize: 10 }}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
              <RechartsTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div
                        style={{
                          backgroundColor: '#fff',
                          border: '1px solid #f0f0f0',
                          borderRadius: 8,
                          padding: 12,
                          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        }}
                      >
                        <Text strong style={{ display: 'block', marginBottom: 8 }}>
                          {data.name}
                        </Text>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                          <Text>DQI Score: <Text strong style={{ color: getBarColor(data.dqi) }}>{data.dqi}%</Text></Text>
                          <Text>Clean Rate: {data.cleanRate}%</Text>
                          <Text>Patients: {data.patients}</Text>
                          <Text>Sites: {data.sites}</Text>
                        </div>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Bar
                dataKey="dqi"
                name="DQI Score"
                radius={[4, 4, 0, 0]}
                animationDuration={800}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getBarColor(entry.dqi)} />
                ))}
              </Bar>
            </BarChart>
          ) : (
            <RadarChart data={radarData} outerRadius={90}>
              <PolarGrid stroke="#f0f0f0" />
              <PolarAngleAxis dataKey="study" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis tick={{ fontSize: 9 }} domain={[0, 100]} />
              <RechartsTooltip />
              <Radar
                name="DQI"
                dataKey="DQI"
                stroke="#1890ff"
                fill="#1890ff"
                fillOpacity={0.3}
              />
              <Radar
                name="Clean Rate"
                dataKey="Clean Rate"
                stroke="#52c41a"
                fill="#52c41a"
                fillOpacity={0.3}
              />
              <Legend />
            </RadarChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Summary Stats */}
      <div
        style={{
          marginTop: 16,
          padding: 12,
          backgroundColor: '#fafafa',
          borderRadius: 8,
          display: 'flex',
          justifyContent: 'space-around',
        }}
      >
        <Tooltip title="Average DQI across all studies">
          <div style={{ textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: 11 }}>Avg DQI</Text>
            <div>
              <Text strong style={{ fontSize: 16, color: '#1890ff' }}>
                {(chartData.reduce((sum, s) => sum + s.dqi, 0) / chartData.length).toFixed(1)}%
              </Text>
            </div>
          </div>
        </Tooltip>
        <Tooltip title="Total patients enrolled">
          <div style={{ textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: 11 }}>Total Patients</Text>
            <div>
              <Text strong style={{ fontSize: 16, color: '#722ed1' }}>
                {chartData.reduce((sum, s) => sum + s.patients, 0).toLocaleString()}
              </Text>
            </div>
          </div>
        </Tooltip>
        <Tooltip title="Average clean patient rate">
          <div style={{ textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: 11 }}>Avg Clean Rate</Text>
            <div>
              <Text strong style={{ fontSize: 16, color: '#52c41a' }}>
                {(chartData.reduce((sum, s) => sum + s.cleanRate, 0) / chartData.length).toFixed(1)}%
              </Text>
            </div>
          </div>
        </Tooltip>
      </div>
    </Card>
  )
}
