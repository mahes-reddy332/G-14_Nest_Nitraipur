import { Card, Row, Col, Progress, Typography, Tooltip, Badge } from 'antd'
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  Legend,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { metricsApi } from '../../api'
import { useStore } from '../../store'
import {
  CheckCircleFilled,
  ClockCircleFilled,
  SafetyCertificateFilled,
  FileTextFilled,
  BarChartOutlined,
} from '@ant-design/icons'

const { Text, Title } = Typography

const COLORS = ['#1890ff', '#52c41a', '#722ed1', '#fa8c16', '#13c2c2']

const dimensionIcons = {
  completeness: <FileTextFilled />,
  accuracy: <CheckCircleFilled />,
  consistency: <SafetyCertificateFilled />,
  timeliness: <ClockCircleFilled />,
}

export default function DQIBreakdownChart() {
  const { selectedStudyId } = useStore()

  const { data: dqi, isLoading } = useQuery({
    queryKey: ['dqiMetrics', selectedStudyId],
    queryFn: () => metricsApi.getDQI(selectedStudyId || undefined),
    refetchInterval: 60000,
  })

  if (isLoading || !dqi) {
    return <Card title="Data Quality Breakdown" loading />
  }

  // Prepare pie chart data
  const pieData = [
    { name: 'Completeness', value: dqi.completeness, color: '#1890ff' },
    { name: 'Accuracy', value: dqi.accuracy, color: '#52c41a' },
    { name: 'Consistency', value: dqi.consistency, color: '#722ed1' },
    { name: 'Timeliness', value: dqi.timeliness, color: '#fa8c16' },
  ]

  // Prepare trend data
  const trendData = (dqi.trend || []).map((value, index) => ({
    day: `Day ${index + 1}`,
    dqi: value,
  }))

  const getStatusColor = (value: number) => {
    if (value >= 85) return '#52c41a'
    if (value >= 70) return '#faad14'
    return '#ff4d4f'
  }

  const getStatusBadge = (value: number) => {
    if (value >= 85) return 'success'
    if (value >= 70) return 'warning'
    return 'error'
  }

  return (
    <Card 
      style={{ height: '100%' }}
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <BarChartOutlined />
          <span>Data Quality Index Breakdown</span>
          <Badge 
            status={getStatusBadge(dqi.overall_dqi)} 
            text={<Text strong style={{ color: getStatusColor(dqi.overall_dqi) }}>{dqi.overall_dqi.toFixed(1)}</Text>}
          />
        </div>
      }
    >
      <Row gutter={[24, 24]}>
        {/* Pie Chart */}
        <Col xs={24} md={12}>
          <div style={{ height: 250 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                  animationBegin={0}
                  animationDuration={800}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip 
                  formatter={(value: number) => [`${value.toFixed(1)}%`, '']}
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
                  formatter={(value) => <Text style={{ fontSize: 12 }}>{value}</Text>}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Col>

        {/* Dimension Progress Bars */}
        <Col xs={24} md={12}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {pieData.map(({ name, value, color }) => (
              <div key={name}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ color, display: 'inline-flex' }}>
                      {dimensionIcons[name.toLowerCase() as keyof typeof dimensionIcons]}
                    </span>
                    <Text style={{ lineHeight: 1.2 }}>{name}</Text>
                  </div>
                  <Text strong style={{ color: getStatusColor(value) }}>
                    {value.toFixed(1)}%
                  </Text>
                </div>
                <Progress 
                  percent={value} 
                  strokeColor={color}
                  showInfo={false}
                  size="small"
                />
              </div>
            ))}
          </div>
        </Col>
      </Row>

      {/* Trend Area Chart */}
      {trendData.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <Text type="secondary" style={{ marginBottom: 8, display: 'block' }}>
            DQI Trend (7 Days)
          </Text>
          <div style={{ height: 120 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="dqiGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#1890ff" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#1890ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                <RechartsTooltip 
                  formatter={(value: number) => [`${value.toFixed(1)}%`, 'DQI']}
                />
                <Area
                  type="monotone"
                  dataKey="dqi"
                  stroke="#1890ff"
                  strokeWidth={2}
                  fill="url(#dqiGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </Card>
  )
}
