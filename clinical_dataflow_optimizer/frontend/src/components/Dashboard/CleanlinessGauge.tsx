import { Card, Progress, Row, Col, Tag, Tooltip } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { metricsApi } from '../../api'
import { useStore } from '../../store'

export default function CleanlinessGauge() {
  const { selectedStudyId } = useStore()

  const { data: metrics, isLoading } = useQuery({
    queryKey: ['cleanlinessMetrics', selectedStudyId],
    queryFn: () => metricsApi.getCleanliness(selectedStudyId || undefined),
    refetchInterval: 60000,
  })

  const getStatusColor = (rate: number) => {
    if (rate >= 80) return '#52c41a'
    if (rate >= 60) return '#faad14'
    return '#ff4d4f'
  }

  if (isLoading || !metrics) {
    return <Card title="Patient Cleanliness" loading />
  }

  return (
    <Card title="Patient Cleanliness Status" style={{ height: '100%' }}>
      <Row gutter={[24, 24]} align="middle">
        <Col span={12}>
          <div style={{ textAlign: 'center' }}>
            <Progress
              type="dashboard"
              percent={Math.round(metrics.cleanliness_rate)}
              strokeColor={getStatusColor(metrics.cleanliness_rate)}
              format={(percent) => (
                <div>
                  <div style={{ fontSize: 28, fontWeight: 'bold' }}>{percent}%</div>
                  <div style={{ fontSize: 12, color: '#8c8c8c' }}>Clean Rate</div>
                </div>
              )}
            />
          </div>
        </Col>
        <Col span={12}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <Tooltip title="Patients with no blocking factors">
              <div>
                <Tag color="success" style={{ marginRight: 8 }}>Clean</Tag>
                <span style={{ fontSize: 18, fontWeight: 'bold' }}>{metrics.clean_count}</span>
              </div>
            </Tooltip>
            <Tooltip title="Patients with unresolved issues">
              <div>
                <Tag color="error" style={{ marginRight: 8 }}>Dirty</Tag>
                <span style={{ fontSize: 18, fontWeight: 'bold' }}>{metrics.dirty_count}</span>
              </div>
            </Tooltip>
            <Tooltip title="Patients approaching dirty status">
              <div>
                <Tag color="warning" style={{ marginRight: 8 }}>At Risk</Tag>
                <span style={{ fontSize: 18, fontWeight: 'bold' }}>{metrics.at_risk_count}</span>
              </div>
            </Tooltip>
          </div>
        </Col>
      </Row>

      {/* Trend sparkline placeholder */}
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 12, color: '#8c8c8c', marginBottom: 8 }}>7-Day Trend</div>
        <div style={{ display: 'flex', gap: 4, alignItems: 'flex-end', height: 40 }}>
          {(metrics.trend || []).map((value, index) => (
            <div
              key={index}
              style={{
                flex: 1,
                height: `${value}%`,
                backgroundColor: getStatusColor(value),
                borderRadius: 2,
              }}
            />
          ))}
        </div>
      </div>
    </Card>
  )
}
