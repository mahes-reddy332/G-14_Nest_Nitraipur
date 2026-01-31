import { Progress, Row, Col, Tag, Tooltip, Skeleton, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { SafetyOutlined, InfoCircleOutlined } from '@ant-design/icons'
import { metricsApi } from '../../api'
import { useStore } from '../../store'
import { getDaysFromRange } from '../../utils/filtering'
import '../../styles/clinical-design-system.css'

export default function CleanlinessGauge() {
  const { selectedStudyId, selectedSiteId, dateRange } = useStore()
  const days = getDaysFromRange(dateRange, 30)

  const { data: metrics, isLoading, isError } = useQuery({
    queryKey: ['cleanlinessMetrics', selectedStudyId, selectedSiteId, days],
    queryFn: () => metricsApi.getCleanliness(selectedStudyId || undefined, selectedSiteId || undefined, days),
    refetchInterval: 60000,
  })

  const getStatusColor = (rate: number) => {
    if (rate >= 80) return '#52c41a'
    if (rate >= 60) return '#faad14'
    return '#ff4d4f'
  }

  if (isLoading) {
    return (
      <div className="clinical-card">
        <div className="clinical-card__header">
          <span className="clinical-card__title">
            <SafetyOutlined className="clinical-card__title-icon" />
            Patient Cleanliness
          </span>
        </div>
        <div className="clinical-card__body">
          <Skeleton active paragraph={{ rows: 5 }} />
        </div>
      </div>
    )
  }

  if (isError || !metrics) {
    return (
      <div className="clinical-card">
        <div className="clinical-card__header">
          <span className="clinical-card__title">
            <SafetyOutlined className="clinical-card__title-icon" />
            Patient Cleanliness
          </span>
        </div>
        <div className="clinical-card__body">
          <Empty description="Cleanliness metrics unavailable" />
        </div>
      </div>
    )
  }

  return (
    <div className="glass-panel" style={{ height: '100%', padding: 24 }}>
      <div className="clinical-card__header">
        <span className="clinical-card__title" style={{ color: 'var(--neon-green)' }}>
          <SafetyOutlined className="clinical-card__title-icon" style={{ color: 'var(--neon-green)' }} />
          Patient Cleanliness
        </span>
        <Tooltip title="Share of patients with no blocking data issues">
          <InfoCircleOutlined style={{ color: 'rgba(255, 255, 255, 0.3)' }} />
        </Tooltip>
      </div>
      <div className="clinical-card__body">
        <Row gutter={[24, 24]} align="middle">
          <Col span={12}>
            <div style={{ textAlign: 'center', position: 'relative' }}>
              <div style={{
                position: 'absolute',
                top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                width: 120, height: 120, borderRadius: '50%',
                background: 'var(--neon-green)', opacity: 0.1, filter: 'blur(20px)'
              }} />
              <Progress
                type="dashboard"
                percent={Math.round(metrics.cleanliness_rate)}
                strokeColor="var(--neon-green)"
                trailColor="rgba(255, 255, 255, 0.1)"
                width={140}
                format={(percent) => (
                  <div>
                    <div style={{ fontSize: 32, fontWeight: 'bold', color: '#fff', textShadow: '0 0 10px var(--neon-green)' }}>{percent}%</div>
                    <div style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.5)' }}>Clean Rate</div>
                  </div>
                )}
              />
            </div>
          </Col>
          <Col span={12}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <Tooltip title="Patients with no blocking factors">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: 8 }}>
                  <span style={{ color: 'var(--neon-green)' }}>Clean</span>
                  <span style={{ fontSize: 18, fontWeight: 'bold', color: '#fff' }}>{metrics.clean_count}</span>
                </div>
              </Tooltip>
              <Tooltip title="Patients with unresolved issues">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: 8 }}>
                  <span style={{ color: 'var(--neon-red)' }}>Dirty</span>
                  <span style={{ fontSize: 18, fontWeight: 'bold', color: '#fff' }}>{metrics.dirty_count}</span>
                </div>
              </Tooltip>
              <Tooltip title="Patients approaching dirty status">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: 'var(--neon-yellow)' }}>At Risk</span>
                  <span style={{ fontSize: 18, fontWeight: 'bold', color: '#fff' }}>{metrics.at_risk_count}</span>
                </div>
              </Tooltip>
            </div>
          </Col>
        </Row>

        {/* Trend sparkline */}
        <div style={{ marginTop: 24 }}>
          <div style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.5)', marginBottom: 8, fontFamily: 'var(--font-mono)' }}>7-DAY TREND</div>
          <div style={{ display: 'flex', gap: 4, alignItems: 'flex-end', height: 40 }}>
            {(metrics.trend || []).length === 0 && (
              <div style={{ fontSize: 12, color: 'var(--gray-400)' }}>No trend data</div>
            )}
            {(metrics.trend || []).map((value, index) => (
              <div
                key={index}
                style={{
                  flex: 1,
                  height: `${value}%`,
                  backgroundColor: getStatusColor(value),
                  borderRadius: 2,
                  boxShadow: `0 0 5px ${getStatusColor(value)}`,
                  opacity: 0.8
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
