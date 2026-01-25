import { Row, Col, Typography, Space, Breadcrumb, Tabs, Badge, Card } from 'antd'
import { HomeOutlined, DashboardOutlined, PieChartOutlined, LineChartOutlined, AlertOutlined } from '@ant-design/icons'
import { useQueryClient } from '@tanstack/react-query'
import {
  // Original Components
  CleanlinessGauge,
  AlertsPanel,
  AgentInsightsPanel,
  DataHeatmap,
  QueryVelocityChart,
  StudyDistributionChart,
  // Enhanced Components
  AtRiskSummary,
  EnhancedKPISections,
  EnhancedQueryStatus,
  EnhancedDQIBreakdown,
  GlobalFilters,
} from '../components/Dashboard'
import { useStore } from '../store'
import '../styles/clinical-design-system.css'
import { useState } from 'react'

const { Title, Text } = Typography

export default function Dashboard() {
  const { selectedStudyId, isConnected } = useStore()
  const queryClient = useQueryClient()
  const [lastUpdated, setLastUpdated] = useState<string>(new Date().toISOString())

  const handleRefresh = () => {
    // Invalidate all dashboard queries
    queryClient.invalidateQueries({ queryKey: ['dashboardSummary'] })
    queryClient.invalidateQueries({ queryKey: ['queryMetrics'] })
    queryClient.invalidateQueries({ queryKey: ['cleanlinessMetrics'] })
    queryClient.invalidateQueries({ queryKey: ['kpiTiles'] })
    queryClient.invalidateQueries({ queryKey: ['dqiBreakdown'] })
    queryClient.invalidateQueries({ queryKey: ['alertSummary'] })
    setLastUpdated(new Date().toISOString())
  }

  return (
    <div className="clinical-dashboard">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: <><DashboardOutlined /> Dashboard</> },
        ]}
        style={{ marginBottom: 16 }}
      />

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0, color: 'var(--gray-800)' }}>
            Clinical Data Dashboard
          </Title>
          <Text type="secondary">
            Real-time overview of clinical trial data quality and operational metrics
            {selectedStudyId && ` • Filtered by ${selectedStudyId}`}
          </Text>
        </Space>
      </div>

      {/* Global Filters - Sticky */}
      <GlobalFilters onRefresh={handleRefresh} lastUpdated={lastUpdated} />

      {/* At-Risk Summary Banner - Critical items first */}
      <AtRiskSummary />

      {/* Enhanced KPI Sections - Organized by category */}
      <EnhancedKPISections />

      {/* Main Grid - Row 1: DQI Breakdown + Query Status + Cleanliness */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={8}>
          <EnhancedDQIBreakdown />
        </Col>
        <Col xs={24} lg={8}>
          <EnhancedQueryStatus />
        </Col>
        <Col xs={24} lg={8}>
          <CleanlinessGauge />
        </Col>
      </Row>

      {/* Main Grid - Row 2: Study Distribution + Query Velocity */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <StudyDistributionChart />
        </Col>
        <Col xs={24} lg={12}>
          <QueryVelocityChart />
        </Col>
      </Row>

      {/* Main Grid - Row 3: Heatmap + Alerts/Insights */}
      <Row gutter={[24, 24]}>
        <Col xs={24} lg={16}>
          <DataHeatmap metric="dqi" title="Site Data Quality Heatmap" />
        </Col>
        <Col xs={24} lg={8}>
          <Space direction="vertical" size={24} style={{ width: '100%' }}>
            <AlertsPanel />
            <AgentInsightsPanel />
          </Space>
        </Col>
      </Row>
    </div>
  )
}
