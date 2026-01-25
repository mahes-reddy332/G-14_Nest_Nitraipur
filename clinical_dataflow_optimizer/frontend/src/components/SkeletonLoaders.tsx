import { Card, Skeleton, Row, Col, Space } from 'antd'

/**
 * Skeleton loader for dashboard KPI cards
 */
export function KPICardsSkeleton() {
  return (
    <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
      {[1, 2, 3, 4].map((i) => (
        <Col xs={24} sm={12} lg={6} key={i}>
          <Card>
            <Skeleton active paragraph={{ rows: 2 }} title={{ width: '60%' }} />
          </Card>
        </Col>
      ))}
    </Row>
  )
}

/**
 * Skeleton loader for dashboard charts
 */
export function ChartSkeleton({ height = 300 }: { height?: number }) {
  return (
    <Card>
      <Skeleton.Input active style={{ width: '30%', marginBottom: 16 }} />
      <div 
        style={{ 
          height, 
          background: 'linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%)',
          backgroundSize: '200% 100%',
          animation: 'shimmer 1.5s infinite',
          borderRadius: 4
        }} 
      />
      <style>{`
        @keyframes shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </Card>
  )
}

/**
 * Skeleton loader for data tables
 */
export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <Card>
      <Skeleton.Input active style={{ width: '40%', marginBottom: 16 }} />
      <Space direction="vertical" style={{ width: '100%' }} size={8}>
        {/* Header row */}
        <Row gutter={16} style={{ marginBottom: 8 }}>
          {[1, 2, 3, 4, 5].map((i) => (
            <Col span={4} key={i}>
              <Skeleton.Input active size="small" style={{ width: '100%' }} />
            </Col>
          ))}
        </Row>
        {/* Data rows */}
        {Array.from({ length: rows }).map((_, i) => (
          <Row gutter={16} key={i}>
            {[1, 2, 3, 4, 5].map((j) => (
              <Col span={4} key={j}>
                <Skeleton.Input active size="small" style={{ width: '100%' }} />
              </Col>
            ))}
          </Row>
        ))}
      </Space>
    </Card>
  )
}

/**
 * Skeleton loader for study details page
 */
export function StudyDetailSkeleton() {
  return (
    <Space direction="vertical" style={{ width: '100%' }} size={24}>
      {/* Breadcrumb */}
      <Skeleton.Input active style={{ width: 200 }} />
      
      {/* Title area */}
      <div>
        <Skeleton.Input active style={{ width: 300, marginBottom: 8 }} />
        <Skeleton.Input active size="small" style={{ width: 150 }} />
      </div>
      
      {/* Stats cards */}
      <KPICardsSkeleton />
      
      {/* Tabs content */}
      <Card>
        <Skeleton.Input active style={{ width: '100%', marginBottom: 16 }} />
        <Skeleton active paragraph={{ rows: 6 }} />
      </Card>
    </Space>
  )
}

/**
 * Skeleton loader for patient list
 */
export function PatientListSkeleton() {
  return (
    <Space direction="vertical" style={{ width: '100%' }} size={16}>
      {/* Filters */}
      <Card>
        <Row gutter={16}>
          {[1, 2, 3, 4].map((i) => (
            <Col span={6} key={i}>
              <Skeleton.Input active style={{ width: '100%' }} />
            </Col>
          ))}
        </Row>
      </Card>
      
      {/* Table */}
      <TableSkeleton rows={10} />
    </Space>
  )
}

/**
 * Skeleton loader for alerts panel
 */
export function AlertsPanelSkeleton() {
  return (
    <Card>
      <Skeleton.Input active style={{ width: '30%', marginBottom: 16 }} />
      <Space direction="vertical" style={{ width: '100%' }} size={12}>
        {[1, 2, 3].map((i) => (
          <Card key={i} size="small" style={{ background: '#fafafa' }}>
            <Skeleton active paragraph={{ rows: 1 }} title={{ width: '50%' }} />
          </Card>
        ))}
      </Space>
    </Card>
  )
}

/**
 * Skeleton loader for agent insights
 */
export function AgentInsightsSkeleton() {
  return (
    <Card>
      <Space direction="vertical" style={{ width: '100%' }} size={16}>
        {[1, 2, 3].map((i) => (
          <div key={i} style={{ display: 'flex', gap: 16 }}>
            <Skeleton.Avatar active size={48} />
            <div style={{ flex: 1 }}>
              <Skeleton active paragraph={{ rows: 2 }} title={{ width: '40%' }} />
            </div>
          </div>
        ))}
      </Space>
    </Card>
  )
}

/**
 * Full dashboard skeleton
 */
export function DashboardSkeleton() {
  return (
    <Space direction="vertical" style={{ width: '100%' }} size={24}>
      {/* Header */}
      <div>
        <Skeleton.Input active style={{ width: 250, marginBottom: 8 }} />
        <Skeleton.Input active size="small" style={{ width: 400 }} />
      </div>
      
      {/* Filters */}
      <Card size="small">
        <Row gutter={16} align="middle">
          <Col flex="auto">
            <Row gutter={8}>
              {[1, 2, 3].map((i) => (
                <Col key={i}>
                  <Skeleton.Input active style={{ width: 120 }} />
                </Col>
              ))}
            </Row>
          </Col>
          <Col>
            <Skeleton.Button active />
          </Col>
        </Row>
      </Card>
      
      {/* KPI Cards */}
      <KPICardsSkeleton />
      
      {/* Charts row */}
      <Row gutter={[24, 24]}>
        <Col span={8}>
          <ChartSkeleton height={250} />
        </Col>
        <Col span={8}>
          <ChartSkeleton height={250} />
        </Col>
        <Col span={8}>
          <ChartSkeleton height={250} />
        </Col>
      </Row>
      
      {/* Bottom section */}
      <Row gutter={[24, 24]}>
        <Col span={12}>
          <ChartSkeleton height={300} />
        </Col>
        <Col span={12}>
          <AlertsPanelSkeleton />
        </Col>
      </Row>
    </Space>
  )
}

/**
 * Generic page loading skeleton
 */
export function PageSkeleton() {
  return (
    <Space direction="vertical" style={{ width: '100%' }} size={24}>
      {/* Breadcrumb */}
      <Skeleton.Input active style={{ width: 200 }} />
      
      {/* Title */}
      <div>
        <Skeleton.Input active style={{ width: 200, marginBottom: 8 }} />
        <Skeleton.Input active size="small" style={{ width: 300 }} />
      </div>
      
      {/* Content */}
      <Card>
        <Skeleton active paragraph={{ rows: 8 }} />
      </Card>
    </Space>
  )
}
