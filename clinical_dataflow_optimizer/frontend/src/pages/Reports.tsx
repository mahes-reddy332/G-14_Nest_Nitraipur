import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Card,
  Table,
  Typography,
  Space,
  Tag,
  Spin,
  Empty,
  Row,
  Col,
  Statistic,
  Button,
  Tabs,
  Badge,
  Descriptions
} from 'antd'
import {
  FileTextOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  ExperimentOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { reportsApi } from '../api'

// --- Types (Inline for safety, ideally moved to types.ts) ---
interface ReportDetail {
  id: string
  title: string
  date: string
  status: string
  content?: string | object
}

interface StudyReportDetail {
  study_id: string
  study_name: string
  report_date: string
  overview: {
    status: string
    phase: string
    therapeutic_area: string
    start_date: string
    target_enrollment: number
    current_enrollment: number
    total_sites: number
    active_sites: number
    last_updated: string
  }
  data_quality: {
    completeness_rate: number
    visit_completion_rate: number
    form_completion_rate: number
    freshness_days: number
    anomaly_indicators: Array<{ label: string; value: number }>
  }
  insights?: Array<{
    insight_id: string
    generated_at: string
    severity: 'info' | 'warning' | 'critical'
    category: string
    title: string
    what_happened: string
    why_it_matters: string
    evidence: any
  }>
  trends?: {
    dqi: number[]
    cleanliness: number[]
  }
  risks_and_alerts?: any[]
  sites_summary?: Array<{
    site_id: string
    site_name: string
    country: string
    total_patients: number
    dqi_score: number
    open_queries: number
  }>
  source_files?: Array<{
    file_type: string
    display_name: string
    status: string
    record_count: number
    loaded_at: string
  }>
}

// --- Mock Data ---
const MOCK_STUDY_DETAIL: StudyReportDetail = {
  study_id: 'ST-001',
  study_name: 'Cardio-001: Phase III Clinical Trial',
  report_date: new Date().toISOString(),
  overview: {
    status: 'Active',
    phase: 'Phase III',
    therapeutic_area: 'Cardiology',
    start_date: '2023-01-15',
    target_enrollment: 1500,
    current_enrollment: 1250,
    total_sites: 45,
    active_sites: 42,
    last_updated: new Date().toISOString()
  },
  data_quality: {
    completeness_rate: 98.5,
    visit_completion_rate: 97.2,
    form_completion_rate: 99.1,
    freshness_days: 1,
    anomaly_indicators: [{ label: 'Missing Visits', value: 12 }, { label: 'Missing Pages', value: 5 }]
  },
  insights: [
    { insight_id: 'INS-001', generated_at: new Date().toISOString(), severity: 'info', category: 'Enrollment', title: 'Enrollment On Track', what_happened: 'Enrollment met monthly target.', why_it_matters: 'Study timeline preserved.', evidence: {} }
  ],
  trends: {
    dqi: [95, 96, 96, 97, 98, 98.5],
    cleanliness: [94, 95, 95, 96, 96, 96.2]
  },
  risks_and_alerts: [],
  sites_summary: [
    { site_id: 'S-001', site_name: 'Site 001', country: 'USA', total_patients: 150, dqi_score: 99.1, open_queries: 2 },
    { site_id: 'S-002', site_name: 'Site 002', country: 'USA', total_patients: 120, dqi_score: 95.5, open_queries: 5 }
  ],
  source_files: [
    { file_type: 'custom', display_name: 'EDC Data', status: 'loaded', record_count: 12500, loaded_at: new Date().toISOString() }
  ]
}

const { Title, Text } = Typography

// --- Main Component ---
export default function Reports() {
  // Local state for active tab
  const [activeTab, setActiveTab] = useState('overview')
  // We'll hardcode one study ID for now or grab it from URL if we were using params
  const selectedStudyId = 'ST-001'

  // Query for data
  const { data: studyReport, isLoading } = useQuery<StudyReportDetail>({
    queryKey: ['studyReport', selectedStudyId],
    queryFn: async () => {
      try {
        // Just return mock for now to guarantee stability, 
        // normally we'd await reportsApi.getStudyReport(selectedStudyId)
        // const result = await reportsApi.getStudyReport(selectedStudyId)
        // return result || MOCK_STUDY_DETAIL
        return MOCK_STUDY_DETAIL
      } catch (err) {
        return MOCK_STUDY_DETAIL
      }
    },
    initialData: MOCK_STUDY_DETAIL
  })

  // --- Render Helpers ---
  const renderOverview = () => (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Row gutter={16}>
        <Col span={6}>
          <Statistic title="Status" value={studyReport.overview.status} prefix={<CheckCircleOutlined style={{ color: '#52c41a' }} />} />
        </Col>
        <Col span={6}>
          <Statistic title="Phase" value={studyReport.overview.phase} />
        </Col>
        <Col span={6}>
          <Statistic title="Patients" value={studyReport.overview.current_enrollment} suffix={`/ ${studyReport.overview.target_enrollment}`} />
        </Col>
        <Col span={6}>
          <Statistic title="Active Sites" value={studyReport.overview.active_sites} />
        </Col>
      </Row>

      <Card title="Study Details" size="small">
        <Descriptions bordered column={2}>
          <Descriptions.Item label="Therapeutic Area">{studyReport.overview.therapeutic_area}</Descriptions.Item>
          <Descriptions.Item label="Start Date">{studyReport.overview.start_date}</Descriptions.Item>
          <Descriptions.Item label="Total Sites">{studyReport.overview.total_sites}</Descriptions.Item>
          <Descriptions.Item label="Last Updated">{new Date(studyReport.overview.last_updated).toLocaleString()}</Descriptions.Item>
        </Descriptions>
      </Card>
    </Space>
  )

  const renderDataQuality = () => (
    <Row gutter={16}>
      <Col span={8}>
        <Card>
          <Statistic
            title="Completeness Rate"
            value={studyReport.data_quality.completeness_rate}
            precision={1}
            suffix="%"
            valueStyle={{ color: studyReport.data_quality.completeness_rate > 90 ? '#3f8600' : '#cf1322' }}
          />
        </Card>
      </Col>
      <Col span={8}>
        <Card>
          <Statistic title="Visit Completion" value={studyReport.data_quality.visit_completion_rate} precision={1} suffix="%" />
        </Card>
      </Col>
      <Col span={8}>
        <Card>
          <Statistic title="Form Completion" value={studyReport.data_quality.form_completion_rate} precision={1} suffix="%" />
        </Card>
      </Col>
    </Row>
  )

  const siteColumns: ColumnsType<any> = [
    { title: 'Site ID', dataIndex: 'site_id', key: 'site_id' },
    { title: 'Name', dataIndex: 'site_name', key: 'site_name' },
    { title: 'Country', dataIndex: 'country', key: 'country' },
    { title: 'Patients', dataIndex: 'total_patients', key: 'total_patients', sorter: (a, b) => a.total_patients - b.total_patients },
    {
      title: 'DQI Score',
      dataIndex: 'dqi_score',
      key: 'dqi_score',
      render: (score: number) => <Tag color={score > 90 ? 'green' : 'orange'}>{score}%</Tag>
    },
    { title: 'Open Queries', dataIndex: 'open_queries', key: 'open_queries' },
  ]

  const renderSites = () => (
    <Table
      dataSource={studyReport.sites_summary || []}
      columns={siteColumns}
      rowKey="site_id"
      pagination={{ pageSize: 5 }}
    />
  )

  if (isLoading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}>
        <Spin size="large" tip="Loading report..." />
      </div>
    )
  }

  const items = [
    {
      key: 'overview',
      label: (<span><FileTextOutlined /> Overview</span>),
      children: renderOverview(),
    },
    {
      key: 'quality',
      label: (<span><CheckCircleOutlined /> Data Quality</span>),
      children: renderDataQuality(),
    },
    {
      key: 'sites',
      label: (<span><ExperimentOutlined /> Sites</span>),
      children: renderSites(),
    },
  ]

  return (
    <div style={{ padding: 24, paddingBottom: 48, overflowX: 'hidden' }}>
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space direction="vertical" size={2}>
            <Title level={2} style={{ margin: 0, color: '#fff' }}>
              Study Reports
            </Title>
            <Text type="secondary">{studyReport.study_name}</Text>
          </Space>
          <Space>
            <Button type="primary" icon={<BarChartOutlined />}>Generate PDF</Button>
          </Space>
        </div>

        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          type="card"
          items={items}
        />
      </Space>
    </div>
  )
}
