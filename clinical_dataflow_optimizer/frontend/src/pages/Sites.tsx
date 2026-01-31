import { useState } from 'react'
import { useParams } from 'react-router-dom'
import {
  Card,
  Table,
  Tag,
  Typography,
  Breadcrumb,
  Space,
  Row,
  Col,
  Statistic,
  Progress,
  Select,
  Drawer,
  Descriptions,
  List,
} from 'antd'
import {
  HomeOutlined,
  BankOutlined,
  TeamOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { sitesApi } from '../api'
import { useStore } from '../store'
import type { Site, SitePerformance } from '../types'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text } = Typography

const statusColors: Record<string, string> = {
  active: 'green',
  enrolling: 'blue',
  closed: 'default',
  suspended: 'orange',
}

export default function Sites() {
  const { siteId } = useParams()
  const { selectedStudyId } = useStore()
  
  const [regionFilter, setRegionFilter] = useState<string | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [selectedSite, setSelectedSite] = useState<Site | null>(null)

  // Fetch sites
  const { data: sites = [], isLoading } = useQuery({
    queryKey: ['sites', selectedStudyId, regionFilter],
    queryFn: () => sitesApi.getAll({
      study_id: selectedStudyId || undefined,
      region: regionFilter || undefined,
    }),
  })

  // Fetch high-risk sites
  const { data: highRiskSites = [] } = useQuery({
    queryKey: ['highRiskSites', selectedStudyId],
    queryFn: () => sitesApi.getHighRisk(selectedStudyId || undefined),
  })

  // Fetch site performance for selected site
  const { data: sitePerformance } = useQuery({
    queryKey: ['sitePerformance', selectedSite?.site_id],
    queryFn: () => sitesApi.getPerformance(selectedSite!.site_id),
    enabled: !!selectedSite,
  })

  // Get unique regions for filter
  const regions = [...new Set(sites.map(s => s.region).filter(Boolean))]

  const handleViewSite = (site: Site) => {
    setSelectedSite(site)
    setDrawerOpen(true)
  }

  const columns: ColumnsType<Site> = [
    {
      title: 'Site ID',
      dataIndex: 'site_id',
      key: 'site_id',
      render: (text, record) => (
        <a onClick={() => handleViewSite(record)}>{text}</a>
      ),
      sorter: (a, b) => a.site_id.localeCompare(b.site_id),
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
    },
    {
      title: 'Country',
      dataIndex: 'country',
      key: 'country',
      render: (country) => <Tag>{country}</Tag>,
      filters: [...new Set(sites.map(s => s.country).filter(Boolean))].map(c => ({
        text: c,
        value: c,
      })),
      onFilter: (value, record) => record.country === value,
    },
    {
      title: 'Region',
      dataIndex: 'region',
      key: 'region',
      filters: regions.map(r => ({ text: r, value: r })),
      onFilter: (value, record) => record.region === value,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={statusColors[status?.toLowerCase()] || 'default'}>
          {status}
        </Tag>
      ),
    },
    {
      title: 'Patients',
      dataIndex: 'patient_count',
      key: 'patient_count',
      sorter: (a, b) => (a.patient_count || 0) - (b.patient_count || 0),
    },
    {
      title: 'DQI Score',
      dataIndex: 'dqi_score',
      key: 'dqi_score',
      render: (score) => (
        <Progress
          percent={Math.round(score || 0)}
          size="small"
          status={score >= 80 ? 'success' : score >= 60 ? 'normal' : 'exception'}
          showInfo
          format={(percent) => `${percent}%`}
        />
      ),
      sorter: (a, b) => (a.dqi_score || 0) - (b.dqi_score || 0),
    },
    {
      title: 'Avg Resolution (days)',
      dataIndex: 'query_resolution_time',
      key: 'query_resolution_time',
      render: (time) => (
        <span style={{ color: time > 7 ? '#ff4d4f' : time > 3 ? '#faad14' : '#52c41a' }}>
          {time?.toFixed(1) || '-'}
        </span>
      ),
      sorter: (a, b) => (a.query_resolution_time || 0) - (b.query_resolution_time || 0),
    },
  ]

  // Calculate summary stats
  const avgDQI = sites.length > 0
    ? sites.reduce((sum, s) => sum + (s.dqi_score || 0), 0) / sites.length
    : 0
  const totalPatients = sites.reduce((sum, s) => sum + (s.patient_count || 0), 0)

  return (
    <div>
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: <><BankOutlined /> Sites</> },
        ]}
        style={{ marginBottom: 16 }}
      />

      <Title level={2}>Sites</Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
        Site performance metrics and data quality overview
      </Text>

      {/* Summary Stats */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Sites"
              value={sites.length}
              prefix={<BankOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Patients"
              value={totalPatients}
              prefix={<TeamOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg DQI Score"
              value={avgDQI.toFixed(1)}
              suffix="%"
              valueStyle={{ color: avgDQI >= 80 ? '#52c41a' : '#faad14' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="High Risk Sites"
              value={highRiskSites.length}
              valueStyle={{ color: highRiskSites.length > 0 ? '#ff4d4f' : '#52c41a' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      <Card style={{ marginBottom: 24 }}>
        <Space wrap>
          <Select
            placeholder="Region"
            allowClear
            style={{ width: 200 }}
            onChange={setRegionFilter}
            options={regions.map(r => ({ value: r, label: r }))}
          />
        </Space>
      </Card>

      {/* Sites Table */}
      <Card>
        <Table
          columns={columns}
          dataSource={sites}
          rowKey="site_id"
          loading={isLoading}
          pagination={{ pageSize: 15, showSizeChanger: true }}
        />
      </Card>

      {/* Site Detail Drawer */}
      <Drawer
        title={`Site: ${selectedSite?.site_id}`}
        placement="right"
        width={600}
        open={drawerOpen}
        onClose={() => {
          setDrawerOpen(false)
          setSelectedSite(null)
        }}
      >
        {selectedSite && (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="Site ID">{selectedSite.site_id}</Descriptions.Item>
              <Descriptions.Item label="Name">{selectedSite.name}</Descriptions.Item>
              <Descriptions.Item label="Country">{selectedSite.country}</Descriptions.Item>
              <Descriptions.Item label="Region">{selectedSite.region}</Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={statusColors[selectedSite.status?.toLowerCase()]}>
                  {selectedSite.status}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Patients">{selectedSite.patient_count}</Descriptions.Item>
            </Descriptions>

            {sitePerformance && (
              <Card title="Performance Metrics" size="small">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Statistic
                      title="DQI Score"
                      value={sitePerformance.dqi_score?.toFixed(1)}
                      suffix="%"
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="Cleanliness Rate"
                      value={sitePerformance.cleanliness_rate?.toFixed(1)}
                      suffix="%"
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="Open Queries"
                      value={sitePerformance.query_count}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="Resolution Rate"
                      value={sitePerformance.query_resolution_rate?.toFixed(1)}
                      suffix="%"
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="Avg Resolution Time"
                      value={sitePerformance.avg_resolution_time?.toFixed(1)}
                      suffix="days"
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="Protocol Deviations"
                      value={sitePerformance.protocol_deviations}
                    />
                  </Col>
                </Row>
              </Card>
            )}
          </Space>
        )}
      </Drawer>
    </div>
  )
}
