import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { labsApi } from '../api'
import {
  Card,
  Table,
  Row,
  Col,
  Typography,
  Space,
  Breadcrumb,
  Select,
  Input,
  Tag,
  Button,
  Tooltip,
  Statistic,
  Badge,
  Tabs,
  Timeline,
  Avatar,
  Divider,
  Progress,
  Alert,
} from 'antd'
import {
  HomeOutlined,
  ExperimentOutlined,
  WarningOutlined,
  ClockCircleOutlined,
  SearchOutlined,
  DownloadOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  UserOutlined,
  FileSearchOutlined,
} from '@ant-design/icons'
// Replaced @ant-design/charts with recharts for consistency
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend
} from 'recharts'
import type { ColumnsType } from 'antd/es/table'
import type { MissingLabData, LabReconciliationSummary } from '../types'
import dayjs from 'dayjs'

const { Title, Text } = Typography

// Colors for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

export default function LaboratoryData() {
  const [activeTab, setActiveTab] = useState('missing')
  const [searchText, setSearchText] = useState('')
  const [filters, setFilters] = useState({
    missingElement: [] as string[],
    sites: [] as string[],
    status: [] as string[],
    priority: [] as string[],
    labType: [] as string[],
  })

  // Mock data for fallback
  const MOCK_LAB_DATA: MissingLabData[] = Array.from({ length: 50 }, (_, i) => ({
    id: `LB-${1000 + i}`,
    study_id: 'Study_1',
    subject_id: `SUBJ-${1000 + (i % 20)}`,
    site_id: `SITE-${100 + (i % 5)}`,
    site_name: `Site ${100 + (i % 5)}`,
    visit_name: 'Visit 1',
    lab_test_name: 'Hemoglobin',
    missing_element: 'Unit',
    collection_date: '2023-01-01',
    received_date: '2023-01-02',
    days_since_collection: 5,
    priority_level: 'High',
    assigned_to: 'User',
    resolution_status: 'Open',
    comments: ''
  }))

  // Allow fetching all studies by default
  const [selectedStudyId, setSelectedStudyId] = useState<string | null>(null)

  // Fetch Missing Lab Data
  const { data: missingData = [], isLoading, refetch } = useQuery<MissingLabData[]>({
    queryKey: ['labMissingData', selectedStudyId],
    queryFn: () => labsApi.getMissingLabData(selectedStudyId || undefined),
    // Always enabled to fetch initial data
  })

  const missingLabData = missingData

  // Fetch Lab Summary
  const { data: summaryMetrics } = useQuery<LabReconciliationSummary>({
    queryKey: ['lab-summary'],
    queryFn: () => labsApi.getLabSummary(),
    refetchInterval: 30000,
  })

  // Filter data
  const filteredLabData = useMemo(() => {
    if (!Array.isArray(missingLabData)) return []

    return missingLabData.filter((item: MissingLabData) => {
      if (searchText && !item.subject_id.toLowerCase().includes(searchText.toLowerCase())) {
        return false
      }
      if (filters.missingElement.length > 0 && !filters.missingElement.includes(item.missing_element)) {
        return false
      }
      if (filters.sites.length > 0 && !filters.sites.includes(item.site_id)) {
        return false
      }
      if (filters.status.length > 0 && !filters.status.includes(item.resolution_status)) {
        return false
      }
      if (filters.priority.length > 0 && !filters.priority.includes(item.priority_level)) {
        return false
      }
      if (filters.labType.length > 0 && !filters.labType.includes(item.lab_test_name)) {
        return false
      }
      return true
    })
  }, [filters, searchText, missingLabData])

  // Calculate metrics (derived from real data now)
  const labMetrics = useMemo(() => {
    const data = Array.isArray(missingLabData) ? missingLabData : []

    const missingLabNames = data.filter(d => d.missing_element === 'Lab Name').length
    const missingRefRanges = data.filter(d => d.missing_element === 'Reference Range').length
    const missingUnits = data.filter(d => d.missing_element === 'Unit').length
    const resolved = data.filter(d => d.resolution_status === 'Resolved').length
    const avgDays = data.reduce((sum, d) => sum + d.days_since_collection, 0) / (data.length || 1)

    const byLabTypeRaw = data.reduce((acc, d) => {
      acc[d.lab_test_name] = (acc[d.lab_test_name] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    // specific logic for pie chart: Top 5 + Others
    const sortedLabTypes = Object.entries(byLabTypeRaw)
      .sort((a, b) => b[1] - a[1])

    const top5LabTypes = sortedLabTypes.slice(0, 5).map(([type, count]) => ({ name: type, value: count }))
    const otherCount = sortedLabTypes.slice(5).reduce((sum, [, count]) => sum + count, 0)

    if (otherCount > 0) {
      top5LabTypes.push({ name: 'Others', value: otherCount })
    }

    const bySite = data.reduce((acc, d) => {
      acc[d.site_id] = (acc[d.site_id] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    return {
      total: data.length,
      missingLabNames,
      missingRefRanges,
      missingUnits,
      resolved,
      open: data.length - resolved,
      avgResolutionTime: avgDays.toFixed(1),
      byLabType: top5LabTypes,
      bySite: Object.entries(bySite)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10) // Top 10 sites
        .map(([site, issues]) => ({ name: site, value: issues })),
    }
  }, [missingLabData])

  // Table columns
  const columns: ColumnsType<MissingLabData> = [
    {
      title: 'Subject ID',
      dataIndex: 'subject_id',
      key: 'subject_id',
      width: 120,
      fixed: 'left',
      render: (text) => <a style={{ fontWeight: 500 }}>{text}</a>,
      sorter: (a, b) => a.subject_id.localeCompare(b.subject_id),
    },
    {
      title: 'Site',
      dataIndex: 'site_id',
      key: 'site_id',
      width: 100,
      render: (text, record) => (
        <Tooltip title={record.site_name}>
          <span>{text}</span>
        </Tooltip>
      ),
    },
    {
      title: 'Visit',
      dataIndex: 'visit_name',
      key: 'visit_name',
      width: 90,
    },
    {
      title: 'Lab Test',
      dataIndex: 'lab_test_name',
      key: 'lab_test_name',
      width: 180,
      render: (text) => (
        <Tooltip title={text}>
          <span>{text}</span>
        </Tooltip>
      ),
    },
    {
      title: 'Missing Element',
      dataIndex: 'missing_element',
      key: 'missing_element',
      width: 130,
      render: (element) => {
        const colors: Record<string, string> = {
          'Lab Name': 'red',
          'Reference Range': 'orange',
          'Unit': 'blue',
        }
        return <Tag color={colors[element] || 'default'}>{element || 'Unknown'}</Tag>
      },
    },
    {
      title: 'Collection Date',
      dataIndex: 'collection_date',
      key: 'collection_date',
      width: 120,
      sorter: (a, b) => dayjs(a.collection_date).unix() - dayjs(b.collection_date).unix(),
    },
    {
      title: 'Received Date',
      dataIndex: 'received_date',
      key: 'received_date',
      width: 120,
    },
    {
      title: 'Days Since Collection',
      dataIndex: 'days_since_collection',
      key: 'days_since_collection',
      width: 140,
      sorter: (a, b) => a.days_since_collection - b.days_since_collection,
      render: (days) => (
        <Tag color={days > 20 ? 'red' : days > 10 ? 'orange' : 'green'}>
          {days} days
        </Tag>
      ),
    },
    {
      title: 'Priority',
      dataIndex: 'priority_level',
      key: 'priority_level',
      width: 90,
      render: (priority) => {
        const colors: Record<string, string> = {
          'Critical': 'red',
          'High': 'orange',
          'Medium': 'blue',
          'Low': 'default',
        }
        return <Tag color={colors[priority] || 'default'}>{priority || 'Normal'}</Tag>
      },
    },
    {
      title: 'Assigned To',
      dataIndex: 'assigned_to',
      key: 'assigned_to',
      width: 120,
      render: (name) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          <Text>{name}</Text>
        </Space>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'resolution_status',
      key: 'resolution_status',
      width: 110,
      render: (status) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          'Open': { color: 'error', icon: <ExclamationCircleOutlined /> },
          'In Progress': { color: 'processing', icon: <ClockCircleOutlined /> },
          'Resolved': { color: 'success', icon: <CheckCircleOutlined /> },
        }
        const statusConfig = config[status] || config['Open'] || { color: 'default', icon: null }
        return (
          <Tag color={statusConfig.color} icon={statusConfig.icon}>
            {status || 'Unknown'}
          </Tag>
        )
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Button type="link" size="small" icon={<FileSearchOutlined />}>
            View
          </Button>
        </Space>
      ),
    },
  ]

  const sites = labMetrics.bySite.map(s => s.name)
  const labTypes = labMetrics.byLabType.map(l => l.name)

  return (
    <div className="laboratory-data-page" >
      {/* Breadcrumb */}
      <Breadcrumb
        items={
          [
            { href: '/', title: <HomeOutlined /> },
            { title: 'Laboratory Data' },
          ]}
        style={{ marginBottom: 16 }}
      />

      {/* Header */}
      < div style={{ marginBottom: 24 }}>
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0, color: 'var(--gray-800)' }}>
            Laboratory Data Management
          </Title>
          <Text type="secondary">
            Track missing lab data, reference ranges, and reconciliation status
          </Text>
        </Space>
      </div >

      {/* Debug Info */}
      <Alert
        message="Data Status"
        description={
          <div>
            Data loaded: {missingLabData?.length} records.
            Charts are now enabled.
          </div>
        }
        type="success"
        showIcon
        closable
        style={{ marginBottom: 16 }}
      />

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Total Missing Lab Names"
              value={labMetrics.missingLabNames}
              valueStyle={{ color: '#f5222d' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Missing Reference Ranges"
              value={labMetrics.missingRefRanges}
              valueStyle={{ color: '#fa8c16' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Missing Units"
              value={labMetrics.missingUnits}
              valueStyle={{ color: '#1890ff' }}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Avg Resolution Time"
              value={labMetrics.avgResolutionTime}
              suffix="days"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Tabs */}
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'missing',
            label: (
              <span>
                <WarningOutlined />
                Missing Lab Data
                <Badge count={labMetrics.open} style={{ marginLeft: 8 }} />
              </span>
            ),
            children: (
              <>
                {/* Filters */}
                <Card size="small" style={{ marginBottom: 16 }}>
                  <Row gutter={[16, 16]} align="middle">
                    <Col xs={24} md={4}>
                      <Input
                        placeholder="Search Subject ID"
                        prefix={<SearchOutlined />}
                        value={searchText}
                        onChange={(e) => setSearchText(e.target.value)}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={3}>
                      <Select
                        mode="multiple"
                        placeholder="Missing Element"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Lab Name', value: 'Lab Name' },
                          { label: 'Reference Range', value: 'Reference Range' },
                          { label: 'Unit', value: 'Unit' },
                        ]}
                        value={filters.missingElement}
                        onChange={(v) => setFilters({ ...filters, missingElement: v })}
                        maxTagCount={1}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={3}>
                      <Select
                        mode="multiple"
                        placeholder="Lab Type"
                        style={{ width: '100%' }}
                        options={labTypes.map(t => ({ label: t, value: t }))}
                        value={filters.labType}
                        onChange={(v) => setFilters({ ...filters, labType: v })}
                        maxTagCount={1}
                        showSearch
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={3}>
                      <Select
                        mode="multiple"
                        placeholder="Site"
                        style={{ width: '100%' }}
                        options={sites.map(s => ({ label: s, value: s }))}
                        value={filters.sites}
                        onChange={(v) => setFilters({ ...filters, sites: v })}
                        maxTagCount={1}
                        showSearch
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={3}>
                      <Select
                        mode="multiple"
                        placeholder="Priority"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Critical', value: 'Critical' },
                          { label: 'High', value: 'High' },
                          { label: 'Medium', value: 'Medium' },
                          { label: 'Low', value: 'Low' },
                        ]}
                        value={filters.priority}
                        onChange={(v) => setFilters({ ...filters, priority: v })}
                        maxTagCount={1}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={3}>
                      <Select
                        mode="multiple"
                        placeholder="Status"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Open', value: 'Open' },
                          { label: 'In Progress', value: 'In Progress' },
                          { label: 'Resolved', value: 'Resolved' },
                        ]}
                        value={filters.status}
                        onChange={(v) => setFilters({ ...filters, status: v })}
                        maxTagCount={1}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={5}>
                      <Space>
                        <Button icon={<ReloadOutlined />} onClick={() => refetch()}>Refresh</Button>
                        <Button icon={<DownloadOutlined />}>Export</Button>
                      </Space>
                    </Col>
                  </Row>
                </Card>

                {/* Table */}
                <Card size="small">
                  <Table
                    columns={columns}
                    dataSource={filteredLabData}
                    rowKey="id"
                    loading={isLoading}
                    scroll={{ x: 1500, y: 500 }}
                    pagination={{
                      pageSize: 25,
                      showSizeChanger: true,
                      pageSizeOptions: ['25', '50', '100'],
                      showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} records`,
                    }}
                    size="small"
                  />
                </Card>
              </>
            ),
          },
          {
            key: 'reconciliation',
            label: (
              <span>
                <ExperimentOutlined />
                Lab Reconciliation
              </span>
            ),
            children: (
              <Row gutter={[16, 16]}>
                <Col xs={24} md={8}>
                  <Card title="Lab Type Breakdown" size="small">
                    <div style={{ width: '100%', height: 300 }}>
                      <ResponsiveContainer>
                        <PieChart>
                          <Pie
                            data={labMetrics.byLabType}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {labMetrics.byLabType.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <RechartsTooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </Card>
                </Col>
                <Col xs={24} md={8}>
                  <Card title="Site Performance" size="small">
                    <div style={{ width: '100%', height: 300 }}>
                      <ResponsiveContainer>
                        <BarChart
                          data={labMetrics.bySite}
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" />
                          <YAxis type="category" dataKey="name" width={80} />
                          <RechartsTooltip />
                          <Bar dataKey="value" fill="#fa8c16" name="Issues" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </Card>
                </Col>
                <Col xs={24} md={8}>
                  <Card title="Issue Tracking Summary" size="small">
                    <Space direction="vertical" style={{ width: '100%' }} size={16}>
                      <div>
                        <Text>Open Issues</Text>
                        <Progress
                          percent={Math.round((labMetrics.open / labMetrics.total) * 100)}
                          status="exception"
                        />
                      </div>
                      <div>
                        <Text>In Progress</Text>
                        <Progress
                          percent={30}
                          status="active"
                        />
                      </div>
                      <div>
                        <Text>Resolved</Text>
                        <Progress
                          percent={Math.round((labMetrics.resolved / labMetrics.total) * 100)}
                          status="success"
                        />
                      </div>
                      <Divider />
                      <Timeline
                        items={[
                          { color: 'red', children: `Critical Issues: ${Array.isArray(missingLabData) ? missingLabData.filter((d: MissingLabData) => d.priority_level === 'Critical').length : 0}` },
                          { color: 'orange', children: `High Priority: ${Array.isArray(missingLabData) ? missingLabData.filter((d: MissingLabData) => d.priority_level === 'High').length : 0}` },
                          { color: 'blue', children: `Medium Priority: ${Array.isArray(missingLabData) ? missingLabData.filter((d: MissingLabData) => d.priority_level === 'Medium').length : 0}` },
                          { color: 'green', children: `Low Priority: ${Array.isArray(missingLabData) ? missingLabData.filter((d: MissingLabData) => d.priority_level === 'Low').length : 0}` },
                        ]}
                      />
                    </Space>
                  </Card>
                </Col>
              </Row>
            ),
          },
        ]}
      />
    </div>
  )
}

