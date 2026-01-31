import { useState, useMemo } from 'react'
import {
  Card,
  Table,
  Row,
  Col,
  Typography,
  Space,
  Breadcrumb,
  Select,
  DatePicker,
  Input,
  Tag,
  Button,
  Tooltip,
  Progress,
  Statistic,
  Badge,
  Tabs,
  Timeline,
  Divider,
} from 'antd'
import {
  HomeOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  ClockCircleOutlined,
  SearchOutlined,
  DownloadOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  SendOutlined,
  EyeOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons'
import { Pie, Column, Line } from '@ant-design/charts'
import type { ColumnsType } from 'antd/es/table'
import type { QueryDetail, QueryAging, NonConformantData } from '../types'
import dayjs from 'dayjs'

const { Title, Text } = Typography
const { RangePicker } = DatePicker

// Mock query data
const mockQueryData: QueryDetail[] = Array.from({ length: 150 }, (_, i) => ({
  query_id: `QRY-${String(10000 + i).padStart(5, '0')}`,
  subject_id: `SUBJ-${String(1000 + (i % 50)).padStart(4, '0')}`,
  site_id: `SITE-${String(100 + (i % 15)).padStart(3, '0')}`,
  site_name: `Clinical Site ${100 + (i % 15)}`,
  visit_name: `Visit ${(i % 8) + 1}`,
  form_name: ['Demographics', 'Vital Signs', 'Laboratory', 'Adverse Events', 'Concomitant Meds'][i % 5],
  query_type: ['Data Query', 'Protocol Deviation', 'Safety Query', 'Lab Query', 'Other'][i % 5] as QueryDetail['query_type'],
  query_field: ['Date of Birth', 'Blood Pressure', 'Lab Value', 'AE Term', 'Medication Name'][i % 5],
  query_text: [
    'Please verify the entered value.',
    'Missing required data. Please complete.',
    'Value out of expected range.',
    'Inconsistent with previous entry.',
    'Please clarify or correct.',
  ][i % 5],
  opened_date: dayjs().subtract(Math.floor(Math.random() * 60), 'day').format('YYYY-MM-DD'),
  days_open: Math.floor(Math.random() * 60),
  query_status: ['Open', 'Answered', 'Closed', 'Cancelled'][i % 4] as QueryDetail['query_status'],
  assigned_to: ['CRA Team', 'Site 101', 'Site 102', 'Data Manager'][i % 4],
  response_due_date: dayjs().add(Math.floor(Math.random() * 14), 'day').format('YYYY-MM-DD'),
  last_response_date: dayjs().subtract(Math.floor(Math.random() * 7), 'day').format('YYYY-MM-DD'),
  priority_level: ['Critical', 'High', 'Medium', 'Low'][i % 4] as QueryDetail['priority_level'],
}))

export default function DataQuality() {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('queries')
  const [searchText, setSearchText] = useState('')
  const [filters, setFilters] = useState({
    queryType: [] as string[],
    status: [] as string[],
    priority: [] as string[],
    sites: [] as string[],
  })

  // Filter data
  const filteredQueries = useMemo(() => {
    return mockQueryData.filter(item => {
      if (searchText && !item.query_id.toLowerCase().includes(searchText.toLowerCase()) &&
          !item.subject_id.toLowerCase().includes(searchText.toLowerCase())) {
        return false
      }
      if (filters.queryType.length > 0 && !filters.queryType.includes(item.query_type)) {
        return false
      }
      if (filters.status.length > 0 && !filters.status.includes(item.query_status)) {
        return false
      }
      if (filters.priority.length > 0 && !filters.priority.includes(item.priority_level)) {
        return false
      }
      if (filters.sites.length > 0 && !filters.sites.includes(item.site_id)) {
        return false
      }
      return true
    })
  }, [filters, searchText])

  // Calculate metrics
  const queryMetrics = useMemo(() => {
    const total = mockQueryData.length
    const open = mockQueryData.filter(q => q.query_status === 'Open').length
    const answered = mockQueryData.filter(q => q.query_status === 'Answered').length
    const closed = mockQueryData.filter(q => q.query_status === 'Closed').length
    
    const byType = mockQueryData.reduce((acc, q) => {
      acc[q.query_type] = (acc[q.query_type] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const aging: QueryAging = {
      under_7_days: mockQueryData.filter(q => q.days_open < 7 && q.query_status === 'Open').length,
      days_7_to_14: mockQueryData.filter(q => q.days_open >= 7 && q.days_open < 14 && q.query_status === 'Open').length,
      days_15_to_30: mockQueryData.filter(q => q.days_open >= 14 && q.days_open < 30 && q.query_status === 'Open').length,
      over_30_days: mockQueryData.filter(q => q.days_open >= 30 && q.query_status === 'Open').length,
    }

    const avgResolutionTime = mockQueryData
      .filter(q => q.query_status === 'Closed')
      .reduce((sum, q) => sum + q.days_open, 0) / closed || 0

    return {
      total,
      open,
      answered,
      closed,
      cancelled: total - open - answered - closed,
      resolutionRate: ((closed / total) * 100).toFixed(1),
      avgResolutionTime: avgResolutionTime.toFixed(1),
      byType,
      aging,
    }
  }, [])

  // Chart data
  const typeChartData = Object.entries(queryMetrics.byType).map(([type, value]) => ({
    type,
    value,
  }))

  const agingChartData = [
    { range: '< 7 days', count: queryMetrics.aging.under_7_days, color: '#52c41a' },
    { range: '7-14 days', count: queryMetrics.aging.days_7_to_14, color: '#faad14' },
    { range: '15-30 days', count: queryMetrics.aging.days_15_to_30, color: '#fa8c16' },
    { range: '> 30 days', count: queryMetrics.aging.over_30_days, color: '#f5222d' },
  ]

  const statusChartData = [
    { status: 'Open', count: queryMetrics.open },
    { status: 'Answered', count: queryMetrics.answered },
    { status: 'Closed', count: queryMetrics.closed },
  ]

  // Table columns
  const columns: ColumnsType<QueryDetail> = [
    {
      title: 'Query ID',
      dataIndex: 'query_id',
      key: 'query_id',
      width: 120,
      fixed: 'left',
      render: (text) => <a style={{ fontWeight: 500 }}>{text}</a>,
    },
    {
      title: 'Subject ID',
      dataIndex: 'subject_id',
      key: 'subject_id',
      width: 120,
      render: (text) => <a>{text}</a>,
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
      title: 'Visit / Form',
      key: 'visit_form',
      width: 150,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text>{record.visit_name}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>{record.form_name}</Text>
        </Space>
      ),
    },
    {
      title: 'Type',
      dataIndex: 'query_type',
      key: 'query_type',
      width: 130,
      render: (type) => {
        const colors: Record<string, string> = {
          'Data Query': 'blue',
          'Protocol Deviation': 'orange',
          'Safety Query': 'red',
          'Lab Query': 'purple',
          'Other': 'default',
        }
        return <Tag color={colors[type]}>{type}</Tag>
      },
    },
    {
      title: 'Query Field',
      dataIndex: 'query_field',
      key: 'query_field',
      width: 130,
    },
    {
      title: 'Query Text',
      dataIndex: 'query_text',
      key: 'query_text',
      width: 200,
      ellipsis: true,
      render: (text) => (
        <Tooltip title={text}>
          <span>{text}</span>
        </Tooltip>
      ),
    },
    {
      title: 'Opened',
      dataIndex: 'opened_date',
      key: 'opened_date',
      width: 100,
      sorter: (a, b) => dayjs(a.opened_date).unix() - dayjs(b.opened_date).unix(),
    },
    {
      title: 'Days Open',
      dataIndex: 'days_open',
      key: 'days_open',
      width: 100,
      sorter: (a, b) => a.days_open - b.days_open,
      render: (days, record) => {
        if (record.query_status === 'Closed' || record.query_status === 'Cancelled') {
          return <Text type="secondary">{days}</Text>
        }
        return (
          <Tag color={days > 30 ? 'red' : days > 14 ? 'orange' : days > 7 ? 'gold' : 'green'}>
            {days} days
          </Tag>
        )
      },
    },
    {
      title: 'Status',
      dataIndex: 'query_status',
      key: 'query_status',
      width: 100,
      render: (status) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          'Open': { color: 'blue', icon: <ClockCircleOutlined /> },
          'Answered': { color: 'orange', icon: <ExclamationCircleOutlined /> },
          'Closed': { color: 'green', icon: <CheckCircleOutlined /> },
          'Cancelled': { color: 'default', icon: <CloseCircleOutlined /> },
        }
        return (
          <Tag color={config[status].color} icon={config[status].icon}>
            {status}
          </Tag>
        )
      },
    },
    {
      title: 'Assigned To',
      dataIndex: 'assigned_to',
      key: 'assigned_to',
      width: 120,
    },
    {
      title: 'Due Date',
      dataIndex: 'response_due_date',
      key: 'response_due_date',
      width: 100,
      render: (date, record) => {
        if (record.query_status === 'Closed' || record.query_status === 'Cancelled') {
          return <Text type="secondary">{date}</Text>
        }
        const isOverdue = dayjs(date).isBefore(dayjs())
        return (
          <Text type={isOverdue ? 'danger' : undefined}>
            {date}
          </Text>
        )
      },
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
        return <Tag color={colors[priority]}>{priority}</Tag>
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Tooltip title="View Details">
            <Button type="link" size="small" icon={<EyeOutlined />} />
          </Tooltip>
          {record.query_status === 'Open' && (
            <Tooltip title="Send Reminder">
              <Button type="link" size="small" icon={<SendOutlined />} />
            </Tooltip>
          )}
        </Space>
      ),
    },
  ]

  const sites = [...new Set(mockQueryData.map(q => q.site_id))]

  return (
    <div className="data-quality-page">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: 'Data Quality' },
        ]}
        style={{ marginBottom: 16 }}
      />

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0, color: 'var(--gray-800)' }}>
            Data Quality & Query Management
          </Title>
          <Text type="secondary">
            Monitor and manage data queries, track resolution rates, and ensure data quality
          </Text>
        </Space>
      </div>

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Total Open Queries"
              value={queryMetrics.open}
              valueStyle={{ color: '#1890ff' }}
              prefix={<WarningOutlined />}
              suffix={
                <Text type="secondary" style={{ fontSize: 14 }}>
                  of {queryMetrics.total}
                </Text>
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Resolution Rate"
              value={queryMetrics.resolutionRate}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Avg Resolution Time"
              value={queryMetrics.avgResolutionTime}
              suffix="days"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Critical Queries (>30 days)"
              value={queryMetrics.aging.over_30_days}
              valueStyle={{ color: '#f5222d' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Charts Row */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} md={8}>
          <Card title="Queries by Type" size="small">
            <Pie
              data={typeChartData}
              angleField="value"
              colorField="type"
              radius={0.8}
              innerRadius={0.6}
              height={200}
              label={{
                type: 'inner',
                offset: '-50%',
                content: '{value}',
                style: { textAlign: 'center', fontSize: 12 },
              }}
              legend={{ position: 'bottom' }}
            />
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card title="Query Aging Distribution" size="small">
            <Column
              data={agingChartData}
              xField="range"
              yField="count"
              height={200}
              color={({ range }: { range: string }) => {
                const item = agingChartData.find(d => d.range === range)
                return item?.color || '#1890ff'
              }}
              label={{
                position: 'middle',
                style: { fill: '#fff' },
              }}
            />
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card title="Query Status Overview" size="small">
            <div style={{ padding: '20px 0' }}>
              {statusChartData.map((item, index) => (
                <div key={item.status} style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <Text>{item.status}</Text>
                    <Text strong>{item.count}</Text>
                  </div>
                  <Progress
                    percent={(item.count / queryMetrics.total) * 100}
                    showInfo={false}
                    strokeColor={
                      item.status === 'Open' ? '#1890ff' :
                      item.status === 'Answered' ? '#faad14' : '#52c41a'
                    }
                  />
                </div>
              ))}
            </div>
          </Card>
        </Col>
      </Row>

      {/* Tabs */}
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'queries',
            label: (
              <span>
                <WarningOutlined />
                Query Details
                <Badge count={queryMetrics.open} style={{ marginLeft: 8 }} />
              </span>
            ),
            children: (
              <Card size="small">
                {/* Filters */}
                <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                  <Col xs={24} md={4}>
                    <Input
                      placeholder="Search Query/Subject ID"
                      prefix={<SearchOutlined />}
                      value={searchText}
                      onChange={(e) => setSearchText(e.target.value)}
                      allowClear
                    />
                  </Col>
                  <Col xs={24} md={4}>
                    <Select
                      mode="multiple"
                      placeholder="Query Type"
                      style={{ width: '100%' }}
                      options={[
                        { label: 'Data Query', value: 'Data Query' },
                        { label: 'Protocol Deviation', value: 'Protocol Deviation' },
                        { label: 'Safety Query', value: 'Safety Query' },
                        { label: 'Lab Query', value: 'Lab Query' },
                        { label: 'Other', value: 'Other' },
                      ]}
                      value={filters.queryType}
                      onChange={(v) => setFilters({ ...filters, queryType: v })}
                      maxTagCount={1}
                      allowClear
                    />
                  </Col>
                  <Col xs={24} md={4}>
                    <Select
                      mode="multiple"
                      placeholder="Status"
                      style={{ width: '100%' }}
                      options={[
                        { label: 'Open', value: 'Open' },
                        { label: 'Answered', value: 'Answered' },
                        { label: 'Closed', value: 'Closed' },
                        { label: 'Cancelled', value: 'Cancelled' },
                      ]}
                      value={filters.status}
                      onChange={(v) => setFilters({ ...filters, status: v })}
                      maxTagCount={1}
                      allowClear
                    />
                  </Col>
                  <Col xs={24} md={4}>
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
                  <Col xs={24} md={4}>
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
                  <Col xs={24} md={4}>
                    <Space>
                      <Button icon={<ReloadOutlined />}>Refresh</Button>
                      <Button icon={<DownloadOutlined />}>Export</Button>
                    </Space>
                  </Col>
                </Row>

                {/* Table */}
                <Table
                  columns={columns}
                  dataSource={filteredQueries}
                  rowKey="query_id"
                  loading={loading}
                  scroll={{ x: 1800, y: 500 }}
                  pagination={{
                    pageSize: 25,
                    showSizeChanger: true,
                    pageSizeOptions: ['25', '50', '100'],
                    showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} queries`,
                  }}
                  size="small"
                />
              </Card>
            ),
          },
          {
            key: 'nonconformant',
            label: (
              <span>
                <ExclamationCircleOutlined />
                Non-Conformant Data
              </span>
            ),
            children: (
              <Card size="small">
                <Row gutter={[16, 16]}>
                  <Col xs={24} md={8}>
                    <Card title="Non-Conformant Data Summary" size="small">
                      <Statistic
                        title="Total Non-Conformant Data Points"
                        value={234}
                        valueStyle={{ color: '#f5222d' }}
                      />
                      <Divider />
                      <Timeline
                        items={[
                          { color: 'red', children: 'Date format violations: 45' },
                          { color: 'orange', children: 'Range violations: 67' },
                          { color: 'gold', children: 'Missing required fields: 89' },
                          { color: 'blue', children: 'Logic check failures: 33' },
                        ]}
                      />
                    </Card>
                  </Col>
                  <Col xs={24} md={8}>
                    <Card title="Sites with Highest Non-Conformance" size="small">
                      <div>
                        {[
                          { site: 'SITE-101', rate: 15.3 },
                          { site: 'SITE-105', rate: 12.7 },
                          { site: 'SITE-108', rate: 10.2 },
                          { site: 'SITE-103', rate: 8.5 },
                          { site: 'SITE-110', rate: 6.1 },
                        ].map((item) => (
                          <div key={item.site} style={{ marginBottom: 12 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                              <Text>{item.site}</Text>
                              <Text type="danger">{item.rate}%</Text>
                            </div>
                            <Progress
                              percent={item.rate}
                              showInfo={false}
                              strokeColor={item.rate > 10 ? '#f5222d' : item.rate > 5 ? '#faad14' : '#52c41a'}
                            />
                          </div>
                        ))}
                      </div>
                    </Card>
                  </Col>
                  <Col xs={24} md={8}>
                    <Card title="Trend Over Time" size="small">
                      <Line
                        data={[
                          { month: 'Jan', count: 45 },
                          { month: 'Feb', count: 52 },
                          { month: 'Mar', count: 38 },
                          { month: 'Apr', count: 41 },
                          { month: 'May', count: 29 },
                          { month: 'Jun', count: 24 },
                        ]}
                        xField="month"
                        yField="count"
                        height={200}
                        smooth
                        point={{ size: 3 }}
                      />
                    </Card>
                  </Col>
                </Row>
              </Card>
            ),
          },
          {
            key: 'resolution',
            label: (
              <span>
                <CheckCircleOutlined />
                Resolution Tracking
              </span>
            ),
            children: (
              <Card size="small">
                <Row gutter={[16, 16]}>
                  <Col xs={24} md={12}>
                    <Card title="Resolution Rate by Week" size="small">
                      <Line
                        data={[
                          { week: 'Week 1', rate: 65 },
                          { week: 'Week 2', rate: 72 },
                          { week: 'Week 3', rate: 68 },
                          { week: 'Week 4', rate: 78 },
                          { week: 'Week 5', rate: 82 },
                          { week: 'Week 6', rate: 85 },
                        ]}
                        xField="week"
                        yField="rate"
                        height={250}
                        smooth
                        point={{ size: 4 }}
                        yAxis={{ min: 0, max: 100 }}
                      />
                    </Card>
                  </Col>
                  <Col xs={24} md={12}>
                    <Card title="CRA Performance" size="small">
                      <Table
                        dataSource={[
                          { cra: 'CRA Team A', resolved: 145, avgTime: 5.2, rate: 92 },
                          { cra: 'CRA Team B', resolved: 123, avgTime: 6.8, rate: 85 },
                          { cra: 'CRA Team C', resolved: 98, avgTime: 7.1, rate: 78 },
                          { cra: 'CRA Team D', resolved: 87, avgTime: 8.5, rate: 72 },
                        ]}
                        columns={[
                          { title: 'CRA', dataIndex: 'cra', key: 'cra' },
                          { title: 'Queries Resolved', dataIndex: 'resolved', key: 'resolved' },
                          { 
                            title: 'Avg Time (days)', 
                            dataIndex: 'avgTime', 
                            key: 'avgTime',
                            render: (v) => <Text>{v}</Text>,
                          },
                          {
                            title: 'Resolution Rate',
                            dataIndex: 'rate',
                            key: 'rate',
                            render: (v) => <Progress percent={v} size="small" />,
                          },
                        ]}
                        size="small"
                        pagination={false}
                      />
                    </Card>
                  </Col>
                </Row>
              </Card>
            ),
          },
        ]}
      />
    </div>
  )
}
