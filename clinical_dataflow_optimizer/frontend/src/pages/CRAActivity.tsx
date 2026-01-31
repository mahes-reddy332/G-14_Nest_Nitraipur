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
  Input,
  Tag,
  Button,
  Tooltip,
  Statistic,
  Badge,
  Tabs,
  Progress,
  DatePicker,
  Avatar,
  Rate,
  Timeline,
} from 'antd'
import {
  HomeOutlined,
  UserOutlined,
  TeamOutlined,
  CalendarOutlined,
  SearchOutlined,
  DownloadOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  EnvironmentOutlined,
  FileSearchOutlined,
  RiseOutlined,
  FallOutlined,
  TrophyOutlined,
} from '@ant-design/icons'
import { Column, Line, Radar } from '@ant-design/charts'
import type { ColumnsType } from 'antd/es/table'
import type { CRAPerformance, MonitoringVisit, FollowUpItem } from '../types'
import dayjs from 'dayjs'

const { Title, Text } = Typography
const { RangePicker } = DatePicker

// Mock data for CRA Performance
const mockCRAPerformance: CRAPerformance[] = [
  {
    cra_id: 'CRA-001',
    cra_name: 'Sarah Johnson',
    region: 'North America',
    assigned_sites: 5,
    total_visits_completed: 45,
    avg_monitoring_duration: 6.5,
    avg_queries_per_visit: 12.3,
    sdv_completion_rate: 94.5,
    issue_resolution_rate: 88.2,
    follow_up_adherence: 92.0,
    performance_score: 91,
    performance_trend: 'up',
    last_visit_date: dayjs().subtract(2, 'day').format('YYYY-MM-DD'),
  },
  {
    cra_id: 'CRA-002',
    cra_name: 'Michael Chen',
    region: 'Asia Pacific',
    assigned_sites: 6,
    total_visits_completed: 52,
    avg_monitoring_duration: 7.2,
    avg_queries_per_visit: 15.8,
    sdv_completion_rate: 89.3,
    issue_resolution_rate: 85.6,
    follow_up_adherence: 88.5,
    performance_score: 85,
    performance_trend: 'stable',
    last_visit_date: dayjs().subtract(5, 'day').format('YYYY-MM-DD'),
  },
  {
    cra_id: 'CRA-003',
    cra_name: 'Emma Williams',
    region: 'Europe',
    assigned_sites: 4,
    total_visits_completed: 38,
    avg_monitoring_duration: 5.8,
    avg_queries_per_visit: 10.5,
    sdv_completion_rate: 96.8,
    issue_resolution_rate: 92.1,
    follow_up_adherence: 95.3,
    performance_score: 95,
    performance_trend: 'up',
    last_visit_date: dayjs().subtract(1, 'day').format('YYYY-MM-DD'),
  },
  {
    cra_id: 'CRA-004',
    cra_name: 'David Brown',
    region: 'North America',
    assigned_sites: 5,
    total_visits_completed: 41,
    avg_monitoring_duration: 6.9,
    avg_queries_per_visit: 14.2,
    sdv_completion_rate: 87.5,
    issue_resolution_rate: 82.4,
    follow_up_adherence: 85.0,
    performance_score: 78,
    performance_trend: 'down',
    last_visit_date: dayjs().subtract(8, 'day').format('YYYY-MM-DD'),
  },
  {
    cra_id: 'CRA-005',
    cra_name: 'Lisa Martinez',
    region: 'Latin America',
    assigned_sites: 4,
    total_visits_completed: 35,
    avg_monitoring_duration: 7.5,
    avg_queries_per_visit: 13.1,
    sdv_completion_rate: 91.2,
    issue_resolution_rate: 89.8,
    follow_up_adherence: 90.5,
    performance_score: 88,
    performance_trend: 'up',
    last_visit_date: dayjs().subtract(3, 'day').format('YYYY-MM-DD'),
  },
]

// Mock data for Monitoring Visits
const mockMonitoringVisits: MonitoringVisit[] = Array.from({ length: 40 }, (_, i) => ({
  visit_id: `MV-${String(1000 + i).padStart(4, '0')}`,
  site_id: `SITE-${String(100 + (i % 10)).padStart(3, '0')}`,
  site_name: `Clinical Site ${100 + (i % 10)}`,
  cra_id: mockCRAPerformance[i % 5].cra_id,
  cra_name: mockCRAPerformance[i % 5].cra_name,
  visit_type: ['Initiation', 'Routine', 'Interim', 'Close-out', 'For Cause'][i % 5] as MonitoringVisit['visit_type'],
  planned_date: dayjs().add(Math.floor(Math.random() * 30) - 15, 'day').format('YYYY-MM-DD'),
  actual_date: i % 3 === 0 ? '' : dayjs().subtract(Math.floor(Math.random() * 10), 'day').format('YYYY-MM-DD'),
  status: ['Scheduled', 'In Progress', 'Completed', 'Report Pending', 'Overdue'][i % 5] as MonitoringVisit['status'],
  duration_hours: i % 3 === 0 ? 0 : Math.floor(Math.random() * 4) + 4,
  subjects_reviewed: Math.floor(Math.random() * 20) + 5,
  queries_generated: Math.floor(Math.random() * 30) + 5,
  issues_identified: Math.floor(Math.random() * 8),
  report_submitted: i % 3 !== 0 && i % 2 === 0,
}))

// Mock data for Follow-up Items
const mockFollowUpItems: FollowUpItem[] = Array.from({ length: 35 }, (_, i) => ({
  follow_up_id: `FU-${String(1000 + i).padStart(4, '0')}`,
  site_id: `SITE-${String(100 + (i % 10)).padStart(3, '0')}`,
  site_name: `Clinical Site ${100 + (i % 10)}`,
  cra_id: mockCRAPerformance[i % 5].cra_id,
  cra_name: mockCRAPerformance[i % 5].cra_name,
  issue_category: ['Protocol Deviation', 'Data Discrepancy', 'Consent Issue', 'Storage Issue', 'Documentation'][i % 5],
  description: [
    'Missing signature on informed consent page 3',
    'Lab values not entered for Visit 4',
    'Temperature excursion not documented',
    'Protocol deviation not reported timely',
    'Source document discrepancy for vital signs',
  ][i % 5],
  priority: ['High', 'Medium', 'Low'][i % 3] as FollowUpItem['priority'],
  status: ['Open', 'In Progress', 'Pending Response', 'Resolved'][i % 4] as FollowUpItem['status'],
  date_identified: dayjs().subtract(Math.floor(Math.random() * 30) + 5, 'day').format('YYYY-MM-DD'),
  due_date: dayjs().add(Math.floor(Math.random() * 14) - 7, 'day').format('YYYY-MM-DD'),
  days_open: Math.floor(Math.random() * 25) + 3,
  response_received: i % 4 >= 2,
}))

export default function CRAActivityPage() {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('performance')
  const [searchText, setSearchText] = useState('')
  const [selectedCRA, setSelectedCRA] = useState<string | null>(null)

  // Calculate metrics
  const metrics = useMemo(() => {
    const totalCRAs = mockCRAPerformance.length
    const avgPerformance = (mockCRAPerformance.reduce((sum, c) => sum + c.performance_score, 0) / totalCRAs).toFixed(1)
    const totalVisits = mockMonitoringVisits.length
    const completedVisits = mockMonitoringVisits.filter(v => v.status === 'Completed').length
    const overdueVisits = mockMonitoringVisits.filter(v => v.status === 'Overdue').length
    const openFollowUps = mockFollowUpItems.filter(f => f.status === 'Open' || f.status === 'In Progress').length
    const highPriorityFollowUps = mockFollowUpItems.filter(f => f.priority === 'High' && f.status !== 'Resolved').length

    const visitsByType = mockMonitoringVisits.reduce((acc, v) => {
      acc[v.visit_type] = (acc[v.visit_type] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    return {
      totalCRAs,
      avgPerformance,
      totalVisits,
      completedVisits,
      overdueVisits,
      openFollowUps,
      highPriorityFollowUps,
      visitsByType: Object.entries(visitsByType).map(([type, count]) => ({ type, count })),
    }
  }, [])

  // CRA Performance columns
  const craColumns: ColumnsType<CRAPerformance> = [
    {
      title: 'CRA',
      key: 'cra',
      width: 200,
      fixed: 'left',
      render: (_, record) => (
        <Space>
          <Avatar icon={<UserOutlined />} style={{ backgroundColor: '#1890ff' }} />
          <div>
            <Text strong>{record.cra_name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>{record.cra_id}</Text>
          </div>
        </Space>
      ),
    },
    {
      title: 'Region',
      dataIndex: 'region',
      key: 'region',
      width: 130,
      render: (region) => (
        <Tag icon={<EnvironmentOutlined />} color="blue">{region}</Tag>
      ),
    },
    {
      title: 'Sites',
      dataIndex: 'assigned_sites',
      key: 'assigned_sites',
      width: 80,
      align: 'center',
    },
    {
      title: 'Visits',
      dataIndex: 'total_visits_completed',
      key: 'total_visits_completed',
      width: 80,
      align: 'center',
    },
    {
      title: 'Avg Duration (hrs)',
      dataIndex: 'avg_monitoring_duration',
      key: 'avg_monitoring_duration',
      width: 130,
      render: (val) => `${val} hrs`,
    },
    {
      title: 'SDV Rate',
      dataIndex: 'sdv_completion_rate',
      key: 'sdv_completion_rate',
      width: 120,
      render: (rate) => (
        <Progress
          percent={rate}
          size="small"
          strokeColor={rate >= 90 ? '#52c41a' : rate >= 80 ? '#faad14' : '#f5222d'}
        />
      ),
    },
    {
      title: 'Issue Resolution',
      dataIndex: 'issue_resolution_rate',
      key: 'issue_resolution_rate',
      width: 130,
      render: (rate) => (
        <Progress
          percent={rate}
          size="small"
          strokeColor={rate >= 85 ? '#52c41a' : rate >= 75 ? '#faad14' : '#f5222d'}
        />
      ),
    },
    {
      title: 'Performance',
      key: 'performance',
      width: 150,
      sorter: (a, b) => a.performance_score - b.performance_score,
      render: (_, record) => (
        <Space>
          <Progress
            type="circle"
            percent={record.performance_score}
            width={40}
            strokeColor={record.performance_score >= 90 ? '#52c41a' : record.performance_score >= 80 ? '#faad14' : '#f5222d'}
          />
          {record.performance_trend === 'up' && <RiseOutlined style={{ color: '#52c41a' }} />}
          {record.performance_trend === 'down' && <FallOutlined style={{ color: '#f5222d' }} />}
        </Space>
      ),
    },
    {
      title: 'Last Visit',
      dataIndex: 'last_visit_date',
      key: 'last_visit_date',
      width: 110,
    },
  ]

  // Monitoring Visit columns
  const visitColumns: ColumnsType<MonitoringVisit> = [
    {
      title: 'Visit ID',
      dataIndex: 'visit_id',
      key: 'visit_id',
      width: 100,
      fixed: 'left',
    },
    {
      title: 'Site',
      key: 'site',
      width: 150,
      render: (_, record) => (
        <div>
          <Text>{record.site_id}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 11 }}>{record.site_name}</Text>
        </div>
      ),
    },
    {
      title: 'CRA',
      dataIndex: 'cra_name',
      key: 'cra_name',
      width: 140,
    },
    {
      title: 'Type',
      dataIndex: 'visit_type',
      key: 'visit_type',
      width: 110,
      render: (type) => {
        const colors: Record<string, string> = {
          'Initiation': 'green',
          'Routine': 'blue',
          'Interim': 'cyan',
          'Close-out': 'purple',
          'For Cause': 'red',
        }
        return <Tag color={colors[type]}>{type}</Tag>
      },
    },
    {
      title: 'Planned Date',
      dataIndex: 'planned_date',
      key: 'planned_date',
      width: 110,
    },
    {
      title: 'Actual Date',
      dataIndex: 'actual_date',
      key: 'actual_date',
      width: 110,
      render: (date) => date || <Text type="secondary">-</Text>,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 130,
      render: (status) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          'Scheduled': { color: 'default', icon: <CalendarOutlined /> },
          'In Progress': { color: 'processing', icon: <ClockCircleOutlined /> },
          'Completed': { color: 'success', icon: <CheckCircleOutlined /> },
          'Report Pending': { color: 'warning', icon: <FileSearchOutlined /> },
          'Overdue': { color: 'error', icon: <ExclamationCircleOutlined /> },
        }
        return (
          <Tag color={config[status].color} icon={config[status].icon}>
            {status}
          </Tag>
        )
      },
    },
    {
      title: 'Duration',
      dataIndex: 'duration_hours',
      key: 'duration_hours',
      width: 90,
      render: (hrs) => hrs ? `${hrs} hrs` : '-',
    },
    {
      title: 'Subjects',
      dataIndex: 'subjects_reviewed',
      key: 'subjects_reviewed',
      width: 90,
      align: 'center',
    },
    {
      title: 'Queries',
      dataIndex: 'queries_generated',
      key: 'queries_generated',
      width: 80,
      align: 'center',
    },
    {
      title: 'Issues',
      dataIndex: 'issues_identified',
      key: 'issues_identified',
      width: 80,
      render: (count) => (
        <Badge count={count} style={{ backgroundColor: count > 5 ? '#f5222d' : count > 2 ? '#faad14' : '#52c41a' }} showZero />
      ),
    },
  ]

  // Follow-up columns
  const followUpColumns: ColumnsType<FollowUpItem> = [
    {
      title: 'ID',
      dataIndex: 'follow_up_id',
      key: 'follow_up_id',
      width: 100,
      fixed: 'left',
    },
    {
      title: 'Site',
      dataIndex: 'site_id',
      key: 'site_id',
      width: 100,
    },
    {
      title: 'CRA',
      dataIndex: 'cra_name',
      key: 'cra_name',
      width: 140,
    },
    {
      title: 'Category',
      dataIndex: 'issue_category',
      key: 'issue_category',
      width: 150,
      render: (cat) => <Tag>{cat}</Tag>,
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      width: 250,
      ellipsis: true,
      render: (text) => (
        <Tooltip title={text}>
          <Text>{text}</Text>
        </Tooltip>
      ),
    },
    {
      title: 'Priority',
      dataIndex: 'priority',
      key: 'priority',
      width: 100,
      render: (priority) => {
        const colors: Record<string, string> = { High: 'red', Medium: 'orange', Low: 'blue' }
        return <Tag color={colors[priority]}>{priority}</Tag>
      },
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 130,
      render: (status) => {
        const colors: Record<string, string> = {
          'Open': 'error',
          'In Progress': 'processing',
          'Pending Response': 'warning',
          'Resolved': 'success',
        }
        return <Tag color={colors[status]}>{status}</Tag>
      },
    },
    {
      title: 'Due Date',
      dataIndex: 'due_date',
      key: 'due_date',
      width: 110,
      render: (date) => {
        const isOverdue = dayjs(date).isBefore(dayjs())
        return <Text type={isOverdue ? 'danger' : undefined}>{date}</Text>
      },
    },
    {
      title: 'Days Open',
      dataIndex: 'days_open',
      key: 'days_open',
      width: 100,
      sorter: (a, b) => a.days_open - b.days_open,
      render: (days) => (
        <Tag color={days > 14 ? 'red' : days > 7 ? 'orange' : 'green'}>
          {days} days
        </Tag>
      ),
    },
  ]

  return (
    <div className="cra-activity-page">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: 'CRA Activity' },
        ]}
        style={{ marginBottom: 16 }}
      />

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0, color: 'var(--gray-800)' }}>
            CRA Activity Dashboard
          </Title>
          <Text type="secondary">
            CRA Performance Metrics, Monitoring Visits, Follow-up Tracker
          </Text>
        </Space>
      </div>

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Total CRAs"
              value={metrics.totalCRAs}
              prefix={<TeamOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Avg Performance"
              value={metrics.avgPerformance}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
              prefix={<TrophyOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Total Visits"
              value={metrics.totalVisits}
              prefix={<CalendarOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Completed Visits"
              value={metrics.completedVisits}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Overdue Visits"
              value={metrics.overdueVisits}
              valueStyle={{ color: '#f5222d' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Open Follow-ups"
              value={metrics.openFollowUps}
              valueStyle={{ color: '#fa8c16' }}
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
            key: 'performance',
            label: (
              <span>
                <TrophyOutlined />
                CRA Performance
              </span>
            ),
            children: (
              <Row gutter={[16, 16]}>
                <Col xs={24}>
                  <Card size="small">
                    <Table
                      columns={craColumns}
                      dataSource={mockCRAPerformance}
                      rowKey="cra_id"
                      loading={loading}
                      scroll={{ x: 1300 }}
                      pagination={false}
                      size="small"
                    />
                  </Card>
                </Col>
                <Col xs={24} md={12}>
                  <Card title="Performance Comparison" size="small">
                    <Column
                      data={mockCRAPerformance.map(c => ({
                        cra: c.cra_name.split(' ')[0],
                        metric: 'SDV Rate',
                        value: c.sdv_completion_rate,
                      })).concat(mockCRAPerformance.map(c => ({
                        cra: c.cra_name.split(' ')[0],
                        metric: 'Resolution Rate',
                        value: c.issue_resolution_rate,
                      })))}
                      xField="cra"
                      yField="value"
                      seriesField="metric"
                      isGroup
                      height={250}
                      color={['#1890ff', '#52c41a']}
                      legend={{ position: 'top' }}
                    />
                  </Card>
                </Col>
                <Col xs={24} md={12}>
                  <Card title="Visits by Type" size="small">
                    <Column
                      data={metrics.visitsByType}
                      xField="type"
                      yField="count"
                      height={250}
                      color="#722ed1"
                      label={{
                        position: 'middle',
                        style: { fill: '#fff' },
                      }}
                    />
                  </Card>
                </Col>
              </Row>
            ),
          },
          {
            key: 'visits',
            label: (
              <span>
                <CalendarOutlined />
                Monitoring Visits
                <Badge count={metrics.overdueVisits} style={{ marginLeft: 8, backgroundColor: '#f5222d' }} />
              </span>
            ),
            children: (
              <>
                <Card size="small" style={{ marginBottom: 16 }}>
                  <Row gutter={[16, 16]} align="middle">
                    <Col xs={24} md={6}>
                      <Input
                        placeholder="Search visits..."
                        prefix={<SearchOutlined />}
                        value={searchText}
                        onChange={(e) => setSearchText(e.target.value)}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={4}>
                      <Select
                        placeholder="CRA"
                        style={{ width: '100%' }}
                        options={mockCRAPerformance.map(c => ({ label: c.cra_name, value: c.cra_id }))}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={4}>
                      <Select
                        placeholder="Visit Type"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Initiation', value: 'Initiation' },
                          { label: 'Routine', value: 'Routine' },
                          { label: 'Interim', value: 'Interim' },
                          { label: 'Close-out', value: 'Close-out' },
                          { label: 'For Cause', value: 'For Cause' },
                        ]}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={4}>
                      <Select
                        placeholder="Status"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Scheduled', value: 'Scheduled' },
                          { label: 'In Progress', value: 'In Progress' },
                          { label: 'Completed', value: 'Completed' },
                          { label: 'Report Pending', value: 'Report Pending' },
                          { label: 'Overdue', value: 'Overdue' },
                        ]}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={6}>
                      <Space>
                        <Button icon={<ReloadOutlined />}>Refresh</Button>
                        <Button icon={<DownloadOutlined />}>Export</Button>
                      </Space>
                    </Col>
                  </Row>
                </Card>
                <Card size="small">
                  <Table
                    columns={visitColumns}
                    dataSource={mockMonitoringVisits}
                    rowKey="visit_id"
                    loading={loading}
                    scroll={{ x: 1400, y: 450 }}
                    pagination={{
                      pageSize: 20,
                      showSizeChanger: true,
                      showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} visits`,
                    }}
                    size="small"
                    rowClassName={(record) => record.status === 'Overdue' ? 'row-overdue' : ''}
                  />
                </Card>
              </>
            ),
          },
          {
            key: 'followups',
            label: (
              <span>
                <FileSearchOutlined />
                Follow-up Tracker
                <Badge count={metrics.highPriorityFollowUps} style={{ marginLeft: 8 }} />
              </span>
            ),
            children: (
              <Card size="small">
                <Table
                  columns={followUpColumns}
                  dataSource={mockFollowUpItems}
                  rowKey="follow_up_id"
                  loading={loading}
                  scroll={{ x: 1400, y: 500 }}
                  pagination={{
                    pageSize: 25,
                    showSizeChanger: true,
                    showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} follow-ups`,
                  }}
                  size="small"
                  rowClassName={(record) => record.priority === 'High' && record.status !== 'Resolved' ? 'row-high-priority' : ''}
                />
              </Card>
            ),
          },
        ]}
      />

      <style>{`
        .row-overdue {
          background-color: rgba(255, 77, 79, 0.15);
        }
        .row-overdue:hover > td {
          background-color: rgba(255, 77, 79, 0.25) !important;
        }
        .row-high-priority {
          background-color: rgba(250, 173, 20, 0.15);
        }
        .row-high-priority:hover > td {
          background-color: rgba(250, 173, 20, 0.25) !important;
        }
      `}</style>
    </div>
  )
}
