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
  Tooltip as AntTooltip,
  Progress,
  Statistic,
  Badge,
  Tabs,
  List,
  Avatar,
  theme,
} from 'antd'
import {
  HomeOutlined,
  ScheduleOutlined,
  WarningOutlined,
  ClockCircleOutlined,
  SearchOutlined,
  DownloadOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  SendOutlined,
  CheckCircleOutlined,
  UserOutlined,
  EnvironmentOutlined,
} from '@ant-design/icons'
import { Column } from '@ant-design/charts'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts'
import type { ColumnsType } from 'antd/es/table'
import type { MissingVisit } from '../types'
import dayjs from 'dayjs'

const { Title, Text } = Typography

// Mock data for missing visits
const mockMissingVisits: MissingVisit[] = Array.from({ length: 80 }, (_, i) => ({
  subject_id: `SUBJ-${String(1000 + (i % 40)).padStart(4, '0')}`,
  site_id: `SITE-${String(100 + (i % 12)).padStart(3, '0')}`,
  site_name: `Clinical Site ${100 + (i % 12)}`,
  visit_name: `Visit ${(i % 10) + 1}`,
  visit_number: (i % 10) + 1,
  projected_visit_date: dayjs().subtract(Math.floor(Math.random() * 60), 'day').format('YYYY-MM-DD'),
  days_overdue: Math.floor(Math.random() * 60),
  visit_type: ['Screening', 'Baseline', 'Follow-up', 'End of Study'][i % 4] as MissingVisit['visit_type'],
  last_contact_date: dayjs().subtract(Math.floor(Math.random() * 14), 'day').format('YYYY-MM-DD'),
  cra_assigned: ['John Smith', 'Sarah Johnson', 'Michael Brown', 'Emily Davis'][i % 4],
  follow_up_status: ['Pending', 'In Progress', 'Contacted', 'Resolved'][i % 4] as MissingVisit['follow_up_status'],
}))

const PIE_COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

export default function VisitManagement() {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('missing')
  const [searchText, setSearchText] = useState('')
  const [filters, setFilters] = useState({
    visitType: [] as string[],
    sites: [] as string[],
    status: [] as string[],
    overdueRange: '' as string,
  })

  // Filter data
  const filteredVisits = useMemo(() => {
    return mockMissingVisits.filter(item => {
      if (searchText && !item.subject_id.toLowerCase().includes(searchText.toLowerCase())) {
        return false
      }
      if (filters.visitType.length > 0 && !filters.visitType.includes(item.visit_type)) {
        return false
      }
      if (filters.sites.length > 0 && !filters.sites.includes(item.site_id)) {
        return false
      }
      if (filters.status.length > 0 && !filters.status.includes(item.follow_up_status)) {
        return false
      }
      if (filters.overdueRange) {
        if (filters.overdueRange === '<15' && item.days_overdue >= 15) return false
        if (filters.overdueRange === '15-30' && (item.days_overdue < 15 || item.days_overdue > 30)) return false
        if (filters.overdueRange === '>30' && item.days_overdue <= 30) return false
      }
      return true
    })
  }, [filters, searchText])

  // Calculate metrics
  const complianceMetrics = useMemo(() => {
    const total = mockMissingVisits.length
    const avgOverdue = mockMissingVisits.reduce((sum, v) => sum + v.days_overdue, 0) / total || 0
    const completed = mockMissingVisits.filter(v => v.follow_up_status === 'Resolved').length

    const byType = mockMissingVisits.reduce((acc, v) => {
      acc[v.visit_type] = (acc[v.visit_type] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const bySite = mockMissingVisits.reduce((acc, v) => {
      acc[v.site_id] = (acc[v.site_id] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const sitesWithOver5 = Object.entries(bySite).filter(([_, count]) => count > 5).length
    const subjectsWithOver3 = new Set(
      mockMissingVisits.filter((_, i) => mockMissingVisits.filter(v => v.subject_id === mockMissingVisits[i].subject_id).length > 3)
        .map(v => v.subject_id)
    ).size

    return {
      total,
      avgOverdue: avgOverdue.toFixed(1),
      complianceRate: (((100 - total) / 100) * 100).toFixed(1), // Mock calculation
      sitesWithOver5,
      subjectsWithOver3,
      byType,
      bySite: Object.entries(bySite)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([site, count]) => ({ site, count })),
    }
  }, [])

  // Chart data
  const visitTypeData = Object.entries(complianceMetrics.byType).map(([name, value]) => ({
    name,
    value,
  }))

  const siteComparisonData = complianceMetrics.bySite

  // Table columns
  const columns: ColumnsType<MissingVisit> = [
    {
      title: 'Subject ID',
      dataIndex: 'subject_id',
      key: 'subject_id',
      width: 130,
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
        <AntTooltip title={record.site_name}>
          <span>{text}</span>
        </AntTooltip>
      ),
    },
    {
      title: 'Visit',
      dataIndex: 'visit_name',
      key: 'visit_name',
      width: 100,
    },
    {
      title: 'Visit Type',
      dataIndex: 'visit_type',
      key: 'visit_type',
      width: 110,
      render: (type) => {
        const colors: Record<string, string> = {
          'Screening': 'blue',
          'Baseline': 'purple',
          'Follow-up': 'cyan',
          'End of Study': 'orange',
        }
        return <Tag color={colors[type]}>{type}</Tag>
      },
    },
    {
      title: 'Projected Date',
      dataIndex: 'projected_visit_date',
      key: 'projected_visit_date',
      width: 120,
      sorter: (a, b) => dayjs(a.projected_visit_date).unix() - dayjs(b.projected_visit_date).unix(),
    },
    {
      title: 'Days Overdue',
      dataIndex: 'days_overdue',
      key: 'days_overdue',
      width: 120,
      sorter: (a, b) => a.days_overdue - b.days_overdue,
      render: (days) => {
        let color = 'green'
        if (days > 30) color = 'red'
        else if (days > 15) color = 'orange'
        else if (days > 7) color = 'gold'
        return (
          <Tag color={color} style={{ minWidth: 70, textAlign: 'center' }}>
            {days} days
          </Tag>
        )
      },
    },
    {
      title: 'Last Contact',
      dataIndex: 'last_contact_date',
      key: 'last_contact_date',
      width: 110,
    },
    {
      title: 'CRA Assigned',
      dataIndex: 'cra_assigned',
      key: 'cra_assigned',
      width: 130,
      render: (name) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          <Text>{name}</Text>
        </Space>
      ),
    },
    {
      title: 'Follow-up Status',
      dataIndex: 'follow_up_status',
      key: 'follow_up_status',
      width: 130,
      render: (status) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          'Pending': { color: 'default', icon: <ClockCircleOutlined /> },
          'In Progress': { color: 'processing', icon: <ScheduleOutlined /> },
          'Contacted': { color: 'warning', icon: <SendOutlined /> },
          'Resolved': { color: 'success', icon: <CheckCircleOutlined /> },
        }
        return (
          <Tag color={config[status].color} icon={config[status].icon}>
            {status}
          </Tag>
        )
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 180,
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Button type="link" size="small" icon={<SendOutlined />}>
            Reminder
          </Button>
          {record.follow_up_status !== 'Resolved' && (
            <Button type="link" size="small" icon={<CheckCircleOutlined />}>
              Complete
            </Button>
          )}
        </Space>
      ),
    },
  ]

  const sites = [...new Set(mockMissingVisits.map(v => v.site_id))]

  // Generate heatmap data (visits by day of week and week number)
  const weeks = Array.from({ length: 12 }, (_, i) => `W${i + 1}`)
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

  // Custom Heatmap Grid Implementation
  const getHeatmapColor = (value: number) => {
    if (value === 0) return '#f0f0f005' // Very transparent for empty
    if (value < 3) return '#ffccc7' // light red
    if (value < 6) return '#ff7875' // medium red
    return '#f5222d' // dark red
  }

  // Generate a matrix for rendering
  const heatmapMatrix = useMemo(() => {
    return weeks.map(week => {
      return {
        week,
        days: days.map(day => ({
          day,
          value: Math.floor(Math.random() * 10)
        }))
      }
    })
  }, []) // Stable random data

  return (
    <div className="visit-management-page">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: 'Visit Management' },
        ]}
        style={{ marginBottom: 16 }}
      />

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0, color: 'var(--gray-800)' }}>
            Visit Management
          </Title>
          <Text type="secondary">
            Track missing visits, monitor compliance, and manage visit projections
          </Text>
        </Space>
      </div>

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Total Missing Visits"
              value={complianceMetrics.total}
              valueStyle={{ color: 'var(--neon-red)' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Avg Days Overdue"
              value={complianceMetrics.avgOverdue}
              suffix="days"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Visit Compliance Rate"
              value={complianceMetrics.complianceRate}
              suffix="%"
              valueStyle={{ color: 'var(--neon-green)' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Sites with >5 Overdue"
              value={complianceMetrics.sitesWithOver5}
              valueStyle={{ color: 'var(--neon-yellow)' }}
              prefix={<EnvironmentOutlined />}
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
                Missing Visits
                <Badge count={complianceMetrics.total} style={{ marginLeft: 8 }} />
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
                    <Col xs={24} md={4}>
                      <Select
                        mode="multiple"
                        placeholder="Visit Type"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Screening', value: 'Screening' },
                          { label: 'Baseline', value: 'Baseline' },
                          { label: 'Follow-up', value: 'Follow-up' },
                          { label: 'End of Study', value: 'End of Study' },
                        ]}
                        value={filters.visitType}
                        onChange={(v) => setFilters({ ...filters, visitType: v })}
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
                      <Select
                        placeholder="Days Overdue"
                        style={{ width: '100%' }}
                        options={[
                          { label: '< 15 days', value: '<15' },
                          { label: '15-30 days', value: '15-30' },
                          { label: '> 30 days', value: '>30' },
                        ]}
                        value={filters.overdueRange}
                        onChange={(v) => setFilters({ ...filters, overdueRange: v })}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={4}>
                      <Select
                        mode="multiple"
                        placeholder="Status"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Pending', value: 'Pending' },
                          { label: 'In Progress', value: 'In Progress' },
                          { label: 'Contacted', value: 'Contacted' },
                          { label: 'Resolved', value: 'Resolved' },
                        ]}
                        value={filters.status}
                        onChange={(v) => setFilters({ ...filters, status: v })}
                        maxTagCount={1}
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
                </Card>

                {/* Table */}
                <Card size="small">
                  <Table
                    columns={columns}
                    dataSource={filteredVisits}
                    rowKey={(record) => `${record.subject_id}-${record.visit_name}`}
                    loading={loading}
                    scroll={{ x: 1400, y: 500 }}
                    pagination={{
                      pageSize: 25,
                      showSizeChanger: true,
                      pageSizeOptions: ['25', '50', '100'],
                      showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} visits`,
                    }}
                    size="small"
                    rowClassName={(record) => {
                      if (record.days_overdue > 30) return 'row-critical'
                      if (record.days_overdue > 15) return 'row-warning'
                      return ''
                    }}
                  />
                </Card>
              </>
            ),
          },
          {
            key: 'analytics',
            label: (
              <span>
                <ScheduleOutlined />
                Visual Analytics
              </span>
            ),
            children: (
              <Row gutter={[16, 16]}>
                <Col xs={24} md={12}>
                  <Card title="Missing Visits by Site" size="small">
                    <Column
                      data={siteComparisonData}
                      xField="site"
                      yField="count"
                      height={300}
                      color="#f5222d"
                      label={{
                        position: 'middle',
                        style: { fill: '#fff' },
                      }}
                    />
                  </Card>
                </Col>
                <Col xs={24} md={12}>
                  <Card title="Missing Visits by Type" size="small">
                    <div style={{ height: 300, width: '100%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={visitTypeData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            outerRadius={100}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {visitTypeData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1f1f2e', border: '1px solid #333', borderRadius: 8 }}
                            itemStyle={{ color: '#fff' }}
                          />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </Card>
                </Col>
                <Col xs={24}>
                  <Card title="Overdue Visits Calendar Heatmap" size="small">
                    <div style={{ overflowX: 'auto', padding: '16px 0' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '50px repeat(12, 1fr)', gap: 4, minWidth: 600 }}>
                        {/* Header Row */}
                        <div style={{}}></div>
                        {weeks.map(week => (
                          <div key={week} style={{ textAlign: 'center', fontSize: 12, color: 'var(--gray-500)', fontWeight: 600 }}>
                            {week}
                          </div>
                        ))}

                        {/* Data Rows */}
                        {days.map((day, dayIndex) => (
                          <>
                            {/* Row Label */}
                            <div key={`label-${day}`} style={{ fontSize: 12, color: 'var(--gray-500)', display: 'flex', alignItems: 'center' }}>
                              {day}
                            </div>
                            {/* Cells */}
                            {heatmapMatrix.map((weekData) => {
                              const cellData = weekData.days[dayIndex];
                              return (
                                <AntTooltip key={`${weekData.week}-${day}`} title={`${cellData.value} overdue visits on ${weekData.week} ${day}`}>
                                  <div
                                    style={{
                                      height: 24,
                                      borderRadius: 4,
                                      backgroundColor: getHeatmapColor(cellData.value),
                                      transition: 'all 0.2s',
                                      cursor: 'pointer',
                                    }}
                                  />
                                </AntTooltip>
                              )
                            })}
                          </>
                        ))}
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16, alignItems: 'center', gap: 8 }}>
                        <Text style={{ fontSize: 12 }}>Less</Text>
                        {[0, 3, 6, 9].map(val => (
                          <div key={val} style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: getHeatmapColor(val) }}></div>
                        ))}
                        <Text style={{ fontSize: 12 }}>More</Text>
                      </div>
                    </div>
                  </Card>
                </Col>
              </Row>
            ),
          },
          {
            key: 'compliance',
            label: (
              <span>
                <CheckCircleOutlined />
                Compliance Metrics
              </span>
            ),
            children: (
              <Row gutter={[16, 16]}>
                <Col xs={24} md={12}>
                  <Card title="Sites with >5 Overdue Visits" size="small">
                    <List
                      dataSource={complianceMetrics.bySite.filter(s => s.count > 5)}
                      renderItem={(item) => (
                        <List.Item>
                          <List.Item.Meta
                            avatar={<Avatar icon={<EnvironmentOutlined />} style={{ backgroundColor: '#f5222d' }} />}
                            title={item.site}
                            description={
                              <Space>
                                <Text type="danger">{item.count} overdue visits</Text>
                                <Button type="link" size="small">View Details</Button>
                              </Space>
                            }
                          />
                        </List.Item>
                      )}
                    />
                  </Card>
                </Col>
                <Col xs={24} md={12}>
                  <Card title="Subjects with >3 Overdue Visits" size="small">
                    <List
                      dataSource={[
                        { subject: 'SUBJ-1001', count: 5, site: 'SITE-101' },
                        { subject: 'SUBJ-1015', count: 4, site: 'SITE-105' },
                        { subject: 'SUBJ-1023', count: 4, site: 'SITE-108' },
                        { subject: 'SUBJ-1007', count: 3, site: 'SITE-102' },
                      ]}
                      renderItem={(item) => (
                        <List.Item>
                          <List.Item.Meta
                            avatar={<Avatar icon={<UserOutlined />} style={{ backgroundColor: '#faad14' }} />}
                            title={
                              <a>{item.subject}</a>
                            }
                            description={
                              <Space>
                                <Tag>{item.site}</Tag>
                                <Text type="danger">{item.count} overdue visits</Text>
                              </Space>
                            }
                          />
                        </List.Item>
                      )}
                    />
                  </Card>
                </Col>
                <Col xs={24}>
                  <Card title="Visit Compliance Trend (Past 6 Months)" size="small">
                    <Row gutter={16} style={{ marginBottom: 16 }}>
                      {[
                        { month: 'Aug', rate: 82 },
                        { month: 'Sep', rate: 78 },
                        { month: 'Oct', rate: 85 },
                        { month: 'Nov', rate: 88 },
                        { month: 'Dec', rate: 91 },
                        { month: 'Jan', rate: 89 },
                      ].map((item) => (
                        <Col xs={4} key={item.month}>
                          <Card size="small" style={{ textAlign: 'center' }}>
                            <Text type="secondary">{item.month}</Text>
                            <div style={{ marginTop: 8 }}>
                              <Progress
                                type="circle"
                                percent={item.rate}
                                size={60}
                                strokeColor={item.rate > 85 ? '#52c41a' : item.rate > 75 ? '#faad14' : '#f5222d'}
                              />
                            </div>
                          </Card>
                        </Col>
                      ))}
                    </Row>
                  </Card>
                </Col>
              </Row>
            ),
          },
        ]}
      />

      <style>{`
        .row-critical td {
          background-color: rgba(255, 77, 79, 0.1) !important;
        }
        .row-warning td {
          background-color: rgba(250, 173, 20, 0.1) !important;
        }
      `}</style>
    </div>
  )
}
