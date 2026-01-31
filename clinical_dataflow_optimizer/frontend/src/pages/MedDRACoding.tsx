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
  Avatar,
  Alert,
} from 'antd'
import {
  HomeOutlined,
  MedicineBoxOutlined,
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
import { useQuery } from '@tanstack/react-query'
import type { ColumnsType } from 'antd/es/table'
import type { MedDRACoding } from '../types'
import dayjs from 'dayjs'
import { codingApi } from '../api'
import {
  PieChart,
  Pie,
  Cell,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
} from 'recharts'

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8']

const { Title, Text } = Typography

// Mock data for development/fallback
const MOCK_DATA: MedDRACoding[] = [
  {
    id: '1',
    subject_id: 'SUBJ-001',
    site_id: 'Site-01',
    term_type: 'Adverse Event',
    verbatim_term: 'Headache and dizziness',
    meddra_coded_term: 'Headache',
    preferred_term: 'Headache',
    system_organ_class: 'Nervous system disorders',
    coding_status: 'Approved',
    coder_assigned: 'John Smith',
    date_term_entered: '2024-01-15',
    days_pending_coding: 0,
  },
  {
    id: '2',
    subject_id: 'SUBJ-002',
    site_id: 'Site-01',
    term_type: 'Adverse Event',
    verbatim_term: 'Nausea',
    meddra_coded_term: 'Nausea',
    preferred_term: 'Nausea',
    system_organ_class: 'Gastrointestinal disorders',
    coding_status: 'Coded',
    coder_assigned: 'Jane Doe',
    date_term_entered: '2024-01-18',
    days_pending_coding: 5,
  },
  {
    id: '3',
    subject_id: 'SUBJ-003',
    site_id: 'Site-02',
    term_type: 'Medical History',
    verbatim_term: 'High blood pressure',
    meddra_coded_term: null,
    preferred_term: null,
    system_organ_class: null,
    coding_status: 'Uncoded',
    coder_assigned: 'John Smith',
    date_term_entered: '2024-01-20',
    days_pending_coding: 8,
  },
  {
    id: '4',
    subject_id: 'SUBJ-004',
    site_id: 'Site-02',
    term_type: 'Adverse Event',
    verbatim_term: 'Stomach pain',
    meddra_coded_term: null,
    preferred_term: null,
    system_organ_class: null,
    coding_status: 'Pending Review',
    coder_assigned: 'Jane Doe',
    date_term_entered: '2024-01-22',
    days_pending_coding: 12,
  }
]

import { normalizeArray } from '../utils/data'

export default function MedDRACodingPage() {
  const { data: apiData, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['coding', 'meddra'],
    queryFn: () => codingApi.getMedDRA(),
  })

  // Use API data or fallback to MOCK if empty/error (for dev robustness)
  // Now using robust normalization
  const normalizedApiData = normalizeArray(apiData)
  const codingData = normalizedApiData.length > 0 ? normalizedApiData : MOCK_DATA

  const [activeTab, setActiveTab] = useState('terms')
  const [searchText, setSearchText] = useState('')
  const [filters, setFilters] = useState({
    termType: [] as string[],
    sites: [] as string[],
    status: [] as string[],
    soc: [] as string[],
  })

  // Filter data
  const filteredData = useMemo(() => {
    return codingData.filter((item) => {
      // Search filter
      if (searchText) {
        const searchLower = searchText.toLowerCase()
        const matchesSubject = item.subject_id?.toLowerCase().includes(searchLower)
        const matchesTerm = item.verbatim_term?.toLowerCase().includes(searchLower)
        if (!matchesSubject && !matchesTerm) {
          return false
        }
      }

      // Term type filter
      if (filters.termType.length > 0 && !filters.termType.includes(item.term_type)) {
        return false
      }

      // Site filter
      if (filters.sites.length > 0 && !filters.sites.includes(item.site_id)) {
        return false
      }

      // Status filter
      if (filters.status.length > 0 && !filters.status.includes(item.coding_status || 'Uncoded')) {
        return false
      }

      return true
    })
  }, [filters, searchText, codingData])

  // Calculate metrics
  const codingMetrics = useMemo(() => {
    const total = codingData.length
    const uncoded = codingData.filter((d) => d.coding_status === 'Uncoded' || !d.coding_status).length
    const pendingReview = codingData.filter((d) => d.coding_status === 'Pending Review').length
    const coded = codingData.filter((d) => d.coding_status === 'Coded').length
    const approved = codingData.filter((d) => d.coding_status === 'Approved').length

    const codedItems = codingData.filter((d) => d.coding_status !== 'Uncoded' && d.coding_status)
    const avgTurnaround = codedItems.length > 0
      ? codedItems.reduce((sum, d) => sum + (d.days_pending_coding || 0), 0) / codedItems.length
      : 0

    const bySoc = codingData
      .filter((d) => d.system_organ_class)
      .reduce((acc, d) => {
        const soc = d.system_organ_class!
        acc[soc] = (acc[soc] || 0) + 1
        return acc
      }, {} as Record<string, number>)

    const bySite = codingData
      .filter((d) => d.coding_status === 'Uncoded' || !d.coding_status)
      .reduce((acc, d) => {
        acc[d.site_id] = (acc[d.site_id] || 0) + 1
        return acc
      }, {} as Record<string, number>)

    // Synthetic Trend Data
    const trendData = [
      { week: 'Week 1', coded: Math.floor(total * 0.1), approved: Math.floor(total * 0.05) },
      { week: 'Week 2', coded: Math.floor(total * 0.2), approved: Math.floor(total * 0.1) },
      { week: 'Week 3', coded: Math.floor(total * 0.35), approved: Math.floor(total * 0.2) },
      { week: 'Week 4', coded: Math.floor(total * 0.5), approved: Math.floor(total * 0.3) },
      { week: 'Week 5', coded: Math.floor(total * 0.7), approved: Math.floor(total * 0.5) },
      { week: 'Week 6', coded: coded, approved: approved },
    ].flatMap(d => [
      { week: d.week, type: 'Coded', value: d.coded },
      { week: d.week, type: 'Approved', value: d.approved },
    ])

    return {
      total,
      uncoded,
      pendingReview,
      coded,
      approved,
      completionRate: total > 0 ? (((coded + approved) / total) * 100).toFixed(1) : '0',
      avgTurnaround: avgTurnaround.toFixed(1),
      bySoc: Object.entries(bySoc)
        .sort((a, b) => b[1] - a[1])
        .map(([soc, count]) => ({ name: soc, value: count })),
      bySite: Object.entries(bySite)
        .sort((a, b) => b[1] - a[1])
        .map(([site, count]) => ({ name: site, value: count })),
      trendData
    }
  }, [codingData])

  // Table columns
  const columns: ColumnsType<MedDRACoding> = [
    {
      title: 'Subject ID',
      dataIndex: 'subject_id',
      key: 'subject_id',
      width: 120,
      fixed: 'left',
      render: (text) => <a style={{ fontWeight: 500 }}>{text}</a>,
    },
    {
      title: 'Site',
      dataIndex: 'site_id',
      key: 'site_id',
      width: 100,
    },
    {
      title: 'Term Type',
      dataIndex: 'term_type',
      key: 'term_type',
      width: 120,
      render: (type) => {
        const colors: Record<string, string> = {
          'Adverse Event': 'red',
          'Medical History': 'blue',
          'Indication': 'green',
        }
        return <Tag color={colors[type] || 'default'}>{type}</Tag>
      },
    },
    {
      title: 'Verbatim Term',
      dataIndex: 'verbatim_term',
      key: 'verbatim_term',
      width: 200,
      ellipsis: true,
      render: (text) => (
        <Tooltip title={text}>
          <Text strong>{text}</Text>
        </Tooltip>
      ),
    },
    {
      title: 'MedDRA Coded Term',
      dataIndex: 'meddra_coded_term',
      key: 'meddra_coded_term',
      width: 150,
      render: (text) => text || <Text type="secondary">-</Text>,
    },
    {
      title: 'Preferred Term (PT)',
      dataIndex: 'preferred_term',
      key: 'preferred_term',
      width: 150,
      render: (text) => text || <Text type="secondary">-</Text>,
    },
    {
      title: 'System Organ Class',
      dataIndex: 'system_organ_class',
      key: 'system_organ_class',
      width: 200,
      ellipsis: true,
      render: (text) => text || <Text type="secondary">-</Text>,
    },
    {
      title: 'Status',
      dataIndex: 'coding_status',
      key: 'coding_status',
      width: 130,
      render: (status) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          'Uncoded': { color: 'error', icon: <ExclamationCircleOutlined /> },
          'Pending Review': { color: 'processing', icon: <ClockCircleOutlined /> },
          'Coded': { color: 'warning', icon: <CheckCircleOutlined /> },
          'Approved': { color: 'success', icon: <CheckCircleOutlined /> },
        }
        const statusConfig = config[status] || config['Uncoded']
        return (
          <Tag color={statusConfig.color} icon={statusConfig.icon}>
            {status || 'Uncoded'}
          </Tag>
        )
      },
    },
    {
      title: 'Coder',
      dataIndex: 'coder_assigned',
      key: 'coder_assigned',
      width: 110,
      render: (name) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          <Text>{name}</Text>
        </Space>
      ),
    },
    {
      title: 'Date Entered',
      dataIndex: 'date_term_entered',
      key: 'date_term_entered',
      width: 110,
      sorter: (a, b) => dayjs(a.date_term_entered).unix() - dayjs(b.date_term_entered).unix(),
    },
    {
      title: 'Days Pending',
      dataIndex: 'days_pending_coding',
      key: 'days_pending_coding',
      width: 110,
      sorter: (a, b) => a.days_pending_coding - b.days_pending_coding,
      render: (days, record) => {
        if (record.coding_status === 'Approved' || record.coding_status === 'Coded') {
          return <Text type="secondary">-</Text>
        }
        return (
          <Tag color={days > 14 ? 'red' : days > 7 ? 'orange' : 'green'}>
            {days} days
          </Tag>
        )
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      fixed: 'right',
      render: () => (
        <Button type="link" size="small" icon={<FileSearchOutlined />}>
          View
        </Button>
      ),
    },
  ]

  const sites = [...new Set(codingData.map((d) => d.site_id))]

  return (
    <div className="meddra-coding-page" >
      {/* Error Alert */}
      {isError && (
        <Alert
          message="API Connection Issue"
          description="Unable to connect to the backend API. Displaying sample data. Please check your API configuration."
          type="warning"
          closable
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={() => refetch()}>
              Retry
            </Button>
          }
        />
      )}

      {/* Breadcrumb */}
      < Breadcrumb
        items={
          [
            { href: '/', title: <HomeOutlined /> },
            { title: 'Coding & Reconciliation' },
            { title: 'MedDRA Coding' },
          ]}
        style={{ marginBottom: 16 }}
      />

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0, color: 'var(--gray-800)' }}>
            MedDRA Coding Dashboard
          </Title>
          <Text type="secondary">
            Medical terms requiring MedDRA coding - Adverse Events, Medical History, Indications
          </Text>
        </Space>
      </div>

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Total Terms"
              value={codingMetrics.total}
              prefix={<MedicineBoxOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Uncoded Terms"
              value={codingMetrics.uncoded}
              valueStyle={{ color: '#f5222d' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Completion Rate"
              value={codingMetrics.completionRate}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Avg Turnaround"
              value={codingMetrics.avgTurnaround}
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
            key: 'terms',
            label: (
              <span>
                <MedicineBoxOutlined />
                Coding Status
                <Badge count={codingMetrics.uncoded} style={{ marginLeft: 8 }} />
              </span>
            ),
            children: (
              <>
                {/* Filters */}
                <Card size="small" style={{ marginBottom: 16 }}>
                  <Row gutter={[16, 16]} align="middle">
                    <Col xs={24} md={5}>
                      <Input
                        placeholder="Search Subject/Term"
                        prefix={<SearchOutlined />}
                        value={searchText}
                        onChange={(e) => setSearchText(e.target.value)}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={4}>
                      <Select
                        mode="multiple"
                        placeholder="Term Type"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Adverse Event', value: 'Adverse Event' },
                          { label: 'Medical History', value: 'Medical History' },
                          { label: 'Indication', value: 'Indication' },
                        ]}
                        value={filters.termType}
                        onChange={(v) => setFilters({ ...filters, termType: v })}
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
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={4}>
                      <Select
                        mode="multiple"
                        placeholder="Status"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Uncoded', value: 'Uncoded' },
                          { label: 'Pending Review', value: 'Pending Review' },
                          { label: 'Coded', value: 'Coded' },
                          { label: 'Approved', value: 'Approved' },
                        ]}
                        value={filters.status}
                        onChange={(v) => setFilters({ ...filters, status: v })}
                        maxTagCount={1}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={7}>
                      <Space>
                        <Button icon={<ReloadOutlined />} onClick={() => refetch()}>
                          Refresh
                        </Button>
                        <Button icon={<DownloadOutlined />}>Export</Button>
                      </Space>
                    </Col>
                  </Row>
                </Card>

                {/* Table */}
                <Card size="small">
                  <Table
                    columns={columns}
                    dataSource={filteredData}
                    rowKey="id"
                    loading={isLoading}
                    scroll={{ x: 1700, y: 500 }}
                    pagination={{
                      pageSize: 25,
                      showSizeChanger: true,
                      showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} terms`,
                    }}
                    size="small"
                  />
                </Card>
              </>
            ),
          },
          {
            key: 'metrics',
            label: (
              <span>
                <WarningOutlined />
                Coding Metrics
              </span>
            ),
            children: (
              <Row gutter={[16, 16]}>
                <Col xs={24} md={8}>
                  <Card title="SOC Distribution" size="small" bodyStyle={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={codingMetrics.bySoc}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          label
                        >
                          {codingMetrics.bySoc.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <RechartsTooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
                <Col xs={24} md={8}>
                  <Card title="Uncoded Terms by Site" size="small" bodyStyle={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={codingMetrics.bySite} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="name" width={80} />
                        <RechartsTooltip />
                        <Bar dataKey="value" fill="#f5222d" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
                <Col xs={24} md={8}>
                  <Card title="Coding Progress" size="small">
                    <Space direction="vertical" style={{ width: '100%' }} size={16}>
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <Text>Approved</Text>
                          <Text strong>{codingMetrics.approved}</Text>
                        </div>
                        <Progress
                          percent={codingMetrics.total > 0 ? (codingMetrics.approved / codingMetrics.total) * 100 : 0}
                          showInfo={false}
                          strokeColor="#52c41a"
                        />
                      </div>
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <Text>Coded</Text>
                          <Text strong>{codingMetrics.coded}</Text>
                        </div>
                        <Progress
                          percent={codingMetrics.total > 0 ? (codingMetrics.coded / codingMetrics.total) * 100 : 0}
                          showInfo={false}
                          strokeColor="#faad14"
                        />
                      </div>
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <Text>Pending Review</Text>
                          <Text strong>{codingMetrics.pendingReview}</Text>
                        </div>
                        <Progress
                          percent={codingMetrics.total > 0 ? (codingMetrics.pendingReview / codingMetrics.total) * 100 : 0}
                          showInfo={false}
                          strokeColor="#1890ff"
                        />
                      </div>
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <Text>Uncoded</Text>
                          <Text strong>{codingMetrics.uncoded}</Text>
                        </div>
                        <Progress
                          percent={codingMetrics.total > 0 ? (codingMetrics.uncoded / codingMetrics.total) * 100 : 0}
                          showInfo={false}
                          strokeColor="#f5222d"
                        />
                      </div>
                    </Space>
                  </Card>
                </Col>
                <Col xs={24}>
                  <Card title="Coding Progress Timeline" size="small" bodyStyle={{ height: 250 }}>
                    <div style={{ width: '100%', height: '100%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={codingMetrics.trendData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="week" />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Line type="monotone" dataKey="value" dataKeyName="Count" stroke="#8884d8" activeDot={{ r: 8 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </Card>
                </Col>
              </Row>
            ),
          },
        ]}
      />
    </div >
  )
}