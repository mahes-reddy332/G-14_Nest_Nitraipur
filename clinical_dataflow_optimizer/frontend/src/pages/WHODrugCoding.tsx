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
import type { WHODrugCoding } from '../types'
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

// Mock data for WHO Drug coding
const mockWHODrugData: WHODrugCoding[] = Array.from({ length: 70 }, (_, i) => ({
  subject_id: `SUBJ-${String(1000 + (i % 35)).padStart(4, '0')}`,
  site_id: `SITE-${String(100 + (i % 10)).padStart(3, '0')}`,
  site_name: `Clinical Site ${100 + (i % 10)}`,
  medication_type: ['Concomitant', 'Prior', 'Protocol'][i % 3] as WHODrugCoding['medication_type'],
  verbatim_drug_name: [
    'Tylenol Extra Strength',
    'Advil Cold and Sinus',
    'Lipitor 20mg tablet',
    'Metformin ER 500mg',
    'Omeprazole 20mg capsule',
    'Lisinopril 10mg',
    'Atorvastatin calcium',
    'Aspirin low dose 81mg',
  ][i % 8],
  who_drug_coded_term: i % 4 !== 0 ? [
    'Paracetamol',
    'Ibuprofen',
    'Atorvastatin',
    'Metformin',
    'Omeprazole',
    'Lisinopril',
    'Atorvastatin',
    'Acetylsalicylic acid',
  ][i % 8] : '',
  drug_code: i % 4 !== 0 ? `DC${String(10000 + i).padStart(6, '0')}` : '',
  atc_classification: i % 4 !== 0 ? [
    'N02BE01',
    'M01AE01',
    'C10AA05',
    'A10BA02',
    'A02BC01',
    'C09AA03',
    'C10AA05',
    'B01AC06',
  ][i % 8] : '',
  coding_status: ['Uncoded', 'Pending Review', 'Coded', 'Approved'][i % 4] as WHODrugCoding['coding_status'],
  coder_assigned: ['Drug Coder A', 'Drug Coder B', 'Drug Coder C', 'Drug Coder D'][i % 4],
  date_medication_entered: dayjs().subtract(Math.floor(Math.random() * 30) + 5, 'day').format('YYYY-MM-DD'),
  date_coded: i % 4 !== 0 ? dayjs().subtract(Math.floor(Math.random() * 10), 'day').format('YYYY-MM-DD') : '',
  days_pending_coding: i % 4 === 0 ? Math.floor(Math.random() * 15) + 1 : 0,
  comments: '',
}))

export default function WHODrugCodingPage() {
  const { data: apiData, isLoading, isError, refetch } = useQuery({
    queryKey: ['coding', 'whodrug'],
    queryFn: async () => {
      try {
        return await codingApi.getWHODrug()
      } catch (err) {
        console.warn('Failed to fetch WHO Drug data, using mock data:', err)
        return mockWHODrugData
      }
    },
    initialData: mockWHODrugData,
    retry: 1,
    retryDelay: 1000,
  })

  const codingData = Array.isArray(apiData) ? apiData : mockWHODrugData
  const [activeTab, setActiveTab] = useState('medications')
  const [searchText, setSearchText] = useState('')
  const [filters, setFilters] = useState({
    medicationType: [] as string[],
    sites: [] as string[],
    status: [] as string[],
  })

  // Filter data
  const filteredData = useMemo(() => {
    const data = Array.isArray(codingData) ? codingData : []
    return data.filter((item: any) => {
      if (searchText && !item.subject_id?.toLowerCase().includes(searchText.toLowerCase()) &&
        !item.verbatim_drug_name?.toLowerCase().includes(searchText.toLowerCase())) {
        return false
      }
      if (filters.medicationType.length > 0 && !filters.medicationType.includes(item.medication_type)) {
        return false
      }
      if (filters.sites.length > 0 && !filters.sites.includes(item.site_id)) {
        return false
      }
      if (filters.status.length > 0 && !filters.status.includes(item.coding_status)) {
        return false
      }
      return true
    })
  }, [filters, searchText, codingData])

  // Calculate metrics
  const codingMetrics = useMemo(() => {
    // Type assertion to handle potential 'unknown' type from useQuery
    const data = Array.isArray(codingData) ? codingData : []
    const total = data.length
    const uncoded = data.filter((d: any) => d.coding_status === 'Uncoded' || !d.coding_status).length
    const pendingReview = data.filter((d: any) => d.coding_status === 'Pending Review').length
    const coded = data.filter((d: any) => d.coding_status === 'Coded').length
    const approved = data.filter((d: any) => d.coding_status === 'Approved').length

    const codedItems = data.filter((d: any) => d.coding_status !== 'Uncoded' && d.coding_status)
    const avgTurnaround = codedItems.length > 0
      ? codedItems.reduce((sum: number, d: any) => sum + (d.days_pending_coding || 4), 0) / codedItems.length
      : 0

    const byAtc = data
      .filter((d: any) => d.atc_classification)
      .reduce((acc: any, d: any) => {
        const atcClass = d.atc_classification.substring(0, 1)
        const atcNames: Record<string, string> = {
          'A': 'Alimentary tract',
          'B': 'Blood & forming organs',
          'C': 'Cardiovascular',
          'M': 'Musculo-skeletal',
          'N': 'Nervous system',
        }
        const name = atcNames[atcClass] || 'Other'
        acc[name] = (acc[name] || 0) + 1
        return acc
      }, {} as Record<string, number>)

    const bySite = data
      .filter((d: any) => d.coding_status === 'Uncoded' || !d.coding_status)
      .reduce((acc: any, d: any) => {
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
      byAtc: Object.entries(byAtc)
        .sort((a: any, b: any) => b[1] - a[1])
        .map(([atc, count]) => ({ name: atc, value: count })),
      bySite: Object.entries(bySite)
        .sort((a: any, b: any) => b[1] - a[1])
        .map(([site, count]) => ({ name: site, value: count })),
      trendData
    }
  }, [codingData])

  // Table columns
  const columns: ColumnsType<WHODrugCoding> = [
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
      title: 'Medication Type',
      dataIndex: 'medication_type',
      key: 'medication_type',
      width: 130,
      render: (type) => {
        const colors: Record<string, string> = {
          'Concomitant': 'blue',
          'Prior': 'purple',
          'Protocol': 'green',
        }
        return <Tag color={colors[type]}>{type}</Tag>
      },
    },
    {
      title: 'Verbatim Drug Name',
      dataIndex: 'verbatim_drug_name',
      key: 'verbatim_drug_name',
      width: 200,
      ellipsis: true,
      render: (text) => (
        <Tooltip title={text}>
          <Text strong>{text}</Text>
        </Tooltip>
      ),
    },
    {
      title: 'WHO Drug Coded Term',
      dataIndex: 'who_drug_coded_term',
      key: 'who_drug_coded_term',
      width: 150,
      render: (text) => text || <Text type="secondary">-</Text>,
    },
    {
      title: 'Drug Code',
      dataIndex: 'drug_code',
      key: 'drug_code',
      width: 120,
      render: (text) => text || <Text type="secondary">-</Text>,
    },
    {
      title: 'ATC Classification',
      dataIndex: 'atc_classification',
      key: 'atc_classification',
      width: 130,
      render: (text) => text ? <Tag color="cyan">{text}</Tag> : <Text type="secondary">-</Text>,
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
      width: 120,
      render: (name) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          <Text>{name}</Text>
        </Space>
      ),
    },
    {
      title: 'Date Entered',
      dataIndex: 'date_medication_entered',
      key: 'date_medication_entered',
      width: 110,
      sorter: (a, b) => dayjs(a.date_medication_entered).unix() - dayjs(b.date_medication_entered).unix(),
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
          <Tag color={days > 10 ? 'red' : days > 5 ? 'orange' : 'green'}>
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

  const sites = [...new Set((Array.isArray(codingData) ? codingData : []).map((d: any) => d.site_id))]

  return (
    <div className="whodrug-coding-page">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: 'Coding & Reconciliation' },
          { title: 'WHO Drug Coding' },
        ]}
        style={{ marginBottom: 16 }}
      />

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0, color: 'var(--gray-800)' }}>
            WHO Drug Coding Dashboard
          </Title>
          <Text type="secondary">
            Medications requiring WHO Drug dictionary coding - Concomitant, Prior, Protocol
          </Text>
        </Space>
      </div>

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Total Medications"
              value={codingMetrics.total}
              prefix={<MedicineBoxOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Uncoded Medications"
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
            key: 'medications',
            label: (
              <span>
                <MedicineBoxOutlined />
                Medication Coding
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
                        placeholder="Search Subject/Drug"
                        prefix={<SearchOutlined />}
                        value={searchText}
                        onChange={(e) => setSearchText(e.target.value)}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={4}>
                      <Select
                        mode="multiple"
                        placeholder="Medication Type"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Concomitant', value: 'Concomitant' },
                          { label: 'Prior', value: 'Prior' },
                          { label: 'Protocol', value: 'Protocol' },
                        ]}
                        value={filters.medicationType}
                        onChange={(v) => setFilters({ ...filters, medicationType: v })}
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
                    dataSource={filteredData}
                    rowKey="id"
                    loading={isLoading}
                    scroll={{ x: 1600, y: 500 }}
                    pagination={{
                      pageSize: 25,
                      showSizeChanger: true,
                      showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} medications`,
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
                  <Card title="ATC Classification Distribution" size="small" bodyStyle={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={codingMetrics.byAtc}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          label
                        >
                          {codingMetrics.byAtc.map((entry, index) => (
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
                  <Card title="Uncoded Medications by Site" size="small" bodyStyle={{ height: 300 }}>
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
                          percent={(codingMetrics.approved / codingMetrics.total) * 100}
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
                          percent={(codingMetrics.coded / codingMetrics.total) * 100}
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
                          percent={(codingMetrics.pendingReview / codingMetrics.total) * 100}
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
                          percent={(codingMetrics.uncoded / codingMetrics.total) * 100}
                          showInfo={false}
                          strokeColor="#f5222d"
                        />
                      </div>
                    </Space>
                  </Card>
                </Col>
                <Col xs={24}>
                  <Card title="Coding Progress Over Time" size="small">
                    <Card title="Coding Progress Over Time" size="small" bodyStyle={{ height: 250 }}>
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
