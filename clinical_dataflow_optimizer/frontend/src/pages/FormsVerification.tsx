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
  Alert,
} from 'antd'
import {
  HomeOutlined,
  FileProtectOutlined,
  WarningOutlined,
  ClockCircleOutlined,
  SearchOutlined,
  DownloadOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  LockOutlined,
  UnlockOutlined,
  AuditOutlined,
  FormOutlined,
  FieldTimeOutlined,
} from '@ant-design/icons'
import { Pie, Column, Heatmap } from '@ant-design/charts'
import type { ColumnsType } from 'antd/es/table'
import type { SDVStatus, CRFTracker, FormStatus } from '../types'
import dayjs from 'dayjs'

const { Title, Text } = Typography
const { RangePicker } = DatePicker

// Mock data for SDV Status
const mockSDVData: SDVStatus[] = Array.from({ length: 50 }, (_, i) => ({
  subject_id: `SUBJ-${String(1000 + i).padStart(4, '0')}`,
  site_id: `SITE-${String(100 + (i % 10)).padStart(3, '0')}`,
  site_name: `Clinical Site ${100 + (i % 10)}`,
  visit_name: ['Screening', 'Baseline', 'Week 4', 'Week 8', 'Week 12', 'End of Study'][i % 6],
  form_name: ['Demographics', 'Medical History', 'Vitals', 'Lab Results', 'AE Form', 'Conmed'][i % 6],
  total_fields: Math.floor(Math.random() * 30) + 10,
  fields_verified: Math.floor(Math.random() * 25) + 5,
  sdv_percentage: Math.floor(Math.random() * 60) + 40,
  verification_status: ['Not Started', 'In Progress', 'Complete', 'Requires Re-verification'][i % 4] as SDVStatus['verification_status'],
  last_verified_date: i % 4 === 0 ? '' : dayjs().subtract(Math.floor(Math.random() * 10), 'day').format('YYYY-MM-DD'),
  verified_by: i % 4 === 0 ? '' : ['CRA Smith', 'CRA Johnson', 'CRA Williams'][i % 3],
  critical_fields_pending: Math.floor(Math.random() * 5),
}))

// Mock data for Form Status
const mockFormStatus: FormStatus[] = Array.from({ length: 80 }, (_, i) => ({
  subject_id: `SUBJ-${String(1000 + (i % 40)).padStart(4, '0')}`,
  site_id: `SITE-${String(100 + (i % 10)).padStart(3, '0')}`,
  site_name: `Clinical Site ${100 + (i % 10)}`,
  visit_name: ['Screening', 'Baseline', 'Week 4', 'Week 8', 'Week 12', 'End of Study'][i % 6],
  form_name: ['Demographics', 'Medical History', 'Vitals', 'Lab Results', 'AE Form', 'Conmed'][i % 6],
  form_status: ['Incomplete', 'Complete', 'Frozen', 'Locked', 'Signed'][i % 5] as FormStatus['form_status'],
  date_created: dayjs().subtract(Math.floor(Math.random() * 60) + 10, 'day').format('YYYY-MM-DD'),
  date_completed: i % 5 === 0 ? '' : dayjs().subtract(Math.floor(Math.random() * 30) + 5, 'day').format('YYYY-MM-DD'),
  date_frozen: i % 5 >= 2 ? dayjs().subtract(Math.floor(Math.random() * 20), 'day').format('YYYY-MM-DD') : '',
  date_locked: i % 5 >= 3 ? dayjs().subtract(Math.floor(Math.random() * 10), 'day').format('YYYY-MM-DD') : '',
  date_signed: i % 5 >= 4 ? dayjs().subtract(Math.floor(Math.random() * 5), 'day').format('YYYY-MM-DD') : '',
  has_queries: Math.random() > 0.7,
  days_since_last_update: Math.floor(Math.random() * 15),
}))

// Mock data for overdue CRFs
const mockOverdueCRFs: CRFTracker[] = Array.from({ length: 30 }, (_, i) => ({
  subject_id: `SUBJ-${String(1000 + i).padStart(4, '0')}`,
  site_id: `SITE-${String(100 + (i % 10)).padStart(3, '0')}`,
  site_name: `Clinical Site ${100 + (i % 10)}`,
  visit_name: ['Baseline', 'Week 4', 'Week 8', 'Week 12', 'End of Study'][i % 5],
  form_name: ['Lab Results', 'AE Form', 'Conmed', 'Vitals', 'ECG'][i % 5],
  expected_date: dayjs().subtract(Math.floor(Math.random() * 20) + 5, 'day').format('YYYY-MM-DD'),
  days_overdue: Math.floor(Math.random() * 20) + 1,
  priority: ['High', 'Medium', 'Low'][i % 3] as CRFTracker['priority'],
  assigned_to: ['Data Entry A', 'Data Entry B', 'Data Entry C'][i % 3],
  reminder_sent: Math.random() > 0.5,
}))

export default function FormsVerificationPage() {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('sdv')
  const [searchText, setSearchText] = useState('')
  const [filters, setFilters] = useState({
    sites: [] as string[],
    status: [] as string[],
    visits: [] as string[],
  })

  // Calculate metrics
  const metrics = useMemo(() => {
    const totalForms = mockFormStatus.length
    const frozenForms = mockFormStatus.filter(f => f.form_status === 'Frozen').length
    const lockedForms = mockFormStatus.filter(f => f.form_status === 'Locked').length
    const signedForms = mockFormStatus.filter(f => f.form_status === 'Signed').length
    const incompleteForms = mockFormStatus.filter(f => f.form_status === 'Incomplete').length

    const totalSDVFields = mockSDVData.reduce((sum, d) => sum + d.total_fields, 0)
    const verifiedFields = mockSDVData.reduce((sum, d) => sum + d.fields_verified, 0)
    const avgSDVPercent = (mockSDVData.reduce((sum, d) => sum + d.sdv_percentage, 0) / mockSDVData.length).toFixed(1)

    const overdueCRFs = mockOverdueCRFs.length
    const highPriorityOverdue = mockOverdueCRFs.filter(c => c.priority === 'High').length

    // Form status by site
    const bySite = mockFormStatus.reduce((acc, f) => {
      if (!acc[f.site_id]) {
        acc[f.site_id] = { frozen: 0, locked: 0, signed: 0, total: 0 }
      }
      acc[f.site_id].total++
      if (f.form_status === 'Frozen') acc[f.site_id].frozen++
      if (f.form_status === 'Locked') acc[f.site_id].locked++
      if (f.form_status === 'Signed') acc[f.site_id].signed++
      return acc
    }, {} as Record<string, { frozen: number; locked: number; signed: number; total: number }>)

    return {
      totalForms,
      frozenForms,
      lockedForms,
      signedForms,
      incompleteForms,
      totalSDVFields,
      verifiedFields,
      avgSDVPercent,
      overdueCRFs,
      highPriorityOverdue,
      formStatusDistribution: [
        { status: 'Signed', count: signedForms },
        { status: 'Locked', count: lockedForms },
        { status: 'Frozen', count: frozenForms },
        { status: 'Complete', count: mockFormStatus.filter(f => f.form_status === 'Complete').length },
        { status: 'Incomplete', count: incompleteForms },
      ],
      siteProgressData: Object.entries(bySite).map(([site, data]) => ({
        site,
        'Signed': data.signed,
        'Locked': data.locked,
        'Frozen': data.frozen,
      })),
    }
  }, [])

  // Filter SDV data
  const filteredSDVData = useMemo(() => {
    return mockSDVData.filter(item => {
      if (searchText && !item.subject_id.toLowerCase().includes(searchText.toLowerCase()) &&
        !item.form_name.toLowerCase().includes(searchText.toLowerCase())) {
        return false
      }
      if (filters.sites.length > 0 && !filters.sites.includes(item.site_id)) {
        return false
      }
      if (filters.status.length > 0 && !filters.status.includes(item.verification_status)) {
        return false
      }
      return true
    })
  }, [filters, searchText])

  // SDV columns
  const sdvColumns: ColumnsType<SDVStatus> = [
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
      title: 'Visit',
      dataIndex: 'visit_name',
      key: 'visit_name',
      width: 120,
    },
    {
      title: 'Form',
      dataIndex: 'form_name',
      key: 'form_name',
      width: 140,
    },
    {
      title: 'SDV Progress',
      key: 'sdv_progress',
      width: 200,
      render: (_, record) => (
        <Space>
          <Progress
            percent={record.sdv_percentage}
            size="small"
            style={{ width: 100 }}
            strokeColor={record.sdv_percentage === 100 ? '#52c41a' : record.sdv_percentage >= 50 ? '#1890ff' : '#faad14'}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.fields_verified}/{record.total_fields}
          </Text>
        </Space>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'verification_status',
      key: 'verification_status',
      width: 160,
      render: (status) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          'Not Started': { color: 'default', icon: <ClockCircleOutlined /> },
          'In Progress': { color: 'processing', icon: <FieldTimeOutlined /> },
          'Complete': { color: 'success', icon: <CheckCircleOutlined /> },
          'Requires Re-verification': { color: 'warning', icon: <ExclamationCircleOutlined /> },
        }
        return (
          <Tag color={config[status].color} icon={config[status].icon}>
            {status}
          </Tag>
        )
      },
    },
    {
      title: 'Critical Fields Pending',
      dataIndex: 'critical_fields_pending',
      key: 'critical_fields_pending',
      width: 150,
      sorter: (a, b) => a.critical_fields_pending - b.critical_fields_pending,
      render: (count) => (
        <Badge
          count={count}
          style={{ backgroundColor: count > 0 ? '#f5222d' : '#52c41a' }}
          showZero
        />
      ),
    },
    {
      title: 'Last Verified',
      dataIndex: 'last_verified_date',
      key: 'last_verified_date',
      width: 110,
      render: (date) => date || <Text type="secondary">-</Text>,
    },
    {
      title: 'Verified By',
      dataIndex: 'verified_by',
      key: 'verified_by',
      width: 120,
      render: (name) => name || <Text type="secondary">-</Text>,
    },
  ]

  // Form status columns
  const formColumns: ColumnsType<FormStatus> = [
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
      title: 'Visit',
      dataIndex: 'visit_name',
      key: 'visit_name',
      width: 120,
    },
    {
      title: 'Form',
      dataIndex: 'form_name',
      key: 'form_name',
      width: 140,
    },
    {
      title: 'Status',
      dataIndex: 'form_status',
      key: 'form_status',
      width: 120,
      render: (status) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          'Incomplete': { color: 'error', icon: <FormOutlined /> },
          'Complete': { color: 'processing', icon: <CheckCircleOutlined /> },
          'Frozen': { color: 'cyan', icon: <WarningOutlined /> },
          'Locked': { color: 'orange', icon: <LockOutlined /> },
          'Signed': { color: 'success', icon: <AuditOutlined /> },
        }
        return (
          <Tag color={config[status].color} icon={config[status].icon}>
            {status}
          </Tag>
        )
      },
    },
    {
      title: 'Created',
      dataIndex: 'date_created',
      key: 'date_created',
      width: 100,
    },
    {
      title: 'Completed',
      dataIndex: 'date_completed',
      key: 'date_completed',
      width: 100,
      render: (date) => date || <Text type="secondary">-</Text>,
    },
    {
      title: 'Frozen',
      dataIndex: 'date_frozen',
      key: 'date_frozen',
      width: 100,
      render: (date) => date || <Text type="secondary">-</Text>,
    },
    {
      title: 'Locked',
      dataIndex: 'date_locked',
      key: 'date_locked',
      width: 100,
      render: (date) => date || <Text type="secondary">-</Text>,
    },
    {
      title: 'Signed',
      dataIndex: 'date_signed',
      key: 'date_signed',
      width: 100,
      render: (date) => date || <Text type="secondary">-</Text>,
    },
    {
      title: 'Queries',
      dataIndex: 'has_queries',
      key: 'has_queries',
      width: 80,
      render: (has) => has ? <Tag color="red">Yes</Tag> : <Tag color="green">No</Tag>,
    },
  ]

  // Overdue CRF columns
  const overdueColumns: ColumnsType<CRFTracker> = [
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
      title: 'Visit',
      dataIndex: 'visit_name',
      key: 'visit_name',
      width: 120,
    },
    {
      title: 'Form',
      dataIndex: 'form_name',
      key: 'form_name',
      width: 140,
    },
    {
      title: 'Expected Date',
      dataIndex: 'expected_date',
      key: 'expected_date',
      width: 120,
    },
    {
      title: 'Days Overdue',
      dataIndex: 'days_overdue',
      key: 'days_overdue',
      width: 120,
      sorter: (a, b) => a.days_overdue - b.days_overdue,
      render: (days) => (
        <Tag color={days > 14 ? 'red' : days > 7 ? 'orange' : 'gold'}>
          {days} days
        </Tag>
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
      title: 'Assigned To',
      dataIndex: 'assigned_to',
      key: 'assigned_to',
      width: 120,
    },
    {
      title: 'Reminder Sent',
      dataIndex: 'reminder_sent',
      key: 'reminder_sent',
      width: 120,
      render: (sent) => (
        <Tag color={sent ? 'green' : 'default'}>
          {sent ? 'Yes' : 'No'}
        </Tag>
      ),
    },
  ]

  const sites = [...new Set(mockSDVData.map(d => d.site_id))]

  return (
    <div className="forms-verification-page">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: 'Forms & Verification' },
        ]}
        style={{ marginBottom: 16 }}
      />

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0, color: 'var(--gray-800)' }}>
            Forms & Verification Dashboard
          </Title>
          <Text type="secondary">
            SDV Status, Form Lifecycle (Frozen/Locked/Signed), Overdue CRF Tracking
          </Text>
        </Space>
      </div>

      {/* Alert for high priority items */}
      {metrics.highPriorityOverdue > 0 && (
        <Alert
          message={`${metrics.highPriorityOverdue} High Priority Overdue CRFs Require Immediate Attention`}
          type="error"
          showIcon
          icon={<ExclamationCircleOutlined />}
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" danger onClick={() => setActiveTab('overdue')}>
              View Overdue
            </Button>
          }
        />
      )}

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Total Forms"
              value={metrics.totalForms}
              prefix={<FileProtectOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Signed Forms"
              value={metrics.signedForms}
              valueStyle={{ color: '#52c41a' }}
              prefix={<AuditOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Locked Forms"
              value={metrics.lockedForms}
              valueStyle={{ color: '#fa8c16' }}
              prefix={<LockOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Avg SDV %"
              value={metrics.avgSDVPercent}
              suffix="%"
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Overdue CRFs"
              value={metrics.overdueCRFs}
              valueStyle={{ color: '#f5222d' }}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={4}>
          <Card size="small">
            <Statistic
              title="Incomplete Forms"
              value={metrics.incompleteForms}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<FormOutlined />}
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
            key: 'sdv',
            label: (
              <span>
                <CheckCircleOutlined />
                SDV Status
              </span>
            ),
            children: (
              <>
                {/* Filters */}
                <Card size="small" style={{ marginBottom: 16 }}>
                  <Row gutter={[16, 16]} align="middle">
                    <Col xs={24} md={6}>
                      <Input
                        placeholder="Search Subject/Form"
                        prefix={<SearchOutlined />}
                        value={searchText}
                        onChange={(e) => setSearchText(e.target.value)}
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
                    <Col xs={24} md={5}>
                      <Select
                        mode="multiple"
                        placeholder="Verification Status"
                        style={{ width: '100%' }}
                        options={[
                          { label: 'Not Started', value: 'Not Started' },
                          { label: 'In Progress', value: 'In Progress' },
                          { label: 'Complete', value: 'Complete' },
                          { label: 'Requires Re-verification', value: 'Requires Re-verification' },
                        ]}
                        value={filters.status}
                        onChange={(v) => setFilters({ ...filters, status: v })}
                        maxTagCount={1}
                        allowClear
                      />
                    </Col>
                    <Col xs={24} md={9}>
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
                    columns={sdvColumns}
                    dataSource={filteredSDVData}
                    rowKey={(record) => `${record.subject_id}-${record.visit_name}-${record.form_name}`}
                    loading={loading}
                    scroll={{ x: 1400, y: 450 }}
                    pagination={{
                      pageSize: 20,
                      showSizeChanger: true,
                      showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} records`,
                    }}
                    size="small"
                  />
                </Card>
              </>
            ),
          },
          {
            key: 'forms',
            label: (
              <span>
                <LockOutlined />
                Form Status
                <Badge count={metrics.incompleteForms} style={{ marginLeft: 8 }} />
              </span>
            ),
            children: (
              <Row gutter={[16, 16]}>
                <Col xs={24} md={8}>
                  <Card title="Form Status Distribution" size="small">
                    <Pie
                      data={metrics.formStatusDistribution}
                      angleField="count"
                      colorField="status"
                      radius={0.8}
                      innerRadius={0.5}
                      height={250}
                      label={{
                        type: 'inner',
                        content: '{value}',
                      }}
                      legend={{ position: 'bottom' }}
                      color={['#52c41a', '#fa8c16', '#13c2c2', '#1890ff', '#f5222d']}
                    />
                  </Card>
                </Col>
                <Col xs={24} md={16}>
                  <Card title="Form Status by Site" size="small">
                    <Column
                      data={metrics.siteProgressData.flatMap(d => [
                        { site: d.site, status: 'Signed', count: d['Signed'] },
                        { site: d.site, status: 'Locked', count: d['Locked'] },
                        { site: d.site, status: 'Frozen', count: d['Frozen'] },
                      ])}
                      xField="site"
                      yField="count"
                      seriesField="status"
                      isStack
                      height={250}
                      color={['#52c41a', '#fa8c16', '#13c2c2']}
                      legend={{ position: 'top-right' }}
                    />
                  </Card>
                </Col>
                <Col xs={24}>
                  <Card size="small">
                    <Table
                      columns={formColumns}
                      dataSource={mockFormStatus}
                      rowKey={(record) => `${record.subject_id}-${record.visit_name}-${record.form_name}`}
                      loading={loading}
                      scroll={{ x: 1400, y: 350 }}
                      pagination={{
                        pageSize: 15,
                        showSizeChanger: true,
                        showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} forms`,
                      }}
                      size="small"
                    />
                  </Card>
                </Col>
              </Row>
            ),
          },
          {
            key: 'overdue',
            label: (
              <span>
                <WarningOutlined />
                Overdue CRFs
                <Badge count={metrics.overdueCRFs} style={{ marginLeft: 8 }} />
              </span>
            ),
            children: (
              <Card size="small">
                <Table
                  columns={overdueColumns}
                  dataSource={mockOverdueCRFs}
                  rowKey={(record) => `${record.subject_id}-${record.visit_name}-${record.form_name}`}
                  loading={loading}
                  scroll={{ x: 1200, y: 500 }}
                  pagination={{
                    pageSize: 25,
                    showSizeChanger: true,
                    showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} overdue CRFs`,
                  }}
                  size="small"
                  rowClassName={(record) => record.priority === 'High' ? 'row-high-priority' : ''}
                />
              </Card>
            ),
          },
        ]}
      />

      <style>{`
        .row-high-priority {
          background-color: rgba(255, 77, 79, 0.15);
        }
        .row-high-priority:hover > td {
          background-color: rgba(255, 77, 79, 0.25) !important;
        }
      `}</style>
    </div>
  )
}
