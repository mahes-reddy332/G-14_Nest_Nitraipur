import { useState, useMemo, useCallback } from 'react'
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
  Dropdown,
  Switch,
  Badge,
  Statistic,
  Divider,
  message,
} from 'antd'
import {
  HomeOutlined,
  UserOutlined,
  SearchOutlined,
  DownloadOutlined,
  ReloadOutlined,
  FilterOutlined,
  SettingOutlined,
  EyeOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons'
import type { ColumnsType, TableProps } from 'antd/es/table'
import type { SubjectEDCMetrics } from '../types'
import {
  useEDCMetrics,
  useCleanPatientMetrics,
  useExportEDCMetrics,
} from '../hooks'
import { useRegions, useCountries, useSites } from '../hooks/useMetadata'
import type { EDCMetricsFilters } from '../services/api/types'
import dayjs from 'dayjs'

const { Title, Text } = Typography
const { RangePicker } = DatePicker

// Feature flag for mock data - set to true to use mock data during development
const USE_MOCK_DATA = import.meta.env.VITE_ENABLE_MOCK_DATA === 'true'

// Mock data for demonstration (used when API is not available)
const mockSubjectData: SubjectEDCMetrics[] = Array.from({ length: 100 }, (_, i) => ({
  region: ['North America', 'Europe', 'Asia Pacific', 'Latin America'][i % 4],
  country: ['USA', 'Germany', 'Japan', 'Brazil', 'UK', 'France', 'China', 'Canada'][i % 8],
  site_id: `SITE-${String(100 + (i % 20)).padStart(3, '0')}`,
  site_name: `Clinical Site ${100 + (i % 20)}`,
  subject_id: `SUBJ-${String(1000 + i).padStart(4, '0')}`,
  subject_status: ['Screened', 'Enrolled', 'Completed', 'Withdrawn', 'Screen Failed'][i % 5],
  enrollment_date: dayjs().subtract(Math.floor(Math.random() * 365), 'day').format('YYYY-MM-DD'),
  last_visit_date: dayjs().subtract(Math.floor(Math.random() * 30), 'day').format('YYYY-MM-DD'),
  visits_planned: 12,
  visits_completed: Math.floor(Math.random() * 12),
  missing_visits_count: Math.floor(Math.random() * 5),
  missing_visits_percent: Math.random() * 30,
  missing_pages_count: Math.floor(Math.random() * 10),
  missing_pages_percent: Math.random() * 20,
  open_queries_total: Math.floor(Math.random() * 15),
  data_queries: Math.floor(Math.random() * 8),
  protocol_deviation_queries: Math.floor(Math.random() * 4),
  safety_queries: Math.floor(Math.random() * 3),
  non_conformant_data_count: Math.floor(Math.random() * 5),
  sdv_percentage: Math.random() * 100,
  frozen_forms_count: Math.floor(Math.random() * 20),
  locked_forms_count: Math.floor(Math.random() * 15),
  signed_forms_count: Math.floor(Math.random() * 10),
  overdue_crfs_count: Math.floor(Math.random() * 8),
  inactivated_folders_count: Math.floor(Math.random() * 3),
  is_clean_patient: Math.random() > 0.3,
  last_update_timestamp: dayjs().subtract(Math.floor(Math.random() * 7), 'day').format('YYYY-MM-DD HH:mm:ss'),
}))

export default function EDCMetrics() {
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([])
  const [searchText, setSearchText] = useState('')
  const [filters, setFilters] = useState<EDCMetricsFilters>({
    regions: [],
    countries: [],
    sites: [],
    status: [],
    cleanOnly: false,
    page: 1,
    pageSize: 25,
  })
  const [visibleColumns, setVisibleColumns] = useState<string[]>([
    'region', 'country', 'site_id', 'subject_id', 'subject_status',
    'open_queries_total', 'missing_visits_count', 'sdv_percentage', 'is_clean_patient',
  ])
  const [pageSize, setPageSize] = useState(25)

  // API hooks
  const edcMetricsQuery = useEDCMetrics(filters)
  const cleanPatientQuery = useCleanPatientMetrics()
  const exportMutation = useExportEDCMetrics()

  // Metadata hooks for filter options
  const regionsQuery = useRegions()
  const countriesQuery = useCountries(
    filters.regions?.length === 1 ? filters.regions[0] : undefined
  )
  const sitesQuery = useSites(
    filters.countries?.length === 1 ? filters.countries[0] : undefined
  )

  // Determine data source
  const isLoading = USE_MOCK_DATA ? false : edcMetricsQuery.isLoading
  const rawData = USE_MOCK_DATA
    ? mockSubjectData
    : (edcMetricsQuery.data?.items ?? [])

  // Filter data locally for search (API already handles other filters)
  const filteredData = useMemo(() => {
    if (USE_MOCK_DATA) {
      return rawData.filter(item => {
        if (searchText && !item.subject_id.toLowerCase().includes(searchText.toLowerCase())) {
          return false
        }
        if (filters.regions && filters.regions.length > 0 && !filters.regions.includes(item.region)) {
          return false
        }
        if (filters.countries && filters.countries.length > 0 && !filters.countries.includes(item.country)) {
          return false
        }
        if (filters.sites && filters.sites.length > 0 && !filters.sites.includes(item.site_id)) {
          return false
        }
        if (filters.status && filters.status.length > 0 && !filters.status.includes(item.subject_status)) {
          return false
        }
        if (filters.cleanOnly && !item.is_clean_patient) {
          return false
        }
        return true
      })
    }
    // For API data, filter only by search text (other filters sent to API)
    return rawData.filter(item => {
      if (searchText && !item.subject_id.toLowerCase().includes(searchText.toLowerCase())) {
        return false
      }
      return true
    })
  }, [rawData, filters, searchText])

  // Get filter options - from API or mock data
  const regions = USE_MOCK_DATA
    ? [...new Set(mockSubjectData.map(d => d.region))]
    : (regionsQuery.data?.map(r => r.name) ?? [])
  const countries = USE_MOCK_DATA
    ? [...new Set(mockSubjectData.map(d => d.country))]
    : (countriesQuery.data?.map(c => c.name) ?? [])
  const sites = USE_MOCK_DATA
    ? [...new Set(mockSubjectData.map(d => d.site_id))]
    : (sitesQuery.data?.map(s => s.id) ?? [])
  const statuses = ['Screened', 'Enrolled', 'Completed', 'Withdrawn', 'Screen Failed']

  // Calculate summary metrics
  const summaryMetrics = useMemo(() => {
    const total = filteredData.length
    const clean = filteredData.filter(d => d.is_clean_patient).length
    const avgQueries = filteredData.reduce((sum, d) => sum + d.open_queries_total, 0) / total || 0
    const avgSDV = filteredData.reduce((sum, d) => sum + d.sdv_percentage, 0) / total || 0
    return {
      total,
      clean,
      cleanPercent: (clean / total * 100) || 0,
      avgQueries: avgQueries.toFixed(1),
      avgSDV: avgSDV.toFixed(1),
    }
  }, [filteredData])

  // Handlers
  const handleRefresh = useCallback(() => {
    if (!USE_MOCK_DATA) {
      edcMetricsQuery.refetch()
      cleanPatientQuery.refetch()
    }
    message.success('Data refreshed')
  }, [edcMetricsQuery, cleanPatientQuery])

  const handleExport = useCallback((format: 'excel' | 'csv' | 'pdf') => {
    if (USE_MOCK_DATA) {
      message.info('Export is not available with mock data')
      return
    }
    exportMutation.mutate(
      { filters, format },
      {
        onSuccess: () => message.success(`Export to ${format.toUpperCase()} started`),
        onError: (error) => message.error(`Export failed: ${error.message}`),
      }
    )
  }, [exportMutation, filters])

  const handleFilterChange = useCallback((key: keyof EDCMetricsFilters, value: unknown) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Enrolled': return 'green'
      case 'Completed': return 'blue'
      case 'Withdrawn': return 'red'
      case 'Screen Failed': return 'orange'
      default: return 'default'
    }
  }

  const getQuerySeverity = (count: number) => {
    if (count > 10) return 'critical'
    if (count > 5) return 'warning'
    return 'normal'
  }

  const columns: ColumnsType<SubjectEDCMetrics> = [
    {
      title: 'Region',
      dataIndex: 'region',
      key: 'region',
      sorter: (a, b) => a.region.localeCompare(b.region),
      width: 120,
    },
    {
      title: 'Country',
      dataIndex: 'country',
      key: 'country',
      sorter: (a, b) => a.country.localeCompare(b.country),
      width: 100,
    },
    {
      title: 'Site ID',
      dataIndex: 'site_id',
      key: 'site_id',
      sorter: (a, b) => a.site_id.localeCompare(b.site_id),
      width: 120,
      render: (text, record) => (
        <Tooltip title={record.site_name}>
          <a>{text}</a>
        </Tooltip>
      ),
    },
    {
      title: 'Subject ID',
      dataIndex: 'subject_id',
      key: 'subject_id',
      sorter: (a, b) => a.subject_id.localeCompare(b.subject_id),
      fixed: 'left',
      width: 130,
      render: (text) => <a style={{ fontWeight: 500 }}>{text}</a>,
    },
    {
      title: 'Status',
      dataIndex: 'subject_status',
      key: 'subject_status',
      width: 120,
      render: (status) => <Tag color={getStatusColor(status)}>{status}</Tag>,
    },
    {
      title: 'Enrollment Date',
      dataIndex: 'enrollment_date',
      key: 'enrollment_date',
      sorter: (a, b) => dayjs(a.enrollment_date).unix() - dayjs(b.enrollment_date).unix(),
      width: 130,
    },
    {
      title: 'Last Visit',
      dataIndex: 'last_visit_date',
      key: 'last_visit_date',
      sorter: (a, b) => dayjs(a.last_visit_date).unix() - dayjs(b.last_visit_date).unix(),
      width: 120,
    },
    {
      title: 'Visits (Plan/Done)',
      key: 'visits',
      width: 130,
      render: (_, record) => (
        <Space>
          <Text>{record.visits_completed}/{record.visits_planned}</Text>
          <Progress
            percent={Math.round(record.visits_completed / record.visits_planned * 100)}
            size="small"
            style={{ width: 60 }}
            showInfo={false}
          />
        </Space>
      ),
    },
    {
      title: 'Missing Visits',
      dataIndex: 'missing_visits_count',
      key: 'missing_visits_count',
      sorter: (a, b) => a.missing_visits_count - b.missing_visits_count,
      width: 120,
      render: (count, record) => (
        <Tooltip title={`${record.missing_visits_percent.toFixed(1)}% missing`}>
          <Badge
            count={count}
            showZero
            color={count > 3 ? 'red' : count > 1 ? 'orange' : 'green'}
          />
        </Tooltip>
      ),
    },
    {
      title: 'Missing Pages',
      dataIndex: 'missing_pages_count',
      key: 'missing_pages_count',
      sorter: (a, b) => a.missing_pages_count - b.missing_pages_count,
      width: 120,
      render: (count, record) => (
        <Tooltip title={`${record.missing_pages_percent.toFixed(1)}% missing`}>
          <Badge
            count={count}
            showZero
            color={count > 5 ? 'red' : count > 2 ? 'orange' : 'green'}
          />
        </Tooltip>
      ),
    },
    {
      title: 'Open Queries',
      dataIndex: 'open_queries_total',
      key: 'open_queries_total',
      sorter: (a, b) => a.open_queries_total - b.open_queries_total,
      width: 120,
      render: (count, record) => {
        const severity = getQuerySeverity(count)
        return (
          <Tooltip
            title={
              <div>
                <div>Data: {record.data_queries}</div>
                <div>Protocol: {record.protocol_deviation_queries}</div>
                <div>Safety: {record.safety_queries}</div>
              </div>
            }
          >
            <Tag
              color={severity === 'critical' ? 'red' : severity === 'warning' ? 'orange' : 'green'}
              style={{ minWidth: 40, textAlign: 'center' }}
            >
              {count}
            </Tag>
          </Tooltip>
        )
      },
    },
    {
      title: 'Non-Conformant',
      dataIndex: 'non_conformant_data_count',
      key: 'non_conformant_data_count',
      sorter: (a, b) => a.non_conformant_data_count - b.non_conformant_data_count,
      width: 120,
      render: (count) => (
        <Badge count={count} showZero color={count > 0 ? 'red' : 'green'} />
      ),
    },
    {
      title: 'SDV %',
      dataIndex: 'sdv_percentage',
      key: 'sdv_percentage',
      sorter: (a, b) => a.sdv_percentage - b.sdv_percentage,
      width: 120,
      render: (percent) => (
        <Progress
          percent={Math.round(percent)}
          size="small"
          status={percent < 50 ? 'exception' : percent < 80 ? 'active' : 'success'}
        />
      ),
    },
    {
      title: 'Frozen',
      dataIndex: 'frozen_forms_count',
      key: 'frozen_forms_count',
      sorter: (a, b) => a.frozen_forms_count - b.frozen_forms_count,
      width: 80,
    },
    {
      title: 'Locked',
      dataIndex: 'locked_forms_count',
      key: 'locked_forms_count',
      sorter: (a, b) => a.locked_forms_count - b.locked_forms_count,
      width: 80,
    },
    {
      title: 'Signed',
      dataIndex: 'signed_forms_count',
      key: 'signed_forms_count',
      sorter: (a, b) => a.signed_forms_count - b.signed_forms_count,
      width: 80,
    },
    {
      title: 'Overdue CRFs',
      dataIndex: 'overdue_crfs_count',
      key: 'overdue_crfs_count',
      sorter: (a, b) => a.overdue_crfs_count - b.overdue_crfs_count,
      width: 110,
      render: (count) => (
        <Badge count={count} showZero color={count > 5 ? 'red' : count > 0 ? 'orange' : 'green'} />
      ),
    },
    {
      title: 'Clean Patient',
      dataIndex: 'is_clean_patient',
      key: 'is_clean_patient',
      width: 110,
      fixed: 'right',
      render: (isClean) => (
        isClean ? (
          <Tag icon={<CheckCircleOutlined />} color="success">Clean</Tag>
        ) : (
          <Tag icon={<CloseCircleOutlined />} color="error">Not Clean</Tag>
        )
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 80,
      fixed: 'right',
      render: (_, record) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => console.log('View details for', record.subject_id)}
        >
          View
        </Button>
      ),
    },
  ]

  const visibleColumnsList = columns.filter(col =>
    col.key === 'actions' || visibleColumns.includes(col.key as string)
  )

  const rowSelection = {
    selectedRowKeys,
    onChange: (keys: React.Key[]) => setSelectedRowKeys(keys),
  }

  const exportItems = [
    { key: 'excel', label: 'Export to Excel' },
    { key: 'csv', label: 'Export to CSV' },
    { key: 'pdf', label: 'Export to PDF' },
  ]

  return (
    <div className="edc-metrics-page">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: 'Patient & Site Metrics' },
          { title: 'EDC Metrics' },
        ]}
        style={{ marginBottom: 16 }}
      />

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0, color: 'var(--gray-800)' }}>
            Patient & Site Metrics (EDC)
          </Title>
          <Text type="secondary">
            Comprehensive subject-level data from EDC system with filtering and export capabilities
          </Text>
        </Space>
      </div>

      {/* Summary Cards */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Total Subjects"
              value={summaryMetrics.total}
              prefix={<UserOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Clean Patients"
              value={summaryMetrics.clean}
              suffix={`(${summaryMetrics.cleanPercent.toFixed(1)}%)`}
              valueStyle={{ color: '#3f8600' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Avg Open Queries"
              value={summaryMetrics.avgQueries}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Avg SDV %"
              value={summaryMetrics.avgSDV}
              suffix="%"
              prefix={<InfoCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} md={4}>
            <Select
              mode="multiple"
              placeholder="Region"
              style={{ width: '100%' }}
              options={regions.map(r => ({ label: r, value: r }))}
              value={filters.regions}
              onChange={(v) => setFilters({ ...filters, regions: v })}
              allowClear
              maxTagCount={1}
            />
          </Col>
          <Col xs={24} md={4}>
            <Select
              mode="multiple"
              placeholder="Country"
              style={{ width: '100%' }}
              options={countries.map(c => ({ label: c, value: c }))}
              value={filters.countries}
              onChange={(v) => setFilters({ ...filters, countries: v })}
              allowClear
              maxTagCount={1}
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
              allowClear
              maxTagCount={1}
              showSearch
            />
          </Col>
          <Col xs={24} md={4}>
            <Select
              mode="multiple"
              placeholder="Status"
              style={{ width: '100%' }}
              options={statuses.map(s => ({ label: s, value: s }))}
              value={filters.status}
              onChange={(v) => setFilters({ ...filters, status: v })}
              allowClear
              maxTagCount={1}
            />
          </Col>
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
            <Space>
              <Switch
                checkedChildren="Clean Only"
                unCheckedChildren="All"
                checked={filters.cleanOnly}
                onChange={(v) => setFilters({ ...filters, cleanOnly: v })}
              />
              <Button
                icon={<FilterOutlined />}
                onClick={() => setFilters({ regions: [], countries: [], sites: [], status: [], cleanOnly: false, page: 1, pageSize: 25 })}
              >
                Clear
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Data Table */}
      <Card
        title={
          <Space>
            <Text strong>Subject Level Metrics</Text>
            <Badge count={filteredData.length} showZero color="blue" />
          </Space>
        }
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={handleRefresh}>
              Refresh
            </Button>
            <Dropdown
              menu={{
                items: exportItems,
                onClick: ({ key }) => handleExport(key as 'excel' | 'csv' | 'pdf'),
              }}
            >
              <Button icon={<DownloadOutlined />}>Export</Button>
            </Dropdown>
            <Dropdown
              menu={{
                items: columns
                  .filter(c => c.key !== 'actions')
                  .map(c => ({
                    key: c.key as string,
                    label: c.title as string,
                  })),
                selectable: true,
                multiple: true,
                selectedKeys: visibleColumns,
                onSelect: ({ selectedKeys }) => setVisibleColumns(selectedKeys),
                onDeselect: ({ selectedKeys }) => setVisibleColumns(selectedKeys),
              }}
            >
              <Button icon={<SettingOutlined />}>Columns</Button>
            </Dropdown>
          </Space>
        }
      >
        <Table
          columns={visibleColumnsList}
          dataSource={filteredData}
          rowKey="subject_id"
          loading={isLoading || exportMutation.isPending}
          rowSelection={rowSelection}
          scroll={{ x: 1800, y: 600 }}
          pagination={{
            pageSize,
            pageSizeOptions: ['25', '50', '100'],
            showSizeChanger: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} subjects`,
            onShowSizeChange: (_, size) => setPageSize(size),
          }}
          size="small"
          rowClassName={(record) => {
            if (record.open_queries_total > 10 || record.missing_visits_percent > 20) {
              return 'row-critical'
            }
            if (record.open_queries_total > 5 || record.missing_visits_percent > 10) {
              return 'row-warning'
            }
            if (record.is_clean_patient) {
              return 'row-clean'
            }
            return ''
          }}
        />
      </Card>

      <style>{`
        .row-critical {
          background-color: rgba(255, 51, 51, 0.15) !important;
        }
        .row-critical:hover > td {
          background-color: rgba(255, 51, 51, 0.25) !important;
        }
        .row-warning {
          background-color: rgba(252, 238, 10, 0.15) !important;
        }
        .row-warning:hover > td {
          background-color: rgba(252, 238, 10, 0.25) !important;
        }
        .row-clean {
          background-color: rgba(0, 255, 153, 0.1) !important;
        }
        .row-clean:hover > td {
          background-color: rgba(0, 255, 153, 0.2) !important;
        }
      `}</style>
    </div>
  )
}
