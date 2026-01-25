import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
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
  Input,
  Select,
  Button,
  Drawer,
  Descriptions,
  List,
  Timeline,
  Tooltip,
} from 'antd'
import {
  HomeOutlined,
  UserOutlined,
  SearchOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { patientsApi } from '../api'
import { useStore } from '../store'
import type { Patient, CleanPatientStatus, BlockingFactor } from '../types'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text } = Typography
const { Search } = Input

const statusColors = {
  clean: 'green',
  dirty: 'red',
  'at-risk': 'orange',
}

const severityColors = {
  critical: 'red',
  high: 'orange',
  medium: 'gold',
  low: 'green',
}

export default function Patients() {
  const { patientId } = useParams()
  const navigate = useNavigate()
  const { selectedStudyId } = useStore()
  
  const [searchText, setSearchText] = useState('')
  const [statusFilter, setStatusFilter] = useState<string | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)

  // Fetch patients
  const { data: patients = [], isLoading } = useQuery({
    queryKey: ['patients', selectedStudyId, statusFilter],
    queryFn: () => patientsApi.getAll({
      study_id: selectedStudyId || undefined,
      status: statusFilter || undefined,
    }),
  })

  // Fetch clean status for selected patient
  const { data: cleanStatus } = useQuery({
    queryKey: ['patientCleanStatus', selectedPatient?.patient_id],
    queryFn: () => patientsApi.getCleanStatus(selectedPatient!.patient_id),
    enabled: !!selectedPatient,
  })

  // Fetch dirty patients count
  const { data: dirtyPatients = [] } = useQuery({
    queryKey: ['dirtyPatients', selectedStudyId],
    queryFn: () => patientsApi.getDirtyPatients(selectedStudyId || undefined),
  })

  // Filter patients by search text
  const filteredPatients = patients.filter(p =>
    p.patient_id.toLowerCase().includes(searchText.toLowerCase()) ||
    p.site_id?.toLowerCase().includes(searchText.toLowerCase())
  )

  const handleViewPatient = (patient: Patient) => {
    setSelectedPatient(patient)
    setDrawerOpen(true)
  }

  const columns: ColumnsType<Patient> = [
    {
      title: 'Patient ID',
      dataIndex: 'patient_id',
      key: 'patient_id',
      render: (text, record) => (
        <a onClick={() => handleViewPatient(record)}>{text}</a>
      ),
      sorter: (a, b) => a.patient_id.localeCompare(b.patient_id),
    },
    {
      title: 'Study',
      dataIndex: 'study_id',
      key: 'study_id',
      render: (text) => <Tag>{text}</Tag>,
    },
    {
      title: 'Site',
      dataIndex: 'site_id',
      key: 'site_id',
    },
    {
      title: 'Status',
      key: 'clean_status',
      render: (_, record) => (
        <Space>
          {record.is_clean ? (
            <Tag icon={<CheckCircleOutlined />} color="success">Clean</Tag>
          ) : (
            <Tag icon={<CloseCircleOutlined />} color="error">Dirty</Tag>
          )}
        </Space>
      ),
      filters: [
        { text: 'Clean', value: 'clean' },
        { text: 'Dirty', value: 'dirty' },
      ],
      onFilter: (value, record) => 
        value === 'clean' ? record.is_clean : !record.is_clean,
    },
    {
      title: 'Cleanliness Score',
      dataIndex: 'cleanliness_score',
      key: 'cleanliness_score',
      render: (score) => (
        <Progress
          percent={Math.round(score || 0)}
          size="small"
          status={score >= 80 ? 'success' : score >= 60 ? 'normal' : 'exception'}
        />
      ),
      sorter: (a, b) => (a.cleanliness_score || 0) - (b.cleanliness_score || 0),
    },
    {
      title: 'Visits',
      dataIndex: 'visit_count',
      key: 'visit_count',
      sorter: (a, b) => (a.visit_count || 0) - (b.visit_count || 0),
    },
    {
      title: 'Open Queries',
      dataIndex: 'query_count',
      key: 'query_count',
      render: (count) => (
        <span style={{ color: count > 0 ? '#ff4d4f' : '#52c41a' }}>
          {count || 0}
        </span>
      ),
      sorter: (a, b) => (a.query_count || 0) - (b.query_count || 0),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Button type="link" size="small" onClick={() => handleViewPatient(record)}>
          View Details
        </Button>
      ),
    },
  ]

  // Blocking factor renderer
  const renderBlockingFactor = (factor: BlockingFactor) => (
    <List.Item>
      <List.Item.Meta
        avatar={
          <Tag color={severityColors[factor.severity]}>
            {factor.severity}
          </Tag>
        }
        title={factor.factor_type}
        description={
          <>
            <Text>{factor.description}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              Domain: {factor.domain} | Action: {factor.resolution_action}
            </Text>
          </>
        }
      />
    </List.Item>
  )

  return (
    <div>
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: <><UserOutlined /> Patients</> },
        ]}
        style={{ marginBottom: 16 }}
      />

      <Title level={2}>Patients</Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
        Patient cleanliness status and data quality overview
      </Text>

      {/* Summary Stats */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Patients"
              value={patients.length}
              prefix={<UserOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Clean Patients"
              value={patients.filter(p => p.is_clean).length}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Dirty Patients"
              value={dirtyPatients.length}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<CloseCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Clean Rate"
              value={patients.length > 0 
                ? ((patients.filter(p => p.is_clean).length / patients.length) * 100).toFixed(1)
                : 0
              }
              suffix="%"
              valueStyle={{ 
                color: patients.filter(p => p.is_clean).length / patients.length >= 0.8 
                  ? '#52c41a' : '#faad14' 
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      <Card style={{ marginBottom: 24 }}>
        <Space wrap>
          <Search
            placeholder="Search patients..."
            allowClear
            onSearch={setSearchText}
            onChange={(e) => setSearchText(e.target.value)}
            style={{ width: 250 }}
          />
          <Select
            placeholder="Status"
            allowClear
            style={{ width: 150 }}
            onChange={setStatusFilter}
            options={[
              { value: 'clean', label: 'Clean' },
              { value: 'dirty', label: 'Dirty' },
            ]}
          />
        </Space>
      </Card>

      {/* Patients Table */}
      <Card>
        <Table
          columns={columns}
          dataSource={filteredPatients}
          rowKey="patient_id"
          loading={isLoading}
          pagination={{ pageSize: 15, showSizeChanger: true }}
        />
      </Card>

      {/* Patient Detail Drawer */}
      <Drawer
        title={`Patient: ${selectedPatient?.patient_id}`}
        placement="right"
        width={600}
        open={drawerOpen}
        onClose={() => {
          setDrawerOpen(false)
          setSelectedPatient(null)
        }}
      >
        {selectedPatient && (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            {/* Patient Info */}
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="Patient ID">{selectedPatient.patient_id}</Descriptions.Item>
              <Descriptions.Item label="Study">{selectedPatient.study_id}</Descriptions.Item>
              <Descriptions.Item label="Site">{selectedPatient.site_id}</Descriptions.Item>
              <Descriptions.Item label="Status">
                {selectedPatient.is_clean ? (
                  <Tag color="success">Clean</Tag>
                ) : (
                  <Tag color="error">Dirty</Tag>
                )}
              </Descriptions.Item>
              <Descriptions.Item label="Cleanliness Score">
                <Progress
                  percent={Math.round(selectedPatient.cleanliness_score || 0)}
                  status={selectedPatient.cleanliness_score >= 80 ? 'success' : 'exception'}
                />
              </Descriptions.Item>
              <Descriptions.Item label="Visits">{selectedPatient.visit_count}</Descriptions.Item>
              <Descriptions.Item label="Open Queries">{selectedPatient.query_count}</Descriptions.Item>
            </Descriptions>

            {/* Clean Status Details */}
            {cleanStatus && (
              <Card
                title="Clean Patient Status Details"
                size="small"
                extra={
                  <Tag color={cleanStatus.is_clean ? 'success' : 'error'}>
                    {cleanStatus.lock_readiness}
                  </Tag>
                }
              >
                <Text type="secondary">Last checked: {cleanStatus.last_checked}</Text>
                
                {/* Blocking Factors */}
                {cleanStatus.blocking_factors && cleanStatus.blocking_factors.length > 0 && (
                  <div style={{ marginTop: 16 }}>
                    <Text strong style={{ display: 'block', marginBottom: 8 }}>
                      <WarningOutlined style={{ color: '#faad14' }} /> Blocking Factors ({cleanStatus.blocking_factors.length})
                    </Text>
                    <List
                      size="small"
                      dataSource={cleanStatus.blocking_factors}
                      renderItem={renderBlockingFactor}
                    />
                  </div>
                )}
              </Card>
            )}
          </Space>
        )}
      </Drawer>
    </div>
  )
}
