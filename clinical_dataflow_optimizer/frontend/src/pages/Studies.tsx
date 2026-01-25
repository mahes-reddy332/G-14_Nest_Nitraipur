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
  Tabs,
  Badge,
} from 'antd'
import {
  HomeOutlined,
  ExperimentOutlined,
  TeamOutlined,
  FileTextOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  DatabaseOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { studiesApi, patientsApi, sitesApi } from '../api'
import type { Study, SourceFile } from '../types'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text } = Typography

const statusColors: Record<string, string> = {
  active: 'green',
  enrolling: 'blue',
  completed: 'default',
  suspended: 'orange',
}

export default function Studies() {
  const { studyId } = useParams()
  const [activeTab, setActiveTab] = useState('overview')

  // Fetch all studies
  const { data: studies = [], isLoading } = useQuery({
    queryKey: ['studies'],
    queryFn: studiesApi.getAll,
    enabled: !studyId,
  })

  // Fetch single study if studyId provided
  const { data: study } = useQuery({
    queryKey: ['study', studyId],
    queryFn: () => studiesApi.getById(studyId!),
    enabled: !!studyId,
  })

  // Fetch study metrics
  const { data: metrics } = useQuery({
    queryKey: ['studyMetrics', studyId],
    queryFn: () => studiesApi.getMetrics(studyId!),
    enabled: !!studyId,
  })

  // Fetch source files for the study
  const { data: sourceFiles = [], isLoading: isSourceFilesLoading } = useQuery<SourceFile[]>({
    queryKey: ['sourceFiles', studyId],
    queryFn: () => studiesApi.getSourceFiles(studyId!),
    enabled: !!studyId,
  })

  const columns: ColumnsType<Study> = [
    {
      title: 'Study ID',
      dataIndex: 'study_id',
      key: 'study_id',
      render: (text) => <a href={`/studies/${text}`}>{text}</a>,
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Phase',
      dataIndex: 'phase',
      key: 'phase',
      render: (phase) => <Tag color="blue">{phase}</Tag>,
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
      dataIndex: 'total_patients',
      key: 'total_patients',
      sorter: (a, b) => (a.total_patients || 0) - (b.total_patients || 0),
    },
    {
      title: 'Clean %',
      key: 'clean_rate',
      render: (_, record) => {
        const rate = record.total_patients > 0
          ? ((record.clean_patients || 0) / record.total_patients) * 100
          : 0
        return (
          <Progress
            percent={Math.round(rate)}
            size="small"
            status={rate >= 80 ? 'success' : rate >= 60 ? 'normal' : 'exception'}
          />
        )
      },
      sorter: (a, b) => {
        const rateA = a.total_patients > 0 ? (a.clean_patients || 0) / a.total_patients : 0
        const rateB = b.total_patients > 0 ? (b.clean_patients || 0) / b.total_patients : 0
        return rateA - rateB
      },
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
        />
      ),
      sorter: (a, b) => (a.dqi_score || 0) - (b.dqi_score || 0),
    },
    {
      title: 'Sites',
      dataIndex: 'total_sites',
      key: 'total_sites',
    },
  ]

  // Study detail view
  if (studyId && study) {
    return (
      <div>
        <Breadcrumb
          items={[
            { href: '/', title: <HomeOutlined /> },
            { href: '/studies', title: <><ExperimentOutlined /> Studies</> },
            { title: study.study_id },
          ]}
          style={{ marginBottom: 16 }}
        />

        <Title level={2}>{study.name || study.study_id}</Title>
        <Space style={{ marginBottom: 24 }}>
          <Tag color={statusColors[study.status?.toLowerCase()]}>{study.status}</Tag>
          <Tag color="blue">{study.phase}</Tag>
          {study.therapeutic_area && <Tag>{study.therapeutic_area}</Tag>}
        </Space>

        {/* Summary Stats */}
        <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Total Patients"
                value={study.total_patients}
                prefix={<TeamOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Clean Patients"
                value={study.clean_patients}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="DQI Score"
                value={study.dqi_score?.toFixed(1)}
                suffix="%"
                valueStyle={{
                  color: (study.dqi_score || 0) >= 80 ? '#52c41a' : '#faad14',
                }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Total Sites"
                value={study.total_sites}
                prefix={<FileTextOutlined />}
              />
            </Card>
          </Col>
        </Row>

        {/* Tabs for detail views */}
        <Card>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={[
              {
                key: 'overview',
                label: 'Overview',
                children: (
                  <div>
                    {metrics && (
                      <Row gutter={[24, 24]}>
                        <Col span={12}>
                          <Card size="small" title="Data Quality">
                            <Progress
                              percent={Math.round(metrics.dqi_score || 0)}
                              status={metrics.dqi_score >= 80 ? 'success' : 'normal'}
                            />
                            <Text type="secondary">Cleanliness Rate: {metrics.cleanliness_rate?.toFixed(1)}%</Text>
                          </Card>
                        </Col>
                        <Col span={12}>
                          <Card size="small" title="Queries">
                            <Statistic value={metrics.query_count} suffix="open" />
                            <Text type="secondary">Resolution Rate: {metrics.query_resolution_rate?.toFixed(1)}%</Text>
                          </Card>
                        </Col>
                      </Row>
                    )}
                  </div>
                ),
              },
              {
                key: 'source-files',
                label: (
                  <span>
                    <DatabaseOutlined /> Source Files
                    <Badge 
                      count={sourceFiles.filter(f => f.status === 'loaded').length} 
                      style={{ marginLeft: 8, backgroundColor: '#52c41a' }}
                      showZero
                    />
                  </span>
                ),
                children: (
                  <div>
                    <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
                      Excel files processed for this study from the QC Anonymized Study Files folder
                    </Text>
                    <Table
                      dataSource={sourceFiles}
                      rowKey="file_type"
                      loading={isSourceFilesLoading}
                      pagination={false}
                      columns={[
                        {
                          title: 'File Type',
                          dataIndex: 'display_name',
                          key: 'display_name',
                          render: (name: string) => <Text strong>{name}</Text>,
                        },
                        {
                          title: 'Status',
                          dataIndex: 'status',
                          key: 'status',
                          render: (status: string) => (
                            status === 'loaded' ? (
                              <Tag icon={<CheckCircleOutlined />} color="success">Loaded</Tag>
                            ) : (
                              <Tag icon={<CloseCircleOutlined />} color="default">Not Found</Tag>
                            )
                          ),
                        },
                        {
                          title: 'Records',
                          dataIndex: 'record_count',
                          key: 'record_count',
                          render: (count: number, record: SourceFile) => (
                            record.status === 'loaded' 
                              ? <Text>{count.toLocaleString()}</Text>
                              : <Text type="secondary">-</Text>
                          ),
                        },
                        {
                          title: 'Loaded At',
                          dataIndex: 'loaded_at',
                          key: 'loaded_at',
                          render: (date: string | null) => (
                            date 
                              ? <Text type="secondary">{new Date(date).toLocaleString()}</Text>
                              : <Text type="secondary">-</Text>
                          ),
                        },
                      ]}
                    />
                  </div>
                ),
              },
              {
                key: 'patients',
                label: 'Patients',
                children: <Text>Patient list for {studyId}</Text>,
              },
              {
                key: 'sites',
                label: 'Sites',
                children: <Text>Site list for {studyId}</Text>,
              },
              {
                key: 'metrics',
                label: 'Metrics',
                children: <Text>Detailed metrics for {studyId}</Text>,
              },
            ]}
          />
        </Card>
      </div>
    )
  }

  // Studies list view
  return (
    <div>
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: <><ExperimentOutlined /> Studies</> },
        ]}
        style={{ marginBottom: 16 }}
      />

      <Title level={2}>Studies</Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
        Overview of all clinical studies and their data quality metrics
      </Text>

      <Card>
        <Table
          columns={columns}
          dataSource={studies}
          rowKey="study_id"
          loading={isLoading}
          pagination={{ pageSize: 10 }}
        />
      </Card>
    </div>
  )
}
