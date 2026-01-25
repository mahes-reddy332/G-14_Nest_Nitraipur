import { useState } from 'react'
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
  Button,
  Select,
  Tabs,
  Modal,
  Input,
  Form,
  message,
  Timeline,
  Tooltip,
} from 'antd'
import {
  HomeOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { alertsApi } from '../api'
import type { Alert } from '../types'
import type { ColumnsType } from 'antd/es/table'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'

dayjs.extend(relativeTime)

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input

const severityConfig = {
  critical: { color: '#ff4d4f', icon: <AlertOutlined />, tagColor: 'red' },
  high: { color: '#fa8c16', icon: <ExclamationCircleOutlined />, tagColor: 'orange' },
  medium: { color: '#faad14', icon: <InfoCircleOutlined />, tagColor: 'gold' },
  low: { color: '#52c41a', icon: <CheckCircleOutlined />, tagColor: 'green' },
}

const statusConfig: Record<string, string> = {
  new: 'red',
  acknowledged: 'orange',
  in_progress: 'blue',
  resolved: 'green',
  dismissed: 'default',
}

const categoryLabels: Record<string, string> = {
  data_quality: 'Data Quality',
  safety: 'Safety',
  operational: 'Operational',
  compliance: 'Compliance',
  system: 'System',
}

export default function Alerts() {
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState('active')
  const [statusFilter, setStatusFilter] = useState<string | null>(null)
  const [severityFilter, setSeverityFilter] = useState<string | null>(null)
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null)
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [resolveModalOpen, setResolveModalOpen] = useState(false)
  const [form] = Form.useForm()

  // Fetch alerts
  const { data: alerts = [], isLoading } = useQuery({
    queryKey: ['alerts', statusFilter, severityFilter, categoryFilter],
    queryFn: () => alertsApi.getAll({
      status: statusFilter || undefined,
      severity: severityFilter || undefined,
      category: categoryFilter || undefined,
    }),
  })

  // Fetch summary
  const { data: summary } = useQuery({
    queryKey: ['alertSummary'],
    queryFn: alertsApi.getSummary,
  })

  // Fetch critical alerts
  const { data: criticalAlerts = [] } = useQuery({
    queryKey: ['criticalAlerts'],
    queryFn: alertsApi.getCritical,
  })

  // Acknowledge mutation
  const acknowledgeMutation = useMutation({
    mutationFn: ({ alertId, user }: { alertId: string; user: string }) =>
      alertsApi.acknowledge(alertId, user),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] })
      queryClient.invalidateQueries({ queryKey: ['alertSummary'] })
      message.success('Alert acknowledged')
    },
  })

  // Resolve mutation
  const resolveMutation = useMutation({
    mutationFn: ({ alertId, user, notes }: { alertId: string; user: string; notes?: string }) =>
      alertsApi.resolve(alertId, user, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] })
      queryClient.invalidateQueries({ queryKey: ['alertSummary'] })
      message.success('Alert resolved')
      setResolveModalOpen(false)
      form.resetFields()
    },
  })

  const handleAcknowledge = (alert: Alert) => {
    acknowledgeMutation.mutate({ alertId: alert.alert_id, user: 'Current User' })
  }

  const handleResolve = (values: { notes: string }) => {
    if (selectedAlert) {
      resolveMutation.mutate({
        alertId: selectedAlert.alert_id,
        user: 'Current User',
        notes: values.notes,
      })
    }
  }

  const showAlertDetail = (alert: Alert) => {
    setSelectedAlert(alert)
    setModalOpen(true)
  }

  const columns: ColumnsType<Alert> = [
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity) => {
        const config = severityConfig[severity as keyof typeof severityConfig]
        return (
          <Tag icon={config?.icon} color={config?.tagColor}>
            {severity}
          </Tag>
        )
      },
      filters: Object.entries(severityConfig).map(([key]) => ({
        text: key.charAt(0).toUpperCase() + key.slice(1),
        value: key,
      })),
      onFilter: (value, record) => record.severity === value,
    },
    {
      title: 'Title',
      dataIndex: 'title',
      key: 'title',
      render: (text, record) => (
        <a onClick={() => showAlertDetail(record)}>{text}</a>
      ),
    },
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      render: (category) => (
        <Tag>{categoryLabels[category] || category}</Tag>
      ),
      filters: Object.entries(categoryLabels).map(([key, label]) => ({
        text: label,
        value: key,
      })),
      onFilter: (value, record) => record.category === value,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={statusConfig[status]}>{status.replace('_', ' ')}</Tag>
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => (
        <Tooltip title={dayjs(date).format('YYYY-MM-DD HH:mm:ss')}>
          <span>{dayjs(date).fromNow()}</span>
        </Tooltip>
      ),
      sorter: (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 200,
      render: (_, record) => (
        <Space>
          {record.status === 'new' && (
            <Button
              type="primary"
              size="small"
              onClick={() => handleAcknowledge(record)}
              loading={acknowledgeMutation.isPending}
            >
              Acknowledge
            </Button>
          )}
          {['new', 'acknowledged', 'in_progress'].includes(record.status) && (
            <Button
              size="small"
              onClick={() => {
                setSelectedAlert(record)
                setResolveModalOpen(true)
              }}
            >
              Resolve
            </Button>
          )}
        </Space>
      ),
    },
  ]

  // Filter active vs resolved alerts
  const activeAlerts = alerts.filter(a => !['resolved', 'dismissed'].includes(a.status))
  const resolvedAlerts = alerts.filter(a => ['resolved', 'dismissed'].includes(a.status))

  return (
    <div>
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: <><AlertOutlined /> Alerts</> },
        ]}
        style={{ marginBottom: 16 }}
      />

      <Title level={2}>Alerts</Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
        Real-time alerts and notifications for data quality and operational issues
      </Text>

      {/* Summary Stats */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Active"
              value={summary?.active_alerts || 0}
              prefix={<AlertOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Critical"
              value={summary?.by_severity?.critical || 0}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<CloseCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="High Priority"
              value={summary?.by_severity?.high || 0}
              valueStyle={{ color: '#fa8c16' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Resolved Today"
              value={summary?.by_status?.resolved || 0}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      <Card style={{ marginBottom: 24 }}>
        <Space wrap>
          <Select
            placeholder="Severity"
            allowClear
            style={{ width: 150 }}
            onChange={setSeverityFilter}
            options={Object.keys(severityConfig).map(s => ({
              value: s,
              label: s.charAt(0).toUpperCase() + s.slice(1),
            }))}
          />
          <Select
            placeholder="Category"
            allowClear
            style={{ width: 150 }}
            onChange={setCategoryFilter}
            options={Object.entries(categoryLabels).map(([key, label]) => ({
              value: key,
              label,
            }))}
          />
          <Select
            placeholder="Status"
            allowClear
            style={{ width: 150 }}
            onChange={setStatusFilter}
            options={Object.keys(statusConfig).map(s => ({
              value: s,
              label: s.charAt(0).toUpperCase() + s.slice(1).replace('_', ' '),
            }))}
          />
        </Space>
      </Card>

      {/* Alerts Tabs */}
      <Card>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            {
              key: 'active',
              label: `Active (${activeAlerts.length})`,
              children: (
                <Table
                  columns={columns}
                  dataSource={activeAlerts}
                  rowKey="alert_id"
                  loading={isLoading}
                  pagination={{ pageSize: 10 }}
                />
              ),
            },
            {
              key: 'critical',
              label: (
                <span>
                  Critical{' '}
                  {criticalAlerts.length > 0 && (
                    <Tag color="red">{criticalAlerts.length}</Tag>
                  )}
                </span>
              ),
              children: (
                <Table
                  columns={columns}
                  dataSource={criticalAlerts}
                  rowKey="alert_id"
                  loading={isLoading}
                  pagination={{ pageSize: 10 }}
                />
              ),
            },
            {
              key: 'resolved',
              label: `Resolved (${resolvedAlerts.length})`,
              children: (
                <Table
                  columns={columns}
                  dataSource={resolvedAlerts}
                  rowKey="alert_id"
                  loading={isLoading}
                  pagination={{ pageSize: 10 }}
                />
              ),
            },
          ]}
        />
      </Card>

      {/* Alert Detail Modal */}
      <Modal
        title={`Alert: ${selectedAlert?.title}`}
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        footer={null}
        width={600}
      >
        {selectedAlert && (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Row gutter={16}>
              <Col span={8}>
                <Tag
                  icon={severityConfig[selectedAlert.severity]?.icon}
                  color={severityConfig[selectedAlert.severity]?.tagColor}
                >
                  {selectedAlert.severity}
                </Tag>
              </Col>
              <Col span={8}>
                <Tag color={statusConfig[selectedAlert.status]}>
                  {selectedAlert.status.replace('_', ' ')}
                </Tag>
              </Col>
              <Col span={8}>
                <Tag>{categoryLabels[selectedAlert.category]}</Tag>
              </Col>
            </Row>

            <Paragraph>{selectedAlert.description}</Paragraph>

            <div>
              <Text strong>Source: </Text>
              <Text>{selectedAlert.source}</Text>
            </div>

            <div>
              <Text strong>Affected Entity: </Text>
              <Text>
                {selectedAlert.affected_entity?.type}: {selectedAlert.affected_entity?.id}
              </Text>
            </div>

            <div>
              <Text strong>Created: </Text>
              <Text>{dayjs(selectedAlert.created_at).format('YYYY-MM-DD HH:mm:ss')}</Text>
            </div>

            {selectedAlert.actions_taken && selectedAlert.actions_taken.length > 0 && (
              <div>
                <Text strong>Actions Taken:</Text>
                <Timeline style={{ marginTop: 8 }}>
                  {selectedAlert.actions_taken.map((action, idx) => (
                    <Timeline.Item key={idx}>
                      {action.action} - {action.performed_by} ({dayjs(action.performed_at).fromNow()})
                    </Timeline.Item>
                  ))}
                </Timeline>
              </div>
            )}
          </Space>
        )}
      </Modal>

      {/* Resolve Modal */}
      <Modal
        title="Resolve Alert"
        open={resolveModalOpen}
        onCancel={() => setResolveModalOpen(false)}
        footer={null}
      >
        <Form form={form} onFinish={handleResolve} layout="vertical">
          <Form.Item name="notes" label="Resolution Notes">
            <TextArea rows={4} placeholder="Enter resolution notes..." />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={resolveMutation.isPending}>
                Resolve
              </Button>
              <Button onClick={() => setResolveModalOpen(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
