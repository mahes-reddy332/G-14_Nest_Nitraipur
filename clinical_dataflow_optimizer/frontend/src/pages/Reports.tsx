import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, Table, Typography, Breadcrumb, Space, Tag, Spin, Empty } from 'antd'
import { HomeOutlined, FileTextOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { reportsApi } from '../api'
import type { ReportSummary, ReportDetail } from '../types'

const { Title, Text } = Typography

export default function Reports() {
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null)

  const { data: reports = [], isLoading } = useQuery({
    queryKey: ['reports'],
    queryFn: reportsApi.list,
  })

  const { data: reportDetail, isLoading: isDetailLoading } = useQuery<ReportDetail>({
    queryKey: ['reportDetail', selectedReportId],
    queryFn: () => reportsApi.getById(selectedReportId!),
    enabled: !!selectedReportId,
  })

  const columns: ColumnsType<ReportSummary> = [
    {
      title: 'Report ID',
      dataIndex: 'report_id',
      key: 'report_id',
      render: (text, record) => (
        <a onClick={() => setSelectedReportId(record.report_id)}>{text}</a>
      ),
    },
    {
      title: 'Type',
      dataIndex: 'report_type',
      key: 'report_type',
      render: (value) => <Tag color={value === 'html' ? 'blue' : 'green'}>{value}</Tag>,
    },
    {
      title: 'Size',
      dataIndex: 'size_bytes',
      key: 'size_bytes',
      render: (value) => `${(value / 1024).toFixed(1)} KB`,
    },
    {
      title: 'Last Modified',
      dataIndex: 'last_modified',
      key: 'last_modified',
    },
  ]

  return (
    <div>
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: <><FileTextOutlined /> Reports</> },
        ]}
        style={{ marginBottom: 16 }}
      />

      <Title level={2}>Reports</Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
        Generated reports and dashboard artifacts
      </Text>

      <Card style={{ marginBottom: 24 }}>
        <Table
          columns={columns}
          dataSource={reports}
          rowKey="report_id"
          loading={isLoading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      <Card title="Report Viewer">
        {isDetailLoading && <Spin />}
        {!isDetailLoading && !reportDetail && <Empty description="Select a report to view" />}
        {!isDetailLoading && reportDetail && (
          <div>
            <Space direction="vertical" size={8} style={{ width: '100%' }}>
              <Text strong>{reportDetail.name}</Text>
              <Text type="secondary">{reportDetail.report_id}</Text>
            </Space>

            <div style={{ marginTop: 16 }}>
              {reportDetail.report_type === 'html' ? (
                <iframe
                  title="report-viewer"
                  srcDoc={String(reportDetail.content)}
                  style={{ width: '100%', height: 600, border: '1px solid #f0f0f0' }}
                />
              ) : (
                <pre style={{ maxHeight: 600, overflow: 'auto', background: '#fafafa', padding: 16 }}>
                  {JSON.stringify(reportDetail.content, null, 2)}
                </pre>
              )}
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
