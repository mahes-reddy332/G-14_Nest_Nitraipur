import { useState } from 'react'
import {
  Card,
  Typography,
  Breadcrumb,
  Space,
  Row,
  Col,
  Tag,
  Avatar,
  List,
  Progress,
  Tabs,
  Collapse,
  Button,
  Modal,
  Timeline,
  Tooltip,
  Badge,
} from 'antd'
import {
  HomeOutlined,
  RobotOutlined,
  BulbOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  QuestionCircleOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { agentsApi } from '../api'
import type { AgentStatus, AgentInsight, AgentRecommendation } from '../types'

const { Title, Text, Paragraph } = Typography
const { Panel } = Collapse

const priorityColors = {
  critical: 'red',
  high: 'orange',
  medium: 'gold',
  low: 'green',
}

const agentColors: Record<string, string> = {
  reconciliation: '#1890ff',
  coding: '#722ed1',
  data_quality: '#13c2c2',
  predictive: '#eb2f96',
  site_liaison: '#52c41a',
  supervisor: '#fa8c16',
}

const statusColors: Record<string, string> = {
  active: 'success',
  idle: 'warning',
  error: 'error',
}

export default function Agents() {
  const [selectedInsight, setSelectedInsight] = useState<AgentInsight | null>(null)
  const [explainModalOpen, setExplainModalOpen] = useState(false)

  // Fetch agent status
  const { data: agentStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['agentStatus'],
    queryFn: agentsApi.getStatus,
  })

  // Fetch insights
  const { data: insights = [], isLoading: insightsLoading } = useQuery({
    queryKey: ['agentInsights'],
    queryFn: () => agentsApi.getInsights({ limit: 50 }),
  })

  // Fetch recommendations
  const { data: recommendations = [], isLoading: recsLoading } = useQuery({
    queryKey: ['agentRecommendations'],
    queryFn: () => agentsApi.getRecommendations({ limit: 20 }),
  })

  // Fetch explainability for selected insight
  const { data: explainability } = useQuery({
    queryKey: ['explainability', selectedInsight?.insight_id],
    queryFn: () => agentsApi.getExplainability(selectedInsight!.insight_id),
    enabled: !!selectedInsight && explainModalOpen,
  })

  const handleExplain = (insight: AgentInsight) => {
    setSelectedInsight(insight)
    setExplainModalOpen(true)
  }

  const agents = agentStatus ? Object.entries(agentStatus) : []
  const activeCount = agents.filter(([_, a]) => a.status === 'active').length

  // Group insights by agent
  const insightsByAgent = insights.reduce<Record<string, AgentInsight[]>>((acc, insight) => {
    if (!acc[insight.agent]) {
      acc[insight.agent] = []
    }
    acc[insight.agent].push(insight)
    return acc
  }, {})

  return (
    <div>
      <Breadcrumb
        items={[
          { href: '/', title: <HomeOutlined /> },
          { title: <><RobotOutlined /> AI Agents</> },
        ]}
        style={{ marginBottom: 16 }}
      />

      <Title level={2}>AI Agents</Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
        Monitor AI agent status, insights, and recommendations
      </Text>

      {/* Agent Status Cards */}
      <div style={{ marginBottom: 24 }}>
        <Title level={4}>
          Agent Status{' '}
          <Tag color="blue">{activeCount}/{agents.length} Active</Tag>
        </Title>
        <Row gutter={[16, 16]}>
          {agents.map(([key, agent]) => (
            <Col key={key} xs={24} sm={12} md={8} lg={4}>
              <Card size="small" hoverable>
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <Space>
                    <Avatar
                      style={{ backgroundColor: agentColors[key] || '#8c8c8c' }}
                      icon={<RobotOutlined />}
                    />
                    <div>
                      <Text strong style={{ display: 'block', fontSize: 12 }}>
                        {agent.name}
                      </Text>
                      <Badge
                        status={statusColors[agent.status] as 'success' | 'warning' | 'error'}
                        text={<Text type="secondary" style={{ fontSize: 11 }}>{agent.status}</Text>}
                      />
                    </div>
                  </Space>
                  <div style={{ fontSize: 11, color: '#8c8c8c' }}>
                    {agent.tasks_completed} completed • {agent.tasks_pending} pending
                  </div>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </div>

      {/* Tabs for Insights and Recommendations */}
      <Card>
        <Tabs
          defaultActiveKey="insights"
          items={[
            {
              key: 'insights',
              label: (
                <span>
                  <BulbOutlined /> Insights ({insights.length})
                </span>
              ),
              children: (
                <Collapse accordion>
                  {Object.entries(insightsByAgent).map(([agent, agentInsights]) => (
                    <Panel
                      key={agent}
                      header={
                        <Space>
                          <Avatar
                            size="small"
                            style={{ backgroundColor: agentColors[agent] || '#8c8c8c' }}
                            icon={<RobotOutlined />}
                          />
                          <Text strong>
                            {agentStatus?.[agent]?.name || agent}
                          </Text>
                          <Tag>{agentInsights.length} insights</Tag>
                        </Space>
                      }
                    >
                      <List
                        dataSource={agentInsights}
                        renderItem={(insight) => (
                          <List.Item
                            actions={[
                              <Tooltip title="Explain this insight" key="explain">
                                <Button
                                  type="link"
                                  icon={<QuestionCircleOutlined />}
                                  onClick={() => handleExplain(insight)}
                                >
                                  Explain
                                </Button>
                              </Tooltip>,
                            ]}
                          >
                            <List.Item.Meta
                              avatar={
                                <Tag color={priorityColors[insight.priority]}>
                                  {insight.priority}
                                </Tag>
                              }
                              title={insight.title}
                              description={
                                <>
                                  <Paragraph
                                    ellipsis={{ rows: 2 }}
                                    style={{ marginBottom: 8 }}
                                  >
                                    {insight.description}
                                  </Paragraph>
                                  <Space wrap size="small">
                                    <Progress
                                      percent={Math.round(insight.confidence * 100)}
                                      size="small"
                                      style={{ width: 80 }}
                                      format={(p) => `${p}%`}
                                    />
                                    <Text type="secondary" style={{ fontSize: 12 }}>
                                      Confidence
                                    </Text>
                                    {insight.affected_entities && (
                                      <Tag color="blue">
                                        {insight.affected_entities.count} {insight.affected_entities.type}
                                      </Tag>
                                    )}
                                  </Space>
                                </>
                              }
                            />
                          </List.Item>
                        )}
                      />
                    </Panel>
                  ))}
                </Collapse>
              ),
            },
            {
              key: 'recommendations',
              label: (
                <span>
                  <CheckCircleOutlined /> Recommendations ({recommendations.length})
                </span>
              ),
              children: (
                <List
                  loading={recsLoading}
                  dataSource={recommendations}
                  renderItem={(rec) => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={
                          <div style={{ textAlign: 'center' }}>
                            <Progress
                              type="circle"
                              percent={rec.priority_score}
                              size={50}
                              format={(p) => p}
                            />
                          </div>
                        }
                        title={
                          <Space>
                            <Text strong>{rec.title}</Text>
                            <Tag color={
                              rec.impact === 'high' ? 'red' :
                              rec.impact === 'medium' ? 'orange' : 'green'
                            }>
                              {rec.impact} impact
                            </Tag>
                            <Tag color={
                              rec.effort === 'high' ? 'red' :
                              rec.effort === 'medium' ? 'orange' : 'green'
                            }>
                              {rec.effort} effort
                            </Tag>
                          </Space>
                        }
                        description={
                          <>
                            <Paragraph>{rec.description}</Paragraph>
                            <div style={{ marginTop: 8 }}>
                              <Text strong>Action Items:</Text>
                              <ul style={{ marginTop: 4, paddingLeft: 20 }}>
                                {rec.action_items.map((item, idx) => (
                                  <li key={idx}><Text type="secondary">{item}</Text></li>
                                ))}
                              </ul>
                            </div>
                            <Space style={{ marginTop: 8 }}>
                              <ClockCircleOutlined />
                              <Text type="secondary">
                                Est. time: {rec.estimated_completion_time}
                              </Text>
                              <Tag color="blue">{rec.source_agent}</Tag>
                            </Space>
                          </>
                        }
                      />
                    </List.Item>
                  )}
                />
              ),
            },
          ]}
        />
      </Card>

      {/* Explainability Modal */}
      <Modal
        title={
          <Space>
            <InfoCircleOutlined />
            Insight Explanation
          </Space>
        }
        open={explainModalOpen}
        onCancel={() => setExplainModalOpen(false)}
        footer={null}
        width={700}
      >
        {selectedInsight && (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Card size="small" title="Insight">
              <Text strong>{selectedInsight.title}</Text>
              <Paragraph type="secondary">{selectedInsight.description}</Paragraph>
            </Card>

            {explainability && (
              <>
                <Card size="small" title="Decision Path">
                  <Timeline>
                    {explainability.decision_path?.map((step: { step: number; description: string }) => (
                      <Timeline.Item key={step.step}>
                        <Text strong>Step {step.step}:</Text> {step.description}
                      </Timeline.Item>
                    ))}
                  </Timeline>
                </Card>

                <Card size="small" title="Key Factors">
                  <List
                    size="small"
                    dataSource={explainability.key_factors || []}
                    renderItem={(factor: { factor: string; contribution: number; description: string }) => (
                      <List.Item>
                        <List.Item.Meta
                          title={
                            <Space>
                              <Text>{factor.factor}</Text>
                              <Progress
                                percent={Math.round(factor.contribution * 100)}
                                size="small"
                                style={{ width: 100 }}
                              />
                            </Space>
                          }
                          description={factor.description}
                        />
                      </List.Item>
                    )}
                  />
                </Card>

                <Card size="small" title="Data Sources & Algorithms">
                  <Space wrap>
                    {explainability.data_sources?.map((source: string) => (
                      <Tag key={source} color="blue">{source}</Tag>
                    ))}
                    {explainability.algorithms_used?.map((algo: string) => (
                      <Tag key={algo} color="purple">{algo}</Tag>
                    ))}
                  </Space>
                </Card>
              </>
            )}
          </Space>
        )}
      </Modal>
    </div>
  )
}
