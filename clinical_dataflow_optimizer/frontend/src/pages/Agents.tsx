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
    <div className="agents-dashboard" style={{ padding: '24px', minHeight: '100vh' }}>
      {/* Custom Styles */}
      <style>{`
        :root {
          --glass-bg: rgba(30, 30, 40, 0.6);
          --glass-border: rgba(255, 255, 255, 0.08);
          --neon-blue: #00f3ff;
          --neon-purple: #bc13fe;
          --neon-green: #0aff60;
          --card-radius: 16px;
        }

        .agents-dashboard {
          color: #e0e0e0;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .agent-card {
          background: var(--glass-bg);
          backdrop-filter: blur(12px);
          border: 1px solid var(--glass-border);
          border-radius: var(--card-radius);
          padding: 24px;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
          height: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
        }

        .agent-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 10px 30px -10px rgba(0, 243, 255, 0.15);
          border-color: rgba(0, 243, 255, 0.3);
        }

        .agent-avatar-container {
          position: relative;
          margin-bottom: 16px;
        }

        .agent-avatar {
          width: 64px;
          height: 64px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 28px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
          transition: all 0.3s ease;
        }

        .agent-card:hover .agent-avatar {
          transform: scale(1.1);
          box-shadow: 0 0 25px var(--neon-blue);
          border-color: var(--neon-blue);
        }

        .status-badge {
          display: inline-flex;
          align-items: center;
          padding: 4px 12px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: 600;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          margin-top: 12px;
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          margin-right: 8px;
          box-shadow: 0 0 8px currentColor;
        }

        .status-dot.active {
          background-color: var(--neon-green);
          color: var(--neon-green);
          animation: pulse 2s infinite;
        }
        
        .status-dot.idle {
           background-color: #faad14;
           color: #faad14;
        }

        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(10, 255, 96, 0.4); }
          70% { box-shadow: 0 0 0 6px rgba(10, 255, 96, 0); }
          100% { box-shadow: 0 0 0 0 rgba(10, 255, 96, 0); }
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
        }
        
        /* Modern Tabs */
        .modern-tabs .ant-tabs-nav {
          margin-bottom: 24px !important;
        }
        .modern-tabs .ant-tabs-tab {
          padding: 12px 24px !important;
          border-radius: 12px !important;
          background: transparent !important;
          border: 1px solid transparent !important;
          color: #a0a0a0 !important;
          transition: all 0.3s ease !important;
        }
        .modern-tabs .ant-tabs-tab-active {
            background: rgba(255, 255, 255, 0.05) !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
            color: #fff !important;
        }
        .modern-tabs .ant-tabs-ink-bar {
            background: var(--neon-blue) !important;
            height: 3px !important;
            border-radius: 2px;
        }

        /* Insight Card */
        .insight-panel {
            background: var(--glass-bg) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 12px !important;
            margin-bottom: 16px !important;
            overflow: hidden;
        }
        
        .insight-panel .ant-collapse-header {
            color: #fff !important;
            align-items: center !important;
            padding: 16px 24px !important;
        }
        
        .insight-panel .ant-collapse-content {
            background: transparent !important;
            border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
            color: #d0d0d0 !important;
        }

        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.5px;
        }
        
        .agent-name {
            font-size: 16px;
            font-weight: 600;
            color: #fff;
            letter-spacing: 0.5px;
        }
      `}</style>

      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <Space direction="vertical" size={4}>
          <Breadcrumb
            items={[
              { href: '/', title: <HomeOutlined style={{ color: 'rgba(255,255,255,0.5)' }} /> },
              { title: <span style={{ color: 'rgba(255,255,255,0.8)' }}>AI Agents</span> },
            ]}
          />
          <Title level={2} style={{ margin: 0, color: '#fff', fontSize: '32px', fontWeight: 700, letterSpacing: '-1px' }}>
            <span style={{ background: 'linear-gradient(45deg, #fff, #a5a5a5)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Command Center
            </span>
          </Title>
          <Text style={{ color: 'rgba(255,255,255,0.45)', fontSize: '16px' }}>
            Orchestrate your clinical data intelligence agents
          </Text>
        </Space>
      </div>

      {/* Agents Grid */}
      <div style={{ marginBottom: 40 }}>
        <div className="section-header">
          <Title level={4} style={{ color: '#fff', margin: 0 }}>Active Agents</Title>
          <Tag color="rgba(0, 243, 255, 0.15)" style={{ color: '#00f3ff', border: '1px solid rgba(0, 243, 255, 0.3)', borderRadius: '12px', padding: '4px 12px' }}>
            {activeCount}/{agents.length} Operational
          </Tag>
        </div>

        <Row gutter={[20, 20]}>
          {agents.map(([key, agent]) => (
            <Col key={key} xs={24} sm={12} md={8} lg={4}>
              <div className="agent-card">
                <div className="agent-avatar-container">
                  <div className="agent-avatar" style={{ color: agentColors[key] || '#fff' }}>
                    <RobotOutlined />
                  </div>
                </div>
                <div className="agent-name">{agent.name}</div>
                <div className="status-badge">
                  <span className={`status-dot ${agent.status}`}></span>
                  {agent.status.toUpperCase()}
                </div>
              </div>
            </Col>
          ))}
        </Row>
      </div>

      {/* Analytics & Insights Section */}
      <div style={{ background: 'rgba(20, 20, 30, 0.4)', borderRadius: '24px', padding: '32px', border: '1px solid rgba(255,255,255,0.05)' }}>
        <Tabs
          defaultActiveKey="insights"
          className="modern-tabs"
          items={[
            {
              key: 'insights',
              label: (<span><BulbOutlined /> Live Insights</span>),
              children: (
                <div style={{ marginTop: '16px' }}>
                  <Collapse accordion ghost>
                    {Object.entries(insightsByAgent).map(([agent, agentInsights]) => (
                      <Panel
                        key={agent}
                        className="insight-panel"
                        header={
                          <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                            <Avatar
                              size="small"
                              icon={<RobotOutlined />}
                              style={{ backgroundColor: agentColors[agent] || '#555', marginRight: 12 }}
                            />
                            <span style={{ fontSize: '15px', fontWeight: 600, color: '#fff', flex: 1 }}>
                              {agentStatus?.[agent]?.name || agent}
                            </span>
                            <Tag color="blue" style={{ borderRadius: '8px' }}>
                              {agentInsights.length} New
                            </Tag>
                          </div>
                        }
                      >
                        <List
                          itemLayout="vertical"
                          dataSource={agentInsights}
                          renderItem={(insight) => (
                            <List.Item style={{ padding: '16px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                              <Space align="start" style={{ width: '100%' }}>
                                <div style={{ marginTop: 4 }}>
                                  {insight.priority === 'critical' ? <ExclamationCircleOutlined style={{ color: '#ff4d4f', fontSize: 18 }} /> :
                                    insight.priority === 'high' ? <InfoCircleOutlined style={{ color: '#faad14', fontSize: 18 }} /> :
                                      <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 18 }} />}
                                </div>
                                <div style={{ flex: 1 }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                    <Text style={{ color: '#fff', fontWeight: 600, fontSize: 15 }}>{insight.title}</Text>
                                    <Tag color={priorityColors[insight.priority]} style={{ margin: 0 }}>{insight.priority.toUpperCase()}</Tag>
                                  </div>
                                  <Paragraph style={{ color: 'rgba(255,255,255,0.6)', marginBottom: 12 }}>{insight.description}</Paragraph>

                                  <Space size="large">
                                    <div>
                                      <Text style={{ color: 'rgba(255,255,255,0.4)', fontSize: 12, display: 'block' }}>CONFIDENCE</Text>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                        <Progress percent={Math.round(insight.confidence * 100)} size="small" steps={5} strokeColor={agentColors[agent] || '#1890ff'} trailColor="rgba(255,255,255,0.1)" style={{ width: 80 }} showInfo={false} />
                                        <span style={{ color: '#fff', fontSize: 12 }}>{Math.round(insight.confidence * 100)}%</span>
                                      </div>
                                    </div>
                                    <Button type="link" size="small" onClick={() => handleExplain(insight)} style={{ padding: 0, color: 'var(--neon-blue)' }}>
                                      View Analysis <QuestionCircleOutlined />
                                    </Button>
                                  </Space>
                                </div>
                              </Space>
                            </List.Item>
                          )}
                        />
                      </Panel>
                    ))}
                  </Collapse>
                </div>
              ),
            },
            {
              key: 'recommendations',
              label: (<span><CheckCircleOutlined /> Action Plan</span>),
              children: (
                <List
                  grid={{ gutter: 16, column: 2 }}
                  dataSource={recommendations}
                  renderItem={(rec) => (
                    <List.Item>
                      <div style={{
                        background: 'rgba(255,255,255,0.03)',
                        padding: 20,
                        borderRadius: 12,
                        border: '1px solid rgba(255,255,255,0.05)'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                          <Space>
                            <Tag color={rec.impact === 'high' ? 'red' : 'orange'}>{rec.impact} impact</Tag>
                            <Text style={{ color: '#00f3ff', fontSize: 12 }}>Est: {rec.estimated_completion_time}</Text>
                          </Space>
                          <div style={{ width: 30 }}>
                            <Progress type="circle" percent={rec.priority_score} width={30} strokeColor={rec.priority_score > 80 ? '#ff4d4f' : '#faad14'} format={() => ''} />
                          </div>
                        </div>
                        <Text style={{ color: '#fff', fontWeight: 600, fontSize: 15, display: 'block', marginBottom: 8 }}>{rec.title}</Text>
                        <Paragraph style={{ color: 'rgba(255,255,255,0.5)', fontSize: 13, marginBottom: 16 }} ellipsis={{ rows: 2 }}>{rec.description}</Paragraph>

                        <div style={{ background: 'rgba(0,0,0,0.2)', padding: 12, borderRadius: 8 }}>
                          <Text style={{ color: 'rgba(255,255,255,0.7)', fontSize: 12, fontWeight: 600 }}>NEXT STEPS:</Text>
                          <ul style={{ margin: '8px 0 0 20px', padding: 0, color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>
                            {rec.action_items.slice(0, 2).map((item, i) => <li key={i}>{item}</li>)}
                          </ul>
                        </div>
                      </div>
                    </List.Item>
                  )}
                />
              ),
            },
          ]}
        />
      </div>

      {/* Explainability Modal */}
      <Modal
        title={<span style={{ color: '#fff' }}>Insight Explanation</span>}
        open={explainModalOpen}
        onCancel={() => setExplainModalOpen(false)}
        footer={null}
        width={700}
        styles={{ content: { background: '#1f1f2e', borderRadius: 16, border: '1px solid #333' }, header: { background: 'transparent', borderBottom: '1px solid #333' } }}
        closeIcon={<span style={{ color: '#fff' }}>×</span>}
      >
        {selectedInsight && (
          <div style={{ color: '#d0d0d0' }}>
            <Title level={4} style={{ color: '#fff' }}>{selectedInsight.title}</Title>
            <Paragraph style={{ color: '#a0a0a0' }}>{selectedInsight.description}</Paragraph>

            {explainability && (
              <div style={{ marginTop: 24 }}>
                <div style={{ marginBottom: 24 }}>
                  <Text strong style={{ color: '#00f3ff' }}>KEY FACTORS</Text>
                  <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                    {explainability.key_factors?.map((f: any, i: number) => (
                      <div key={i} style={{ background: 'rgba(255,255,255,0.05)', padding: '8px 12px', borderRadius: 8 }}>
                        <Text style={{ color: '#fff', marginRight: 8 }}>{f.factor}</Text>
                        <Tag color="blue">{(f.contribution * 100).toFixed(0)}%</Tag>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <Text strong style={{ color: '#00f3ff' }}>ANALYSIS PATH</Text>
                  <Timeline style={{ marginTop: 16, color: '#fff' }}>
                    {explainability.decision_path?.map((step: any, i: number) => (
                      <Timeline.Item key={i} color="gray">
                        <Text style={{ color: '#ccc' }}>{step.description}</Text>
                      </Timeline.Item>
                    ))}
                  </Timeline>
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  )
}
