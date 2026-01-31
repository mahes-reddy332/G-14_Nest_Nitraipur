import React, { useState, useEffect, useRef } from 'react';
import { Layout, Typography, Input, Button, Card, List, Tag, Spin, message, Space, Avatar } from 'antd';
import { SendOutlined, RobotOutlined, UserOutlined, BulbOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import axios from 'axios';
import { useStore } from '../store';

const { Content } = Layout;
const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

interface Insight {
  category: string;
  severity: string;
  title: string;
  description: string;
  confidence: number;
  related_entities: string[];
  next_steps: string[];
}

interface ConversationalResponse {
  query: string;
  understanding: string;
  answer: string;
  insights: Insight[];
  follow_up_questions: string[];
  confidence: number;
  processing_time_ms: number;
  session_id: string;
}

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  response?: ConversationalResponse;
}

const Conversational: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const selectedStudyId = useStore((state) => state.selectedStudyId);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Start a new session when component mounts or study changes
    startNewSession();
  }, [selectedStudyId]);

  const startNewSession = async () => {
    try {
      const response = await axios.post('/api/conversational/session/start', {
        study_id: selectedStudyId || undefined,
      });
      setSessionId(response.data.session_id);
      setMessages([]);
      message.success('Conversation session started');
    } catch (error) {
      message.error('Failed to start conversation session');
    }
  };

  const sendQuery = async (query: string) => {
    if (!query.trim() || !sessionId) return;

    setLoading(true);

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: query,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    try {
      const response = await axios.post('/api/conversational/query', {
        query: query,
        session_id: sessionId,
        study_id: selectedStudyId || undefined,
      });

      const data: ConversationalResponse = response.data;

      // Add assistant message
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        response: data,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      message.error('Failed to process query');
      console.error('Query error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = () => {
    if (inputValue.trim()) {
      sendQuery(inputValue);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFollowUpClick = (question: string) => {
    sendQuery(question);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'red';
      case 'high': return 'orange';
      case 'warning': return 'gold';
      default: return 'blue';
    }
  };

  const exampleQueries = [
    "Show me sites with high missing visits",
    "What are the top 5 sites by open queries?",
    "Find correlations between metrics",
    "Show me patients with SAE issues",
    "Analyze trends in data quality",
    "Find sites with protocol deviations"
  ];

  return (
    <Content style={{ padding: '24px', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <Title level={2}>
          <RobotOutlined style={{ marginRight: '12px' }} />
          Knowledge Graph Chatbot
        </Title>
        <Paragraph>
          Ask questions about clinical trial data in natural language. The chatbot uses the study knowledge graph
          and LongCat reasoning to provide grounded answers.
        </Paragraph>
        <Space size="middle" style={{ marginBottom: '16px' }}>
          <Tag color="blue">
            Knowledge Graph: {selectedStudyId || 'All Studies'}
          </Tag>
          <Tag color="green">LongCat Reasoning Enabled</Tag>
        </Space>

        <div style={{ display: 'flex', gap: '24px', height: 'calc(100vh - 200px)' }}>
          {/* Chat Interface */}
          <div style={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
            <Card
              title="Conversation"
              style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
              bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '16px' }}
            >
              {/* Messages */}
              <div
                style={{
                  flex: 1,
                  overflowY: 'auto',
                  marginBottom: '16px',
                  padding: '16px',
                  border: '1px solid #303030',
                  borderRadius: '12px',
                  backgroundColor: 'rgba(0, 0, 0, 0.2)'
                }}
              >
                {messages.length === 0 && (
                  <div style={{ textAlign: 'center', color: 'rgba(255, 255, 255, 0.45)', padding: '40px' }}>
                    <RobotOutlined style={{ fontSize: '48px', marginBottom: '16px', color: '#1890ff' }} />
                    <div>Start a conversation by asking a question about clinical trial data.</div>
                  </div>
                )}

                {messages.map((message) => (
                  <div
                    key={message.id}
                    style={{
                      marginBottom: '16px',
                      display: 'flex',
                      justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start'
                    }}
                  >
                    <div
                      style={{
                        maxWidth: '70%',
                        padding: '12px 16px',
                        borderRadius: '12px',
                        backgroundColor: message.type === 'user' ? '#1890ff' : '#1f1f1f',
                        color: 'white',
                        border: message.type === 'user' ? 'none' : '1px solid #303030',
                        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
                        <Avatar
                          size="small"
                          icon={message.type === 'user' ? <UserOutlined /> : <RobotOutlined />}
                          style={{ marginRight: '8px' }}
                        />
                        <Text strong style={{ color: 'white' }}>
                          {message.type === 'user' ? 'You' : 'AI Assistant'}
                        </Text>
                        <Text
                          type="secondary"
                          style={{
                            fontSize: '12px',
                            marginLeft: '8px',
                            color: 'rgba(255, 255, 255, 0.45)'
                          }}
                        >
                          {message.timestamp.toLocaleTimeString()}
                        </Text>
                      </div>

                      <div style={{ marginBottom: '8px' }}>
                        {message.content}
                      </div>

                      {/* Show insights for assistant messages */}
                      {message.type === 'assistant' && message.response && (
                        <div>
                          {message.response.insights.length > 0 && (
                            <div style={{ marginTop: '12px' }}>
                              <Text strong style={{ fontSize: '14px', marginBottom: '8px', display: 'block' }}>
                                <BulbOutlined style={{ marginRight: '4px' }} />
                                Key Insights:
                              </Text>
                              <List
                                size="small"
                                dataSource={message.response.insights}
                                renderItem={(insight: Insight) => (
                                  <List.Item style={{ padding: '8px 0' }}>
                                    <div style={{ width: '100%' }}>
                                      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
                                        <Tag color={getSeverityColor(insight.severity)} style={{ marginRight: '8px' }}>
                                          {insight.severity.toUpperCase()}
                                        </Tag>
                                        <Text strong>{insight.title}</Text>
                                      </div>
                                      <Text>{insight.description}</Text>
                                      {insight.confidence && (
                                        <div style={{ marginTop: '4px' }}>
                                          <Text type="secondary" style={{ fontSize: '12px' }}>
                                            Confidence: {(insight.confidence * 100).toFixed(0)}%
                                          </Text>
                                        </div>
                                      )}
                                    </div>
                                  </List.Item>
                                )}
                              />
                            </div>
                          )}

                          {/* Follow-up questions */}
                          {message.response.follow_up_questions.length > 0 && (
                            <div style={{ marginTop: '12px' }}>
                              <Text strong style={{ fontSize: '14px', marginBottom: '8px', display: 'block' }}>
                                <QuestionCircleOutlined style={{ marginRight: '4px' }} />
                                Suggested Follow-ups:
                              </Text>
                              <Space wrap>
                                {message.response.follow_up_questions.map((question, index) => (
                                  <Button
                                    key={index}
                                    type="link"
                                    size="small"
                                    onClick={() => handleFollowUpClick(question)}
                                  >
                                    {question}
                                  </Button>
                                ))}
                              </Space>
                            </div>
                          )}

                          {/* Processing time and confidence */}
                          <div style={{ marginTop: '8px', fontSize: '12px', color: '#999' }}>
                            Processing: {message.response.processing_time_ms.toFixed(0)}ms |
                            Confidence: {(message.response.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Spin size="large" />
                    <div style={{ marginTop: '8px', color: '#999' }}>Analyzing your query...</div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div style={{ display: 'flex', gap: '8px' }}>
                <TextArea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask a question about clinical trial data..."
                  autoSize={{ minRows: 1, maxRows: 4 }}
                  style={{ flex: 1 }}
                  disabled={loading}
                />
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={handleSend}
                  loading={loading}
                  disabled={!inputValue.trim() || loading}
                >
                  Send
                </Button>
              </div>
            </Card>
          </div>

          {/* Sidebar with capabilities and examples */}
          <div style={{ flex: 1 }}>
            <Card title="Capabilities" style={{ marginBottom: '16px' }}>
              <List
                size="small"
                dataSource={[
                  "Natural language understanding",
                  "Multi-turn conversations",
                  "RAG-powered insights",
                  "Cross-dataset analysis",
                  "Trend analysis",
                  "Correlation detection"
                ]}
                renderItem={(item) => <List.Item>{item}</List.Item>}
              />
            </Card>

            <Card title="Example Queries">
              <Space direction="vertical" style={{ width: '100%' }}>
                {exampleQueries.map((query, index) => (
                  <Button
                    key={index}
                    type="link"
                    block
                    style={{ textAlign: 'left' }}
                    onClick={() => sendQuery(query)}
                    disabled={loading}
                  >
                    {query}
                  </Button>
                ))}
              </Space>
            </Card>

            <Card title="Supported Metrics" style={{ marginTop: '16px' }}>
              <Space wrap>
                <Tag>missing_visits</Tag>
                <Tag>open_queries</Tag>
                <Tag>data_quality_index</Tag>
                <Tag>sae_count</Tag>
                <Tag>uncoded_terms</Tag>
                <Tag>frozen_crfs</Tag>
                <Tag>locked_crfs</Tag>
              </Space>
            </Card>
          </div>
        </div>
      </div>
    </Content>
  );
};

export default Conversational;