
import React from 'react';
import { Card, Tag, Button, Typography, Space } from 'antd';
import { RobotOutlined, CheckCircleOutlined, SafetyCertificateOutlined } from '@ant-design/icons';

const { Paragraph, Text } = Typography;

interface AISummaryProps {
    summary: string;
    source: string; // e.g., "Generated from Site 101 Data"
    citations?: string[];
    isLoading?: boolean;
}

export const AISummaryCard: React.FC<AISummaryProps> = ({ summary, source, citations = [], isLoading = false }) => {
    return (
        <Card
            loading={isLoading}
            title={
                <Space>
                    <RobotOutlined style={{ color: '#1890ff' }} />
                    <span>AI Insight Summary</span>
                    <Tag color="cyan">Beta</Tag>
                </Space>
            }
            extra={
                <Space>
                    <Tag icon={<SafetyCertificateOutlined color="#52c41a" />}>Guardrails Active</Tag>
                </Space>
            }
            style={{ background: '#f0f5ff', borderColor: '#d6e4ff' }}
            size="small"
        >
            <Paragraph style={{ fontSize: '14px', lineHeight: '1.6' }}>
                {summary}
            </Paragraph>

            <div style={{ marginTop: '12px', borderTop: '1px solid #d6e4ff', paddingTop: '8px' }}>
                <Space direction="vertical" size={0}>
                    <Text type="secondary" style={{ fontSize: '11px' }}>Source: {source}</Text>
                    {citations.length > 0 && (
                        <Text type="secondary" style={{ fontSize: '11px' }}>
                            Citations: {citations.join(', ')}
                        </Text>
                    )}
                </Space>
            </div>
        </Card>
    );
};
