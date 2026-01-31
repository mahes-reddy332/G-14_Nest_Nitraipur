
import React from 'react';
import { Card, Tooltip } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';

interface HeatmapProps {
    data: Array<{
        site_id: string;
        risk_score: number;
        risk_level: 'Low' | 'Medium' | 'High';
        region: string;
    }>;
    title?: string;
}

export const RiskHeatmap: React.FC<HeatmapProps> = ({ data, title = "Site Risk Heatmap" }) => {

    const getColor = (score: number) => {
        // Gradient from Green to Red
        // Simple implementation: 0-3 Green, 3-7 Orange, 7+ Red
        if (score > 7) return '#ff4d4f'; // Red
        if (score > 3) return '#faad14'; // Orange
        return '#52c41a'; // Green
    };

    return (
        <Card
            title={
                <span>
                    {title}
                    <Tooltip title="Visual representation of site risk scores. Red indicates high risk requiring immediate attention.">
                        <InfoCircleOutlined style={{ marginLeft: 8, color: '#999' }} />
                    </Tooltip>
                </span>
            }
            size="small"
        >
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: '8px' }}>
                {data.map((site) => (
                    <Tooltip
                        key={site.site_id}
                        title={
                            <div>
                                <strong>{site.site_id}</strong><br />
                                Score: {site.risk_score}<br />
                                Region: {site.region}
                            </div>
                        }
                    >
                        <div
                            style={{
                                height: '60px',
                                backgroundColor: getColor(site.risk_score),
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: 'white',
                                fontWeight: 'bold',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '12px'
                            }}
                        >
                            {site.site_id}
                        </div>
                    </Tooltip>
                ))}
            </div>
            <div style={{ marginTop: '16px', display: 'flex', gap: '16px', fontSize: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <div style={{ width: 12, height: 12, backgroundColor: '#52c41a', borderRadius: 2 }} /> Low Risk
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <div style={{ width: 12, height: 12, backgroundColor: '#faad14', borderRadius: 2 }} /> Medium Risk
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <div style={{ width: 12, height: 12, backgroundColor: '#ff4d4f', borderRadius: 2 }} /> High Risk
                </div>
            </div>
        </Card>
    );
};
