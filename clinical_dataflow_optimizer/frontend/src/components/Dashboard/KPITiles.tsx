import { Card, Statistic, Row, Col, Tooltip } from 'antd'
import {
  ArrowUpOutlined,
  ArrowDownOutlined,
  MinusOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { metricsApi } from '../../api'
import { useStore } from '../../store'
import type { KPITile as KPITileType } from '../../types'

const statusColors = {
  good: '#52c41a',
  warning: '#faad14',
  critical: '#ff4d4f',
}

const statusIcons = {
  good: <CheckCircleOutlined />,
  warning: <ExclamationCircleOutlined />,
  critical: <CloseCircleOutlined />,
}

const trendIcons = {
  up: <ArrowUpOutlined />,
  down: <ArrowDownOutlined />,
  stable: <MinusOutlined />,
}

interface KPITileProps {
  tile: KPITileType
}

function KPITileCard({ tile }: KPITileProps) {
  const color = statusColors[tile.status]
  const trendColor = tile.trend === 'up' ? '#52c41a' : tile.trend === 'down' ? '#ff4d4f' : '#8c8c8c'

  return (
    <Card
      className="kpi-tile"
      hoverable
      style={{
        borderLeft: `4px solid ${color}`,
      }}
    >
      <Statistic
        title={
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {tile.title}
            <Tooltip title={`Status: ${tile.status}`}>
              <span style={{ color }}>{statusIcons[tile.status]}</span>
            </Tooltip>
          </span>
        }
        value={tile.value}
        suffix={tile.unit}
        valueStyle={{ color }}
      />
      <div style={{ marginTop: 8, color: trendColor, fontSize: 12 }}>
        {trendIcons[tile.trend]} {tile.trend_value > 0 ? '+' : ''}
        {tile.trend_value}% vs last period
      </div>
    </Card>
  )
}

export default function KPITiles() {
  const { selectedStudyId } = useStore()

  const { data: tiles = [], isLoading } = useQuery({
    queryKey: ['kpiTiles', selectedStudyId],
    queryFn: () => metricsApi.getKPITiles(selectedStudyId || undefined),
    refetchInterval: 60000,
  })

  if (isLoading) {
    return (
      <Row gutter={[16, 16]}>
        {[1, 2, 3, 4].map((i) => (
          <Col key={i} xs={24} sm={12} md={6}>
            <Card loading />
          </Col>
        ))}
      </Row>
    )
  }

  return (
    <Row gutter={[16, 16]}>
      {tiles.map((tile) => (
        <Col key={tile.id} xs={24} sm={12} md={6}>
          <KPITileCard tile={tile} />
        </Col>
      ))}
    </Row>
  )
}
