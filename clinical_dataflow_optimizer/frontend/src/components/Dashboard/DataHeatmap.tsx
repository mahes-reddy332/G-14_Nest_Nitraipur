import { Card, Tooltip, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { metricsApi } from '../../api'
import { useStore } from '../../store'

interface HeatmapCellProps {
  value: number
  row: string
  col: string
  metric: string
}

function HeatmapCell({ value, row, col, metric }: HeatmapCellProps) {
  // Color scale from red to green
  const getColor = (val: number) => {
    if (val >= 80) return '#52c41a'
    if (val >= 60) return '#95de64'
    if (val >= 40) return '#fadb14'
    if (val >= 20) return '#ffa940'
    return '#ff4d4f'
  }

  const getTextColor = (val: number) => {
    return val >= 40 && val <= 60 ? '#262626' : '#fff'
  }

  return (
    <Tooltip title={`${row} - ${col}: ${value.toFixed(1)}%`}>
      <div
        className="heatmap-cell"
        style={{
          backgroundColor: getColor(value),
          color: getTextColor(value),
          width: '100%',
          height: 32,
        }}
      >
        {value.toFixed(0)}
      </div>
    </Tooltip>
  )
}

interface DataHeatmapProps {
  metric?: string
  title?: string
}

export default function DataHeatmap({ metric = 'dqi', title = 'Site DQI Heatmap' }: DataHeatmapProps) {
  const { selectedStudyId } = useStore()

  const { data: heatmapData, isLoading } = useQuery({
    queryKey: ['heatmap', metric, selectedStudyId],
    queryFn: () => metricsApi.getHeatmap(selectedStudyId || undefined, metric),
    refetchInterval: 120000,
  })

  if (isLoading || !heatmapData) {
    return <Card title={title} loading />
  }

  const rows = Array.isArray(heatmapData.rows) ? heatmapData.rows : []
  const columns = Array.isArray(heatmapData.columns) ? heatmapData.columns : []
  const values = Array.isArray(heatmapData.values) ? heatmapData.values : []

  if (rows.length === 0 || columns.length === 0) {
    return (
      <Card title={title}>
        <Empty description="No heatmap data available" />
      </Card>
    )
  }

  return (
    <Card title={title}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 2 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '8px 12px', minWidth: 100 }}></th>
              {columns.map((col) => (
                <th
                  key={col}
                  style={{
                    textAlign: 'center',
                    padding: '8px 4px',
                    fontSize: 11,
                    fontWeight: 500,
                    minWidth: 60,
                  }}
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={row}>
                <td
                  style={{
                    padding: '4px 12px',
                    fontSize: 12,
                    fontWeight: 500,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {row}
                </td>
                {columns.map((col, colIndex) => (
                  <td key={`${row}-${col}`} style={{ padding: 2 }}>
                    <HeatmapCell
                      value={values[rowIndex]?.[colIndex] ?? 0}
                      row={row}
                      col={col}
                      metric={metric}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div
        style={{
          marginTop: 16,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
        }}
      >
        <span style={{ fontSize: 11, color: '#8c8c8c' }}>Low</span>
        <div
          style={{
            display: 'flex',
            height: 12,
            borderRadius: 2,
            overflow: 'hidden',
          }}
        >
          {['#ff4d4f', '#ffa940', '#fadb14', '#95de64', '#52c41a'].map((color) => (
            <div key={color} style={{ width: 24, backgroundColor: color }} />
          ))}
        </div>
        <span style={{ fontSize: 11, color: '#8c8c8c' }}>High</span>
      </div>
    </Card>
  )
}
