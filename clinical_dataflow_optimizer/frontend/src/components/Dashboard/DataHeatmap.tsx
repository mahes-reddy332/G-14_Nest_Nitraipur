import { Card, Tooltip, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { metricsApi } from '../../api'
import { useStore } from '../../store'

interface HeatmapCellProps {
  value: number
  row: string
  col: string
  metric: string
  getColor: (value: number) => string
  getTextColor: (value: number) => string
  formatValue: (value: number) => string
}

function HeatmapCell({ value, row, col, metric, getColor, getTextColor, formatValue }: HeatmapCellProps) {
  return (
    <Tooltip title={`${row} - ${col}: ${formatValue(value)}`}>
      <div
        className="heatmap-cell"
        style={{
          backgroundColor: getColor(value),
          color: getTextColor(value),
          width: '100%',
          height: 32,
        }}
      >
        {Number.isFinite(value) ? value.toFixed(0) : '—'}
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
  const flatValues = values.flat().filter((val) => Number.isFinite(val)) as number[]
  const minValue = flatValues.length ? Math.min(...flatValues) : 0
  const maxValue = flatValues.length ? Math.max(...flatValues) : 0
  const colors = ['#ff4d4f', '#ffa940', '#fadb14', '#95de64', '#52c41a']
  const isPercentMetric = ['dqi', 'dqi_score', 'data_quality', 'cleanliness', 'clean_rate', 'risk'].includes(metric)
  const formatValue = (val: number) => {
    if (!Number.isFinite(val)) return 'N/A'
    return isPercentMetric ? `${val.toFixed(1)}%` : val.toFixed(1)
  }
  const getColor = (val: number) => {
    if (!Number.isFinite(val)) return '#f0f0f0'
    if (maxValue === minValue) return colors[2]
    const ratio = (val - minValue) / (maxValue - minValue)
    const index = Math.max(0, Math.min(colors.length - 1, Math.floor(ratio * (colors.length - 1))))
    return colors[index]
  }
  const getTextColor = (val: number) => {
    if (!Number.isFinite(val)) return '#8c8c8c'
    if (maxValue === minValue) return '#262626'
    const ratio = (val - minValue) / (maxValue - minValue)
    return ratio >= 0.35 && ratio <= 0.65 ? '#262626' : '#fff'
  }

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
                      getColor={getColor}
                      getTextColor={getTextColor}
                      formatValue={formatValue}
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
        <span style={{ fontSize: 11, color: '#8c8c8c' }}>{formatValue(minValue)}</span>
        <div
          style={{
            display: 'flex',
            height: 12,
            borderRadius: 2,
            overflow: 'hidden',
          }}
        >
          {colors.map((color) => (
            <div key={color} style={{ width: 24, backgroundColor: color }} />
          ))}
        </div>
        <span style={{ fontSize: 11, color: '#8c8c8c' }}>{formatValue(maxValue)}</span>
      </div>
    </Card>
  )
}
