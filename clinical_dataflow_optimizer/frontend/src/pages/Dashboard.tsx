import { Typography } from 'antd'
import {
  AlertsPanel,
  CleanlinessGauge,
  EnhancedDQIBreakdown,
  EnhancedKPISections,
  EnhancedQueryStatus,
  GlobalFilters,
  QueryVelocityChart,
} from '../components/Dashboard'
import { useStore } from '../store'
import '../styles/clinical-design-system.css'
import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

const { Title } = Typography

export default function Dashboard() {
  const { isConnected } = useStore()
  const queryClient = useQueryClient()
  const [lastUpdated, setLastUpdated] = useState<string>(new Date().toISOString())

  const handleRefresh = () => {
    // Invalidate all dashboard queries
    queryClient.invalidateQueries({ queryKey: ['dashboardSummary'] })
    queryClient.invalidateQueries({ queryKey: ['queryMetrics'] })
    queryClient.invalidateQueries({ queryKey: ['cleanlinessMetrics'] })
    queryClient.invalidateQueries({ queryKey: ['kpiTiles'] })
    queryClient.invalidateQueries({ queryKey: ['dqiBreakdown'] })
    queryClient.invalidateQueries({ queryKey: ['alertSummary'] })
    setLastUpdated(new Date().toISOString())
  }

  return (
    <div className="clinical-dashboard">
      <div
        style={{
          maxWidth: 1600, // Increased max-width for the grid
          margin: '0 auto',
          padding: '16px 24px 32px',
        }}
      >
        {/* Global Filters - Sticky pill-style bar */}
        <GlobalFilters onRefresh={handleRefresh} lastUpdated={lastUpdated} />

        <div className="dashboard-grid">
          {/* Row 1: Priority Alert (2 cols), Charts (1 col each) */}
          <div className="grid-item-span-2">
            <AlertsPanel />
          </div>
          <div className="grid-item-span-1">
            {/* Alert Trends Chart */}
            <div className="glass-panel" style={{ height: '100%', padding: 20, display: 'flex', flexDirection: 'column' }}>
              <Title level={5} style={{ color: 'var(--neon-cyan)', margin: '0 0 16px 0', fontFamily: 'var(--font-display)' }}>
                Alert Trends
              </Title>
              <div style={{ flex: 1, minHeight: 200 }}>
                <div style={{ flex: 1, minHeight: 200 }}>
                  <QueryVelocityChart embedded={true} />
                </div>
              </div>
            </div>
          </div>
          <div className="grid-item-span-1">
            {/* Total Patients Card */}
            <div className="glass-panel" style={{ height: '100%', padding: 24, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <Title level={5} style={{ color: 'var(--neon-cyan)', margin: 0, textTransform: 'uppercase', letterSpacing: 1 }}>Total Patients</Title>
              <div style={{ fontSize: 48, fontWeight: 'bold', color: '#fff', textShadow: '0 0 20px rgba(255, 255, 255, 0.3)', marginTop: 8 }}>
                57,997
              </div>
              <div style={{ marginTop: 'auto' }}>
                <div style={{ marginTop: 'auto' }}>
                  <div style={{ height: 60, display: 'flex', alignItems: 'flex-end', gap: 2 }}>
                    {[4, 7, 5, 9, 8, 5, 2, 4, 3, 6, 8, 10, 5, 7, 3].map((h, i) => (
                      <div key={i} style={{ flex: 1, height: `${h * 10}%`, background: 'var(--clinical-primary)', opacity: 0.3 + (i * 0.05), borderRadius: '2px 2px 0 0' }} />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Row 2: Stats Cards */}
          <div className="grid-item-span-4">
            <EnhancedKPISections />
          </div>

          {/* Row 3: Protocol Deviations */}
          <div className="grid-item-span-4">
            <div className="glass-panel" style={{ height: '100%', padding: 0 }}>
              <EnhancedQueryStatus />
            </div>
          </div>

          {/* Row 3: Data Quality */}
          <div className="grid-item-span-2">
            <EnhancedDQIBreakdown />
          </div>
          <div className="grid-item-span-2">
            <CleanlinessGauge />
          </div>
        </div>
      </div>
    </div>
  )
}
