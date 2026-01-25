import { Layout, Badge, Select, Space, Button, Dropdown, Avatar, Tooltip, Spin } from 'antd'
import {
  BellOutlined,
  UserOutlined,
  SyncOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  CloudSyncOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { studiesApi, alertsApi } from '../../api'
import { useStore } from '../../store'
import { useWebSocket, type ConnectionState } from '../../hooks/useWebSocket'
import type { Study } from '../../types'

const { Header: AntHeader } = Layout

// Connection status display component
function ConnectionStatus({ connectionState, isConnected }: { connectionState: ConnectionState; isConnected: boolean }) {
  const statusConfig: Record<ConnectionState, { color: string; icon: React.ReactNode; text: string; tooltip: string }> = {
    'connected': {
      color: '#52c41a',
      icon: <CheckCircleOutlined />,
      text: 'Connected',
      tooltip: 'Real-time connection active'
    },
    'connecting': {
      color: '#1890ff',
      icon: <LoadingOutlined spin />,
      text: 'Connecting...',
      tooltip: 'Establishing connection to server'
    },
    'waiting_for_backend': {
      color: '#faad14',
      icon: <CloudSyncOutlined />,
      text: 'Starting...',
      tooltip: 'Waiting for backend services to initialize'
    },
    'disconnected': {
      color: '#ff4d4f',
      icon: <CloseCircleOutlined />,
      text: 'Disconnected',
      tooltip: 'Connection lost. Reconnecting automatically...'
    }
  }

  // Use isConnected from store as final authority
  const effectiveState = isConnected ? 'connected' : connectionState
  const config = statusConfig[effectiveState]

  return (
    <Tooltip title={config.tooltip}>
      <div className="realtime-indicator" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {effectiveState === 'connected' && <div className="realtime-dot" />}
        <span style={{ color: config.color, display: 'flex', alignItems: 'center', gap: 4 }}>
          {config.icon} {config.text}
        </span>
      </div>
    </Tooltip>
  )
}

export default function Header() {
  const { isConnected, selectedStudyId, setSelectedStudyId, unreadNotifications, resetNotifications } =
    useStore()
  
  const { connectionState, forceReconnect } = useWebSocket()

  // Fetch studies for selector
  const { data: studies = [] } = useQuery({
    queryKey: ['studies'],
    queryFn: studiesApi.getAll,
  })

  // Fetch alert summary
  const { data: alertSummary } = useQuery({
    queryKey: ['alertSummary'],
    queryFn: alertsApi.getSummary,
    refetchInterval: 30000,
  })

  const studyOptions = [
    { value: '', label: 'All Studies' },
    ...studies.map((s: Study) => ({ value: s.study_id, label: s.name || s.study_id })),
  ]

  const userMenuItems = [
    { key: 'profile', label: 'Profile' },
    { key: 'settings', label: 'Settings' },
    { type: 'divider' as const },
    { key: 'logout', label: 'Logout' },
  ]

  return (
    <AntHeader
      style={{
        padding: '0 24px',
        background: '#fff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
      }}
    >
      <Space size="large">
        <Select
          style={{ width: 200 }}
          placeholder="Select Study"
          value={selectedStudyId || ''}
          onChange={(value: string) => setSelectedStudyId(value || null)}
          options={studyOptions}
        />

        <ConnectionStatus connectionState={connectionState} isConnected={isConnected} />
      </Space>

      <Space size="middle">
        <Tooltip title="Force reconnect to server">
          <Button
            type="text"
            icon={<SyncOutlined />}
            onClick={forceReconnect}
          >
            Reconnect
          </Button>
        </Tooltip>

        <Badge
          count={alertSummary?.by_severity?.critical || 0}
          style={{ backgroundColor: '#ff4d4f' }}
        >
          <Badge count={unreadNotifications} offset={[0, 0]}>
            <Button
              type="text"
              icon={<BellOutlined style={{ fontSize: 20 }} />}
              onClick={resetNotifications}
            />
          </Badge>
        </Badge>

        <Dropdown menu={{ items: userMenuItems }}>
          <Avatar icon={<UserOutlined />} style={{ cursor: 'pointer' }} />
        </Dropdown>
      </Space>
    </AntHeader>
  )
}
