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
import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { studiesApi, alertsApi } from '../../api'
import { useStore } from '../../store'
import type { ConnectionState } from '../../types'
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
    'retrying': {
      color: '#faad14',
      icon: <CloudSyncOutlined />,
      text: 'Retrying...',
      tooltip: 'Retrying connection with backoff'
    },
    'failed': {
      color: '#ff4d4f',
      icon: <CloseCircleOutlined />,
      text: 'Failed',
      tooltip: 'Connection failed. Please retry.'
    },
    'disconnected': {
      color: '#ff4d4f',
      icon: <CloseCircleOutlined />,
      text: 'Disconnected',
      tooltip: 'Connection lost. Reconnecting automatically...'
    },
    'idle': {
      color: '#8c8c8c',
      icon: <CloseCircleOutlined />,
      text: 'Idle',
      tooltip: 'Connection idle'
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
  const navigate = useNavigate()
  const {
    isConnected,
    selectedStudyId,
    setSelectedStudyId,
    unreadNotifications,
    resetNotifications,
    connectionState,
    requestReconnect,
  } =
    useStore()

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

  const studyOptions = useMemo(
    () => [
      { value: '', label: 'All Studies' },
      ...studies.map((s: Study) => ({ value: s.study_id, label: s.name || s.study_id })),
    ],
    [studies]
  )

  const userMenuItems = useMemo(
    () => [
      { key: 'profile', label: 'Profile' },
      { key: 'settings', label: 'Settings' },
      { type: 'divider' as const },
      { key: 'logout', label: 'Logout' },
    ],
    []
  )

  const handleMenuClick = ({ key }: { key: string }) => {
    switch (key) {
      case 'profile':
        navigate('/profile')
        break
      case 'settings':
        navigate('/settings')
        break
      case 'logout':
        // Clear token logic if implemented, or just redirect
        // localStorage.removeItem('token')
        navigate('/login')
        break
    }
  }

  return (
    <AntHeader
      style={{
        padding: '0 24px',
        background: 'transparent',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end', // Move everything to the right
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
            onClick={requestReconnect}
            aria-label="Reconnect real-time data"
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
              onClick={() => {
                resetNotifications()
                navigate('/alerts')
              }}
              aria-label="Open notifications"
            />
          </Badge>
        </Badge>

        <Dropdown menu={{ items: userMenuItems, onClick: handleMenuClick }}>
          <Avatar icon={<UserOutlined />} style={{ cursor: 'pointer' }} />
        </Dropdown>
      </Space>
    </AntHeader>
  )
}
