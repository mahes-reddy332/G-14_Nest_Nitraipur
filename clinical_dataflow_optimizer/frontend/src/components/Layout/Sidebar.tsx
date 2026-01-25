import { Layout, Menu } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  DashboardOutlined,
  ExperimentOutlined,
  UserOutlined,
  BankOutlined,
  AlertOutlined,
  RobotOutlined,
  SettingOutlined,
  MessageOutlined,
  FileTextOutlined,
} from '@ant-design/icons'

const { Sider } = Layout

const menuItems = [
  {
    key: '/dashboard',
    icon: <DashboardOutlined />,
    label: 'Dashboard',
  },
  {
    key: '/studies',
    icon: <ExperimentOutlined />,
    label: 'Studies',
  },
  {
    key: '/patients',
    icon: <UserOutlined />,
    label: 'Patients',
  },
  {
    key: '/sites',
    icon: <BankOutlined />,
    label: 'Sites',
  },
  {
    key: '/alerts',
    icon: <AlertOutlined />,
    label: 'Alerts',
  },
  {
    key: '/agents',
    icon: <RobotOutlined />,
    label: 'AI Agents',
  },
  {
    key: '/conversational',
    icon: <MessageOutlined />,
    label: 'Chatbot',
  },
  {
    key: '/reports',
    icon: <FileTextOutlined />,
    label: 'Reports',
  },
]

interface SidebarProps {
  collapsed: boolean
  onCollapse: (collapsed: boolean) => void
}

export default function Sidebar({ collapsed, onCollapse }: SidebarProps) {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={onCollapse}
      style={{
        overflow: 'auto',
        height: '100vh',
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
      }}
    >
      <div
        style={{
          height: 64,
          margin: 16,
          display: 'flex',
          alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'flex-start',
        }}
      >
        <SettingOutlined style={{ fontSize: 24, color: '#fff' }} />
        {!collapsed && (
          <span
            style={{
              color: '#fff',
              fontSize: 16,
              fontWeight: 'bold',
              marginLeft: 12,
            }}
          >
            Clinical DM
          </span>
        )}
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={({ key }: { key: string }) => navigate(key)}
      />
    </Sider>
  )
}
