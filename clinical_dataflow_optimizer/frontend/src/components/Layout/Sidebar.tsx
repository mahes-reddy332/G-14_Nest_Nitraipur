import { Layout, Menu, type MenuProps } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import { BarChartOutlined, LeftOutlined, RightOutlined } from '@ant-design/icons'
import { routes, RouteConfig } from '../../config/routes'

const { Sider } = Layout

type MenuItem = Required<MenuProps>['items'][number]

export default function Sidebar({ collapsed, onCollapse }: { collapsed: boolean; onCollapse: (c: boolean) => void }) {
  const navigate = useNavigate()
  const location = useLocation()

  // Helper to convert route config to menu items
  const generateMenuItems = (routeList: RouteConfig[]): MenuItem[] => {
    const items: MenuItem[] = []

    routeList.forEach(route => {
      if (route.hideInMenu) return

      // If it has children that are meant to be in menu
      const visibleChildren = route.children?.filter(c => !c.hideInMenu)

      if (visibleChildren && visibleChildren.length > 0) {
        items.push({
          key: route.key,
          icon: route.icon,
          label: route.label,
          children: generateMenuItems(visibleChildren),
        })
      } else {
        items.push({
          key: route.key,
          icon: route.icon,
          label: route.label,
        })
      }
    })

    return items
  }

  const menuItems = generateMenuItems(routes)

  // Find the currently selected key and its parent for menu highlighting
  // This needs to map the current path back to the route key
  const getSelectedKeys = () => {
    const path = location.pathname
    // Simple direct match first
    return [path]
  }

  // Calculate open keys based on path hierarchy
  const getOpenKeys = () => {
    const path = location.pathname
    const openKeys: string[] = []

    routes.forEach(route => {
      if (route.children) {
        // If current path starts with a child path or equals it
        const hasChild = route.children.some(c =>
          path === c.path || path.startsWith(c.path.replace('/*', ''))
        )
        if (hasChild || path.startsWith(route.path)) {
          openKeys.push(route.key)
        }
      }
    })

    return openKeys
  }

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={onCollapse}
      width={240}
      trigger={null} // Disable default absolute trigger
      className="sidebar-container"
      style={{
        height: 'calc(100vh - 48px)', // Floating effect
        position: 'fixed',
        left: 24, // Floating offset
        top: 24,
        bottom: 24,
        borderRadius: 24, // Rounded corners
        zIndex: 1000,
        // background is handled by className in index.css
        overflow: 'hidden', // Hide Sider-level overflow, handle internally
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Logo Header */}
        <div
          style={{
            height: 64,
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'flex-start',
            paddingLeft: collapsed ? 0 : 24,
            flexShrink: 0,
          }}
        >
          <BarChartOutlined style={{ fontSize: 28, color: '#00f3ff' }} />
          {!collapsed && (
            <span
              className="font-display"
              style={{
                color: '#fff',
                fontSize: 18,
                fontWeight: 600,
                marginLeft: 12,
                letterSpacing: '1px',
                textShadow: '0 0 10px rgba(0, 243, 255, 0.5)'
              }}
            >
              NCDM
            </span>
          )}
        </div>

        {/* Scrollable Menu Area */}
        <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden' }}>
          <Menu
            theme="dark"
            mode="inline"
            selectedKeys={getSelectedKeys()}
            defaultOpenKeys={collapsed ? [] : getOpenKeys()}
            items={menuItems}
            onClick={({ key }) => {
              // We can rely on the key being the path for leaf nodes in our new structure
              const findRoute = (list: RouteConfig[]): boolean => {
                for (const r of list) {
                  if (r.key === key) {
                    if (r.children && r.children.some(c => !c.hideInMenu)) {
                      return false // It's a parent menu, don't navigate
                    }
                    if (r.path) {
                      navigate(r.path)
                      return true
                    }
                  }
                  if (r.children) {
                    if (findRoute(r.children)) return true
                  }
                }
                return false
              }
              findRoute(routes)
            }}
            style={{
              background: 'transparent',
              borderRight: 0,
            }}
          />
        </div>

        {/* Custom Footer Trigger */}
        <div
          onClick={() => onCollapse(!collapsed)}
          style={{
            height: 48,
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            borderTop: '1px solid rgba(255, 255, 255, 0.1)',
            color: 'rgba(255, 255, 255, 0.65)',
            transition: 'color 0.3s',
          }}
          className="sidebar-trigger"
        >
          {collapsed ? <RightOutlined /> : <LeftOutlined />}
        </div>
      </div>
    </Sider>
  )
}
