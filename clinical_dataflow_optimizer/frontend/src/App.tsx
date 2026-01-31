import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Spin, Result, Button, ConfigProvider, theme } from 'antd'
import { useState, Suspense, lazy } from 'react'
import Sidebar from './components/Layout/Sidebar'
import Header from './components/Layout/Header'
import ErrorBoundary from './components/ErrorBoundary'
import { useWebSocket } from './hooks/useWebSocket'
import { useStore } from './store'
import ErrorNotifications from './components/ErrorNotifications'

import { routes, RouteConfig } from './config/routes'

const { Content } = Layout

// Loading spinner for lazy-loaded pages
function PageLoader() {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: 400,
      padding: 48
    }}>
      <Spin size="large" tip="Loading page..." />
    </div>
  )
}

// Fallback UI for error boundary
function ErrorFallback({ error, resetError }: { error: Error; resetError: () => void }) {
  return (
    <Result
      status="error"
      title="Something went wrong"
      subTitle={error.message}
      extra={[
        <Button type="primary" key="retry" onClick={resetError}>
          Try Again
        </Button>,
        <Button key="home" onClick={() => window.location.href = '/'}>
          Go to Dashboard
        </Button>
      ]}
    />
  )
}

function App() {
  // Initialize WebSocket connection for real-time updates
  useWebSocket()

  const [collapsed, setCollapsed] = useState(false)
  // Keep this in sync with `Sidebar`'s `<Sider width={240} />` and antd's default `collapsedWidth={80}`
  const siderWidth = collapsed ? 80 : 240
  const siderGutter = 16

  const addError = useStore(state => state.addError)

  const handleError = (error: Error, errorInfo: React.ErrorInfo) => {
    addError({
      message: error.message,
      code: 'UI_ERROR',
      timestamp: new Date(),
      context: 'App Component',
    })
  }

  return (
    <ErrorBoundary onError={handleError}>
      <ConfigProvider
        theme={{
          algorithm: theme.darkAlgorithm,
          token: {
            colorPrimary: '#00f3ff', // Neon Cyan
            colorBgBase: '#050510',
            colorBgContainer: '#141423',
            colorInfo: '#00f3ff',
            colorSuccess: '#00ff99',
            colorWarning: '#fcee0a',
            colorError: '#ff3333',
            fontFamily: 'Inter, sans-serif',
            borderRadius: 8,
          },
          components: {
            Layout: {
              bodyBg: 'transparent',
              headerBg: 'transparent',
              siderBg: 'rgba(5, 5, 16, 0.5)',
            },
            Menu: {
              darkItemBg: 'transparent',
              darkItemColor: 'rgba(255, 255, 255, 0.65)',
              darkItemSelectedBg: 'rgba(0, 243, 255, 0.1)',
              darkItemSelectedColor: '#00f3ff',
            },
            Card: {
              colorBgContainer: 'rgba(20, 20, 35, 0.6)',
            }
          }
        }}
      >
        <Router>
          <Layout style={{ minHeight: '100vh', background: 'transparent' }}>
            <Sidebar collapsed={collapsed} onCollapse={setCollapsed} />
            <Layout style={{ marginLeft: siderWidth + siderGutter, background: 'transparent' }}>
              <Header />
              <Content
                style={{
                  margin: '24px 24px',
                  padding: 0,
                  minHeight: 280,
                  background: 'transparent',
                  overflow: 'visible', // Changed from auto to visible for better scrolling with floating elements
                }}
              >
                <ErrorBoundary onError={handleError}>
                  <Suspense fallback={<PageLoader />}>
                    <Routes>
                      <Route path="/" element={<Navigate to="/dashboard" replace />} />
                      {routes.map(route => {
                        const renderRoute = (r: RouteConfig): React.ReactNode[] => {
                          const nodes: React.ReactNode[] = []

                          if (r.redirect) {
                            nodes.push(<Route key={r.key} path={r.path} element={<Navigate to={r.redirect} replace />} />)
                          } else if (r.element) {
                            // Only render element if it's NOT a redirect (unless we specifically want both, but typically redirect takes precedence for the exact path match)
                            nodes.push(<Route key={r.key} path={r.path} element={<r.element />} />)
                          }

                          if (r.children) {
                            r.children.forEach(child => {
                              nodes.push(...renderRoute(child))
                            })
                          }
                          return nodes
                        }

                        // Flatten the tree for Router as we are mostly using absolute paths in the registry keys/paths
                        // except where explicitly nested.
                        // Our registry design used absolute paths for everything for clarity.
                        return renderRoute(route)
                      })}
                    </Routes>
                  </Suspense>
                </ErrorBoundary>
              </Content>
            </Layout>
          </Layout>
          <ErrorNotifications />
        </Router>
      </ConfigProvider>
    </ErrorBoundary>
  )
}

export default App
