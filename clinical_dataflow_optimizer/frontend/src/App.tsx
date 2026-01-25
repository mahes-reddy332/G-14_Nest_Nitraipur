import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Spin, Result, Button } from 'antd'
import { useState, Suspense, lazy } from 'react'
import Sidebar from './components/Layout/Sidebar'
import Header from './components/Layout/Header'
import ErrorBoundary from './components/ErrorBoundary'
import { useWebSocket } from './hooks/useWebSocket'
import { useStore } from './store'
import ErrorNotifications from './components/ErrorNotifications'

// Lazy load pages for code splitting - reduces initial bundle size
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Studies = lazy(() => import('./pages/Studies'))
const Patients = lazy(() => import('./pages/Patients'))
const Sites = lazy(() => import('./pages/Sites'))
const Alerts = lazy(() => import('./pages/Alerts'))
const Agents = lazy(() => import('./pages/Agents'))
const Conversational = lazy(() => import('./pages/Conversational'))
const Reports = lazy(() => import('./pages/Reports'))

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
  const siderWidth = collapsed ? 80 : 200

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
      <Router>
        <Layout style={{ minHeight: '100vh' }}>
          <Sidebar collapsed={collapsed} onCollapse={setCollapsed} />
          <Layout style={{ marginLeft: siderWidth }}>
            <Header />
            <Content
              style={{
                margin: '24px 16px',
                padding: 24,
                minHeight: 280,
                background: '#f0f2f5',
                overflow: 'auto',
              }}
            >
              <ErrorBoundary onError={handleError}>
                <Suspense fallback={<PageLoader />}>
                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/studies" element={<Studies />} />
                    <Route path="/studies/:studyId" element={<Studies />} />
                    <Route path="/patients" element={<Patients />} />
                    <Route path="/patients/:patientId" element={<Patients />} />
                    <Route path="/sites" element={<Sites />} />
                    <Route path="/sites/:siteId" element={<Sites />} />
                    <Route path="/alerts" element={<Alerts />} />
                    <Route path="/agents" element={<Agents />} />
                    <Route path="/conversational" element={<Conversational />} />
                    <Route path="/reports" element={<Reports />} />
                  </Routes>
                </Suspense>
              </ErrorBoundary>
            </Content>
          </Layout>
        </Layout>
        <ErrorNotifications />
      </Router>
    </ErrorBoundary>
  )
}

export default App
