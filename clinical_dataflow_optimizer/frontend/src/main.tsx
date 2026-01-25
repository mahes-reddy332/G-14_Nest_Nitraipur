import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ConfigProvider } from 'antd'
import App from './App'
import ErrorBoundary from './components/ErrorBoundary'
import './index.css'
import './styles/clinical-design-system.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30000,
    },
  },
})

// Clinical-grade theme configuration
const clinicalTheme = {
  token: {
    // Primary brand color - clinical navy
    colorPrimary: '#1a365d',
    colorLink: '#2b6cb0',
    
    // Border radius for clinical look
    borderRadius: 8,
    borderRadiusLG: 12,
    borderRadiusSM: 4,
    
    // Typography
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    fontSize: 14,
    
    // Semantic colors (muted clinical tones)
    colorSuccess: '#2f855a',
    colorWarning: '#c05621',
    colorError: '#c53030',
    colorInfo: '#2b6cb0',
    
    // Background
    colorBgContainer: '#ffffff',
    colorBgLayout: '#f7fafc',
  },
  components: {
    Card: {
      borderRadiusLG: 12,
      boxShadowTertiary: '0 1px 3px 0 rgba(0, 0, 0, 0.08), 0 1px 2px 0 rgba(0, 0, 0, 0.04)',
    },
    Button: {
      borderRadius: 6,
    },
    Select: {
      borderRadius: 6,
    },
    Input: {
      borderRadius: 6,
    },
  },
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ConfigProvider theme={clinicalTheme}>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </ConfigProvider>
    </QueryClientProvider>
  </React.StrictMode>,
)
