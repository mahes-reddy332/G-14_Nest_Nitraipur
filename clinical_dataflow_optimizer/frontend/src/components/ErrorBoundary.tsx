import React, { Component, ErrorInfo, ReactNode } from 'react'
import { Result, Button, Typography, Space } from 'antd'
import { ExclamationCircleOutlined, ReloadOutlined, HomeOutlined } from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)

    this.setState({
      error,
      errorInfo,
    })

    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    }

    // Report error to backend (if implemented)
    this.reportError(error, errorInfo)
  }

  reportError = async (error: Error, errorInfo: ErrorInfo) => {
    try {
      // TODO: Implement error reporting to backend
      // await api.post('/errors/report', {
      //   message: error.message,
      //   stack: error.stack,
      //   componentStack: errorInfo.componentStack,
      //   timestamp: new Date().toISOString(),
      //   userAgent: navigator.userAgent,
      //   url: window.location.href,
      // })
    } catch (reportError) {
      console.error('Failed to report error:', reportError)
    }
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined })
  }

  handleGoHome = () => {
    window.location.href = '/'
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div style={{
          minHeight: '400px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px'
        }}>
          <Result
            status="error"
            icon={<ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />}
            title={
              <Title level={3} style={{ color: '#ff4d4f', marginBottom: 8 }}>
                Something went wrong
              </Title>
            }
            subTitle={
              <Space direction="vertical" size="small">
                <Text type="secondary">
                  An unexpected error occurred while loading this page.
                </Text>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Error: {this.state.error?.message || 'Unknown error'}
                </Text>
              </Space>
            }
            extra={
              <Space>
                <Button
                  type="primary"
                  icon={<ReloadOutlined />}
                  onClick={this.handleRetry}
                >
                  Try Again
                </Button>
                <Button
                  icon={<HomeOutlined />}
                  onClick={this.handleGoHome}
                >
                  Go Home
                </Button>
              </Space>
            }
          >
            {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
              <div style={{ marginTop: 16, textAlign: 'left' }}>
                <Paragraph strong>Development Error Details:</Paragraph>
                <pre style={{
                  background: '#f5f5f5',
                  padding: '12px',
                  borderRadius: '4px',
                  fontSize: '12px',
                  overflow: 'auto',
                  maxHeight: '200px'
                }}>
                  {this.state.error?.stack}
                  {'\n\nComponent Stack:\n'}
                  {this.state.errorInfo.componentStack}
                </pre>
              </div>
            )}
          </Result>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
