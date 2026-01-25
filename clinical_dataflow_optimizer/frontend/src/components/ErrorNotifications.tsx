import React, { useEffect } from 'react'
import { Alert, notification, Button, Space } from 'antd'
import { CloseOutlined, ExclamationCircleOutlined } from '@ant-design/icons'
import { useStore } from '../store'

const ErrorNotifications: React.FC = () => {
  const { errors, clearError, clearAllErrors } = useStore()

  useEffect(() => {
    errors.forEach((error, index) => {
      // Show notification for new errors
      notification.error({
        message: 'Error Occurred',
        description: error.message,
        duration: 5000, // Auto close after 5 seconds
        key: `error-${index}-${error.timestamp.getTime()}`,
        icon: <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />,
        onClose: () => clearError(index),
      })
    })
  }, [errors, clearError])

  if (errors.length === 0) {
    return null
  }

  return (
    <div
      style={{
        position: 'fixed',
        top: '70px', // Below header
        right: '20px',
        zIndex: 1000,
        maxWidth: '400px',
      }}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        {errors.slice(0, 3).map((error, index) => (
          <Alert
            key={index}
            message="Error"
            description={
              <div>
                <div style={{ marginBottom: '8px' }}>{error.message}</div>
                {error.code && (
                  <div style={{ fontSize: '12px', color: '#8c8c8c' }}>
                    Code: {error.code}
                  </div>
                )}
                <div style={{ fontSize: '12px', color: '#8c8c8c' }}>
                  {error.timestamp.toLocaleTimeString()}
                </div>
              </div>
            }
            type="error"
            closable
            onClose={() => clearError(index)}
            action={
              <Button
                size="small"
                type="text"
                icon={<CloseOutlined />}
                onClick={() => clearError(index)}
              />
            }
            style={{
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
              borderRadius: '6px',
            }}
          />
        ))}
        {errors.length > 3 && (
          <Button
            size="small"
            type="link"
            onClick={clearAllErrors}
            style={{ padding: 0 }}
          >
            Clear all {errors.length} errors
          </Button>
        )}
      </Space>
    </div>
  )
}

export default ErrorNotifications