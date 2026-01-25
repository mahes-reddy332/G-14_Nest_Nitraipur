import React from 'react'
import { Spin, Progress, Typography, Space } from 'antd'
import { LoadingOutlined } from '@ant-design/icons'

const { Text } = Typography

interface LoadingSpinnerProps {
  size?: 'small' | 'default' | 'large'
  tip?: string
  progress?: number
  showProgress?: boolean
  fullScreen?: boolean
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'default',
  tip,
  progress,
  showProgress = false,
  fullScreen = false,
}) => {
  const spinner = (
    <Space direction="vertical" align="center" size="middle">
      {showProgress && progress !== undefined ? (
        <Progress
          type="circle"
          percent={Math.round(progress)}
          size={size === 'large' ? 80 : size === 'small' ? 40 : 60}
          status={progress >= 100 ? 'success' : 'active'}
        />
      ) : (
        <Spin
          size={size}
          indicator={<LoadingOutlined spin />}
          tip={tip}
        />
      )}
      {tip && (
        <Text type="secondary" style={{ fontSize: '14px' }}>
          {tip}
        </Text>
      )}
    </Space>
  )

  if (fullScreen) {
    return (
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          zIndex: 9999,
        }}
      >
        {spinner}
      </div>
    )
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '200px',
        padding: '20px',
      }}
    >
      {spinner}
    </div>
  )
}

export default LoadingSpinner