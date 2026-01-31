/**
 * Error Message Component
 * Reusable error display with retry functionality
 */

import React from 'react'
import { AlertCircle, RefreshCw, XCircle, AlertTriangle, Info } from 'lucide-react'

interface ErrorMessageProps {
  title?: string
  message: string
  variant?: 'error' | 'warning' | 'info'
  onRetry?: () => void
  retryLabel?: string
  fullScreen?: boolean
  className?: string
  showIcon?: boolean
  dismissible?: boolean
  onDismiss?: () => void
}

const variantStyles = {
  error: {
    container: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
    icon: 'text-red-500',
    title: 'text-red-800 dark:text-red-200',
    message: 'text-red-700 dark:text-red-300',
    button: 'bg-red-600 hover:bg-red-700 text-white',
  },
  warning: {
    container: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
    icon: 'text-yellow-500',
    title: 'text-yellow-800 dark:text-yellow-200',
    message: 'text-yellow-700 dark:text-yellow-300',
    button: 'bg-yellow-600 hover:bg-yellow-700 text-white',
  },
  info: {
    container: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    icon: 'text-blue-500',
    title: 'text-blue-800 dark:text-blue-200',
    message: 'text-blue-700 dark:text-blue-300',
    button: 'bg-blue-600 hover:bg-blue-700 text-white',
  },
}

const icons = {
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  title,
  message,
  variant = 'error',
  onRetry,
  retryLabel = 'Try Again',
  fullScreen = false,
  className = '',
  showIcon = true,
  dismissible = false,
  onDismiss,
}) => {
  const styles = variantStyles[variant]
  const IconComponent = icons[variant]

  const content = (
    <div
      className={`relative rounded-lg border p-4 ${styles.container} ${className}`}
      role="alert"
    >
      <div className="flex items-start gap-3">
        {showIcon && (
          <IconComponent
            className={`h-5 w-5 flex-shrink-0 mt-0.5 ${styles.icon}`}
          />
        )}
        <div className="flex-1 min-w-0">
          {title && (
            <h3 className={`text-sm font-medium mb-1 ${styles.title}`}>
              {title}
            </h3>
          )}
          <p className={`text-sm ${styles.message}`}>{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className={`mt-3 inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${styles.button}`}
            >
              <RefreshCw className="h-4 w-4" />
              {retryLabel}
            </button>
          )}
        </div>
        {dismissible && onDismiss && (
          <button
            onClick={onDismiss}
            className="flex-shrink-0 p-1 rounded hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
            aria-label="Dismiss"
          >
            <XCircle className="h-4 w-4 text-gray-400" />
          </button>
        )}
      </div>
    </div>
  )

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm p-4">
        <div className="max-w-md w-full">{content}</div>
      </div>
    )
  }

  return content
}

/**
 * Empty State Component
 * For when there's no data to display
 */
interface EmptyStateProps {
  icon?: React.ComponentType<{ className?: string }>
  title: string
  description?: string
  action?: {
    label: string
    onClick: () => void
  }
  className?: string
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon: Icon = AlertCircle,
  title,
  description,
  action,
  className = '',
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center p-8 text-center ${className}`}
    >
      <Icon className="h-12 w-12 text-gray-400 dark:text-gray-500 mb-4" />
      <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-1">
        {title}
      </h3>
      {description && (
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4 max-w-sm">
          {description}
        </p>
      )}
      {action && (
        <button
          onClick={action.onClick}
          className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-md transition-colors"
        >
          {action.label}
        </button>
      )}
    </div>
  )
}

/**
 * Error Boundary Fallback Component
 */
interface ErrorBoundaryFallbackProps {
  error: Error
  resetErrorBoundary?: () => void
}

export const ErrorBoundaryFallback: React.FC<ErrorBoundaryFallbackProps> = ({
  error,
  resetErrorBoundary,
}) => {
  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gray-50 dark:bg-gray-900">
      <div className="max-w-md w-full text-center">
        <XCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
          Something went wrong
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          An unexpected error occurred. Please try again.
        </p>
        {process.env.NODE_ENV === 'development' && (
          <pre className="text-left text-xs bg-gray-100 dark:bg-gray-800 p-4 rounded-lg mb-4 overflow-auto max-h-40">
            {error.message}
            {error.stack && `\n\n${error.stack}`}
          </pre>
        )}
        <div className="flex gap-3 justify-center">
          {resetErrorBoundary && (
            <button
              onClick={resetErrorBoundary}
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-md transition-colors"
            >
              <RefreshCw className="h-4 w-4" />
              Try Again
            </button>
          )}
          <button
            onClick={() => window.location.reload()}
            className="inline-flex items-center px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 rounded-md transition-colors"
          >
            Reload Page
          </button>
        </div>
      </div>
    </div>
  )
}

export default ErrorMessage
