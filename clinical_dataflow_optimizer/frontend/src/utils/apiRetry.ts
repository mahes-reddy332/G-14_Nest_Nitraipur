import axios, { AxiosRequestConfig, AxiosResponse } from 'axios'

export interface RetryConfig {
  maxRetries?: number
  baseDelay?: number
  maxDelay?: number
  retryCondition?: (error: any) => boolean
  onRetry?: (attempt: number, error: any) => void
}

export interface LoadingConfig {
  operation: string
  onLoadingChange?: (isLoading: boolean, progress?: number) => void
}

const defaultRetryCondition = (error: any): boolean => {
  // Retry on network errors, 5xx server errors, or specific 4xx errors
  if (!error.response) return true // Network error
  const status = error.response.status
  return status >= 500 || status === 408 || status === 429 // Server errors, timeout, rate limit
}

export async function apiRequestWithRetry<T = any>(
  config: AxiosRequestConfig,
  retryConfig: RetryConfig = {},
  loadingConfig?: LoadingConfig
): Promise<AxiosResponse<T>> {
  const {
    maxRetries = 3,
    baseDelay = 1000,
    maxDelay = 10000,
    retryCondition = defaultRetryCondition,
    onRetry,
  } = retryConfig

  let lastError: any

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Set loading state
      if (loadingConfig?.onLoadingChange) {
        const progress = attempt === 0 ? 0 : (attempt / (maxRetries + 1)) * 100
        loadingConfig.onLoadingChange(true, progress)
      }

      const response = await axios(config)

      // Clear loading state on success
      if (loadingConfig?.onLoadingChange) {
        loadingConfig.onLoadingChange(false, 100)
      }

      return response
    } catch (error) {
      lastError = error

      // Clear loading state on error
      if (loadingConfig?.onLoadingChange) {
        loadingConfig.onLoadingChange(false)
      }

      // Don't retry on last attempt or if condition not met
      if (attempt === maxRetries || !retryCondition(error)) {
        throw error
      }

      // Calculate delay with exponential backoff and jitter
      const delay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay)
      const jitter = Math.random() * 0.1 * delay // Add 10% jitter
      const finalDelay = delay + jitter

      // Call retry callback
      if (onRetry) {
        onRetry(attempt + 1, error)
      }

      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, finalDelay))
    }
  }

  throw lastError
}

// Hook for using retry logic with Zustand store
export function useApiWithRetry() {
  const { setLoading, addError, addToRetryQueue } = useStore()

  const apiCall = async <T = any>(
    config: AxiosRequestConfig,
    operationName: string,
    retryConfig?: Partial<RetryConfig>
  ): Promise<AxiosResponse<T>> => {
    const fullRetryConfig: RetryConfig = {
      maxRetries: 3,
      onRetry: (attempt, error) => {
        console.warn(`Retrying ${operationName} (attempt ${attempt}):`, error.message)
        addError({
          message: `${operationName} failed, retrying (attempt ${attempt})`,
          code: 'API_RETRY',
          timestamp: new Date(),
          context: operationName,
        })
      },
      ...retryConfig,
    }

    const loadingConfig: LoadingConfig = {
      operation: operationName,
      onLoadingChange: (isLoading, progress) => {
        setLoading(operationName, isLoading, progress)
      },
    }

    try {
      return await apiRequestWithRetry(config, fullRetryConfig, loadingConfig)
    } catch (error: unknown) {
      // Add final error to store
      const axiosError = error as { response?: { status?: number } }
      addError({
        message: `${operationName} failed after all retries`,
        code: axiosError.response?.status?.toString() || 'API_ERROR',
        timestamp: new Date(),
        context: operationName,
      })
      throw error
    }
  }

  return { apiCall }
}

// Import here to avoid circular dependency
import { useStore } from '../store'