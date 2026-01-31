/**
 * API Configuration
 * Central configuration for API endpoints and settings
 */

export const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
}

export const WS_CONFIG = {
  wsURL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
  reconnectDelay: 1000,
  maxReconnectAttempts: 5,
}

export const config = {
  apiBaseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
  wsURL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
  environment: import.meta.env.VITE_ENV || 'development',
  isDevelopment: import.meta.env.VITE_ENV === 'development',
  isProduction: import.meta.env.VITE_ENV === 'production',
}

export default config
