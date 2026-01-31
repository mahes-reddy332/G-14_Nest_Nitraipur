/**
 * Authentication Utilities
 * Handles token storage, retrieval, and refresh logic
 */

const TOKEN_KEY = 'auth_token'
const REFRESH_TOKEN_KEY = 'refresh_token'
const USER_KEY = 'user_info'

export interface UserInfo {
  id: string
  email: string
  name: string
  role: string
  permissions: string[]
}

export const getAuthToken = (): string | null => {
  return localStorage.getItem(TOKEN_KEY)
}

export const setAuthToken = (token: string): void => {
  localStorage.setItem(TOKEN_KEY, token)
}

export const getRefreshToken = (): string | null => {
  return localStorage.getItem(REFRESH_TOKEN_KEY)
}

export const setRefreshToken = (token: string): void => {
  localStorage.setItem(REFRESH_TOKEN_KEY, token)
}

export const getUserInfo = (): UserInfo | null => {
  const userInfo = localStorage.getItem(USER_KEY)
  return userInfo ? JSON.parse(userInfo) : null
}

export const setUserInfo = (user: UserInfo): void => {
  localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export const clearAuth = (): void => {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

export const refreshAuthToken = async (): Promise<string> => {
  const refreshToken = getRefreshToken()
  
  if (!refreshToken) {
    throw new Error('No refresh token available')
  }

  try {
    const response = await fetch(
      `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/auth/refresh`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refreshToken }),
      }
    )

    if (!response.ok) {
      throw new Error('Token refresh failed')
    }

    const data = await response.json()
    setAuthToken(data.accessToken)
    setRefreshToken(data.refreshToken)
    
    return data.accessToken
  } catch (error) {
    clearAuth()
    throw error
  }
}

export const isAuthenticated = (): boolean => {
  return !!getAuthToken()
}

export const hasPermission = (permission: string): boolean => {
  const user = getUserInfo()
  return user?.permissions?.includes(permission) ?? false
}

export const hasRole = (role: string): boolean => {
  const user = getUserInfo()
  return user?.role === role
}
