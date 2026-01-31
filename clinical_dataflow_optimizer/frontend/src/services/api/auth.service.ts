/**
 * Authentication Service
 * Handles login, logout, and user authentication
 */

import { apiClient } from './client'
import {
  setAuthToken,
  setRefreshToken,
  setUserInfo,
  clearAuth,
} from '../../utils/auth'
import type { LoginCredentials, AuthResponse } from './types'

export class AuthService {
  private basePath = '/auth'

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await apiClient.post<AuthResponse>(
      `${this.basePath}/login`,
      credentials
    )

    // Store tokens and user info
    setAuthToken(response.accessToken)
    setRefreshToken(response.refreshToken)
    setUserInfo(response.user)

    return response
  }

  async logout(): Promise<void> {
    try {
      await apiClient.post(`${this.basePath}/logout`)
    } finally {
      clearAuth()
      window.location.href = '/login'
    }
  }

  async getCurrentUser(): Promise<AuthResponse['user']> {
    return apiClient.get<AuthResponse['user']>(`${this.basePath}/me`)
  }

  async forgotPassword(email: string): Promise<{ message: string }> {
    return apiClient.post(`${this.basePath}/forgot-password`, { email })
  }

  async resetPassword(
    token: string,
    newPassword: string
  ): Promise<{ message: string }> {
    return apiClient.post(`${this.basePath}/reset-password`, {
      token,
      newPassword,
    })
  }

  async changePassword(
    currentPassword: string,
    newPassword: string
  ): Promise<{ message: string }> {
    return apiClient.post(`${this.basePath}/change-password`, {
      currentPassword,
      newPassword,
    })
  }
}

export const authService = new AuthService()
