/**
 * Auth Hooks
 * React Query hooks for authentication
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { authService } from '../services/api'
import { setAuthToken, clearAuth, getAuthToken } from '../utils/auth'

export const QUERY_KEYS = {
  user: 'auth-user',
}

export const useCurrentUser = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.user],
    queryFn: () => authService.getCurrentUser(),
    enabled: !!getAuthToken(),
    staleTime: 10 * 60 * 1000,
    retry: false,
  })
}

export const useLogin = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      username,
      password,
    }: {
      username: string
      password: string
    }) => authService.login(username, password),
    onSuccess: (data) => {
      setAuthToken(data.accessToken, data.refreshToken)
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.user] })
    },
  })
}

export const useLogout = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => authService.logout(),
    onSuccess: () => {
      clearAuth()
      queryClient.clear()
    },
    onError: () => {
      // Clear auth even if logout fails
      clearAuth()
      queryClient.clear()
    },
  })
}

export const useForgotPassword = () => {
  return useMutation({
    mutationFn: (email: string) => authService.forgotPassword(email),
  })
}

export const useResetPassword = () => {
  return useMutation({
    mutationFn: ({
      token,
      newPassword,
    }: {
      token: string
      newPassword: string
    }) => authService.resetPassword(token, newPassword),
  })
}

export const useRefreshToken = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (refreshToken: string) => authService.refreshToken(refreshToken),
    onSuccess: (data) => {
      setAuthToken(data.accessToken, data.refreshToken)
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.user] })
    },
    onError: () => {
      clearAuth()
      queryClient.clear()
    },
  })
}
