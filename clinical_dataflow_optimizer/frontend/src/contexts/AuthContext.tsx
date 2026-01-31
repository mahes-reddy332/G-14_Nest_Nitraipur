/**
 * Auth Context
 * Global authentication state management
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { useCurrentUser, useLogin, useLogout } from '../hooks/useAuth'
import { getAuthToken, isAuthenticated, hasPermission, hasRole } from '../utils/auth'
import type { User, LoginResponse } from '../services/api/types'

interface AuthContextType {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (username: string, password: string) => Promise<LoginResponse>
  logout: () => Promise<void>
  hasPermission: (permission: string) => boolean
  hasRole: (role: string) => boolean
  refresh: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [isAuth, setIsAuth] = useState(() => isAuthenticated())
  const { data: user, isLoading, refetch } = useCurrentUser()
  const loginMutation = useLogin()
  const logoutMutation = useLogout()

  // Check auth status on mount and token changes
  useEffect(() => {
    const checkAuth = () => {
      setIsAuth(isAuthenticated())
    }
    
    // Check periodically
    const interval = setInterval(checkAuth, 60000) // Every minute
    
    // Listen for storage events (token changes in other tabs)
    window.addEventListener('storage', checkAuth)
    
    return () => {
      clearInterval(interval)
      window.removeEventListener('storage', checkAuth)
    }
  }, [])

  const login = useCallback(
    async (username: string, password: string) => {
      const result = await loginMutation.mutateAsync({ username, password })
      setIsAuth(true)
      return result
    },
    [loginMutation]
  )

  const logout = useCallback(async () => {
    await logoutMutation.mutateAsync()
    setIsAuth(false)
  }, [logoutMutation])

  const checkPermission = useCallback((permission: string) => {
    return hasPermission(permission)
  }, [])

  const checkRole = useCallback((role: string) => {
    return hasRole(role)
  }, [])

  const refresh = useCallback(() => {
    refetch()
  }, [refetch])

  const value: AuthContextType = {
    user: user ?? null,
    isLoading,
    isAuthenticated: isAuth,
    login,
    logout,
    hasPermission: checkPermission,
    hasRole: checkRole,
    refresh,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
