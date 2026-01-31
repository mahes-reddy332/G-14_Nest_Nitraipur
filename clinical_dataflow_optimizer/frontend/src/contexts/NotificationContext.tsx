/**
 * Notification Context
 * Global notification/alert management for real-time updates
 */

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { wsClient } from '../services/websocket'

export interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
  timestamp: Date
  read: boolean
  source?: string
  data?: unknown
}

interface NotificationContextType {
  notifications: Notification[]
  unreadCount: number
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void
  markAsRead: (id: string) => void
  markAllAsRead: () => void
  removeNotification: (id: string) => void
  clearAll: () => void
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined)

const MAX_NOTIFICATIONS = 100

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [notifications, setNotifications] = useState<Notification[]>([])

  // Subscribe to WebSocket alerts
  useEffect(() => {
    const unsubscribe = wsClient.subscribe('alert_triggered', (message) => {
      addNotification({
        type: message.data.severity === 'critical' ? 'error' : message.data.severity,
        title: message.data.type,
        message: message.data.message,
        source: 'alert',
        data: message.data,
      })
    })

    return unsubscribe
  }, [])

  // Subscribe to SAE reports
  useEffect(() => {
    const unsubscribe = wsClient.subscribe('sae_reported', (message) => {
      addNotification({
        type: message.data.severity === 'serious' ? 'error' : 'warning',
        title: 'New SAE Reported',
        message: `SAE ${message.data.saeId} reported for subject ${message.data.subjectId}`,
        source: 'sae',
        data: message.data,
      })
    })

    return unsubscribe
  }, [])

  const addNotification = useCallback(
    (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => {
      const newNotification: Notification = {
        ...notification,
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date(),
        read: false,
      }

      setNotifications((prev) => {
        const updated = [newNotification, ...prev]
        // Keep only the most recent notifications
        return updated.slice(0, MAX_NOTIFICATIONS)
      })
    },
    []
  )

  const markAsRead = useCallback((id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    )
  }, [])

  const markAllAsRead = useCallback(() => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })))
  }, [])

  const removeNotification = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id))
  }, [])

  const clearAll = useCallback(() => {
    setNotifications([])
  }, [])

  const unreadCount = notifications.filter((n) => !n.read).length

  const value: NotificationContextType = {
    notifications,
    unreadCount,
    addNotification,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearAll,
  }

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  )
}

export const useNotifications = () => {
  const context = useContext(NotificationContext)
  if (context === undefined) {
    throw new Error('useNotifications must be used within a NotificationProvider')
  }
  return context
}
