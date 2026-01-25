import { create, StateCreator, StoreApi } from 'zustand'
import type { Alert, WebSocketEvent, DashboardSummary } from '../types'

interface ErrorState {
  message: string
  code?: string
  timestamp: Date
  context?: string
}

interface LoadingState {
  isLoading: boolean
  operation: string
  progress?: number
}

interface RealTimeState {
  // Connection status
  isConnected: boolean
  setConnected: (status: boolean) => void

  // Dashboard summary (cached for quick access)
  dashboardSummary: DashboardSummary | null
  setDashboardSummary: (summary: DashboardSummary) => void

  // Real-time updates
  recentUpdates: WebSocketEvent[]
  addUpdate: (update: WebSocketEvent) => void
  clearUpdates: () => void

  // Alerts
  activeAlerts: Alert[]
  addAlert: (alert: Alert) => void
  updateAlert: (alertId: string, updates: Partial<Alert>) => void
  removeAlert: (alertId: string) => void
  setAlerts: (alerts: Alert[]) => void

  // Notifications
  unreadNotifications: number
  incrementNotifications: () => void
  resetNotifications: () => void

  // Selected filters (global state)
  selectedStudyId: string | null
  setSelectedStudyId: (studyId: string | null) => void

  // Error handling
  errors: ErrorState[]
  addError: (error: ErrorState) => void
  clearError: (index: number) => void
  clearAllErrors: () => void

  // Loading states
  loadingStates: Record<string, LoadingState>
  setLoading: (operation: string, isLoading: boolean, progress?: number) => void
  getLoading: (operation: string) => LoadingState | undefined

  // Retry logic
  retryQueue: Array<{
    id: string
    operation: () => Promise<any>
    retries: number
    maxRetries: number
  }>
  addToRetryQueue: (id: string, operation: () => Promise<any>, maxRetries?: number) => void
  removeFromRetryQueue: (id: string) => void
  processRetryQueue: () => void
}

export const useStore = create<RealTimeState>()((set, get) => ({
  // Connection status
  isConnected: false,
  setConnected: (status: boolean) => set({ isConnected: status }),

  // Dashboard summary
  dashboardSummary: null,
  setDashboardSummary: (summary: DashboardSummary) => set({ dashboardSummary: summary }),

  // Real-time updates
  recentUpdates: [],
  addUpdate: (update: WebSocketEvent) =>
    set((state: RealTimeState) => ({
      recentUpdates: [update, ...state.recentUpdates].slice(0, 100),
    })),
  clearUpdates: () => set({ recentUpdates: [] }),

  // Alerts
  activeAlerts: [],
  addAlert: (alert: Alert) =>
    set((state: RealTimeState) => ({
      activeAlerts: [alert, ...state.activeAlerts],
      unreadNotifications: state.unreadNotifications + 1,
    })),
  updateAlert: (alertId: string, updates: Partial<Alert>) =>
    set((state: RealTimeState) => ({
      activeAlerts: state.activeAlerts.map((a: Alert) =>
        a.alert_id === alertId ? { ...a, ...updates } : a
      ),
    })),
  removeAlert: (alertId: string) =>
    set((state: RealTimeState) => ({
      activeAlerts: state.activeAlerts.filter((a: Alert) => a.alert_id !== alertId),
    })),
  setAlerts: (alerts: Alert[]) => set({ activeAlerts: alerts }),

  // Notifications
  unreadNotifications: 0,
  incrementNotifications: () =>
    set((state: RealTimeState) => ({ unreadNotifications: state.unreadNotifications + 1 })),
  resetNotifications: () => set({ unreadNotifications: 0 }),

  // Selected filters
  selectedStudyId: null,
  setSelectedStudyId: (studyId: string | null) => set({ selectedStudyId: studyId }),

  // Error handling
  errors: [],
  addError: (error: ErrorState) =>
    set((state: RealTimeState) => ({
      errors: [error, ...state.errors].slice(0, 50), // Keep last 50 errors
    })),
  clearError: (index: number) =>
    set((state: RealTimeState) => ({
      errors: state.errors.filter((_, i) => i !== index),
    })),
  clearAllErrors: () => set({ errors: [] }),

  // Loading states
  loadingStates: {},
  setLoading: (operation: string, isLoading: boolean, progress?: number) =>
    set((state: RealTimeState) => ({
      loadingStates: {
        ...state.loadingStates,
        [operation]: {
          isLoading,
          operation,
          progress,
        },
      },
    })),
  getLoading: (operation: string) => (get() as RealTimeState).loadingStates[operation],

  // Retry logic
  retryQueue: [],
  addToRetryQueue: (id: string, operation: () => Promise<any>, maxRetries: number = 3) =>
    set((state: RealTimeState) => ({
      retryQueue: [
        ...state.retryQueue,
        { id, operation, retries: 0, maxRetries },
      ],
    })),
  removeFromRetryQueue: (id: string) =>
    set((state: RealTimeState) => ({
      retryQueue: state.retryQueue.filter(item => item.id !== id),
    })),
  processRetryQueue: () =>
    set((state: RealTimeState) => {
      const newQueue = [...state.retryQueue]
      newQueue.forEach(async (item, index) => {
        if (item.retries < item.maxRetries) {
          try {
            await item.operation()
            newQueue.splice(index, 1) // Remove successful operation
          } catch (error) {
            item.retries += 1
            // Exponential backoff: wait 1s, 2s, 4s, etc.
            setTimeout(() => {
              // Re-trigger processing after delay
              ;(get() as RealTimeState).processRetryQueue()
            }, Math.pow(2, item.retries) * 1000)
          }
        } else {
          // Max retries reached, add to errors
          ;(get() as RealTimeState).addError({
            message: `Operation ${item.id} failed after ${item.maxRetries} retries`,
            code: 'MAX_RETRIES_EXCEEDED',
            timestamp: new Date(),
            context: item.id,
          })
          newQueue.splice(index, 1) // Remove failed operation
        }
      })
      return { retryQueue: newQueue }
    }),
}))
