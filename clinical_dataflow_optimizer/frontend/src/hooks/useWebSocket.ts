import { useEffect, useRef, useCallback, useState } from 'react'
import { useStore } from '../store'
import type { WebSocketEvent, Alert } from '../types'

const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss' : 'ws'
const WS_DEFAULT_BASE = import.meta.env.DEV
  ? ''  // Use relative URL in dev to go through Vite proxy
  : `${WS_PROTOCOL}://${window.location.host}`
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || WS_DEFAULT_BASE
const WS_URL = `${WS_BASE_URL.replace(/\/$/, '')}/ws/dashboard`

// Configuration for robust reconnection
const INITIAL_RECONNECT_DELAY = 1000  // Start with 1 second
const MAX_RECONNECT_DELAY = 30000     // Max 30 seconds between attempts
const RECONNECT_MULTIPLIER = 1.5      // Exponential backoff multiplier
const PING_INTERVAL = 25000           // Send ping every 25 seconds
const PONG_TIMEOUT = 10000            // Expect pong within 10 seconds
const STARTUP_CHECK_INTERVAL = 2000   // Check backend readiness every 2 seconds
const MAX_STARTUP_WAIT = 60000        // Max 60 seconds waiting for backend

interface StartupStatus {
  is_ready: boolean
  is_starting: boolean
  data_loaded: boolean
  services_initialized: boolean
  startup_error: string | null
}

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'waiting_for_backend'

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pongTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const startupCheckRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY)
  const startupWaitStartRef = useRef<number | null>(null)
  const isConnectingRef = useRef(false)
  
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected')
  const [backendReady, setBackendReady] = useState(false)
  
  const { setConnected, addUpdate, addAlert } = useStore()

  // Check if backend is ready before attempting WebSocket connection
  const checkBackendReady = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch('/api/ready')
      if (response.ok) {
        const data: StartupStatus = await response.json()
        return data.is_ready
      }
      // 503 means starting up - not ready yet
      if (response.status === 503) {
        return false
      }
      return false
    } catch {
      // Network error - backend not reachable
      return false
    }
  }, [])

  // Wait for backend to be ready before connecting WebSocket
  const waitForBackend = useCallback(async (): Promise<boolean> => {
    if (startupWaitStartRef.current === null) {
      startupWaitStartRef.current = Date.now()
    }

    setConnectionState('waiting_for_backend')
    
    return new Promise((resolve) => {
      const check = async () => {
        const elapsed = Date.now() - (startupWaitStartRef.current || Date.now())
        
        if (elapsed > MAX_STARTUP_WAIT) {
          console.warn('Backend startup timeout - attempting connection anyway')
          startupWaitStartRef.current = null
          if (startupCheckRef.current) {
            clearInterval(startupCheckRef.current)
            startupCheckRef.current = null
          }
          resolve(false)
          return
        }

        const ready = await checkBackendReady()
        if (ready) {
          console.log('Backend ready!')
          setBackendReady(true)
          startupWaitStartRef.current = null
          if (startupCheckRef.current) {
            clearInterval(startupCheckRef.current)
            startupCheckRef.current = null
          }
          resolve(true)
        }
      }

      // Immediate check
      check()

      // Periodic checks
      if (!startupCheckRef.current) {
        startupCheckRef.current = setInterval(check, STARTUP_CHECK_INTERVAL)
      }
    })
  }, [checkBackendReady])

  const clearTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
    if (pongTimeoutRef.current) {
      clearTimeout(pongTimeoutRef.current)
      pongTimeoutRef.current = null
    }
    if (startupCheckRef.current) {
      clearInterval(startupCheckRef.current)
      startupCheckRef.current = null
    }
  }, [])

  const startPingPong = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
    }
    
    pingIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({ type: 'ping' }))
          
          // Set timeout for pong response
          if (pongTimeoutRef.current) {
            clearTimeout(pongTimeoutRef.current)
          }
          pongTimeoutRef.current = setTimeout(() => {
            console.warn('Pong timeout - connection may be dead')
            wsRef.current?.close()
          }, PONG_TIMEOUT)
        } catch (error) {
          console.warn('Failed to send ping:', error)
        }
      }
    }, PING_INTERVAL)
  }, [])

  const scheduleReconnect = useCallback((delay?: number) => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    
    const reconnectDelay = delay ?? Math.min(reconnectDelayRef.current, MAX_RECONNECT_DELAY)
    console.log(`Scheduling reconnect in ${reconnectDelay}ms`)
    
    reconnectTimeoutRef.current = setTimeout(() => {
      // Increase delay for next attempt (exponential backoff)
      reconnectDelayRef.current = Math.min(
        reconnectDelayRef.current * RECONNECT_MULTIPLIER,
        MAX_RECONNECT_DELAY
      )
      connectInternal()
    }, reconnectDelay)
  }, [])

  const connectInternal = useCallback(async () => {
    // Prevent concurrent connection attempts
    if (isConnectingRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return

    isConnectingRef.current = true

    try {
      // Wait for backend to be ready first
      if (!backendReady) {
        const ready = await waitForBackend()
        if (!ready) {
          console.warn('Backend not ready, will retry')
          isConnectingRef.current = false
          scheduleReconnect(STARTUP_CHECK_INTERVAL)
          return
        }
      }

      setConnectionState('connecting')

      console.log('Connecting to WebSocket:', WS_URL)
      wsRef.current = new WebSocket(WS_URL)

      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
        setConnectionState('connected')
        setConnected(true)
        isConnectingRef.current = false
        // Reset reconnect delay on successful connection
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY
        // Start ping-pong heartbeat
        startPingPong()
      }

      wsRef.current.onmessage = (event: MessageEvent) => {
        try {
          const rawMessage = JSON.parse(event.data) as { type: string; data?: unknown }
          
          // Handle pong response - clear timeout
          if (rawMessage.type === 'pong' || rawMessage.type === 'heartbeat') {
            if (pongTimeoutRef.current) {
              clearTimeout(pongTimeoutRef.current)
              pongTimeoutRef.current = null
            }
            return
          }
          
          // Cast to WebSocketEvent for standard message types
          const message = rawMessage as WebSocketEvent
          
          // Add to recent updates
          addUpdate(message)

          // Handle specific event types
          if (message.type === 'new_alert' && message.data) {
            addAlert(message.data as unknown as Alert)
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        setConnectionState('disconnected')
        setConnected(false)
        isConnectingRef.current = false
        clearTimers()
        
        // Only reconnect if not a clean close (code 1000)
        if (event.code !== 1000) {
          scheduleReconnect()
        }
      }

      wsRef.current.onerror = () => {
        console.error('WebSocket error')
        isConnectingRef.current = false
        // The close event will handle reconnection
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnectionState('disconnected')
      setConnected(false)
      isConnectingRef.current = false
      scheduleReconnect()
    }
  }, [setConnected, addUpdate, addAlert, backendReady, waitForBackend, startPingPong, scheduleReconnect, clearTimers])

  const connect = useCallback(() => {
    connectInternal()
  }, [connectInternal])

  const disconnect = useCallback(() => {
    clearTimers()
    isConnectingRef.current = false
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnect')
      wsRef.current = null
    }
    setConnectionState('disconnected')
    setConnected(false)
    setBackendReady(false)
    startupWaitStartRef.current = null
    reconnectDelayRef.current = INITIAL_RECONNECT_DELAY
  }, [setConnected, clearTimers])

  const send = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket not connected, cannot send message')
    }
  }, [])

  // Force reconnect (useful after backend restart)
  const forceReconnect = useCallback(() => {
    disconnect()
    setBackendReady(false)
    setTimeout(() => connect(), 100)
  }, [disconnect, connect])

  useEffect(() => {
    connect()

    // Handle visibility change - reconnect when tab becomes visible
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
          console.log('Tab visible, reconnecting WebSocket')
          connect()
        }
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)

    // Handle online/offline events
    const handleOnline = () => {
      console.log('Network online, reconnecting WebSocket')
      setBackendReady(false) // Re-check backend on network restore
      connect()
    }
    window.addEventListener('online', handleOnline)

    return () => {
      disconnect()
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('online', handleOnline)
    }
  }, [connect, disconnect])

  return { connect, disconnect, send, forceReconnect, connectionState, backendReady }
}
