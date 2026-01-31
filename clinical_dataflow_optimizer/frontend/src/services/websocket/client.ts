/**
 * WebSocket Client
 * Handles real-time data updates via WebSocket connection
 */

import { WS_CONFIG } from '../../config/api.config'
import { getAuthToken } from '../../utils/auth'

export type WebSocketMessageType =
  | 'query_updated'
  | 'sae_reported'
  | 'visit_completed'
  | 'form_signed'
  | 'alert_triggered'
  | 'coding_updated'
  | 'subject_updated'
  | 'cra_visit_completed'

export interface WebSocketMessage {
  type: WebSocketMessageType
  payload: unknown
  timestamp: string
}

type MessageCallback = (payload: unknown) => void

export class WebSocketClient {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = WS_CONFIG.maxReconnectAttempts
  private reconnectDelay = WS_CONFIG.reconnectDelay
  private listeners: Map<WebSocketMessageType, Set<MessageCallback>> = new Map()
  private connectionListeners: Set<(connected: boolean) => void> = new Set()
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null

  connect(): void {
    const token = getAuthToken()

    if (!token) {
      console.warn('No auth token available for WebSocket connection')
      return
    }

    try {
      this.ws = new WebSocket(`${WS_CONFIG.wsURL}?token=${token}`)

      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.reconnectAttempts = 0
        this.notifyConnectionListeners(true)
      }

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          this.handleMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      this.ws.onclose = () => {
        console.log('WebSocket disconnected')
        this.notifyConnectionListeners(false)
        this.attemptReconnect()
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
    }
  }

  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }
    
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(
        `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`
      )
      
      this.reconnectTimeout = setTimeout(() => {
        this.connect()
      }, this.reconnectDelay * this.reconnectAttempts)
    } else {
      console.error('Max reconnection attempts reached')
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    const listeners = this.listeners.get(message.type)
    if (listeners) {
      listeners.forEach((callback) => callback(message.payload))
    }
  }

  private notifyConnectionListeners(connected: boolean): void {
    this.connectionListeners.forEach((callback) => callback(connected))
  }

  subscribe(type: WebSocketMessageType, callback: MessageCallback): () => void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set())
    }
    this.listeners.get(type)!.add(callback)

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(type)
      if (listeners) {
        listeners.delete(callback)
      }
    }
  }

  onConnectionChange(callback: (connected: boolean) => void): () => void {
    this.connectionListeners.add(callback)
    return () => {
      this.connectionListeners.delete(callback)
    }
  }

  send(message: WebSocketMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket is not connected')
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

export const wsClient = new WebSocketClient()
