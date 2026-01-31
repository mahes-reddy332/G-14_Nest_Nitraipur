# Backend-Frontend Integration Implementation Summary

## Overview

This document summarizes the implementation of the backend-frontend integration layer for the Clinical Trial Operational Dataflow Metrics Web Application.

## Files Created

### 1. Configuration Files

| File | Purpose |
|------|---------|
| `src/config/api.config.ts` | Central API configuration with base URLs and timeouts |
| `.env.development` | Development environment variables |
| `.env.production` | Production environment variables |
| `.env.example` | Template for local configuration |

### 2. API Service Layer

| File | Purpose |
|------|---------|
| `src/services/api/client.ts` | Axios-based API client with request/response interceptors |
| `src/services/api/types.ts` | TypeScript interfaces for all API requests and responses |
| `src/services/api/auth.service.ts` | Authentication endpoints (login, logout, refresh) |
| `src/services/api/dashboard.service.ts` | Dashboard summary and metrics |
| `src/services/api/edcMetrics.service.ts` | EDC metrics and subject data |
| `src/services/api/dataQuality.service.ts` | Query management and DQI |
| `src/services/api/visitManagement.service.ts` | Visit compliance and scheduling |
| `src/services/api/laboratory.service.ts` | Lab reconciliation data |
| `src/services/api/safety.service.ts` | SAE dashboard and management |
| `src/services/api/coding.service.ts` | MedDRA and WHO Drug coding |
| `src/services/api/forms.service.ts` | SDV and form status |
| `src/services/api/craActivity.service.ts` | CRA performance and visits |
| `src/services/api/metadata.service.ts` | Reference data (regions, countries, sites) |
| `src/services/api/index.ts` | Central export for all services |

### 3. WebSocket Client

| File | Purpose |
|------|---------|
| `src/services/websocket/client.ts` | WebSocket client with auto-reconnect |
| `src/services/websocket/index.ts` | WebSocket exports |

### 4. React Query Hooks

| File | Purpose |
|------|---------|
| `src/hooks/useAuth.ts` | Authentication hooks |
| `src/hooks/useDashboard.ts` | Dashboard data hooks |
| `src/hooks/useEDCMetrics.ts` | EDC metrics hooks |
| `src/hooks/useDataQuality.ts` | Query management hooks |
| `src/hooks/useVisitManagement.ts` | Visit data hooks |
| `src/hooks/useLaboratory.ts` | Lab data hooks |
| `src/hooks/useSafety.ts` | SAE data hooks |
| `src/hooks/useCoding.ts` | MedDRA/WHO Drug coding hooks |
| `src/hooks/useForms.ts` | Forms/SDV hooks |
| `src/hooks/useCRAActivity.ts` | CRA activity hooks |
| `src/hooks/useMetadata.ts` | Reference data hooks |
| `src/hooks/index.ts` | Central export for all hooks |

### 5. Utility Files

| File | Purpose |
|------|---------|
| `src/utils/auth.ts` | Authentication token management |

### 6. Context Providers

| File | Purpose |
|------|---------|
| `src/contexts/AuthContext.tsx` | Authentication state context |
| `src/contexts/NotificationContext.tsx` | Real-time notification context |
| `src/contexts/index.ts` | Context exports |

### 7. Common Components

| File | Purpose |
|------|---------|
| `src/components/common/LoadingSpinner.tsx` | Loading states and skeletons |
| `src/components/common/ErrorMessage.tsx` | Error display components |
| `src/components/common/DataWrapper.tsx` | Data fetching wrapper component |
| `src/components/common/index.ts` | Common component exports |

## Key Features

### API Client
- Axios-based with request/response interceptors
- Automatic Bearer token injection
- 401 error handling with token refresh
- Request queue for concurrent requests during refresh
- Configurable timeout and base URL

### React Query Integration
- Caching with configurable stale times
- Automatic refetching on window focus
- Optimistic updates for mutations
- Query invalidation on successful mutations

### WebSocket Support
- Auto-reconnect with exponential backoff
- Message type subscriptions
- Connection state management
- Type-safe message handling

### Feature Flags
- Mock data mode for development
- WebSocket toggle
- Notifications toggle
- Debug logging

## Environment Variables

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000/api
VITE_API_TIMEOUT=30000

# WebSocket Configuration
VITE_WS_BASE_URL=ws://localhost:8000
VITE_WS_RECONNECT_INTERVAL=5000
VITE_WS_MAX_RECONNECT_ATTEMPTS=5

# Feature Flags
VITE_ENABLE_MOCK_DATA=false
VITE_ENABLE_WEBSOCKET=true
VITE_ENABLE_NOTIFICATIONS=true

# Debug
VITE_DEBUG_MODE=true
VITE_LOG_API_CALLS=true
```

## Usage Example

### Using a Hook

```tsx
import { useEDCMetrics, useExportEDCMetrics } from '../hooks'

function EDCMetricsPage() {
  const [filters, setFilters] = useState<EDCMetricsFilters>({})
  
  // Fetch data
  const { data, isLoading, error, refetch } = useEDCMetrics(filters)
  
  // Export mutation
  const exportMutation = useExportEDCMetrics()
  
  const handleExport = (format: 'excel' | 'csv' | 'pdf') => {
    exportMutation.mutate({ filters, format })
  }
  
  if (isLoading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error.message} onRetry={refetch} />
  
  return <Table data={data?.items} />
}
```

### Using the Data Wrapper

```tsx
import { QueryDataWrapper } from '../components/common'
import { useQueries } from '../hooks'

function DataQualityPage() {
  const query = useQueries(filters)
  
  return (
    <QueryDataWrapper
      query={query}
      loadingType="table"
      emptyTitle="No queries found"
    >
      {(data) => <QueriesTable data={data.items} />}
    </QueryDataWrapper>
  )
}
```

### Subscribing to WebSocket Events

```tsx
import { useSAEReports, useQueryUpdates } from '../hooks'
import { useQueryClient } from '@tanstack/react-query'

function Dashboard() {
  const queryClient = useQueryClient()
  
  // Subscribe to SAE reports
  useSAEReports((data) => {
    // Show notification
    notification.warning({
      message: 'New SAE Reported',
      description: `SAE ${data.saeId} for subject ${data.subjectId}`,
    })
    // Invalidate related queries
    queryClient.invalidateQueries({ queryKey: ['safety-dashboard'] })
  })
  
  // Subscribe to query updates
  useQueryUpdates((data) => {
    queryClient.invalidateQueries({ queryKey: ['data-quality-queries'] })
  })
  
  return <DashboardContent />
}
```

## Next Steps

1. **Backend API Implementation**: Implement the corresponding backend API endpoints
2. **Error Handling Enhancement**: Add more specific error types and handling
3. **Offline Support**: Consider adding offline caching with React Query persistence
4. **Testing**: Add unit tests for hooks and services
5. **Performance Monitoring**: Add request timing and error tracking
