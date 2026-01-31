/**
 * Data Wrapper Component
 * Wraps data fetching logic with loading, error, and empty states
 */

import React from 'react'
import {
  LoadingSpinner,
  Skeleton,
  TableSkeleton,
  CardSkeleton,
  ChartSkeleton,
} from './LoadingSpinner'
import { ErrorMessage, EmptyState } from './ErrorMessage'

interface DataWrapperProps<T> {
  data: T | undefined
  isLoading: boolean
  error: Error | null
  onRetry?: () => void
  loadingComponent?: React.ReactNode
  loadingType?: 'spinner' | 'skeleton' | 'table' | 'card' | 'chart'
  skeletonProps?: {
    rows?: number
    columns?: number
    count?: number
    height?: number
  }
  emptyTitle?: string
  emptyDescription?: string
  emptyAction?: {
    label: string
    onClick: () => void
  }
  isEmpty?: (data: T) => boolean
  children: (data: T) => React.ReactNode
  className?: string
}

export function DataWrapper<T>({
  data,
  isLoading,
  error,
  onRetry,
  loadingComponent,
  loadingType = 'spinner',
  skeletonProps,
  emptyTitle = 'No data available',
  emptyDescription,
  emptyAction,
  isEmpty,
  children,
  className = '',
}: DataWrapperProps<T>) {
  // Show loading state
  if (isLoading) {
    if (loadingComponent) {
      return <>{loadingComponent}</>
    }

    switch (loadingType) {
      case 'skeleton':
        return (
          <div className={className}>
            <Skeleton count={skeletonProps?.count ?? 3} />
          </div>
        )
      case 'table':
        return (
          <div className={className}>
            <TableSkeleton
              rows={skeletonProps?.rows ?? 5}
              columns={skeletonProps?.columns ?? 4}
            />
          </div>
        )
      case 'card':
        return (
          <div className={className}>
            <CardSkeleton count={skeletonProps?.count ?? 4} />
          </div>
        )
      case 'chart':
        return (
          <div className={className}>
            <ChartSkeleton height={skeletonProps?.height ?? 300} />
          </div>
        )
      default:
        return (
          <div className={`flex items-center justify-center p-8 ${className}`}>
            <LoadingSpinner size="lg" text="Loading..." />
          </div>
        )
    }
  }

  // Show error state
  if (error) {
    return (
      <div className={className}>
        <ErrorMessage
          title="Failed to load data"
          message={error.message || 'An unexpected error occurred'}
          onRetry={onRetry}
        />
      </div>
    )
  }

  // Show empty state
  if (!data || (isEmpty && isEmpty(data))) {
    return (
      <div className={className}>
        <EmptyState
          title={emptyTitle}
          description={emptyDescription}
          action={emptyAction}
        />
      </div>
    )
  }

  // Default check for common empty data patterns
  if (
    (Array.isArray(data) && data.length === 0) ||
    (typeof data === 'object' &&
      data !== null &&
      'items' in data &&
      Array.isArray((data as { items: unknown[] }).items) &&
      (data as { items: unknown[] }).items.length === 0)
  ) {
    return (
      <div className={className}>
        <EmptyState
          title={emptyTitle}
          description={emptyDescription}
          action={emptyAction}
        />
      </div>
    )
  }

  // Render children with data
  return <>{children(data)}</>
}

/**
 * Query Data Wrapper
 * Specifically for React Query results
 */
interface QueryDataWrapperProps<T> {
  query: {
    data: T | undefined
    isLoading: boolean
    isError: boolean
    error: Error | null
    refetch: () => void
  }
  loadingType?: 'spinner' | 'skeleton' | 'table' | 'card' | 'chart'
  skeletonProps?: {
    rows?: number
    columns?: number
    count?: number
    height?: number
  }
  emptyTitle?: string
  emptyDescription?: string
  isEmpty?: (data: T) => boolean
  children: (data: T) => React.ReactNode
  className?: string
}

export function QueryDataWrapper<T>({
  query,
  loadingType,
  skeletonProps,
  emptyTitle,
  emptyDescription,
  isEmpty,
  children,
  className,
}: QueryDataWrapperProps<T>) {
  return (
    <DataWrapper<T>
      data={query.data}
      isLoading={query.isLoading}
      error={query.isError ? query.error : null}
      onRetry={() => query.refetch()}
      loadingType={loadingType}
      skeletonProps={skeletonProps}
      emptyTitle={emptyTitle}
      emptyDescription={emptyDescription}
      isEmpty={isEmpty}
      className={className}
    >
      {children}
    </DataWrapper>
  )
}

export default DataWrapper
