/**
 * Loading Spinner Component
 * Reusable loading indicator with various sizes
 */

import React from 'react'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  text?: string
  fullScreen?: boolean
  className?: string
}

const sizeClasses = {
  sm: 'h-4 w-4 border-2',
  md: 'h-8 w-8 border-2',
  lg: 'h-12 w-12 border-3',
  xl: 'h-16 w-16 border-4',
}

const textSizeClasses = {
  sm: 'text-xs',
  md: 'text-sm',
  lg: 'text-base',
  xl: 'text-lg',
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  text,
  fullScreen = false,
  className = '',
}) => {
  const spinner = (
    <div className={`flex flex-col items-center justify-center gap-3 ${className}`}>
      <div
        className={`${sizeClasses[size]} border-primary-200 border-t-primary-600 animate-spin rounded-full`}
        role="status"
        aria-label="Loading"
      />
      {text && (
        <p className={`${textSizeClasses[size]} text-gray-600 dark:text-gray-400`}>
          {text}
        </p>
      )}
    </div>
  )

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
        {spinner}
      </div>
    )
  }

  return spinner
}

/**
 * Skeleton Loader Component
 * For content placeholders during loading
 */
interface SkeletonProps {
  className?: string
  variant?: 'text' | 'rectangular' | 'circular'
  width?: string | number
  height?: string | number
  count?: number
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'text',
  width,
  height,
  count = 1,
}) => {
  const baseClasses = 'animate-pulse bg-gray-200 dark:bg-gray-700'
  
  const variantClasses = {
    text: 'rounded',
    rectangular: 'rounded-md',
    circular: 'rounded-full',
  }

  const style: React.CSSProperties = {
    width: width ?? (variant === 'circular' ? height : '100%'),
    height: height ?? (variant === 'text' ? '1rem' : '100%'),
  }

  const skeletonItems = Array.from({ length: count }, (_, i) => (
    <div
      key={i}
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      style={style}
    />
  ))

  if (count === 1) {
    return skeletonItems[0]
  }

  return <div className="flex flex-col gap-2">{skeletonItems}</div>
}

/**
 * Table Skeleton
 * For loading state of tables
 */
interface TableSkeletonProps {
  rows?: number
  columns?: number
}

export const TableSkeleton: React.FC<TableSkeletonProps> = ({
  rows = 5,
  columns = 4,
}) => {
  return (
    <div className="w-full">
      {/* Header */}
      <div className="flex gap-4 p-4 border-b border-gray-200 dark:border-gray-700">
        {Array.from({ length: columns }, (_, i) => (
          <Skeleton key={i} variant="text" height={20} className="flex-1" />
        ))}
      </div>
      {/* Rows */}
      {Array.from({ length: rows }, (_, rowIndex) => (
        <div
          key={rowIndex}
          className="flex gap-4 p-4 border-b border-gray-100 dark:border-gray-800"
        >
          {Array.from({ length: columns }, (_, colIndex) => (
            <Skeleton
              key={colIndex}
              variant="text"
              height={16}
              className="flex-1"
            />
          ))}
        </div>
      ))}
    </div>
  )
}

/**
 * Card Skeleton
 * For loading state of metric cards
 */
interface CardSkeletonProps {
  count?: number
}

export const CardSkeleton: React.FC<CardSkeletonProps> = ({ count = 1 }) => {
  const cards = Array.from({ length: count }, (_, i) => (
    <div
      key={i}
      className="p-4 rounded-lg border border-gray-200 dark:border-gray-700"
    >
      <Skeleton variant="text" height={14} width="60%" className="mb-2" />
      <Skeleton variant="text" height={32} width="40%" className="mb-3" />
      <Skeleton variant="text" height={12} width="80%" />
    </div>
  ))

  if (count === 1) {
    return cards[0]
  }

  return <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">{cards}</div>
}

/**
 * Chart Skeleton
 * For loading state of charts
 */
interface ChartSkeletonProps {
  height?: number
}

export const ChartSkeleton: React.FC<ChartSkeletonProps> = ({ height = 300 }) => {
  return (
    <div
      className="w-full flex items-end justify-center gap-2 p-4"
      style={{ height }}
    >
      {[40, 65, 50, 80, 45, 70, 55, 90, 60, 75].map((h, i) => (
        <Skeleton
          key={i}
          variant="rectangular"
          width={20}
          height={`${h}%`}
        />
      ))}
    </div>
  )
}

export default LoadingSpinner
