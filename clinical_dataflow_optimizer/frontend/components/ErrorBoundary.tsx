/**
 * Error Handling Components for Clinical Dataflow Optimizer
 * =========================================================
 * 
 * React Error Boundary and error state management components.
 * Provides graceful degradation and user-friendly error displays.
 */

import React, { Component, ReactNode, ErrorInfo } from 'react';
import { create } from 'zustand';

// Declare process for Node.js environment variable access
declare const process: { env: { NODE_ENV: string } };

// =============================================================================
// Error Types
// =============================================================================

export interface ErrorDetails {
  errorType: string;
  errorCode: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
  recoverable: boolean;
  errorId?: string;
}

export interface ErrorState {
  errors: ErrorDetails[];
  lastError: ErrorDetails | null;
  isRecovering: boolean;
  recoveryAttempts: number;
}

// =============================================================================
// Error Store (Zustand)
// =============================================================================

interface ErrorStore extends ErrorState {
  addError: (error: ErrorDetails) => void;
  clearError: (errorId?: string) => void;
  clearAllErrors: () => void;
  setRecovering: (isRecovering: boolean) => void;
  incrementRecoveryAttempts: () => void;
  resetRecoveryAttempts: () => void;
}

export const useErrorStore = create<ErrorStore>((set) => ({
  errors: [],
  lastError: null,
  isRecovering: false,
  recoveryAttempts: 0,

  addError: (error: ErrorDetails) =>
    set((state) => ({
      errors: [...state.errors, error].slice(-50), // Keep last 50 errors
      lastError: error,
    })),

  clearError: (errorId?: string) =>
    set((state) => ({
      errors: errorId
        ? state.errors.filter((e) => e.errorId !== errorId)
        : state.errors.slice(1),
      lastError: errorId === state.lastError?.errorId ? null : state.lastError,
    })),

  clearAllErrors: () =>
    set({
      errors: [],
      lastError: null,
    }),

  setRecovering: (isRecovering: boolean) =>
    set({ isRecovering }),

  incrementRecoveryAttempts: () =>
    set((state) => ({ recoveryAttempts: state.recoveryAttempts + 1 })),

  resetRecoveryAttempts: () =>
    set({ recoveryAttempts: 0 }),
}));

// =============================================================================
// Styles (defined early so components can reference them)
// =============================================================================

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem',
    backgroundColor: '#FFF5F5',
    border: '1px solid #FEB2B2',
    borderRadius: '8px',
    margin: '1rem',
  },
  icon: {
    fontSize: '3rem',
    marginBottom: '1rem',
  },
  title: {
    color: '#C53030',
    marginBottom: '0.5rem',
  },
  message: {
    color: '#742A2A',
    textAlign: 'center',
  },
  component: {
    color: '#9B2C2C',
    fontSize: '0.875rem',
  },
  actions: {
    display: 'flex',
    gap: '1rem',
    marginTop: '1rem',
  },
  button: {
    padding: '0.5rem 1rem',
    backgroundColor: '#3182CE',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  buttonSecondary: {
    padding: '0.5rem 1rem',
    backgroundColor: '#718096',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  details: {
    marginTop: '1rem',
    width: '100%',
  },
  stack: {
    padding: '1rem',
    backgroundColor: '#1A202C',
    color: '#68D391',
    borderRadius: '4px',
    overflow: 'auto',
    fontSize: '0.75rem',
    maxHeight: '200px',
  },
  errorDisplay: {
    padding: '1rem',
    borderRadius: '8px',
    border: '1px solid',
    marginBottom: '0.5rem',
  },
  errorHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.5rem',
  },
  errorIcon: {
    fontSize: '1.25rem',
  },
  errorType: {
    fontWeight: 'bold',
    flex: 1,
  },
  dismissButton: {
    background: 'none',
    border: 'none',
    fontSize: '1.5rem',
    cursor: 'pointer',
    padding: '0',
    lineHeight: 1,
  },
  errorMessage: {
    margin: '0.5rem 0',
  },
  errorDetails: {
    padding: '0.5rem',
    backgroundColor: 'rgba(0,0,0,0.05)',
    borderRadius: '4px',
    fontSize: '0.75rem',
    overflow: 'auto',
  },
  timestamp: {
    fontSize: '0.75rem',
    color: '#666',
  },
  toast: {
    position: 'fixed',
    top: '1rem',
    right: '1rem',
    zIndex: 9999,
    maxWidth: '400px',
  },
  loadingContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '2rem',
  },
  spinner: {
    textAlign: 'center',
  },
  spinnerInner: {
    width: '40px',
    height: '40px',
    border: '4px solid #E2E8F0',
    borderTop: '4px solid #3182CE',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
    margin: '0 auto',
  },
  errorContainer: {
    padding: '1rem',
  },
  retryButton: {
    marginTop: '1rem',
    padding: '0.5rem 1rem',
    backgroundColor: '#3182CE',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  emptyState: {
    padding: '2rem',
    textAlign: 'center',
    color: '#718096',
  },
};

// =============================================================================
// Error Boundary Component
// =============================================================================

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  showReload?: boolean;
  componentName?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    
    // Log error to console
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
    
    // Add to global error store
    const errorDetails: ErrorDetails = {
      errorType: error.name,
      errorCode: 'UI_ERROR',
      message: error.message,
      details: {
        componentStack: errorInfo.componentStack,
        componentName: this.props.componentName,
      },
      timestamp: new Date().toISOString(),
      recoverable: true,
    };
    
    useErrorStore.getState().addError(errorDetails);
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
    useErrorStore.getState().clearAllErrors();
  };

  handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="error-boundary-fallback" style={styles.container}>
          <div style={styles.icon}>⚠️</div>
          <h2 style={styles.title}>Something went wrong</h2>
          <p style={styles.message}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          {this.props.componentName && (
            <p style={styles.component}>
              Component: {this.props.componentName}
            </p>
          )}
          <div style={styles.actions}>
            <button onClick={this.handleReset} style={styles.button}>
              Try Again
            </button>
            {this.props.showReload && (
              <button onClick={this.handleReload} style={styles.buttonSecondary}>
                Reload Page
              </button>
            )}
          </div>
          {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
            <details style={styles.details}>
              <summary>Error Details</summary>
              <pre style={styles.stack}>
                {this.state.error?.stack}
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

// =============================================================================
// Error Display Component
// =============================================================================

interface ErrorDisplayProps {
  error: ErrorDetails;
  onDismiss?: () => void;
  showDetails?: boolean;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  onDismiss,
  showDetails = false,
}) => {
  const isWarning = error.recoverable;
  
  return (
    <div
      className={`error-display ${isWarning ? 'warning' : 'error'}`}
      style={{
        ...styles.errorDisplay,
        backgroundColor: isWarning ? '#FFF3CD' : '#F8D7DA',
        borderColor: isWarning ? '#FFC107' : '#DC3545',
      }}
    >
      <div style={styles.errorHeader}>
        <span style={styles.errorIcon}>{isWarning ? '⚠️' : '❌'}</span>
        <span style={styles.errorType}>[{error.errorCode}] {error.errorType}</span>
        {onDismiss && (
          <button onClick={onDismiss} style={styles.dismissButton}>
            ×
          </button>
        )}
      </div>
      <p style={styles.errorMessage}>{error.message}</p>
      {showDetails && error.details && (
        <pre style={styles.errorDetails}>
          {JSON.stringify(error.details, null, 2)}
        </pre>
      )}
      <span style={styles.timestamp}>
        {new Date(error.timestamp).toLocaleString()}
      </span>
    </div>
  );
};

// =============================================================================
// Error Toast Component
// =============================================================================

export const ErrorToast: React.FC = () => {
  const { lastError, clearError } = useErrorStore();

  if (!lastError) return null;

  return (
    <div style={styles.toast}>
      <ErrorDisplay
        error={lastError}
        onDismiss={() => clearError(lastError.errorId)}
      />
    </div>
  );
};

// =============================================================================
// Loading State with Error Handling
// =============================================================================

interface AsyncStateProps<T> {
  loading: boolean;
  error: ErrorDetails | null;
  data: T | null;
  children: (data: T) => ReactNode;
  loadingComponent?: ReactNode;
  errorComponent?: ReactNode;
  onRetry?: () => void;
}

export function AsyncState<T>({
  loading,
  error,
  data,
  children,
  loadingComponent,
  errorComponent,
  onRetry,
}: AsyncStateProps<T>): JSX.Element {
  if (loading) {
    return (
      <div style={styles.loadingContainer}>
        {loadingComponent || (
          <div style={styles.spinner}>
            <div style={styles.spinnerInner} />
            <p>Loading...</p>
          </div>
        )}
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.errorContainer}>
        {errorComponent || (
          <div>
            <ErrorDisplay error={error} showDetails />
            {onRetry && (
              <button onClick={onRetry} style={styles.retryButton}>
                Retry
              </button>
            )}
          </div>
        )}
      </div>
    );
  }

  if (data === null || data === undefined) {
    return (
      <div style={styles.emptyState}>
        <p>No data available</p>
      </div>
    );
  }

  return <>{children(data)}</>;
}

// =============================================================================
// API Error Handler Hook
// =============================================================================

interface UseApiErrorReturn {
  handleError: (error: unknown) => ErrorDetails;
  wrapAsync: <T,>(promise: Promise<T>) => Promise<T>;
}

export const useApiError = (): UseApiErrorReturn => {
  const { addError } = useErrorStore();

  const handleError = (error: unknown): ErrorDetails => {
    let errorDetails: ErrorDetails;

    if (error instanceof Error) {
      errorDetails = {
        errorType: error.name,
        errorCode: 'API_ERROR',
        message: error.message,
        timestamp: new Date().toISOString(),
        recoverable: true,
      };
    } else if (typeof error === 'object' && error !== null && 'error' in error) {
      // API error response
      const apiError = error as { error: Partial<ErrorDetails> };
      errorDetails = {
        errorType: apiError.error.errorType || 'APIError',
        errorCode: apiError.error.errorCode || 'API_ERROR',
        message: apiError.error.message || 'An API error occurred',
        details: apiError.error.details,
        timestamp: new Date().toISOString(),
        recoverable: apiError.error.recoverable ?? true,
      };
    } else {
      errorDetails = {
        errorType: 'UnknownError',
        errorCode: 'UNKNOWN',
        message: String(error),
        timestamp: new Date().toISOString(),
        recoverable: true,
      };
    }

    addError(errorDetails);
    return errorDetails;
  };

  const wrapAsync = async <T,>(promise: Promise<T>): Promise<T> => {
    try {
      return await promise;
    } catch (error) {
      handleError(error);
      throw error;
    }
  };

  return { handleError, wrapAsync };
};

// =============================================================================
// Retry Hook
// =============================================================================

interface UseRetryOptions {
  maxRetries?: number;
  baseDelay?: number;
  onRetry?: (attempt: number) => void;
}

interface UseRetryReturn<T> {
  execute: () => Promise<T>;
  isRetrying: boolean;
  attempts: number;
  reset: () => void;
}

export const useRetry = <T,>(
  asyncFn: () => Promise<T>,
  options: UseRetryOptions = {}
): UseRetryReturn<T> => {
  const { maxRetries = 3, baseDelay = 1000, onRetry } = options;
  const [isRetrying, setIsRetrying] = React.useState(false);
  const [attempts, setAttempts] = React.useState(0);

  const execute = async (): Promise<T> => {
    let lastError: Error | null = null;
    setIsRetrying(true);

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        setAttempts(attempt);
        const result = await asyncFn();
        setIsRetrying(false);
        setAttempts(0);
        return result;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        if (attempt < maxRetries) {
          const delay = baseDelay * Math.pow(2, attempt);
          if (onRetry) onRetry(attempt + 1);
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    setIsRetrying(false);
    throw lastError;
  };

  const reset = () => {
    setAttempts(0);
    setIsRetrying(false);
  };

  return { execute, isRetrying, attempts, reset };
};

export default ErrorBoundary;
