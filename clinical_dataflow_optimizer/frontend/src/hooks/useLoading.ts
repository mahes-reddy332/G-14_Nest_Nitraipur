import { useStore } from '../store'

export function useLoading(operation: string) {
  const loadingState = useStore(state => state.getLoading(operation))
  const setLoading = useStore(state => state.setLoading)

  return {
    isLoading: loadingState?.isLoading || false,
    progress: loadingState?.progress,
    setLoading: (isLoading: boolean, progress?: number) =>
      setLoading(operation, isLoading, progress),
  }
}

export function useGlobalLoading() {
  const loadingStates = useStore(state => state.loadingStates)
  const isAnyLoading = Object.values(loadingStates).some(state => state.isLoading)

  return {
    isAnyLoading,
    loadingStates,
  }
}