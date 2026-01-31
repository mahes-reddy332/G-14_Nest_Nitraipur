
/**
 * Data Normalization Utilities
 * 
 * Provides robust handling for API responses to prevent frontend crashes
 * when data is missing, null, or in an unexpected format.
 */

/**
 * Normalizes an API response that is expected to be an array.
 * 
 * @param data The raw data from the API
 * @param fallbackValue Value to return if data is invalid (default: [])
 * @returns An array (either the data itself or the fallback)
 */
export function normalizeArray<T>(data: any, fallbackValue: T[] = []): T[] {
    if (data === null || data === undefined) {
        return fallbackValue
    }

    if (Array.isArray(data)) {
        return data
    }

    // If it's an object but we expected an array, it might be a paginated response
    // Check common pagination patterns
    if (typeof data === 'object') {
        if (Array.isArray(data.items)) return data.items
        if (Array.isArray(data.data)) return data.data
        if (Array.isArray(data.results)) return data.results
    }

    // If we can't find an array, return the fallback
    console.warn('API Response expected array but got:', typeof data, data)
    return fallbackValue
}

/**
 * Safely accesses a string property, handling null/undefined/numbers.
 */
export function safeString(value: any, fallback: string = ''): string {
    if (value === null || value === undefined) return fallback
    return String(value)
}

/**
 * Safely accesses a number property, handling strings parsing.
 */
export function safeNumber(value: any, fallback: number = 0): number {
    if (value === null || value === undefined) return fallback
    if (typeof value === 'number') return value
    const parsed = Number(value)
    return isNaN(parsed) ? fallback : parsed
}
