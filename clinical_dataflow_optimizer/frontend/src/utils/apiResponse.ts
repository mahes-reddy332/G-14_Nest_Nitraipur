
/**
 * Global API Response Normalizer
 * 
 * Handles various backend response shapes to ensure the frontend always receives
 * the expected data structure (usually an array), preventing runtime crashes.
 */

export function normalizeApiResponse<T>(responseBody: any, fallback: T[] = []): T[] {
    if (!responseBody) return fallback;

    // 1. Direct Array
    if (Array.isArray(responseBody)) {
        return responseBody;
    }

    // 2. Wrapped in 'data' (Standard REST/Axios sometimes if double wrapped)
    if (Array.isArray(responseBody.data)) {
        return responseBody.data;
    }

    // 3. Wrapped in 'payload' (Common enterprise pattern)
    // Check payload.data
    if (responseBody.payload && Array.isArray(responseBody.payload.data)) {
        return responseBody.payload.data;
    }
    // Check payload directly
    if (Array.isArray(responseBody.payload)) {
        return responseBody.payload;
    }

    // 4. Wrapped in 'items' (Pagination)
    if (Array.isArray(responseBody.items)) {
        return responseBody.items;
    }

    // 5. Wrapped in 'results' (Django/DRF style)
    if (Array.isArray(responseBody.results)) {
        return responseBody.results;
    }

    // 6. Wrapped in 'value' (OData style)
    if (Array.isArray(responseBody.value)) {
        return responseBody.value;
    }

    console.warn('[API Normalizer] Could not extract array from response:', responseBody);
    return fallback;
}
