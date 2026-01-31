# Dashboard API Integration, Rendering, and E2E Validation

## Scope
- API integration tests for dashboard endpoints and data correctness.
- Frontend rendering tests for loading/error paths and visible error states.
- End-to-end dashboard validation using Playwright + raw source file comparison.

## Implemented Test Suites

### API Integration + Data Integrity
- [clinical_dataflow_optimizer/tests/test_dashboard_data_integrity.py](clinical_dataflow_optimizer/tests/test_dashboard_data_integrity.py)
  - Validates `/api/dashboard/initial-load` against raw CPID source data.
  - Validates `/api/metrics/cleanliness` and `/api/metrics/queries` against raw source aggregates.
  - Fails if required CPID columns are missing or study folders are absent.

### Authentication Enforcement
- [clinical_dataflow_optimizer/tests/test_auth_enforcement.py](clinical_dataflow_optimizer/tests/test_auth_enforcement.py)
  - Verifies 401 when auth is enabled and token is missing.
  - Issues bootstrap token and confirms authorized access.

### Frontend Rendering Tests
- [clinical_dataflow_optimizer/frontend/src/components/Dashboard/__tests__/EnhancedKPISections.test.tsx](clinical_dataflow_optimizer/frontend/src/components/Dashboard/__tests__/EnhancedKPISections.test.tsx)
  - Ensures loading skeletons render during async fetch.
  - Confirms KPI values are rendered from API payloads.
- [clinical_dataflow_optimizer/frontend/src/components/__tests__/ErrorBoundary.test.tsx](clinical_dataflow_optimizer/frontend/src/components/__tests__/ErrorBoundary.test.tsx)
  - Confirms error UI is shown (prevents white-screen regressions).

### End-to-End Dashboard Workflow
- [clinical_dataflow_optimizer/tests/test_e2e_dashboard_playwright.py](clinical_dataflow_optimizer/tests/test_e2e_dashboard_playwright.py)
  - Opens the dashboard UI and extracts KPI values.
  - Computes expected KPIs directly from raw CPID source files.
  - Fails if UI KPIs deviate from raw values.

## Test Tooling Added
- Frontend: Vitest + Testing Library
  - [clinical_dataflow_optimizer/frontend/vitest.config.ts](clinical_dataflow_optimizer/frontend/vitest.config.ts)
  - [clinical_dataflow_optimizer/frontend/src/setupTests.ts](clinical_dataflow_optimizer/frontend/src/setupTests.ts)
- Backend: pytest-playwright for E2E
  - Added to requirements

## Execution Commands

### Backend API Tests
- From [clinical_dataflow_optimizer](clinical_dataflow_optimizer):
  - `pytest -m integration`

### E2E (Dashboard UI + Raw Data)
- Start API server first, then:
  - `pytest -m e2e`

### Frontend Component Tests
- From [clinical_dataflow_optimizer/frontend](clinical_dataflow_optimizer/frontend):
  - `npm run test`

## Quality Gates
- All API tests must pass with real data present.
- No frontend test should allow blank screens without an error or loader.
- E2E KPI comparisons must match raw source values exactly (rounded to 1 decimal for DQI).

## Production Readiness Checklist
- [x] Dashboard KPIs are validated against raw source data.
- [x] No mock/demo data is used in API or UI fallback paths.
- [x] Frontend renders explicit error state on crash.
- [x] Loading states prevent blank/white screens.
- [x] Auth enforcement verified when enabled.

## Notes
- E2E test requires the API server to be running at `E2E_BASE_URL` (default: http://127.0.0.1:8000).
- If raw data is missing, tests will fail early with an explicit assertion.
