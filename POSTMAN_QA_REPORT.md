# Postman API Test Suite & UX Validation Summary

## Postman Assets
- Collection: postman/Clinical_Data_Mesh_Dashboard.postman_collection.json
- Environment: postman/Clinical_Data_Mesh_Environment.postman_environment.json

## Coverage
- Health/Readiness
- Dashboard summary + initial load
- Metrics (DQI, cleanliness, queries, SAEs, coding, velocity)
- Core entities (studies, patients, sites)
- Alerts + agents
- NLQ + narratives
- Reports

## Assertions Included
- HTTP status code validation
- Response JSON presence
- Numeric fields validated for type correctness
- KPI fields checked for presence and non-null values

## Usage
1. Import the collection and environment into Postman.
2. Set baseUrl if different from local default.
3. Add authToken only if auth is enabled and a token is required.
4. Run the collection with the environment.

## UX & Data Authenticity Validation
- Automated integration tests compare API outputs to raw source data in:
  - clinical_dataflow_optimizer/tests/test_dashboard_data_integrity.py
- UI rendering safeguards are verified in:
  - clinical_dataflow_optimizer/frontend/src/components/Dashboard/__tests__/EnhancedKPISections.test.tsx
  - clinical_dataflow_optimizer/frontend/src/components/__tests__/ErrorBoundary.test.tsx

## Notes
- Postman API automation via MCP is not executed because a Postman API key is not configured.
- Provide a Postman API key if you want automated runs via API.
