# Backend-Frontend Integration Guide
## Clinical Trial Operational Dataflow Metrics Web Application

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [API Design Specifications](#api-design-specifications)
3. [Frontend Service Layer Implementation](#frontend-service-layer-implementation)
4. [CORS Configuration](#cors-configuration)
5. [Authentication & Authorization](#authentication--authorization)
6. [Data Flow & State Management](#data-flow--state-management)
7. [Error Handling & Retry Logic](#error-handling--retry-logic)
8. [Environment Configuration](#environment-configuration)
9. [Type Safety & Validation](#type-safety--validation)
10. [Real-time Data Updates](#real-time-data-updates)
11. [Performance Optimization](#performance-optimization)
12. [Testing Strategy](#testing-strategy)
13. [Deployment Checklist](#deployment-checklist)

---

## 1. Architecture Overview

### Backend Architecture Requirements
```
Backend Stack:
├── API Gateway (Node.js/Express or FastAPI/Django)
├── Authentication Service (JWT/OAuth2)
├── Database Layer (PostgreSQL/MongoDB)
├── Data Integration Services
│   ├── EDC System Connector
│   ├── CTMS Connector
│   ├── Lab System Connector
│   └── Safety Database Connector
├── Business Logic Layer
│   ├── Metrics Calculation Engine
│   ├── Data Quality Index Calculator
│   ├── AI/ML Services Integration
│   └── Report Generation Service
└── WebSocket Server (for real-time updates)
```

### Frontend Architecture Requirements
```
Frontend Stack:
├── React Application (TypeScript)
├── API Service Layer (Axios/Fetch)
├── State Management (Redux Toolkit/Zustand/React Query)
├── Authentication Context
├── WebSocket Client
├── Type Definitions
└── Error Boundary Components
```

---

## 2. API Design Specifications

### Base URL Configuration
```typescript
// src/config/api.config.ts
export const API_CONFIG = {
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
};

export const WS_CONFIG = {
  wsURL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws',
};
```

### API Endpoint Structure

#### 2.1 Authentication Endpoints
```typescript
POST   /auth/login
POST   /auth/logout
POST   /auth/refresh
GET    /auth/me
POST   /auth/forgot-password
POST   /auth/reset-password
```

#### 2.2 Dashboard & Metrics Endpoints
```typescript
GET    /dashboard/summary
       Response: {
         totalSites: number;
         totalSubjects: number;
         cleanPatientPercentage: number;
         dataQualityIndex: number;
         openQueriesCount: number;
         missingSAECount: number;
         overdueCRFsCount: number;
         unresolvedSAEsCount: number;
       }

GET    /dashboard/regional-performance
       Query Params: ?regionId=string&countryId=string
       Response: RegionalPerformanceData[]

GET    /dashboard/alerts
       Query Params: ?severity=critical|warning|info&limit=number
       Response: Alert[]

GET    /dashboard/trends
       Query Params: ?metric=string&startDate=string&endDate=string
       Response: TrendData[]
```

#### 2.3 EDC Metrics Endpoints
```typescript
GET    /edc-metrics/subjects
       Query Params: {
         region?: string[];
         country?: string[];
         siteId?: string[];
         subjectId?: string;
         status?: string[];
         dateFrom?: string;
         dateTo?: string;
         isClean?: boolean;
         page?: number;
         pageSize?: number;
         sortBy?: string;
         sortOrder?: 'asc' | 'desc';
       }
       Response: {
         data: SubjectMetric[];
         total: number;
         page: number;
         pageSize: number;
         totalPages: number;
       }

GET    /edc-metrics/subjects/:subjectId
       Response: SubjectDetailMetric

GET    /edc-metrics/derived-metrics
       Query Params: ?aggregateBy=site|region|country
       Response: DerivedMetrics[]

POST   /edc-metrics/export
       Body: { filters: FilterObject; format: 'excel' | 'csv' | 'pdf' }
       Response: { downloadUrl: string; expiresAt: string }
```

#### 2.4 Data Quality Endpoints
```typescript
GET    /data-quality/queries
       Query Params: {
         type?: string[];
         status?: string[];
         siteId?: string[];
         ageInDays?: { min: number; max: number };
         priority?: string[];
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<Query>

GET    /data-quality/queries/:queryId
       Response: QueryDetail

PUT    /data-quality/queries/:queryId
       Body: { status?: string; response?: string; assignedTo?: string }
       Response: Query

POST   /data-quality/queries/:queryId/comments
       Body: { comment: string; attachments?: string[] }
       Response: Comment

GET    /data-quality/query-metrics
       Response: {
         totalOpen: number;
         byType: Record<string, number>;
         byAge: Record<string, number>;
         averageResolutionTime: number;
         resolutionRate: number;
       }

GET    /data-quality/non-conformant
       Query Params: ?siteId=string&ruleType=string
       Response: NonConformantData[]

GET    /data-quality/dqi
       Query Params: ?aggregateBy=overall|region|country|site|subject
       Response: DataQualityIndex[]
```

#### 2.5 Visit Management Endpoints
```typescript
GET    /visits/missing
       Query Params: {
         siteId?: string[];
         daysOverdue?: { min: number; max: number };
         visitType?: string[];
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<MissingVisit>

GET    /visits/compliance-metrics
       Query Params: ?siteId=string&dateFrom=string&dateTo=string
       Response: {
         averageDaysOverdue: number;
         visitComplianceRate: number;
         sitesWithMultipleOverdue: SiteOverdueCount[];
         subjectsWithMultipleOverdue: SubjectOverdueCount[];
       }

GET    /visits/calendar-heatmap
       Query Params: ?year=number&month=number
       Response: CalendarHeatmapData[]

PUT    /visits/:visitId/status
       Body: { status: string; notes?: string }
       Response: Visit
```

#### 2.6 Laboratory Data Endpoints
```typescript
GET    /laboratory/missing-data
       Query Params: {
         missingElement?: 'lab_name' | 'reference_range' | 'unit';
         priority?: string[];
         siteId?: string[];
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<MissingLabData>

GET    /laboratory/reconciliation-metrics
       Response: {
         totalMissingLabNames: number;
         totalMissingRanges: number;
         totalMissingUnits: number;
         averageResolutionTime: number;
         byLabType: Record<string, number>;
       }

PUT    /laboratory/:labId
       Body: { labName?: string; referenceRange?: string; unit?: string }
       Response: LabData
```

#### 2.7 Safety Monitoring Endpoints
```typescript
GET    /safety/sae-dashboard
       Query Params: {
         view: 'data_management' | 'safety';
         status?: string[];
         severity?: string[];
         siteId?: string[];
         daysOpen?: { min: number; max: number };
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<SAE>

GET    /safety/sae/:saeId
       Response: SAEDetail

PUT    /safety/sae/:saeId
       Body: {
         status?: string;
         reviewNotes?: string;
         assignedTo?: string;
         causalityAssessment?: string;
       }
       Response: SAE

GET    /safety/sae-metrics
       Response: {
         totalSAEs: number;
         bySeverity: Record<string, number>;
         averageTimeToResolution: number;
         pendingReviews: number;
         expeditedReportsDue: number;
       }

POST   /safety/sae/:saeId/comments
       Body: { comment: string; type: 'dm' | 'safety' }
       Response: Comment
```

#### 2.8 Coding Endpoints
```typescript
// MedDRA Coding
GET    /coding/meddra
       Query Params: {
         status?: string[];
         termType?: string[];
         siteId?: string[];
         daysPending?: { min: number; max: number };
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<MedDRATerm>

PUT    /coding/meddra/:termId
       Body: {
         codedTerm?: string;
         preferredTerm?: string;
         soc?: string;
         status?: string;
       }
       Response: MedDRATerm

GET    /coding/meddra/metrics
       Response: CodingMetrics

// WHO Drug Coding
GET    /coding/whodrug
       Query Params: { /* similar to meddra */ }
       Response: PaginatedResponse<WHODrugTerm>

PUT    /coding/whodrug/:termId
       Body: {
         codedTerm?: string;
         drugCode?: string;
         atcClass?: string;
         status?: string;
       }
       Response: WHODrugTerm

GET    /coding/whodrug/metrics
       Response: CodingMetrics
```

#### 2.9 Forms & Verification Endpoints
```typescript
GET    /forms/sdv-status
       Query Params: {
         siteId?: string[];
         subjectId?: string;
         verificationStatus?: string[];
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<SDVStatus>

GET    /forms/form-status
       Query Params: {
         status?: ('frozen' | 'locked' | 'signed')[];
         siteId?: string[];
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<FormStatus>

GET    /forms/overdue-crfs
       Query Params: {
         daysOverdue?: { min: number; max: number };
         priority?: string[];
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<OverdueCRF>

GET    /forms/inactivated
       Query Params: { siteId?: string[]; reason?: string[] }
       Response: PaginatedResponse<InactivatedForm>

PUT    /forms/:formId/status
       Body: { status: string; notes?: string }
       Response: FormStatus
```

#### 2.10 CRA Activity Endpoints
```typescript
GET    /cra/performance-metrics
       Query Params: ?craId=string
       Response: {
         totalActiveCRAs: number;
         totalMonitoringVisits: number;
         averageVisitsPerCRA: number;
         sitesPerCRA: CRAWorkload[];
       }

GET    /cra/monitoring-visits
       Query Params: {
         craId?: string;
         siteId?: string[];
         visitType?: string[];
         dateFrom?: string;
         dateTo?: string;
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<MonitoringVisit>

GET    /cra/follow-up-items
       Query Params: {
         craId?: string;
         status?: string[];
         priority?: string[];
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<FollowUpItem>

PUT    /cra/follow-up-items/:itemId
       Body: { status?: string; resolutionNotes?: string }
       Response: FollowUpItem
```

#### 2.11 Reports & Analytics Endpoints
```typescript
GET    /reports/clean-data-milestones
       Response: {
         milestones: Milestone[];
         overallReadiness: number;
         blockingIssues: BlockingIssue[];
         daysToTargetLock: number;
       }

GET    /reports/derived-metrics-summary
       Query Params: ?aggregateBy=site|region|overall
       Response: DerivedMetricsSummary

POST   /reports/generate-cra-report
       Body: { craId?: string; siteId?: string; dateFrom: string; dateTo: string }
       Response: { reportUrl: string; expiresAt: string }

POST   /reports/data-readiness-check
       Body: { milestoneId: string }
       Response: DataReadinessReport
```

#### 2.12 Third-Party Data Endpoints
```typescript
GET    /third-party/unresolved-issues
       Query Params: {
         dataSource?: string[];
         priority?: string[];
         siteId?: string[];
         page?: number;
         pageSize?: number;
       }
       Response: PaginatedResponse<ThirdPartyIssue>

PUT    /third-party/issues/:issueId
       Body: { status?: string; notes?: string }
       Response: ThirdPartyIssue
```

#### 2.13 Collaboration Endpoints
```typescript
POST   /collaboration/alerts
       Body: {
         type: string;
         recipients: string[];
         message: string;
         triggerCondition?: object;
       }
       Response: Alert

GET    /collaboration/notifications
       Query Params: ?unreadOnly=boolean
       Response: Notification[]

PUT    /collaboration/notifications/:notificationId/read
       Response: Notification

POST   /collaboration/tasks
       Body: {
         title: string;
         description: string;
         assignedTo: string;
         dueDate: string;
         priority: string;
         relatedEntity: { type: string; id: string };
       }
       Response: Task

GET    /collaboration/tasks
       Query Params: ?assignedTo=string&status=string
       Response: Task[]
```

#### 2.14 AI-Powered Endpoints
```typescript
POST   /ai/natural-language-query
       Body: { query: string; context?: object }
       Response: {
         answer: string;
         supportingData: any[];
         suggestedActions: string[];
         relatedQueries: string[];
       }

POST   /ai/generate-report
       Body: { reportType: string; filters: object }
       Response: { reportContent: string; visualizations: any[] }

GET    /ai/recommendations
       Query Params: ?entityType=site|subject|cra&entityId=string
       Response: Recommendation[]

GET    /ai/predictive-analytics
       Query Params: ?metric=string
       Response: PredictiveAnalytics
```

#### 2.15 Filters & Metadata Endpoints
```typescript
GET    /metadata/regions
       Response: Region[]

GET    /metadata/countries
       Query Params: ?regionId=string
       Response: Country[]

GET    /metadata/sites
       Query Params: ?countryId=string&regionId=string
       Response: Site[]

GET    /metadata/subjects
       Query Params: ?siteId=string
       Response: Subject[]

GET    /metadata/visit-types
       Response: VisitType[]

GET    /metadata/query-types
       Response: QueryType[]

GET    /metadata/user-roles
       Response: Role[]
```

---

## 3. Frontend Service Layer Implementation

### 3.1 Base API Client Setup

```typescript
// src/services/api/client.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { API_CONFIG } from '../../config/api.config';
import { getAuthToken, refreshAuthToken, clearAuth } from '../../utils/auth';

class APIClient {
  private client: AxiosInstance;
  private isRefreshing = false;
  private failedQueue: Array<{
    resolve: (value?: any) => void;
    reject: (reason?: any) => void;
  }> = [];

  constructor() {
    this.client = axios.create({
      baseURL: API_CONFIG.baseURL,
      timeout: API_CONFIG.timeout,
      headers: API_CONFIG.headers,
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request Interceptor
    this.client.interceptors.request.use(
      (config) => {
        const token = getAuthToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response Interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

        // Handle 401 Unauthorized - Token Refresh Logic
        if (error.response?.status === 401 && !originalRequest._retry) {
          if (this.isRefreshing) {
            return new Promise((resolve, reject) => {
              this.failedQueue.push({ resolve, reject });
            })
              .then((token) => {
                originalRequest.headers!.Authorization = `Bearer ${token}`;
                return this.client(originalRequest);
              })
              .catch((err) => {
                return Promise.reject(err);
              });
          }

          originalRequest._retry = true;
          this.isRefreshing = true;

          try {
            const newToken = await refreshAuthToken();
            this.failedQueue.forEach((prom) => prom.resolve(newToken));
            this.failedQueue = [];
            originalRequest.headers!.Authorization = `Bearer ${newToken}`;
            return this.client(originalRequest);
          } catch (refreshError) {
            this.failedQueue.forEach((prom) => prom.reject(refreshError));
            this.failedQueue = [];
            clearAuth();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          } finally {
            this.isRefreshing = false;
          }
        }

        return Promise.reject(error);
      }
    );
  }

  // Generic request methods
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response: AxiosResponse<T> = await this.client.get(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response: AxiosResponse<T> = await this.client.post(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response: AxiosResponse<T> = await this.client.put(url, data, config);
    return response.data;
  }

  async patch<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response: AxiosResponse<T> = await this.client.patch(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response: AxiosResponse<T> = await this.client.delete(url, config);
    return response.data;
  }
}

export const apiClient = new APIClient();
```

### 3.2 Service Layer for Each Module

```typescript
// src/services/api/edcMetrics.service.ts
import { apiClient } from './client';
import {
  SubjectMetric,
  SubjectDetailMetric,
  DerivedMetrics,
  EDCMetricsFilters,
  PaginatedResponse,
} from '../../types';

export class EDCMetricsService {
  private basePath = '/edc-metrics';

  async getSubjects(
    filters: EDCMetricsFilters
  ): Promise<PaginatedResponse<SubjectMetric>> {
    const params = this.buildQueryParams(filters);
    return apiClient.get<PaginatedResponse<SubjectMetric>>(
      `${this.basePath}/subjects`,
      { params }
    );
  }

  async getSubjectById(subjectId: string): Promise<SubjectDetailMetric> {
    return apiClient.get<SubjectDetailMetric>(
      `${this.basePath}/subjects/${subjectId}`
    );
  }

  async getDerivedMetrics(aggregateBy: string): Promise<DerivedMetrics[]> {
    return apiClient.get<DerivedMetrics[]>(
      `${this.basePath}/derived-metrics`,
      { params: { aggregateBy } }
    );
  }

  async exportData(
    filters: EDCMetricsFilters,
    format: 'excel' | 'csv' | 'pdf'
  ): Promise<{ downloadUrl: string; expiresAt: string }> {
    return apiClient.post(`${this.basePath}/export`, { filters, format });
  }

  private buildQueryParams(filters: EDCMetricsFilters): Record<string, any> {
    const params: Record<string, any> = {};
    
    if (filters.region?.length) params.region = filters.region.join(',');
    if (filters.country?.length) params.country = filters.country.join(',');
    if (filters.siteId?.length) params.siteId = filters.siteId.join(',');
    if (filters.subjectId) params.subjectId = filters.subjectId;
    if (filters.status?.length) params.status = filters.status.join(',');
    if (filters.dateFrom) params.dateFrom = filters.dateFrom;
    if (filters.dateTo) params.dateTo = filters.dateTo;
    if (filters.isClean !== undefined) params.isClean = filters.isClean;
    if (filters.page) params.page = filters.page;
    if (filters.pageSize) params.pageSize = filters.pageSize;
    if (filters.sortBy) params.sortBy = filters.sortBy;
    if (filters.sortOrder) params.sortOrder = filters.sortOrder;
    
    return params;
  }
}

export const edcMetricsService = new EDCMetricsService();
```

```typescript
// src/services/api/dataQuality.service.ts
import { apiClient } from './client';
import {
  Query,
  QueryDetail,
  QueryFilters,
  QueryMetrics,
  NonConformantData,
  DataQualityIndex,
  Comment,
  PaginatedResponse,
} from '../../types';

export class DataQualityService {
  private basePath = '/data-quality';

  async getQueries(filters: QueryFilters): Promise<PaginatedResponse<Query>> {
    const params = this.buildQueryParams(filters);
    return apiClient.get<PaginatedResponse<Query>>(
      `${this.basePath}/queries`,
      { params }
    );
  }

  async getQueryById(queryId: string): Promise<QueryDetail> {
    return apiClient.get<QueryDetail>(`${this.basePath}/queries/${queryId}`);
  }

  async updateQuery(
    queryId: string,
    updates: Partial<Query>
  ): Promise<Query> {
    return apiClient.put<Query>(`${this.basePath}/queries/${queryId}`, updates);
  }

  async addComment(
    queryId: string,
    comment: string,
    attachments?: string[]
  ): Promise<Comment> {
    return apiClient.post<Comment>(
      `${this.basePath}/queries/${queryId}/comments`,
      { comment, attachments }
    );
  }

  async getQueryMetrics(): Promise<QueryMetrics> {
    return apiClient.get<QueryMetrics>(`${this.basePath}/query-metrics`);
  }

  async getNonConformantData(
    siteId?: string,
    ruleType?: string
  ): Promise<NonConformantData[]> {
    return apiClient.get<NonConformantData[]>(
      `${this.basePath}/non-conformant`,
      { params: { siteId, ruleType } }
    );
  }

  async getDataQualityIndex(
    aggregateBy: string
  ): Promise<DataQualityIndex[]> {
    return apiClient.get<DataQualityIndex[]>(`${this.basePath}/dqi`, {
      params: { aggregateBy },
    });
  }

  private buildQueryParams(filters: QueryFilters): Record<string, any> {
    const params: Record<string, any> = {};
    
    if (filters.type?.length) params.type = filters.type.join(',');
    if (filters.status?.length) params.status = filters.status.join(',');
    if (filters.siteId?.length) params.siteId = filters.siteId.join(',');
    if (filters.ageInDays) {
      params.ageInDaysMin = filters.ageInDays.min;
      params.ageInDaysMax = filters.ageInDays.max;
    }
    if (filters.priority?.length) params.priority = filters.priority.join(',');
    if (filters.page) params.page = filters.page;
    if (filters.pageSize) params.pageSize = filters.pageSize;
    
    return params;
  }
}

export const dataQualityService = new DataQualityService();
```

```typescript
// src/services/api/index.ts
// Export all services from a central location
export { apiClient } from './client';
export { edcMetricsService } from './edcMetrics.service';
export { dataQualityService } from './dataQuality.service';
export { visitManagementService } from './visitManagement.service';
export { laboratoryService } from './laboratory.service';
export { safetyService } from './safety.service';
export { meddraCodingService } from './meddraCoding.service';
export { whoDrugCodingService } from './whoDrugCoding.service';
export { formsService } from './forms.service';
export { craActivityService } from './craActivity.service';
export { reportsService } from './reports.service';
export { thirdPartyService } from './thirdParty.service';
export { collaborationService } from './collaboration.service';
export { aiService } from './ai.service';
export { metadataService } from './metadata.service';
export { dashboardService } from './dashboard.service';
export { authService } from './auth.service';
```

### 3.3 React Query / TanStack Query Integration

```typescript
// src/hooks/useEDCMetrics.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { edcMetricsService } from '../services/api';
import { EDCMetricsFilters } from '../types';

export const useEDCMetrics = (filters: EDCMetricsFilters) => {
  return useQuery({
    queryKey: ['edc-metrics', 'subjects', filters],
    queryFn: () => edcMetricsService.getSubjects(filters),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: true,
  });
};

export const useSubjectDetail = (subjectId: string) => {
  return useQuery({
    queryKey: ['edc-metrics', 'subject', subjectId],
    queryFn: () => edcMetricsService.getSubjectById(subjectId),
    enabled: !!subjectId,
  });
};

export const useDerivedMetrics = (aggregateBy: string) => {
  return useQuery({
    queryKey: ['edc-metrics', 'derived', aggregateBy],
    queryFn: () => edcMetricsService.getDerivedMetrics(aggregateBy),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const useExportEDCMetrics = () => {
  return useMutation({
    mutationFn: ({
      filters,
      format,
    }: {
      filters: EDCMetricsFilters;
      format: 'excel' | 'csv' | 'pdf';
    }) => edcMetricsService.exportData(filters, format),
    onSuccess: (data) => {
      // Trigger download
      window.open(data.downloadUrl, '_blank');
    },
  });
};
```

```typescript
// src/hooks/useDataQuality.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { dataQualityService } from '../services/api';
import { QueryFilters } from '../types';

export const useQueries = (filters: QueryFilters) => {
  return useQuery({
    queryKey: ['data-quality', 'queries', filters],
    queryFn: () => dataQualityService.getQueries(filters),
    staleTime: 5 * 60 * 1000,
  });
};

export const useQueryDetail = (queryId: string) => {
  return useQuery({
    queryKey: ['data-quality', 'query', queryId],
    queryFn: () => dataQualityService.getQueryById(queryId),
    enabled: !!queryId,
  });
};

export const useUpdateQuery = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      queryId,
      updates,
    }: {
      queryId: string;
      updates: Partial<any>;
    }) => dataQualityService.updateQuery(queryId, updates),
    onSuccess: (data, variables) => {
      // Invalidate and refetch
      queryClient.invalidateQueries({ queryKey: ['data-quality', 'queries'] });
      queryClient.invalidateQueries({
        queryKey: ['data-quality', 'query', variables.queryId],
      });
      queryClient.invalidateQueries({ queryKey: ['data-quality', 'query-metrics'] });
    },
  });
};

export const useQueryMetrics = () => {
  return useQuery({
    queryKey: ['data-quality', 'query-metrics'],
    queryFn: () => dataQualityService.getQueryMetrics(),
    staleTime: 5 * 60 * 1000,
  });
};

export const useDataQualityIndex = (aggregateBy: string) => {
  return useQuery({
    queryKey: ['data-quality', 'dqi', aggregateBy],
    queryFn: () => dataQualityService.getDataQualityIndex(aggregateBy),
    staleTime: 10 * 60 * 1000,
  });
};
```

### 3.4 Component Integration Example

```typescript
// src/pages/EDCMetrics.tsx
import React, { useState } from 'react';
import { useEDCMetrics, useDerivedMetrics, useExportEDCMetrics } from '../hooks/useEDCMetrics';
import { EDCMetricsFilters } from '../types';
import { DataTable, FilterPanel, LoadingSpinner, ErrorMessage } from '../components';

export const EDCMetrics: React.FC = () => {
  const [filters, setFilters] = useState<EDCMetricsFilters>({
    page: 1,
    pageSize: 50,
    sortBy: 'subjectId',
    sortOrder: 'asc',
  });

  // Fetch data using React Query
  const {
    data: metricsData,
    isLoading,
    isError,
    error,
    refetch,
  } = useEDCMetrics(filters);

  const {
    data: derivedMetrics,
    isLoading: isDerivedLoading,
  } = useDerivedMetrics('site');

  const exportMutation = useExportEDCMetrics();

  // NO MOCK DATA - Only use data from API
  if (isLoading) {
    return <LoadingSpinner message="Loading EDC metrics..." />;
  }

  if (isError) {
    return (
      <ErrorMessage
        error={error}
        onRetry={refetch}
        message="Failed to load EDC metrics. Please try again."
      />
    );
  }

  // Only render if data exists from API
  if (!metricsData) {
    return <div>No data available</div>;
  }

  const handleFilterChange = (newFilters: Partial<EDCMetricsFilters>) => {
    setFilters((prev) => ({ ...prev, ...newFilters, page: 1 }));
  };

  const handlePageChange = (page: number) => {
    setFilters((prev) => ({ ...prev, page }));
  };

  const handleSort = (sortBy: string, sortOrder: 'asc' | 'desc') => {
    setFilters((prev) => ({ ...prev, sortBy, sortOrder }));
  };

  const handleExport = (format: 'excel' | 'csv' | 'pdf') => {
    exportMutation.mutate({ filters, format });
  };

  return (
    <div className="edc-metrics-container">
      <div className="header">
        <h1>Patient & Site Metrics</h1>
        <div className="actions">
          <button onClick={() => refetch()}>
            Refresh
          </button>
          <button onClick={() => handleExport('excel')}>
            Export to Excel
          </button>
        </div>
      </div>

      <FilterPanel
        filters={filters}
        onFilterChange={handleFilterChange}
      />

      {isDerivedLoading ? (
        <LoadingSpinner size="small" />
      ) : (
        <DerivedMetricsPanel data={derivedMetrics} />
      )}

      <DataTable
        data={metricsData.data}
        columns={edcMetricsColumns}
        totalRows={metricsData.total}
        currentPage={metricsData.page}
        pageSize={metricsData.pageSize}
        onPageChange={handlePageChange}
        onSort={handleSort}
        sortBy={filters.sortBy}
        sortOrder={filters.sortOrder}
      />
    </div>
  );
};
```

---

## 4. CORS Configuration

### 4.1 Backend CORS Setup (Node.js/Express Example)

```javascript
// backend/src/middleware/cors.js
const cors = require('cors');

const corsOptions = {
  origin: function (origin, callback) {
    // Allow requests from these origins
    const allowedOrigins = [
      'http://localhost:3000',
      'http://localhost:3001',
      'https://clinicaltrial-app.example.com',
      'https://staging.clinicaltrial-app.example.com',
    ];

    // Allow requests with no origin (mobile apps, Postman, etc.)
    if (!origin) return callback(null, true);

    if (allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true, // Allow cookies to be sent
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: [
    'Content-Type',
    'Authorization',
    'X-Requested-With',
    'Accept',
    'Origin',
  ],
  exposedHeaders: ['Content-Range', 'X-Content-Range'],
  maxAge: 600, // Cache preflight request for 10 minutes
};

module.exports = cors(corsOptions);
```

```javascript
// backend/src/app.js
const express = require('express');
const corsMiddleware = require('./middleware/cors');

const app = express();

// Apply CORS middleware BEFORE routes
app.use(corsMiddleware);

// Other middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/v1', require('./routes'));

// Error handling
app.use((err, req, res, next) => {
  if (err.message === 'Not allowed by CORS') {
    res.status(403).json({ error: 'CORS policy violation' });
  } else {
    next(err);
  }
});

module.exports = app;
```

### 4.2 Backend CORS Setup (FastAPI/Python Example)

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

app = FastAPI(title="Clinical Trial API", version="1.0.0")

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://clinicaltrial-app.example.com",
    "https://staging.clinicaltrial-app.example.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
    expose_headers=["Content-Range", "X-Content-Range"],
    max_age=600,
)

# Include routers
from app.routers import (
    auth,
    dashboard,
    edc_metrics,
    data_quality,
    visits,
    laboratory,
    safety,
    coding,
    forms,
    cra,
    reports,
    collaboration,
    ai,
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(edc_metrics.router, prefix="/api/v1/edc-metrics", tags=["EDC Metrics"])
app.include_router(data_quality.router, prefix="/api/v1/data-quality", tags=["Data Quality"])
# ... include other routers
```

### 4.3 Frontend Axios Configuration (Already Handled in Client)

The CORS credentials are automatically handled in the API client setup:

```typescript
// src/services/api/client.ts
// Already configured with credentials in axios instance
this.client = axios.create({
  baseURL: API_CONFIG.baseURL,
  timeout: API_CONFIG.timeout,
  headers: API_CONFIG.headers,
  withCredentials: true, // IMPORTANT: Enable credentials for CORS
});
```

---

## 5. Authentication & Authorization

### 5.1 Auth Utilities

```typescript
// src/utils/auth.ts
const TOKEN_KEY = 'auth_token';
const REFRESH_TOKEN_KEY = 'refresh_token';
const USER_KEY = 'user_info';

export const getAuthToken = (): string | null => {
  return localStorage.getItem(TOKEN_KEY);
};

export const setAuthToken = (token: string): void => {
  localStorage.setItem(TOKEN_KEY, token);
};

export const getRefreshToken = (): string | null => {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
};

export const setRefreshToken = (token: string): void => {
  localStorage.setItem(REFRESH_TOKEN_KEY, token);
};

export const getUserInfo = (): any | null => {
  const userInfo = localStorage.getItem(USER_KEY);
  return userInfo ? JSON.parse(userInfo) : null;
};

export const setUserInfo = (user: any): void => {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
};

export const clearAuth = (): void => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
};

export const refreshAuthToken = async (): Promise<string> => {
  const refreshToken = getRefreshToken();
  
  if (!refreshToken) {
    throw new Error('No refresh token available');
  }

  try {
    const response = await fetch(`${process.env.REACT_APP_API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refreshToken }),
    });

    if (!response.ok) {
      throw new Error('Token refresh failed');
    }

    const data = await response.json();
    setAuthToken(data.accessToken);
    setRefreshToken(data.refreshToken);
    
    return data.accessToken;
  } catch (error) {
    clearAuth();
    throw error;
  }
};

export const isAuthenticated = (): boolean => {
  return !!getAuthToken();
};
```

### 5.2 Auth Service

```typescript
// src/services/api/auth.service.ts
import { apiClient } from './client';
import { setAuthToken, setRefreshToken, setUserInfo, clearAuth } from '../../utils/auth';

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface AuthResponse {
  accessToken: string;
  refreshToken: string;
  user: {
    id: string;
    email: string;
    name: string;
    role: string;
    permissions: string[];
  };
}

export class AuthService {
  private basePath = '/auth';

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await apiClient.post<AuthResponse>(
      `${this.basePath}/login`,
      credentials
    );

    // Store tokens and user info
    setAuthToken(response.accessToken);
    setRefreshToken(response.refreshToken);
    setUserInfo(response.user);

    return response;
  }

  async logout(): Promise<void> {
    try {
      await apiClient.post(`${this.basePath}/logout`);
    } finally {
      clearAuth();
      window.location.href = '/login';
    }
  }

  async getCurrentUser(): Promise<AuthResponse['user']> {
    return apiClient.get<AuthResponse['user']>(`${this.basePath}/me`);
  }

  async forgotPassword(email: string): Promise<{ message: string }> {
    return apiClient.post(`${this.basePath}/forgot-password`, { email });
  }

  async resetPassword(token: string, newPassword: string): Promise<{ message: string }> {
    return apiClient.post(`${this.basePath}/reset-password`, {
      token,
      newPassword,
    });
  }
}

export const authService = new AuthService();
```

### 5.3 Protected Route Component

```typescript
// src/components/ProtectedRoute.tsx
import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { isAuthenticated, getUserInfo } from '../utils/auth';

interface ProtectedRouteProps {
  allowedRoles?: string[];
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ allowedRoles }) => {
  const authenticated = isAuthenticated();
  const user = getUserInfo();

  if (!authenticated) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles && allowedRoles.length > 0) {
    if (!user || !allowedRoles.includes(user.role)) {
      return <Navigate to="/unauthorized" replace />;
    }
  }

  return <Outlet />;
};
```

### 5.4 App Router with Protected Routes

```typescript
// src/App.tsx
import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ProtectedRoute } from './components/ProtectedRoute';
import { LoadingSpinner } from './components/LoadingSpinner';
import { MainLayout } from './layouts/MainLayout';

// Lazy load pages
const Login = lazy(() => import('./pages/Login'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const EDCMetrics = lazy(() => import('./pages/EDCMetrics'));
const DataQuality = lazy(() => import('./pages/DataQuality'));
const VisitManagement = lazy(() => import('./pages/VisitManagement'));
const LaboratoryData = lazy(() => import('./pages/LaboratoryData'));
const SafetyMonitoring = lazy(() => import('./pages/SafetyMonitoring'));
const MedDRACoding = lazy(() => import('./pages/MedDRACoding'));
const WHODrugCoding = lazy(() => import('./pages/WHODrugCoding'));
const FormsVerification = lazy(() => import('./pages/FormsVerification'));
const CRAActivity = lazy(() => import('./pages/CRAActivity'));
const Reports = lazy(() => import('./pages/Reports'));
const ThirdPartyData = lazy(() => import('./pages/ThirdPartyData'));

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: true,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Suspense fallback={<LoadingSpinner fullScreen />}>
          <Routes>
            {/* Public Routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route path="/reset-password/:token" element={<ResetPassword />} />

            {/* Protected Routes */}
            <Route element={<ProtectedRoute />}>
              <Route element={<MainLayout />}>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                
                {/* EDC Metrics - All authenticated users */}
                <Route path="/edc-metrics" element={<EDCMetrics />} />
                
                {/* Data Quality - DQT, CRA, Study Manager */}
                <Route
                  path="/data-quality"
                  element={<DataQuality />}
                />
                
                {/* Visit Management */}
                <Route path="/visit-management" element={<VisitManagement />} />
                
                {/* Laboratory */}
                <Route path="/laboratory" element={<LaboratoryData />} />
                
                {/* Safety - Medical Monitor, Safety team, Study Manager */}
                <Route
                  path="/safety"
                  element={<SafetyMonitoring />}
                />
                
                {/* Coding */}
                <Route path="/coding/meddra" element={<MedDRACoding />} />
                <Route path="/coding/whodrug" element={<WHODrugCoding />} />
                
                {/* Forms */}
                <Route path="/forms" element={<FormsVerification />} />
                
                {/* CRA Activity */}
                <Route path="/cra-activity" element={<CRAActivity />} />
                
                {/* Reports */}
                <Route path="/reports" element={<Reports />} />
                
                {/* Third Party Data */}
                <Route path="/third-party" element={<ThirdPartyData />} />
              </Route>
            </Route>

            {/* 404 */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
};

export default App;
```

---

## 6. Data Flow & State Management

### 6.1 React Query Configuration

```typescript
// src/config/queryClient.ts
import { QueryClient } from '@tanstack/react-query';
import { toast } from 'react-toastify';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Data refetching
      refetchOnWindowFocus: true,
      refetchOnMount: true,
      refetchOnReconnect: true,
      
      // Retry configuration
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false;
        }
        // Retry up to 3 times for 5xx errors
        return failureCount < 3;
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      
      // Stale time
      staleTime: 5 * 60 * 1000, // 5 minutes
      
      // Cache time
      cacheTime: 10 * 60 * 1000, // 10 minutes
      
      // Error handling
      onError: (error: any) => {
        const message = error?.response?.data?.message || 'An error occurred';
        toast.error(message);
      },
    },
    mutations: {
      // Error handling for mutations
      onError: (error: any) => {
        const message = error?.response?.data?.message || 'Operation failed';
        toast.error(message);
      },
      
      // Success handling
      onSuccess: () => {
        toast.success('Operation completed successfully');
      },
    },
  },
});
```

### 6.2 Global State for Filters (Optional - Zustand Example)

```typescript
// src/store/filterStore.ts
import create from 'zustand';
import { persist } from 'zustand/middleware';

interface FilterState {
  globalFilters: {
    region?: string[];
    country?: string[];
    siteId?: string[];
    dateFrom?: string;
    dateTo?: string;
  };
  setGlobalFilters: (filters: Partial<FilterState['globalFilters']>) => void;
  clearGlobalFilters: () => void;
}

export const useFilterStore = create<FilterState>()(
  persist(
    (set) => ({
      globalFilters: {},
      setGlobalFilters: (filters) =>
        set((state) => ({
          globalFilters: { ...state.globalFilters, ...filters },
        })),
      clearGlobalFilters: () => set({ globalFilters: {} }),
    }),
    {
      name: 'clinical-trial-filters',
    }
  )
);
```

---

## 7. Error Handling & Retry Logic

### 7.1 Error Boundary Component

```typescript
// src/components/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
    
    // Send to error tracking service (e.g., Sentry)
    // logErrorToService(error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="error-boundary">
            <h1>Something went wrong</h1>
            <p>{this.state.error?.message}</p>
            <button onClick={() => this.setState({ hasError: false })}>
              Try again
            </button>
          </div>
        )
      );
    }

    return this.props.children;
  }
}
```

### 7.2 Error Message Component

```typescript
// src/components/ErrorMessage.tsx
import React from 'react';

interface ErrorMessageProps {
  error: any;
  onRetry?: () => void;
  message?: string;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  error,
  onRetry,
  message,
}) => {
  const errorMessage =
    message ||
    error?.response?.data?.message ||
    error?.message ||
    'An unexpected error occurred';

  return (
    <div className="error-message">
      <div className="error-icon">⚠️</div>
      <h3>Error</h3>
      <p>{errorMessage}</p>
      {onRetry && (
        <button onClick={onRetry} className="retry-button">
          Retry
        </button>
      )}
    </div>
  );
};
```

### 7.3 Global Error Handler

```typescript
// src/utils/errorHandler.ts
import { AxiosError } from 'axios';
import { toast } from 'react-toastify';

export const handleApiError = (error: AxiosError | Error) => {
  if (error instanceof AxiosError) {
    const status = error.response?.status;
    const message = error.response?.data?.message;

    switch (status) {
      case 400:
        toast.error(message || 'Invalid request. Please check your input.');
        break;
      case 401:
        toast.error('Session expired. Please log in again.');
        // Redirect to login handled by interceptor
        break;
      case 403:
        toast.error('You do not have permission to perform this action.');
        break;
      case 404:
        toast.error(message || 'The requested resource was not found.');
        break;
      case 500:
        toast.error('Server error. Please try again later.');
        break;
      case 503:
        toast.error('Service temporarily unavailable. Please try again later.');
        break;
      default:
        toast.error(message || 'An unexpected error occurred.');
    }
  } else {
    toast.error(error.message || 'An unexpected error occurred.');
  }

  // Log to error tracking service
  // logErrorToService(error);
};
```

---

## 8. Environment Configuration

### 8.1 Environment Variables (.env files)

```bash
# .env.development
REACT_APP_API_BASE_URL=http://localhost:8000/api/v1
REACT_APP_WS_URL=ws://localhost:8000/ws
REACT_APP_ENV=development

# .env.staging
REACT_APP_API_BASE_URL=https://staging-api.clinicaltrial-app.example.com/api/v1
REACT_APP_WS_URL=wss://staging-api.clinicaltrial-app.example.com/ws
REACT_APP_ENV=staging

# .env.production
REACT_APP_API_BASE_URL=https://api.clinicaltrial-app.example.com/api/v1
REACT_APP_WS_URL=wss://api.clinicaltrial-app.example.com/ws
REACT_APP_ENV=production
```

### 8.2 Config File

```typescript
// src/config/index.ts
export const config = {
  apiBaseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1',
  wsURL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws',
  environment: process.env.REACT_APP_ENV || 'development',
  isDevelopment: process.env.REACT_APP_ENV === 'development',
  isProduction: process.env.REACT_APP_ENV === 'production',
};
```

---

## 9. Type Safety & Validation

### 9.1 Comprehensive Type Definitions

```typescript
// src/types/api.types.ts

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

export interface SubjectMetric {
  id: string;
  region: string;
  country: string;
  siteId: string;
  siteName: string;
  subjectId: string;
  subjectStatus: string;
  enrollmentDate: string;
  lastVisitDate: string;
  totalVisitsPlanned: number;
  totalVisitsCompleted: number;
  missingVisitsCount: number;
  missingVisitsPercentage: number;
  missingPagesCount: number;
  missingPagesPercentage: number;
  openQueriesTotal: number;
  openQueriesByType: Record<string, number>;
  nonConformantDataCount: number;
  sdvStatusPercentage: number;
  frozenFormsCount: number;
  lockedFormsCount: number;
  signedFormsCount: number;
  overdueCRFsCount: number;
  inactivatedFoldersCount: number;
  isClean: boolean;
  lastUpdated: string;
}

export interface Query {
  id: string;
  subjectId: string;
  siteId: string;
  siteName: string;
  visitName: string;
  formName: string;
  fieldName: string;
  queryType: string;
  queryText: string;
  status: 'open' | 'answered' | 'closed' | 'cancelled';
  priority: 'critical' | 'high' | 'medium' | 'low';
  openedDate: string;
  daysOpen: number;
  assignedTo: string;
  responseDueDate: string;
  lastResponseDate?: string;
}

export interface MissingVisit {
  id: string;
  subjectId: string;
  siteId: string;
  siteName: string;
  visitName: string;
  visitType: string;
  projectedDate: string;
  daysOverdue: number;
  lastContactDate?: string;
  craAssigned: string;
  followUpStatus: 'pending' | 'in_progress' | 'contacted' | 'resolved';
}

export interface SAE {
  id: string;
  subjectId: string;
  siteId: string;
  siteName: string;
  saeDescription: string;
  onsetDate: string;
  reportDate: string;
  severity: 'mild' | 'moderate' | 'severe';
  discrepancyType?: string;
  discrepancyStatus?: string;
  reviewStatus: 'pending' | 'under_review' | 'completed';
  daysOpen: number;
  assignedDataManager?: string;
  assignedSafetyPhysician?: string;
  causalityAssessment?: string;
  expectedness?: string;
  lastUpdateDate: string;
}

export interface DataQualityIndex {
  entityId: string;
  entityType: 'overall' | 'region' | 'country' | 'site' | 'subject';
  entityName: string;
  dqiScore: number;
  parameterScores: {
    missingVisits: number;
    missingPages: number;
    openQueries: number;
    nonConformantData: number;
    unverifiedForms: number;
    uncodedTerms: number;
    unresolvedSAEs: number;
  };
  lastCalculated: string;
}

// Add more type definitions for other entities...
```

### 9.2 Runtime Validation (Zod Example)

```typescript
// src/utils/validation.ts
import { z } from 'zod';

export const SubjectMetricSchema = z.object({
  id: z.string(),
  region: z.string(),
  country: z.string(),
  siteId: z.string(),
  siteName: z.string(),
  subjectId: z.string(),
  subjectStatus: z.string(),
  enrollmentDate: z.string(),
  lastVisitDate: z.string(),
  totalVisitsPlanned: z.number(),
  totalVisitsCompleted: z.number(),
  missingVisitsCount: z.number(),
  missingVisitsPercentage: z.number(),
  missingPagesCount: z.number(),
  missingPagesPercentage: z.number(),
  openQueriesTotal: z.number(),
  openQueriesByType: z.record(z.number()),
  nonConformantDataCount: z.number(),
  sdvStatusPercentage: z.number(),
  frozenFormsCount: z.number(),
  lockedFormsCount: z.number(),
  signedFormsCount: z.number(),
  overdueCRFsCount: z.number(),
  inactivatedFoldersCount: z.number(),
  isClean: z.boolean(),
  lastUpdated: z.string(),
});

export const validateSubjectMetric = (data: unknown) => {
  return SubjectMetricSchema.parse(data);
};
```

---

## 10. Real-time Data Updates

### 10.1 WebSocket Client

```typescript
// src/services/websocket/client.ts
import { WS_CONFIG } from '../../config/api.config';
import { getAuthToken } from '../../utils/auth';

export type WebSocketMessageType =
  | 'query_updated'
  | 'sae_reported'
  | 'visit_completed'
  | 'form_signed'
  | 'alert_triggered';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  payload: any;
  timestamp: string;
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<WebSocketMessageType, Set<(payload: any) => void>> = new Map();

  connect() {
    const token = getAuthToken();
    
    if (!token) {
      console.warn('No auth token available for WebSocket connection');
      return;
    }

    this.ws = new WebSocket(`${WS_CONFIG.wsURL}?token=${token}`);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.attemptReconnect();
    };
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  private handleMessage(message: WebSocketMessage) {
    const listeners = this.listeners.get(message.type);
    if (listeners) {
      listeners.forEach((callback) => callback(message.payload));
    }
  }

  subscribe(type: WebSocketMessageType, callback: (payload: any) => void) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(callback);

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(type);
      if (listeners) {
        listeners.delete(callback);
      }
    };
  }

  send(message: WebSocketMessage) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }
}

export const wsClient = new WebSocketClient();
```

### 10.2 WebSocket Hook

```typescript
// src/hooks/useWebSocket.ts
import { useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { wsClient, WebSocketMessageType } from '../services/websocket/client';

export const useWebSocket = () => {
  const queryClient = useQueryClient();

  useEffect(() => {
    // Connect WebSocket
    wsClient.connect();

    // Subscribe to query updates
    const unsubscribeQueryUpdated = wsClient.subscribe('query_updated', (payload) => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({ queryKey: ['data-quality', 'queries'] });
      queryClient.invalidateQueries({ queryKey: ['data-quality', 'query-metrics'] });
    });

    // Subscribe to SAE reports
    const unsubscribeSAEReported = wsClient.subscribe('sae_reported', (payload) => {
      queryClient.invalidateQueries({ queryKey: ['safety', 'sae-dashboard'] });
      queryClient.invalidateQueries({ queryKey: ['safety', 'sae-metrics'] });
    });

    // Subscribe to visit completions
    const unsubscribeVisitCompleted = wsClient.subscribe('visit_completed', (payload) => {
      queryClient.invalidateQueries({ queryKey: ['visits', 'missing'] });
      queryClient.invalidateQueries({ queryKey: ['edc-metrics', 'subjects'] });
    });

    // Subscribe to form signed
    const unsubscribeFormSigned = wsClient.subscribe('form_signed', (payload) => {
      queryClient.invalidateQueries({ queryKey: ['forms', 'form-status'] });
      queryClient.invalidateQueries({ queryKey: ['edc-metrics', 'subjects'] });
    });

    // Cleanup on unmount
    return () => {
      unsubscribeQueryUpdated();
      unsubscribeSAEReported();
      unsubscribeVisitCompleted();
      unsubscribeFormSigned();
      wsClient.disconnect();
    };
  }, [queryClient]);
};
```

### 10.3 Use WebSocket in App

```typescript
// src/App.tsx
import { useWebSocket } from './hooks/useWebSocket';

const App: React.FC = () => {
  // Initialize WebSocket connection
  useWebSocket();

  return (
    // ... rest of app
  );
};
```

---

## 11. Performance Optimization

### 11.1 Data Pagination

All table components must implement proper pagination:

```typescript
// Example: EDCMetrics.tsx pagination
const [pagination, setPagination] = useState({
  page: 1,
  pageSize: 50,
});

const { data } = useEDCMetrics({
  ...filters,
  page: pagination.page,
  pageSize: pagination.pageSize,
});

// Handle page change
const handlePageChange = (newPage: number) => {
  setPagination((prev) => ({ ...prev, page: newPage }));
};
```

### 11.2 Virtual Scrolling for Large Tables

```typescript
// Use react-virtual or react-window for tables with 500+ rows
import { useVirtualizer } from '@tanstack/react-virtual';

const TableVirtualized: React.FC<{ data: any[] }> = ({ data }) => {
  const parentRef = React.useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtualizer({
    count: data.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
    overscan: 10,
  });

  return (
    <div ref={parentRef} style={{ height: '600px', overflow: 'auto' }}>
      <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, position: 'relative' }}>
        {rowVirtualizer.getVirtualItems().map((virtualRow) => (
          <div
            key={virtualRow.index}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualRow.size}px`,
              transform: `translateY(${virtualRow.start}px)`,
            }}
          >
            {/* Render row data[virtualRow.index] */}
          </div>
        ))}
      </div>
    </div>
  );
};
```

### 11.3 Debouncing Search/Filter Inputs

```typescript
// src/hooks/useDebounce.ts
import { useState, useEffect } from 'react';

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

// Usage in component
const [searchTerm, setSearchTerm] = useState('');
const debouncedSearchTerm = useDebounce(searchTerm, 500);

useEffect(() => {
  // Trigger API call with debouncedSearchTerm
}, [debouncedSearchTerm]);
```

### 11.4 Lazy Loading Images and Charts

```typescript
import { lazy, Suspense } from 'react';

// Lazy load chart components
const ChartComponent = lazy(() => import('./ChartComponent'));

// Use in render
<Suspense fallback={<div>Loading chart...</div>}>
  <ChartComponent data={data} />
</Suspense>
```

---

## 12. Testing Strategy

### 12.1 API Service Tests

```typescript
// src/services/api/__tests__/edcMetrics.service.test.ts
import { edcMetricsService } from '../edcMetrics.service';
import { apiClient } from '../client';

jest.mock('../client');

describe('EDCMetricsService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('getSubjects calls API with correct parameters', async () => {
    const mockResponse = {
      data: [],
      total: 0,
      page: 1,
      pageSize: 50,
      totalPages: 0,
    };

    (apiClient.get as jest.Mock).mockResolvedValue(mockResponse);

    const filters = { page: 1, pageSize: 50 };
    const result = await edcMetricsService.getSubjects(filters);

    expect(apiClient.get).toHaveBeenCalledWith('/edc-metrics/subjects', {
      params: { page: 1, pageSize: 50 },
    });
    expect(result).toEqual(mockResponse);
  });
});
```

### 12.2 Component Integration Tests

```typescript
// src/pages/__tests__/EDCMetrics.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { EDCMetrics } from '../EDCMetrics';
import { edcMetricsService } from '../../services/api';

jest.mock('../../services/api');

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

describe('EDCMetrics Page', () => {
  test('displays data from API', async () => {
    const mockData = {
      data: [
        {
          id: '1',
          subjectId: 'S001',
          siteName: 'Site A',
          // ... other fields
        },
      ],
      total: 1,
      page: 1,
      pageSize: 50,
      totalPages: 1,
    };

    (edcMetricsService.getSubjects as jest.Mock).mockResolvedValue(mockData);

    render(<EDCMetrics />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText('S001')).toBeInTheDocument();
      expect(screen.getByText('Site A')).toBeInTheDocument();
    });
  });

  test('displays error message when API fails', async () => {
    (edcMetricsService.getSubjects as jest.Mock).mockRejectedValue(
      new Error('API Error')
    );

    render(<EDCMetrics />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText(/Failed to load/i)).toBeInTheDocument();
    });
  });
});
```

---

## 13. Deployment Checklist

### 13.1 Backend Deployment Checklist

- [ ] Environment variables configured for production
- [ ] CORS origins updated to production frontend URLs
- [ ] Database connections secured (SSL, credentials)
- [ ] API rate limiting enabled
- [ ] Logging configured (structured logs)
- [ ] Error monitoring setup (Sentry, Datadog, etc.)
- [ ] Health check endpoint implemented (`/health`, `/api/health`)
- [ ] Load balancer configured
- [ ] SSL certificates installed and configured
- [ ] Database migrations run
- [ ] Backup strategy in place
- [ ] API documentation deployed (Swagger/OpenAPI)

### 13.2 Frontend Deployment Checklist

- [ ] Environment variables set for production API URL
- [ ] Build process optimized (`npm run build`)
- [ ] Code splitting configured
- [ ] Static assets optimized (images, fonts)
- [ ] Service worker configured (if using PWA)
- [ ] Analytics configured (Google Analytics, etc.)
- [ ] Error tracking configured (Sentry)
- [ ] CDN configured for static assets
- [ ] SSL certificate configured
- [ ] Meta tags and SEO configured
- [ ] Browser compatibility tested
- [ ] Performance tested (Lighthouse)
- [ ] Security headers configured (CSP, X-Frame-Options, etc.)

### 13.3 Integration Testing Checklist

- [ ] All API endpoints tested with real backend
- [ ] Authentication flow tested end-to-end
- [ ] File upload/download tested
- [ ] Real-time updates (WebSocket) tested
- [ ] Error scenarios tested (network failure, timeout, etc.)
- [ ] Cross-browser testing completed
- [ ] Mobile responsiveness tested
- [ ] Load testing completed
- [ ] Security testing completed (penetration testing)

---

## 14. Mock Data Removal Strategy

### 14.1 Identify and Remove Mock Data

**Step 1: Search for Mock Data Patterns**

```bash
# Search for common mock data patterns
grep -r "mockData" src/
grep -r "MOCK_" src/
grep -r "dummyData" src/
grep -r "sampleData" src/
grep -r "TODO: Replace with API" src/
```

**Step 2: Remove Mock Data Files**

```bash
# Remove any mock data files
rm -rf src/mocks/
rm -rf src/__mocks__/
rm src/data/mockData.ts
```

**Step 3: Update Components to Only Use API Data**

Before (with mock data):
```typescript
const EDCMetrics = () => {
  const [data, setData] = useState(MOCK_EDC_METRICS); // ❌ Remove this
  
  // ... component code
};
```

After (only API data):
```typescript
const EDCMetrics = () => {
  const { data, isLoading, isError, error, refetch } = useEDCMetrics(filters);
  
  if (isLoading) return <LoadingSpinner />;
  if (isError) return <ErrorMessage error={error} onRetry={refetch} />;
  if (!data) return <div>No data available</div>;
  
  // ... component code using data from API
};
```

**Step 4: Remove Conditional Mock/Real Data Logic**

Before:
```typescript
const fetchData = async () => {
  if (process.env.NODE_ENV === 'development') {
    return MOCK_DATA; // ❌ Remove this
  }
  return await apiClient.get('/endpoint');
};
```

After:
```typescript
const fetchData = async () => {
  return await apiClient.get('/endpoint'); // ✅ Only real API
};
```

### 14.2 Verification Steps

1. **Build the Application**
   ```bash
   npm run build
   ```
   Ensure no references to mock data remain.

2. **Start with Production API**
   ```bash
   REACT_APP_API_BASE_URL=https://api.example.com npm start
   ```

3. **Test All Pages**
   - Navigate to each page
   - Verify data loads from API
   - Verify loading states work
   - Verify error states work

4. **Check Network Tab**
   - Open DevTools > Network
   - Verify all requests go to real API
   - Verify no 404s or failed requests
   - Verify proper authentication headers

5. **Test Error Scenarios**
   - Disconnect network
   - Verify error messages display
   - Reconnect and verify retry works

---

## 15. Final Integration Verification

### 15.1 End-to-End Integration Test Script

```typescript
// integration-tests/e2e.test.ts
describe('Clinical Trial App E2E', () => {
  beforeAll(async () => {
    // Login
    await page.goto('http://localhost:3000/login');
    await page.type('input[name="email"]', 'test@example.com');
    await page.type('input[name="password"]', 'password');
    await page.click('button[type="submit"]');
    await page.waitForNavigation();
  });

  test('Dashboard loads with real data', async () => {
    await page.goto('http://localhost:3000/dashboard');
    
    // Wait for API call to complete
    await page.waitForSelector('[data-testid="dashboard-summary"]');
    
    // Verify no mock data placeholders
    const content = await page.content();
    expect(content).not.toContain('MOCK');
    expect(content).not.toContain('Sample Data');
    expect(content).not.toContain('Lorem ipsum');
  });

  test('EDC Metrics loads and displays subjects', async () => {
    await page.goto('http://localhost:3000/edc-metrics');
    
    // Wait for table to load
    await page.waitForSelector('table tbody tr');
    
    // Verify data is present
    const rows = await page.$$('table tbody tr');
    expect(rows.length).toBeGreaterThan(0);
  });

  // Add more E2E tests for other pages...
});
```

---

## 16. Summary: Key Implementation Requirements

### ✅ Mandatory Requirements

1. **NO Mock Data**: All components must fetch data exclusively from backend APIs
2. **CORS Properly Configured**: Backend must allow frontend origin with credentials
3. **Authentication**: JWT token-based auth with automatic refresh
4. **Error Handling**: Global error handling with user-friendly messages and retry logic
5. **Loading States**: Show loading indicators while data is being fetched
6. **Type Safety**: All API responses must be typed with TypeScript interfaces
7. **Real-time Updates**: WebSocket integration for live data updates
8. **Pagination**: Implement server-side pagination for all large datasets
9. **Environment Config**: Separate configs for dev/staging/production
10. **Testing**: Integration tests to verify API connectivity

### 🔧 Technical Stack

- **Frontend**: React 18+ with TypeScript
- **State Management**: React Query (TanStack Query)
- **HTTP Client**: Axios with interceptors
- **WebSocket**: Native WebSocket API with reconnection logic
- **Routing**: React Router v6
- **Forms**: React Hook Form with validation
- **UI Components**: Reusable components library

### 📝 Next Steps

1. Set up backend API endpoints as specified
2. Configure CORS on backend
3. Implement authentication service
4. Create API service layer in frontend
5. Replace all mock data with API hooks
6. Test integration end-to-end
7. Deploy to staging environment
8. Conduct UAT (User Acceptance Testing)
9. Deploy to production

---

**This guide ensures proper backend-frontend integration with NO mock data and production-ready architecture.**
