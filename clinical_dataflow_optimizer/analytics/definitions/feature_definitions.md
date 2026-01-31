# Feature Definitions - Clinical Analytics

## Overview
This document defines the derived metrics and features used for Risk Scoring and Operational Insights (Phase 2).

## 1. Operational Metrics

### 1.1 Missing Visit Ratio (Subject Level)
- **Definition**: The percentage of expected visits that have not been completed/entered.
- **Logic**: `(Count(Expected Visits) - Count(Completed Visits)) / Count(Expected Visits)`
- **Granularity**: Subject
- **Risk Threshold**: > 10%

### 1.2 Query Aging Index (Site Level)
- **Definition**: Weighted average of days queries have been open.
- **Logic**: `Sum(Days Open * Priority Weight) / Count(Open Queries)`
    - Priority Weights: High=3, Medium=2, Low=1
- **Granularity**: Site

### 1.3 SAE Reporting Lag (Site Level)
- **Definition**: Average time between `Event Date` and `Reported Date`.
- **Logic**: `Avg(Reported Date - Event Date)` in days.
- **Risk Threshold**: > 2 days

## 2. Quality Metrics

### 2.1 Coding Completeness
- **Definition**: Percentage of verbatim terms successfully coded (MedDRA/WHO).
- **Logic**: `Count(Coded Terms) / Count(Total Verbatim Terms)`

### 2.2 SDV Coverage
- **Definition**: Percentage of verifiable data points that have been Source Data Verified.
- **Logic**: `Count(SDV Verified Fields) / Count(SDV Required Fields)`
