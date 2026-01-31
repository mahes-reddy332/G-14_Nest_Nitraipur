
/*
    Feature Model: Query Aging (Site Level)
    Logic: Weighted average of open query duration.
*/

{{ config(materialized='table') }}

WITH query_data AS (
    SELECT 
        study_id,
        site_id,
        -- Calculate days open
        DATEDIFF('day', created_date, COALESCE(resolved_date, CURRENT_DATE)) as days_open,
        status,
        priority
    FROM {{ ref('stg_clinical_data') }}
    WHERE domain = 'QUERY' AND status = 'Open'
),

weighted_data AS (
    SELECT
        study_id,
        site_id,
        days_open,
        CASE 
            WHEN priority = 'High' THEN 3
            WHEN priority = 'Medium' THEN 2
            ELSE 1
        END as weight
    FROM query_data
),

site_agg AS (
    SELECT
        study_id,
        site_id,
        COUNT(*) as open_query_count,
        SUM(days_open * weight) as weighted_sum_days,
        SUM(weight) as total_weight
    FROM weighted_data
    GROUP BY study_id, site_id
)

SELECT
    study_id,
    site_id,
    open_query_count,
    CASE 
        WHEN open_query_count = 0 THEN 0
        ELSE ROUND(weighted_sum_days::FLOAT / total_weight, 2)
    END as query_aging_index,
    CURRENT_TIMESTAMP as metrics_calculated_at
FROM site_agg
