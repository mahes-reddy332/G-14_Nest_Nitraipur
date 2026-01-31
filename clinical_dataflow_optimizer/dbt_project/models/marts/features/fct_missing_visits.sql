
/*
    Feature Model: Missing Visits (Subject Level)
    Logic: Calculates the ratio of missed vs expected visits per subject.
*/

{{ config(materialized='table') }}

WITH visit_data AS (
    SELECT 
        study_id,
        site_id,
        subject_id,
        -- In real data, we would have a flag or status. 
        -- Here we assume 'status' column exists from staging
        CASE WHEN status = 'Missed' THEN 1 ELSE 0 END AS is_missed,
        1 AS is_expected
    FROM {{ ref('stg_clinical_data') }}
    WHERE domain = 'VISIT' -- Assuming domain logic in staging captures visits
),

aggregated AS (
    SELECT
        study_id,
        site_id,
        subject_id,
        SUM(is_missed) as missed_count,
        SUM(is_expected) as expected_count
    FROM visit_data
    GROUP BY study_id, site_id, subject_id
)

SELECT
    study_id,
    site_id,
    subject_id,
    missed_count,
    expected_count,
    CASE 
        WHEN expected_count = 0 THEN 0
        ELSE ROUND(missed_count::FLOAT / expected_count, 4) 
    END AS missing_visit_ratio,
    CURRENT_TIMESTAMP as metrics_calculated_at
FROM aggregated
