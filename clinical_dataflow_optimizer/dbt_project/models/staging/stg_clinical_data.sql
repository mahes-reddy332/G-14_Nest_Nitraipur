
/*
    Staging Model: Clinical Data
    - Standardizes column names
    - Generates Canonical IDs for lineage
*/

{{ config(materialized='view') }}

WITH source_data AS (
    -- In a real dbt project, this would reference a source defined in src_clinical.yml
    -- For now, we assume a raw table 'raw_clinical_data' exists
    SELECT * FROM raw_clinical_data
),

renamed AS (
    SELECT
        -- Standardize to snake_case
        "Subject ID" AS subject_id,
        "Study ID" AS study_id,
        "Site ID" AS site_id,
        "Visit ID" AS visit_id,
        "Visit Date" AS visit_date,
        
        -- Domain specific columns
        CASE 
            WHEN "Lab Result" IS NOT NULL THEN 'LAB'
            WHEN "AE Term" IS NOT NULL THEN 'AE'
            ELSE 'EDC'
        END AS domain,
        
        -- Generate Canonical ID: Hash of Study + Subject
        -- This enables cross-system linkage even if IDs drift
        md5(COALESCE("Study ID", '') || '-' || COALESCE("Subject ID", '')) AS canonical_subject_id

    FROM source_data
)

SELECT * FROM renamed
