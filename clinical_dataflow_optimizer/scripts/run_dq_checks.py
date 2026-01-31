
import great_expectations as gx
import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataQuality")

def run_dq_checkpoint(df: pd.DataFrame, suite_name: str = "clinical_data_suite"):
    """
    Run Great Expectations validation on a pandas DataFrame.
    """
    context = gx.get_context()

    # Define a Validator
    validator = context.sources.pandas_default.read_dataframe(df)
    
    # create expectation suite
    # in a real scenario, this would be loaded from a JSON file
    validator.expect_column_values_to_not_be_null(column="subject_id")
    validator.expect_column_values_to_not_be_null(column="study_id")
    
    # Conformance check: Study ID should follow pattern (e.g., alphanumeric)
    validator.expect_column_values_to_match_regex(column="study_id", regex=r"^[A-Za-z0-9_]+$")
    
    # Save the suite
    validator.save_expectation_suite(discard_failed_expectations=False)
    
    # Run Validation
    checkpoint = context.add_or_update_checkpoint(
        name="daily_clinical_checkpoint",
        validator=validator,
    )
    
    results = checkpoint.run()
    
    # Parse Result
    success = results["success"]
    logger.info(f"DQ Checkpoint '{suite_name}' finished. Success: {success}")
    
    if not success:
        logger.warning("Data Quality issues found!")
        # In a real pipeline, we might raise an error here to stop the DAG
        # raise ValueError("Data Quality Checks Failed")
        
    return success

if __name__ == "__main__":
    # Mock data for demonstration
    data = {
        "subject_id": ["SUBJ-001", "SUBJ-002", None],
        "study_id": ["STUDY_A", "STUDY_A", "STUDY_B"],
        "visit_date": ["2023-01-01", "2023-01-02", "2023-01-03"]
    }
    df = pd.DataFrame(data)
    
    logger.info("Running DQ checks on sample data...")
    success = run_dq_checkpoint(df)
    logger.info(f"Final Result: {'PASS' if success else 'FAIL'}")
