from src.logger import logging
import pandas as pd
from src.logger import logging

def get_features(dataset: pd.DataFrame, target_column_name: str):
    """
    Automatically detect numerical and categorical columns 
    from the given DataFrame, excluding the target column.
    """
    # detect columns
    numerical_columns = dataset.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = dataset.select_dtypes(include=["object"]).columns.tolist()

    # drop target column if present
    if target_column_name in numerical_columns:
        numerical_columns.remove(target_column_name)
    if target_column_name in categorical_columns:
        categorical_columns.remove(target_column_name)

    logging.info("seperating the numerical and categorical columns")


    return numerical_columns, categorical_columns

def create_target_column(dataset: pd.DataFrame):
    """
    Create the target column IsCounselingNeeded based on the given conditions.
    """
    # Create target column IsCounselingNeeded before splitting
    try:
        dataset['IsCounselingNeeded'] = (
            (dataset['Depression'] == 1) |
            (dataset['Anxiety'] == 1) |
            (dataset['PanicAttack'] == 1) |
            (dataset['SpecialistTreatment'] == 1) |
            (dataset['SymptomFrequency_Last7Days'] >= 3) |
            ((dataset['HasMentalHealthSupport'] == 0) &
             ((dataset['SleepQuality'] <= 2) | (dataset['StudyStressLevel'] >= 3))) |
            (dataset['CGPA'] <= 2.5) |
            (dataset['AcademicEngagement'] <= 2)
        ).astype(int)
        logging.info('Created target column IsCounselingNeeded')
    except Exception as e:
        logging.info(f"Skipping IsCounselingNeeded creation due to error: {e}")

    return dataset
