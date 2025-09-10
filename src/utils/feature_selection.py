"""
Feature Selection Utility Functions

This module provides convenient wrapper functions for feature selection
to be used in the data ingestion pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import sys

from src.exception import CustomeException
from src.logger import logging
from src.components.feature_selection import FeatureSelectionComponent


def select_important_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                            target_column: str = 'IsCounselingNeeded',
                            method: str = 'ensemble', k: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Select important features using comprehensive feature selection methods
    
    This function integrates with the main feature selection component to provide
    a streamlined interface for the data ingestion pipeline.
    
    Args:
        train_df: Training dataframe with features and target
        test_df: Test dataframe with features and target  
        target_column: Name of the target column (default: 'IsCounselingNeeded')
        method: Feature selection method to use
                Options: 'ensemble', 'statistical', 'model_based', 'wrapper', 'correlation'
                (default: 'ensemble')
        k: Number of top features to select (default: 8)
        
    Returns:
        Tuple containing:
        - train_df_selected: Training dataframe with selected features and target
        - test_df_selected: Test dataframe with selected features and target
        - selected_features: List of selected feature names
    """
    try:
        logging.info(f"Starting feature selection with method: {method}, k: {k}")
        
        # Validate input dataframes
        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in test data")
            
        # Initialize feature selection component
        feature_selection_component = FeatureSelectionComponent()
        
        # Perform feature selection
        X_train_selected, X_test_selected, selected_features = feature_selection_component.initiate_feature_selection(
            train_df=train_df,
            test_df=test_df,
            target_column=target_column,
            method=method,
            k=k
        )
        
        # Combine selected features with target column for output
        train_df_selected = X_train_selected.copy()
        train_df_selected[target_column] = train_df[target_column]
        
        test_df_selected = X_test_selected.copy()
        test_df_selected[target_column] = test_df[target_column]
        
        logging.info(f"Feature selection completed successfully. Selected {len(selected_features)} features:")
        logging.info(f"Selected features: {', '.join(selected_features)}")
        
        return train_df_selected, test_df_selected, selected_features
        
    except Exception as e:
        raise CustomeException(e, sys)


def get_recommended_features_count(n_total_features: int) -> int:
    """
    Get recommended number of features to select based on total features
    
    Args:
        n_total_features: Total number of input features
        
    Returns:
        Recommended number of features to select
    """
    if n_total_features <= 5:
        return n_total_features
    elif n_total_features <= 10:
        return max(3, n_total_features - 2)
    elif n_total_features <= 20:
        return max(5, n_total_features // 2)
    else:
        return max(8, min(15, n_total_features // 3))


def quick_feature_analysis(df: pd.DataFrame, target_column: str = 'IsCounselingNeeded') -> dict:
    """
    Perform quick feature analysis to understand the dataset
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        
    Returns:
        Dictionary with basic feature analysis results
    """
    try:
        logging.info("Performing quick feature analysis")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Basic statistics
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Feature types
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Target distribution
        target_distribution = y.value_counts().to_dict()
        
        # Missing values
        missing_values = X.isnull().sum()
        features_with_missing = missing_values[missing_values > 0].to_dict()
        
        analysis = {
            'dataset_shape': (n_samples, n_features),
            'n_numerical_features': len(numerical_features),
            'n_categorical_features': len(categorical_features),
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'target_distribution': target_distribution,
            'features_with_missing': features_with_missing,
            'recommended_k': get_recommended_features_count(n_features)
        }
        
        logging.info(f"Feature analysis completed. Dataset: {analysis['dataset_shape']}, "
                    f"Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}")
        
        return analysis
        
    except Exception as e:
        logging.warning(f"Error in feature analysis: {str(e)}")
        return {}


def validate_feature_selection_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                    target_column: str, k: int) -> bool:
    """
    Validate inputs for feature selection
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        target_column: Target column name
        k: Number of features to select
        
    Returns:
        True if inputs are valid, raises exception otherwise
    """
    try:
        # Check if dataframes are not empty
        if train_df.empty or test_df.empty:
            raise ValueError("Input dataframes cannot be empty")
            
        # Check target column exists
        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in test data")
            
        # Check number of features
        train_features = train_df.drop(columns=[target_column]).shape[1]
        test_features = test_df.drop(columns=[target_column]).shape[1]
        
        if train_features != test_features:
            raise ValueError(f"Feature count mismatch: train={train_features}, test={test_features}")
            
        if k > train_features:
            logging.warning(f"k ({k}) is greater than available features ({train_features}). Using k={train_features}")
            k = train_features
            
        # Check if features are the same
        train_feature_names = set(train_df.drop(columns=[target_column]).columns)
        test_feature_names = set(test_df.drop(columns=[target_column]).columns)
        
        if train_feature_names != test_feature_names:
            missing_in_test = train_feature_names - test_feature_names
            missing_in_train = test_feature_names - train_feature_names
            
            error_msg = "Feature mismatch between train and test sets."
            if missing_in_test:
                error_msg += f" Missing in test: {list(missing_in_test)}"
            if missing_in_train:
                error_msg += f" Missing in train: {list(missing_in_train)}"
                
            raise ValueError(error_msg)
            
        logging.info(f"Feature selection validation passed. Features: {train_features}, k: {k}")
        return True
        
    except Exception as e:
        raise CustomeException(e, sys)
