"""
Feature Analysis Utility Functions

This module provides convenient functions for feature analysis and selection
in the Mental Health Analyzer project.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.components.feature_selection import FeatureSelector, FeatureSelectionComponent
from src.logger import logging


def quick_feature_selection(data_path: str, target_column: str = 'IsCounselingNeeded', 
                           method: str = 'ensemble', k: int = 8) -> Dict:
    """
    Perform quick feature selection on a dataset
    
    Args:
        data_path: Path to the CSV dataset
        target_column: Name of target column
        method: Feature selection method ('ensemble', 'statistical', 'model_based', etc.)
        k: Number of features to select
        
    Returns:
        Dictionary with selected features and performance results
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Perform feature selection
    selector = FeatureSelector()
    X_selected, selected_features = selector.fit_transform(X, y, method=method, k=k)
    
    # Get evaluation results
    evaluation_results = selector.evaluate_feature_sets(X, y)
    
    return {
        'selected_features': selected_features,
        'original_shape': X.shape,
        'selected_shape': X_selected.shape,
        'evaluation_results': evaluation_results,
        'feature_scores': selector.feature_scores_,
        'correlation_pairs': selector.high_corr_pairs_
    }


def compare_feature_methods(data_path: str, target_column: str = 'IsCounselingNeeded', 
                           k: int = 8) -> pd.DataFrame:
    """
    Compare different feature selection methods
    
    Args:
        data_path: Path to the CSV dataset
        target_column: Name of target column
        k: Number of features to select
        
    Returns:
        DataFrame comparing different methods
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Test different methods
    methods = ['statistical', 'model_based', 'wrapper', 'ensemble']
    results = []
    
    for method in methods:
        try:
            logging.info(f"Testing method: {method}")
            selector = FeatureSelector()
            X_selected, selected_features = selector.fit_transform(X, y, method=method, k=k)
            
            # Simple evaluation
            evaluation_results = selector.evaluate_feature_sets(X, y)
            
            if method in evaluation_results:
                method_result = evaluation_results[method]
                results.append({
                    'Method': method,
                    'Num_Features': len(selected_features),
                    'F1_Score': method_result['mean_f1'],
                    'F1_Std': method_result['std_f1'],
                    'Accuracy': method_result['mean_accuracy'],
                    'Accuracy_Std': method_result['std_accuracy'],
                    'Selected_Features': ', '.join(selected_features)
                })
        except Exception as e:
            logging.warning(f"Error with method {method}: {e}")
            results.append({
                'Method': method,
                'Num_Features': 0,
                'F1_Score': 0,
                'F1_Std': 0,
                'Accuracy': 0,
                'Accuracy_Std': 0,
                'Selected_Features': 'Error'
            })
    
    return pd.DataFrame(results)


def analyze_feature_importance(data_path: str, target_column: str = 'IsCounselingNeeded') -> Dict:
    """
    Analyze feature importance using multiple methods
    
    Args:
        data_path: Path to the CSV dataset
        target_column: Name of target column
        
    Returns:
        Dictionary with feature importance analysis
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    selector = FeatureSelector()
    
    # Run all methods to get feature scores
    selector.statistical_feature_selection(X, y, method='all', k=len(X.columns))
    selector.model_based_selection(X, y, method='all', k=len(X.columns))
    
    # Combine scores
    importance_summary = {}
    for feature in X.columns:
        scores = []
        methods_used = []
        
        for method_name, feature_scores in selector.feature_scores_.items():
            if feature in feature_scores:
                scores.append(feature_scores[feature])
                methods_used.append(method_name)
        
        if scores:
            importance_summary[feature] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores) if len(scores) > 1 else 0,
                'methods': methods_used,
                'num_methods': len(methods_used)
            }
    
    # Sort by mean score
    sorted_features = sorted(importance_summary.items(), 
                           key=lambda x: x[1]['mean_score'], reverse=True)
    
    return {
        'feature_importance': dict(sorted_features),
        'top_features': [f[0] for f in sorted_features[:8]],
        'feature_scores_by_method': selector.feature_scores_
    }


def validate_selected_features(train_path: str, test_path: str, 
                              selected_features: List[str],
                              target_column: str = 'IsCounselingNeeded') -> Dict:
    """
    Validate selected features on train/test split
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV  
        selected_features: List of feature names to validate
        target_column: Name of target column
        
    Returns:
        Dictionary with validation results
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Prepare data with selected features
    X_train = train_df[selected_features]
    y_train = train_df[target_column]
    X_test = test_df[selected_features]
    y_test = test_df[target_column]
    
    # Encode categorical features
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    for col in categorical_features:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col])
        # Handle unknown categories in test set
        X_test_encoded[col] = X_test_encoded[col].map(
            dict(zip(le.classes_, le.transform(le.classes_)))
        ).fillna(-1)
    
    # Test different models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }
    
    results = {}
    for model_name, model in models.items():
        # Train model
        model.fit(X_train_encoded, y_train)
        
        # Predict
        y_pred = model.predict(X_test_encoded)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[model_name] = {
            'f1_score': f1,
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        logging.info(f"{model_name} - F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
    
    return {
        'selected_features': selected_features,
        'num_features': len(selected_features),
        'model_results': results,
        'data_shapes': {
            'train': X_train.shape,
            'test': X_test.shape
        }
    }


def get_feature_recommendations(data_path: str, target_column: str = 'IsCounselingNeeded') -> Dict:
    """
    Get feature selection recommendations based on comprehensive analysis
    
    Args:
        data_path: Path to the CSV dataset
        target_column: Name of target column
        
    Returns:
        Dictionary with recommendations
    """
    # Perform comprehensive analysis
    quick_results = quick_feature_selection(data_path, target_column)
    importance_analysis = analyze_feature_importance(data_path, target_column)
    
    recommendations = {
        'recommended_features': quick_results['selected_features'],
        'feature_reduction': f"Reduced from {quick_results['original_shape'][1]} to {len(quick_results['selected_features'])} features",
        'top_important_features': importance_analysis['top_features'][:5],
        'correlated_features_to_review': [
            f"{pair['feature1']} <-> {pair['feature2']} (r={pair['correlation']:.3f})"
            for pair in quick_results['correlation_pairs']
        ],
        'best_method': max(quick_results['evaluation_results'].items(), 
                          key=lambda x: x[1]['mean_f1'] if 'mean_f1' in x[1] else 0)[0],
        'performance_summary': {
            method: f"F1: {results['mean_f1']:.3f}±{results['std_f1']:.3f}"
            for method, results in quick_results['evaluation_results'].items()
            if 'mean_f1' in results
        }
    }
    
    return recommendations


if __name__ == "__main__":
    # Example usage
    try:
        data_path = "artifacts/data.csv"
        
        print("=== Quick Feature Selection ===")
        results = quick_feature_selection(data_path)
        print(f"Selected {len(results['selected_features'])} features:")
        for i, feature in enumerate(results['selected_features'], 1):
            print(f"{i:2d}. {feature}")
        
        print("\n=== Feature Recommendations ===")
        recommendations = get_feature_recommendations(data_path)
        print(f"Recommended features: {recommendations['recommended_features']}")
        print(f"Best method: {recommendations['best_method']}")
        
    except Exception as e:
        print(f"Error in example usage: {e}")
