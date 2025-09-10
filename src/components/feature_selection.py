import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE, SelectFromModel, SequentialFeatureSelector
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

from src.exception import CustomeException
from src.logger import logging
from src.utils.filesManager import save_object


@dataclass
class FeatureSelectionConfig:
    feature_selection_obj_file_path: str = os.path.join('artifacts', "feature_selector.pkl")
    feature_analysis_report_path: str = os.path.join('artifacts', "feature_analysis_report.txt")
    feature_importance_plot_path: str = os.path.join('artifacts', "feature_importance_plots")


class FeatureSelector:
    """
    Comprehensive Feature Selection Class
    
    This class provides multiple feature selection techniques:
    1. Statistical Tests (Chi-square, F-test, Mutual Information)
    2. Correlation Analysis
    3. Model-based Selection (LASSO, Tree-based)
    4. Wrapper Methods (RFE, Sequential Selection)
    5. Ensemble Selection (combines multiple methods)
    """
    
    def __init__(self):
        self.config = FeatureSelectionConfig()
        self.feature_scores_ = {}
        self.selected_features_ = {}
        self.feature_rankings_ = {}
        self.correlation_matrix_ = None
        self.high_corr_pairs_ = []
        self.feature_importance_df_ = None
        
        # Create plots directory if it doesn't exist
        os.makedirs(self.config.feature_importance_plot_path, exist_ok=True)
        
    def statistical_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                    method: str = 'all', k: int = 10) -> Dict[str, List[str]]:
        """
        Perform statistical feature selection using various tests
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'chi2', 'f_classif', 'mutual_info', or 'all'
            k: Number of top features to select
            
        Returns:
            Dictionary with selected features for each method
        """
        try:
            logging.info(f"Starting statistical feature selection with method: {method}")
            
            results = {}
            feature_names = list(X.columns)
            
            # Separate categorical and numerical features
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Encode categorical features for statistical tests
            X_encoded = X.copy()
            le_dict = {}
            for col in categorical_features:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
                le_dict[col] = le
            
            if method in ['chi2', 'all']:
                # Chi-square test (for categorical/discrete features)
                logging.info("Performing Chi-square test")
                selector_chi2 = SelectKBest(score_func=chi2, k=k)
                
                # Ensure all values are non-negative for chi2
                X_chi2 = X_encoded.copy()
                X_chi2 = X_chi2 - X_chi2.min() + 0.01  # Make all values positive
                
                X_selected_chi2 = selector_chi2.fit_transform(X_chi2, y)
                chi2_scores = selector_chi2.scores_
                chi2_selected = [feature_names[i] for i in selector_chi2.get_support(indices=True)]
                
                results['chi2'] = chi2_selected
                self.feature_scores_['chi2'] = dict(zip(feature_names, chi2_scores))
                logging.info(f"Chi2 selected features: {chi2_selected}")
            
            if method in ['f_classif', 'all']:
                # F-test (for numerical features)
                logging.info("Performing F-test (ANOVA)")
                selector_f = SelectKBest(score_func=f_classif, k=k)
                X_selected_f = selector_f.fit_transform(X_encoded, y)
                f_scores = selector_f.scores_
                f_selected = [feature_names[i] for i in selector_f.get_support(indices=True)]
                
                results['f_classif'] = f_selected
                self.feature_scores_['f_classif'] = dict(zip(feature_names, f_scores))
                logging.info(f"F-test selected features: {f_selected}")
            
            if method in ['mutual_info', 'all']:
                # Mutual Information
                logging.info("Performing Mutual Information analysis")
                selector_mi = SelectKBest(score_func=mutual_info_classif, k=k)
                X_selected_mi = selector_mi.fit_transform(X_encoded, y)
                mi_scores = selector_mi.scores_
                mi_selected = [feature_names[i] for i in selector_mi.get_support(indices=True)]
                
                results['mutual_info'] = mi_selected
                self.feature_scores_['mutual_info'] = dict(zip(feature_names, mi_scores))
                logging.info(f"Mutual Info selected features: {mi_selected}")
            
            self.selected_features_.update(results)
            logging.info("Statistical feature selection completed")
            
            return results
            
        except Exception as e:
            raise CustomeException(e, sys)
    
    def correlation_analysis(self, X: pd.DataFrame, y: pd.Series, 
                           threshold: float = 0.8, target_corr_threshold: float = 0.1) -> Dict[str, any]:
        """
        Analyze correlations between features and with target
        
        Args:
            X: Feature matrix
            y: Target variable
            threshold: Correlation threshold for identifying highly correlated features
            target_corr_threshold: Minimum correlation with target to keep feature
            
        Returns:
            Dictionary with correlation analysis results
        """
        try:
            logging.info("Starting correlation analysis")
            
            # Encode categorical features
            X_encoded = X.copy()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_features:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
            
            # Calculate correlation matrix
            data_with_target = X_encoded.copy()
            data_with_target['target'] = y
            self.correlation_matrix_ = data_with_target.corr()
            
            # Find highly correlated feature pairs
            corr_matrix = X_encoded.corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            self.high_corr_pairs_ = high_corr_pairs
            
            # Calculate correlation with target
            target_correlations = self.correlation_matrix_['target'].drop('target').sort_values(key=abs, ascending=False)
            
            # Select features based on target correlation
            corr_selected_features = target_correlations[abs(target_correlations) >= target_corr_threshold].index.tolist()
            
            results = {
                'high_corr_pairs': high_corr_pairs,
                'target_correlations': target_correlations.to_dict(),
                'corr_selected_features': corr_selected_features,
                'correlation_matrix': self.correlation_matrix_
            }
            
            self.selected_features_['correlation'] = corr_selected_features
            logging.info(f"Correlation analysis completed. Found {len(high_corr_pairs)} highly correlated pairs")
            logging.info(f"Selected {len(corr_selected_features)} features based on target correlation")
            
            return results
            
        except Exception as e:
            raise CustomeException(e, sys)
    
    def model_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                            method: str = 'all', k: int = 10) -> Dict[str, List[str]]:
        """
        Model-based feature selection using LASSO, Random Forest, etc.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'lasso', 'random_forest', or 'all'
            k: Number of top features to select
            
        Returns:
            Dictionary with selected features for each method
        """
        try:
            logging.info(f"Starting model-based feature selection with method: {method}")
            
            results = {}
            
            # Encode categorical features
            X_encoded = X.copy()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_features:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
            
            if method in ['lasso', 'all']:
                # LASSO Regularization
                logging.info("Performing LASSO feature selection")
                lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
                lasso.fit(X_encoded, y)
                
                # Select features with non-zero coefficients
                lasso_selected = X_encoded.columns[lasso.coef_ != 0].tolist()
                
                # If no features selected or too few, select top k by coefficient magnitude
                if len(lasso_selected) < k:
                    coef_abs = np.abs(lasso.coef_)
                    top_k_indices = np.argsort(coef_abs)[-k:]
                    lasso_selected = X_encoded.columns[top_k_indices].tolist()
                
                results['lasso'] = lasso_selected
                self.feature_scores_['lasso'] = dict(zip(X_encoded.columns, np.abs(lasso.coef_)))
                logging.info(f"LASSO selected features: {lasso_selected}")
            
            if method in ['random_forest', 'all']:
                # Random Forest Feature Importance
                logging.info("Performing Random Forest feature selection")
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_encoded, y)
                
                # Get feature importances
                feature_importance = rf.feature_importances_
                
                # Select top k features
                top_k_indices = np.argsort(feature_importance)[-k:]
                rf_selected = X_encoded.columns[top_k_indices].tolist()
                
                results['random_forest'] = rf_selected
                self.feature_scores_['random_forest'] = dict(zip(X_encoded.columns, feature_importance))
                logging.info(f"Random Forest selected features: {rf_selected}")
            
            self.selected_features_.update(results)
            logging.info("Model-based feature selection completed")
            
            return results
            
        except Exception as e:
            raise CustomeException(e, sys)
    
    def wrapper_methods(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'rfe', k: int = 10) -> Dict[str, List[str]]:
        """
        Wrapper methods for feature selection
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'rfe', 'forward', 'backward', or 'all'
            k: Number of features to select
            
        Returns:
            Dictionary with selected features for each method
        """
        try:
            logging.info(f"Starting wrapper methods with method: {method}")
            
            results = {}
            
            # Encode categorical features
            X_encoded = X.copy()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_features:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
            
            # Base estimator for wrapper methods
            base_estimator = LogisticRegression(max_iter=1000, random_state=42)
            
            if method in ['rfe', 'all']:
                # Recursive Feature Elimination
                logging.info("Performing Recursive Feature Elimination (RFE)")
                rfe = RFE(estimator=base_estimator, n_features_to_select=k)
                rfe.fit(X_encoded, y)
                
                rfe_selected = X_encoded.columns[rfe.support_].tolist()
                results['rfe'] = rfe_selected
                self.feature_rankings_['rfe'] = dict(zip(X_encoded.columns, rfe.ranking_))
                logging.info(f"RFE selected features: {rfe_selected}")
            
            if method in ['forward', 'all'] and k <= 10:  # Limit forward selection for computational efficiency
                # Forward Selection
                logging.info("Performing Forward Selection")
                sfs_forward = SequentialFeatureSelector(
                    base_estimator, n_features_to_select=k, direction='forward',
                    cv=3, scoring='f1', n_jobs=-1
                )
                sfs_forward.fit(X_encoded, y)
                
                forward_selected = X_encoded.columns[sfs_forward.get_support()].tolist()
                results['forward'] = forward_selected
                logging.info(f"Forward selection selected features: {forward_selected}")
            
            if method in ['backward', 'all'] and k <= 10:  # Limit backward elimination for computational efficiency
                # Backward Elimination
                logging.info("Performing Backward Elimination")
                sfs_backward = SequentialFeatureSelector(
                    base_estimator, n_features_to_select=k, direction='backward',
                    cv=3, scoring='f1', n_jobs=-1
                )
                sfs_backward.fit(X_encoded, y)
                
                backward_selected = X_encoded.columns[sfs_backward.get_support()].tolist()
                results['backward'] = backward_selected
                logging.info(f"Backward elimination selected features: {backward_selected}")
            
            self.selected_features_.update(results)
            logging.info("Wrapper methods completed")
            
            return results
            
        except Exception as e:
            raise CustomeException(e, sys)
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series, 
                          k: int = 10, min_votes: int = 2) -> List[str]:
        """
        Ensemble feature selection combining multiple methods
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            min_votes: Minimum number of methods that should select a feature
            
        Returns:
            List of selected features
        """
        try:
            logging.info("Starting ensemble feature selection")
            
            # Run all individual methods
            self.statistical_feature_selection(X, y, method='all', k=k)
            self.correlation_analysis(X, y)
            self.model_based_selection(X, y, method='all', k=k)
            if k <= 10:  # Only run wrapper methods for small k due to computational cost
                self.wrapper_methods(X, y, method='rfe', k=k)
            
            # Count votes for each feature
            feature_votes = {}
            all_features = list(X.columns)
            
            for feature in all_features:
                votes = 0
                for method_name, selected_features in self.selected_features_.items():
                    if feature in selected_features:
                        votes += 1
                feature_votes[feature] = votes
            
            # Select features with minimum votes
            ensemble_selected = [feature for feature, votes in feature_votes.items() 
                               if votes >= min_votes]
            
            # If not enough features, add top-voted features
            if len(ensemble_selected) < k:
                sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
                for feature, votes in sorted_features:
                    if len(ensemble_selected) >= k:
                        break
                    if feature not in ensemble_selected:
                        ensemble_selected.append(feature)
            
            # If too many features, keep top k by votes
            if len(ensemble_selected) > k:
                feature_vote_scores = [(feature, feature_votes[feature]) for feature in ensemble_selected]
                feature_vote_scores.sort(key=lambda x: x[1], reverse=True)
                ensemble_selected = [feature for feature, _ in feature_vote_scores[:k]]
            
            self.selected_features_['ensemble'] = ensemble_selected
            logging.info(f"Ensemble selected features: {ensemble_selected}")
            
            return ensemble_selected
            
        except Exception as e:
            raise CustomeException(e, sys)
    
    def evaluate_feature_sets(self, X: pd.DataFrame, y: pd.Series, 
                             cv: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluate different feature selection methods using cross-validation
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with evaluation results for each method
        """
        try:
            logging.info("Starting feature set evaluation")
            
            # Encode categorical features
            X_encoded = X.copy()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_features:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
            
            results = {}
            base_estimator = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            
            # Evaluate each feature selection method
            for method_name, selected_features in self.selected_features_.items():
                if not selected_features:
                    continue
                    
                logging.info(f"Evaluating {method_name} with {len(selected_features)} features")
                
                X_selected = X_encoded[selected_features]
                
                # Cross-validation scores
                cv_scores_f1 = cross_val_score(base_estimator, X_selected, y, 
                                              cv=cv, scoring='f1', n_jobs=-1)
                cv_scores_acc = cross_val_score(base_estimator, X_selected, y, 
                                               cv=cv, scoring='accuracy', n_jobs=-1)
                
                results[method_name] = {
                    'num_features': len(selected_features),
                    'features': selected_features,
                    'mean_f1': cv_scores_f1.mean(),
                    'std_f1': cv_scores_f1.std(),
                    'mean_accuracy': cv_scores_acc.mean(),
                    'std_accuracy': cv_scores_acc.std()
                }
            
            # Also evaluate with all features for comparison
            cv_scores_f1_all = cross_val_score(base_estimator, X_encoded, y, 
                                              cv=cv, scoring='f1', n_jobs=-1)
            cv_scores_acc_all = cross_val_score(base_estimator, X_encoded, y, 
                                               cv=cv, scoring='accuracy', n_jobs=-1)
            
            results['all_features'] = {
                'num_features': len(X.columns),
                'features': list(X.columns),
                'mean_f1': cv_scores_f1_all.mean(),
                'std_f1': cv_scores_f1_all.std(),
                'mean_accuracy': cv_scores_acc_all.mean(),
                'std_accuracy': cv_scores_acc_all.std()
            }
            
            logging.info("Feature set evaluation completed")
            
            return results
            
        except Exception as e:
            raise CustomeException(e, sys)
    
    def plot_feature_importance(self, X: pd.DataFrame, save_plots: bool = True) -> None:
        """
        Create visualizations for feature importance and selection results
        
        Args:
            X: Feature matrix
            save_plots: Whether to save plots to disk
        """
        try:
            logging.info("Creating feature importance plots")
            
            if save_plots:
                plt.style.use('default')
                fig_size = (12, 8)
                
                # Plot 1: Feature scores from different methods
                if self.feature_scores_:
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    fig.suptitle('Feature Selection Scores by Method', fontsize=16)
                    
                    plot_idx = 0
                    for method_name, scores in self.feature_scores_.items():
                        if plot_idx >= 4:
                            break
                        
                        row, col = plot_idx // 2, plot_idx % 2
                        ax = axes[row, col]
                        
                        features = list(scores.keys())
                        values = list(scores.values())
                        
                        # Sort by score
                        sorted_data = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
                        features, values = zip(*sorted_data)
                        
                        ax.barh(features, values)
                        ax.set_title(f'{method_name.replace("_", " ").title()} Scores')
                        ax.set_xlabel('Score')
                        
                        # Rotate labels if too many features
                        if len(features) > 10:
                            ax.tick_params(axis='y', labelsize=8)
                        
                        plot_idx += 1
                    
                    # Remove unused subplots
                    for i in range(plot_idx, 4):
                        row, col = i // 2, i % 2
                        fig.delaxes(axes[row, col])
                    
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(os.path.join(self.config.feature_importance_plot_path, 'feature_scores.png'))
                    plt.close()
                
                # Plot 2: Correlation heatmap
                if self.correlation_matrix_ is not None:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(self.correlation_matrix_, annot=True, cmap='coolwarm', center=0,
                               square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                    plt.title('Feature Correlation Matrix')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(os.path.join(self.config.feature_importance_plot_path, 'correlation_matrix.png'))
                    plt.close()
                
                # Plot 3: Feature selection comparison
                if self.selected_features_:
                    feature_selection_summary = {}
                    for method, features in self.selected_features_.items():
                        feature_selection_summary[method] = len(features)
                    
                    plt.figure(figsize=(10, 6))
                    methods = list(feature_selection_summary.keys())
                    counts = list(feature_selection_summary.values())
                    
                    bars = plt.bar(methods, counts)
                    plt.title('Number of Features Selected by Each Method')
                    plt.xlabel('Selection Method')
                    plt.ylabel('Number of Features')
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(os.path.join(self.config.feature_importance_plot_path, 'feature_selection_comparison.png'))
                    plt.close()
                
                logging.info(f"Feature importance plots saved to {self.config.feature_importance_plot_path}")
            
        except Exception as e:
            logging.warning(f"Error creating plots: {str(e)}")
    
    def generate_report(self, X: pd.DataFrame, y: pd.Series, 
                       evaluation_results: Optional[Dict] = None) -> str:
        """
        Generate a comprehensive feature selection report
        
        Args:
            X: Feature matrix
            y: Target variable
            evaluation_results: Results from evaluate_feature_sets
            
        Returns:
            Report string
        """
        try:
            logging.info("Generating feature selection report")
            
            report_lines = []
            report_lines.append("="*80)
            report_lines.append("FEATURE SELECTION ANALYSIS REPORT")
            report_lines.append("="*80)
            report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Dataset shape: {X.shape}")
            report_lines.append(f"Target distribution: {y.value_counts().to_dict()}")
            report_lines.append("")
            
            # Feature types summary
            report_lines.append("FEATURE TYPES SUMMARY:")
            report_lines.append("-" * 40)
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            report_lines.append(f"Categorical features ({len(categorical_features)}): {', '.join(categorical_features)}")
            report_lines.append(f"Numerical features ({len(numerical_features)}): {', '.join(numerical_features)}")
            report_lines.append("")
            
            # Feature selection results
            if self.selected_features_:
                report_lines.append("FEATURE SELECTION RESULTS:")
                report_lines.append("-" * 40)
                
                for method_name, selected_features in self.selected_features_.items():
                    report_lines.append(f"{method_name.upper()} ({len(selected_features)} features):")
                    report_lines.append(f"  {', '.join(selected_features)}")
                    report_lines.append("")
            
            # High correlation pairs
            if self.high_corr_pairs_:
                report_lines.append("HIGH CORRELATION PAIRS:")
                report_lines.append("-" * 40)
                for pair in self.high_corr_pairs_:
                    report_lines.append(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
                report_lines.append("")
            
            # Evaluation results
            if evaluation_results:
                report_lines.append("PERFORMANCE EVALUATION:")
                report_lines.append("-" * 40)
                report_lines.append(f"{'Method':<15} {'# Features':<12} {'F1 Score':<12} {'Accuracy':<12}")
                report_lines.append("-" * 55)
                
                # Sort by F1 score
                sorted_results = sorted(evaluation_results.items(), 
                                      key=lambda x: x[1]['mean_f1'], reverse=True)
                
                for method_name, results in sorted_results:
                    report_lines.append(
                        f"{method_name:<15} {results['num_features']:<12} "
                        f"{results['mean_f1']:.3f}±{results['std_f1']:.3f} "
                        f"{results['mean_accuracy']:.3f}±{results['std_accuracy']:.3f}"
                    )
                report_lines.append("")
            
            # Recommendations
            report_lines.append("RECOMMENDATIONS:")
            report_lines.append("-" * 40)
            
            if evaluation_results:
                best_method = max(evaluation_results.items(), key=lambda x: x[1]['mean_f1'])
                best_method_name, best_results = best_method
                
                report_lines.append(f"+ Best performing method: {best_method_name}")
                report_lines.append(f"  Features: {', '.join(best_results['features'])}")
                report_lines.append(f"  F1 Score: {best_results['mean_f1']:.3f} +/- {best_results['std_f1']:.3f}")
                
                # Check if feature selection improved performance
                if 'all_features' in evaluation_results:
                    all_features_f1 = evaluation_results['all_features']['mean_f1']
                    improvement = best_results['mean_f1'] - all_features_f1
                    
                    if improvement > 0.01:
                        report_lines.append(f"+ Feature selection improved F1 score by {improvement:.3f}")
                    elif improvement < -0.01:
                        report_lines.append(f"! Feature selection decreased F1 score by {abs(improvement):.3f}")
                    else:
                        report_lines.append("> Feature selection had minimal impact on performance")
            
            if self.high_corr_pairs_:
                report_lines.append(f"! Found {len(self.high_corr_pairs_)} highly correlated feature pairs")
                report_lines.append("  Consider removing redundant features to reduce multicollinearity")
            
            report_lines.append("")
            report_lines.append("="*80)
            
            report_text = "\n".join(report_lines)
            
            # Save report to file with UTF-8 encoding
            with open(self.config.feature_analysis_report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            logging.info(f"Feature selection report saved to {self.config.feature_analysis_report_path}")
            
            return report_text
            
        except Exception as e:
            raise CustomeException(e, sys)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, 
                     method: str = 'ensemble', k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Main method to perform feature selection and return selected features
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method to use
            k: Number of features to select
            
        Returns:
            Tuple of (selected feature matrix, list of selected feature names)
        """
        try:
            logging.info(f"Starting complete feature selection with method: {method}")
            
            if method == 'ensemble':
                selected_features = self.ensemble_selection(X, y, k=k)
            elif method == 'statistical':
                results = self.statistical_feature_selection(X, y, method='all', k=k)
                # Use mutual info as default from statistical methods
                selected_features = results.get('mutual_info', list(X.columns[:k]))
            elif method == 'correlation':
                corr_results = self.correlation_analysis(X, y)
                selected_features = corr_results['corr_selected_features'][:k]
            elif method == 'model_based':
                results = self.model_based_selection(X, y, method='all', k=k)
                # Use random forest as default from model-based methods
                selected_features = results.get('random_forest', list(X.columns[:k]))
            elif method == 'wrapper':
                results = self.wrapper_methods(X, y, method='rfe', k=k)
                selected_features = results.get('rfe', list(X.columns[:k]))
            else:
                logging.warning(f"Unknown method {method}, using all features")
                selected_features = list(X.columns)
            
            # Evaluate different methods
            evaluation_results = self.evaluate_feature_sets(X, y)
            
            # Generate plots
            self.plot_feature_importance(X)
            
            # Generate report
            report = self.generate_report(X, y, evaluation_results)
            
            # Save feature selector object
            save_object(
                file_path=self.config.feature_selection_obj_file_path,
                obj=self
            )
            
            logging.info(f"Feature selection completed. Selected {len(selected_features)} features: {selected_features}")
            
            return X[selected_features], selected_features
            
        except Exception as e:
            raise CustomeException(e, sys)


class FeatureSelectionComponentConfig:
    """Configuration class for feature selection component"""
    def __init__(self):
        self.feature_selection_obj_file_path = os.path.join('artifacts', "feature_selector.pkl")
        self.selected_features_file_path = os.path.join('artifacts', "selected_features.pkl")
        self.feature_analysis_report_path = os.path.join('artifacts', "feature_analysis_report.txt")


class FeatureSelectionComponent:
    """
    Main component class for integrating feature selection into the ML pipeline
    """
    
    def __init__(self):
        self.feature_selection_config = FeatureSelectionComponentConfig()
        self.feature_selector = FeatureSelector()
    
    def initiate_feature_selection(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                 target_column: str = 'IsCounselingNeeded',
                                 method: str = 'ensemble', k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Initiate feature selection process
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe 
            target_column: Name of target column
            method: Feature selection method
            k: Number of features to select
            
        Returns:
            Tuple of (selected train features, selected test features, selected feature names)
        """
        try:
            logging.info("Starting feature selection process")
            
            # Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            
            # Perform feature selection on training data
            X_train_selected, selected_features = self.feature_selector.fit_transform(
                X_train, y_train, method=method, k=k
            )
            
            # Apply same feature selection to test data
            X_test_selected = X_test[selected_features]
            
            # Save selected features list
            save_object(
                file_path=self.feature_selection_config.selected_features_file_path,
                obj=selected_features
            )
            
            logging.info(f"Feature selection process completed. Selected features: {selected_features}")
            
            return X_train_selected, X_test_selected, selected_features
            
        except Exception as e:
            raise CustomeException(e, sys)


if __name__ == "__main__":
    # Example usage and testing
    try:
        import pandas as pd
        from sklearn.datasets import make_classification
        
        # Create sample data for testing
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                                 n_redundant=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        
        # Test feature selection
        selector = FeatureSelector()
        X_selected, selected_features = selector.fit_transform(X_df, y_series, method='ensemble', k=10)
        
        print(f"Original features: {X_df.shape[1]}")
        print(f"Selected features: {len(selected_features)}")
        print(f"Selected feature names: {selected_features}")
        
    except Exception as e:
        print(f"Error in testing: {e}")
