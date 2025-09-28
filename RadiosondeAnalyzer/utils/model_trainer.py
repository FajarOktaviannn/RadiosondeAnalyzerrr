import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        pass
    
    def prepare_data(self, data, feature_columns, target_column, label_rules):
        """Prepare data for training"""
        # Apply labels
        from utils.data_processor import DataProcessor
        data_processor = DataProcessor()
        labeled_data = data_processor.apply_labels(data, target_column, label_rules)
        
        # Prepare features and target
        X = labeled_data[feature_columns]
        y = labeled_data['label']
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        return X, y_encoded, label_encoder, labeled_data
    
    def get_model(self, algorithm, params=None):
        """Get model instance based on algorithm"""
        if params is None:
            params = {}
        
        if algorithm == "Random Forest":
            return RandomForestClassifier(**params)
        elif algorithm == "SVM":
            return SVC(**params)
        elif algorithm == "Decision Tree":
            return DecisionTreeClassifier(**params)
        elif algorithm == "K-Nearest Neighbors":
            return KNeighborsClassifier(**params)
        elif algorithm == "Gradient Boosting":
            return GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def train_model_with_params(self, data, feature_columns, target_column, label_rules, 
                               algorithm, params, test_size, random_state, use_smote=False, smote_params=None):
        """Train model with specified parameters"""
        
        # Prepare data
        X, y_encoded, label_encoder, labeled_data = self.prepare_data(data, feature_columns, target_column, label_rules)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Store original distribution
        unique, counts = np.unique(y_train, return_counts=True)
        original_distribution = {}
        for i, label_idx in enumerate(unique):
            original_label = label_encoder.inverse_transform([label_idx])[0]
            original_distribution[original_label] = counts[i]
        
        # Apply SMOTE if requested
        resampled_distribution = original_distribution.copy()
        if use_smote:
            if smote_params is None:
                smote_params = {}
            
            smote = SMOTE(random_state=random_state, **smote_params)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Update resampled distribution
            unique, counts = np.unique(y_train, return_counts=True)
            resampled_distribution = {}
            for i, label_idx in enumerate(unique):
                original_label = label_encoder.inverse_transform([label_idx])[0]
                resampled_distribution[original_label] = counts[i]
        
        # Train model
        model = self.get_model(algorithm, params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return X_train, X_test, y_train, y_test, model, y_pred, accuracy, report, conf_matrix, label_encoder, original_distribution, resampled_distribution
    
    def train_model_with_hyperparameter_tuning(self, data, feature_columns, target_column, label_rules, 
                                             algorithm, tuning_params, test_size, random_state, 
                                             use_smote=False, smote_params=None):
        """Train model with hyperparameter tuning"""
        
        # Prepare data
        X, y_encoded, label_encoder, labeled_data = self.prepare_data(data, feature_columns, target_column, label_rules)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Store original distribution
        unique, counts = np.unique(y_train, return_counts=True)
        original_distribution = {}
        for i, label_idx in enumerate(unique):
            original_label = label_encoder.inverse_transform([label_idx])[0]
            original_distribution[original_label] = counts[i]
        
        # Apply SMOTE if requested
        resampled_distribution = original_distribution.copy()
        if use_smote:
            if smote_params is None:
                smote_params = {}
            
            smote = SMOTE(random_state=random_state, **smote_params)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Update resampled distribution
            unique, counts = np.unique(y_train, return_counts=True)
            resampled_distribution = {}
            for i, label_idx in enumerate(unique):
                original_label = label_encoder.inverse_transform([label_idx])[0]
                resampled_distribution[original_label] = counts[i]
        
        # Get base model
        base_model = self.get_model(algorithm)
        
        # Setup hyperparameter tuning
        param_grid = tuning_params['param_grid']
        cv_folds = tuning_params['cv_folds']
        scoring = tuning_params['scoring']
        method = tuning_params['method']
        
        print(f"Starting {method} with {len(param_grid)} parameters...")
        print(f"Parameter grid: {param_grid}")
        
        if method == "GridSearchCV":
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        else:  # RandomizedSearchCV
            n_iter = tuning_params.get('n_iter', 50)
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                random_state=random_state,
                verbose=1
            )
        
        # Perform hyperparameter tuning
        search.fit(X_train, y_train)
        
        # Get best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_results = search.cv_results_
        
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")
        
        # Make predictions with best model
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return (X_train, X_test, y_train, y_test, best_model, y_pred, accuracy, report, 
                conf_matrix, label_encoder, original_distribution, resampled_distribution, 
                best_params, cv_results)
    
    def get_hyperparameter_ranges(self, algorithm):
        """Get default hyperparameter ranges for each algorithm"""
        ranges = {
            "Random Forest": {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "SVM": {
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            },
            "Decision Tree": {
                'max_depth': [3, 5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            "K-Nearest Neighbors": {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        return ranges.get(algorithm, {})
