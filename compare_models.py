import argparse
import inspect
import os
import re
import tempfile
import warnings

import numpy as np
import openai
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor

from clf_fit_agent import *
from datasets.data import DataReader
from llms.llm import LLM
from models.model import Model
from models.utils import *


def run_regression_pipeline(df_list, target_column):
 
    models = {
        "Linear Regression": (LinearRegression(), {}),
        "KNN Regressor": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}),
        "Random Forest Regressor": (RandomForestRegressor(), {"n_estimators": [100, 200], "max_depth": [5, 10, None]}),
        "XGBoost Regressor": (XGBRegressor(eval_metric='rmse'), {"learning_rate": [0.01, 0.1], "max_depth": [3, 5], "n_estimators": [100, 200]})
    }

    results = []
    
    for i, df in enumerate(df_list):
        print(f"\nProcessing Dataset {i+1}...")
        X, y = df.drop(target_column, axis=1), df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  
            ('scaler', StandardScaler()),  
            ('feature_selection', SelectKBest(score_func=f_regression, k=30))  
        ])
        
        X_train = pipeline.fit_transform(X_train, y_train)
        X_test = pipeline.transform(X_test)

        for model_name, (model, params) in models.items():
            print(f"Training {model_name} on Dataset {i+1}...")
            grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='neg_mean_squared_error', cv=5, verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            results.append({
                "Dataset": f"Dataset {i+1}",
                "Model": model_name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R²": r2
            })

            print(f"{model_name} on Dataset {i+1} -> MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Across Datasets:")
    print(results_df)
    return results_df


def run_classification_pipeline(df_list, target_column):

    models = {
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
        "Logistic Regression": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(), {"n_estimators": [100, 200], "max_depth": [5, 10, None]}),
        "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {"learning_rate": [0.01, 0.1], "max_depth": [3, 5], "n_estimators": [100, 200]})
    }

    results = []
    
    for i, df in enumerate(df_list):
        print(f"\nProcessing Dataset {i+1}...")
        X, y = df.drop(target_column, axis=1), df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  
            ('scaler', StandardScaler()),  
            ('feature_selection', SelectKBest(score_func=f_classif, k=30))  
        ])
        
        X_train = pipeline.fit_transform(X_train, y_train)
        X_test = pipeline.transform(X_test)

        for model_name, (model, params) in models.items():
            print(f"Training {model_name} on Dataset {i+1}...")
            grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=5, verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results.append({
                "Dataset": f"Dataset {i+1}",
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

            print(f"{model_name} on Dataset {i+1} -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Across Datasets:")
    print(results_df)
    return results_df



titanic = pd.read_csv('datasets/Titanic/train.csv')
titanic_features = pd.read_csv('Titanic.csv')
titanic_with_hyp = pd.read_csv('Titanic_hypothesis.csv')

cars = pd.read_csv('datasets/Cars/train.csv')
cars_features = pd.read_csv('Cars.csv')
cars_with_hyp = pd.read_csv('Cars_hypothesis.csv')

diamonds = pd.read_csv('datasets/Diamonds/train.csv')
diamonds_features = pd.read_csv('Diamonds.csv')
diamonds_with_hyp = pd.read_csv('Diamonds_hypothesis.csv')

smoking = pd.read_csv('datasets/Smoking/train.csv')
#smoking_features = pd.read_csv('Smoking.csv')
smoking_with_hyp = pd.read_csv('Smoking_hypothesis.csv')

titanic_comparison = run_classification_pipeline(
    df_list=[titanic_features, titanic_with_hyp],
    target_column='Survived'
)
# cars_comparison = run_regression_pipeline(
#     df_list=[ cars_features, cars_with_hyp],
#     target_column='price'
# )
diamonds_comparison = run_regression_pipeline(
    df_list=[diamonds_features, diamonds_with_hyp],
    target_column='price'
)

# smoking_comparison = run_classification_pipeline(
#     df_list=[smoking_with_hyp],
#     target_column='smoking'
# )

titanic_comparison.to_csv('titanic_comparison.csv')
#cars_comparison.to_csv('cars_comparison.csv')
diamonds_comparison.to_csv('diamonds_comparison.csv')
#smoking_comparison.to_csv('smoking_comparison.csv')