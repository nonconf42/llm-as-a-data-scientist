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










ml_model = Model('XGB', 'classification')

api_key = "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"  # Replace with your actual API key
llm_model = LLM('gpt')
model_name = "gpt-4o"

hypothesis_results = []

df = DataReader('Titanic')

h_list = [
    "General Hypothesis",
    "Clustering Hypothesis",
    "Outlier Detection",
    "Anomaly Detection",
    "Cause-Effect Relationships",
    "Latent Variable Hypotheses"
]

for hyp_type in h_list:
    prompt = generate_hypothesis_prompt(hypothesis_type=hyp_type, dataset=df,num_hypothesis=10)
    print('Below is hypothesis list:::::::::')
    print(prompt)
    llm_output = llm_model.llm_call(prompt)
    python_code = extract_python_code_hyp(llm_output)
    hypothesis_list = extract_hypotheses_text(python_code)
    dataset = df.train_base
    imports = ''' 
                    # Data manipulation and analysis
                    from scipy.stats import chi2_contingency
                    import scipy.stats
                    import numpy as np
                    import pandas as pd
                    # Machine learning models and preprocessing
                    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
                    from sklearn.pipeline import Pipeline
                    # Commonly used machine learning algorithms
                    from sklearn.linear_model import LogisticRegression, LinearRegression
                    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
                    from sklearn.svm import SVC, SVR
                    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                    from sklearn.naive_bayes import GaussianNB
                    from sklearn.cluster import KMeans
                    # Model evaluation metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, mean_squared_error, r2_score
                    # Dimensionality reduction
                    from sklearn.decomposition import PCA
                '''
    imports = strip_multiline_string(imports)
    for code in hypothesis_list:
        code = imports + '\n' + code 
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_module:
            temp_module_name = os.path.basename(temp_module.name).split('.')[0]
            temp_module.write(code.encode('utf-8'))
            temp_module_path = temp_module.name
        try:
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(temp_module_name, temp_module_path)
            module = importlib.util.module_from_spec(spec)
            module.dataset = dataset
            spec.loader.exec_module(module)
            #print(f'##########\n {dir(module)}')
            if hasattr(module, 'test'):
                result = module.test(dataset)
                description = module.description
                insights = module.insights
                hypothesis_results.append((insights, result))
                print(module.description)
                print(f'INSIGHTS: {module.insights}')
                print(result)
                print('#####')
                print(f'COT:{module.COT}')
        except Exception as e:
            print('ERROR')
            print(e)
            print(code)
            print('#####')
        finally:
            # Clean up the temporary file
            os.remove(temp_module_path)
print('$$$$$$$$$$$$$$$$$$')
print(hypothesis_results)

print('Cleaned hypothesis results:')
cleaned_hypothesis_results = [hyp for hyp in hypothesis_results if hyp[1] != None]
print(cleaned_hypothesis_results)
print(f'Len: {len(cleaned_hypothesis_results)}')
print('$$$$$$$$$$$$$$$$$$$$$$$$')
#bag_of_hyp = hyp_results_to_text(cleaned_hypothesis_results, llm_model=llm_model)
bag_of_hyp = hyp_res_to_text(cleaned_hypothesis_results)
print('HYPOTHESIS TEXT:')
print(bag_of_hyp)



auto_ml_agent(
    dataset_name='Titanic',
    ml_model_type='XGB',
    llm_model_type='gpt',
    model_name='gpt-4o',
    num_of_generations=1,
    num_of_features=20,
    debug=True,
    output_file='df_without_hypothesis.csv'
)

auto_ml_agent(
    dataset_name='Titanic',
    ml_model_type='XGB',
    llm_model_type='gpt',
    model_name='gpt-4o',
    num_of_generations=1,
    num_of_features=20,
    debug=True,
    output_file='df_with_hypothesis.csv',
    hypothesis=bag_of_hyp
)

df_to_test = pd.read_csv('df_without_hypothesis.csv')
df_to_test_hyp = pd.read_csv('df_with_hypothesis.csv')
print(f'# of NAN without hyp:{df_to_test.isna().sum()}')
print(f'# of NAN wit hyp:{df_to_test_hyp.isna().sum()}')
comparison = run_classification_pipeline(df_to_test=df_to_test, df_to_test_hyp=df_to_test_hyp,target_column='Survived')