import argparse
import inspect
import os
import re
import tempfile
import warnings

import openai
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

from clf_fit_agent import *
from datasets.data import DataReader
from llms.llm import LLM
from models.model import Model
from models.utils import *


def run_classification_pipeline(df_to_test, df_to_test_hyp, target_column):
    """
    Run multiple ML models with feature selection and hyperparameter tuning 
    on two datasets and compare their accuracy.

    Parameters:
        df_to_test (pd.DataFrame): First dataset.
        df_to_test_hyp (pd.DataFrame): Second dataset.
        target_column (str): The target column name.

    Returns:
        None
    """
    
    X1, y1 = df_to_test.drop(target_column, axis=1), df_to_test[target_column]
    X2, y2 = df_to_test_hyp.drop(target_column, axis=1), df_to_test_hyp[target_column]
    
   
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)

   
    models = {
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
        "Logistic Regression": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(), {"n_estimators": [100, 200], "max_depth": [5, 10, None]}),
        "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {"learning_rate": [0.01, 0.1], "max_depth": [3, 5], "n_estimators": [100, 200]})
    }

    results = {"Model": [], "Accuracy on df_to_test": [], "Accuracy on df_to_test_hyp": []}
    
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  
        ('scaler', StandardScaler()),  
        ('feature_selection', SelectKBest(score_func=f_classif, k=30))  
    ])
    
    X_train1 = pipeline.fit_transform(X_train1, y_train1)
    X_test1 = pipeline.transform(X_test1)
    X_train2 = pipeline.fit_transform(X_train2, y_train2)
    X_test2 = pipeline.transform(X_test2)
    
    for model_name, (model, params) in models.items():
        print(f"Training {model_name}...")
       
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=5, verbose=1)
        grid_search.fit(X_train1, y_train1)
        best_model = grid_search.best_estimator_
        
       
        accuracy1 = accuracy_score(y_test1, best_model.predict(X_test1))
        accuracy2 = accuracy_score(y_test2, best_model.predict(X_test2))
        
        results["Model"].append(model_name)
        results["Accuracy on df_to_test"].append(accuracy1)
        results["Accuracy on df_to_test_hyp"].append(accuracy2)
        
        print(f"{model_name} -> Accuracy on df_to_test: {accuracy1:.4f}, Accuracy on df_to_test_hyp: {accuracy2:.4f}")
    
    
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
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

# TODO: validation with and without hypothesis
# TODO: generation followed by hypothesis
# TODO: 