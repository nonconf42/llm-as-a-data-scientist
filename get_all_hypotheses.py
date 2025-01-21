import inspect
import os
import re
import tempfile
import warnings
from textwrap import dedent
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType, ArgumentTypeError

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
from tqdm import tqdm

from clf_fit_agent import *
from datasets.data import DataReader
from llms.llm import LLM
from models.model import Model
from models.utils import *

llm_models = get_llms_list()
datasets = get_dataset_names()
hypothesis_list = [
    "General Hypothesis",
    "Clustering Hypothesis",
    "Outlier Detection",
    "Anomaly Detection",
    "Cause-Effect Relationships",
    "Latent Variable Hypotheses"
]

for dataset_name in tqdm(datasets):
    for platform, model_name in tqdm(llm_models):
        model_name_to_save = model_name.replace('/', '_')
        file_path = f'logs/all_hypotheses/{dataset_name}_{model_name_to_save}_hypotheses_text_only.txt'
        if os.path.isfile(file_path):
            print(f'{dataset_name}_{model_name_to_save}_hypotheses_text_only.txt already exist')
            continue
        hypothesis_dict = {}
        df = DataReader(dataset_name=dataset_name)
        hypothesis_results = []
        file = open(f'logs/all_hypotheses/{dataset_name}_{model_name_to_save}.txt', 'w')
        llm_model = LLM(platform=platform)
        for hyp_type in hypothesis_list:
            prompt = generate_hypothesis_prompt(hypothesis_type=hyp_type, dataset=df,num_hypothesis=10)
            llm_output = llm_model.llm_call(prompt=prompt, model_name=model_name )
            python_code = extract_python_code(llm_output)
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
                        hypothesis_results.append((module.insights, result))
                        file.write(f'DESCRIPTION: {module.description}')
                        file.write('\n')
                        file.write(f'INSIGHTS: {module.insights}')
                        file.write('\n')
                        file.write(str(result))
                        file.write('\n')
                        file.write('#' * 20)
                        file.write('\n')
                        file.write(f'COT:{module.COT}')
                        file.write('CODE:\n')
                        file.write(code + '\n')
                        file.write('END:\n')
                        hypothesis_dict[module.insights] = result
                except Exception as e:
                    file.write('ERROR')
                    file.write('\n')
                    file.write(str(e))
                    file.write('\n')
                    file.write(code)
                    file.write('\n')
                    file.write('#####')
                    file.write('\n')
                finally:
                    # Clean up the temporary file
                    os.remove(temp_module_path)

        file.write('#'*20 + '\n')
        file.write('HYPOTHESIS RESULTS:\n')
        file.write(str(hypothesis_results))
        file.write('\n\n\n\n')

        file.write('#'*20 + '\n')
        file.write('CLEANED HYPOTHESIS RESULTS:')
        file.write('\n')
        cleaned_hypothesis_results = [hyp for hyp in hypothesis_results if hyp[1] != None]
        file.write(str(cleaned_hypothesis_results))
        file.write('\n')
        file.write(f'Len: {len(cleaned_hypothesis_results)}')

        file.write('\n\n\n\n')
        file.write('#'*20 + '\n')
        file.write('HYPOTHESIS DICTIONARY:\n')
        # bag_of_hyp = hyp_results_to_text(cleaned_hypothesis_results, llm_model=llm_model)
        # file.write(bag_of_hyp)
        file.write(str(hypotheses_dict) + '\n')

        file.write('HYPOTHESIS RESULTS:\n')
        file.write(str(hypothesis_results))

        file.close()
        hypothesis_tuple_to_text(
            cleaned_hypothesis_results, 
            f'logs/all_hypotheses/{dataset_name}_{model_name_to_save}_hypotheses_text_only.txt'
        )
        print(f'Generation of hypothesis finished for dataset {dataset_name} with {model_name}')