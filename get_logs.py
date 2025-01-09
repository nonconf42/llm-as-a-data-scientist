import inspect
import os
import re
import tempfile
import warnings
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

def main():
    parser = ArgumentParser(
        description='Helps to test some functionality of auto ml'
    )
    

    subparsers = parser.add_subparsers(help='Choose command', dest='command')
    
    prompt_parser = subparsers.add_parser(
        'prompt',
        help='Checks whether general prompt or hypothesis prompt',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )


    prompt_parser.add_argument(
        '-t', '--type', 
        help='checks general prompt',
        default=None
    )

    hypothesis_parser = subparsers.add_parser(
        'hypothesis',
        help='Shows generated hypothesis',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    hypothesis_parser.add_argument(
        '-d', '--dataset',
        default='Titanic'
    )


  
  

    args = parser.parse_args()
    ml_model = Model('XGB', 'classification')
    hypothesis_list = [
        "General Hypothesis",
        "Clustering Hypothesis",
        "Outlier Detection",
        "Anomaly Detection",
        "Cause-Effect Relationships",
        "Latent Variable Hypotheses"
    ]

    api_key = "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"  # Replace with your actual API key
    llm_model = LLM('gpt')
    model_name = "gpt-4o"

    dataset_names = ['Cars', 'Titanic', 'Diamonds', 'Smoking']
   
    if args.command == 'prompt':
        if args.type == 'general':
            for name in dataset_names:
                dataset = DataReader(name)
                prep_prompt = generate_prompt(
                    dataset=dataset,
                    instruction_type='preprocessing',
                    transformation_type='Encoding Data'
                )

                with open(f'logs/prompts/{name}_prep.txt', 'w') as file:
                    file.write(prep_prompt) 

                eng_prompt = generate_prompt(
                    dataset=dataset,
                    instruction_type='engineering',
                    transformation_type='Feature Scaling'
                )

                with open(f'logs/prompts/{name}_eng.txt', 'w') as file:
                    file.write(eng_prompt) 

        elif args.type == 'hypothesis':
            for name in dataset_names:
                dataset = DataReader(name)
                for hypothesis in hypothesis_list:
                    hypothesis_prompt = generate_hypothesis_prompt(hypothesis, dataset)
                    with open(f'logs/prompts/hypothesis_prompts/{name}_{hypothesis}.txt', 'a+') as file:
                        file.write(hypothesis_prompt)
    
    elif args.command == 'hypothesis':
        hypothesis_results = []
        file = open(f'logs/hypothesis/{args.dataset}.txt', 'w')
        df = DataReader(args.dataset)

        for hyp_type in tqdm(hypothesis_list):
            prompt = generate_hypothesis_prompt(hypothesis_type=hyp_type, dataset=df,num_hypothesis=10)
            llm_output = llm_model.llm_call(prompt)
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
            for code in tqdm(hypothesis_list):
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
                        hypothesis_results.append((description, result))
                        file.write(module.description)
                        file.write('\n')
                        file.write(f'INSIGHTS: {module.insights}')
                        file.write('\n')
                        file.write(str(result))
                        file.write('\n')
                        file.write('#' * 20)
                        file.write('\n')
                        file.write(f'COT:{module.COT}')
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
        file.write('HYPOTHESIS TEXT:\n')
        bag_of_hyp = hyp_results_to_text(cleaned_hypothesis_results, llm_model=llm_model)
        file.write(bag_of_hyp)
        file.close()
if __name__ == "__main__":
    main()