import inspect
import os
import re
import tempfile
import warnings
import sys
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

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush() 

    def flush(self):
        for stream in self.streams:
            stream.flush()

log_file = open("output.log", "a")

sys.stdout = Tee(sys.stdout, log_file)
print('All logs appear here')

NUM_OF_EXPERIMENTS = 3

llm_models = get_llms_list()
#datasets = get_dataset_names()
datasets = ['Churn', 'Smoking', 'Cars', 'Diamonds']

print(datasets)
print(llm_models)
f_path_beginning = 'logs/all_dataframes'
for dataset_name in tqdm(datasets):
    for platform, model_name in tqdm(llm_models):
        model_name_for_fpath =  model_name.replace('/', '_')
        new_dir_name = f'{dataset_name}_{model_name_for_fpath}'
        new_dir_path = os.path.join(f_path_beginning, new_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
        print(f'{new_dir_name} is created successfully...')

        hypothesis_path = f'logs/all_hypotheses/{dataset_name}_{model_name_for_fpath}_hypotheses_text_only.txt'
        with open(hypothesis_path, 'r') as file:
            hypothesis_content = file.read()
            print(f'\nHypotheses fetched successfully from  {hypothesis_path}')
        
        for i in range(NUM_OF_EXPERIMENTS):
            print(f'ITERATION {i+1} started......')
            if not os.path.isfile(f'{new_dir_path}/prep_{i+1}_exp.csv'):
                auto_ml_agent(
                    dataset_name=dataset_name,
                    ml_model_type='XGB',
                    llm_model_type=platform,
                    model_name=model_name,
                    num_of_generations=1,
                    num_of_features=50,
                    debug=False,
                    hypothesis=None,
                    preprocessing=True,
                    engineering=False,
                    df_csv_path=f'{new_dir_path}/prep_{i+1}_exp.csv',
                    df_features_path=f'{new_dir_path}/features_prep_{i+1}_exp.csv',
                )
            print(f'{new_dir_path}/prep_{i+1}_exp.csv finished......')

            if not os.path.isfile(f'{new_dir_path}/full_{i+1}_exp.csv'):
                auto_ml_agent(
                    dataset_name=dataset_name,
                    ml_model_type='XGB',
                    llm_model_type=platform,
                    model_name=model_name,
                    num_of_generations=1,
                    num_of_features=50,
                    debug=False,
                    hypothesis=None,
                    preprocessing=True,
                    engineering=True,
                    df_csv_path=f'{new_dir_path}/full_{i+1}_exp.csv',
                    df_features_path=f'{new_dir_path}/features_full_{i+1}_exp.csv',
                )
            print(f'{new_dir_path}/full_{i+1}_exp.csv finished......')

            # if not os.path.isfile(f'{new_dir_path}/prep_hyp_{i+1}_exp.csv'):
            #     auto_ml_agent(
            #         dataset_name=dataset_name,
            #         ml_model_type='XGB',
            #         llm_model_type=platform,
            #         model_name=model_name,
            #         num_of_generations=1,
            #         num_of_features=50,
            #         debug=False,
            #         hypothesis=hypothesis_content,
            #         preprocessing=True,
            #         engineering=False,
            #         df_csv_path=f'{new_dir_path}/prep_hyp_{i+1}_exp.csv',
            #         df_features_path=f'{new_dir_path}/features_prep_hyp_{i+1}_exp.csv',
            #     )

            if not os.path.isfile(f'{new_dir_path}/all_hyp_{i+1}_exp.csv'):
                auto_ml_agent(
                    dataset_name=dataset_name,
                    ml_model_type='XGB',
                    llm_model_type=platform,
                    model_name=model_name,
                    num_of_generations=1,
                    num_of_features=50,
                    debug=False,
                    hypothesis=hypothesis_content,
                    preprocessing=True,
                    engineering=True,
                    df_csv_path=f'{new_dir_path}/all_hyp_{i+1}_exp.csv',
                    df_features_path=f'{new_dir_path}/features_all_hyp_{i+1}_exp.csv',
                )
            print(f'{new_dir_path}/all_hyp_{i+1}_exp.csv finished......')

            print(f'ITERATION {i+1} finished......')
            
log_file.close()