import os
import glob
import sys
from collections import defaultdict

# Core Libraries
import pandas as pd
import numpy as np

# Model Selection and Preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Metrics for Classification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Metrics for Regression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

# Feature Selection
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, mutual_info_classif,
    mutual_info_regression, VarianceThreshold, RFE
)

# Linear Models (Classification)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveClassifier, Perceptron
)

# Linear Models (Regression)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
    OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor, HuberRegressor,
    TheilSenRegressor, RANSACRegressor
)

# Tree-Based Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

# Neighbor-Based Models
from sklearn.neighbors import (
    KNeighborsClassifier, RadiusNeighborsClassifier,
    KNeighborsRegressor, RadiusNeighborsRegressor
)

# Ensemble Models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)

# SVM Models
from sklearn.svm import SVC, LinearSVC, NuSVC, SVR, LinearSVR

# Naive Bayes Models
from sklearn.naive_bayes import (
    GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
)

# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor

# Kernel Ridge
from sklearn.kernel_ridge import KernelRidge

# Cross-Decomposition
from sklearn.cross_decomposition import PLSRegression

# Gradient Boosting Libraries
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from tqdm import tqdm

from reg_fit_agent import *
from clf_fit_agent import *

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

log_file = open("model_training_2.log", "a")

sys.stdout = Tee(sys.stdout, log_file)
print('All logs appear here')

dict_clf = defaultdict(list)
dict_reg = defaultdict(list)

llms_list = [v.replace('/','_') for k,v in get_llms_list()]
dataset_list = ['Churn', 'Smoking', 'Diamonds', 'Cars']
file_names ={
    'prep_1_exp.csv' : 'P',     #Preprocessing
    'full_1_exp.csv' : 'PE',    #Preprocessing + Engineering
    'all_hyp_1_exp.csv' : 'HPE' #Hypothesis + Preprocessing + Engineering
}

df_to_target = {
    'Cars' : 'price',
    'Diamonds' : 'price',
    'Churn' : 'Exited',
    'Smoking' : 'smoking'
}

main_path = 'logs/all_dataframes'

for dataset in tqdm(dataset_list):
    for llm in llms_list:
        for csv in list(file_names.keys()):
            print(f'Start of {file_names[csv]} for {dataset} using {llm}')
            print('#' * 60)
            file_path = f'{main_path}/{dataset}_{llm}/{csv}'

            df = pd.read_csv(file_path)
            target = df_to_target[dataset]
            y = df[target]
            X = df.drop([target], axis = 1)

            if f'Cars' in file_path or 'Diamonds' in file_path:
                dict_reg['Dataset'].append(dataset)
                dict_reg['LLM'].append(llm)
                dict_reg['ToT'].append(file_names[csv])

                X = X.dropna(axis=1, how='all')
                X = impute_missing_values(X)
                X = remove_linearly_dependent_columns(X)
                X = remove_zero_variance_columns(X)
                #Check for categorical features
                if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
                    raise ValueError("Categorical features detected. Please encode categorical variables before training.")
                performance_metrics = ['r2', 'rmse']
                results = main_reg(X=X, y=y, metrics=performance_metrics)
                i = 1
                for k,v in results[1].items():
                    dict_reg[f'M{i}'].append(k)
                    dict_reg[f'Performance of M{i}'].append(list(v.values()))
                    i += 1
                dict_reg = dict(dict_reg)

            else:
                dict_clf['Dataset'].append(dataset)
                dict_clf['LLM'].append(llm)
                dict_clf['ToT'].append(file_names[csv])

                df = pd.read_csv(file_path)
                target = df_to_target[dataset]
                y = df[target]
                X = df.drop([target], axis = 1)
                X.fillna(X.median(), inplace=True)
                X.dropna(axis=1, how='all', inplace=True)
                X = X.loc[:, X.nunique() > 1]
                results = main_clf(X,y)

                i = 1
                for k,v in results[1].items():
                    dict_clf[f'M{i}'].append(k)
                    dict_clf[f'Performance of M{i}'].append(list(v.values())) 
                    i += 1
                dict_clf = dict(dict_clf)
  
    df_clf = pd.DataFrame(dict_clf)
    df_reg = pd.DataFrame(dict_reg)

    df_clf.to_csv('clf_results.csv')
    df_reg.to_csv('reg_results.csv ')


#{'LogisticRegression': {'accuracy': 0.859375, 'f1': 0.8467265963115485}, 'Voting Classifier': {'accuracy': 0.8525, 'f1': 0.8335107407407406}, 'Stacking Classifier': {'accuracy': 0.858125, 'f1': 0.8464360432972236}}