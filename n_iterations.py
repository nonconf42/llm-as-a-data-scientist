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

with open(f'logs/hypothesis/{args.dataset}_hypothesis_text.txt') as file:
            hypothesis_text = file.read()

auto_ml_agent(
    dataset_name='Titanic',
    ml_model_type='XGB',
    llm_model_type='gpt',
    model_name='gpt-4o',
    debug=True,
    num_of_generations=5
)