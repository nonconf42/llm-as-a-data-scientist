# Import necessary libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, VarianceThreshold, RFE, f_classif

from sklearn.linear_model import (LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier,
                                  Perceptron)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier,
                              ExtraTreesClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# You may need to install these libraries if not already installed
# pip install xgboost lightgbm catboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from models.utils import *

import warnings
warnings.filterwarnings('ignore')

def load_data(data_path, target_column):
    """Load dataset from a CSV file."""
    data = pd.read_csv(data_path)

    # Check if target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical target variables if necessary
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def get_classifiers(n_features):
    """Define classifiers, feature selection methods, and their hyperparameter grids."""
    classifiers = []
    params = []

    # Preprocessing (scaling)
    scaler = StandardScaler()

    # List of base classifiers and their parameters

    
    base_classifiers = [
        #works. good performance
        ('LogisticRegression', LogisticRegression(max_iter=10000), {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'None'],
            'classifier__C': np.logspace(-4, 4, 20),
            #'classifier__solver': ['lbfgs', 'saga'],
            #'classifier__l1_ratio': np.linspace(0, 1, 5)
        }, True),

        #works. poor performance
    #     ('RidgeClassifier', RidgeClassifier(max_iter=100000), {
    #         'classifier__alpha': np.logspace(-4, 4, 20),
    #         'classifier__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
    #     }, True),

    #    # works. good performance
    #     ('SGDClassifier', SGDClassifier(max_iter=1000, tol=1e-3), {
    #         'classifier__loss': ['log_loss', 'modified_huber', 'squared_hinge'],
    #         'classifier__penalty': ['l2', 'l1', 'elasticnet'],
    #         'classifier__alpha': np.logspace(-6, -1, 10),
    #         'classifier__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    #     }, True),

    #     #fixed
    #     ('PassiveAggressiveClassifier', PassiveAggressiveClassifier(max_iter=1000, tol=1e-3), {
    #         'classifier__C': np.logspace(-4, 1, 10),
    #         'classifier__loss': ['hinge', 'squared_hinge'],
    #     }, True),

        #works. good performance on Iris
        ('KNeighborsClassifier', KNeighborsClassifier(), {
            'classifier__n_neighbors': range(1, 31),
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
        }, True),

        #works. good performance on Iris
        ('DecisionTreeClassifier', DecisionTreeClassifier(), {
            'classifier__criterion': ['gini'],#, 'entropy', 'log_loss'],
            'classifier__max_depth': [None] + list(range(2, 20)),
            'classifier__min_samples_split': range(2, 20),
            'classifier__min_samples_leaf': range(1, 20)
        }, False),

        #works. good performance on Iris
        ('RandomForestClassifier', RandomForestClassifier(), {
            'classifier__n_estimators': [100, 200, 500],
            'classifier__criterion': ['gini'],#, 'entropy', 'log_loss'],
            'classifier__max_depth': [None] + list(range(2, 20, 3)),
            'classifier__min_samples_split': range(2, 20, 3),
            'classifier__min_samples_leaf': range(1, 20, 3)
        }, False),

       # works. good performance on Iris
        # ('GradientBoostingClassifier', GradientBoostingClassifier(), {
        #     'classifier__n_estimators': [100, 200, 500],
        #     'classifier__learning_rate': [0.001, 0.01, 0.1],
        #     'classifier__loss': ['log_loss', 'exponential'],
        #     'classifier__max_depth': range(2, 10),
        #     'classifier__min_samples_split': range(2, 20),
        #     'classifier__min_samples_leaf': range(1, 20)
        # }, False),

        #gives some warnings
        # ('AdaBoostClassifier', AdaBoostClassifier(), {
        #     'classifier__n_estimators': [50, 100, 200],
        #     'classifier__learning_rate': [0.001, 0.01, 0.1, 1.0],
        #     'classifier__algorithm': ['SAMME', 'SAMME.R']
        # }, False),

        # ('SVC', SVC(probability=True), [
        #     {
        #         'classifier__kernel': ['linear'],
        #         'classifier__C': np.logspace(-3, 3, 10),
        #     },
        #     {
        #         'classifier__kernel': ['poly'],
        #         'classifier__C': np.logspace(-3, 3, 10),
        #         'classifier__degree': [2, 3, 4],
        #         'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 5)),
        #     },
        #     {
        #         'classifier__kernel': ['rbf', 'sigmoid'],
        #         'classifier__C': np.logspace(-3, 3, 10),
        #         'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 5)),
        #     },
        # ], True),

        # ('SVC_linear', SVC(probability=True), 
        #     {
        #         'classifier__kernel': ['linear'],
        #         'classifier__C': np.logspace(-3, 3, 10),
        #     }, True),

        # ('LinearSVC', LinearSVC(max_iter=10000), {
        #     'classifier__C': np.logspace(-3, 3, 10),
        #     'classifier__penalty': ['l2'],
        #     'classifier__loss': ['hinge', 'squared_hinge'],
        # }, True),

        # ('NuSVC', NuSVC(probability=True), [
        #     {
        #         'classifier__kernel': ['linear'],
        #         'classifier__nu': np.linspace(0.1, 0.9, 9),
        #     },
        #     {
        #         'classifier__kernel': ['poly'],
        #         'classifier__nu': np.linspace(0.1, 0.9, 9),
        #         'classifier__degree': [2, 3, 4],
        #         'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 5)),
        #     },
        #     {
        #         'classifier__kernel': ['rbf', 'sigmoid'],
        #         'classifier__nu': np.linspace(0.1, 0.9, 9),
        #         'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 5)),
        #     },
        # ], True),

        #works
        # ('GaussianNB', GaussianNB(), {}, False), 

        # #works. But good only for Binary classification
        # ('BernoulliNB', BernoulliNB(), {        
        #     'classifier__alpha': np.logspace(-4, 0, 10),
        #     'classifier__binarize': [0.0, 0.5, 1.0],
        # }, False),

        #works. Gives good performance on Iris
        # ('MultinomialNB', MultinomialNB(), {       
        #     'classifier__alpha': np.logspace(-4, 0, 10),
        #     'classifier__fit_prior': [True, False],
        # }, False),

        #works. Poor performance on Iris
        # ('ComplementNB', ComplementNB(), {         
        #     'classifier__alpha': np.logspace(-4, 0, 10),
        #     'classifier__fit_prior': [True, False],
        # }, False),

        #works. Good performance on Iris
        # ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis(), { 
        #     'classifier__solver': ['svd', 'lsqr', 'eigen'],
        #     'classifier__shrinkage': [None, 'auto'] + list(np.linspace(0, 1, 5)),
        # }, True),

        # # # works. Good performance on Iris
        # ('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis(), {
        #     'classifier__reg_param': np.linspace(0, 1, 5),
        # }, True),

        # #works. Good performance on Iris
        ('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
            'classifier__n_estimators': [100, 200, 500],
            'classifier__learning_rate': [0.001, 0.01, 0.1],
            'classifier__max_depth': range(2, 10),
            'classifier__subsample': [0.5, 0.7, 0.9, 1.0],
            'classifier__colsample_bytree': [0.5, 0.7, 0.9, 1.0]
        }, False),

       
        # ('LGBMClassifier', LGBMClassifier(force_row_wise=True), {
        #     'classifier__n_estimators': [100, 200, 500],
        #     'classifier__learning_rate': [0.001, 0.01, 0.1],
        #     'classifier__max_depth': range(2, 6),
        #     'classifier__num_leaves': [70],
        #     'classifier__subsample': [0.5, 0.7, 0.9, 1.0]
        # }, False),

        #works. Good performance on Iris
        ('CatBoostClassifier', CatBoostClassifier(silent=True), {
            'classifier__iterations': [100, 200, 500],
            'classifier__learning_rate': [0.001, 0.01, 0.1],
            'classifier__depth': range(2, 10),
            'classifier__l2_leaf_reg': [1, 3, 5, 7, 9],
            'classifier__bagging_temperature': [0, 1, 2, 5, 10]
        }, False)
     ]

    # Define feature selectors and their parameters
    feature_selector_list = [
        ('variance_threshold', VarianceThreshold(), {
            'feature_selection__threshold': [0.0]#, 0.01, 0.1]
        }),
        # ('select_k_best_chi2', SelectKBest(score_func=f_classif), {
        #     'feature_selection__k': [10, 20, 30,  'all']
        # }),
        # ('select_k_best_mi', SelectKBest(score_func=mutual_info_classif), {
        #     'feature_selection__k': [10, 20, 30,  'all']
        #}),
        # ('rfe', RFE(estimator=LogisticRegression(max_iter=1000)), {
        #     'feature_selection__n_features_to_select': [10, 20, 30]
        # })
    ]

    max_features = n_features

    for clf_name, clf, clf_params, needs_scaling in base_classifiers:
        for fs_name, fs, fs_params in feature_selector_list:
            # Create pipeline
            steps = []
            if needs_scaling:
                steps.append(('scaler', scaler))
            steps.append(('feature_selection', fs))
            steps.append(('classifier', clf))
            pipe = Pipeline(steps)
            classifiers.append(pipe)
            # Merge classifier parameters with feature selection parameters
            if isinstance(clf_params, list):  # For classifiers with multiple parameter grids
                for grid in clf_params:
                    param_grid = grid.copy()
                    param_grid.update(fs_params)
                    params.append(param_grid)
            else:
                param_grid = clf_params.copy()
                param_grid.update(fs_params)
                params.append(param_grid)

            # Bagging Classifier
            # bagging_clf = BaggingClassifier(estimator=clf, n_estimators=10, max_samples=0.8,
            #                                 bootstrap=True, n_jobs=-1, random_state=42)
            # steps_bagging = []
            # if needs_scaling:
            #     steps_bagging.append(('scaler', scaler))
            # steps_bagging.append(('feature_selection', fs))
            # steps_bagging.append(('classifier', bagging_clf))
            # pipe_bagging = Pipeline(steps_bagging)
            # classifiers.append(pipe_bagging)
            # # Bagging parameters
            # if isinstance(clf_params, list):  # For classifiers with multiple parameter grids
            #     for grid in clf_params:
            #         param_grid_bagging = {}
            #         param_grid_bagging.update({f'classifier__estimator__{k.split("__")[1]}': v for k, v in grid.items()})
            #         param_grid_bagging.update(fs_params)
            #         params.append(param_grid_bagging)
            # else:
            #     param_grid_bagging = {}
            #     param_grid_bagging.update({f'classifier__estimator__{k.split("__")[1]}': v for k, v in clf_params.items()})
            #     param_grid_bagging.update(fs_params)
            #     params.append(param_grid_bagging)

    return classifiers, params

def hyperparameter_tuning(clf, params, X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    rand_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=50, cv=5,
                                     scoring='accuracy', n_jobs=-1, random_state=42)
    rand_search.fit(X_train, y_train)
    return rand_search.best_estimator_, rand_search.best_score_

def calculate_metrics(y_true, y_pred, y_prob, metrics_list):
    """Calculate specified performance metrics."""
    metrics_results = {}
    for metric in metrics_list:
        if metric == 'accuracy':
            acc = accuracy_score(y_true, y_pred)
            metrics_results['accuracy'] = acc
        elif metric == 'precision':
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics_results['precision'] = prec
        elif metric == 'recall':
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics_results['recall'] = rec
        elif metric == 'f1':
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics_results['f1'] = f1
        elif metric == 'roc_auc':
            if y_prob is not None and len(np.unique(y_true)) == 2:
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                metrics_results['roc_auc'] = roc_auc
            else:
                metrics_results['roc_auc'] = np.nan
        else:
            print(f"Warning: Unknown metric '{metric}'")
    return metrics_results

def main_clf(X, y,metrics=['accuracy', 'f1']):
    # Set data path and target column
    # data_path = 'datasets/Iris/IRIS.csv'      # Replace with your data file path
    # target_column = 'Survived'    # Replace with your target column name

    # # Load data
    # X, y = load_data(data_path, target_column='species')

    # Check for categorical features
    # if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
    #     raise ValueError("Categorical features detected. Please encode categorical variables before training.")

    n_features = X.shape[1]

    # Split data
    X_train, X_valid, y_train, y_valid = split_data(X, y)

    # Get classifiers and hyperparameter grids
    classifiers, params = get_classifiers(n_features)

    # Dictionary to store best models and their scores
    best_models = {}

    # For each classifier
    for clf, param in zip(classifiers, params):
        # Determine if it's a bagging classifier
        if isinstance(clf.named_steps['classifier'], BaggingClassifier):
            base_clf = clf.named_steps['classifier'].base_estimator
            base_clf_name = base_clf.__class__.__name__
            fs_name = clf.named_steps['feature_selection'].__class__.__name__
            clf_name = f"Bagging_{base_clf_name}_{fs_name}"
        else:
            clf_name = clf.named_steps['classifier'].__class__.__name__
            fs_name = clf.named_steps['feature_selection'].__class__.__name__
            clf_name = f"{clf_name}_{fs_name}"
        print(f"Training {clf_name}...")
        try:
            best_clf, best_score = hyperparameter_tuning(clf, param, X_train, y_train)
            y_pred = best_clf.predict(X_valid)
            # Handle classifiers that output probabilities
            if hasattr(best_clf, "predict_proba"):
                y_prob = best_clf.predict_proba(X_valid)
            elif hasattr(best_clf, "_predict_proba_lr"):
                y_prob = best_clf._predict_proba_lr(X_valid)
            else:
                y_prob = None
            metrics_results = calculate_metrics(y_valid, y_pred, y_prob, metrics)
            print(f"{clf_name} performance:")
            for metric_name, metric_value in metrics_results.items():
                print(f"  {metric_name}: {metric_value:.4f}")
            best_models[clf_name] = {
                'model': best_clf,
                'metrics': metrics_results,
                'predictions': y_pred,
                'probabilities': y_prob
            }
        except Exception as e:
            print(f"{clf_name} failed with error: {e}")

    # Check if any models succeeded
    if not best_models:
        print("No models were successfully trained.")
        return

    # Ensure sort_metric is valid
    sort_metric = metrics[0]
    valid_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
    if sort_metric not in valid_metrics:
        raise ValueError(f"Sort metric '{sort_metric}' is not recognized. Please choose from {valid_metrics}.")

    # Remove models that do not have the sort_metric
    sorted_models = []
    for clf_name, clf_info in best_models.items():
        if sort_metric in clf_info['metrics']:
            sorted_models.append((clf_name, clf_info))
        else:
            print(f"Model {clf_name} does not have the metric '{sort_metric}' and will be excluded from sorting.")

    # Sort models by the first metric specified
    sorted_models = sorted(sorted_models, key=lambda x: x[1]['metrics'][sort_metric], reverse=True)
    
    dict_to_return = {}

    # Print best models
    print("\nBest Models:")
    best = True
    for clf_name, clf_info in sorted_models[:10]:  # Show top 10 models
        d = {}
        print(f"{clf_name} performance:")
        for metric_name, metric_value in clf_info['metrics'].items():
            if best:
                d[metric_name] = metric_value
            print(f"  {metric_name}: {metric_value:.4f}")
        if best:
            dict_to_return[clf_name.split('_')[0]] = d
        best = False

    # Ensembling with top 3 models
    top_models = [info['model'] for name, info in sorted_models[:3]]
    estimators = []
    for i, model in enumerate(top_models):
        estimator_name = f"model_{i+1}"
        estimators.append((estimator_name, model))

    # Voting Classifier
    voting_clf = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_valid)
    if hasattr(voting_clf, "predict_proba"):
        y_prob = voting_clf.predict_proba(X_valid)
    elif hasattr(voting_clf, "_predict_proba_lr"):
        y_prob = voting_clf._predict_proba_lr(X_valid)
    else:
        y_prob = None
    metrics_results = calculate_metrics(y_valid, y_pred, y_prob, metrics)
    print(f"\nVoting Classifier performance:")
    dict_to_return['Voting Classifier'] = {metric_name : metric_value for metric_name, metric_value in metrics_results.items()}
    for metric_name, metric_value in metrics_results.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    best_models['VotingClassifier'] = {
        'model': voting_clf,
        'metrics': metrics_results,
        'predictions': y_pred,
        'probabilities': y_prob
    }

    # Stacking Classifier
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_valid)
    if hasattr(stacking_clf, "predict_proba"):
        y_prob = stacking_clf.predict_proba(X_valid)
    elif hasattr(stacking_clf, "_predict_proba_lr"):
        y_prob = stacking_clf._predict_proba_lr(X_valid)
    else:
        y_prob = None
    metrics_results = calculate_metrics(y_valid, y_pred, y_prob, metrics)
    print(f"\nStacking Classifier performance:")
    dict_to_return['Stacking Classifier'] = {metric_name : metric_value for metric_name, metric_value in metrics_results.items()}
    for metric_name, metric_value in metrics_results.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    best_models['StackingClassifier'] = {
        'model': stacking_clf,
        'metrics': metrics_results,
        'predictions': y_pred,
        'probabilities': y_prob
    }
    #print(f'#######################\n{best_models}')
    print('#' * 100)
    print(dict_to_return)
    return best_models, dict_to_return

if __name__ == "__main__":
    # Specify the performance metrics you want to use
    # Available metrics: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    #performance_metrics = ['accuracy', 'precision', 'recall', 'f1']
    # TODO 
    # 1. load data - X, y
    # 2. delete nonnumeric
    # 3. replace NAN
    # 4. give as arg below
    df = pd.read_csv('logs/all_dataframes/Churn_gpt-4o/all_hyp_1_exp.csv')
    y = df['Exited']
    X = df.drop(['Exited'], axis = 1)
    X.fillna(X.median(), inplace=True)
    X.dropna(axis=1, how='all', inplace=True)
    X = X.loc[:, X.nunique() > 1]
    results = main_clf(X,y)
    print('#############RESULTS')
    print(results[1])
