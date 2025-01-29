# Import necessary libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error, explained_variance_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, VarianceThreshold, RFE

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit,
                                  BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, HuberRegressor,
                                  TheilSenRegressor, RANSACRegressor)
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor,
                              ExtraTreesRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor)
from sklearn.svm import SVR, LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression

# You may need to install these libraries if not already installed
# pip install xgboost lightgbm catboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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
        try:
            y = pd.to_numeric(y)
        except ValueError:
            raise ValueError("Target variable is non-numeric and cannot be converted. Please encode it appropriately.")

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_regressors(n_features):
    """Define regressors, feature selection methods, and their hyperparameter grids."""
    regressors = []
    params = []

    # Preprocessing (scaling)
    scaler = StandardScaler()

    # List of base regressors and their parameters
    base_regressors = [
        #works on diabetes
    #     ('LinearRegression', LinearRegression(), {}, True),

    # #     #works on diabetes
    # #     ('Ridge', Ridge(), {
    # #         'regressor__alpha': np.logspace(-4, 4, 50),
    # #         'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
    # #     }, True),

    # #     #works on diabetes
    #     ('Lasso', Lasso(max_iter=10000), {
    #         'regressor__alpha': np.logspace(-4, 1, 50),
    #         'regressor__selection': ['cyclic', 'random'],
    #     }, True),

    #     #works on diabetes
    #     ('ElasticNet', ElasticNet(max_iter=10000), {
    #         'regressor__alpha': np.logspace(-4, 1, 50),
    #         'regressor__l1_ratio': np.linspace(0, 1, 20),
    #         'regressor__selection': ['cyclic', 'random'],
    #     }, True),

    #     #works on diabetes
    #     ('BayesianRidge', BayesianRidge(), {
    #         'regressor__alpha_1': np.logspace(-6, -1, 10),
    #         'regressor__alpha_2': np.logspace(-6, -1, 10),
    #         'regressor__lambda_1': np.logspace(-6, -1, 10),
    #         'regressor__lambda_2': np.logspace(-6, -1, 10)
    #     }, True),

    #     #works on diabetes
    #     ('ARDRegression', ARDRegression(), {
    #         'regressor__alpha_1': np.logspace(-6, -1, 10),
    #         'regressor__alpha_2': np.logspace(-6, -1, 10),
    #         'regressor__lambda_1': np.logspace(-6, -1, 10),
    #         'regressor__lambda_2': np.logspace(-6, -1, 10)
    #     }, True),

    #     #works on diabetes
    #     ('SGDRegressor', SGDRegressor(max_iter=1000, tol=1e-3), {
    #         'regressor__loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    #         'regressor__penalty': ['l2', 'l1', 'elasticnet'],
    #         'regressor__alpha': np.logspace(-6, -1, 10),
    #         'regressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    #     }, True),

    #     #works on diabetes
    #     ('PassiveAggressiveRegressor', PassiveAggressiveRegressor(max_iter=1000, tol=1e-3), {
    #         'regressor__C': np.logspace(-4, 1, 50),
    #         'regressor__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    #     }, True),

    #     #works on diabetes
    #     ('HuberRegressor', HuberRegressor(max_iter=1000), {
    #         'regressor__epsilon': np.linspace(1.1, 2.0, 10),
    #         'regressor__alpha': np.logspace(-6, -1, 10),
    #     }, True),

    #     #works on diabetes
        ('KNeighborsRegressor', KNeighborsRegressor(), {
            'regressor__n_neighbors': range(1, 31),
            'regressor__weights': ['uniform', 'distance'],
            'regressor__metric': ['euclidean', 'manhattan', 'minkowski'],
        }, True),

    #     #works on diabetes
        ('DecisionTreeRegressor', DecisionTreeRegressor(), {
            'regressor__criterion': ['squared_error'], #'friedman_mse', 'absolute_error', 'poisson'],
            'regressor__max_depth': [None] + list(range(2, 20)),
            'regressor__min_samples_split': range(2, 20),
            'regressor__min_samples_leaf': range(1, 20)
        }, False),

    #     #works on diabetes
        ('RandomForestRegressor', RandomForestRegressor(), {
            'regressor__n_estimators': [100, 200, 500],
            'regressor__criterion': ['squared_error'],#, 'absolute_error', 'friedman_mse'],
            'regressor__max_depth': [None] + list(range(2, 20, 3)),
            'regressor__min_samples_split': range(2, 20, 3),
            'regressor__min_samples_leaf': range(1, 20, 3)
        }, False),

    #     #works on diabetes
    #     ('GradientBoostingRegressor', GradientBoostingRegressor(), {
    #         'regressor__n_estimators': [100, 200, 500],
    #         'regressor__learning_rate': [0.001, 0.01, 0.1],
    #         'regressor__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    #         'regressor__max_depth': range(2, 20),
    #         'regressor__min_samples_split': range(2, 20),
    #         'regressor__min_samples_leaf': range(1, 20)
    #     }, False),

    #    # works on diabetes
    #     ('AdaBoostRegressor', AdaBoostRegressor(), {
    #         'regressor__n_estimators': [50, 100, 200],
    #         'regressor__learning_rate': [0.001, 0.01, 0.1, 1.0],
    #         'regressor__loss': ['linear', 'square', 'exponential']
    #     }, False),

        #does not work on diabetes
        # ('SVR', SVR(), [
        #     {
        #         'regressor__kernel': ['linear'],
        #         'regressor__C': np.logspace(-3, 3, 10),
        #         'regressor__epsilon': np.logspace(-4, 0, 10),
        #     },
        #     {
        #         'regressor__kernel': ['poly'],
        #         'regressor__C': np.logspace(-3, 3, 10),
        #         'regressor__epsilon': np.logspace(-4, 0, 10),
        #         'regressor__degree': [2, 3, 4],
        #         'regressor__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 5)),
        #     },
        #     {
        #         'regressor__kernel': ['rbf', 'sigmoid'],
        #         'regressor__C': np.logspace(-3, 3, 10),
        #         'regressor__epsilon': np.logspace(-4, 0, 10),
        #         'regressor__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 5)),
        #     },
        # ], True),

        # ('KernelRidge', KernelRidge(), [
        #     {
        #         'regressor__kernel': ['linear'],
        #         'regressor__alpha': np.logspace(-6, 0, 10),
        #     },
        #     {
        #         'regressor__kernel': ['polynomial'],
        #         'regressor__alpha': np.logspace(-6, 0, 10),
        #         'regressor__degree': [2, 3, 4],
        #         'regressor__gamma': np.logspace(-4, 0, 5),
        #     },
        #     {
        #         'regressor__kernel': ['rbf', 'sigmoid'],
        #         'regressor__alpha': np.logspace(-6, 0, 10),
        #         'regressor__gamma': np.logspace(-4, 0, 5),
        #     },
        # ], True),

        
        ('XGBRegressor', XGBRegressor(), {
            'regressor__n_estimators': [100, 200, 500],
            'regressor__learning_rate': [0.001, 0.01, 0.1],
            'regressor__max_depth': range(2, 10),
            'regressor__subsample': [0.5, 0.7, 0.9, 1.0],
            'regressor__colsample_bytree': [0.5, 0.7, 0.9, 1.0]
        }, False),

        #need to be fixed
        # ('LGBMRegressor', LGBMRegressor(), {
        #     'regressor__n_estimators': [100, 200, 500],
        #     'regressor__learning_rate': [0.001, 0.01, 0.1],
        #     'regressor__max_depth': range(2, 10),
        #     'regressor__num_leaves': [31, 50, 70],
        #     'regressor__subsample': [0.5, 0.7, 0.9, 1.0]
        # }, False),

        # ('LGBMRegressor', LGBMRegressor(n_jobs=-1), {
        #     'regressor__n_estimators': [5, 100, 200, 500],
        #     'regressor__learning_rate': [0.01, 0.1, 0.2],
        #     'regressor__max_depth': [3, 5, 7, 9],
        #     'regressor__num_leaves': [1, 2, 5], #31, 50],
        #     'regressor__subsample': [0.6, 0.8, 1.0],
        #     'regressor__min_child_samples': [5, 10, 20]
        # }, False),

        ('CatBoostRegressor', CatBoostRegressor(silent=True), {
            'regressor__iterations': [100, 200, 500],
            'regressor__learning_rate': [0.001, 0.01, 0.1],
            'regressor__depth': range(2, 10),
            'regressor__l2_leaf_reg': [1, 3, 5, 7, 9],
            'regressor__bagging_temperature': [0, 1, 2, 5, 10]
        }, False)
    ]

    # Define feature selectors and their parameters
    feature_selector_list = [
        ('variance_threshold', VarianceThreshold(), {
            'feature_selection__threshold': [0.0]#, 0.01, 0.1]
        }),
        # # ('select_k_best_f', SelectKBest(score_func=f_regression), {
        #     'feature_selection__k': [5, 10, 15, 20, 'all']
        # }),
        # ('select_k_best_mi', SelectKBest(score_func=mutual_info_regression), {
        #     'feature_selection__k': [5, 10, 15, 20, 'all']
        # }),
        # ('rfe', RFE(estimator=LinearRegression()), {
        #     'feature_selection__n_features_to_select': [5, 10, 15, 20, None]
        # })
    ]

    max_features = n_features

    # Adjust n_nonzero_coefs for models that require it
    # n_nonzero_coefs_range = list(range(1, min(max_features, 500)))

    # # Regressors that need n_nonzero_coefs adjusted
    # base_regressors_with_n_nonzero_coefs = [
    #     ('Lars', Lars(), {
    #         'regressor__n_nonzero_coefs': n_nonzero_coefs_range
    #     }, True),
    #     ('LassoLars', LassoLars(max_iter=1000), {
    #         'regressor__alpha': np.logspace(-4, 1, 50)
    #     }, True),
    #     ('OrthogonalMatchingPursuit', OrthogonalMatchingPursuit(), {
    #         'regressor__n_nonzero_coefs': n_nonzero_coefs_range
    #     }, True),
    # ]

    # base_regressors.extend(base_regressors_with_n_nonzero_coefs)

    for reg_name, reg, reg_params, needs_scaling in base_regressors:
        for fs_name, fs, fs_params in feature_selector_list:
            # Create pipeline
            steps = []
            if needs_scaling:
                steps.append(('scaler', scaler))
            steps.append(('feature_selection', fs))
            steps.append(('regressor', reg))
            pipe = Pipeline(steps)
            regressors.append(pipe)
            # Merge regressor parameters with feature selection parameters
            if isinstance(reg_params, list):  # For regressors with multiple parameter grids
                for grid in reg_params:
                    param_grid = grid.copy()
                    param_grid.update(fs_params)
                    params.append(param_grid)
            else:
                param_grid = reg_params.copy()
                param_grid.update(fs_params)
                params.append(param_grid)

            # # Bagging Regressor
            # bagging_reg = BaggingRegressor(base_estimator=reg, n_estimators=10, max_samples=0.8,
            #                                bootstrap=True, n_jobs=-1, random_state=42)
            # steps_bagging = []
            # if needs_scaling:
            #     steps_bagging.append(('scaler', scaler))
            # steps_bagging.append(('feature_selection', fs))
            # steps_bagging.append(('regressor', bagging_reg))
            # pipe_bagging = Pipeline(steps_bagging)
            # regressors.append(pipe_bagging)
            # # Bagging parameters
            # if isinstance(reg_params, list):  # For regressors with multiple parameter grids
            #     for grid in reg_params:
            #         param_grid_bagging = {}
            #         param_grid_bagging.update({f'regressor__base_estimator__{k.split("__")[1]}': v for k, v in grid.items()})
            #         param_grid_bagging.update(fs_params)
            #         params.append(param_grid_bagging)
            # else:
            #     param_grid_bagging = {}
            #     param_grid_bagging.update({f'regressor__base_estimator__{k.split("__")[1]}': v for k, v in reg_params.items()})
            #     param_grid_bagging.update(fs_params)
            #     params.append(param_grid_bagging)

    return regressors, params

def hyperparameter_tuning(reg, params, X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    rand_search = RandomizedSearchCV(reg, param_distributions=params, n_iter=2, cv=5,
                                     scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    rand_search.fit(X_train, y_train)
    return rand_search.best_estimator_, rand_search.best_score_

def calculate_metrics(y_true, y_pred, metrics_list):
    """Calculate specified performance metrics."""
    metrics_results = {}
    for metric in metrics_list:
        if metric == 'mse':
            mse = mean_squared_error(y_true, y_pred)
            metrics_results['mse'] = mse
        elif metric == 'mae':
            mae = mean_absolute_error(y_true, y_pred)
            metrics_results['mae'] = mae
        elif metric == 'r2':
            r2 = r2_score(y_true, y_pred)
            metrics_results['r2'] = r2
        elif metric == 'rmse':
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics_results['rmse'] = rmse
        elif metric == 'mape':
            mape = mean_absolute_percentage_error(y_true, y_pred)
            metrics_results['mape'] = mape
        elif metric == 'explained_variance':
            evs = explained_variance_score(y_true, y_pred)
            metrics_results['explained_variance'] = evs
        else:
            print(f"Warning: Unknown metric '{metric}'")
    return metrics_results

import pandas as pd

def impute_missing_values(df):
    """
    Imputes missing values in a DataFrame.
    - Numerical columns: Imputed with the median.
    - Non-numerical columns: Imputed with the mode.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with potential missing values.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    df_imputed = df.copy()
    
    for column in df_imputed.columns:
        if df_imputed[column].dtype in ['int64', 'float64']:
            # Impute numerical columns with median
            median_value = df_imputed[column].median()
            df_imputed[column].fillna(median_value, inplace=True)
        else:
            # Impute non-numerical columns with mode
            mode_value = df_imputed[column].mode()[0]
            df_imputed[column].fillna(mode_value, inplace=True)
    
    return df_imputed

import pandas as pd
import numpy as np
from datasets.data import DataReader
def remove_multicollinear_columns(df, threshold=0.9):
    """
    Removes columns with high multicollinearity based on a correlation threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Correlation threshold above which columns are considered multicollinear.

    Returns:
        pd.DataFrame: DataFrame with multicollinear columns removed.
    """
    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Select upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns with a correlation greater than the threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > abs(threshold))]

    # Drop the multicollinear columns
    return df.drop(columns=to_drop)
def encode_non_numerical_columns(df):
    df_encoded = df.copy()  
    label_encoders = {}  
    
    for column in df_encoded.select_dtypes(include=['object', 'category']).columns:
        encoder = LabelEncoder()
        df_encoded[column] = encoder.fit_transform(df_encoded[column])
        label_encoders[column] = encoder  
    
    return df_encoded
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_series(series):
    """
    Scales a Pandas Series using sklearn's StandardScaler.

    Parameters:
        series (pd.Series): Input Pandas Series to scale.

    Returns:
        pd.Series: Scaled Pandas Series.
    """
    scaler = StandardScaler()
    # Reshape the Series to a 2D array
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    # Return a Pandas Series
    return pd.Series(scaled_data.flatten(), index=series.index)


def remove_linearly_dependent_columns(df):
    """
    Removes linearly dependent columns from a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with linearly dependent columns removed.
    """
    # Ensure the DataFrame contains only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Perform Singular Value Decomposition (SVD)
    _, singular_values, vh = np.linalg.svd(df_numeric, full_matrices=False)
    
    # Identify columns to keep
    tolerance = 1e-10  # Adjust tolerance for linear dependence
    independent_columns = np.abs(singular_values) > tolerance

    # Keep only independent columns
    dependent_columns = [
        df_numeric.columns[i] 
        for i, is_independent in enumerate(independent_columns) 
        if not is_independent
    ]
    
    # Drop dependent columns
    df_cleaned = df.drop(columns=dependent_columns)
    
    return df_cleaned

import pandas as pd

def remove_zero_variance_columns(df):
    """
    Removes columns with zero variance from the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with zero-variance columns removed.
    """
    # Calculate variance for each column
    variances = df.var()
    
    # Identify columns with zero variance
    zero_variance_columns = variances[variances == 0].index

    # Drop the zero-variance columns
    return df.drop(columns=zero_variance_columns)


def main_reg(X,y,metrics=['mse', 'r2']):
    # Set data path and target column
   

    n_features = X.shape[1]

    # Split data
    X_train, X_valid, y_train, y_valid = split_data(X, y)

    # Get regressors and hyperparameter grids
    regressors, params = get_regressors(n_features)

    # Dictionary to store best models and their scores
    best_models = {}

    # For each regressor
    for reg, param in zip(regressors, params):
        # Determine if it's a bagging regressor
        if isinstance(reg.named_steps['regressor'], BaggingRegressor):
            base_reg = reg.named_steps['regressor'].base_estimator
            base_reg_name = base_reg.__class__.__name__
            fs_name = reg.named_steps['feature_selection'].__class__.__name__
            reg_name = f"Bagging_{base_reg_name}_{fs_name}"
        else:
            reg_name = reg.named_steps['regressor'].__class__.__name__
            fs_name = reg.named_steps['feature_selection'].__class__.__name__
            reg_name = f"{reg_name}_{fs_name}"
        print(f"Training {reg_name}...")
        try:
            best_reg, best_score = hyperparameter_tuning(reg, param, X_train, y_train)
            y_pred = best_reg.predict(X_valid)
            metrics_results = calculate_metrics(y_valid, y_pred, metrics)
            print(f"{reg_name} performance:")
            for metric_name, metric_value in metrics_results.items():
                print(f"  {metric_name}: {metric_value:.4f}")
            best_models[reg_name] = {
                'model': best_reg,
                'metrics': metrics_results,
                'predictions': y_pred
            }
        except Exception as e:
            print(f"{reg_name} failed with error: {e}")

    # Check if any models succeeded
    if not best_models:
        print("No models were successfully trained.")
        return

    # Ensure sort_metric is valid
    sort_metric = metrics[0]
    valid_metrics = {'mse', 'mae', 'r2', 'rmse', 'mape', 'explained_variance'}
    if sort_metric not in valid_metrics:
        raise ValueError(f"Sort metric '{sort_metric}' is not recognized. Please choose from {valid_metrics}.")

    # Remove models that do not have the sort_metric
    sorted_models = []
    for reg_name, reg_info in best_models.items():
        if sort_metric in reg_info['metrics']:
            sorted_models.append((reg_name, reg_info))
        else:
            print(f"Model {reg_name} does not have the metric '{sort_metric}' and will be excluded from sorting.")

    # Sort models by the first metric specified
    sorted_models = sorted(sorted_models, key=lambda x: x[1]['metrics'][sort_metric],
                           reverse=(sort_metric == 'r2' or sort_metric == 'explained_variance'))

    # Print best models
    dict_to_return = {}
    best = True
    print("\nBest Models:")
    for reg_name, reg_info in sorted_models[:10]:  # Show top 10 models
        print(f"{reg_name} performance:")
        for metric_name, metric_value in reg_info['metrics'].items():
            if best:
                d = {metric_name : metric_value for metric_name, metric_value in reg_info['metrics'].items()}
            print(f"  {metric_name}: {metric_value:.4f}")
        if best:
            dict_to_return[reg_name.split('_')[0]] = d
        best = False

    # Ensembling with top 3 models
    top_models = [info['model'] for name, info in sorted_models[:3]]
    estimators = []
    for i, model in enumerate(top_models):
        estimator_name = f"model_{i+1}"
        estimators.append((estimator_name, model))

    # Voting Regressor
    voting_reg = VotingRegressor(estimators=estimators, n_jobs=-1)
    voting_reg.fit(X_train, y_train)
    y_pred = voting_reg.predict(X_valid)
    metrics_results = calculate_metrics(y_valid, y_pred, metrics)
    print(f"\nVoting Regressor performance:")
    dict_to_return['Voting Regressor'] = {metric_name : metric_value for metric_name, metric_value in metrics_results.items()}
    for metric_name, metric_value in metrics_results.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    best_models['VotingRegressor'] = {
        'model': voting_reg,
        'metrics': metrics_results,
        'predictions': y_pred
    }

    # Stacking Regressor
    stacking_reg = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5, n_jobs=-1)
    stacking_reg.fit(X_train, y_train)
    y_pred = stacking_reg.predict(X_valid)
    metrics_results = calculate_metrics(y_valid, y_pred, metrics)
    print(f"\nStacking Regressor performance:")
    dict_to_return['Stacking Regressor'] = {metric_name : metric_value for metric_name, metric_value in metrics_results.items()}
    for metric_name, metric_value in metrics_results.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    best_models['StackingRegressor'] = {
        'model': stacking_reg,
        'metrics': metrics_results,
        'predictions': y_pred
    }

    return best_models, dict_to_return

if __name__ == "__main__":
     data_path = 'logs/all_dataframes/Diamonds_gpt-4o/all_hyp_1_exp.csv'      # Replace with your data file path
     target_column = 'price'    # Replace with your target column name
    #diamonds = DataReader('Diamonds')
    # Load data
     df = pd.read_csv(data_path)
     y = df[target_column]
     X = df.drop([target_column], axis = 1)
    #X = encode_non_numerical_columns(X)
     X = X.dropna(axis=1, how='all')
     X = impute_missing_values(X)
     X = remove_linearly_dependent_columns(X)
     X = remove_zero_variance_columns(X)
    #Check for categorical features
     if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
        raise ValueError("Categorical features detected. Please encode categorical variables before training.")
    # Specify the performance metrics you want to use
    # Available metrics: 'mse', 'mae', 'r2', 'rmse', 'mape', 'explained_variance'
     performance_metrics = ['r2', 'rmse']
     results = main_reg(X=X, y=y)
     print(results[1])
