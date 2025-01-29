from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from datasets.data import DataReader
from collections import defaultdict
import pandas as pd



def encode_non_numerical_columns(df):
    df_encoded = df.copy()  
    label_encoders = {}  
    
    for column in df_encoded.select_dtypes(include=['object', 'category']).columns:
        encoder = LabelEncoder()
        df_encoded[column] = encoder.fit_transform(df_encoded[column])
        label_encoders[column] = encoder  
    
    return df_encoded



# car = DataReader('Cars')
# churn = DataReader('Churn')
#diamonds = DataReader('Diamonds')
smoking = DataReader('Smoking')

dataframes_clf = [smoking]
#dataframes_reg = [diamonds, car]


results = defaultdict(list)
for df in dataframes_clf:
    print(f"Start running for dataset {df.dataset_name}.....")
    X = encode_non_numerical_columns(df.train_input)
    y = df.train_labels

    class_counts = Counter(y)
    print(f"Class distribution for {df.dataset_name}: {class_counts}")

    if min(class_counts.values()) < 2:
        print(f"Skipping stratification for dataset {df.dataset_name} due to insufficient class samples.")
        stratify_param = None  
    else:
        stratify_param = y  

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    feature_selector = SelectKBest(score_func=f_classif)
    pipeline = Pipeline(steps=[("selector", feature_selector), ("classifier", xgb_model)])

    param_grid = {
        "selector__k": [7, 10, "all"],  # Adjust the number of features
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,  
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    classification_report_result = classification_report(y_test, y_pred)

    results['dataset'].append(df.dataset_name)
    results['accuracy'].append(accuracy_score(y_test, y_pred))
    results['f1-score'].append(f1_score(y_test, y_pred))

    print(f"RESULTS for dataset {df.dataset_name}")
    print("Best Parameters:", grid_search.best_params_)
    print("\nClassification Report:\n", classification_report_result)

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
    print("\nCross-validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())



# results_reg = defaultdict(list)
# for df in dataframes_reg:
#     print(f"Start running for dataset {df.dataset_name}.....")
#     X = encode_non_numerical_columns(df.train_input)
#     y = df.train_labels

#     if len(set(y)) < 2:
#         print(f"Skipping dataset {df.dataset_name} due to insufficient variance in target.")
#         continue

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     xgb_model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse")
#     feature_selector = SelectKBest(score_func=f_regression)
#     pipeline = Pipeline(steps=[("selector", feature_selector), ("regressor", xgb_model)])

#     param_grid = {
#         "selector__k": [7, 10, "all"],  # Adjust the number of features
#         "regressor__n_estimators": [50, 100, 200],
#         "regressor__max_depth": [3, 5, 7],
#         "regressor__learning_rate": [0.01, 0.1, 0.2],
#         "regressor__subsample": [0.8, 1.0],
#         "regressor__colsample_bytree": [0.8, 1.0],
#     }

#     grid_search = GridSearchCV(
#         estimator=pipeline,
#         param_grid=param_grid,
#         cv=5,  
#         scoring="neg_mean_squared_error",
#         verbose=1,
#         n_jobs=-1,
#     )

#     grid_search.fit(X_train, y_train)

#     best_model = grid_search.best_estimator_

#     y_pred = best_model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

    
    
#     results_reg['dataset'].append(df.dataset_name)
#     results_reg['mse'].append(mse)
#     results_reg['mae'].append(mae)
#     results_reg['r2'].append(r2)
#     print(f"RESULTS for dataset {df.dataset_name}")
#     print("\nEvaluation Metrics:")
#     print(f"Mean Squared Error (MSE): {mse}")
#     print(f"Mean Absolute Error (MAE): {mae}")
#     print(f"R^2 Score: {r2}")

#     cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
#     print("\nCross-validation MSE Scores:", -cv_scores)
#     print("Mean CV MSE:", -cv_scores.mean())
# print("DATAFRAME FOR CLASSIFICATION TASK:\n")
# print(pd.DataFrame(results))
# print("DATAFRAME FOR REGRESSION TASK:\n")
# print(pd.DataFrame(results_reg))