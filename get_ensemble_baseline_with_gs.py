from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier, StackingClassifier, BaggingRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
import pandas as pd
from datasets.data import DataReader

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def encode_non_numerical_columns(df):
    df_encoded = df.copy()
    label_encoders = {}

    for column in df_encoded.select_dtypes(include=["object", "category"]).columns:
        encoder = LabelEncoder()
        df_encoded[column] = encoder.fit_transform(df_encoded[column])
        label_encoders[column] = encoder

    return df_encoded

# Hyperparameter grids for classification models
param_grids_clf = {
    "xgb": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
    },
    "dt": {"max_depth": [3, 5, None]},
    "rf": {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
    "knn": {"n_neighbors": [3, 5, 7]},
    "lr": {"C": [0.1, 1, 10], "max_iter": [100, 1000]},
}

# Hyperparameter grids for regression models
param_grids_reg = {
    "xgb": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
    },
    "dt": {"max_depth": [3, 5, None]},
    "rf": {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
    "knn": {"n_neighbors": [3, 5, 7]},
    "lr": {},
}

# Models initialization
def get_models_clf():
    return {
        "xgb": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "dt": DecisionTreeClassifier(),
        "rf": RandomForestClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
        "lr": LogisticRegression(max_iter=1000),
    }

def get_models_reg():
    return {
        "xgb": xgb.XGBRegressor(objective="reg:squarederror"),
        "dt": DecisionTreeRegressor(),
        "rf": RandomForestRegressor(random_state=42),
        "knn": KNeighborsRegressor(),
        "lr": LinearRegression(),
    }

# Function for grid search
def perform_grid_search(model, param_grid, X_train, y_train, task="classification"):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy" if task == "classification" else "neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
# Read datasets
car = DataReader("Cars")
churn = DataReader("Churn")
diamonds = DataReader("Diamonds")
smoking = DataReader("Smoking")

dataframes_clf = [churn, smoking]
dataframes_reg = [diamonds, car]

# Classification task
results_clf = defaultdict(list)
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

    # Tune each base model
    tuned_models_clf = {}
    for name, model in get_models_clf().items():
        print(f"Tuning {name} for dataset {df.dataset_name}...")
        tuned_model = perform_grid_search(model, param_grids_clf[name], X_train, y_train, task="classification")
        tuned_models_clf[name] = tuned_model

    # Bagging
    bagging_clf = BaggingClassifier(base_estimator=tuned_models_clf["dt"], n_estimators=10, random_state=42)

    # Stacking
    stacking_clf = StackingClassifier(estimators=list(tuned_models_clf.items()), final_estimator=LogisticRegression())

    # Evaluate Bagging and Stacking
    models_clf = {"Bagging": bagging_clf, "Stacking": stacking_clf}
    for model_name, model in models_clf.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results_clf["dataset"].append(df.dataset_name)
        results_clf["model"].append(model_name)
        results_clf["accuracy"].append(accuracy_score(y_test, y_pred))
        results_clf["f1-score"].append(f1_score(y_test, y_pred, average="weighted"))

        print(f"RESULTS for dataset {df.dataset_name} using {model_name}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

# Regression task
results_reg = defaultdict(list)
for df in dataframes_reg:
    print(f"Start running for dataset {df.dataset_name}.....")
    X = encode_non_numerical_columns(df.train_input)
    y = df.train_labels

    if len(set(y)) < 2:
        print(f"Skipping dataset {df.dataset_name} due to insufficient variance in target.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Tune each base model
    tuned_models_reg = {}
    for name, model in get_models_reg().items():
        print(f"Tuning {name} for dataset {df.dataset_name}...")
        tuned_model = perform_grid_search(model, param_grids_reg[name], X_train, y_train, task="regression")
        tuned_models_reg[name] = tuned_model

    # Bagging
    bagging_reg = BaggingRegressor(base_estimator=tuned_models_reg["dt"], n_estimators=10, random_state=42)

    # Stacking
    stacking_reg = StackingRegressor(estimators=list(tuned_models_reg.items()), final_estimator=LinearRegression())

    # Evaluate Bagging and Stacking
    models_reg = {"Bagging": bagging_reg, "Stacking": stacking_reg}
    for model_name, model in models_reg.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results_reg["dataset"].append(df.dataset_name)
        results_reg["model"].append(model_name)
        results_reg["mse"].append(mse)
        results_reg["mae"].append(mae)
        results_reg["r2"].append(r2)

        print(f"RESULTS for dataset {df.dataset_name} using {model_name}")
        print("\nEvaluation Metrics:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R^2 Score: {r2}")

# Display results
print("DATAFRAME FOR CLASSIFICATION TASK:\n")
print(pd.DataFrame(results_clf))
print("DATAFRAME FOR REGRESSION TASK:\n")
print(pd.DataFrame(results_reg))
