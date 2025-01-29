from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    BaggingClassifier,
    StackingClassifier,
    BaggingRegressor,
    StackingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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


# Encode non-numerical columns
def encode_non_numerical_columns(df):
    df_encoded = df.copy()
    label_encoders = {}

    for column in df_encoded.select_dtypes(include=["object", "category"]).columns:
        encoder = LabelEncoder()
        df_encoded[column] = encoder.fit_transform(df_encoded[column])
        label_encoders[column] = encoder

    return df_encoded


# Read datasets
car = DataReader("Cars")
churn = DataReader("Churn")
diamonds = DataReader("Diamonds")
smoking = DataReader("Smoking")

dataframes_clf = [churn, smoking]
dataframes_reg = [diamonds, car]

# CLASSIFICATION TASK
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

    base_models_clf = [
        ("xgb", xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
        ("dt", DecisionTreeClassifier()),
        ("rf", RandomForestClassifier(random_state=42)),
        ("knn", KNeighborsClassifier()),
        ("lr", LogisticRegression(max_iter=1000)),
    ]

    bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)

    stacking_clf = StackingClassifier(estimators=base_models_clf, final_estimator=LogisticRegression())

    models_clf = {"Bagging": bagging_clf, "Stacking": stacking_clf}
    for model_name, model in models_clf.items():
        pipeline = Pipeline(steps=[("selector", SelectKBest(score_func=f_classif)), ("classifier", model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results_clf["dataset"].append(df.dataset_name)
        results_clf["model"].append(model_name)
        results_clf["accuracy"].append(accuracy_score(y_test, y_pred))
        results_clf["f1-score"].append(f1_score(y_test, y_pred, average="weighted"))

        print(f"RESULTS for dataset {df.dataset_name} using {model_name}")
        print("Classification Report:\n", classification_report(y_test, y_pred))


# REGRESSION TASK
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

    base_models_reg = [
        ("xgb", xgb.XGBRegressor(objective="reg:squarederror")),
        ("dt", DecisionTreeRegressor()),
        ("rf", RandomForestRegressor(random_state=42)),
        ("knn", KNeighborsRegressor()),
        ("lr", LinearRegression()),
    ]

    bagging_reg = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42)

    stacking_reg = StackingRegressor(estimators=base_models_reg, final_estimator=LinearRegression())

    models_reg = {"Bagging": bagging_reg, "Stacking": stacking_reg}
    for model_name, model in models_reg.items():
        pipeline = Pipeline(steps=[("selector", SelectKBest(score_func=f_regression)), ("regressor", model)])

        # Fit and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

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


print("DATAFRAME FOR CLASSIFICATION TASK:\n")
print(pd.DataFrame(results_clf))
print("DATAFRAME FOR REGRESSION TASK:\n")
print(pd.DataFrame(results_reg))
