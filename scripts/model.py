"""Editable model definitions and evaluation helpers."""

from __future__ import annotations

import inspect

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from prepare import RANDOM_STATE


def _make_one_hot_encoder() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def build_preprocessor(
    X: pd.DataFrame,
    polynomial_numeric: bool = False,
) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if polynomial_numeric:
        numeric_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0,
    )


def build_logistic_regression(X_train: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def build_tuned_logistic_search_space(X_train: pd.DataFrame) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("clf", LogisticRegression(max_iter=5000)),
        ]
    )
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__class_weight": [None, "balanced"],
        "clf__solver": ["lbfgs"],
    }
    return pipeline, param_grid


def build_random_forest_search_space(X_train: pd.DataFrame) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    param_grid = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 10, 20, 40],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
        "clf__class_weight": [None, "balanced"],
    }
    return pipeline, param_grid


def build_extra_trees_search_space(X_train: pd.DataFrame) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("clf", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    param_grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
        "clf__class_weight": [None, "balanced"],
    }
    return pipeline, param_grid


def build_polynomial_logistic_search_space(X_train: pd.DataFrame) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train, polynomial_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }
    return pipeline, param_grid


def build_boosted_tree_search_space(X_train: pd.DataFrame) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth": [2, 3, 4],
        "clf__subsample": [0.8, 1.0],
    }
    return pipeline, param_grid


def build_hist_gradient_boosting_search_space(
    X_train: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_iter": [100, 200],
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__l2_regularization": [0.0, 0.1, 1.0],
    }
    return pipeline, param_grid
