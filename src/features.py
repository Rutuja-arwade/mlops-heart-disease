import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        if self.enabled:
            X["hr_achieved_ratio"] = X["thalach"] / (220 - X["age"]).clip(lower=1)
            X["rate_pressure_index"] = X["thalach"] * X["trestbps"] / 1000.0
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.enabled:
            names += config.ENGINEERED_FEATURES
        return np.asarray(names, dtype=object)


def select_numeric(X):
    return [c for c in config.CONTINUOUS_FEATURES + config.ENGINEERED_FEATURES if c in X.columns]

def select_categorical(X):
    return [c for c in config.CATEGORICAL_FEATURES if c in X.columns]

def select_binary(X):
    return [c for c in config.BINARY_FEATURES if c in X.columns]


def build_preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer([("num", num, select_numeric), ("cat", cat, select_categorical), ("bin", binary, select_binary)],
                             remainder="drop", verbose_feature_names_out=False)


def build_pipeline(estimator, engineered=True):
    return Pipeline([("features", ClinicalFeatureBuilder(enabled=engineered)), ("preprocess", build_preprocessor()), ("model", estimator)])


def transformed_feature_names(pipeline):
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
