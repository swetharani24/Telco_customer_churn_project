import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import os, sys, joblib
from log_file import setup_logging

logger = setup_logging("missingvalues")


class MISSING_VALUE_TECHNIQUES:

    # ---------------- Numeric-only imputations ----------------
    @staticmethod
    def mean_imputation(X_train, X_test):
        X_tr, X_te = X_train.copy(), X_test.copy()
        cols = X_tr.select_dtypes(include=np.number).columns
        imp = SimpleImputer(strategy="mean")
        X_tr[cols] = imp.fit_transform(X_tr[cols])
        X_te[cols] = imp.transform(X_te[cols])
        return X_tr, X_te

    @staticmethod
    def median_imputation(X_train, X_test):
        X_tr, X_te = X_train.copy(), X_test.copy()
        cols = X_tr.select_dtypes(include=np.number).columns
        imp = SimpleImputer(strategy="median")
        X_tr[cols] = imp.fit_transform(X_tr[cols])
        X_te[cols] = imp.transform(X_te[cols])
        return X_tr, X_te

    @staticmethod
    def knn_imputation(X_train, X_test):
        X_tr, X_te = X_train.copy(), X_test.copy()
        cols = X_tr.select_dtypes(include=np.number).columns
        imp = KNNImputer(n_neighbors=5)
        X_tr[cols] = imp.fit_transform(X_tr[cols])
        X_te[cols] = imp.transform(X_te[cols])
        return X_tr, X_te

    # ---------------- Categorical-safe imputations ----------------
    @staticmethod
    def mode_imputation(X_train, X_test):
        imp = SimpleImputer(strategy="most_frequent")
        X_tr = pd.DataFrame(
            imp.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_te = pd.DataFrame(
            imp.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        return X_tr, X_te

    # ---------------- Pipeline runner (optional) ----------------
    @staticmethod
    def run_imputation_pipeline(df):
        """
        Optional standalone pipeline (NOT used in main.py)
        """
        logger.info("🔄 Running standalone imputation pipeline")

        # -------- Robust target handling --------
        y_raw = df["Churn"]

        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw
        else:
            y = (
                y_raw.astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "yes": 1, "no": 0,
                    "true": 1, "false": 0,
                    "1": 1, "0": 0
                })
            )

        X = df.drop(columns=["Churn", "customerID"], errors="ignore")

        invalid = y.isna()
        X, y = X.loc[~invalid], y.loc[~invalid]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("✅ Standalone imputation pipeline completed")
        return X_train, X_test, y_train, y_test
