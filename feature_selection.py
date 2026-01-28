import numpy as np
import pandas as pd
import logging

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from log_file import setup_logging

logger = setup_logging("feature_selection")


class NumericalFeatureSelector:
    def __init__(self, X_train, X_test, y_train):
        self.logger = logger
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train

        self.numerical_cols = None
        self.categorical_cols = None
        self.feature_votes = None

    def identify_columns(self):
        try:
            self.logger.info("Identifying numerical and categorical columns")

            self.numerical_cols = self.X_train.select_dtypes(
                include=np.number
            ).columns.tolist()

            self.categorical_cols = self.X_train.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            self.logger.info(f"Numerical Columns: {self.numerical_cols}")
            self.logger.info(f"Categorical Columns: {self.categorical_cols}")

        except Exception as e:
            self.logger.error("Error identifying columns", exc_info=True)
            raise

    def prepare_numerical_data(self):
        try:
            self.logger.info("Preparing numerical datasets")

            X_train_num = self.X_train[self.numerical_cols]
            X_test_num = self.X_test[self.numerical_cols]

            self.logger.info(f"X_train numerical shape: {X_train_num.shape}")
            self.logger.info(f"X_test numerical shape : {X_test_num.shape}")

            self.feature_votes = pd.Series(0, index=self.numerical_cols)

            return X_train_num, X_test_num

        except Exception as e:
            self.logger.error("Error preparing numerical data", exc_info=True)
            raise

    def constant_quasi_constant(self, X_train_num):
        try:
            self.logger.info("Applying Constant & Quasi-Constant Feature Selection")

            vt_const = VarianceThreshold(0.0)
            vt_const.fit(X_train_num)

            vt_quasi = VarianceThreshold(0.01)
            vt_quasi.fit(X_train_num)

            selected = X_train_num.columns[
                vt_quasi.get_support()
            ]

            self.feature_votes[selected] += 1

        except Exception as e:
            self.logger.error("Constant feature selection failed", exc_info=True)


    def variance_threshold(self, X_train_num):
        try:
            self.logger.info("Applying Variance Threshold")

            vt = VarianceThreshold(0.05)
            vt.fit(X_train_num)

            selected = X_train_num.columns[vt.get_support()]
            self.feature_votes[selected] += 1

        except Exception as e:
            self.logger.error("Variance Threshold failed", exc_info=True)


    def correlation_method(self, X_train_num):
        try:
            self.logger.info("Applying Correlation Method")

            corr = X_train_num.corr().abs()
            upper = corr.where(
                np.triu(np.ones(corr.shape), k=1).astype(bool)
            )

            drop_features = [
                col for col in upper.columns if any(upper[col] > 0.85)
            ]

            selected = list(set(X_train_num.columns) - set(drop_features))
            self.feature_votes[selected] += 1

        except Exception as e:
            self.logger.error("Correlation method failed", exc_info=True)

    def anova_test(self, X_train_num):
        try:
            self.logger.info("Applying ANOVA F-Test")

            anova = SelectKBest(score_func=f_classif, k=4)
            anova.fit(X_train_num, self.y_train)

            selected = X_train_num.columns[anova.get_support()]
            self.feature_votes[selected] += 1

        except Exception as e:
            self.logger.error("ANOVA test failed", exc_info=True)


    def lasso_method(self, X_train_num):
        try:
            self.logger.info("Applying LASSO (Standard Scaling)")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train_num)

            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=1000
            )
            model.fit(X_scaled, self.y_train)

            selected = X_train_num.columns[
                model.coef_[0] != 0
            ]

            self.feature_votes[selected] += 1

        except Exception as e:
            self.logger.error("LASSO failed", exc_info=True)


    def tree_methods(self, X_train_num):
        try:
            self.logger.info("Applying Tree-Based Feature Selection")

            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X_train_num, self.y_train)

            dt_selected = X_train_num.columns[
                dt.feature_importances_ > np.mean(dt.feature_importances_)
            ]
            self.feature_votes[dt_selected] += 1

            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train_num, self.y_train)

            rf_selected = X_train_num.columns[
                rf.feature_importances_ > np.mean(rf.feature_importances_)
            ]
            self.feature_votes[rf_selected] += 1

            # Robust scaling
            scaler = RobustScaler()
            X_robust = scaler.fit_transform(X_train_num)

            rf.fit(X_robust, self.y_train)
            robust_selected = X_train_num.columns[
                rf.feature_importances_ > np.mean(rf.feature_importances_)
            ]
            self.feature_votes[robust_selected] += 1

        except Exception as e:
            self.logger.error("Tree methods failed", exc_info=True)

    def select_best_features(self, X_train_num, X_test_num):
        try:
            self.logger.info("Selecting best features using voting")

            best_features = (
                self.feature_votes
                .sort_values(ascending=False)
                .head()
                .index
                .tolist()
            )

            self.logger.info(f"BEST  FEATURES: {best_features}")

            X_train_selected = X_train_num[best_features]
            X_test_selected = X_test_num[best_features]

            self.logger.info(f"Final X_train shape: {X_train_selected.shape}")
            self.logger.info(f"Final X_test shape : {X_test_selected.shape}")

            return X_train_selected, X_test_selected, best_features

        except Exception as e:
            self.logger.error("Final selection failed", exc_info=True)
            raise


    def run(self):
        self.identify_columns()
        X_train_num, X_test_num = self.prepare_numerical_data()

        self.constant_quasi_constant(X_train_num)
        self.variance_threshold(X_train_num)
        self.correlation_method(X_train_num)
        self.anova_test(X_train_num)
        self.lasso_method(X_train_num)
        self.tree_methods(X_train_num)

        return self.select_best_features(X_train_num, X_test_num)
