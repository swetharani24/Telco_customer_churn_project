import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from log_file import setup_logging
logger = setup_logging("train_models")


class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()

        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_auc = 0.0

    # ---------------- CLEAN DATA ----------------
    def clean_data(self):
        try:
            X_train = (
                self.X_train
                .select_dtypes(exclude=["object"])
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            X_test = (
                self.X_test
                .select_dtypes(exclude=["object"])
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            return X_train, X_test

        except Exception as e:
            logger.error("❌ Error during data cleaning", exc_info=True)
            raise e

    # ---------------- EVALUATE MODEL ----------------
    def evaluate_model(self, model, model_name):
        try:
            X_train, X_test = self.clean_data()

            model.fit(X_train, self.y_train.values.ravel())

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_prob)

            self.results[model_name] = {
                "model": model,
                "accuracy": acc,
                "auc": auc
            }

            logger.info(f"\n===== {model_name} =====")
            logger.info(f"Accuracy : {acc:.4f}")
            logger.info(f"ROC-AUC  : {auc:.4f}")
            logger.info("\n" + classification_report(self.y_test, y_pred))

            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")

        except Exception as e:
            logger.error(f"❌ Error evaluating {model_name}", exc_info=True)

    # ---------------- LOGISTIC REGRESSION ----------------
    def train_logistic_regression(self):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ])
        self.evaluate_model(model, "Logistic Regression")

    # ---------------- KNN ----------------
    def train_knn(self):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
        self.evaluate_model(model, "KNN")

    # ---------------- NAIVE BAYES ----------------
    def train_naive_bayes(self):
        model = GaussianNB()
        self.evaluate_model(model, "Naive Bayes")

    # ---------------- DECISION TREE ----------------
    def train_decision_tree(self):
        model = DecisionTreeClassifier(random_state=42)
        self.evaluate_model(model, "Decision Tree")

    # ---------------- RANDOM FOREST ----------------
    def train_random_forest(self):
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        self.evaluate_model(model, "Random Forest")

    # ---------------- ADABOOST ----------------
    def train_adaboost(self):
        model = AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=42
        )
        self.evaluate_model(model, "AdaBoost")

    # ---------------- XGBOOST ----------------
    def train_xgboost(self):
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            use_label_encoder=False
        )
        self.evaluate_model(model, "XGBoost")

    # ---------------- TRAIN ALL ----------------
    def train_all_models(self):
        logger.info("🚀 Training all models started...")
        plt.figure(figsize=(10, 8))

        self.train_logistic_regression()
        self.train_knn()
        self.train_naive_bayes()
        self.train_decision_tree()
        self.train_random_forest()
        self.train_adaboost()
        self.train_xgboost()

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend()
        plt.show()

        self.select_best_model()

    # ---------------- SELECT BEST MODEL ----------------
    def select_best_model(self):
        try:
            self.best_model_name = max(
                self.results,
                key=lambda x: self.results[x]["auc"]
            )

            self.best_model = self.results[self.best_model_name]["model"]
            self.best_auc = self.results[self.best_model_name]["auc"]

            logger.info(
                f"🏆 Best Model: {self.best_model_name} | AUC={self.best_auc:.4f}"
            )

        except Exception as e:
            logger.error("❌ Error selecting best model", exc_info=True)

    # ---------------- SAVE MODEL & SCALER ----------------
    def save_artifacts(
        self,
        model_path="artifacts/best_model.pkl",
        scaler_path="artifacts/best_scaler.pkl"
    ):
        try:
            joblib.dump(self.best_model, model_path)
            logger.info(f"💾 Best model saved: {model_path}")

            # Save scaler ONLY if pipeline has scaler
            if isinstance(self.best_model, Pipeline) and "scaler" in self.best_model.named_steps:
                scaler = self.best_model.named_steps["scaler"]
                joblib.dump(scaler, scaler_path)
                logger.info(f"💾 Best scaler saved: {scaler_path}")
            else:
                logger.info("ℹ️ Best model does not use scaler")

        except Exception as e:
            logger.error("❌ Error saving artifacts", exc_info=True)

    # ---------------- RUN FULL PIPELINE ----------------
    def run(self):
        self.train_all_models()
        self.save_artifacts()
