import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from log_file import setup_logging

logger = setup_logging("feature_encoding")


class FeatureEncoding:
    """
    Pipeline-safe Feature Encoding for categorical columns:
    - Handles binary and multi-class categorical features
    - Safely manages unseen categories in test set
    - Drops identifier columns if specified
    - Saves encoders as artifacts
    """

    def __init__(self, X_train, X_test, y_train=None, artifact_dir="artifacts", drop_cols=None):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train
        self.artifact_dir = artifact_dir
        self.drop_cols = drop_cols if drop_cols else []

        # Drop identifier columns
        self.X_train.drop(columns=self.drop_cols, errors="ignore", inplace=True)
        self.X_test.drop(columns=self.drop_cols, errors="ignore", inplace=True)

        # Identify categorical columns
        self.cat_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        # Encoder storage
        self.label_encoders = {}
        self.freq_encoders = {}

        os.makedirs(self.artifact_dir, exist_ok=True)

        logger.info("📊 FEATURE ENCODING INITIALIZED")
        logger.info(f"Dropped columns: {self.drop_cols}")
        logger.info(f"Categorical columns ({len(self.cat_cols)}): {self.cat_cols}")
        logger.info(f"X_train shape before encoding: {self.X_train.shape}")
        logger.info(f"X_test shape before encoding: {self.X_test.shape}")

    def encode(self):
        try:
            X_train_enc = self.X_train.copy()
            X_test_enc = self.X_test.copy()

            for col in self.cat_cols:
                unique_vals = self.X_train[col].nunique()

                # -----------------------------
                # Binary categorical → Label Encoding
                # -----------------------------
                if unique_vals == 2:
                    le = LabelEncoder()
                    X_train_enc[col] = le.fit_transform(self.X_train[col])
                    # Convert X_test to object for safety
                    X_test_col_safe = X_test_enc[col].astype(object)
                    X_test_enc[col] = X_test_col_safe.map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                    self.label_encoders[col] = le
                    logger.info(f"🔢 Label encoded column: {col}")

                # -----------------------------
                # Multi-class categorical → Frequency Encoding
                # -----------------------------
                else:
                    freq_map = self.X_train[col].value_counts(normalize=True)
                    X_train_enc[col] = self.X_train[col].map(freq_map)
                    # Convert to object first to avoid Categorical issues
                    X_test_enc[col] = X_test_enc[col].astype(object).map(freq_map).fillna(0)
                    self.freq_encoders[col] = freq_map
                    logger.info(f"📈 Frequency encoded column: {col}")

            # -----------------------------
            # Save encoders as artifacts
            # -----------------------------
            if self.label_encoders:
                joblib.dump(self.label_encoders, os.path.join(self.artifact_dir, "label_encoders.pkl"))
            if self.freq_encoders:
                joblib.dump(self.freq_encoders, os.path.join(self.artifact_dir, "frequency_encoders.pkl"))

            logger.info(f"📐 Shapes after encoding | X_train: {X_train_enc.shape}, X_test: {X_test_enc.shape}")

            return X_train_enc, X_test_enc, {
                "label_encoders": self.label_encoders,
                "frequency_encoders": self.freq_encoders
            }

        except Exception:
            logger.exception("❌ Feature Encoding failed")
            raise
