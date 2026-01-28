import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from collections import Counter
import logging
from log_file import setup_logging
logger = setup_logging("data_balancing")
logger.setLevel("INFO")


class DataBalancing:
    def __init__(self, X_train_num, y_train, X_train_cat, X_test_num, X_test_cat):
        self.X_train_num = X_train_num.reset_index(drop=True)
        self.X_train_cat = X_train_cat.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)

        self.X_test_num = X_test_num.reset_index(drop=True)
        self.X_test_cat = X_test_cat.reset_index(drop=True)

        self.X_train = None
        self.X_test = None

    # ------------------------
    # Resampling methods
    # ------------------------
    def _apply_sampler(self, sampler, name):
        logger.info(f"🔁 Applying {name}")
        X_res, y_res = sampler.fit_resample(self.X_train_num, self.y_train)

        # IMPORTANT: repeat categorical rows
        cat_resampled = self.X_train_cat.iloc[
            sampler.sample_indices_
        ].reset_index(drop=True)

        X_final = pd.concat(
            [X_res.reset_index(drop=True), cat_resampled],
            axis=1
        )

        return X_final, y_res.reset_index(drop=True)

    def select_best_method(self):
        logger.info("⚖️ Selecting best data balancing method")
        logger.info(f"Before balancing: {Counter(self.y_train)}")

        methods = {
            "SMOTE": SMOTE(random_state=42),
            "RandomOverSampler": RandomOverSampler(random_state=42),
            "RandomUnderSampler": RandomUnderSampler(random_state=42),
        }

        best_method = None
        best_X, best_y = None, None
        max_rows = 0

        for name, sampler in methods.items():
            try:
                X_res, y_res = self._apply_sampler(sampler, name)
                if len(X_res) > max_rows:
                    max_rows = len(X_res)
                    best_method = name
                    best_X, best_y = X_res, y_res
            except Exception:
                logger.exception(f"❌ {name} failed")

        if best_X is None:
            raise ValueError("❌ All balancing methods failed")

        logger.info(f"After balancing ({best_method}): {Counter(best_y)}")

        self.X_train = best_X
        self.y_train = best_y

        # Test data (NO balancing)
        self.X_test = pd.concat(
            [self.X_test_num, self.X_test_cat],
            axis=1
        )

        logger.info(f"Final X_train shape: {self.X_train.shape}")
        logger.info(f"Final X_test shape : {self.X_test.shape}")

        return self.X_train, self.y_train, best_method

    # ------------------------
    # Scaling numeric columns
    # ------------------------
    def scale_numeric(self):
        logger.info("📏 Scaling numeric features")

        scaler = StandardScaler()
        num_cols = self.X_train_num.columns

        self.X_train[num_cols] = scaler.fit_transform(self.X_train[num_cols])
        self.X_test[num_cols] = scaler.transform(self.X_test[num_cols])

        logger.info("✅ Scaling completed")
