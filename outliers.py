import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
from log_file import setup_logging
logger = setup_logging("outliers")

class OUTLIER_HANDLING():
    plot_dir = "plot_outliers"
    os.makedirs(plot_dir, exist_ok=True)

    # =====================
    # Existing Techniques
    # =====================
    @staticmethod
    def iqr_method(X_train, X_test):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        numeric_cols = X_tr.select_dtypes(include='number').columns
        Q1 = X_tr[numeric_cols].quantile(0.25)
        Q3 = X_tr[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((X_tr[numeric_cols] < (Q1 - 1.5 * IQR)) |
                 (X_tr[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        return X_tr.loc[mask], X_te

    @staticmethod
    def zscore_method(X_train, X_test):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        numeric_cols = X_tr.select_dtypes(include='number').columns
        z_scores = np.abs(stats.zscore(X_tr[numeric_cols]))
        mask = (z_scores < 3).all(axis=1)
        return X_tr.loc[mask], X_te

    @staticmethod
    def winsorization(X_train, X_test):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        numeric_cols = X_tr.select_dtypes(include='number').columns
        lower = X_tr[numeric_cols].quantile(0.05)
        upper = X_tr[numeric_cols].quantile(0.95)
        X_tr[numeric_cols] = X_tr[numeric_cols].clip(lower=lower, upper=upper, axis=1)
        X_te[numeric_cols] = X_te[numeric_cols].clip(lower=lower, upper=upper, axis=1)
        return X_tr, X_te

    @staticmethod
    def clipping(X_train, X_test):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        numeric_cols = X_tr.select_dtypes(include='number').columns
        for col in numeric_cols:
            lower = X_tr[col].mean() - 3 * X_tr[col].std()
            upper = X_tr[col].mean() + 3 * X_tr[col].std()
            X_tr[col] = X_tr[col].clip(lower, upper)
            X_te[col] = X_te[col].clip(lower, upper)
        return X_tr, X_te

    @staticmethod
    def log_outlier(X_train, X_test):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        numeric_cols = X_tr.select_dtypes(include='number').columns
        min_val = min(X_tr[numeric_cols].min().min(), X_te[numeric_cols].min().min())
        if min_val < 0:
            shift = abs(min_val) + 1
            X_tr[numeric_cols] += shift
            X_te[numeric_cols] += shift
        X_tr[numeric_cols] = np.log1p(X_tr[numeric_cols])
        X_te[numeric_cols] = np.log1p(X_te[numeric_cols])
        return X_tr, X_te

    @staticmethod
    def no_outlier(X_train, X_test):
        return X_train, X_test

    # =====================
    # Count Outliers
    # =====================
    @staticmethod
    def count_outliers(df):
        numeric_cols = df.select_dtypes(include='number').columns
        counts = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            counts[col] = ((df[col] < lower) | (df[col] > upper)).sum()
        return counts

    # =====================
    # Save Boxplots
    # =====================
    @staticmethod
    @staticmethod
    def save_boxplots(X_train, X_test, technique_name):
        try:
            # Absolute path (VERY IMPORTANT)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            plot_dir = os.path.join(base_dir, ploy_outliers.plot_dir)
            os.makedirs(plot_dir, exist_ok=True)

            numeric_cols = X_train.select_dtypes(include='number').columns

            if len(numeric_cols) == 0:
                logger.warning(f"No numeric columns found for {technique_name}")
                return

            for col in numeric_cols:
                try:
                    # Skip constant columns
                    if X_train[col].nunique() <= 1:
                        logger.warning(f"Skipping {col} ({technique_name}) → constant column")
                        continue

                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    sns.boxplot(x=X_train[col], ax=axes[0])
                    axes[0].set_title(f"Train - {col} ({technique_name})")

                    sns.boxplot(x=X_test[col], ax=axes[1])
                    axes[1].set_title(f"Test - {col} ({technique_name})")

                    filename = os.path.join(plot_dir, f"{technique_name}_{col}.png")
                    fig.savefig(filename, bbox_inches="tight")
                    plt.close(fig)

                    logger.info(f"✅ Outlier plot saved → {filename}")

                except Exception as col_err:
                    logger.error(
                        f"❌ Failed to save plot for {col} ({technique_name})",
                        exc_info=True
                    )

        except Exception as e:
            logger.error(
                f"❌ Boxplot saving failed for technique: {technique_name}",
                exc_info=True
            )

    # =====================
    # Apply All Techniques & Count Outliers
    # =====================
    @staticmethod
    def apply_all_techniques(X_train, X_test):
        techniques = ['iqr_method','zscore_method','winsorization','clipping','log_outlier','no_outlier']
        results = {}
        original_counts = OUTLIER_HANDLING.count_outliers(X_train)
        logger.info(f"Original Outliers per column: {original_counts}")

        for tech in techniques:
            func = getattr(OUTLIER_HANDLING, tech)
            X_tr, X_te = func(X_train, X_test)
            results[tech] = (X_tr, X_te)

            # Count handled outliers
            new_counts = OUTLIER_HANDLING.count_outliers(X_tr)
            handled_counts = {col: original_counts[col]-new_counts[col] for col in original_counts}
            total_handled = sum(handled_counts.values())
            logger.info(f"{tech} handled outliers: {handled_counts} | Total handled: {total_handled}")

            # Save boxplots
            OUTLIER_HANDLING.save_boxplots(X_tr, X_te, tech)

        return results
