import os
import sys
import numpy as np
import pandas as pd
import joblib
from log_file import setup_logging
from sklearn.preprocessing import PowerTransformer

logger = setup_logging("variable_transformation")

# ============================================================
# Step 1: Split numerical & categorical
# ============================================================

def split_numerical_categorical(X_train, X_test):
    """
    Split dataset into numerical and categorical features
    """
    try:
        num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

        X_train_num = X_train[num_cols].copy()
        X_test_num = X_test[num_cols].copy()

        X_train_cat = X_train[cat_cols].copy()
        X_test_cat = X_test[cat_cols].copy()

        logger.info("📊 BEFORE TRANSFORMATION SHAPES")
        logger.info(f"X_train_numerical: {X_train_num.shape}")
        logger.info(f"X_test_numerical: {X_test_num.shape}")
        logger.info(f"X_train_categorical: {X_train_cat.shape}")
        logger.info(f"X_test_categorical: {X_test_cat.shape}")
        logger.info(f"Numerical columns: {list(num_cols)}")
        logger.info(f"Categorical columns: {list(cat_cols)}")

        return X_train_num, X_test_num, X_train_cat, X_test_cat
    except Exception:
        logger.error("❌ Error in split_numerical_categorical", exc_info=True)
        raise

# ============================================================
# Step 2: Safe numerical transformations (NO SCALING)
# ============================================================

def iqr_transform(X_train, X_test):
    """IQR-based transformation (robust, no scaling object)"""
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1
    IQR.replace(0, 1, inplace=True)  # avoid division by zero
    return ((X_train - Q1) / IQR).values, ((X_test - Q1) / IQR).values


def log_transform(X_train, X_test):
    """Safe log transform (handles negatives)"""
    min_val = min(X_train.min().min(), X_test.min().min())
    shift = abs(min_val) + 1 if min_val <= 0 else 0
    return np.log1p(X_train + shift).values, np.log1p(X_test + shift).values


def exponential_transform(X_train, X_test):
    """Exponential transform with clipping"""
    X_train_clip = X_train.clip(-20, 20)
    X_test_clip = X_test.clip(-20, 20)
    return np.exp(X_train_clip).values, np.exp(X_test_clip).values


def boxcox_transform(X_train, X_test):
    """Box-Cox (only positive values)"""
    min_val = min(X_train.min().min(), X_test.min().min())
    shift = abs(min_val) + 1 if min_val <= 0 else 0

    pt = PowerTransformer(method="box-cox")
    X_train_bc = pt.fit_transform(X_train + shift)
    X_test_bc = pt.transform(X_test + shift)

    return X_train_bc, X_test_bc


def yeojohnson_transform(X_train, X_test):
    """Yeo-Johnson (handles zero & negative values safely)"""
    pt = PowerTransformer(method="yeo-johnson")
    X_train_yj = pt.fit_transform(X_train)
    X_test_yj = pt.transform(X_test)
    return X_train_yj, X_test_yj

# ============================================================
# Step 3: Apply all transformations (TRAIN ONLY for scoring)
# ============================================================

def apply_all_transformations(X_train_num, X_test_num):
    try:
        transformers = {
            "iqr": iqr_transform,
            "log": log_transform,
            "exponential": exponential_transform,
            "boxcox": boxcox_transform,
            "yeojohnson": yeojohnson_transform
        }

        train_results = {}
        for name, func in transformers.items():
            X_tr, _ = func(X_train_num, X_test_num)
            train_results[name] = pd.DataFrame(X_tr, columns=X_train_num.columns)

        return transformers, train_results
    except Exception:
        logger.error("❌ Error in apply_all_transformations", exc_info=True)
        raise

# ============================================================
# Step 4: Select best transformation (lowest skewness)
# ============================================================

def select_best_transformation(train_results):
    try:
        scores = {}
        for name, df in train_results.items():
            skew_val = df.skew().abs().mean()
            scores[name] = skew_val

        best_transform = min(scores, key=scores.get)

        logger.info(f"📈 Transformation skew scores: {scores}")
        logger.info(f"🏆 Best transformation selected: {best_transform}")

        return best_transform, scores
    except Exception:
        logger.error("❌ Error in select_best_transformation", exc_info=True)
        raise

# ============================================================
# Step 5: Apply best transformation (TRAIN + TEST)
# ============================================================

def apply_best_transformation(best_transform, transformers, X_train_num, X_test_num):
    try:
        logger.info("📊 APPLYING BEST TRANSFORMATION")
        logger.info(f"Transformation used: {best_transform}")

        logger.info(f"Before transformation X_train shape: {X_train_num.shape}")
        logger.info(f"Before transformation X_test shape: {X_test_num.shape}")

        X_train_final, X_test_final = transformers[best_transform](X_train_num, X_test_num)

        X_train_final = pd.DataFrame(X_train_final, columns=X_train_num.columns)
        X_test_final = pd.DataFrame(X_test_final, columns=X_test_num.columns)

        logger.info(f"After transformation X_train shape: {X_train_final.shape}")
        logger.info(f"After transformation X_test shape: {X_test_final.shape}")

        return X_train_final, X_test_final
    except Exception:
        logger.error("❌ Error in apply_best_transformation", exc_info=True)
        raise

# ============================================================
# Step 6: Categorical passthrough
# ============================================================

def encode_categorical(X_train_cat, X_test_cat):
    logger.info("📊 CATEGORICAL FEATURES (NO TRANSFORMATION)")
    logger.info(f"X_train_categorical shape: {X_train_cat.shape}")
    logger.info(f"X_test_categorical shape: {X_test_cat.shape}")
    return X_train_cat, X_test_cat

# ============================================================
# Step 7: Combine & save
# ============================================================

def create_and_save_final_dataset(
    X_train_num_final, X_test_num_final,
    X_train_cat_final, X_test_cat_final,
    best_transform
):
    try:
        logger.info("📊 FINAL DATASET SHAPES")
        logger.info(f"X_train_num: {X_train_num_final.shape}")
        logger.info(f"X_train_cat: {X_train_cat_final.shape}")
        logger.info(f"X_test_num: {X_test_num_final.shape}")
        logger.info(f"X_test_cat: {X_test_cat_final.shape}")

        X_train_final = pd.concat([X_train_num_final, X_train_cat_final], axis=1)
        X_test_final = pd.concat([X_test_num_final, X_test_cat_final], axis=1)

        os.makedirs("artifacts", exist_ok=True)
        X_train_final.to_csv("artifacts/X_train_transformed.csv", index=False)
        X_test_final.to_csv("artifacts/X_test_transformed.csv", index=False)
        joblib.dump(best_transform, "artifacts/best_transformation.pkl")

        logger.info(f"✅ Final X_train shape: {X_train_final.shape}")
        logger.info(f"✅ Final X_test shape: {X_test_final.shape}")
        logger.info("Artifacts saved successfully")

        return X_train_final, X_test_final
    except Exception:
        logger.error("❌ Error in create_and_save_final_dataset", exc_info=True)
        raise
