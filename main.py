import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Pipeline modules
from visualisation import generate_all_plots
from datacleaning import DataCleaning
from missingvalues import MISSING_VALUE_TECHNIQUES
from outliers import OUTLIER_HANDLING
from feature_encoding import FeatureEncoding
from variable_transformation import (
    split_numerical_categorical,
    apply_all_transformations,
    select_best_transformation,
    apply_best_transformation,
    encode_categorical,
    create_and_save_final_dataset
)
from feature_selection import NumericalFeatureSelector
from data_balancing import DataBalancing
from train_models import ModelTrainer
from log_file import setup_logging

# --------------------------
logger = setup_logging("main")
logger.setLevel("INFO")

ARTIFACT_PATH = "artifacts"
os.makedirs(ARTIFACT_PATH, exist_ok=True)


class ChurnPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

        # Train/Test datasets
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.X_train_final = self.X_test_final = None
        self.y_train_final = self.y_test_final = None

    # --------------------------
    # 1️⃣ Load Data
    # --------------------------
    def load_data(self):
        logger.info("📥 Loading dataset")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Dataset loaded | Shape: {self.df.shape}")

    # --------------------------
    # 2️⃣ Visualisation
    # --------------------------
    def visualise_data(self):
        logger.info("📊 Running visualisation")
        generate_all_plots(self.df)

    # --------------------------
    # 3️⃣ Data Cleaning
    # --------------------------
    def clean_data(self):
        logger.info("🧹 Data Cleaning")
        cleaner = DataCleaning(self.df, ARTIFACT_PATH)
        self.df = cleaner.run_data_cleaning()

    # --------------------------
    # 4️⃣ Missing Values & Train/Test Split
    # --------------------------
    def handle_missing_values(self):
        logger.info("💧 Missing value handling")

        # Target mapping
        y_raw = self.df["Churn"]
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw
        else:
            y = y_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

        X = self.df.drop(columns=["Churn", "customerID"], errors="ignore")
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train/Test shapes | X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")

        # Impute numeric columns
        self.X_train, self.X_test = MISSING_VALUE_TECHNIQUES.median_imputation(self.X_train, self.X_test)
        logger.info("✅ Missing value imputation done")

    # --------------------------
    # 5️⃣ Outlier Handling
    # --------------------------
    def handle_outliers(self):
        logger.info("🔄 Outlier handling")
        results = OUTLIER_HANDLING.apply_all_techniques(self.X_train, self.X_test)
        self.X_train, self.X_test = results["winsorization"]
        logger.info(f"✅ Outlier handling completed | X_train shape: {self.X_train.shape}")



    # --------------------------
    # 6️⃣ Feature Encoding (safe NaN handling)
        # --------------------------
    def fill_categorical_na(self):
        logger.info("🧹 Filling missing values in categorical columns")

        for df in [self.X_train, self.X_test]:
            cat_cols = df.select_dtypes(exclude="number").columns
            for col in cat_cols:
                df[col] = df[col].astype("object")
                df[col].fillna("Missing", inplace=True)

        logger.info("✅ Categorical NaNs handled safely")

    # --------------------------
    # 6️⃣ Feature Encoding
    # --------------------------
    def feature_encoding(self):
        logger.info("🔢 Feature Encoding")
        encoder = FeatureEncoding(
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            drop_cols=["customerID"],
            artifact_dir=ARTIFACT_PATH
        )
        self.X_train_final, self.X_test_final, _ = encoder.encode()

        # Reset indexes to keep alignment with y
        self.X_train_final.reset_index(drop=True, inplace=True)
        self.X_test_final.reset_index(drop=True, inplace=True)
        self.y_train_final = self.y_train.reset_index(drop=True)
        self.y_test_final = self.y_test.reset_index(drop=True)

        # Safely fill categorical NaNs



        logger.info("✅ Feature encoding completed")

    # --------------------------
    # 7️⃣ Variable Transformation + Safe Categorical NA handling
    # --------------------------
    def variable_transformation(self):

        logger.info("🔄 Variable Transformation Started")

        # 1️⃣ Split numeric and categorical
        X_train_num, X_test_num, X_train_cat, X_test_cat = split_numerical_categorical(
            self.X_train_final, self.X_test_final
        )

        # 2️⃣ Apply numeric transformations
        transformers, train_results = apply_all_transformations(X_train_num, X_test_num)
        best_transform, _ = select_best_transformation(train_results)
        X_train_num_final, X_test_num_final = apply_best_transformation(
            best_transform, transformers, X_train_num, X_test_num
        )

        # 3️⃣ Pass categorical features as-is
        X_train_cat_final, X_test_cat_final = encode_categorical(X_train_cat, X_test_cat)

        # 4️⃣ Combine numeric + categorical
        self.X_train_final, self.X_test_final = create_and_save_final_dataset(
            X_train_num_final, X_test_num_final,
            X_train_cat_final, X_test_cat_final,
            best_transform
        )



        logger.info("✅ Variable Transformation Completed")

    # --------------------------
    # 8️⃣ Feature Selection (SAFE)
    # --------------------------
    def feature_selection(self):
        logger.info("🔄 Feature Selection Started")

        selector = NumericalFeatureSelector(
            X_train=self.X_train_final,
            X_test=self.X_test_final,
            y_train=self.y_train_final
        )

        X_train_num, X_test_num, selected_features = selector.run()

        # keep categorical columns
        cat_cols = self.X_train_final.select_dtypes(exclude="number").columns.tolist()

        self.X_train_final = pd.concat(
            [X_train_num, self.X_train_final[cat_cols]], axis=1
        )
        self.X_test_final = pd.concat(
            [X_test_num, self.X_test_final[cat_cols]], axis=1
        )

        # safety reset
        self.X_train_final.reset_index(drop=True, inplace=True)
        self.X_test_final.reset_index(drop=True, inplace=True)
        self.y_train_final.reset_index(drop=True, inplace=True)
        self.y_test_final.reset_index(drop=True, inplace=True)

        logger.info(
            f"✅ Feature selection completed | "
            f"X_train: {self.X_train_final.shape} | "
            f"Selected numerical features: {len(selected_features)}"
        )

    # --------------------------
    # 9️⃣ Data Balancing + Scaling
    # --------------------------
    def balance_data(self):
        logger.info("⚖️ Data Balancing")
        X_train_num = self.X_train_final.select_dtypes(include="number")
        X_train_cat = self.X_train_final.select_dtypes(exclude="number")
        X_test_num = self.X_test_final.select_dtypes(include="number")
        X_test_cat = self.X_test_final.select_dtypes(exclude="number")

        balancer = DataBalancing(
            X_train_num, self.y_train_final, X_train_cat, X_test_num, X_test_cat
        )
        X_train_bal, _, _ = balancer.select_best_method()
        balancer.scale_numeric()

        # Update pipeline datasets
        self.X_train_final = balancer.X_train.reset_index(drop=True)
        self.X_test_final = balancer.X_test.reset_index(drop=True)
        self.y_train_final = balancer.y_train.reset_index(drop=True)
        # y_test_final remains the same
        self.y_test_final = self.y_test_final.reset_index(drop=True)
        logger.info(f"✅ Data balancing completed | X_train_final: {self.X_train_final.shape}")


    # --------------------------
    # 10️⃣ Model Training
    # --------------------------
    def train_models(self):
        logger.info("🤖 Model Training")
        trainer = ModelTrainer(
            self.X_train_final, self.X_test_final, self.y_train_final, self.y_test_final
        )
        trainer.run()

    # --------------------------
    # Full pipeline run
    # --------------------------
    def run(self):
        self.load_data()
        self.visualise_data()
        self.clean_data()
        self.handle_missing_values()
        self.handle_outliers()
        self.fill_categorical_na()
        self.feature_encoding()
        self.variable_transformation()
        self.feature_selection()
        self.balance_data()
        self.train_models()
        logger.info("🎯 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")

# ==========================
# Execute pipeline
# ==========================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\Suresh Goud\Documents\project intern\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    pipeline = ChurnPipeline(DATA_PATH)
    pipeline.run()
