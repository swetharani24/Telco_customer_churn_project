import pandas as pd
import os
import numpy as np
import logging
from log_file import setup_logging

logger = setup_logging("datacleaning")

class DataCleaning:
    def __init__(self, df, artifact_dir="artifacts"):
        self.df = df
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def add_simtype_column(self):
        """
        Adds synthetic telecom provider column: simtype
        """
        self.df = self.df.copy()
        providers = ["Airtel", "Jio", "VI", "BSNL"]

        np.random.seed(42)
        self.df["simtype"] = np.random.choice(providers, size=len(self.df))

        logger.info("🧪 Synthetic column 'simtype' added")

    def run_data_cleaning(self):
        try:
            logger.info("🧹 Data Cleaning Started")

            # Shape & columns BEFORE cleaning
            logger.info(f"📐 Data shape (before cleaning): {self.df.shape}")
            logger.info(f"📋 Columns before cleaning: {self.df.columns.tolist()}")
            print("📋 Data info before cleaning:")
            self.df.info()

            # Basic cleaning
            self.df.columns = self.df.columns.str.strip()

            # Add synthetic column
            self.add_simtype_column()

            # Shape & columns AFTER cleaning
            logger.info(f"📐 Data shape (after cleaning): {self.df.shape}")
            logger.info(f"📋 Columns after cleaning: {self.df.columns.tolist()}")
            print("📋 Data info after cleaning:")
            self.df.info()

            # Save cleaned data
            output_path = os.path.join(self.artifact_dir, "cleaned_data.csv")
            self.df.to_csv(output_path, index=False)
            logger.info(f"✅ Cleaned data saved at {output_path}")

            return self.df

        except Exception as e:
            logger.error(f"❌ Data Cleaning Failed: {e}")
            raise
