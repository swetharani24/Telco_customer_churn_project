import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from log_file import setup_logging

logger = setup_logging("visualisation")

# =========================
# Plot directory
# =========================
PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================
# Save plot helper
# =========================
def save_plot(filename):
    try:
        path = os.path.join(PLOT_DIR, filename)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"Plot saved: {path}")
    except Exception as e:
        logger.error(f"Plot save failed ({filename}): {e}")

# =========================
# Add SIM Operator column
# =========================
def add_sim_operator_column(df, seed=42):
    try:
        np.random.seed(seed)
        operators = ['Airtel', 'BSNL', 'Vodafone', 'Jio']
        probabilities = [0.30, 0.10, 0.20, 0.40]
        df['SIM_Operator'] = np.random.choice(
            operators, size=len(df), p=probabilities
        )
        logger.info("SIM_Operator column added")
        return df
    except Exception as e:
        logger.error(f"SIM operator column error: {e}")
        return df

# =========================
# Save dataset
# =========================
def save_dataset(df, path):
    try:
        df.to_csv(path, index=False)
        logger.info(f"Dataset saved at {path}")
    except Exception as e:
        logger.error(f"Dataset save failed: {e}")

# =========================
# Generic bar plot
# =========================
def bar_plot(x, y, xlabel, ylabel, title, filename):
    try:
        plt.figure(figsize=(6, 4))
        plt.bar(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        save_plot(filename)
    except Exception as e:
        logger.error(f"Bar plot failed ({title}): {e}")

# =========================
# Plot functions
# =========================
def plot_churn_distribution(df):
    churn = df['Churn'].value_counts()
    bar_plot(
        churn.index, churn.values,
        "Churn", "Customers",
        "Churn Distribution",
        "churn_distribution.png"
    )

def plot_gender_distribution(df):
    gender = df['gender'].value_counts()
    bar_plot(
        gender.index, gender.values,
        "Gender", "Customers",
        "Gender Distribution",
        "gender_distribution.png"
    )

def plot_tenure_vs_churn(df):
    try:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
        df['Tenure_Quarter'] = pd.cut(
            df['tenure'],
            bins=[0, 18, 36, 54, 72],
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        ct = pd.crosstab(df['Tenure_Quarter'], df['Churn'])
        ct.plot(kind='bar', figsize=(6, 4))
        plt.title("Tenure vs Churn")
        save_plot("tenure_vs_churn.png")
    except Exception as e:
        logger.error(f"Tenure vs churn failed: {e}")

def plot_payment_vs_churn(df):
    try:
        ct = pd.crosstab(df['PaymentMethod'], df['Churn'])
        ct.plot(kind='bar', figsize=(7, 4))
        plt.title("Payment Method vs Churn")
        save_plot("payment_vs_churn.png")
    except Exception as e:
        logger.error(f"Payment vs churn failed: {e}")

def plot_monthly_charges_vs_churn(df):
    try:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
        df['Charges_Quartile'] = pd.qcut(
            df['MonthlyCharges'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        ct = pd.crosstab(df['Charges_Quartile'], df['Churn'])
        ct.plot(kind='bar', figsize=(6, 4))
        plt.title("Monthly Charges vs Churn")
        save_plot("monthly_charges_vs_churn.png")
    except Exception as e:
        logger.error(f"Monthly charges plot failed: {e}")

def plot_churn_vs_feature(df, feature, title, filename):
    try:
        ct = pd.crosstab(df[feature], df['Churn'])
        ct.plot(kind='bar', figsize=(6, 4))
        plt.title(title)
        save_plot(filename)
    except Exception as e:
        logger.error(f"{feature} churn plot failed: {e}")

def plot_gender_vs_internet(df):
    try:
        ct = pd.crosstab(df['gender'], df['InternetService'])
        ct.plot(kind='bar', figsize=(7, 4))
        plt.title("Gender vs Internet Service")
        save_plot("gender_vs_internet.png")
    except Exception as e:
        logger.error(f"Gender vs internet failed: {e}")

def plot_churn_vs_sim_operator(df):
    try:
        ct = pd.crosstab(df['SIM_Operator'], df['Churn'])
        ct.plot(kind='bar', figsize=(7, 4))
        plt.title("Churn vs SIM Operator")
        save_plot("churn_vs_sim_operator.png")
    except Exception as e:
        logger.error(f"SIM operator churn failed: {e}")

def plot_sim_operator_vs_gender(df):
    try:
        ct = pd.crosstab(df['SIM_Operator'], df['gender'])
        ct.plot(kind='bar', figsize=(7, 4))
        plt.title("SIM Operator vs Gender")
        save_plot("sim_operator_vs_gender.png")
    except Exception as e:
        logger.error(f"SIM vs gender failed: {e}")

# =========================
# MAIN PIPELINE FUNCTION
# =========================
def generate_all_plots(df):
    """
    Industry-style visualisation pipeline
    Called from main.py
    """
    try:
        logger.info("Visualisation pipeline started")

        df = add_sim_operator_column(df)
        save_dataset(df, "outputs/Telco_with_SIM.csv")

        plot_churn_distribution(df)
        plot_gender_distribution(df)
        plot_tenure_vs_churn(df)
        plot_payment_vs_churn(df)
        plot_monthly_charges_vs_churn(df)
        plot_churn_vs_feature(df, 'Contract',
                              "Churn by Contract Type",
                              "churn_vs_contract.png")
        plot_churn_vs_feature(df, 'gender',
                              "Churn by Gender",
                              "churn_vs_gender.png")
        plot_gender_vs_internet(df)
        plot_churn_vs_sim_operator(df)
        plot_sim_operator_vs_gender(df)

        logger.info("Visualisation pipeline completed successfully")

    except Exception as e:
        logger.error(f"Visualisation pipeline failed: {e}")
        raise
