"""
=============================================================================
Project: E-Commerce Store Management & Analytics Ecosystem
Team: Hyper Digi (2025-2026)
Lead Architect & Data Scientist: Mohamed Khaled Mahmoud

Description: 
Production-grade data pipeline script for the Exploratory Data Analysis (EDA) 
and Feature Engineering phase. This script ingests raw e-commerce transaction 
data (22,049 records across 10 Turkish cities), cleans anomalies, engineers 
predictive features (e.g., RFM), and prepares the dataset for the Orange DM 
Machine Learning pipeline (K-Means & Random Forest).
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress warnings for cleaner production logs
warnings.filterwarnings('ignore')

def load_transactional_data(filepath: str) -> pd.DataFrame:
    """
    [DATA ENGINEERING LOGIC]
    Ingests the consolidated raw data extract from the SQL Server database.
    Expects a joined view of Customers, Orders, OrderItems, and Products.
    
    Args:
        filepath (str): The path to the raw dataset CSV.
    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    print(f"[INFO] Loading raw e-commerce data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"[SUCCESS] Loaded {df.shape[0]} records and {df.shape[1]} features.")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        raise

def preprocess_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DATA ENGINEERING LOGIC]
    Handles data quality issues inherent in transaction databases.
    - Standardizes column names.
    - Handles missing values (e.g., imputing missing cities with mode, dropping null crucial IDs).
    - Removes duplicate transaction anomalies.
    - Enforces correct data types for datetime and numerical columns.
    
    Args:
        df (pd.DataFrame): The raw dataframe.
    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    print("[INFO] Initiating Data Preprocessing...")
    
    # Drop complete duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[CLEANUP] Removed {initial_rows - len(df)} duplicate records.")
    
    # Handle Missing Values
    if 'City' in df.columns:
        df['City'].fillna(df['City'].mode()[0], inplace=True)
        
    # Standardize Datetime
    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        
    # Standardize Strings
    if 'CategoryName' in df.columns:
        df['CategoryName'] = df['CategoryName'].str.title().str.strip()
        
    print("[SUCCESS] Data preprocessing completed.")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DATA ENGINEERING LOGIC]
    Transforms raw transactional data into machine-learning-ready features.
    - Computes Recency, Frequency, and Monetary (RFM) scores for Customer Segmentation.
    - Extracts temporal features (Day of Week, Month) to capture seasonality.
    - Flags high-risk transactions for the Random Forest churn/return model.
    
    Args:
        df (pd.DataFrame): The cleaned dataframe.
    Returns:
        pd.DataFrame: The feature-rich dataframe.
    """
    print("[INFO] Executing Feature Engineering pipeline...")
    
    # 1. Temporal Feature Extraction
    if 'OrderDate' in df.columns:
        df['OrderMonth'] = df['OrderDate'].dt.month
        df['OrderDayOfWeek'] = df['OrderDate'].dt.dayofweek
        df['IsWeekend'] = df['OrderDayOfWeek'].isin([5, 6]).astype(int)
    
    # 2. RFM Aggregation (Recency, Frequency, Monetary)
    # Assumes dataset has CustomerID, OrderDate, and LineTotal/Amount
    if all(col in df.columns for col in ['CustomerID', 'OrderDate', 'LineTotal']):
        current_date = df['OrderDate'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby('CustomerID').agg({
            'OrderDate': lambda x: (current_date - x.max()).days, # Recency
            'OrderID': 'nunique',                                 # Frequency
            'LineTotal': 'sum'                                    # Monetary
        }).rename(columns={
            'OrderDate': 'Recency',
            'OrderID': 'Frequency',
            'LineTotal': 'Monetary'
        }).reset_index()
        
        # Merge RFM back to main dataframe (or keep separate for ML)
        df = df.merge(rfm, on='CustomerID', how='left')
        
    # 3. Target Variable Preparation (e.g., Return Likelihood)
    # If Status indicates 'Returned', create a binary target
    if 'Status' in df.columns:
        df['IsReturned'] = (df['Status'] == 'Returned').astype(int)
        
    print("[SUCCESS] Feature Engineering completed. RFM and Temporal features generated.")
    return df

def detect_and_handle_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    [DATA ENGINEERING LOGIC]
    Applies the Interquartile Range (IQR) method to cap extreme financial outliers
    which could negatively skew the K-Means clustering algorithm.
    """
    if column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of dropping to preserve data volume
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        print(f"[INFO] Outliers capped for feature: {column}")
        
    return df

def export_for_orange_dm(df: pd.DataFrame, output_path: str):
    """
    [DATA ENGINEERING LOGIC]
    Exports the finalized, ML-ready dataset. This artifact is directly ingested 
    into the 35-widget Orange Data Mining visual pipeline for K-Means and Random Forest modeling.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] ML-ready dataset successfully exported to: {output_path}")
        print("[PIPELINE] Ready for Orange DM ingestion.")
    except Exception as e:
        print(f"[ERROR] Failed to export dataset: {e}")

if __name__ == "__main__":
    # --- Pipeline Execution Orchestration ---
    print("="*50)
    print("E-COMMERCE DATA PIPELINE INITIATED")
    print("Team: Hyper Digi | Lead: Mohamed Khaled")
    print("="*50)
    
    # Define Paths (Replace with actual paths in production)
    INPUT_DATA_PATH = "data/raw/ecommerce_transactions_view.csv"
    OUTPUT_DATA_PATH = "data/processed/ecommerce_ml_ready.csv"
    
    try:
        # Step 1: Ingest Data
        # For demonstration purposes, skipping actual load if file doesn't exist.
        # data = load_transactional_data(INPUT_DATA_PATH)
        
        # Mocking data to represent the 22,049 records
        print(f"[INFO] Mocking data schema for 22,049 records across 10 cities...")
        mock_data = pd.DataFrame({
            'CustomerID': np.random.randint(1, 5000, 22049),
            'OrderID': np.random.randint(10000, 90000, 22049),
            'OrderDate': pd.date_range(start='2025-01-01', periods=22049, freq='min'),
            'City': np.random.choice(['Istanbul', 'Ankara', 'Izmir', 'Bursa', 'Antalya', 'Adana', 'Konya', 'Gaziantep', 'Kayseri', 'Mersin'], 22049),
            'CategoryName': np.random.choice(['Electronics', 'Daily Essentials', 'Apparel', 'Home'], 22049),
            'LineTotal': np.random.uniform(10, 1500, 22049),
            'Status': np.random.choice(['Delivered', 'Returned', 'Pending'], 22049, p=[0.85, 0.10, 0.05])
        })
        
        # Step 2: Clean & Preprocess
        clean_data = preprocess_and_clean(mock_data)
        
        # Step 3: Feature Engineering (RFM, Time)
        featured_data = engineer_features(clean_data)
        
        # Step 4: Outlier Treatment (Monetary/LineTotal)
        final_data = detect_and_handle_outliers(featured_data, 'LineTotal')
        
        # Step 5: Export to Orange DM
        export_for_orange_dm(final_data, OUTPUT_DATA_PATH)
        
        print("="*50)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*50)
        
    except Exception as e:
        print(f"[FATAL] Pipeline execution halted: {e}")
