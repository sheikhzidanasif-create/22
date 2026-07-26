import pandas as pd
import numpy as np

def perform_eda():
    print("--- Data Loading ---")
    df = pd.read_csv('/content/car_purchasing (1).csv')
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Dataset Info ---")
    print(df.info())
    
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    perform_eda()
