import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df = df.drop(columns=["Day", "Month", "Year"], errors="ignore")
    df = df.dropna()

    df["Classes"] = df["Classes"].str.strip().str.lower()
    df["Classes"] = df["Classes"].map({"fire": 1, "not fire": 0})

    df = df.dropna(subset=["Classes"])

    # Tách đặc trưng và nhãn
    X = df.drop(columns=["Classes"]).values
    y = df["Classes"].astype(int).values  

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    X, y, scaler = load_and_preprocess_data("../data/Algerian_forest_fires_dataset.csv")
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
