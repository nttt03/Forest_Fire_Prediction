import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    # Đọc dữ liệu
    df = pd.read_csv(file_path)

    # Loại bỏ các cột không cần thiết (nếu có)
    df = df.drop(columns=["Day", "Month", "Year"], errors="ignore")

    # Xử lý giá trị NaN bằng cách loại bỏ dòng chứa NaN
    df = df.dropna()

    # Chuẩn hóa nhãn (Classes) về giá trị cố định và loại bỏ khoảng trắng thừa
    df["Classes"] = df["Classes"].str.strip().str.lower()
    df["Classes"] = df["Classes"].map({"fire": 1, "not fire": 0})
    
    # Kiểm tra nếu còn giá trị NaN trong nhãn sau khi ánh xạ
    df = df.dropna(subset=["Classes"])

    # Tách đặc trưng và nhãn
    X = df.drop(columns=["Classes"]).values
    y = df["Classes"].astype(int).values  # Đảm bảo y là số nguyên

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    X, y, scaler = load_and_preprocess_data("../data/Algerian_forest_fires_dataset.csv")
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
