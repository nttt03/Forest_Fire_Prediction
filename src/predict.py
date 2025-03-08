import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNN_MODEL_PATH = os.path.join(BASE_DIR, "../models/knn_model.pkl")
DT_MODEL_PATH = os.path.join(BASE_DIR, "../models/decision_tree_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

# Load mô hình và scaler
knn = joblib.load(KNN_MODEL_PATH)
dt = joblib.load(DT_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_fire_risk(input_features):
    # Chuyển đổi input thành numpy array
    input_array = np.array(input_features).reshape(1, -1)

    # Chuẩn hóa dữ liệu đầu vào
    input_scaled = scaler.transform(input_array)

    # Dự đoán bằng KNN
    prediction_knn = knn.predict(input_scaled)[0]
    result_knn = "🔥 Fire Risk!" if prediction_knn == 1 else "✅ No Fire Risk"
    
    # Dự đoán bằng Decision Tree
    prediction_dt = dt.predict(input_scaled)[0]
    result_dt = "🔥 Fire Risk!" if prediction_dt == 1 else "✅ No Fire Risk"
    
    # Hiển thị kết quả
    print("--- Prediction Results ---")
    print(f"KNN Prediction: {result_knn}")
    print(f"Decision Tree Prediction: {result_dt}")
    
    return result_knn, result_dt

if __name__ == "__main__":
    # Ví dụ dự đoán
    sample_input = [20, 50, 10, 0, 85, 6, 8, 9, 10, 5]  # Giá trị mẫu
    predict_fire_risk(sample_input)
