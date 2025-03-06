import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

# Load mô hình và scaler
knn = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_fire_risk(input_features):
    # Chuyển đổi input thành numpy array
    input_array = np.array(input_features).reshape(1, -1)

    # Chuẩn hóa dữ liệu đầu vào
    input_scaled = scaler.transform(input_array)

    # Dự đoán
    prediction = knn.predict(input_scaled)
    return "🔥 Fire Risk!" if prediction[0] == 1 else "✅ No Fire Risk"

if __name__ == "__main__":
    # Ví dụ dự đoán
    sample_input = [20, 50, 10, 0, 85, 6, 8, 9, 10, 5, 12]  # Giá trị mẫu
    result = predict_fire_risk(sample_input)
    print("Prediction:", result)
