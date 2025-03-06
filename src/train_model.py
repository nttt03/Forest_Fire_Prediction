import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/Algerian_forest_fires_dataset.csv")

# Load và tiền xử lý dữ liệu
X, y, scaler = load_and_preprocess_data(DATA_PATH)

# Chia tập dữ liệu (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Lưu mô hình và scaler
model_dir = os.path.join(BASE_DIR, "../models")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(knn, os.path.join(model_dir, "knn_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
print("Mô hình đã được lưu thành công!")
