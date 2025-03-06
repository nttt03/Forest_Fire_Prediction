import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import joblib
import numpy as np
import os
import tkinter as tk

# Load model và scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")
ICON_PATH = os.path.join(BASE_DIR, "../data/imgicon.ico")  # Đường dẫn icon

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Hàm căn giữa cửa sổ
def center_window(root, width=600, height=700):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

# Hàm xóa dữ liệu trong các ô nhập liệu
def clear_inputs():
    for entry in entries:
        entry.delete(0, tk.END)

# Hàm hiển thị hộp thoại lỗi căn giữa cửa sổ chính
def show_error_dialog():
    dialog = tk.Toplevel(root)
    dialog.title("Lỗi nhập liệu")
    dialog.geometry("420x150")  
    dialog.resizable(False, False)

    # Căn giữa hộp thoại so với cửa sổ chính
    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_width = root.winfo_width()
    root_height = root.winfo_height()

    dialog_x = root_x + (root_width // 2) - (420 // 2)
    dialog_y = root_y + (root_height // 2) - (150 // 2)
    dialog.geometry(f"420x150+{dialog_x}+{dialog_y}")

    # Thiết lập icon cho hộp thoại nếu có
    if os.path.exists(ICON_PATH):
        dialog.iconbitmap(ICON_PATH)

    # Frame chính để dễ căn chỉnh
    content_frame = ttk.Frame(dialog, padding=10)
    content_frame.pack(fill="both", expand=True)

    # Nội dung thông báo
    label = ttk.Label(content_frame, 
                      text="⚠ Vui lòng nhập số hợp lệ vào tất cả các ô!", 
                      font=("Roboto", 12), 
                      wraplength=380,   # Tự động xuống dòng
                      justify="left", 
                      anchor="w")
    label.pack(pady=(10, 5), fill="both", expand=True)

    # Thêm đường ngăn cách giữa nội dung và nút OK
    separator = ttk.Separator(content_frame, orient="horizontal")
    separator.pack(fill="x", pady=5)

    # Căn nút OK ở góc phải
    btn_ok = ttk.Button(content_frame, text="OK", bootstyle="danger", command=dialog.destroy)
    btn_ok.pack(side="right", padx=10, pady=5)  # Căn phải

    dialog.transient(root)  # Giữ hộp thoại trên cửa sổ chính
    dialog.grab_set()  # Chặn thao tác với cửa sổ chính cho đến khi đóng hộp thoại
    root.wait_window(dialog)  # Đợi hộp thoại đóng trước khi tiếp tục

# Hàm dự đoán
def predict_fire_risk():
    try:
        values = [float(entry.get()) for entry in entries]
        values_scaled = scaler.transform([values])
        prediction = model.predict(values_scaled)[0]

        result_text = "🔥 Nguy cơ cháy rừng cao!" if prediction == 1 else "An toàn, không có nguy cơ cháy rừng."

        # Hộp thoại kết quả với nút OK sẽ xóa dữ liệu nhập
        dialog = Messagebox.show_info(result_text, title="Kết quả dự đoán")
        clear_inputs()  # Xóa dữ liệu sau khi nhấn OK

    except ValueError:
        show_error_dialog()  # Sử dụng hộp thoại lỗi căn giữa

# Tạo giao diện chính
root = ttk.Window(themename="journal")
root.title("Dự đoán nguy cơ cháy rừng")

# Thiết lập icon cho cửa sổ chính
if os.path.exists(ICON_PATH):
    root.iconbitmap(ICON_PATH)

center_window(root)
root.resizable(False, False)

# Khung chứa các input
frame = ttk.Frame(root, bootstyle="light")
frame.place(relx=0.5, rely=0.4, anchor="center")

# Tạo danh sách nhãn và ô nhập
labels = [
    "Temperature (°C)", "RH (%)", "Ws (km/h)", "Rain (mm)", 
    "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"
]
entries = []

for i, label_text in enumerate(labels):
    ttk.Label(frame, text=label_text, font=("Roboto", 12, "bold")).grid(row=i, column=0, sticky="w", padx=20, pady=8)
    entry = ttk.Entry(frame, font=("Roboto", 12), width=15, bootstyle="success")
    entry.grid(row=i, column=1, padx=20, pady=8)
    entries.append(entry)

# Nút dự đoán
btn_predict = ttk.Button(root, text="🔍 Dự đoán", bootstyle="success-outline", 
                         command=predict_fire_risk, padding=(15, 5))
btn_predict.place(relx=0.5, rely=0.9, anchor="center")

root.mainloop()
