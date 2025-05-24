import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ẩn tất cả các cảnh báo từ TensorFlow
import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Cấu hình ---
# Thay đổi MODEL_PATH nếu file model của bạn có tên khác hoặc nằm ở thư mục con
print (" Loading......")
MODEL_PATH = 'xception_brain_tumor_classifier.keras'
IMAGE_WIDTH = 299  # Chiều rộng ảnh đầu vào model (từ notebook của bạn)
IMAGE_HEIGHT = 299 # Chiều cao ảnh đầu vào model (từ notebook của bạn)

# Các class label theo đúng thứ tự mà model đã học (từ notebook của bạn)
# {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Tải Model ---
# Kiểm tra xem file model có tồn tại không
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Please make sure the model file is in the same directory as app.py, "
        "or update the MODEL_PATH variable."
    )

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise IOError(f"Error loading Keras model: {e}. Ensure the model file is valid.")

# --- Hàm tiền xử lý ảnh ---
def preprocess_image(image_pil):
    """
    Tiền xử lý ảnh PIL đầu vào để phù hợp với input của model.
    1. Resize ảnh về IMAGE_WIDTH, IMAGE_HEIGHT.
    2. Chuyển ảnh sang NumPy array.
    3. Chuẩn hóa giá trị pixel về khoảng [0, 1] (chia cho 255.0).
    4. Mở rộng chiều (dimension) để tạo batch size là 1.
    """
    image_resized = image_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image_array = np.array(image_resized)

    # Nếu ảnh là grayscale, chuyển thành RGB bằng cách lặp lại kênh đó
    if image_array.ndim == 2: # Grayscale (height, width)
        image_array = np.stack((image_array,)*3, axis=-1) # (height, width, 3)
    elif image_array.shape[2] == 1: # Grayscale với 1 kênh (height, width, 1)
        image_array = np.concatenate([image_array]*3, axis=-1) # (height, width, 3)
    elif image_array.shape[2] == 4: # RGBA, loại bỏ kênh alpha
        image_array = image_array[:, :, :3]

    # Chuẩn hóa và mở rộng batch dimension
    image_normalized = image_array / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0) # (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    return image_batch

# --- Hàm dự đoán ---
def predict(image_pil):
    """
    Nhận ảnh PIL, tiền xử lý, dự đoán và trả về dictionary các nhãn và xác suất.
    """
    if image_pil is None:
        return {label: 0.0 for label in CLASS_NAMES} # Trả về nếu không có ảnh

    preprocessed_image = preprocess_image(image_pil)
    predictions = model.predict(preprocessed_image)
    
    # predictions[0] là mảng các xác suất cho từng lớp
    # Ví dụ: [0.1, 0.05, 0.8, 0.05]
    # Kết hợp CLASS_NAMES với predictions để tạo dictionary
    confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    return confidences

# --- Tạo giao diện Gradio ---
# Sử dụng kiểu input "pil" cho gr.Image để nhận đối tượng PIL Image
# Output là gr.Label sẽ hiển thị đẹp các nhãn và xác suất
inputs = gr.Image(type="pil", label="Upload Brain MRI Image")
outputs = gr.Label(num_top_classes=len(CLASS_NAMES), label="Prediction Results")

# Tiêu đề và mô tả cho ứng dụng Gradio
title = "Brain Tumor MRI Classification"
description = (
    "Upload an MRI image of a brain to classify if it's a glioma, "
    "meningioma, pituitary tumor, or no tumor. "
    f"Model: Xception based (expects {IMAGE_WIDTH}x{IMAGE_HEIGHT} images)."
)
article = "<p style='text-align: center'>Model trained on Brain Tumor MRI Dataset.</p>"

# Chạy giao diện
iface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    article=article,
    examples=[ # Bạn có thể thêm đường dẫn đến các ảnh ví dụ nếu có
        # os.path.join(os.path.dirname(__file__), "sample_images/glioma_example.jpg"),
        # os.path.join(os.path.dirname(__file__), "sample_images/notumor_example.jpg"),
    ]
)

if __name__ == "__main__":
    iface.launch()