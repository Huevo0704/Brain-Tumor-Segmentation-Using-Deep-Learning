# config.py
"""
File cấu hình trung tâm cho dự án.
Chứa tất cả các đường dẫn, siêu tham số và các hằng số.
"""

# ==============================================================================
# 1. CẤU HÌNH DỮ LIỆU VÀ ĐƯỜNG DẪN
# ==============================================================================
# Thư mục gốc chứa dữ liệu
DATA_DIR = '/content/gdrive/MyDrive/DATA'

# Đường dẫn đến thư mục chứa ảnh và mask cho việc huấn luyện
IMAGE_DIR = f'{DATA_DIR}/images'
MASK_DIR = f'{DATA_DIR}/masks'

# Đường dẫn đến bộ dữ liệu test (dạng .npy)
X_TEST_PATH = f'{DATA_DIR}/test/X_test.npy'
Y_TEST_PATH = f'{DATA_DIR}/test/Y_test.npy'

# Thư mục để lưu các kết quả (model, history, logs, ảnh phân tích)
OUTPUT_DIR = DATA_DIR
EVALUATION_OUTPUT_DIR = f'{DATA_DIR}/evaluation_results' # Thư mục riêng cho kết quả đánh giá

# ==============================================================================
# 2. CẤU HÌNH HUẤN LUYỆN (TRAINING)
# ==============================================================================
IMG_SIZE = (512, 512)
EPOCHS = 65
BATCH_SIZE = 2  # Giảm để tiết kiệm bộ nhớ, có thể tăng nếu VRAM cho phép
LEARNING_RATE = 5e-5

# ==============================================================================
# 3. CẤU HÌNH TÊN FILE ĐẦU RA
# ==============================================================================
BEST_MODEL_NAME = "attention_unet_best_model.keras"
FINAL_MODEL_KERAS_NAME = "attention_unet_final_model.keras"
FINAL_MODEL_H5_NAME = "attention_unet_final_model.h5"
HISTORY_FILE_NAME = "training_history.pkl"
LOG_FILE_NAME = "training_log.csv"
TENSORBOARD_LOG_DIR = "logs/fit/"

# ==============================================================================
# 4. CẤU HÌNH ĐÁNH GIÁ (EVALUATION)
# ==============================================================================
# Đường dẫn đến model tốt nhất để chạy đánh giá. Mặc định là model đã lưu từ quá trình training.
EVAL_MODEL_PATH = f"{OUTPUT_DIR}/{BEST_MODEL_NAME}"

# Các tham số cho việc phân tích diện tích
KNOWN_IMAGE_DPI = 96  # DPI của màn hình hoặc máy quét (quan trọng để tính mm)
NUM_RANDOM_IMAGES_TO_TEST = 10  # Số lượng ảnh ngẫu nhiên để kiểm tra từ bộ test
TARGET_SIZE_EVAL = (512, 512) # Kích thước ảnh khi đánh giá
