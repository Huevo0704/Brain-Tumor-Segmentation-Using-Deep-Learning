# training.py
"""
File chính để điều khiển quá trình huấn luyện mô hình.
Thực hiện các bước:
1. Cài đặt môi trường (mixed precision).
2. Chuẩn bị dữ liệu.
3. Xây dựng và biên dịch mô hình.
4. Thiết lập callbacks.
5. Bắt đầu huấn luyện.
6. Lưu kết quả và mô hình.
7. Trực quan hóa một vài dự đoán.
"""

import os
import pickle
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard

# Import các module tự định nghĩa
import config
from data_loader import create_dataset
from model import build_unet_with_attention
from utils import dice_coefficient, iou, sensitivity, specificity, combined_loss, display_predictions

def main():
    """Hàm chính để thực hiện toàn bộ quá trình huấn luyện."""
    
    # 1. CÀI ĐẶT MÔI TRƯỜNG
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ Mixed precision policy set to 'mixed_float16'")
    
    # 2. CHUẨN BỊ DỮ LIỆU
    print(f" Đang xử lý dataset từ {config.IMAGE_DIR} và {config.MASK_DIR}...")
    image_files = sorted(os.listdir(config.IMAGE_DIR))
    image_paths = [os.path.join(config.IMAGE_DIR, f) for f in image_files if f.endswith('.jpg')]
    mask_paths = [os.path.join(config.MASK_DIR, f.replace('.jpg', '_mask.png')) for f in image_files]
    
    X_train, X_val, y_train, y_val = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    train_dataset = create_dataset(X_train, y_train, augment=True)
    val_dataset = create_dataset(X_val, y_val, augment=False)
    print(f"✅ Dataset Created: {len(X_train)} train, {len(X_val)} validation")

    # 3. XÂY DỰNG VÀ BIÊN DỊCH MÔ HÌNH
    model = build_unet_with_attention()
    model.summary()
    optimizer = AdamW(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=['accuracy', dice_coefficient, iou, sensitivity, specificity])
    print("✅ Model compiled successfully.")

    # 4. THIẾT LẬP CALLBACKS
    log_dir = config.TENSORBOARD_LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(config.OUTPUT_DIR, config.BEST_MODEL_NAME), monitor='val_dice_coefficient', save_best_only=True, mode='max', verbose=1),
        CSVLogger(os.path.join(config.OUTPUT_DIR, config.LOG_FILE_NAME), append=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    print("✅ Callbacks configured.")

    # 5. BẮT ĐẦU HUẤN LUYỆN
    print("\n" + "="*50 + "\n BẮT ĐẦU HUẤN LUYỆN \n" + "="*50)
    history = model.fit(train_dataset, epochs=config.EPOCHS, validation_data=val_dataset, callbacks=callbacks)

    # 6. LƯU KẾT QUẢ
    print("\n" + "="*50 + "\n HUẤN LUYỆN HOÀN TẤT - ĐANG LƯU KẾT QUẢ \n" + "="*50)
    history_path = os.path.join(config.OUTPUT_DIR, config.HISTORY_FILE_NAME)
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✅ History đã được lưu vào {history_path}")

    model.save(os.path.join(config.OUTPUT_DIR, config.FINAL_MODEL_KERAS_NAME))
    model.save(os.path.join(config.OUTPUT_DIR, config.FINAL_MODEL_H5_NAME))
    print("✅ Mô hình cuối cùng đã được lưu ở cả định dạng .keras và .h5!")

    # 7. TRỰC QUAN HÓA KẾT QUẢ
    print("\n Hiển thị một vài ví dụ dự đoán trên tập validation:")
    display_predictions(model, val_dataset)

if __name__ == '__main__':
    main()
