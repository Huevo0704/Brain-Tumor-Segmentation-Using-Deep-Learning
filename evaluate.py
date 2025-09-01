# evaluate.py
"""
File chính để đánh giá mô hình đã được huấn luyện trên bộ dữ liệu test.
Thực hiện các bước:
1. Tải mô hình và dữ liệu test.
2. Lặp qua một số ảnh ngẫu nhiên.
3. Với mỗi ảnh:
    a. Dự đoán bằng U-Net.
    b. Áp dụng các bước hậu xử lý (Median, Adaptive, Active Contour).
    c. Tính toán diện tích khối u.
    d. Trực quan hóa kết quả.
4. Tạo bảng tổng hợp kết quả phân tích diện tích.
"""

import os
import random
import numpy as np
import tensorflow as tf
import cv2
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Import các module tự định nghĩa
import config
from utils import (
    dice_coefficient, iou, combined_loss,
    segment_tumor_adaptive, calculate_final_tumor_area,
    visualize_decomposition, plot_evaluation_results, create_summary_table
)

def load_model_with_custom_objects(model_path):
    """Tải mô hình Keras với các đối tượng tùy chỉnh."""
    print(f"\n1. Đang tải model từ '{model_path}'...")
    custom_objects = {
        'combined_loss': combined_loss,
        'dice_coefficient': dice_coefficient,
        'iou': iou,
        'sensitivity': tf.keras.metrics.SensitivityAtSpecificity(0.5), # Cần cung cấp một metric instance
        'specificity': tf.keras.metrics.SpecificityAtSensitivity(0.5)
    }
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✅ Model đã được tải thành công!"); model.summary()
        return model
    except Exception as e:
        print(f"❌ Lỗi: Không thể tải model. Dừng chương trình.\nChi tiết: {e}"); exit()

def main():
    """Hàm chính thực hiện quy trình đánh giá."""
    model = load_model_with_custom_objects(config.EVAL_MODEL_PATH)

    print(f"\n2. Đang tải bộ dữ liệu test từ '{config.X_TEST_PATH}'...")
    try:
        X_test = np.load(config.X_TEST_PATH)
        Y_test = np.load(config.Y_TEST_PATH)
        print(f"✅ Tải dữ liệu thành công! Tìm thấy {len(X_test)} mẫu.")
    except Exception as e:
        print(f"❌ Lỗi: Không thể tải tệp dữ liệu .npy. Dừng chương trình.\nChi tiết: {e}"); exit()

    num_samples = len(X_test)
    image_indices = random.sample(range(num_samples), min(num_samples, config.NUM_RANDOM_IMAGES_TO_TEST))
    print(f"\n🚀 Sẽ xử lý {len(image_indices)} ảnh ngẫu nhiên: {image_indices}\n")

    all_results = []
    pixel_area_mm2 = (25.4 / config.KNOWN_IMAGE_DPI) ** 2

    for i, idx in enumerate(image_indices):
        print("="*80 + f"\n====== XỬ LÝ ẢNH {i+1}/{len(image_indices)} (INDEX: {idx}) ======\n" + "="*80)
        
        original_image, ground_truth = X_test[idx], Y_test[idx].squeeze()
        image_resized = cv2.resize(original_image, config.TARGET_SIZE_EVAL).astype("float32")
        gt_resized = (cv2.resize(ground_truth, config.TARGET_SIZE_EVAL, cv2.INTER_NEAREST) > 0).astype(np.uint8)

        # 3. DỰ ĐOÁN VÀ HẬU XỬ LÝ
        pred_mask = model.predict(np.expand_dims(image_resized, axis=[0, -1]))[0, ..., 0]
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
        
        if np.count_nonzero(pred_mask_binary) == 0:
            print(f"*** ❗ [Ảnh {idx}] KẾT QUẢ: KHÔNG PHÁT HIỆN KHỐI U. Bỏ qua. ***\n"); continue

        # Pipeline hậu xử lý đơn giản hóa
        mask_median = (cv2.medianBlur(pred_mask_binary * 255, 5) > 127).astype(np.uint8)
        mask_adaptive = segment_tumor_adaptive(mask_median * 255, 21, 5)
        final_mask = mask_adaptive
        
        contours, _ = cv2.findContours(mask_adaptive * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            snake = active_contour(gaussian(image_resized, 1), np.squeeze(max(contours, key=cv2.contourArea)), alpha=0.01, beta=0.1, gamma=0.001)
            temp_mask = np.zeros_like(final_mask); cv2.fillPoly(temp_mask, [snake.astype(np.int32)], 255)
            final_mask = (temp_mask / 255).astype(np.uint8)

        # 4. TÍNH TOÁN VÀ LƯU KẾT QUẢ
        gt_geom_mm2, gt_pixel_mm2 = calculate_final_tumor_area(gt_resized, pixel_area_mm2)
        pred_geom_mm2, pred_pixel_mm2 = calculate_final_tumor_area(final_mask, pixel_area_mm2)
        
        all_results.append({"index": idx, "gt_geom": gt_geom_mm2, "pred_geom": pred_geom_mm2, "gt_pixel": gt_pixel_mm2, "pred_pixel": pred_pixel_mm2})
        
        # 5. TRỰC QUAN HÓA
        img_vis = (image_resized * 255).astype(np.uint8)
        triangulated_img = visualize_decomposition(img_vis, final_mask, show=False)
        plot_evaluation_results(image_resized, gt_resized, final_mask, triangulated_img, idx)

    # 6. TẠO BẢNG TỔNG HỢP
    os.makedirs(config.EVALUATION_OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(config.EVALUATION_OUTPUT_DIR, "summary_area_analysis.png")
    create_summary_table(all_results, summary_path)
    
    print("\n🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH ĐÁNH GIÁ. 🎉")

if __name__ == '__main__':
    main()
