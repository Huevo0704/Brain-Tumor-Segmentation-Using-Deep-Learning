# utils.py
"""
Chứa các hàm tiện ích được sử dụng trong toàn bộ dự án, bao gồm:
- Các chỉ số đánh giá (metrics) và hàm mất mát (loss functions) cho Keras.
- Các hàm xử lý ảnh và phân tích hình học cho việc đánh giá.
- Các hàm trực quan hóa kết quả.
"""

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from typing import Tuple, List

# ==============================================================================
# PHẦN 1: METRICS VÀ LOSS FUNCTIONS CHO KERAS (Dùng trong training.py và tải model)
# ==============================================================================

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def sensitivity(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positive / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negative = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negative / (possible_negatives + K.epsilon())

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return K.mean(alpha * K.pow((1 - p_t), gamma) * bce)
    return loss

def combined_loss(y_true, y_pred):
    """Hàm loss kết hợp giữa Dice Loss và Focal Loss."""
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss()(y_true, y_pred)
    return 0.7 * dice + 0.3 * focal

# ==============================================================================
# PHẦN 2: CÁC HÀM XỬ LÝ VÀ PHÂN TÍCH (Dùng trong evaluate.py và app.py)
# ==============================================================================

def segment_tumor_adaptive(image, block_size, C, kernel_size=5):
    """Phân đoạn ảnh sử dụng Adaptive Thresholding."""
    if block_size % 2 == 0: block_size += 1 # Block size phải là số lẻ
    processed_image = cv2.convertScaleAbs(image)
    mask = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (mask / 255).astype(np.uint8)

def calculate_evaluation_metrics(ground_truth_mask, prediction_mask):
    """
    Tính các chỉ số đánh giá (IoU, Dice, Precision, Recall) cho numpy arrays.
    **CẬP NHẬT:** Đổi tên từ calculate_metrics_eval thành tên này cho nhất quán.
    """
    gt = ground_truth_mask.astype(bool)
    pred = prediction_mask.astype(bool)
    
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    
    true_pos = np.sum(intersection)
    false_pos = np.sum(np.logical_and(~gt, pred))
    false_neg = np.sum(np.logical_and(gt, ~pred))
    
    iou_val = true_pos / (np.sum(union) + 1e-7)
    dice_val = (2. * true_pos) / (np.sum(gt) + np.sum(pred) + 1e-7)
    precision_val = true_pos / (true_pos + false_pos + 1e-7)
    recall_val = true_pos / (true_pos + false_neg + 1e-7)
    
    return iou_val, dice_val, precision_val, recall_val

def calculate_geometric_area(mask: np.ndarray) -> Tuple[float, List]:
    """Tính diện tích bằng phương pháp hình học (tam giác hóa)."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0.0
    all_shapes = []
    if not contours or hierarchy is None: return 0.0, []

    for i, contour in enumerate(contours):
        if len(contour) < 3: continue
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(np.mean(contour[:, 0, 0]))
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(np.mean(contour[:, 0, 1]))
        centroid = (cx, cy)
        points = np.squeeze(contour)
        contour_shapes = [[centroid, tuple(p1), tuple(p2)] for p1, p2 in zip(points, np.roll(points, -1, axis=0))]
        all_shapes.append(contour_shapes)
        contour_area = sum(Polygon(tri).area for tri in contour_shapes)
        total_area += contour_area if hierarchy[0][i][3] == -1 else -contour_area
    return total_area, all_shapes

def calculate_final_tumor_area(mask: np.ndarray, pixel_area_mm2: float) -> Tuple[float, float]:
    """Hàm chính tính diện tích hình học và pixel."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    geometric_area_pixels, _ = calculate_geometric_area(mask_uint8)
    pixel_count_area_mm2 = np.count_nonzero(mask) * pixel_area_mm2
    return geometric_area_pixels * pixel_area_mm2, pixel_count_area_mm2

# ==============================================================================
# PHẦN 3: CÁC HÀM TRỰC QUAN HÓA (VISUALIZATION)
# ==============================================================================

def display_predictions(model, dataset, num_samples=3):
    """Hiển thị kết quả dự đoán sau khi huấn luyện."""
    plt.figure(figsize=(15, num_samples * 5))
    for i, (img, mask) in enumerate(dataset.take(num_samples)):
        pred_mask = model.predict(img)[0]
        plt.subplot(num_samples, 3, i * 3 + 1); plt.imshow(img[0, ..., 0], cmap='gray'); plt.title("Ảnh gốc"); plt.axis('off')
        plt.subplot(num_samples, 3, i * 3 + 2); plt.imshow(mask[0, ..., 0], cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
        plt.subplot(num_samples, 3, i * 3 + 3); plt.imshow(pred_mask[..., 0], cmap='gray'); plt.title("Dự đoán"); plt.axis('off')
    plt.tight_layout(); plt.show()

def visualize_decomposition(original_image: np.ndarray, mask: np.ndarray, show=True) -> np.ndarray:
    """Trực quan hóa quá trình tam giác hóa để tính diện tích."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    _, all_shapes = calculate_geometric_area(mask_uint8)
    for contour_shapes in all_shapes:
        for triangle in contour_shapes:
            pts = np.array(triangle, dtype=np.int32)
            color = tuple(np.random.randint(50, 256, 3).tolist())
            cv2.polylines(vis_image, [pts], isClosed=True, color=color, thickness=1)
            overlay = vis_image.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, vis_image, 0.8, 0, vis_image)
    if show:
        plt.figure(figsize=(8,8)); plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)); plt.title("Decomposition Visualization"); plt.axis('off'); plt.show()
    return vis_image

def plot_evaluation_results(image, gt_mask, final_mask, triangulated_img, index):
    """Hiển thị kết quả 2x2 cho quá trình đánh giá."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 12)); fig.suptitle(f"Kết Quả Phân Tích - Ảnh Index: {index}", fontsize=18)
    axs[0, 0].imshow(image, cmap='gray'); axs[0, 0].set_title("Ảnh Gốc"); axs[0, 0].axis('off')
    axs[0, 1].imshow(gt_mask, cmap='gray'); axs[0, 1].set_title("Ground Truth"); axs[0, 1].axis('off')
    axs[1, 0].imshow(final_mask, cmap='gray'); axs[1, 0].set_title("Mask Dự Đoán Cuối Cùng"); axs[1, 0].axis('off')
    axs[1, 1].imshow(cv2.cvtColor(triangulated_img, cv2.COLOR_BGR2RGB)); axs[1, 1].set_title("Vùng U Tam Giác Hóa"); axs[1, 1].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

def create_summary_table(results, output_path):
    """Tạo và lưu bảng tổng hợp kết quả phân tích diện tích."""
    if not results: return
    df = pd.DataFrame(results).sort_values(by="index")
    df['diff_geom'] = (df['pred_geom'] - df['gt_geom']).abs()
    df['diff_pixel'] = (df['pred_pixel'] - df['gt_pixel']).abs()
    
    fig, ax = plt.subplots(figsize=(16, len(df) * 0.6 + 1)); ax.axis('tight'); ax.axis('off')
    ax.set_title("Bảng Tổng Hợp Phân Tích Diện Tích (đơn vị: mm²)", fontsize=18, pad=20)
    
    table_data = df[['gt_geom', 'pred_geom', 'diff_geom', 'gt_pixel', 'pred_pixel', 'diff_pixel']].round(2).values
    row_labels = [f"Index {idx}" for idx in df['index']]
    col_labels = ["GT DT (HH)", "Dự đoán (HH)", "Chênh lệch (HH)", "GT DT (Pixel)", "Dự đoán (Pixel)", "Chênh lệch (Pixel)"]
    
    table = ax.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('gray')
        if row == 0: cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#40466e')
        if col == -1: cell.set_text_props(weight='bold'); cell.set_facecolor('#f2f2f2')
        if col in [2, 5]: cell.set_facecolor('#e6e6fa')

    plt.tight_layout(); fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\n✅ Bảng tổng hợp đã được lưu tại: {output_path}"); plt.show()
