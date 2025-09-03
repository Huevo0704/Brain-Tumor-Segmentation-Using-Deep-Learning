# app.py

import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Import các hàm cần thiết từ utils.py
from utils import (
    combined_loss, dice_coefficient, iou, sensitivity, specificity,
    segment_tumor_adaptive, calculate_final_tumor_area, calculate_evaluation_metrics
)

# --- CẤU HÌNH VÀ TẢI MODEL ---
MODEL_PATH = "U_Net_model_attention_512.keras"
IMG_SIZE = 512
INCH_TO_MM = 25.4

# Hàm tải model, được cache để tăng tốc
@gr.cache
def load_trained_model(path):
    print("----- Đang tải mô hình... -----")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file model tại '{path}'.")
    custom_objects = {
        'combined_loss': combined_loss,
        'dice_coefficient': dice_coefficient, 'iou': iou,
        'sensitivity': sensitivity, 'specificity': specificity
    }
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)
    print("----- Tải mô hình thành công! -----")
    return model

model = load_trained_model(MODEL_PATH)

# --- HÀM LOGIC CHÍNH ---
def process_and_compare(
    mri_image, gt_mask_image, dpi,
    median_kernel,
    adaptive_block_size, adaptive_c,
    ac_alpha, ac_beta, ac_gamma
):
    if mri_image is None or gt_mask_image is None:
        return None, "Vui lòng tải lên cả ảnh MRI và ảnh Ground Truth."

    # --- 1. Tiền xử lý và Dự đoán U-Net ---
    original_shape = mri_image.shape[:2]
    image_gray = cv2.cvtColor(mri_image, cv2.COLOR_BGR2GRAY) if len(mri_image.shape) > 2 else mri_image
    gt_mask_gray = cv2.cvtColor(gt_mask_image, cv2.COLOR_BGR2GRAY) if len(gt_mask_image.shape) > 2 else gt_mask_image

    image_resized = cv2.resize(image_gray, (IMG_SIZE, IMG_SIZE))
    gt_mask_resized = (cv2.resize(gt_mask_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

    input_tensor = np.expand_dims(image_resized / 255.0, axis=[0, -1]).astype(np.float32)
    pred_mask_unet = (model.predict(input_tensor)[0] > 0.5).astype(np.uint8).squeeze()
    
    if np.count_nonzero(pred_mask_unet) == 0:
        return [
            (mri_image, "Ảnh MRI Gốc"), 
            (gt_mask_image, "Ground Truth")
        ], "Mô hình U-Net không phát hiện thấy khối u."

    # --- 2. Áp dụng chuỗi Hậu xử lý ---
    if median_kernel % 2 == 0: median_kernel += 1
    mask_median = (cv2.medianBlur(pred_mask_unet * 255, median_kernel) > 127).astype(np.uint8)
    mask_adaptive = segment_tumor_adaptive(mask_median * 255, adaptive_block_size, adaptive_c)
    
    final_mask = mask_adaptive
    contours, _ = cv2.findContours(mask_adaptive * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        try:
            snake = active_contour(gaussian(image_resized, 1), np.squeeze(max(contours, key=cv2.contourArea)), 
                                   alpha=ac_alpha, beta=ac_beta, gamma=ac_gamma)
            temp_mask = np.zeros_like(final_mask); cv2.fillPoly(temp_mask, [snake.astype(np.int32)], 255)
            final_mask = (temp_mask / 255).astype(np.uint8)
        except Exception:
            pass # Giữ nguyên mask_adaptive nếu có lỗi

    # --- 3. Tính toán Chỉ số và Diện tích ---
    iou_val, dice_val, prec_val, rec_val = calculate_evaluation_metrics(gt_mask_resized, final_mask)
    pixel_area_mm2 = (INCH_TO_MM / dpi) ** 2
    gt_geom_area, gt_pixel_area = calculate_final_tumor_area(gt_mask_resized, pixel_area_mm2)
    pred_geom_area, pred_pixel_area = calculate_final_tumor_area(final_mask, pixel_area_mm2)

    # --- 4. Chuẩn bị kết quả đầu ra ---
    # Tạo ảnh tô màu
    final_mask_original_size = cv2.resize(final_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    output_colored = cv2.cvtColor(mri_image, cv2.COLOR_GRAY2BGR) if len(mri_image.shape) < 3 else mri_image.copy()
    red_overlay = np.zeros_like(output_colored); red_overlay[final_mask_original_size == 1] = [0, 0, 255]
    final_colored_image = cv2.addWeighted(output_colored, 1.0, red_overlay, 0.6, 0)
    
    # Tạo Gallery ảnh
    image_gallery = [
        (mri_image, "Ảnh MRI Gốc"),
        (gt_mask_image, "Ground Truth"),
        (final_colored_image, "Dự đoán của Model")
    ]
    
    # Tạo chuỗi Markdown kết quả
    result_text = f"""
    ### Bảng Chỉ số Đánh giá
    | Chỉ số | Giá trị |
    | :--- | :---: |
    | **Dice Coefficient** | `{dice_val:.4f}` |
    | **IoU (Jaccard)** | `{iou_val:.4f}` |
    | **Precision** | `{prec_val:.4f}` |
    | **Recall** | `{rec_val:.4f}` |

    ### So sánh Diện tích Khối u (mm²)
    | Loại diện tích | Ground Truth | Dự đoán | Chênh lệch |
    | :--- | :---: | :---: | :---: |
    | **Đếm Pixel** | `{gt_pixel_area:.2f}` | `{pred_pixel_area:.2f}` | `{abs(gt_pixel_area - pred_pixel_area):.2f}` |
    | **Hình học** | `{gt_geom_area:.2f}` | `{pred_geom_area:.2f}` | `{abs(gt_geom_area - pred_geom_area):.2f}` |
    """
    return image_gallery, result_text

# --- TẠO GIAO DIỆN GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 Demo Đánh giá và Phân tích Mô hình Phân đoạn Khối u Não")
    gr.Markdown("Tải lên ảnh MRI và ảnh mặt nạ Ground Truth tương ứng để so sánh trực quan, xem các chỉ số đánh giá và phân tích diện tích.")

    with gr.Row():
        with gr.Column(scale=1):
            mri_input = gr.Image(type="numpy", label="1. Tải ảnh MRI")
            gt_mask_input = gr.Image(type="numpy", label="2. Tải ảnh Ground Truth Mask")
            dpi_input = gr.Slider(minimum=50, maximum=600, value=96, step=1, label="DPI của ảnh (để tính mm²)")
            
            with gr.Accordion("⚙️ Tùy chọn Hậu xử lý (Tùy chọn)", open=False):
                median_kernel = gr.Slider(3, 15, 5, step=2, label="Median Blur Kernel")
                adaptive_block_size = gr.Slider(3, 51, 21, step=2, label="Adaptive Threshold Block Size")
                adaptive_c = gr.Slider(1, 20, 5, label="Adaptive Threshold C Value")
                ac_alpha = gr.Slider(0.001, 0.1, 0.01, label="Active Contour Alpha")
                ac_beta = gr.Slider(0.01, 1.0, 0.1, label="Active Contour Beta")
                ac_gamma = gr.Slider(0.001, 0.1, 0.01, label="Active Contour Gamma")

            analyze_button = gr.Button("So sánh và Phân tích", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("## So sánh Trực quan")
            output_gallery = gr.Gallery(label="So sánh kết quả", columns=3, object_fit="contain", height="auto")
            gr.Markdown("## Kết quả Đánh giá và Phân tích")
            output_markdown = gr.Markdown()
    
    analyze_button.click(
        fn=process_and_compare,
        inputs=[
            mri_input, gt_mask_input, dpi_input,
            median_kernel, adaptive_block_size, adaptive_c,
            ac_alpha, ac_beta, ac_gamma
        ],
        outputs=[output_gallery, output_markdown]
    )
    
    gr.Examples(
        examples=[
            ["demo_images/demo_image_1.jpg", "demo_images/demo_gt_1.png", 96, 5, 21, 5, 0.01, 0.1, 0.01]
        ],
        inputs=[
            mri_input, gt_mask_input, dpi_input,
            median_kernel, adaptive_block_size, adaptive_c,
            ac_alpha, ac_beta, ac_gamma
        ]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
