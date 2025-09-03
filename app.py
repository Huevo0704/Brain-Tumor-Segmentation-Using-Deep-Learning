# app.py

import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Import các hàm đã được tổ chức lại từ utils.py
from utils import (
    combined_loss, dice_coefficient, iou, sensitivity, specificity,
    segment_tumor_adaptive, calculate_final_tumor_area, visualize_decomposition
)

# --- CẤU HÌNH VÀ TẢI MODEL ---
MODEL_PATH = "U_Net_model_attention_512.keras" # File model phải nằm cùng thư mục
IMG_SIZE = 512
KNOWN_IMAGE_DPI = 96
INCH_TO_MM = 25.4

# Hàm tải model (chạy 1 lần)
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
def full_analysis_pipeline(
    input_image, dpi,
    median_kernel,
    adaptive_block_size, adaptive_c,
    ac_alpha, ac_beta, ac_gamma
):
    if input_image is None:
        return None, None, None, None, "Vui lòng tải ảnh lên."

    # --- 1. Tiền xử lý và Dự đoán U-Net ---
    original_shape = input_image.shape[:2]
    image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) if len(input_image.shape) > 2 else input_image
    image_resized = cv2.resize(image_gray, (IMG_SIZE, IMG_SIZE))
    input_tensor = np.expand_dims(image_resized / 255.0, axis=[0, -1]).astype(np.float32)
    
    pred_mask_unet = model.predict(input_tensor)[0]
    pred_mask_unet_binary = (pred_mask_unet > 0.5).astype(np.uint8).squeeze()
    
    if np.count_nonzero(pred_mask_unet_binary) == 0:
        return input_image, None, None, None, "Mô hình U-Net không phát hiện thấy khối u."

    # --- 2. Hậu xử lý Giai đoạn 1: Median Blur ---
    if median_kernel % 2 == 0: median_kernel += 1 # Kernel phải là số lẻ
    mask_median = (cv2.medianBlur(pred_mask_unet_binary * 255, median_kernel) > 127).astype(np.uint8)

    # --- 3. Hậu xử lý Giai đoạn 2: Adaptive Thresholding ---
    mask_adaptive = segment_tumor_adaptive(mask_median * 255, adaptive_block_size, adaptive_c)

    # --- 4. Hậu xử lý Giai đoạn 3: Active Contour (Snake) ---
    final_mask = mask_adaptive # Bắt đầu với mask từ bước trước
    smoothed_image = gaussian(image_resized, sigma=1)
    contours, _ = cv2.findContours(mask_adaptive * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        try:
            initial_snake = np.squeeze(max(contours, key=cv2.contourArea))
            if len(initial_snake) > 2: # Cần ít nhất 3 điểm để tạo contour
                snake = active_contour(smoothed_image, initial_snake, alpha=ac_alpha, beta=ac_beta, gamma=ac_gamma)
                temp_mask = np.zeros_like(final_mask)
                cv2.fillPoly(temp_mask, [snake.astype(np.int32)], 255)
                final_mask = (temp_mask / 255).astype(np.uint8)
        except Exception as e:
            print(f"Lỗi Active Contour: {e}. Sử dụng mask từ bước trước.")
            # Giữ nguyên final_mask = mask_adaptive

    # --- 5. Tính toán diện tích ---
    pixel_area_mm2 = (INCH_TO_MM / dpi) ** 2
    geom_area, pixel_area = calculate_final_tumor_area(final_mask, pixel_area_mm2)
    
    # --- 6. Tạo ảnh kết quả và trực quan hóa ---
    # Resize mask cuối cùng về kích thước ảnh gốc
    final_mask_resized = cv2.resize(final_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Tạo ảnh tô màu
    output_image_color = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR) if len(input_image.shape) < 3 else input_image.copy()
    red_overlay = np.zeros_like(output_image_color)
    red_overlay[final_mask_resized == 1] = [0, 0, 255] # BGR
    final_colored_image = cv2.addWeighted(output_image_color, 1.0, red_overlay, 0.6, 0)

    # Tạo ảnh tam giác hóa
    triangulated_image = visualize_decomposition(cv2.resize((image_resized * 255).astype(np.uint8), (512, 512)), final_mask)
    
    # Tạo chuỗi kết quả
    result_text = (
        f"--- KẾT QUẢ PHÂN TÍCH ---\n"
        f"Diện tích (Đếm Pixel): {pixel_area:.2f} mm²\n"
        f"Diện tích (Hình học): {geom_area:.2f} mm²"
    )

    return final_colored_image, final_mask_resized * 255, triangulated_image, result_text


# --- TẠO GIAO DIỆN GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 Demo Phân tích và Phân đoạn Khối u Não Toàn diện")
    gr.Markdown("Tải lên ảnh MRI, điều chỉnh các tham số hậu xử lý và xem kết quả phân tích chi tiết.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # CỘT ĐẦU VÀO
            input_image = gr.Image(type="numpy", label="Tải ảnh MRI tại đây")
            dpi_input = gr.Slider(minimum=50, maximum=600, value=96, step=1, label="DPI của ảnh (để tính mm²)")
            
            with gr.Accordion("Tinh chỉnh các bước hậu xử lý", open=False):
                gr.Markdown("### Giai đoạn 1: Median Blur (Làm mịn mask)")
                median_kernel = gr.Slider(minimum=3, maximum=15, value=5, step=2, label="Kích thước Kernel")
                
                gr.Markdown("### Giai đoạn 2: Adaptive Thresholding (Tinh chỉnh biên)")
                adaptive_block_size = gr.Slider(minimum=3, maximum=51, value=21, step=2, label="Block Size")
                adaptive_c = gr.Slider(minimum=1, maximum=20, value=5, label="Giá trị C")
                
                gr.Markdown("### Giai đoạn 3: Active Contour (Làm mượt biên)")
                ac_alpha = gr.Slider(minimum=0.001, maximum=0.1, value=0.01, step=0.001, label="Alpha (Độ co dãn)")
                ac_beta = gr.Slider(minimum=0.01, maximum=1.0, value=0.1, step=0.01, label="Beta (Độ cứng)")
                ac_gamma = gr.Slider(minimum=0.001, maximum=0.1, value=0.01, step=0.001, label="Gamma (Lực bên ngoài)")
            
            analyze_button = gr.Button("Bắt đầu Phân tích", variant="primary")

        with gr.Column(scale=2):
            # CỘT ĐẦU RA
            gr.Markdown("## Kết quả Phân tích")
            output_colored = gr.Image(label="Kết quả Phân đoạn (Đã tô màu)")
            with gr.Row():
                output_mask = gr.Image(label="Mask nhị phân cuối cùng")
                output_triangulated = gr.Image(label="Trực quan hóa tính diện tích")
            
            output_textbox = gr.Textbox(label="Thông tin Diện tích")

    # Kết nối logic
    analyze_button.click(
        fn=full_analysis_pipeline,
        inputs=[
            input_image, dpi_input,
            median_kernel,
            adaptive_block_size, adaptive_c,
            ac_alpha, ac_beta, ac_gamma
        ],
        outputs=[output_colored, output_mask, output_triangulated, output_textbox]
    )
    
    gr.Examples(
        examples=[["demo_images/demo_image_1.jpg", 96, 5, 21, 5, 0.01, 0.1, 0.01]],
        inputs=[
            input_image, dpi_input,
            median_kernel,
            adaptive_block_size, adaptive_c,
            ac_alpha, ac_beta, ac_gamma
        ]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
