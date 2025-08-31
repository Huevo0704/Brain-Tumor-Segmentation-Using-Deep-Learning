# Brain-Tumor-Segmentation-Using-Deep-Learning
This thesis focuses on applying deep learning techniques for brain tumor segmentation in medical images, followed by accurate computation of the tumor area. The primary goal is to improve the precision of medical image analysis, thereby providing effective support for clinical diagnosis and treatment planning.
** **
- Data source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
**Giới thiệu tổng quan**

Với sự bùng nổ của học sâu trong thị giác máy tính, ứng dụng của nó trong phân tích ảnh y tế ngày càng trở nên quan trọng. Trong đó, phân đoạn khối u não từ ảnh cộng hưởng từ (MRI) là bước then chốt, quyết định hiệu quả chẩn đoán, lập kế hoạch điều trị và tiên lượng cho bệnh nhân. Tuy nhiên, đặc điểm hình dạng bất thường và ranh giới mờ của khối u khiến việc phân đoạn thủ công vừa khó khăn vừa thiếu nhất quán.

Để giải quyết thách thức này, chúng tôi đề xuất một hệ thống lai (hybrid) hoàn toàn tự động, kết hợp mô hình Attention U-Net – một biến thể nâng cao của U-Net với cơ chế chú ý – và các thuật toán xử lý ảnh cổ điển như lọc Median, phân ngưỡng thích ứng và đường viền chủ động (Active Contour). Bên cạnh phân đoạn, hệ thống còn cung cấp phép đo định lượng chính xác diện tích khối u bằng phương pháp hình học, thay vì chỉ đếm pixel, cho kết quả bằng đơn vị mm².

Hệ thống được đánh giá bằng các chỉ số Dice, IoU, Precision và Recall, đạt độ chính xác cao và khẳng định tiềm năng ứng dụng như một công cụ hỗ trợ lâm sàng mạnh mẽ, hiệu quả và đáng tin cậy.
