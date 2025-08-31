# Brain-Tumor-Segmentation-Using-Deep-Learning
This thesis focuses on applying deep learning techniques for brain tumor segmentation in medical images, followed by accurate computation of the tumor area. The primary goal is to improve the precision of medical image analysis, thereby providing effective support for clinical diagnosis and treatment planning.
** **
- Data source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- 
**Giới thiệu tổng quan**

Với sự bùng nổ của học sâu trong thị giác máy tính, ứng dụng của nó trong phân tích ảnh y tế ngày càng trở nên quan trọng. Trong đó, phân đoạn khối u não từ ảnh cộng hưởng từ (MRI) là bước then chốt, quyết định hiệu quả chẩn đoán, lập kế hoạch điều trị và tiên lượng cho bệnh nhân. Tuy nhiên, đặc điểm hình dạng bất thường và ranh giới mờ của khối u khiến việc phân đoạn thủ công vừa khó khăn vừa thiếu nhất quán.

Để giải quyết thách thức này, chúng tôi đề xuất một hệ thống lai (hybrid) hoàn toàn tự động, kết hợp mô hình Attention U-Net – một biến thể nâng cao của U-Net với cơ chế chú ý – và các thuật toán xử lý ảnh cổ điển như lọc Median, phân ngưỡng thích ứng và đường viền chủ động (Active Contour). Bên cạnh phân đoạn, hệ thống còn cung cấp phép đo định lượng chính xác diện tích khối u bằng phương pháp hình học, thay vì chỉ đếm pixel, cho kết quả bằng đơn vị mm².

Hệ thống được đánh giá bằng các chỉ số Dice, IoU, Precision và Recall, đạt độ chính xác cao và khẳng định tiềm năng ứng dụng như một công cụ hỗ trợ lâm sàng mạnh mẽ, hiệu quả và đáng tin cậy.

**Các Thành Phần Chính Trong Hệ Thống**
Để đạt được mục tiêu nghiên cứu, hệ thống được xây dựng dựa trên các kỹ thuật và tính năng tiên tiến sau:
 - Kiến trúc Attention U-Net:: Sử dụng kiến trúc mạng U-Net được tích hợp các cổng chú ý (Attention Gates), cho phép mô hình tập trung vào những vùng quan trọng chứa khối u và loại bỏ các vùng nền không liên quan, từ đó nâng cao độ chính xác trong phân đoạn.
 - Hàm mất mát kết hợp (Combined Loss): Hệ thống sử dụng đồng thời Dice Loss và Focal Loss để khắc phục vấn đề mất cân bằng dữ liệu giữa vùng khối u (chiếm diện tích nhỏ) và vùng nền (chiếm diện tích lớn), đảm bảo mô hình học được đặc trưng của cả hai lớp.
 - Tăng cường Dữ liệu Nâng cao: Thư viện Albumentations được sử dụng để tạo ra nhiều biến thể dữ liệu đa dạng (xoay, lật, thay đổi độ sáng, tương phản, v.v.), giúp mô hình tăng khả năng khái quát hóa và giảm thiểu hiện tượng overfitting.
 - Hậu xử lý Lai (Hybrid Post-processing): Tự động tinh chỉnh đường viền khối u qua 3 bước: lọc nhiễu bằng Median Blur, làm sắc nét biên bằng phương pháp phân ngưỡng thích ứng (Adaptive Thresholding), và tối ưu hóa đường viền bằng thuật toán đường viền chủ động (Active Contour).
 - Tính toán Diện tích Hình học: Khác với phương pháp đếm pixel thông thường, hệ thống áp dụng kỹ thuật phân rã đa giác thành các tam giác để tính toán diện tích khối u, cho phép đo lường chính xác hơn và có ý nghĩa vật lý (mm²).

Hệ thống được thiết kế theo một kiến trúc hai giai đoạn chính: (1) Huấn luyện mô hình để học cách nhận diện khối u, và (2) Phân tích và Đo lường để áp dụng mô hình và tinh chỉnh kết quả trên dữ liệu thực tế. 

  **Giai đoạn 1**
  
  <img width="325" height="562" alt="image" src="https://github.com/user-attachments/assets/e89b0f56-6e45-4579-adeb-790387d1195f" />

    Quy trình huấn luyện mô hình U-Net để nhận dạng ảnh não được thực hiện theo các bước: 
    đầu tiên xây dựng mô hình U-Net, 
    sau đó tiền xử lý dữ liệu bằng cách thay đổi kích thước ảnh về 512×512. 
    Tiếp theo, dữ liệu được chia thành ba tập gồm tập huấn luyện (70%), tập kiểm định (20%) và tập kiểm tra (10%).
    Dữ liệu đầu vào được tăng cường bằng thư viện Albumentations nhằm nâng cao khả năng khái quát hóa của mô hình. 
    Mô hình U-Net có tích hợp Attention Gate được huấn luyện và đánh giá hiệu suất, cuối cùng áp dụng mô hình đã huấn luyện để nhận dạng ảnh não.





    
    
  **Giai đoạn 2**

  
  <img width="589" height="836" alt="image" src="https://github.com/user-attachments/assets/3e0ec96a-99a4-420f-900a-287b64ccf6c6" />





    



   
