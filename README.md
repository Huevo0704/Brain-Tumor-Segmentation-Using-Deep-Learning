# Brain-Tumor-Segmentation-Using-Deep-Learning
This thesis focuses on applying deep learning techniques for brain tumor segmentation in medical images, followed by accurate computation of the tumor area. The primary goal is to improve the precision of medical image analysis, thereby providing effective support for clinical diagnosis and treatment planning.
** **
- Data source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- 
**Giới thiệu tổng quan**

Với sự bùng nổ của học sâu trong thị giác máy tính, ứng dụng của nó trong phân tích ảnh y tế ngày càng trở nên quan trọng. Trong đó, phân đoạn khối u não từ ảnh cộng hưởng từ (MRI) là bước then chốt, quyết định hiệu quả chẩn đoán, lập kế hoạch điều trị và tiên lượng cho bệnh nhân. Tuy nhiên, đặc điểm hình dạng bất thường và ranh giới mờ của khối u khiến việc phân đoạn thủ công vừa khó khăn vừa thiếu nhất quán.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fb4fafc0-5204-47ce-9693-1940c578b107" alt="Segmentation Result" width="750"/>
</p>

Để giải quyết thách thức này, chúng tôi đề xuất một hệ thống lai (hybrid) hoàn toàn tự động, kết hợp mô hình Attention U-Net – một biến thể nâng cao của U-Net với cơ chế chú ý – và các thuật toán xử lý ảnh cổ điển như lọc Median, phân ngưỡng thích ứng và đường viền chủ động (Active Contour). Bên cạnh phân đoạn, hệ thống còn cung cấp phép đo định lượng chính xác diện tích khối u bằng phương pháp hình học, thay vì chỉ đếm pixel, cho kết quả bằng đơn vị mm².

Hệ thống được đánh giá bằng các chỉ số Dice, IoU, Precision và Recall, đạt độ chính xác cao và khẳng định tiềm năng ứng dụng như một công cụ hỗ trợ lâm sàng mạnh mẽ, hiệu quả và đáng tin cậy.

**Các Thành Phần Chính Trong Hệ Thống**
Để đạt được mục tiêu nghiên cứu, hệ thống được xây dựng dựa trên các kỹ thuật và tính năng tiên tiến sau:
 - Kiến trúc Attention U-Net:: Sử dụng kiến trúc mạng U-Net được tích hợp các cổng chú ý (Attention Gates), cho phép mô hình tập trung vào những vùng quan trọng chứa khối u và loại bỏ các vùng nền không liên quan, từ đó nâng cao độ chính xác trong phân đoạn.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5a24561-777f-4c5b-b1e1-3fa82983c9bf" alt="Attention U-Net Architecture" width="550"/>
</p>

<p align="center"><em>Hình 1. Sơ đồ kiến trúc mạng Attention U-Net</em></p>

 - Hàm mất mát kết hợp (Combined Loss): Hệ thống sử dụng đồng thời Dice Loss và Focal Loss để khắc phục vấn đề mất cân bằng dữ liệu giữa vùng khối u (chiếm diện tích nhỏ) và vùng nền (chiếm diện tích lớn), đảm bảo mô hình học được đặc trưng của cả hai lớp.
   
<p align="center">
  <img src="https://github.com/user-attachments/assets/5d67d788-21d6-4150-9ac3-8f531f1c4a48" alt="Segmentation Result" width="550"/>
</p>
 - Tăng cường Dữ liệu Nâng cao: Thư viện Albumentations được sử dụng để tạo ra nhiều biến thể dữ liệu đa dạng (xoay, lật, thay đổi độ sáng, tương phản, v.v.), giúp mô hình tăng khả năng khái quát hóa và giảm thiểu hiện tượng overfitting.
 - Hậu xử lý Lai (Hybrid Post-processing): Tự động tinh chỉnh đường viền khối u qua 3 bước: lọc nhiễu bằng Median Blur, làm sắc nét biên bằng phương pháp phân ngưỡng thích ứng (Adaptive Thresholding), và tối ưu hóa đường viền bằng thuật toán đường viền chủ động (Active Contour).
 - Tính toán Diện tích Hình học: Khác với phương pháp đếm pixel thông thường, hệ thống áp dụng kỹ thuật phân rã đa giác thành các tam giác để tính toán diện tích khối u, cho phép đo lường chính xác hơn và có ý nghĩa vật lý (mm²).

 Các bước thực hiện 
 
 1 Thu thập dữ liệu: Tải bộ dữ liệu ảnh MRI não từ cơ sở dữ liệu Kaggle tại địa chỉ [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

 2 Tiền xử lý và gán nhãn: Giải nén bộ dữ liệu, tiến hành gán nhãn cho các ảnh MRI có khối.

 3 Xây dựng và huấn luyện mô hình: Thiết kế mô hình Attention U-Net, huấn luyện trên tập dữ liệu đã chuẩn bị để học cách phát hiện và phân đoạn khối u não.

 4 Hậu xử lý và đo lường: Áp dụng các kỹ thuật hậu xử lý trên kết quả dự đoán của mô hình, sau đó tính toán diện tích khối u từ các vùng được phân đoạn.






    
    
 




    



   
