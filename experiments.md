Ý tưởng cốt lõi: Đưa hiệu ứng "Nghẽn cổ chai Lập lịch" (Scheduler Bottleneck) vào file Calibration bằng một hàm Phạt phi tuyến tính (Non-linear Penalty) dựa trên Độ dài hàng đợi (Queue Length).

Chúng ta biết rằng K8s Master xử lý rất mượt 5-10 Pod cùng lúc, nhưng nếu nhồi 50 Pod, thời gian khởi tạo (Startup Time) sẽ bị cộng dồn lên tới hàng chục giây. Vậy ta chỉ cần sửa lại công thức mô phỏng tính toán thời gian predict_total_ms.

Kế hoạch sửa đổi (Chỉ cần 2 bước):
Bước 1: Trong file calibration/calibrated_constants.py, tôi sẽ sửa hàm predict_total_ms bằng cách thêm tham số queue_length. Tôi sẽ cài đặt logic: "Thời gian khởi động là hằng số (709ms), NHƯNG nếu hàng đợi vượt quá 5 task, thì cứ mỗi task xếp hàng dư ra, thời gian khởi động sẽ bị cộng dồn thêm 1.5 giây." (VD: Có 50 task xếp hàng -> Thời gian khởi động = $0.7s + (50 - 5) \times 1.5s = 68.2$ giây!)

Bước 2: Trong file rl_env/edge_cloud_env_calibrated.py, tôi sẽ bổ sung tham số queue_length lấy từ môi trường ảo nạp vào hàm predict_total_ms.
cách
