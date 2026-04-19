"1. Chủ đề Ứng dụng Deep Reinforcement Learning để tối ưu hóa điều phối tác vụ trong hệ thống Hybrid Edge-Cloud. 2. Nội dung thực hiện dự kiến
Thiết lập hạ tầng Hybrid Cloud: Xây dựng cụm K3s đóng vai trò các Node biên (Edge) trên môi trường , kết nối với cụm AWS EKS/EC2 (Cloud) và cài đặt hệ thống giám sát Prometheus để thu thập metrics.
Mô hình hóa bài toán học tăng cường (RL): Xác định các thành phần cốt lõi bao gồm trạng thái hệ thống (State - CPU, RAM, Latency), tập hành động (Action - chọn nơi xử lý) và hàm thưởng (Reward) để cân bằng giữa tốc độ xử lý và chi phí tài nguyên.
Xây dựng môi trường giả lập (Simulation): Sử dụng thư viện Gymnasium để tạo môi trường ảo, giúp huấn luyện AI (thuật toán PPO hoặc DQN) thử sai hàng ngàn lần một cách nhanh chóng trước khi đưa vào hệ thống thật.
Phát triển bộ điều phối thông minh (Smart Dispatcher): Viết script Python đóng vai trò ""trung tâm điều khiển"", có nhiệm vụ đọc dữ liệu thời gian thực từ Prometheus, nạp mô hình AI đã huấn luyện và thực hiện điều hướng task tới node tối ưu nhất. 3. Kết quả dự kiến đạt được
Giảm độ trễ phản hồi (Latency): Chứng minh được AI có khả năng chọn vị trí xử lý giúp hoàn thành tác vụ nhanh hơn đáng kể so với việc chỉ sử dụng Edge hoặc chỉ sử dụng Cloud đơn thuần.
Tối ưu hóa tài nguyên thiết bị biên: Hệ thống tự động đẩy các tác vụ nặng (compute-intensive) lên Cloud đúng lúc, giúp các node K3s tránh được tình trạng quá tải và tiết kiệm điện năng.
Khả năng thích nghi linh hoạt: Hệ thống vẫn hoạt động ổn định và đưa ra quyết định chính xác ngay cả khi môi trường mạng có biến động lớn hoặc tài nguyên giữa các node thay đổi đột ngột.
Tỉ lệ hoàn thành tác vụ (SLA) vượt trội: Giảm thiểu tối đa số lượng task bị quá hạn (deadline miss) so với các phương pháp lập lịch truyền thống như Round-Robin hay Least Connection.
Hệ thống Demo trực quan: Xây dựng Dashboard trên Grafana hiển thị thời gian thực quá trình AI ""suy nghĩ"" và ra quyết định đẩy task đi đâu dựa trên biểu đồ tài nguyên."
State: CPU, RAM, Latency
Action: quyết định đẩy task đi đâu dựa trên biểu đồ tài nguyên
Environment: K3s + AWS/Gymnasium
Reward: tính toán dựa trên log thời gian hoàn thành task so vs deadline
