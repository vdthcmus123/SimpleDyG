# SimpleDyG trên Kaggle

Bản khuyến nghị để chạy **từ đầu trên hai T4, có checkpoint đầy đủ**: [SimpleDyG_Kaggle_2GPU_Resume.ipynb](notebooks/SimpleDyG_Kaggle_2GPU_Resume.ipynb) và [ZIP source/dataset](notebooks/SimpleDyG_Kaggle_2GPU_Inputs.zip). Đọc [hướng dẫn hai GPU và chạy tiếp](reports/2GPU_VÀ_CHECKPOINT.md). Mỗi GPU nhận một lượt thí nghiệm độc lập; mặc định tạm dừng sau 10,5 giờ và đóng gói checkpoint để tiếp tục ở phiên sau. Bản này đã kiểm thử hai tiến trình trên CPU, chưa chạy xác nhận trên hai T4 thật.

Để chạy tiếp Output đã lưu, dùng [SimpleDyG_Kaggle_2GPU_Continue_Fixed.ipynb](notebooks/SimpleDyG_Kaggle_2GPU_Continue_Fixed.ipynb), mặc định `RESUME=True`. Bản sửa ngày 07/10 xử lý lỗi `Existing output has different config/core/source` khi chuyển từ bundle rút gọn sang full clone. Training runtime và run ID được giữ nguyên; source được đối chiếu với manifest từng file trước khi dựng lại đúng tập file cũ. Xem [chi tiết lỗi và cách tiếp tục](reports/FIX_RESUME_SOURCE_INVENTORY.md).

Bản một GPU trước đây: [SimpleDyG_Kaggle_Full_Pipeline.ipynb](notebooks/SimpleDyG_Kaggle_Full_Pipeline.ipynb). Các hướng dẫn sau đây dành cho bản cũ.
Sau phiên paper bị ngắt, dùng [notebook tiếp tục](notebooks/SimpleDyG_Kaggle_Continue.ipynb) và đọc [hướng dẫn khôi phục](reports/TIẾP_TỤC_SAU_KHI_KAGGLE_HẾT_PHIÊN.md). Gói kết quả nhỏ `recovery/SimpleDyG_Continue_Input.zip` giữ 35 lượt đã hoàn thành; core/cấu hình được giữ nguyên để chia lịch chạy qua nhiều phiên.
Nếu đã gặp lỗi cài Tokenizers trên Python 3.13, dùng [bản sửa notebook đã gửi](notebooks/notebook77b2d6b8b8_fixed.ipynb), restart kernel rồi Run All. Bản sửa giữ cấu hình của notebook người dùng và đã kiểm thử smoke trên Python 3.13; xem [chi tiết sửa lỗi](reports/FIX_PYTHON_313.md).
Đọc [báo cáo rà soát](reports/RÀ_SOÁT_VÀ_HƯỚNG_DẪN_KAGGLE.md) để biết lỗi của các file cũ, protocol và giới hạn benchmark.

1. Import notebook vào Kaggle, bật GPU và Internet. Nếu offline, Add Input [SimpleDyG_Kaggle_Inputs.zip](notebooks/SimpleDyG_Kaggle_Inputs.zip) và các wheel cần thiết khi dependency chưa có; đặt `OFFLINE=True`.
2. Chạy `PROFILE="smoke"` trước, sau đó `pilot` cho full UCI/ML-10M với ba seed. `paper` dùng 10 seed và author_compat; có thể chạy từng dataset/nhóm seed để vừa budget.
3. Chạy các cell theo thứ tự và tải ZIP kết quả ở Output. Kết quả smoke không được dùng làm kết quả reproduce.

Notebook đã chứa toàn bộ mã pipeline; chỉ cần một `.ipynb`. Bundle là input tùy chọn giúp chạy offline. Repo tác giả và các notebook cũ được giữ làm nguồn đối chiếu.

Khi bảo trì source, sửa `tools/kaggle_pipeline.py`/`tools/build_kaggle_notebook.py`, rồi chạy generator bằng Python có `nbformat` để cập nhật notebook. Kiểm tra khoa học/packaging ở `tests/test_kaggle_pipeline.py`. Môi trường `.venv-validation` là môi trường kiểm thử CPU cục bộ, không cần upload Kaggle.

Kết quả 35 lượt từ phiên Kaggle cũ và giới hạn khôi phục được ghi trong báo cáo sau khi hết phiên. Evidence kiểm thử bản một GPU ở `reports/validation_summary.json`; bản hai GPU ở `reports/dual_gpu_resume_validation.json`.
