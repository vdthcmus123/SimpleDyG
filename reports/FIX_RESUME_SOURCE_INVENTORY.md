# Sửa lỗi chạy tiếp khi source bundle và full clone có inventory khác nhau

Ngày 07/10/2026. Log đính kèm dừng ở `Existing output has different config/core/source. Do not mix experiments.` trước khi khởi động worker. Log cũng đã xác nhận đọc đúng thư mục:

`/kaggle/input/notebooks/thuvuduy16/notebook77b2d6b8b8/simpledyg_2gpu_results`

Runtime hash của cả hai phiên là `e6e6142856ea68e46feba431bdfbf9748380818caffaa37c604745af7d972ab8`. Phiên thành công gần nhất đã lưu 34 run complete, 2 paused và 64 pending, cùng ZIP 6.351.245.097 byte. Lần lỗi dừng trước train nên không tạo thêm kết quả training.

Source bundle trước gồm 12 file Python được dùng bởi pipeline. Lần chạy tiếp thiếu bundle và clone repo đầy đủ. Core cũ tính fingerprint trên mọi file `.py` trong cây source; các file bổ sung của full clone làm inventory khác dù những file dùng để train có thể giữ nguyên nội dung. Lỗi này đã tái hiện bằng fixture có hai file bổ sung; bỏ các file đó khỏi source view sau khi kiểm tra checksum khôi phục đúng fingerprint và run ID cũ.

## File bàn giao và cách chạy

1. Import [SimpleDyG_Kaggle_2GPU_Continue_Fixed.ipynb](../notebooks/SimpleDyG_Kaggle_2GPU_Continue_Fixed.ipynb).
2. Add Input Output của phiên đã lưu thành công với 34 complete / 2 paused / 64 pending. Giữ ZIP source/dataset hai GPU cũ nếu đã gắn; gói đó vẫn tương thích. Có thể gắn trực tiếp Output trên Kaggle, không cần tải xuống rồi upload lại ZIP 6,35 GB.
3. Chọn GPU T4 x2 và cùng môi trường Python/PyTorch/GPU type. Bản này mặc định `RESUME=True`, `RESUME_FROM=""`; để trống nếu chỉ gắn một bộ Output.
4. Run All. Khi inventory có file thêm, cần thấy `RESUME source aligned: ... verified Python files; ... extra files excluded; original runtime/config identity preserved.` Sau đó các checkpoint đang dở có log `RESUME ... epoch=... next_batch=... phase=...`.
5. Giữ Output/snapshot mới sau mỗi phiên. Nếu có nhiều bộ Output trong Input, chọn đúng bộ bằng `RESUME_FROM`. Không đổi config khoa học khi resume.

Notebook chạy từ đầu cũng được cập nhật phần sửa này, nhưng vẫn mặc định `RESUME=False`. ZIP input trước và sau bản sửa có cùng source/data; không cần thay gói input cũ để khôi phục checkpoint.

## Cách sửa và các kiểm tra được giữ

Phần sửa nằm trong `tools/resume_source_alignment.py`, được nhúng vào cell cấu hình notebook. Training runtime `tools/kaggle_pipeline_dual.py` giữ nguyên byte và fingerprint, nên định dạng checkpoint và cách tính run ID không thay đổi.

Helper đọc `source_hashes.json` từ Output, xác nhận manifest khớp fingerprint trong config và runtime khớp phiên trước. Mọi file source đã ghi nhận phải tồn tại và đúng SHA-256. Nếu full clone có file bổ sung, helper sao chép đúng những file đã xác nhận vào một thư mục mới trong Working, cùng dữ liệu/vocab/raw graph từ `data_audit.json`, kiểm tra lại hash dữ liệu, rồi yêu cầu core tự tính fingerprint/config bằng đúng cây source đó. Chỉ tiếp tục khi kết quả bằng config đã lưu.

Source gốc, Input, checkpoint và fingerprint đã lưu không được chỉnh sửa để vượt kiểm tra. Nội dung source hoặc dữ liệu khác, runtime khác và môi trường khác vẫn bị từ chối. Helper chỉ xử lý sự khác biệt do các file source bổ sung không thuộc inventory cũ.

## Evidence

- 7 kiểm thử regression qua: inventory giống nhau, lỗi do file thêm và cách sửa, từ chối source/dữ liệu/runtime đổi nội dung, bảo toàn audit của cả UCI và ML-10M, hai baseline worker tạm dừng rồi khôi phục đầy đủ checkpoint TGAT/GraphMixer với run ID giữ nguyên.
- Thực thi toàn bộ code cell của notebook bàn giao từ đầu, sau đó resume bằng ZIP và bằng thư mục Output từ source có file bổ sung. Các run hoàn thành và best checkpoint được tái sử dụng.
- CPU cục bộ Python 3.13; chưa chạy bản sửa trực tiếp trên Kaggle T4. Log của người dùng trước đó xác nhận hai T4 thật đã chạy và snapshot đã đóng gói thành công.

[Báo cáo kiểm thử](resume_source_fix_validation.json) · [Evidence thực thi notebook](resume_source_fix_notebook_validation.json).
