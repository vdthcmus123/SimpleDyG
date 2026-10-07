# Sửa lỗi setup Kaggle Python 3.13

Notebook lỗi do người dùng cung cấp: `C:/Users/ASUS/Downloads/notebook77b2d6b8b8.ipynb`.
Traceback ở cell Dependency: `/usr/lib/python3.13/subprocess.py`, `CalledProcessError` khi cài `transformers==4.44.2` và `tokenizers==0.19.1`.
PyPI release tokenizers 0.19.1 không có wheel CPython 3.13; pip chọn source distribution rồi lỗi khi build wheel.

Bản đầu đã được kiểm thử trên Python 3.12, chưa bao phủ môi trường Python 3.13 này.

## Thay đổi

- Setup không còn cài/ép phiên bản Transformers hay Tokenizers; không yêu cầu Rust.
- Tokenization vẫn là `NodeTokenizer` nhúng, giữ mapping node và special token của release.
- Load chính source GPT-2 của tác giả, chỉ chuyển hai activation import sang GELU/get_activation tương đương trong runtime; file repo không bị sửa.
- Chế độ `author_compat` dùng AdamW legacy nhúng, giữ epsilon/bias-correction và scheduler tuyến tính của HF bản cũ. `corrected` vẫn dùng PyTorch AdamW.
- Environment manifest ghi rõ runtime nhúng và không import thư viện HF để lấy version.
- Các dependency phụ còn thiếu được cài bằng wheel (`--only-binary=:all:`), giữ PyTorch/CUDA có sẵn trên Kaggle.
- `notebooks/notebook77b2d6b8b8_fixed.ipynb` giữ cấu hình trong notebook người dùng gửi; thay cell setup/core bằng bản sửa và xóa output lỗi cũ.

## Kiểm thử

Suite kiểm tra phép tính bổ sung đối chiếu trực tiếp bản HF cũ trên Python 3.12: model initialization/forward/gradient, GELU, optimizer và scheduler.
Đường chạy notebook được kiểm tra với import `transformers`/`tokenizers` bị chặn, để phát hiện mọi dependency còn sót.
Python 3.12: 13/13 kiểm tra đạt, gồm các phép đối chiếu trực tiếp HF cũ. Python 3.13: 10 kiểm tra đạt; 3 kiểm tra đối chiếu HF được bỏ qua vì môi trường này chủ động không cài HF.
Toàn bộ 7 code cell đã chạy thành công trên cả Python 3.12 và 3.13 với import HF bị chặn: 5 lượt smoke hoàn tất, input offline được xác minh và chạy lại tái sử dụng kết quả đã hoàn tất.
Evidence và hash của bản notebook cuối ở `reports/python313_fix_validation.json`.

Chưa chạy full training Kaggle GPU. Việc sửa setup không phải xác nhận điểm số reproduce.

## Chạy lại trên Kaggle

Import file `_fixed.ipynb`, bật GPU và Internet như cấu hình trước. Restart session/kernel rồi Run All từ cell đầu.
Nếu muốn kiểm tra nhanh trước, đổi `PROFILE="smoke"`, chạy xong mới đổi về `pilot`.
Input bundle cũ vẫn sử dụng được; dữ liệu và commit không thay đổi.

Nguồn: [tokenizers 0.19.1 files](https://pypi.org/project/tokenizers/0.19.1/#files), [HF optimizer v4.44.2](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/optimization.py).
