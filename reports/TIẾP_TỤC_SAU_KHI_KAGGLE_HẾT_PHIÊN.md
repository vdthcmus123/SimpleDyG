# Tiếp tục phiên paper bị ngắt ngày 06/10/2026

Nguồn kiểm tra: log và `C:/Users/ASUS/Downloads/simpledyg_paper_author_compat.zip`.
ZIP gốc có 2.412.290.995 byte, thiếu central directory/end record và đang ghi dở
`runs/UCI_13_SimpleDyG_no_time_s8_ddee4221376b/best.pt`. ZIP không mở được theo chuẩn.
Đây là lỗi đóng gói khi phiên hết thời gian; log không có traceback/FAILED của training.

## Kết quả đã giữ được

Manifest đã ghi đầy đủ 35 lượt hoàn thành của lịch 100 lượt:

| Dataset | Model/biến thể | Seed hoàn thành | Số lượt |
|---|---|---|---:|
| UCI | SimpleDyG/full | 0–9 | 10 |
| UCI | SimpleDyG/same_time | 0–9 | 10 |
| UCI | SimpleDyG/no_time | 0–9 | 10 |
| UCI | TGAT | 0–4 | 5 |

Còn 65 lượt: UCI TGAT seed 5–9 (5), UCI GraphMixer seed 0–9 (10),
ML-10M cả ba biến thể SimpleDyG và hai baseline (50).

Có 126 local record hoàn chỉnh trước file cuối bị cắt. Đã tạo ZIP mới từ những
record này, giữ byte của payload rồi dựng lại central directory. 17 `result.json`
đã có nguyên file; 18 file kết quả còn thiếu được khôi phục **nguyên nội dung record**
trong manifest. Không chỉnh số đo, không tạo điểm số mới. Cấu hình từng run còn thiếu
được chép từ cấu hình gốc chung. 16 checkpoint cứu được đã qua kiểm tra CRC;
19 checkpoint và 18 file test prediction không nằm trong phần archive đã ghi đủ.
Các metric của 35 lượt vẫn còn. Thư mục Output gốc trên Kaggle có thể vẫn giữ những
file thiếu trong ZIP; nên lưu/attach thư mục đó nếu còn truy cập được.

## File sử dụng

- `notebooks/SimpleDyG_Kaggle_Continue.ipynb`: notebook tiếp tục.
- `recovery/SimpleDyG_Continue_Input.zip`: khoảng 839 KB, chứa kết quả/metadata và
  các file prediction đã cứu được; không chứa checkpoint.
- `recovery/simpledyg_paper_author_compat_recovered.zip`: khoảng 2,326 GB,
  bản backup có 16 checkpoint và kết quả đã khôi phục.
- `reports/kaggle_timeout_recovery.json`: danh sách chính xác các file mất/khôi phục.

## Chạy tiếp

1. Import notebook Continue. Tạo Dataset từ ZIP Input nhỏ và Add Input dataset đó;
   giữ input repo SimpleDyG/DyGLib và dữ liệu cũ, bật GPU.
   Cũng có thể dùng Output thư mục đầy đủ của phiên trước làm nguồn khôi phục.
2. Để `PREVIOUS_OUTPUT_PATH=""` nếu chỉ có một Input kết quả cũ. Nếu có nhiều
   snapshot, điền đúng ZIP/thư mục chứa `config.json`, `source_hashes.json`,
   `data_audit.json` và `runs/`.
3. Run All. Cell khôi phục phải báo 35/100 lượt hoàn thành, source/data/core
   fingerprint khớp. Mặc định huấn luyện hai lượt TGAT UCI tiếp theo, seed 5 và 6.
4. Lưu Output mới; phiên sau Add Input **Output mới nhất** và bỏ Input snapshot cũ.
   Notebook mặc định sẽ chọn hai seed TGAT còn thiếu tiếp theo.
5. Khi đủ TGAT, đặt `SESSION_MODELS=["GraphMixer"]`, `MAX_NEW_RUNS=1`.
   Sau UCI, đặt `SESSION_DATASETS=["ML_10M_13"]` và chọn model cần chạy.

Các bộ lọc SESSION_* ở ngoài cấu hình thí nghiệm. Không đổi CFG, seed gốc,
history cap, epoch, protocol, dimension candidates hay export_checkpoints để
chia phiên: fingerprint cũ bao gồm những trường đó. Notebook tiếp tục giữ nguyên
core pipeline SHA-256 `2134687a7c5548d638a3ecddc8b1d0bb3fd2837177abf588a02f1e6ae2170406`.
Nó dừng trước training nếu source/data/core khác, thay vì âm thầm train lại.

## Giới hạn checkpoint và thời gian

`best.pt` chỉ chứa trọng số được chọn bằng validation. Phiên bản cũ không lưu
optimizer, scheduler, scaler, RNG hoặc vị trí batch, nên không hỗ trợ tiếp tục
đúng epoch của run đang dở. Run có `result.json` hoàn thành được tái sử dụng dù
checkpoint không có trong Input nhỏ. Run đang dở phải chạy lại; không tự dùng
best.pt để fine-tune rồi gọi đó là cùng thí nghiệm.

MAX_SESSION_HOURS chỉ ngăn bắt đầu run mới khi hết budget; không dừng run đang chạy.
Notebook tiếp tục ghi ZIP mới vào `.writing`, chỉ thay ZIP cũ khi ghi và đóng xong.
Nếu phiên ngắt, ưu tiên lưu/attach toàn bộ thư mục Output vì result.json được ghi
trước export; ZIP hoàn chỉnh gần nhất có thể thiếu run cuối vừa hoàn thành.

## Xác minh

- ZIP backup và ZIP Input nhỏ: mọi member qua kiểm tra CRC.
- Source 12 file và core khớp phiên Kaggle; SHA dữ liệu và 100 run ID được kiểm tra.
- 5 kiểm tra tự động đạt: lập lịch 35/100 từ dữ liệu thật, tái sử dụng kết quả và
  huấn luyện tiếp hai phiên CPU smoke, giữ run ID/cấu hình/metric, bảo toàn ZIP cũ
  khi mô phỏng ngắt export, chặn path traversal và khôi phục ZIP bị cắt.
- Notebook mới hợp lệ về schema/cú pháp, core giống notebook cũ. Chưa chạy lại
  các lượt paper còn thiếu trên GPU từ máy này.

Kaggle hỗ trợ dùng Notebook Output làm nguồn dữ liệu: [tài liệu Datasets](https://www.kaggle.com/docs/datasets).
