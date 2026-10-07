# Chạy từ đầu trên hai T4 và tiếp tục từ checkpoint

Cập nhật: 07/10/2026.

Phiên trước chạy tuần tự trên `cuda:0`; nó không phân chia thí nghiệm cho hai GPU. Bản mới dùng hai tiến trình Python. Mỗi tiến trình chỉ nhìn thấy một GPU và lấy một run độc lập từ hàng đợi chung. Hai run có thể chạy đồng thời; khi một worker xong, nó nhận run kế tiếp. Đây là song song giữa các thí nghiệm, không thay batch size bằng cách nhân với số GPU.

## Hai file cần dùng

- [Notebook chạy từ đầu](../notebooks/SimpleDyG_Kaggle_2GPU_Resume.ipynb): toàn bộ runtime nằm trong notebook, không cần upload thêm file `.py`.
- [Notebook chạy tiếp đã sửa source inventory](../notebooks/SimpleDyG_Kaggle_2GPU_Continue_Fixed.ipynb): mặc định `RESUME=True`, dùng với Output của phiên gần nhất.
- [ZIP source và dữ liệu mới](../notebooks/SimpleDyG_Kaggle_2GPU_Inputs.zip): khoảng 1,4 MB; chứa source và dataset đã preprocess của bundle cũ, thêm manifest của bản hai GPU. Đây là gói input để chạy từ đầu, không phải ZIP kết quả 2,41 GB của phiên trước.

Repo tác giả và DyGLib giữ nguyên commit được pin. Các file của phiên cũ và gói khôi phục 35 lượt được giữ lại để đối chiếu. Bản mới bắt đầu một chiến dịch thí nghiệm riêng; không đưa 35 kết quả cũ vào bảng kết quả mới.

## Phiên đầu

1. Import notebook mới vào Kaggle; Add Input ZIP source/dataset mới. Kaggle có thể giữ ZIP hoặc giải nén nó thành thư mục; notebook hỗ trợ cả hai trường hợp.
2. Trong Settings, chọn **GPU T4 x2** và bật Internet cho dependency còn thiếu. Setup giữ PyTorch/CUDA của image hiện tại, không cài `transformers`/`tokenizers`.
3. Giữ `PROFILE="paper"`, `GPU_IDS=[0,1]` và `RESUME=False`. Mặc định chạy UCI + ML-10M, 10 seed, SimpleDyG với `full/same_time/no_time`, cùng TGAT và GraphMixer: **100 run**, bao gồm tìm dimension baseline trên validation. Có thể thử `smoke` trong một phiên riêng trước khi bắt đầu chiến dịch paper.
4. Run All. Log `Visible GPUs: 2`, hai dòng tên GPU và `Two workers READY` xác nhận hai worker đã khởi động. Các dòng `[worker 0 / device 0]` và `[worker 1 / device 1]` cho biết worker đang nhận và chạy thí nghiệm nào.
5. Khi kết thúc hoặc tạm dừng, lưu Version/Output của Kaggle. Giữ toàn bộ `simpledyg_2gpu_results/`, đặc biệt `resume_snapshot.zip`, `runs/`, config và manifest. File snapshot là gói dùng cho phiên tiếp theo.

`SESSION_HOURS=10.5` tính từ lúc bắt đầu bộ điều phối, dành thời gian còn lại để đóng ZIP và lưu Output trước giới hạn 12 giờ. Nó không đảm bảo toàn bộ 100 run xong trong một phiên. Hai T4 cũng không cộng VRAM cho một model; từng run vẫn phải vừa bộ nhớ của một GPU.

## Checkpoint lưu những gì

`CHECKPOINT_SECONDS=300` lưu theo khoảng năm phút tại ranh giới optimizer step; runtime còn lưu trước validation và ở mỗi ranh giới epoch. Khi đến deadline, worker lưu trạng thái rồi tạm dừng. Checkpoint đang train là `latest.pt` của SimpleDyG hoặc `latest_dimN.pt` của baseline.

Checkpoint gồm trọng số hiện tại, trọng số tốt nhất theo validation, optimizer, scheduler/scaler khi có, RNG Python/NumPy/Torch/CUDA, epoch, batch cursor, lịch sử loss/validation và early stopping. SimpleDyG lưu trạng thái shuffle; baseline lưu RNG của negative sampler và tiến độ tìm dimension. Không phải tất cả model đều có scheduler hoặc dùng AMP.

Phiên sau tiếp tục ở batch đã được checkpoint. Nếu bị tắt đột ngột, đoạn train sau checkpoint hoàn chỉnh gần nhất được tính lại; nếu đang evaluation, phần evaluation đó được chạy lại. Khi run hoàn thành, `result.json` được commit rồi mới xóa checkpoint optimizer tạm, và giữ `best.pt` để đánh giá lại.

Checkpoint và ZIP ghi qua file tạm rồi thay thế file đích. Một ZIP đang ghi dở không thay thế snapshot hoàn chỉnh trước đó. **Cần giữ Output của Kaggle**: checkpoint trên đĩa không tự chuyển sang Input của phiên mới.

Mặc định `SNAPSHOT_INCLUDE_COMPLETED_BEST=True`: ZIP chứa checkpoint đầy đủ của các run còn dở, trọng số `best.pt` của các run hoàn thành, prediction và kết quả. ZIP kết quả vì vậy lớn hơn ZIP input 1,4 MB. Nếu đổi tùy chọn này thành False để giảm dung lượng, phải giữ riêng Output gốc có các trọng số đã hoàn thành.

## Phiên tiếp theo

1. Dùng notebook hai GPU hoặc bản `Continue_Fixed` trên, cùng ZIP source và image PyTorch/GPU type của phiên trước. Chọn **GPU T4 x2**.
2. Add Input **Output mới nhất** của notebook hoặc ZIP `resume_snapshot.zip` mới nhất. Bỏ snapshot cũ khỏi Input để tránh chọn nhầm.
3. Đổi `RESUME=True`. Giữ `RESUME_FROM=""` để tự tìm nếu chỉ có một bộ kết quả đúng; nếu có nhiều bộ, điền đường dẫn ZIP hoặc thư mục kết quả cần tiếp tục trong `/kaggle/input/...`.
4. Run All. Notebook khôi phục Input vào Working, tải cấu hình đã lưu, bỏ qua run hoàn thành và ưu tiên run có checkpoint. Log `RESUME ... epoch=... next_batch=... phase=...` cho biết đã khôi phục lượt đang dở.
5. Tiếp tục lưu Output và snapshot mới sau mỗi phiên cho tới khi manifest ghi đủ số run `complete` theo kế hoạch.

Khi `RESUME=True`, cấu hình khoa học lấy từ Output cũ. Có thể chỉnh `SESSION_HOURS`, `CHECKPOINT_SECONDS` và tùy chọn đóng gói; thay dataset/seed/epoch/protocol/dimension/history/micro-batch sẽ làm thành thí nghiệm khác. Runtime kiểm tra hash source/dữ liệu/config và môi trường trước khi tiếp tục. Nếu image hoặc package liên quan thay đổi, nó dừng với thông báo thay vì âm thầm dùng checkpoint trong môi trường khác.

Bản sửa 07/10 hỗ trợ trường hợp phiên trước dùng source bundle rút gọn, còn phiên sau clone đầy đủ cùng commit. Sau khi xác nhận checksum của từng file trong manifest cũ, notebook dựng thư mục source đúng inventory cũ và đối chiếu lại toàn bộ cấu hình. Runtime train giữ nguyên hash nên checkpoint và kết quả trước vẫn tương thích. Source hoặc dữ liệu thay đổi nội dung vẫn bị từ chối. Xem [chi tiết bản sửa](FIX_RESUME_SOURCE_INVENTORY.md).

Checkpoint của notebook một GPU cũ chỉ có trọng số và không tương thích định dạng mới. Không dùng ZIP kết quả cũ để đặt `RESUME=True` ở notebook hai GPU này.

## Đối chiếu kết quả và giới hạn kiểm chứng

SimpleDyG giữ kiến trúc, loss, logical batch, protocol `author_compat`, split thời gian và chọn model bằng validation của core trước. Baseline mới dùng gradient accumulation với micro-batch mặc định 8 event, giữ logical batch 128 và một optimizer step cho mỗi logical batch. Cách chia forward có thể đổi thứ tự dropout và sai số số thực so với baseline một forward lớn; cần ghi thiết lập này trong báo cáo. Các baseline vẫn là backbone DyGLib có BPR adapter và zero features, chưa phải bản tái lập chính xác toàn bộ baseline trong bảng paper.

Đọc `results_per_seed.csv`, `summary_mean_std.csv`, `paper_comparison.csv`, `ablation_paired_deltas.csv` và `run_manifest.json` trong Output. Báo cáo số seed thực tế hoàn thành; các run `paused/pending` còn thiếu, các run `failed` có log và `runs/.../error.txt`. Không kết luận đã tái lập thành công chỉ vì code chạy hết: còn phải đối chiếu metric, protocol và sai lệch với paper.

Đã kiểm chứng cục bộ trên Python 3.13/PyTorch CPU:

- 9 kiểm thử đều qua: resume giữa epoch và khi validation, khôi phục optimizer/scheduler/RNG, TGAT/GraphMixer, legacy AdamW, hàng đợi không nhận trùng run, ZIP ghi dở, hai tiến trình cùng tạm dừng rồi chạy tiếp.
- Trọng số và các trạng thái train trong các kiểm thử resume khớp chính xác với chạy liên tục trên CPU. Một kiểm thử `author_compat` thay phần validation bằng score cố định để giới hạn thời gian, còn model và optimizer thực sự được train.
- Toàn bộ 7 code cell của notebook bàn giao đã chạy qua trên CPU với hai worker và 5 run smoke; khôi phục ZIP trong thư mục mới tái sử dụng đủ 5 kết quả và trọng số, không train lại.

Chưa có phép chạy xác nhận notebook mới trên **hai T4 thật**. Các kiểm thử này xác nhận logic thực thi/checkpoint, không chứng minh tốc độ, VRAM của cấu hình paper trên Kaggle hay metric reproduce. Thời gian từng run có checkpoint I/O; phần công việc bị mất sau checkpoint chưa chắc được cộng đầy đủ. Dùng `session_history_*.json` để báo wall time toàn chiến dịch và ghi rõ chạy song song khi so sánh thời gian.

Evidence: [kiểm thử runtime và artifact](dual_gpu_resume_validation.json), [thực thi notebook từ đầu và khôi phục](dual_notebook_execution_validation.json).

## Bảo trì

Sửa core chung `tools/kaggle_pipeline.py` hoặc phần mở rộng `tools/dual_runtime_extensions.py`, sau đó chạy lần lượt `tools/build_dual_runtime.py` và `tools/build_dual_notebook.py`. Notebook và manifest ZIP phải cùng runtime hash. Kiểm thử chuyên biệt ở `tests/test_dual_resume.py`; script thực thi notebook ở `tools/validate_dual_notebook.py`. Không upload môi trường `.venv-validation313` lên Kaggle.

Tham khảo: [PyTorch về checkpoint để tiếp tục training](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training), [PyTorch về reproducibility](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md).
