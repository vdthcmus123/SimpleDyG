# Rà soát SimpleDyG và pipeline Kaggle

Ngày rà soát: 05/10/2026. Nguồn: paper `2401.14009v2.pdf` trong thư mục, code tác giả commit `fa0dc82587d8ce787345a0379f65a0dcf5d9d9ff`, và DyGLib commit `3aacc36b94b8d2d8293d70a74fdf6d39089b4163`.

**Kết luận: thư mục có đúng paper và đủ dữ liệu cho phần đồ án UCI/ML-10M, nhưng các script/notebook cũ chưa đủ để chạy benchmark và rút kết luận khoa học.** Notebook mới là đường chạy được khuyến nghị. Repo và các notebook cũ được giữ để đối chiếu nguồn.

**Bản sửa sau lỗi Kaggle Python 3.13:** setup đã bỏ các pin `transformers==4.44.2`/`tokenizers==0.19.1`. Tokenizer node đã nhúng; hai activation import của model tác giả được chuyển sang phép tính tương đương và optimizer/scheduler legacy được nhúng. Các kiểm tra so với thư viện cũ xác nhận forward, gradient, token ID, AdamW và learning-rate schedule giữ nguyên. Đọc `reports/FIX_PYTHON_313.md` và evidence mới để phân biệt bản sửa với kiểm thử Python 3.12 ban đầu.

## 1. Đối chiếu yêu cầu đồ án

| Yêu cầu | Hiện trạng trước rà soát | Đã đưa vào notebook mới |
|---|---|---|
| Đọc và tái lập paper | Đúng paper WWW 2024, đúng repo; wrapper mới chưa kiểm tra được chạy hết | Dùng chính implementation GPT-2 trong `models/` của tác giả; có chế độ tương thích release |
| 1–2 dataset preprocess | Có đủ sequence, vocabulary và event CSV UCI/ML-10M | Chọn UCI và ML-10M; validate tất cả split/target trước train |
| Link prediction/dynamic prediction | Release là dự đoán tập node ở bin tương lai, không phải binary link classification AP/AUC | Tái lập dynamic link prediction một bước; NDCG@5, Jaccard và Jaccard@5 |
| Benchmark model mã nguồn mở | Chỉ có SimpleDyG và các cấu hình architecture | Thêm backbone TGAT, GraphMixer từ DyGLib; DyGFormer tùy chọn; BPR training và full-candidate ranking |
| Vai trò temporal alignment | Có bỏ/shuffle/reverse nhãn time, thiếu `same_time` như Table 4 | Full, same_time, no_time, paired delta cùng seed; optional per-ego permutation |
| Một notebook Kaggle | Notebook gộp cũ có 49 cell, vẫn gọi đường chạy bị lỗi | Một `.ipynb` mới chứa toàn bộ mã điều phối, setup, dữ liệu, train, evaluate, export |
| Hướng nghiên cứu mới | Architecture/token ablation chưa đủ để tuyên bố novelty | Hoãn nghiên cứu mới đến sau kết quả full dataset; không cài sẵn kết luận |

## 2. Paper thực sự yêu cầu gì?

Paper chuyển temporal ego-graph thành sequence node, chèn token theo bin thời gian để căn các ego-sequence trên cùng timeline. Loss là causal next-token trên cả history và prediction, **không phải** fine-tune pretrained GPT-2 cho ngôn ngữ. Trong `main.py`, tác giả tạo `model_class(config=config)` từ đầu; `gpt2` chỉ cung cấp config.

Section 5.1: dataset được chia theo bin thời gian, bin cuối là test, bin liền trước là validation. Paper lặp **10 lần**, báo mean và standard deviation. Baseline được chuyển sang loss **BPR** ở Appendix B. Không được đưa AP/AUC của binary event classification cạnh NDCG/Jaccard của dự đoán tập node rồi gọi là cùng benchmark.

Paper Table 2 và Table 4 (đã đối chiếu hình trang 7):

| Dataset | Temporal variant | NDCG@5 | Jaccard |
|---|---|---:|---:|
| UCI | full | 0.104 ± 0.010 | 0.092 ± 0.014 |
| UCI | same_time | 0.090 ± 0.013 | 0.083 ± 0.012 |
| UCI | no_time | 0.111 ± 0.015 | 0.091 ± 0.014 |
| ML-10M | full | 0.138 ± 0.009 | 0.131 ± 0.008 |
| ML-10M | same_time | 0.147 ± 0.001 | 0.139 ± 0.001 |
| ML-10M | no_time | 0.117 ± 0.062 | 0.111 ± 0.059 |

Đây là số paper công bố, **chưa phải số reproduce của nhóm**. Temporal alignment không luôn tốt hơn; `same_time` và `no_time` đã được paper nghiên cứu. Không có cơ sở để áp tiêu chí tùy ý “±2% là reproduce thành công”.

## 3. Dữ liệu thực có trong clone

| Dataset | Node | Event CSV | Train sequence | Val query | Test query | Quyết định |
|---|---:|---:|---:|---:|---:|---|
| UCI_13/12 | 1,781 | 16,743 | 1,708 | 146 | 110 | Chạy mặc định |
| ML_10M_13/12 | 15,841 | 48,561 | 13,701 | 738 | 396 | Chạy mặc định |
| hepth/11 | Theo paper 4,737 | Có CSV | 3,965 | 240 | 305 | Thiếu `resources/hepth/node_features.npy` |
| dialog/15 | Theo paper 7,415 | Có CSV và sequence | Có | Có | Có | Không chọn trong phạm vi hai dataset đầu |

Train của UCI/ML-10M chỉ dùng bin ≤10; val target bin 11; test target bin 12. Test input có thể chứa history bin 11 đã quan sát; điều này hợp lệ theo giao thức dự đoán bin 12 và không phải train lại trên test.

Release xây vocabulary từ tất cả node trên timeline, gồm node identity về sau. Đây là giả định closed world; notebook ghi rõ và không dùng cạnh test làm history. Release chỉ tạo val/test query theo cột `u`; reverse target không tự được thêm. Baseline lấy history undirected theo paper, trong khi sequence release có lịch sử reverse của train và cột `u` cho bin validation. Đây là khác biệt đầu vào cần ghi khi viết phương pháp.

ML-10M có sequence dài hơn context: train tối đa 2,980 token, val 2,605, test 2,985. Notebook thống kê truncation; giới hạn history là một yếu tố thực nghiệm, không nên gọi mọi phiên bản là tái lập chính xác. UCI node/edge feature `.npy` là toàn zero, dimension 172. Các feature `.npz` của ML-10M chưa được căn thành một array theo vocabulary của release; baseline dùng zero feature được khai báo.

## 4. Các lỗi và hạn chế tìm thấy

| File/nhóm file | Phát hiện | Ảnh hưởng |
|---|---|---|
| `repo_clone/utils/Evaluation.py:106` | Tab và space không nhất quán; `ast.parse` báo `TabError` | `main.py` không khởi chạy được nguyên trạng |
| `repo_clone/requirements.txt` | Hai pin tensorboardX mâu thuẫn; torch 1.9.1/numpy 1.21.2 cũ; apex và en_core_web_sm không cần cho đường chạy này | Cài nguyên requirements dễ thất bại trên Python/Kaggle mới |
| `notebooks/01_reproduce.py` | Gọi Hepth với feature file không có; cho `transformers>=4.24.0` không upper bound | Không đảm bảo tương thích legacy import `AdamW`; Hepth fail |
| `SimpleDyG_Ablation_Study.ipynb`, cell 48 | In sẵn “Pipeline verified”, “within +/-2%”, “Full Model >> No Temporal” và kết luận architecture | Kết luận không phụ thuộc kết quả, mâu thuẫn xu hướng Table 4 |
| Temporal wrappers | Shuffle/reverse **nhãn** thời gian, không đảo thứ tự node event; placeholder còn chứa token gốc và có thể bị thay tiếp | Không thể diễn giải thành kiểm chứng thứ tự/direction của event |
| Temporal wrappers | Thiếu `same_time` | Chưa tái lập đúng các variant Table 4 |
| Architecture wrappers | Width sweep đồng thời đổi số head; count parameter dùng vocab 5,000 giả định và đếm thêm LM head tied | So sánh không cô lập width; số parameter không đáng tin |
| Notebook gộp cũ | Đo `torch.cuda.max_memory_allocated()` ở notebook cha nhưng train trong subprocess | Không đo đúng peak VRAM của process train |
| Các collector | Append nhiều lần vào CSV chung, fallback concat mọi checkpoint/run | Có thể trộn epoch/cấu hình hoặc đếm lặp khi báo mean/std |
| Eval gốc | NDCG tính duplicate prediction như nhiều hit; padding được đưa vào loss; val cap 11 token khác test cap context | Điểm có thể vượt 1 với duplicate và metric có độ nhạy protocol |
| Repo tác giả | Không có code baseline đã chỉnh BPR và toàn bộ hyperparameter/negative policy | Không thể tuyên bố tái lập hoàn toàn Table 2 baseline chỉ bằng clone này |
| Preprocess optional | `csv2resources.py` dùng `../all_data/` trong khi folder hiện nằm trong repo; UCI `graphs.npz` chỉ 22 byte, không có array | Chưa thể khẳng định raw-to-sequence rebuild chạy nguyên trạng; pipeline dùng dữ liệu preprocess sẵn |

## 5. Notebook mới vận hành như thế nào?

File chính: `notebooks/SimpleDyG_Kaggle_Full_Pipeline.ipynb`. Gói tùy chọn: `notebooks/SimpleDyG_Kaggle_Inputs.zip` (~1.4 MB). Bundle có minimal source của tác giả, source các model DyGLib, license DyGLib, hai dataset preprocess và SHA-256 manifest; không phải nguyên repo Git.

Notebook chứa toàn bộ code điều phối. Có thể clone commit được pin khi bật Internet, hoặc Add Input bundle khi offline. Không cần upload các file Python điều phối riêng. Generator và core source ở `tools/` dùng để bảo trì; notebook bàn giao đã nhúng source.

### Profile và protocol

| Profile | Dữ liệu/seed | Model/budget | Cách dùng kết quả |
|---|---|---|---|
| smoke | 32 train sequence, 4 val/test query, 64 baseline train event; 1 seed | Model nhỏ, 1 epoch | Kiểm tra code; tuyệt đối không dùng để viết kết luận chất lượng |
| pilot (mặc định) | Full UCI + ML-10M; 3 seed | SimpleDyG theo script tác giả; baseline tune dim 64/128; history cap 128, Mixer 64; LM ≤100 epoch, baseline ≤50 | Kết quả sơ bộ; ghi rõ giới hạn history/protocol |
| paper | Full UCI + ML-10M; 10 seed | SimpleDyG author_compat; baseline dims 16–512; history/Mixer 1024; ≤100 epoch | Thử tái lập nghiêm túc; vẫn cần báo các khác biệt adapter baseline |

Pilot mặc định có **30 run được báo** và 2 candidate dimension cho mỗi baseline run. Early stopping giảm thời gian nhưng không có cam kết một phiên Kaggle đủ cho toàn bộ lịch. Chọn một dataset hoặc một nhóm seed mỗi phiên được hỗ trợ qua override. Với `paper`, tìm dimension baseline và Mixer 1024 sẽ tốn đáng kể thời gian/VRAM; nên chạy theo từng nhóm và giữ Output phiên trước.

`author_compat` giữ loss padding, left-truncation, greedy/EOS và metric duplicate của release, đồng thời dùng trực tiếp class model của tác giả. Trainer đã thay để chạy trên PyTorch hiện đại; micro-batch, RNG/CPU/GPU có thể khác, nên không gọi bit-for-bit. `corrected` mask padding, dùng cùng cap generation val/test và stop ở end-of-prediction/EOS. Cột `ndcg5` luôn là NDCG chuẩn; `ndcg5_author` là metric tương thích release.

Baseline là adaptation từ backbone official TGAT/GraphMixer/DyGFormer của DyGLib. BPR học event train theo thời gian, cả hai hướng cạnh, negative loại self và các positive cùng train bin. Không dùng val/test label để lọc negative train. Graph store không chứa event test; sampler chỉ lấy cạnh có thời gian **nhỏ hơn** query cutoff. Val/test dự đoán tại đầu bin, không cập nhật bằng cạnh trong bin đang dự đoán.

Mọi baseline chấm **toàn bộ vocabulary trừ self**. TGAT/GraphMixer cache node embedding ở cùng cutoff để tiết kiệm; DyGFormer phụ thuộc cặp node nên phải chấm tất cả cặp và chậm hơn, để tùy chọn. Khi tie, thứ tự vocabulary quyết định như nhau cho các baseline.

Hyperparameter dimension, checkpoint và k của baseline đều chọn bằng validation. Test chỉ chấm model được chọn. Primary comparison dùng **NDCG@5 và Jaccard@5**. `jaccard` của SimpleDyG dùng toàn bộ tập sinh; baseline dùng top-k đã chọn trên validation. `jaccard_oracle_max_k` là max macro trên k={1,5,10,20} ở test, dùng tham chiếu mô tả paper; không được coi là điểm không dùng test để chọn k.

Full/same_time/no_time dùng cùng vocabulary/model initialization, split, seed và architecture. `per_ego_permute` tùy chọn dùng mapping cố định theo ego+seed xuyên split, giữ nguyên thứ tự node; mục đích kiểm tra mất global alignment. Nó không đại diện cho đảo thứ tự sự kiện và chưa được coi là đóng góp mới.

### Output

- `data_audit.json`: count, max length, truncation, SHA-256 dataset/vocab/event CSV.
- `environment.json`, `pip_freeze.txt`, `config.json`, `source_hashes.json`: environment và provenance.
- `results_per_seed.csv`, `summary_mean_std.csv`: kết quả thực đo, mean và sample std. Một seed có std thiếu, không thay bằng 0.
- `paper_reference.csv`: bảng số công bố, có nhãn nguồn riêng.
- `paper_comparison.csv`: chênh lệch kết quả đo với Table 2/4, có nhãn profile/protocol; không tự kết luận tái lập thành công.
- `ablation_paired_deltas.csv`: chênh lệch cùng seed so với full; chưa tự suy luận significance khi ít seed.
- Mỗi run: `best.pt`, history JSON, prediction JSONL val/test, result JSON; lỗi có `error.txt`.
- Chart PNG và ZIP kết quả. Checkpoint vẫn ở Output directory, mặc định không đóng vào ZIP để giảm dung lượng.

Fingerprint bao gồm config, dữ liệu và source. Chạy lại cùng fingerprint sẽ tái sử dụng run hoàn thành. Run dở train lại từ đầu; **chưa resume optimizer giữa epoch**. Muốn dùng Output từ phiên Kaggle trước phải Add Input/copy đúng thư mục output vào thư mục working tương ứng, hoặc thay `OUTPUT`; notebook không tự tải Output phiên khác.

## 6. Kiểm thử và giới hạn xác minh

Đã kiểm tra bằng môi trường Python 3.12, PyTorch 2.5.1 CPU, Transformers 4.44.2:

- Hai dataset qua kiểm tra vocabulary, timestamp train/val/test, và đối chiếu target theo event CSV.
- Token ID của tokenizer nhúng khớp tokenizer release ở UCI và ML-10M.
- Smoke corrected: 12 run thành công, gồm ba biến thể SimpleDyG và ba baseline trên cả hai dataset.
- Smoke author_compat: full/no_time trên hai dataset thành công.
- Mã source DyGLib trong bundle khớp SHA của các file tại commit được pin.

Các phép kiểm tra notebook, metric, leakage và bundle được đặt ở `tests/test_kaggle_pipeline.py`. Kết quả QA cuối cùng được lưu trong `reports/validation_summary.json` sau khi chạy.

**Chưa chạy full training trên Kaggle GPU và chưa xác nhận reproduce được số Table 2/4.** Smoke một epoch cho điểm thấp/zero là bình thường và không thể dùng so sánh model. GPU AMP, thời gian chạy full history và mức VRAM thực cần xác nhận bằng run Kaggle. Notebook sẽ dừng với báo lỗi rõ nếu run thất bại, không in “reproduce thành công” dựa trên việc chạy hết code.

## 7. Phần nghiên cứu sau

Hoàn thành các run full dataset và kiểm tra prediction trước. Sau đó có thể khảo sát thiết kế alignment thích nghi theo độ bursty của timeline hoặc multi-resolution time token, với control global alignment/relative order và giới hạn history được giữ thống nhất. Cần rà soát literature, so sánh với Table 4 và kiểm tra nhiều seed trước khi khẳng định novelty; phần này chưa được triển khai trong lượt hiện tại.

Nguồn công khai: [SimpleDyG](https://github.com/YuxiaWu/SimpleDyG), [DyGLib](https://github.com/yule-BUAA/DyGLib), [paper](https://arxiv.org/abs/2401.14009).
