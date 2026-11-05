# SimpleDyG – tái lập và đánh giá trên Kaggle 2×T4

Đồ án tái lập paper **On the Feasibility of Simple Transformer for Dynamic Graph Modeling** (Wu, Fang, Liao – WWW '24, [arXiv:2401.14009](https://arxiv.org/abs/2401.14009v2)).
Nhóm 6: Vũ Duy Thụ & Đỗ Quốc Thịnh.

- **Báo cáo:** [`reports/BAO_CAO_KET_QUA_THUC_NGHIEM.md`](reports/BAO_CAO_KET_QUA_THUC_NGHIEM.md)
- **Slide:** [`reports/SimpleDyG_Slide_KetQua.pdf`](reports/SimpleDyG_Slide_KetQua.pdf) (29 trang)

## Kết quả chính

| Hạng mục | Kết quả |
|---|---|
| Thực nghiệm | 125/125 lượt (3 đợt), UCI và ML-10M, ≈ 689 GPU-giờ trên 2×T4 |
| Link prediction · UCI | SimpleDyG NDCG@5 **.104 ± .017**, bằng paper (.104); tốt hơn TGAT (.049) và GraphMixer (.070) |
| Link prediction · ML-10M | .099 ± .061 so với paper .138; mô hình suy biến về phim phổ biến nhất (trần .148). Mask padding giúp ổn định hơn nhưng không khắc phục được |
| Baseline đơn giản | "Lặp lại slot gần nhất" đạt .293 trên UCI, khoảng 2,8× mọi mô hình |
| Temporal alignment | Trên UCI, same time / no time / per-ego permute so với full: mọi CI 95% đều chứa 0 |

## Cấu trúc

```
notebooks/
  SimpleDyG_Kaggle_2GPU_Resume.ipynb   đợt 1: 100 lượt (RESUME=True để chạy tiếp phiên trước)
  SimpleDyG_Completion_2GPU.ipynb      đợt 2 (EXPERIMENT="uci_permute") và đợt 3 ("ml_corrected")
  SimpleDyG_Paper_Figures.ipynb        chỉ CPU: vẽ bảng/hình kiểu paper từ Output các đợt
  SimpleDyG_Kaggle_2GPU_Inputs.zip     input Kaggle: mã tác giả + DyGLib (commit đã pin) + dữ liệu đã preprocess
  SimpleDyG_Previous_Results.zip       kết quả rút gọn của đợt 1 (bảng, prediction, đường học)
results/
  round1_100runs/                      bảng + hình của đợt 1
  round2_uci_permute/                  phân tích + hình của đợt 2
  round3_ml_corrected/                 phân tích + hình + đường học (ml_histories.tgz) của đợt 3
reports/
  BAO_CAO_KET_QUA_THUC_NGHIEM.md       báo cáo cuối
  SimpleDyG_Slide_KetQua.pdf           slide cuối
  2GPU_VÀ_CHECKPOINT.md                hướng dẫn chạy 2 GPU, checkpoint và chạy tiếp
  figures/                             hình dùng trong báo cáo và slide
  slides_src/                          mã dựng slide kết quả (charts, HTML, ghép PDF)
tools/
  kaggle_pipeline_dual.py              runtime huấn luyện/đánh giá dùng cho cả 3 đợt (SHA e6e61428…)
  kaggle_pipeline.py, dual_runtime_extensions.py, build_dual_runtime.py   nguồn sinh runtime trên
  build_*_notebook.py                  sinh các notebook trong notebooks/
  completion_analysis.py, paper_figures.py   phân tích và vẽ hình (CPU)
  resume_source_alignment.py           kiểm tra source khi chạy tiếp
```

## Chạy lại thực nghiệm (Kaggle)

1. Import notebook, Add Input `SimpleDyG_Kaggle_2GPU_Inputs.zip` (và `SimpleDyG_Previous_Results.zip` cho notebook hoàn thiện).
2. Settings: **GPU T4 x2**, Internet On, *Pin to original environment*.
3. Đặt cấu hình ở cell đầu (`EXPERIMENT`, `RESUME`), rồi **Save & Run All (Commit)**. Hết phiên (≈ 11 giờ): Add Input Output mới nhất, đặt `RESUME=True` và chạy lại.

Chi tiết checkpoint và chạy tiếp: [`reports/2GPU_VÀ_CHECKPOINT.md`](reports/2GPU_VÀ_CHECKPOINT.md).

## Phân tích, hình và slide (máy cá nhân, không cần GPU)

```bash
pip install pandas numpy matplotlib pypdf playwright
# Bảng/hình kiểu paper: chạy notebooks/SimpleDyG_Paper_Figures.ipynb với INPUT_DIR trỏ tới thư mục chứa các ZIP

cd reports/slides_src
python prepare_data.py            # giải nén dữ liệu cần cho biểu đồ vào data/
python charts.py                  # build/*.svg và ../figures/*.png (cần font Inter)
python build_slides.py 13 29
python render.py $PWD/build/results_slides.html $PWD/build/results_slides.pdf
python merge.py MidTermSlideSimpleDyG.pdf build/results_slides.pdf ../SimpleDyG_Slide_KetQua.pdf
```

## Ghi chú

- Mã tác giả: [YuxiaWu/SimpleDyG](https://github.com/YuxiaWu/SimpleDyG) @ `fa0dc82`; DyGLib: [yule-BUAA/DyGLib](https://github.com/yule-BUAA/DyGLib) @ `3aacc36`. Cả hai được đóng gói sẵn trong `SimpleDyG_Kaggle_2GPU_Inputs.zip`.
- Gói kết quả đầy đủ của các đợt (checkpoint, prediction từng lượt, khoảng 8 GB) không nằm trong repo; chúng ở Output của các notebook trên Kaggle.
- `tools/build_kaggle_notebook.py` chỉ dựng lại được ZIP input khi có bản clone repo tác giả ở `repo_clone/` và DyGLib ở `tmp/inspection/DyGLib`.
