"""Build SimpleDyG_Completion_2GPU.ipynb from the 2-GPU notebook used for the 100 paper runs.

The training runtime cell (kaggle_pipeline_dual.py, SHA e6e6142856...) is copied unchanged,
so new runs use exactly the code that produced simple_dyg_result.zip. Only the settings,
output folder and an extra CPU analysis cell differ.

Usage: python build_completion_notebook.py <base_notebook> <completion_analysis.py> <paper_figures.py> <output.ipynb>
"""
import ast
import json
import sys
import uuid
from pathlib import Path

RUNTIME_SHA = 'e6e6142856ea68e46feba431bdfbf9748380818caffaa37c604745af7d972ab8'


def cell(kind, source):
    c = dict(cell_type=kind, id=uuid.uuid4().hex[:8], metadata={}, source=source)
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


INTRO = '''# SimpleDyG – thí nghiệm bổ sung để hoàn thiện đồ án (2×T4, có checkpoint)

Notebook này bổ sung cho 100 lượt chạy `paper/author_compat` đã có trong `simple_dyg_result.zip`.
Runtime huấn luyện được **giữ nguyên byte-for-byte** (SHA-256 `e6e61428…`); chỉ cấu hình thí nghiệm khác.

| `EXPERIMENT` | Mục tiêu | Số run | Ước tính trên 2×T4 |
|---|---|---|---|
| `"uci_permute"` | Kiểm tra trực tiếp **temporal alignment giữa các chuỗi**: giữ phân đoạn time-slot của từng ego nhưng hoán vị nhãn thời gian riêng cho từng ego (`per_ego_permute`). So cặp theo seed với `full` cũ. | 10 (UCI, seed 0–9) | ~2,5–3 giờ |
| `"ml_corrected"` | Chẩn đoán/khắc phục **suy biến trên ML-10M** (mô hình chỉ đoán phim `1419`): protocol `corrected` (mask padding trong loss, attention mask, dừng ở `<|endofpre|>`), `patience=20`. | 15 (ML-10M, 3 biến thể × seed 0–4) | ~8–12 giờ, có thể cần 1 phiên resume |
| `"smoke"` | Kiểm tra nhanh đường chạy (CPU hoặc GPU), không dùng làm kết quả. | 2 | vài phút |

**Input cần Add trên Kaggle**
1. `SimpleDyG_Kaggle_2GPU_Inputs.zip` – source + dataset đã pin (giống lần chạy 100 run).
2. `SimpleDyG_Previous_Results.zip` – kết quả 100 run cũ (bảng + prediction của SimpleDyG) để ghép cặp và phân tích. Không có file này thì phần phân tích chỉ dùng kết quả mới.

**Không** Add Output của thí nghiệm 100 run cũ vào cùng phiên (runtime resume sẽ thấy nhiều Output).

Chọn **GPU T4 x2**, Internet bật (hoặc OFFLINE với wheel), đặt `EXPERIMENT`, rồi Run All.
Cuối notebook có ZIP kết quả `simpledyg_<EXPERIMENT>.zip`, `resume_snapshot.zip`, `simpledyg_completion_analysis.zip` và `simpledyg_paper_figures.zip` (bảng/biểu đồ kiểu paper).
'''

SETTINGS = '''# ====== CHỌN THÍ NGHIỆM ======
EXPERIMENT = "uci_permute"          # "uci_permute" | "ml_corrected" | "smoke"

EXPERIMENTS = {
    # Temporal alignment giữa các ego: hoán vị nhãn time riêng cho từng ego, giữ protocol cũ.
    "uci_permute": dict(PROFILE="paper", PROTOCOL_OVERRIDE=None, DATASETS_OVERRIDE=["UCI_13"],
                        SEEDS_OVERRIDE=list(range(10)), BASELINE_MODELS=[],
                        VARIANTS=["per_ego_permute"], EXTRA_CONFIG={}),
    # ML-10M: mask padding (protocol corrected) + patience dài hơn để tránh dừng sớm ở NDCG=0.
    "ml_corrected": dict(PROFILE="paper", PROTOCOL_OVERRIDE="corrected", DATASETS_OVERRIDE=["ML_10M_13"],
                         SEEDS_OVERRIDE=list(range(5)), BASELINE_MODELS=[],
                         VARIANTS=["full", "same_time", "no_time"], EXTRA_CONFIG={"patience": 20}),
    # Kiểm tra thực thi: 1 epoch, dữ liệu cắt nhỏ; không dùng để kết luận.
    "smoke": dict(PROFILE="smoke", PROTOCOL_OVERRIDE=None, DATASETS_OVERRIDE=["UCI_13"],
                  SEEDS_OVERRIDE=[0], BASELINE_MODELS=[], VARIANTS=["full", "per_ego_permute"],
                  EXTRA_CONFIG={}),
}
if EXPERIMENT not in EXPERIMENTS:
    raise ValueError("EXPERIMENT phải là một trong " + str(list(EXPERIMENTS)))
globals().update(EXPERIMENTS[EXPERIMENT])
print("EXPERIMENT:", EXPERIMENT, EXPERIMENTS[EXPERIMENT])

# ====== Lịch chạy / checkpoint (giữ như notebook 100 run) ======
GPU_IDS = [0, 1]
SESSION_HOURS = 10.5                 # tạm dừng trước giới hạn 12h của Kaggle
CHECKPOINT_SECONDS = 300
WORKER_CPU_THREADS = 2
SNAPSHOT_INCLUDE_COMPLETED_BEST = True

RESUME = False                      # phiên đầu False; phiên tiếp theo (cùng EXPERIMENT) True
RESUME_FROM = ""                    # ZIP/thư mục Output mới nhất; trống = tự tìm trong Input
OFFLINE = False
INSTALL_DEPENDENCIES = True
AUTHOR_ROOT_OVERRIDE = ""
DYGLIB_ROOT_OVERRIDE = ""
INPUT_BUNDLE_OVERRIDE = ""           # đường dẫn SimpleDyG_Kaggle_2GPU_Inputs.zip nếu không tự tìm được
PREVIOUS_RESULTS_OVERRIDE = ""       # đường dẫn SimpleDyG_Previous_Results.zip nếu không tự tìm được
'''

CONFIG_CHECK = '''
# --- kiểm tra Output resume đúng thí nghiệm ---
_expected = EXPERIMENTS[EXPERIMENT]
if RESUME:
    if CFG["datasets"] != _expected["DATASETS_OVERRIDE"] or CFG["variants"] != _expected["VARIANTS"]:
        raise RuntimeError("Output resume thuộc thí nghiệm khác: " + str(CFG["datasets"]) + str(CFG["variants"]))
'''

ANALYSIS_TEMPLATE = '''# Phân tích hoàn thiện (CPU): baseline tầm thường, kiểm tra suy biến, so cặp temporal alignment.
import types, zipfile, shutil
ANALYSIS_SOURCE = __ANALYSIS_SOURCE__
ca = types.ModuleType("completion_analysis")
exec(compile(ANALYSIS_SOURCE, "completion_analysis.py", "exec"), ca.__dict__)

ANALYSIS = WORK / "simpledyg_completion_analysis"
if ANALYSIS.exists():
    shutil.rmtree(ANALYSIS)
ANALYSIS.mkdir(parents=True)
tables, runs_roots = [], []
previous_zip = [Path(PREVIOUS_RESULTS_OVERRIDE)] if PREVIOUS_RESULTS_OVERRIDE else (
    list(INPUT.rglob("SimpleDyG_Previous_Results.zip")) if INPUT.exists() else [])
previous_dirs = [p.parent for p in INPUT.rglob("previous_results/results_per_seed.csv")] if INPUT.exists() else []
if previous_zip:
    with zipfile.ZipFile(previous_zip[0]) as archive:
        archive.extractall(WORK / "previous_extract")
    previous_dirs = [WORK / "previous_extract" / "previous_results"]
if previous_dirs:
    tables.append(previous_dirs[0] / "results_per_seed.csv")
    runs_roots.append(previous_dirs[0] / "runs")
    print("Kết quả cũ:", previous_dirs[0])
else:
    print("Không tìm thấy SimpleDyG_Previous_Results.zip: chỉ phân tích kết quả mới.")
if (OUTPUT / "results_per_seed.csv").exists() and (OUTPUT / "results_per_seed.csv").stat().st_size > 1:
    tables.append(OUTPUT / "results_per_seed.csv")
    runs_roots.append(OUTPUT / "runs")
if not tables:
    raise RuntimeError("Chưa có bảng kết quả nào để phân tích.")
report = ca.run_all(AUTHOR_ROOT, tables, runs_roots, ANALYSIS)
pd.set_option("display.width", 220)
print("== Baseline tầm thường (không huấn luyện)")
display(report["baselines"].round(4))
print("== Tóm tắt biến thể SimpleDyG")
display(report["summary"].round(4))
print("== So cặp theo seed: biến thể − full (CI 95% theo t, sign test)")
display(report["paired"].round(4))
print("== Số run suy biến (một tập dự đoán phủ >=70% truy vấn test)")
display(report["counts"])

try:
    import matplotlib.pyplot as plt
    summary, base = report["summary"], report["baselines"]
    datasets = list(summary.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 4), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = summary[summary.dataset == ds]
        labels = sub.variant + " / " + sub.protocol
        ax.bar(range(len(sub)), sub.ndcg5_mean, yerr=sub.ndcg5_std.fillna(0), capsize=4, color="#4C78A8")
        ax.set_xticks(range(len(sub)), labels, rotation=30, ha="right")
        for method, style in [("Repeat last active slot", "--"), ("Popular top-1 (train)", ":")]:
            value = base[(base.dataset == ds) & (base.split == "test") & (base.method == method)].ndcg5
            if len(value):
                ax.axhline(float(value.iloc[0]), linestyle=style, color="#E45756", label=method)
        ax.set_title(ds + " – NDCG@5 (test, mean ± std)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(ANALYSIS / "ndcg5_vs_trivial.png", dpi=160)
    plt.show()
except ImportError:
    print("Không có matplotlib: bỏ qua biểu đồ.")

analysis_zip = shutil.make_archive(str(WORK / "simpledyg_completion_analysis"), "zip", ANALYSIS)
from IPython.display import FileLink
display(FileLink(Path(analysis_zip).relative_to(WORK) if str(WORK) == "/kaggle/working" else analysis_zip))
'''

FIGURES_TEMPLATE = '''# Bảng & biểu đồ theo kiểu paper (Table 1/2/4/5, Figure A.1, Fig. 4c) – gộp kết quả cũ + mới.
import types, shutil
PAPER_FIGURES_SOURCE = __FIGURES_SOURCE__
pfig = types.ModuleType("paper_figures")
exec(compile(PAPER_FIGURES_SOURCE, "paper_figures.py", "exec"), pfig.__dict__)
NDCG_METRIC = "ndcg5"               # "ndcg5" (chuẩn) hoặc "ndcg5_author" (giống cách tính của paper)
FIGURE_ROOTS = [p for p in (previous_dirs[:1] + [OUTPUT]) if (p / "results_per_seed.csv").exists()]
FIGURES = WORK / "simpledyg_paper_figures"
if FIGURES.exists():
    shutil.rmtree(FIGURES)
figures = pfig.render_all(FIGURE_ROOTS, AUTHOR_ROOT, FIGURES, ndcg_col=NDCG_METRIC)
pfig.show(figures)
figures_zip = shutil.make_archive(str(WORK / "simpledyg_paper_figures"), "zip", FIGURES)
print("Bảng/biểu đồ:", FIGURES, "| ZIP:", figures_zip, "| HTML tổng hợp: paper_tables.html")
'''

OUTRO = '''## Đọc kết quả và phiên tiếp theo

**uci_permute.** Xem dòng `per_ego_permute - full` (UCI, author_compat) trong bảng so cặp.
- Nếu CI 95% nằm hẳn dưới 0: căn chỉnh thời gian *giữa các ego* có đóng góp.
- Nếu CI chứa 0 (giống same_time/no_time): mô hình chủ yếu dùng thứ tự cục bộ của từng ego; alignment toàn cục không có tác dụng đo được.

**ml_corrected.** Xem `collapse_counts` và `top_share` trong `collapse_census.csv`.
- Nếu số run `personalised` tăng và NDCG vượt baseline phổ biến (.1475): mask padding là nguyên nhân chính; khi đó mới so sánh full/same_time/no_time trên ML-10M.
- Nếu vẫn suy biến: báo cáo ML-10M là dataset bị chi phối bởi popularity dưới cấu hình paper (lr 1e-5, 2 lớp, dim 128); kết luận ablation chỉ dựa trên UCI.

**Hết phiên Kaggle khi đang chạy.** Lưu Version/Output, rồi mở phiên mới với cùng `EXPERIMENT`:
Add Input Output mới nhất (hoặc `resume_snapshot.zip`), giữ hai ZIP input, đặt `RESUME=True`, Run All.
Không đổi seed/epoch/protocol giữa các phiên; runtime kiểm tra config/source/data/environment trước khi tiếp tục.
'''


def main(base_path, analysis_path, figures_path, out_path):
    base = json.loads(Path(base_path).read_text(encoding='utf-8'))
    cells = base['cells']
    src = lambda i: ''.join(cells[i]['source'])
    runtime = src(5)
    assert runtime.count(RUNTIME_SHA) == 2, 'Base notebook does not embed the 100-run runtime'
    configure = src(9)
    old_output = 'OUTPUT = WORK / "simpledyg_2gpu_results"'
    assert configure.count(old_output) == 1
    configure = configure.replace(old_output, 'OUTPUT = WORK / f"simpledyg_{EXPERIMENT}"') + CONFIG_CHECK
    conclusion = src(13).replace(
        'display(pd.read_csv(OUTPUT/"paper_comparison.csv"))',
        'comparison = pd.read_csv(OUTPUT/"paper_comparison.csv")\n'
        'display(comparison if len(comparison) else "per_ego_permute không có số liệu paper để đối chiếu.")')
    analysis = ANALYSIS_TEMPLATE.replace('__ANALYSIS_SOURCE__', repr(Path(analysis_path).read_text(encoding='utf-8')))
    new_cells = [
        cell('markdown', INTRO), cell('code', SETTINGS),
        cell('markdown', src(2)), cell('code', src(3)),
        cell('markdown', src(4)), cell('code', runtime),
        cell('markdown', src(6)), cell('code', src(7)),
        cell('markdown', src(8)), cell('code', configure),
        cell('markdown', src(10)), cell('code', src(11)),
        cell('markdown', src(12)), cell('code', conclusion),
        cell('markdown', '## 7. Phân tích hoàn thiện đồ án (CPU)'), cell('code', analysis),
        cell('markdown', '## 8. Bảng và biểu đồ theo kiểu paper\n\nTable 1 (dataset), Table 2 (so với baseline, ◇ = số paper, đậm = tốt nhất, gạch chân = thứ hai), Table 4 (temporal alignment + Δ theo cặp seed), Table 5 (thời gian), Figure A.1 (phân bố tương tác theo time step), Figure 4(c)-style (hidden dim của baseline), và các biểu đồ chẩn đoán (đường validation, so cặp seed, suy biến dự đoán). Figure 3 (multi-step) và sweep layer/head của Figure 4 không chạy nên không vẽ.'),
        cell('code', FIGURES_TEMPLATE.replace('__FIGURES_SOURCE__', repr(Path(figures_path).read_text(encoding='utf-8')))),
        cell('markdown', OUTRO),
    ]
    for c in new_cells:
        if c['cell_type'] == 'code':
            ast.parse(c['source'])
    notebook = dict(cells=new_cells, metadata=base['metadata'], nbformat=4, nbformat_minor=5)
    Path(out_path).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', out_path, Path(out_path).stat().st_size, 'bytes,', len(new_cells), 'cells')


if __name__ == '__main__':
    main(*sys.argv[1:5])
