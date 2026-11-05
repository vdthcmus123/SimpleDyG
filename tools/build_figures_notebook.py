"""Build SimpleDyG_Paper_Figures.ipynb: CPU-only notebook that draws paper-style tables/figures
from any attached SimpleDyG result output (100-run Output, its ZIP, SimpleDyG_Previous_Results.zip,
or outputs of SimpleDyG_Completion_2GPU.ipynb). No GPU, no torch.

Usage: python build_figures_notebook.py <paper_figures.py> <output.ipynb>
"""
import ast
import json
import sys
import uuid
from pathlib import Path


def cell(kind, source):
    c = dict(cell_type=kind, id=uuid.uuid4().hex[:8], metadata={}, source=source)
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


INTRO = '''# SimpleDyG – bảng và biểu đồ theo kiểu paper (chỉ CPU)

Notebook này **không huấn luyện**. Nó đọc kết quả đã có và vẽ lại các bảng/hình của paper:

| Paper | Ở đây |
|---|---|
| Table 1 – thống kê dataset | `table1_datasets` (+ số truy vấn split, chuỗi bị cắt) |
| Table 2 – so với baseline | bảng (đậm = tốt nhất, gạch chân = thứ hai) + biểu đồ cột, ◇ = số paper, nét đứt = baseline không huấn luyện |
| Table 4 – temporal alignment | bảng + Δ so cặp theo seed với CI 95% + biểu đồ |
| Table 5 – thời gian/epoch | `table5_time` (T4, gồm cả validation mỗi epoch) |
| Figure A.1 – phân bố tương tác theo time step | `fig_A1_temporal_pattern` |
| Figure 4(c) – hidden dimension | chỉ cho TGAT/GraphMixer (lưới chọn dim trên validation) |
| (thêm) chẩn đoán | đường validation theo epoch, so cặp từng seed, kiểm tra suy biến dự đoán |

Figure 3 (dự đoán nhiều bước) và sweep layer/head của Figure 4 không được chạy nên không vẽ.

**Input (Add Input trên Kaggle, chọn một hoặc nhiều):**
- Output của notebook 100 lượt (thư mục có `results_per_seed.csv` và `runs/`), **hoặc** `simple_dyg_result.zip`, **hoặc** `SimpleDyG_Previous_Results.zip` (1,3 MB, đủ cho mọi hình).
- Tùy chọn: Output của `SimpleDyG_Completion_2GPU.ipynb` (`uci_permute`, `ml_corrected`) – được gộp tự động.
- Tùy chọn: `SimpleDyG_Kaggle_2GPU_Inputs.zip` – cần cho Table 1 (số time step), Figure A.1 và các dòng baseline không huấn luyện.

Không cần GPU. Run All. Kết quả: `simpledyg_paper_figures.zip` (PNG, CSV, `paper_tables.html`).
'''

SETTINGS = '''NDCG_METRIC = "ndcg5"        # "ndcg5" = NDCG chuẩn (khử trùng lặp); "ndcg5_author" = cách tính trong mã/paper
INPUT_DIR = ""               # trống = /kaggle/input; khi chạy máy cá nhân: thư mục chứa các ZIP/Output
EXTRA_RESULT_ROOTS = []      # thêm thư mục kết quả thủ công nếu cần (mỗi thư mục có results_per_seed.csv)
AUTHOR_ROOT_OVERRIDE = ""    # thư mục SimpleDyG có resources/ và all_data/ (nếu không dùng ZIP input)
'''

SETUP = '''import os, sys, subprocess, json, zipfile, shutil, types
from pathlib import Path
try:
    import matplotlib
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib>=3.7,<4"], check=True)
import pandas as pd
INPUT = Path(INPUT_DIR) if INPUT_DIR else Path("/kaggle/input")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "simpledyg_figures_work"
WORK.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "mpl_cache"))
print("Input:", INPUT, "| Work:", WORK)
'''

DISCOVER = '''# Tìm kết quả trong thư mục và trong ZIP; chỉ giải nén file nhỏ cần cho bảng/hình (bỏ qua *.pt).
EXTRACT = WORK / "extracted_results"
KEEP_TOP = {"results_per_seed.csv", "data_audit.json"}
KEEP_RUN = ("/history.json", "/validation_search.json", "/test_predictions.jsonl")

def safe_extract(archive, member, target):
    dest = (target / member["rel"]).resolve()
    if not dest.is_relative_to(target.resolve()):
        raise ValueError("ZIP path escapes target: " + member["rel"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member["name"]) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)

roots = [Path(p) for p in EXTRA_RESULT_ROOTS]
if INPUT.exists():
    roots += sorted(p.parent for p in INPUT.rglob("results_per_seed.csv") if p.stat().st_size > 1)
    for z in sorted(INPUT.rglob("*.zip")):
        try:
            archive = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            continue
        with archive:
            names = archive.namelist()
            prefixes = sorted({n[:-len("results_per_seed.csv")] for n in names if n.endswith("results_per_seed.csv")})
            for k, prefix in enumerate(prefixes):
                target = EXTRACT / f"{z.stem}_{k}"
                members = []
                for n in names:
                    if n.startswith(prefix) and not n.endswith("/"):
                        rel = n[len(prefix):]
                        if rel in KEEP_TOP or (rel.startswith("runs/") and rel.endswith(KEEP_RUN)):
                            members.append(dict(name=n, rel=rel))
                for m in members:
                    safe_extract(archive, m, target)
                print(f"ZIP {z.name}: {len(members)} file kết quả -> {target}")
                roots.append(target)
# Bỏ thư mục trùng; kết quả trùng run_id được gộp trong paper_figures.load_results.
seen, RESULT_ROOTS = set(), []
for r in roots:
    key = r.resolve()
    if key not in seen and (r / "results_per_seed.csv").exists():
        seen.add(key)
        RESULT_ROOTS.append(r)
if not RESULT_ROOTS:
    raise FileNotFoundError("Không tìm thấy results_per_seed.csv. Add Input Output 100 lượt, simple_dyg_result.zip "
                            "hoặc SimpleDyG_Previous_Results.zip.")
for r in RESULT_ROOTS:
    n = len(pd.read_csv(r / "results_per_seed.csv"))
    print(f"{n:4d} run | {r}")

# Dữ liệu gốc cho Table 1 / Figure A.1 / baseline không huấn luyện (không bắt buộc).
AUTHOR_ROOT = Path(AUTHOR_ROOT_OVERRIDE) if AUTHOR_ROOT_OVERRIDE else None
if AUTHOR_ROOT is None and INPUT.exists():
    for p in INPUT.rglob("train.link_prediction"):
        if p.parent.name == "12" and p.parent.parent.name == "UCI_13":
            AUTHOR_ROOT = p.parents[3]
            break
if AUTHOR_ROOT is None and INPUT.exists():
    for z in sorted(INPUT.rglob("*.zip")):
        try:
            archive = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            continue
        with archive:
            names = archive.namelist()
            anchor = [n for n in names if n.endswith("resources/UCI_13/12/train.link_prediction")]
            if not anchor:
                continue
            prefix = anchor[0][:-len("resources/UCI_13/12/train.link_prediction")]
            target = WORK / "author_data"
            for n in names:
                rel = n[len(prefix):] if n.startswith(prefix) else None
                if rel and not n.endswith("/") and (rel.startswith("resources/") or (rel.startswith("all_data/") and rel.endswith(".csv"))):
                    safe_extract(archive, dict(name=n, rel=rel), target)
            AUTHOR_ROOT = target
            break
print("Dữ liệu gốc:", AUTHOR_ROOT or "không có – bỏ qua Figure A.1, số time step và baseline không huấn luyện")
'''

RENDER = '''PAPER_FIGURES_SOURCE = __FIGURES_SOURCE__
pfig = types.ModuleType("paper_figures")
exec(compile(PAPER_FIGURES_SOURCE, "paper_figures.py", "exec"), pfig.__dict__)
FIGURES = WORK / "simpledyg_paper_figures"
if FIGURES.exists():
    shutil.rmtree(FIGURES)
figures = pfig.render_all(RESULT_ROOTS, AUTHOR_ROOT, FIGURES, ndcg_col=NDCG_METRIC)
df = figures["df"]
print("Tổng số run:", len(df))
display(df.groupby(["dataset", "model", "variant", "protocol"]).seed.nunique().rename("n_seeds").reset_index())
'''

SHOW = '''pfig.show(figures)
'''

EXPORT = '''archive = shutil.make_archive(str(WORK / "simpledyg_paper_figures"), "zip", FIGURES)
print("Đã lưu:", archive)
print("Gồm:", sorted(p.name for p in FIGURES.iterdir()))
try:
    from IPython.display import FileLink
    display(FileLink(os.path.relpath(archive)))
except Exception:
    pass
'''

NOTES = '''## Gợi ý đọc

- **Table 2:** số của chúng tôi dùng metric chọn ở `NDCG_METRIC`. Jaccard của baseline là max theo k ∈ {1, 5, 10, 20} trên test (như paper, mang tính oracle); Jaccard của SimpleDyG lấy theo tập sinh ra. Sáu baseline paper không chạy được hiển thị “–”.
- **Nét đứt và chấm** là baseline không huấn luyện: lặp lại slot gần nhất có hoạt động, và đích phổ biến nhất trong train. Trên ML-10M, mọi mô hình đều chạm mức “phổ biến” (xem biểu đồ suy biến).
- **Table 4:** cột Δ so cặp cùng seed với `full` cùng dataset/protocol; “Significant?” = CI 95% không chứa 0.
- **Table 5:** giây/epoch trên T4 gồm cả giải mã validation; paper đo trên L40, chỉ phần huấn luyện. Chỉ nên so tương đối.
- Đổi `NDCG_METRIC = "ndcg5_author"` để có số đúng cách tính của paper (khác chủ yếu trên UCI).
'''


def main(figures_path, out_path):
    source = Path(figures_path).read_text(encoding='utf-8')
    cells = [cell('markdown', INTRO), cell('code', SETTINGS),
             cell('markdown', '## 1. Môi trường'), cell('code', SETUP),
             cell('markdown', '## 2. Tìm kết quả và dữ liệu'), cell('code', DISCOVER),
             cell('markdown', '## 3. Dựng bảng và biểu đồ'), cell('code', RENDER.replace('__FIGURES_SOURCE__', repr(source))),
             cell('markdown', '## 4. Bảng (Table 1, 2, 4, 5, suy biến) và biểu đồ'), cell('code', SHOW),
             cell('markdown', '## 5. Tải về'), cell('code', EXPORT),
             cell('markdown', NOTES)]
    for c in cells:
        if c['cell_type'] == 'code':
            ast.parse(c['source'])
    nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                            language_info=dict(name='python', version='3.11'),
                            kaggle=dict(isInternetEnabled=True, isGpuEnabled=False, language='python',
                                        sourceType='notebook')))
    Path(out_path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', out_path, Path(out_path).stat().st_size, 'bytes,', len(cells), 'cells')


if __name__ == '__main__':
    main(*sys.argv[1:3])
