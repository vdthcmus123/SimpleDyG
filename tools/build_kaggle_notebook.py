"""Build the single-file notebook and the optional offline Kaggle input archive."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/'tools/kaggle_pipeline.py'

intro = '''# SimpleDyG: tái lập, benchmark và temporal alignment trên Kaggle

Paper: **On the Feasibility of Simple Transformer for Dynamic Graph Modeling**, WWW 2024.
[Paper](https://arxiv.org/abs/2401.14009) · [Repo tác giả](https://github.com/YuxiaWu/SimpleDyG) · [DyGLib](https://github.com/yule-BUAA/DyGLib)

Notebook này chạy **dynamic link prediction một bước**: dự đoán tập node liên kết với ego-node ở bin cuối.
Mặc định: **UCI + ML-10M**, SimpleDyG và **TGAT, GraphMixer**, ba seed, ablation **full / same_time / no_time**.
DyGFormer có thể bật thêm. Không cần copy các file `.py` bên cạnh; toàn bộ mã điều phối nằm trong notebook.

**Cách chạy**

1. Import notebook vào Kaggle. Bật GPU và Internet nếu tải repo/dependency trực tiếp. Pipeline dùng **một GPU** để giữ batch size rõ ràng.
2. Nên chạy `PROFILE = "smoke"` trước. Đây là kiểm tra thực thi trên subset, không phải kết quả khoa học.
3. Chọn `PROFILE = "pilot"` để chạy toàn bộ hai dataset với ba seed. `paper` dùng 10 seed và thiết lập tương thích tác giả; việc tìm dimension baseline ở chế độ này rất tốn thời gian.
4. Nếu chạy offline, Add Input file `SimpleDyG_Kaggle_Inputs.zip` đi kèm, cùng các wheel cần thiết nếu môi trường chưa có dependency. Đặt `OFFLINE = True`. Chạy tuần tự các cell.
5. Kết quả lưu ở `/kaggle/working/simpledyg_<profile>_<protocol>/`; ZIP kết quả không chứa checkpoint nặng trừ khi bật `EXPORT_CHECKPOINTS`.

**Tương thích Python 3.13:** tokenizer node, GELU và legacy optimizer được nhúng bằng PyTorch/Python.
Setup không cài hay hạ phiên bản `transformers`/`tokenizers`, không biên dịch Rust; giữ PyTorch/CUDA của Kaggle.

**Giới hạn khoa học cần giữ**

- Hepth không được chọn vì release hiện tại thiếu `resources/hepth/node_features.npy`; không thay feature word2vec bằng dữ liệu giả rồi gọi là reproduce.
- Paper chia theo thời gian: train bin 0–10, validation bin 11, test bin 12, cho UCI/ML-10M. Giữ nguyên query và target từ release; các query là source-node, không tự thêm reverse target.
- Vocabulary release bao gồm danh tính node toàn timeline. Đây là giả định **closed world** của release, không phải tuyên bố inductive cho node chưa biết.
- `author_compat` dùng chính GPT-2 của tác giả, loss gồm padding, truncate train bên trái 512, validation tối đa 11 token, test đến giới hạn context/EOS, và NDCG có duplicate. Trainer được thay để chạy trên môi trường mới, nên không đảm bảo bit-for-bit.
- `corrected` mask padding, dừng ở end-of-prediction/EOS, dùng cùng giới hạn generation cho val/test và NDCG chuẩn deduplicate. Cần báo rõ khác protocol khi so với Table 2/4.
- Baseline dùng backbone **thật của DyGLib**, loss BPR thích nghi từ event train; history toàn graph undirected nhưng chỉ lấy cạnh trước cutoff. Evaluation giữ nguyên ego, target, vocabulary của release và chấm toàn bộ candidate, không sampled-negative AP/AUC.
- UCI feature release là toàn zero; ML-10M không có feature `.npy` đã căn theo node ID. Baseline dùng feature zero, tune dimension trên validation. Đây là benchmark có mô tả đầy đủ, chưa phải tái lập chính xác toàn bộ cấu hình baseline Table 2.
- `pilot` giới hạn history baseline để khả thi; `paper` nâng history lên 1024. Jaccard của baseline chọn k bằng validation; max-over-k trên test chỉ là cột **oracle** tham chiếu paper.
- Không in sẵn giả thuyết đúng/sai. Table 4 cho thấy `no_time` có thể tốt hơn `full`; `same_time` và `no_time` đã có trong paper, không phải hướng mới.
'''

settings = '''# Cấu hình duy nhất cần sửa trước Run All.
PROFILE = "pilot"                    # smoke | pilot | paper
PROTOCOL_OVERRIDE = None             # None theo profile; "corrected" | "author_compat"
DATASETS_OVERRIDE = None             # None = hai dataset (smoke chỉ UCI); hoặc ["UCI_13"]
SEEDS_OVERRIDE = None                # None = profile; có thể chọn [0] khi kiểm tra
BASELINE_MODELS = ["TGAT", "GraphMixer"]  # thêm "DyGFormer" nếu đủ thời gian
VARIANTS = ["full", "same_time", "no_time"]  # thêm per_ego_permute chỉ khi thử control bổ sung
OFFLINE = False
INSTALL_DEPENDENCIES = True
AUTHOR_ROOT_OVERRIDE = ""            # root repo được Add Input; tự tìm nếu để trống
DYGLIB_ROOT_OVERRIDE = ""
INPUT_BUNDLE_OVERRIDE = ""           # đường dẫn ZIP input nếu tự tìm không thấy
EXPORT_CHECKPOINTS = False
# Chỉ thay giới hạn/cấu hình ở đây; mọi override sẽ được lưu vào manifest.
EXTRA_CONFIG = {}                    # ví dụ {"epochs": 20}; kết quả không còn cùng budget paper
'''

setup = '''import os, sys, subprocess, importlib.util
from pathlib import Path
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()/"kaggle_working"
WORK.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(WORK/"hf_cache")
os.environ["MPLCONFIGDIR"] = str(WORK/"mpl_cache")
requirements = {"boto3": "boto3>=1.34,<2", "matplotlib": "matplotlib>=3.7,<4",
                "requests": "requests>=2.31,<3", "filelock": "filelock>=3.12,<4",
                "tqdm": "tqdm>=4.65,<5"}
from importlib.metadata import version, PackageNotFoundError
needed = []
for name, requirement in requirements.items():
    try:
        installed = version(name)
    except PackageNotFoundError:
        needed.append(requirement)
if needed and INSTALL_DEPENDENCIES:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--only-binary=:all:"]
    if OFFLINE:
        wheels = sorted({str(p.parent) for p in Path("/kaggle/input").rglob("*.whl")})
        if not wheels:
            raise RuntimeError("Thiếu dependency khi offline. Add Input các wheel hoặc bật Internet rồi chạy lại setup.")
        cmd += ["--no-index"]
        for directory in wheels:
            cmd += ["--find-links", directory]
    subprocess.run(cmd + needed, check=True)
elif needed:
    raise RuntimeError(f"Thiếu/sai phiên bản dependency: {needed}")
import torch
from packaging.version import Version
if Version(torch.__version__.split("+")[0]) < Version("2.3"):
    raise RuntimeError("Pipeline cần PyTorch >=2.3 cho native AMP. Chọn image Kaggle mới; setup giữ PyTorch/CUDA có sẵn.")
print("Tokenizer node và legacy optimizer đã nhúng; không cài transformers/tokenizers, không build Rust.")
print("Python:", sys.version.split()[0], "| Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
if PROFILE != "smoke" and not torch.cuda.is_available():
    raise RuntimeError("Bật GPU Kaggle hoặc chọn PROFILE='smoke' để kiểm tra CPU.")
'''

bootstrap = '''# Ưu tiên bundle/attached repo; chỉ clone đúng commit khi chưa có dữ liệu.
INPUT = Path("/kaggle/input")
bundles = [Path(INPUT_BUNDLE_OVERRIDE)] if INPUT_BUNDLE_OVERRIDE else (list(INPUT.rglob("SimpleDyG_Kaggle_Inputs.zip")) if INPUT.exists() else [])
EXTRACTED = WORK/"simpledyg_inputs"
if bundles:
    with zipfile.ZipFile(bundles[0]) as archive:
        for info in archive.infolist():
            target = (EXTRACTED/info.filename).resolve()
            if not target.is_relative_to(EXTRACTED.resolve()):
                raise ValueError("ZIP chứa đường dẫn vượt ngoài thư mục input")
        archive.extractall(EXTRACTED)
    manifest = json.loads((EXTRACTED/"SOURCE_MANIFEST.json").read_text())
    for relative, expected in manifest["sha256"].items():
        if sha256(EXTRACTED/relative) != expected:
            raise ValueError("Input checksum mismatch: "+relative)
    assert manifest["author_commit"] == AUTHOR_COMMIT and manifest["dyglib_commit"] == DYGLIB_COMMIT

def obtain_repo(label, override, url, commit):
    if override:
        root = Path(override)
    elif (EXTRACTED/label).exists():
        root = EXTRACTED/label
    else:
        root = None
        if INPUT.exists():
            if label == "SimpleDyG":
                for p in INPUT.rglob("train.link_prediction"):
                    if p.parent.name == "12" and p.parent.parent.name == "UCI_13":
                        root = p.parents[3]
                        break
            else:
                found = list(INPUT.rglob("TGAT.py"))
                root = found[0].parents[1] if found else None
        if root is None:
            root = WORK/label
            if not root.exists():
                if OFFLINE:
                    raise FileNotFoundError(f"Offline: không tìm thấy {label}. Add Input bundle được cung cấp.")
                subprocess.run(["git", "clone", "--no-checkout", url, str(root)], check=True)
                subprocess.run(["git", "-C", str(root), "checkout", commit], check=True)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if (root/".git").exists():
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != commit:
            raise ValueError(f"{label} commit khác pin: {revision}. Dùng bundle hoặc clone đúng commit {commit}.")
    print(label, root)
    return root

AUTHOR_ROOT = obtain_repo("SimpleDyG", AUTHOR_ROOT_OVERRIDE, "https://github.com/YuxiaWu/SimpleDyG.git", AUTHOR_COMMIT)
DYGLIB_ROOT = obtain_repo("DyGLib", DYGLIB_ROOT_OVERRIDE, "https://github.com/yule-BUAA/DyGLib.git", DYGLIB_COMMIT)
for root, required in [(AUTHOR_ROOT, ["models/modeling_gpt2.py", "resources/UCI_13/12/train.link_prediction"]),
                       (DYGLIB_ROOT, ["models/TGAT.py", "models/GraphMixer.py", "utils/utils.py"])]:
    for relative in required:
        if not (root/relative).is_file():
            raise FileNotFoundError(root/relative)
'''

configure = '''overrides = dict(variants=VARIANTS, baseline_models=BASELINE_MODELS, export_checkpoints=EXPORT_CHECKPOINTS)
if PROTOCOL_OVERRIDE is not None:
    overrides["protocol"] = PROTOCOL_OVERRIDE
if DATASETS_OVERRIDE is not None:
    overrides["datasets"] = DATASETS_OVERRIDE
if SEEDS_OVERRIDE is not None:
    overrides["seeds"] = SEEDS_OVERRIDE
overrides.update(EXTRA_CONFIG)
CFG = make_config(PROFILE, **overrides)
OUTPUT = WORK/f"simpledyg_{PROFILE}_{CFG['protocol']}"
print(json.dumps(CFG, ensure_ascii=False, indent=2))
expected = len(CFG["datasets"])*len(CFG["seeds"])*(len(CFG["variants"])+len(CFG["baseline_models"]))
print("Số run:", expected, "| Baseline dimension candidates:", CFG["baseline_dims"])
# Validate toàn bộ file/split trước khi train. Không bỏ qua dataset bị thiếu.
for name in CFG["datasets"]:
    dataset = load_dataset(AUTHOR_ROOT, name, CFG)
    print(name, json.dumps(dataset["audit"]["splits"], indent=2))
'''

execution = '''# Cell chạy dài. Có log mỗi epoch, early stopping, best checkpoint bằng validation.
# Mỗi run hoàn thành được lưu atomic; chạy lại sẽ tái sử dụng đúng fingerprint.
# Run bị ngắt giữa chừng sẽ train lại từ đầu; chưa có resume optimizer giữa epoch.
results, summary, result_archive = run_pipeline(AUTHOR_ROOT, DYGLIB_ROOT, CFG, OUTPUT)
display(summary)
'''

conclusion = '''# Chỉ hiển thị kết quả thực đo; không có kết luận cài sẵn.
from IPython.display import display, FileLink, Image
display(FileLink(str(result_archive)))
display(results[["dataset", "model", "variant", "seed", "ndcg5", "jaccard_at5", "jaccard", "n_evaluated"]])
for p in [OUTPUT/"ndcg5.png", OUTPUT/"jaccard_at5.png"]:
    if p.exists():
        display(Image(filename=str(p)))
paired = pd.read_csv(OUTPUT/"ablation_paired_deltas.csv")
display(paired)
display(pd.read_csv(OUTPUT/"paper_comparison.csv"))
print((OUTPUT/"READ_RESULTS.txt").read_text())
if CFG["profile"] == "smoke":
    print("SMOKE PASSED: chỉ xác nhận nhánh code hoạt động. Đổi pilot/paper để lấy kết quả nghiên cứu.")
print("Lưu Kaggle Version và tải ZIP kết quả trước khi hết phiên GPU.")
'''


def build():
    source = CORE.read_text(encoding='utf-8')
    ast.parse(source)
    fingerprint = hashlib.sha256(source.encode()).hexdigest()
    nb = nbf.v4.new_notebook()
    nb.metadata = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                       language_info=dict(name='python', version='3.13'),
                       kaggle=dict(accelerator='gpu', isInternetEnabled=True))
    cells = [nbf.v4.new_markdown_cell(intro), nbf.v4.new_code_cell(settings),
             nbf.v4.new_markdown_cell('## 1. Dependency và GPU'), nbf.v4.new_code_cell(setup),
             nbf.v4.new_markdown_cell('## 2. Mã pipeline nhúng trong notebook\nCell này định nghĩa hàm, chưa train. Có thể thu gọn khi đọc.'),
             nbf.v4.new_code_cell(f'PIPELINE_SOURCE_SHA256 = {fingerprint!r}\n'+source),
             nbf.v4.new_markdown_cell('## 3. Dữ liệu và source được pin'), nbf.v4.new_code_cell(bootstrap),
             nbf.v4.new_markdown_cell('## 4. Kiểm tra split và kế hoạch thực nghiệm'), nbf.v4.new_code_cell(configure),
             nbf.v4.new_markdown_cell('## 5. Train, chọn checkpoint, benchmark và ablation'), nbf.v4.new_code_cell(execution),
             nbf.v4.new_markdown_cell('## 6. Kết quả, sai khác và export'), nbf.v4.new_code_cell(conclusion),
             nbf.v4.new_markdown_cell('''## Sau khi chạy xong

Đọc `data_audit.json`, `environment.json`, `config.json`, `results_per_seed.csv`, `summary_mean_std.csv`,
`ablation_paired_deltas.csv`, và JSONL dự đoán từng query. Số seed chưa đủ hoặc run thất bại phải được báo.
`paper_reference.csv` là số công bố; không được gộp thành kết quả đo mới.
So sánh NDCG@5 và Jaccard@5 chung giữa model trước; xem Jaccard theo cách tạo tập của mỗi model sau.
Chênh lệch với paper cần được phân tích bằng protocol, giới hạn history/generation và seed, không áp ngưỡng ±2% tùy ý.

Nếu nghiên cứu temporal alignment, dùng delta từng seed cùng dữ liệu/architecture; `same_time` và `no_time` là reproduction Table 4.
Control `per_ego_permute` giữ thứ tự node, chỉ phá đồng bộ nhãn thời gian giữa ego; nó không đảo thứ tự sự kiện.
Đề xuất mới chỉ nên phát triển sau khi chạy baseline đủ seed và kiểm tra kết quả, không suy từ smoke test.
''')]
    nb.cells = cells
    nbf.validate(nb)
    for c in nb.cells:
        if c.cell_type == 'code':
            ast.parse(c.source)
    dest = ROOT/'notebooks/SimpleDyG_Kaggle_Full_Pipeline.ipynb'
    nbf.write(nb, dest)
    members = {}
    author = ROOT/'repo_clone'
    for directory in ['models', 'resources/UCI_13/12', 'resources/ML_10M_13/12',
                      'vocabs/UCI_13/12', 'vocabs/ML_10M_13/12']:
        for p in (author/directory).rglob('*'):
            if p.is_file() and p.suffix != '.pyc':
                members['SimpleDyG/'+p.relative_to(author).as_posix()] = p
    for rel in ['README.md', 'main.py', 'requirements.txt', 'train_seed_UCI_13.sh', 'train_seed_ML_10M_13.sh',
                'all_data/uci/12/ml_uci.csv', 'all_data/movie/12/ml_movie.csv']:
        members['SimpleDyG/'+rel] = author/rel
    vendor = ROOT/'tmp/inspection/DyGLib'
    for p in vendor.rglob('*'):
        if p.is_file() and p.suffix != '.pyc':
            members['DyGLib/'+p.relative_to(vendor).as_posix()] = p
    manifest = dict(author_commit='fa0dc82587d8ce787345a0379f65a0dcf5d9d9ff',
                    dyglib_commit='3aacc36b94b8d2d8293d70a74fdf6d39089b4163',
                    scope='minimal source and preprocessed data bundle; not full git checkouts',
                    sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in members.items()})
    archive = ROOT/'notebooks/SimpleDyG_Kaggle_Inputs.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, file in members.items():
            z.write(file, name)
        z.writestr('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2))
    print(dest, dest.stat().st_size, 'bytes;', len(cells), 'cells')
    print(archive, archive.stat().st_size, 'bytes')


if __name__ == '__main__':
    build()
