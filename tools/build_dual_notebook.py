"""Package the dedicated two-GPU/full-checkpoint Kaggle notebook and source ZIP."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import zipfile
import nbformat as nbf
import build_kaggle_notebook as base

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = (ROOT / 'tools/kaggle_pipeline_dual.py').read_text(encoding='utf-8')
    fingerprint = hashlib.sha256(source.encode()).hexdigest()
    settings = '''# Run mới từ đầu: chọn Kaggle Settings > Accelerator > GPU T4 x2.
PROFILE = "paper"                   # smoke | pilot | paper
PROTOCOL_OVERRIDE = None
DATASETS_OVERRIDE = None            # None = UCI và ML-10M
SEEDS_OVERRIDE = None               # None = 10 seed với paper
BASELINE_MODELS = ["TGAT", "GraphMixer"]
VARIANTS = ["full", "same_time", "no_time"]
EXTRA_CONFIG = {}                    # không đổi sau khi đã bắt đầu thí nghiệm

GPU_IDS = [0, 1]
SESSION_HOURS = 10.5                 # tạm dừng trước 12h để lưu Output/ZIP
CHECKPOINT_SECONDS = 300             # thêm checkpoint ở ranh giới mỗi epoch
WORKER_CPU_THREADS = 2
SNAPSHOT_INCLUDE_COMPLETED_BEST = True   # ZIP gồm cả best.pt của các run đã hoàn thành

RESUME = False                      # phiên đầu False; phiên tiếp theo True
RESUME_FROM = ""                    # ZIP/thư mục Output mới nhất; trống = tự tìm Input
OFFLINE = False
INSTALL_DEPENDENCIES = True
AUTHOR_ROOT_OVERRIDE = ""
DYGLIB_ROOT_OVERRIDE = ""
INPUT_BUNDLE_OVERRIDE = ""           # ZIP source SimpleDyG_Kaggle_2GPU_Inputs.zip
'''
    setup = base.setup + '''
if PROFILE != "smoke" and torch.cuda.device_count() < 2:
    raise RuntimeError("Notebook này cần GPU T4 x2. Hãy chọn hai GPU trong Kaggle Settings.")
print("Visible GPUs:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print("GPU", index, torch.cuda.get_device_name(index))
'''
    runtime = '''import json, zipfile, hashlib, importlib.util
import pandas as pd
DUAL_RUNTIME_SOURCE = ''' + repr(source) + '''
RUNTIME_PATH = WORK / "simpledyg_dual_runtime.py"
RUNTIME_PATH.write_text(DUAL_RUNTIME_SOURCE, encoding="utf-8", newline="\\n")
assert hashlib.sha256(RUNTIME_PATH.read_text(encoding="utf-8").encode()).hexdigest() == "''' + fingerprint + '''"
spec = importlib.util.spec_from_file_location("simpledyg_dual_runtime", RUNTIME_PATH)
dual = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dual
spec.loader.exec_module(dual)
sha256, AUTHOR_COMMIT, DYGLIB_COMMIT = dual.sha256, dual.AUTHOR_COMMIT, dual.DYGLIB_COMMIT
print("Runtime SHA-256:", "''' + fingerprint + '''")
'''
    bootstrap = base.bootstrap.replace('list(INPUT.rglob("SimpleDyG_Kaggle_Inputs.zip"))',
                                        'list(INPUT.rglob("SimpleDyG_Kaggle_2GPU_Inputs.zip"))')
    source_alignment = (ROOT / 'tools/resume_source_alignment.py').read_text(encoding='utf-8')
    configure = source_alignment + '\n\n' + '''OUTPUT = WORK / "simpledyg_2gpu_results"
if RESUME:
    previous = Path(RESUME_FROM) if RESUME_FROM else dual.find_dual_output(INPUT)
    dual.restore_dual_output(previous, OUTPUT)
    saved = json.loads((OUTPUT / "config.json").read_text())
    AUTHOR_ROOT, DYGLIB_ROOT, source_alignment_report = align_resume_sources(
        dual, AUTHOR_ROOT, DYGLIB_ROOT, saved, OUTPUT, WORK)
    (OUTPUT / "resume_source_alignment.json").write_text(
        json.dumps(source_alignment_report, indent=2), encoding="utf-8")
    # Full config comes from the previous output; scheduling options remain editable.
    CFG = {k: v for k, v in saved.items() if not k.startswith("_")}
    print("RESUME source:", previous)
else:
    if (OUTPUT / "config.json").exists():
        raise RuntimeError("Output đã có thí nghiệm. Đặt RESUME=True để tiếp tục; dùng phiên mới để chạy từ đầu.")
    overrides = dict(variants=VARIANTS, baseline_models=BASELINE_MODELS, export_checkpoints=True)
    if PROTOCOL_OVERRIDE is not None:
        overrides["protocol"] = PROTOCOL_OVERRIDE
    if DATASETS_OVERRIDE is not None:
        overrides["datasets"] = DATASETS_OVERRIDE
    if SEEDS_OVERRIDE is not None:
        overrides["seeds"] = SEEDS_OVERRIDE
    overrides.update(EXTRA_CONFIG)
    CFG = dual.make_config(PROFILE, **overrides)
print(json.dumps(CFG, indent=2))
for name in CFG["datasets"]:
    audit = dual.load_dataset(AUTHOR_ROOT, name, CFG)["audit"]
    print(name, json.dumps(audit["splits"], indent=2))
print("Planned runs:", len(CFG["datasets"])*len(CFG["seeds"])*(len(CFG["variants"])+len(CFG["baseline_models"])))
'''
    execution = '''# Hai tiến trình độc lập; mỗi GPU xử lý một run, không nhân đôi batch size.
# Runtime ghi latest.pt đầy đủ, best.pt riêng và kết quả bằng atomic replace.
# Smoke có thể kiểm tra cùng đường chạy hai tiến trình trên CPU khi không có GPU.
cpu_check = 2 if CFG["profile"] == "smoke" and not torch.cuda.is_available() else 0
results, summary, resume_archive = dual.run_dual_pipeline(
    AUTHOR_ROOT, DYGLIB_ROOT, CFG, OUTPUT, gpu_ids=GPU_IDS,
    session_hours=SESSION_HOURS, checkpoint_seconds=CHECKPOINT_SECONDS,
    worker_threads=WORKER_CPU_THREADS, include_completed_best=SNAPSHOT_INCLUDE_COMPLETED_BEST,
    cpu_workers=cpu_check)
display(summary)
'''
    conclusion = '''from IPython.display import FileLink, Image, display
display(FileLink(str(resume_archive)))
display(results)
for filename in ["ndcg5.png", "jaccard_at5.png"]:
    if (OUTPUT/filename).exists():
        display(Image(filename=str(OUTPUT/filename)))
display(pd.read_csv(OUTPUT/"paper_comparison.csv"))
display(pd.read_csv(OUTPUT/"ablation_paired_deltas.csv"))
records = json.loads((OUTPUT/"run_manifest.json").read_text())
print("Run status:", dict(dual.Counter(r["status"] for r in records)))
print("Lưu Version/Output và resume_snapshot.zip. Phiên sau Add Input Output mới nhất, đặt RESUME=True.")
print("ZIP resume mặc định gồm checkpoint đầy đủ của run đang dở và best.pt của các run hoàn thành.")
'''
    notebook = nbf.v4.new_notebook()
    notebook.metadata = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                             language_info=dict(name='python', version='3.13'),
                             kaggle=dict(isInternetEnabled=True, isGpuEnabled=True,
                                         accelerator='nvidiaTeslaT4', language='python', sourceType='notebook'))
    notebook.cells = [
        nbf.v4.new_markdown_cell('''# SimpleDyG: hai GPU T4 và checkpoint đầy đủ

Chạy từ đầu hai dataset UCI/ML-10M, 10 seed và ba biến thể temporal alignment với
`PROFILE="paper"`. Add Input ZIP source đi kèm rồi chọn **GPU T4 x2**.
Nên thử `PROFILE="smoke"` trong một phiên riêng trước phiên paper.

Hai worker lấy các run độc lập từ hàng đợi chung; mỗi worker chỉ nhìn thấy một GPU.
Giữ architecture, logical batch size, split, loss, protocol và chọn checkpoint bằng validation.
Baseline dùng gradient accumulation theo micro-batch (mặc định 8 event cho paper),
giữ logical batch 128; luồng dropout có thể khác bản baseline chạy một forward lớn.

`latest.pt` lưu model hiện tại, best model, optimizer, scheduler/scaler khi dùng,
RNG Python/NumPy/Torch/CUDA, trạng thái shuffle/negative sampler, epoch và batch cursor.
Sau khi đổi phiên, runtime khôi phục trạng thái thay vì khởi tạo optimizer mới.
Checkpoint cũ của notebook một GPU không tương thích định dạng mới; lịch này bắt đầu từ đầu.

Mặc định tạm dừng sau 10,5 giờ ở ranh giới batch hoặc evaluation để dành thời gian
đóng ZIP. Nếu bị tắt đột ngột, dùng latest.pt đã ghi đủ gần nhất; đoạn sau checkpoint
sẽ được tính lại. ZIP được ghi sang file tạm rồi mới thay snapshot hoàn chỉnh cũ.
Hai GPU không gộp VRAM cho một model; toàn bộ lịch 100 run có thể cần nhiều phiên.

Các giới hạn benchmark của bản trước vẫn áp dụng: BPR adapter từ DyGLib, zero features,
closed-world vocabulary và metric tương thích tác giả được ghi riêng. Kết quả smoke
chỉ kiểm tra thực thi, không dùng để kết luận reproduce.
'''),
        nbf.v4.new_code_cell(settings),
        nbf.v4.new_markdown_cell('## 1. Môi trường và hai GPU'), nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell('## 2. Runtime tự chứa cho hai tiến trình'), nbf.v4.new_code_cell(runtime),
        nbf.v4.new_markdown_cell('## 3. Source và dữ liệu được pin'), nbf.v4.new_code_cell(bootstrap),
        nbf.v4.new_markdown_cell('## 4. Cấu hình mới hoặc khôi phục phiên trước'), nbf.v4.new_code_cell(configure),
        nbf.v4.new_markdown_cell('## 5. Chạy song song, checkpoint và tạm dừng'), nbf.v4.new_code_cell(execution),
        nbf.v4.new_markdown_cell('## 6. Kết quả và gói chạy tiếp'), nbf.v4.new_code_cell(conclusion),
        nbf.v4.new_markdown_cell('''## Phiên tiếp theo

1. Lưu Output và `simpledyg_2gpu_results/resume_snapshot.zip` của phiên hiện tại.
2. Add Input **Output mới nhất** hoặc ZIP snapshot mới nhất; bỏ các snapshot cũ khỏi Input.
3. Giữ ZIP source/dataset cũ và image PyTorch/GPU type cũ, bật GPU T4 x2.
4. Đặt `RESUME=True`; để `RESUME_FROM=""` nếu chỉ có một snapshot đúng.
5. Run All. Log `RESUME ... epoch=... next_batch=... phase=...` xác nhận đọc checkpoint;
   các run hoàn thành sẽ được bỏ qua. Chỉ chỉnh SESSION_HOURS/CHECKPOINT_SECONDS khi chia phiên.

Checkpoint tương thích source/data/config/environment cũ được kiểm tra trước khi dùng.
Khi source lấy từ full clone có thêm file Python, notebook dựng thư mục source đúng
manifest đã lưu sau khi kiểm tra checksum từng file; giữ nguyên runtime và run ID.
Đổi seed, epoch, history, model, dimension, micro-batch hoặc protocol tạo thí nghiệm khác.
`SNAPSHOT_INCLUDE_COMPLETED_BEST=True` mặc định giữ cả trọng số hoàn thành trong ZIP.
Có thể đặt False để giảm dung lượng ZIP resume; khi đó cần giữ riêng Output gốc chứa best.pt.
Giữ toàn bộ Output để phân tích prediction và đánh giá lại từng checkpoint.

Nếu có run failed, đọc worker_0.log/worker_1.log và runs/.../error.txt; các trạng thái
paused/pending thể hiện lịch còn thiếu. Chỉ báo kết quả theo số seed thực tế đã hoàn thành.
''')]
    nbf.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    target = ROOT / 'notebooks/SimpleDyG_Kaggle_2GPU_Resume.ipynb'
    nbf.write(notebook, target)
    resume_notebook = copy.deepcopy(notebook)
    resume_notebook.cells[0].source = resume_notebook.cells[0].source.replace(
        '# SimpleDyG: hai GPU T4 và checkpoint đầy đủ',
        '# SimpleDyG: bản sửa chạy tiếp từ Output và source manifest', 1)
    resume_notebook.cells[0].source += '''\n\n**Bản chạy tiếp:** mặc định `RESUME=True`.
Add Input Output của phiên thành công gần nhất và chọn GPU T4 x2. Giữ ZIP source
cũ nếu có; khi cần clone repo, notebook vẫn đối chiếu inventory/checksum của Output.
Lỗi `Existing output has different config/core/source` do full clone có thêm file
được xử lý ở phần chuẩn bị source; checkpoint và training runtime không thay đổi.
'''
    resume_notebook.cells[1].source = resume_notebook.cells[1].source.replace(
        '# Run mới từ đầu:', '# Chạy tiếp Output mới nhất:', 1).replace(
        'RESUME = False', 'RESUME = True', 1)
    resume_target = ROOT / 'notebooks/SimpleDyG_Kaggle_2GPU_Continue_Fixed.ipynb'
    nbf.validate(resume_notebook)
    nbf.write(resume_notebook, resume_target)
    archive = ROOT / 'notebooks/SimpleDyG_Kaggle_2GPU_Inputs.zip'
    with zipfile.ZipFile(ROOT / 'notebooks/SimpleDyG_Kaggle_Inputs.zip') as old, zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as fresh:
        for info in old.infolist():
            fresh.writestr(info.filename, old.read(info))
        fresh.writestr('DUAL_RUNTIME_MANIFEST.json', json.dumps(dict(runtime_sha256=fingerprint,
                      execution_format='dual_resume_v1', notebook=target.name, workers=2,
                      source_data_unchanged=True, resume_notebook=resume_target.name,
                      source_alignment_sha256=hashlib.sha256(source_alignment.encode()).hexdigest()), indent=2))
        fresh.writestr('HOW_TO_RUN.txt', 'Import SimpleDyG_Kaggle_2GPU_Resume.ipynb, attach this source bundle, choose GPU T4 x2.\nFirst session RESUME=False. Later attach latest resume_snapshot.zip or full Output and set RESUME=True.\nKeep runtime/config/data fixed. Completed best.pt files remain in full Output.\n')
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
    print(target, target.stat().st_size, 'bytes')
    print(resume_target, resume_target.stat().st_size, 'bytes')
    print(archive, archive.stat().st_size, 'bytes')


if __name__ == '__main__':
    build()
