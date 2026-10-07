"""Build a continuation notebook while retaining the original core byte-for-byte."""
import ast
import copy
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = nbf.read(ROOT / 'notebooks/SimpleDyG_Kaggle_Full_Pipeline.ipynb', as_version=4)
    notebook = copy.deepcopy(source)
    notebook.cells[0].source = '''# SimpleDyG: tiếp tục sau khi Kaggle hết phiên

Add Input `SimpleDyG_Continue_Input.zip` đã khôi phục hoặc Output đầy đủ của phiên trước,
đồng thời giữ input repo/dữ liệu SimpleDyG và DyGLib cũ. Bật GPU rồi Run All.

Notebook giữ nguyên core, dữ liệu và toàn bộ cấu hình thí nghiệm cũ. Cell khôi phục
kiểm tra fingerprint, lập kế hoạch 100 lượt và bỏ qua `result.json` hoàn thành.
Các bộ lọc SESSION_* chỉ chia lịch chạy; không thay cấu hình hay run ID.

Mặc định chạy tối đa 2 lượt TGAT còn thiếu trên UCI. Sau phiên này, lưu Output mới
và dùng chính Output mới làm Input của phiên tiếp theo. Khi đủ TGAT, chọn GraphMixer
với MAX_NEW_RUNS=1; sau đó chuyển sang ML_10M_13.

`best.pt` cũ chỉ là trọng số tốt nhất, chưa có optimizer/scheduler/RNG cho resume epoch.
Lượt đang dở phải chạy lại. Giới hạn giờ chỉ kiểm tra trước khi bắt đầu một lượt;
một lượt dài vẫn có thể vượt thời gian phiên.

Gói Input nhỏ giữ kết quả và file dự đoán đã cứu được, không chứa checkpoint.
Giữ ZIP khôi phục lớn để lưu riêng 16 checkpoint còn nguyên. 19 checkpoint và
18 file test prediction không có trong phần ZIP cũ ghi đầy đủ; kết quả metric
của 35 lượt vẫn có trong manifest và được giữ nguyên.
'''
    notebook.cells[1].source = '''# Setup/source vẫn dùng input cũ. Cấu hình training thực được đọc từ Output cũ.
PROFILE = "paper"
OFFLINE = False
INSTALL_DEPENDENCIES = True
AUTHOR_ROOT_OVERRIDE = ""
DYGLIB_ROOT_OVERRIDE = ""
INPUT_BUNDLE_OVERRIDE = ""  # bundle SOURCE cũ, không phải ZIP kết quả

# Để trống nếu /kaggle/input chỉ có một Output cũ phù hợp.
PREVIOUS_OUTPUT_PATH = ""   # ZIP kết quả hoặc thư mục chứa config.json và runs/

# Chia lịch mà không đổi config/fingerprint của 100 lượt gốc.
SESSION_DATASETS = ["UCI_13"]
SESSION_MODELS = ["TGAT"]    # sau đó ["GraphMixer"] hoặc ["SimpleDyG"]
SESSION_SEEDS = None         # None = các seed còn thiếu trong lịch gốc
MAX_NEW_RUNS = 2             # GraphMixer/ML-10M nên kiểm tra 1 lượt đầu
MAX_SESSION_HOURS = 10.0     # chỉ ngăn bắt đầu lượt MỚI khi đã dùng hết thời gian
'''
    notebook.cells[8].source = '## 4. Khôi phục Output, kiểm tra cấu hình và số lượt còn thiếu'
    helpers = (ROOT / 'tools/kaggle_continue.py').read_text(encoding='utf-8')
    notebook.cells[9].source = helpers + '''

PREVIOUS_SOURCE = Path(PREVIOUS_OUTPUT_PATH) if PREVIOUS_OUTPUT_PATH else find_previous_output(INPUT)
OUTPUT = WORK / "simpledyg_paper_author_compat"
restore_previous_output(PREVIOUS_SOURCE, OUTPUT)
CFG, CONTINUE_DATASETS, CONTINUE_PLAN = verify_previous_output(globals(), AUTHOR_ROOT, DYGLIB_ROOT, OUTPUT)
if CFG["profile"] != "paper":
    raise RuntimeError("Input không phải kết quả profile paper của phiên cần tiếp tục.")
plan_table = pd.DataFrame(CONTINUE_PLAN)
display(plan_table.groupby(["dataset", "model", "variant", "status"]).size().rename("runs").reset_index())
print("Toàn bộ kế hoạch:", len(CONTINUE_PLAN), "| Hoàn thành:", sum(r["status"] == "complete" for r in CONTINUE_PLAN))
print("Cấu hình training giữ nguyên; bộ lọc phiên:", SESSION_DATASETS, SESSION_MODELS, SESSION_SEEDS)
'''
    notebook.cells[10].source = '## 5. Chạy nhóm lượt còn thiếu và lưu ZIP bằng ghi atomic'
    notebook.cells[11].source = '''results, summary, result_archive = continue_previous_output(
    globals(), AUTHOR_ROOT, DYGLIB_ROOT, CFG, CONTINUE_DATASETS, CONTINUE_PLAN, OUTPUT,
    session_datasets=SESSION_DATASETS, session_models=SESSION_MODELS,
    session_seeds=SESSION_SEEDS, max_new_runs=MAX_NEW_RUNS,
    max_session_hours=MAX_SESSION_HOURS)
display(summary)
'''
    notebook.cells[14].source = '''## Sau mỗi phiên

Lưu Version/Output và ZIP kết quả mới. Phiên sau Add Input **Output mới nhất**,
bỏ Input kết quả của các phiên cũ để tránh tự chọn nhầm snapshot.

Tiếp tục UCI/TGAT với cấu hình mặc định đến đủ 10 seed. Sau đó chọn
`SESSION_MODELS=["GraphMixer"]`, `MAX_NEW_RUNS=1`. Cuối cùng chọn
`SESSION_DATASETS=["ML_10M_13"]`, `SESSION_MODELS=["SimpleDyG"]`, rồi các baseline.
Muốn chọn seed cụ thể dùng `SESSION_SEEDS=[5]`; không sửa CFG hoặc danh sách seed gốc.

Giới hạn phiên không dừng được một lượt đang chạy. Notebook giữ ZIP đã đóng hoàn chỉnh
cho đến khi ZIP mới ghi xong; nếu bị ngắt, ưu tiên dùng toàn bộ thư mục Output vì
`result.json` được ghi ngay khi mỗi lượt hoàn thành.

Kết quả metric của 35 lượt khôi phục là số thực đo giữ nguyên từ manifest. Các trọng số
và prediction bị thiếu được ghi trong `recovery_inventory.json`; muốn đánh giá lại
các lượt đó cần lấy file từ Output gốc trên Kaggle hoặc chạy lại riêng và ghi rõ.
'''
    assert notebook.cells[5].source == source.cells[5].source
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            cell.execution_count = None
            cell.outputs = []
    nbf.validate(notebook)
    destination = ROOT / 'notebooks/SimpleDyG_Kaggle_Continue.ipynb'
    nbf.write(notebook, destination)
    print(destination)


if __name__ == '__main__':
    build()
