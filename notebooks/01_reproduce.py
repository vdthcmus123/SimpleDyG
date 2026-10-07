# ============================================================================
# SimpleDyG Reproduction Notebook for Kaggle
# ============================================================================
# Notebook 1/3: Reproduce kết quả trên UCI và Hepth datasets
#
# Hướng dẫn sử dụng trên Kaggle:
# 1. Tạo Notebook mới, bật GPU (Settings > Accelerator > GPU T4 x2)
# 2. Copy từng cell (được đánh dấu bằng # %% [markdown] hoặc # %%)
#    vào từng cell trong Kaggle notebook
# 3. Chạy tuần tự từ trên xuống dưới
# ============================================================================

# %% [markdown]
# # 🔬 SimpleDyG Reproduction — Notebook 1: Reproduce Results
# 
# Paper: **"On the Feasibility of Simple Transformer for Dynamic Graph Modeling"** (WWW 2024)
# 
# Notebook này reproduce kết quả link prediction trên **UCI** và **Hepth** datasets.

# %% Cell 1: Setup — Clone repo và cài dependencies
import subprocess, os

# Clone repository
if not os.path.exists('/kaggle/working/SimpleDyG'):
    subprocess.run(['git', 'clone', 'https://github.com/YuxiaWu/SimpleDyG.git', '/kaggle/working/SimpleDyG'], check=True)
    print("✅ Cloned SimpleDyG repository")
else:
    print("✅ Repository already exists")

# Cài dependencies (không cần apex, spacy)
subprocess.run(['pip', 'install', '-q', 
    'transformers>=4.24.0', 'tokenizers', 'pandas', 'numpy', 
    'scikit-learn', 'scipy', 'tensorboard', 'tqdm'], check=True)
print("✅ Dependencies installed")

# Verify GPU
import torch
print(f"✅ PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# %% Cell 2: Kiểm tra dữ liệu đã preprocessed
import os

BASE_DIR = '/kaggle/working/SimpleDyG'
os.chdir(BASE_DIR)

datasets_info = {
    'UCI_13': {'timestamp': '12', 'dir': 'resources/UCI_13/12'},
    'hepth':  {'timestamp': '11', 'dir': 'resources/hepth/11'},
}

print("=" * 60)
print("📊 KIỂM TRA DỮ LIỆU PREPROCESSED")
print("=" * 60)

for ds_name, info in datasets_info.items():
    print(f"\n📁 Dataset: {ds_name}")
    ds_dir = info['dir']
    for f in ['train.link_prediction', 'val.link_prediction', 'val_gt.link_prediction',
              'test.link_prediction', 'test_gt.link_prediction']:
        fpath = os.path.join(ds_dir, f)
        if os.path.exists(fpath):
            with open(fpath) as fp:
                lines = fp.readlines()
            print(f"  ✅ {f}: {len(lines)} samples")
        else:
            print(f"  ❌ {f}: MISSING!")

    # Check vocab
    vocab_path = os.path.join('vocabs', ds_name, info['timestamp'], 'vocab.json')
    if os.path.exists(vocab_path):
        import json
        with open(vocab_path) as fp:
            vocab = json.load(fp)
        print(f"  ✅ vocab.json: {len(vocab)} tokens")

# Xem vài dòng dữ liệu mẫu
print("\n" + "=" * 60)
print("📝 DỮ LIỆU MẪU (UCI_13 train, 3 dòng đầu)")
print("=" * 60)
with open('resources/UCI_13/12/train.link_prediction') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        # In ngắn gọn
        tokens = line.strip().split()
        if len(tokens) > 30:
            print(f"  Line {i}: {' '.join(tokens[:15])} ... {' '.join(tokens[-10:])}  (total {len(tokens)} tokens)")
        else:
            print(f"  Line {i}: {line.strip()}")


# %% Cell 3: Hàm helper để chạy training
import subprocess
import os
import time
import json

def run_training(dataset, timestamp, n_layer, n_head, n_embed, 
                 batch_size=32, lr=0.0001, seed=0, num_epochs=100,
                 patience=10, node_feat_file=None):
    """Chạy training SimpleDyG cho một cấu hình cụ thể."""
    
    base_dir = '/kaggle/working/SimpleDyG'
    os.chdir(base_dir)
    
    output_dir = f"output/{dataset}/{timestamp}/{n_layer}_{n_head}_{n_embed}_{batch_size}_{lr}_{seed}/gpt2"
    
    cmd = [
        'python', 'main.py',
        '--dataset', dataset,
        '--output_dir', output_dir,
        '--model_type', 'gpt2',
        '--model_name_or_path', 'gpt2',
        '--train_data_file', f'./resources/{dataset}/{timestamp}/train.link_prediction',
        '--do_train',
        '--eval_data_file', f'./resources/{dataset}/{timestamp}/val.link_prediction',
        '--eval_data_gt_file', f'./resources/{dataset}/{timestamp}/val_gt.link_prediction',
        '--test_data_file', f'./resources/{dataset}/{timestamp}/test.link_prediction',
        '--test_data_gt_file', f'./resources/{dataset}/{timestamp}/test_gt.link_prediction',
        '--save_steps', '250',
        '--logging_steps', '500',
        '--per_gpu_train_batch_size', str(batch_size),
        '--num_train_epochs', str(num_epochs),
        '--block_size', '512',
        '--eval_all_checkpoints',
        '--timestamp', str(timestamp),
        '--patience', str(patience),
        '--n_layer', str(n_layer),
        '--n_head', str(n_head),
        '--n_embed', str(n_embed),
        '--learning_rate', str(lr),
        '--seed', str(seed),
        '--run_seed',
    ]
    
    # Hepth cần node features
    if node_feat_file:
        cmd.extend(['--node_feat_file', node_feat_file])
    
    config_str = f"{dataset} | layers={n_layer}, heads={n_head}, embed={n_embed}, seed={seed}"
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING: {config_str}")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    
    start_time = time.time()
    
    # Chạy training, stream output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=base_dir
    )
    
    for line in process.stdout:
        # Chỉ in các dòng quan trọng
        line = line.strip()
        if any(k in line for k in ['Epoch', 'val_loss', 'NDCG', 'jaccard', 'Early', 'cost time', 'ERROR', 'Traceback']):
            print(f"  {line}")
    
    process.wait()
    elapsed = (time.time() - start_time) / 60
    
    if process.returncode == 0:
        print(f"✅ Training hoàn thành trong {elapsed:.1f} phút")
    else:
        print(f"❌ Training thất bại (return code: {process.returncode})")
    
    return output_dir, process.returncode

print("✅ Hàm run_training đã sẵn sàng")


# %% Cell 4: Hàm thu thập kết quả
import pandas as pd
import glob

def collect_results(dataset):
    """Thu thập tất cả kết quả từ các file CSV."""
    base_dir = '/kaggle/working/SimpleDyG'
    
    # Tìm trong topk_scores_seed_jac
    result_file = os.path.join(base_dir, 'topk_scores_seed_jac', f'{dataset}_SimpleDyG.csv')
    
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        print(f"\n📊 Kết quả Test cho {dataset}:")
        print(df.to_string(index=False))
        return df
    
    # Fallback: tìm trong output directories
    pattern = os.path.join(base_dir, f'output/{dataset}/**/results_seed_jac/test_score/test_results_epoch.csv')
    files = glob.glob(pattern, recursive=True)
    
    if files:
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"\n📊 Kết quả Test cho {dataset}:")
        print(df.to_string(index=False))
        return df
    
    print(f"⚠️ Chưa có kết quả cho {dataset}")
    return None


def summarize_results(df, dataset):
    """Tính mean ± std từ nhiều seeds."""
    if df is None:
        return
    
    print(f"\n{'='*60}")
    print(f"📈 TỔNG KẾT: {dataset}")
    print(f"{'='*60}")
    
    numeric_cols = [c for c in df.columns if 'NDCG' in c or 'jaccard' in c]
    
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        print(f"  {col}: {mean:.4f} ± {std:.4f}")
    
    print(f"  Số seeds: {len(df)}")

print("✅ Hàm collect_results đã sẵn sàng")


# %% Cell 5: 🚀 Chạy Reproduce trên UCI_13 (Dataset nhỏ, chạy nhanh)
# Chạy 3 seeds (0, 1, 2) để verify - đủ để reproduce
# Hyperparams từ paper: n_layer=6, n_head=8, n_embed=768

uci_results = []
for seed in [0, 1, 2]:
    output_dir, ret_code = run_training(
        dataset='UCI_13',
        timestamp='12',
        n_layer=6,
        n_head=8,
        n_embed=768,
        batch_size=32,
        lr=0.0001,
        seed=seed,
        num_epochs=100,
        patience=10,
    )
    if ret_code == 0:
        uci_results.append({'seed': seed, 'output_dir': output_dir})

print(f"\n✅ UCI_13: {len(uci_results)}/{3} runs hoàn thành")


# %% Cell 6: Thu thập kết quả UCI
uci_df = collect_results('UCI_13')
summarize_results(uci_df, 'UCI_13')


# %% Cell 7: 🚀 Chạy Reproduce trên Hepth
# Hyperparams từ paper: n_layer=12, n_head=2, n_embed=256
# Lưu ý: Hepth dùng node features

hepth_results = []
for seed in [42, 0, 1]:
    output_dir, ret_code = run_training(
        dataset='hepth',
        timestamp='11',
        n_layer=12,
        n_head=2,
        n_embed=256,
        batch_size=32,
        lr=0.0001,
        seed=seed,
        num_epochs=100,
        patience=10,
        node_feat_file='./resources/hepth/node_features.npy',
    )
    if ret_code == 0:
        hepth_results.append({'seed': seed, 'output_dir': output_dir})

print(f"\n✅ Hepth: {len(hepth_results)}/{3} runs hoàn thành")


# %% Cell 8: Thu thập kết quả Hepth
hepth_df = collect_results('hepth')
summarize_results(hepth_df, 'hepth')


# %% Cell 9: 📊 Tổng hợp & so sánh với Paper
print("=" * 70)
print("📊 TỔNG HỢP KẾT QUẢ REPRODUCE vs. PAPER")
print("=" * 70)

# Bảng so sánh
print("""
╔══════════════╦════════════════════════════╦════════════════════════════╗
║              ║        UCI_13              ║        Hepth               ║
║  Metric      ╠══════════════╦═════════════╬══════════════╦═════════════╣
║              ║  Paper       ║  Ours       ║  Paper       ║  Ours       ║
╠══════════════╬══════════════╬═════════════╬══════════════╬═════════════╣
║  NDCG@5      ║  (xem paper) ║  (xem trên) ║  (xem paper) ║  (xem trên) ║
║  Jaccard     ║  (xem paper) ║  (xem trên) ║  (xem paper) ║  (xem trên) ║
╚══════════════╩══════════════╩═════════════╩══════════════╩═════════════╝

📌 Điền kết quả từ paper (Table 2-3) vào cột "Paper" để so sánh.
📌 Kết quả reproduce nên nằm trong khoảng ±2% so với paper.
""")

# Lưu tất cả kết quả
all_results = {}
if uci_df is not None:
    all_results['UCI_13'] = uci_df
if 'hepth_df' in dir() and hepth_df is not None:
    all_results['hepth'] = hepth_df

# Save to CSV
for ds_name, df in all_results.items():
    save_path = f'/kaggle/working/reproduce_results_{ds_name}.csv'
    df.to_csv(save_path, index=False)
    print(f"💾 Saved: {save_path}")


# %% Cell 10: 📥 Download kết quả
# Kết quả được lưu tại /kaggle/working/reproduce_results_*.csv
# Kaggle sẽ tự động hiện nút download ở tab Output

import shutil
# Copy toàn bộ kết quả vào một thư mục gọn
results_dir = '/kaggle/working/all_results'
os.makedirs(results_dir, exist_ok=True)

for ds_name in ['UCI_13', 'hepth']:
    src = f'/kaggle/working/reproduce_results_{ds_name}.csv'
    if os.path.exists(src):
        shutil.copy(src, results_dir)

# Cũng copy log files
for ds_name in ['UCI_13', 'hepth']:
    log_pattern = os.path.join(BASE_DIR, 'all_logs', ds_name, '**', 'train.log')
    for log_file in glob.glob(log_pattern, recursive=True):
        dst_name = f"train_log_{ds_name}_{os.path.basename(os.path.dirname(log_file))}.log"
        shutil.copy(log_file, os.path.join(results_dir, dst_name))

print(f"📁 Tất cả kết quả đã lưu tại: {results_dir}")
print("   Có thể download từ tab Output của Kaggle")
