# ============================================================================
# SimpleDyG Ablation Study — Model Architecture
# ============================================================================
# Notebook 3/3: Kiểm tra ảnh hưởng của kiến trúc model
#
# Thí nghiệm:
#   1. Ablation Depth   — Thay đổi n_layer (2, 4, 6, 8, 12)
#   2. Ablation Width   — Thay đổi n_embed (128, 256, 512, 768)
#   3. Ablation Heads   — Thay đổi n_head (1, 2, 4, 8)
#   4. Scaling Analysis — Tổng hợp xu hướng scaling
#   5. Cross-dataset    — So sánh best config trên UCI vs Hepth
#
# Hướng dẫn: Chạy notebook 01_reproduce.py trước để verify pipeline
# ============================================================================

# %% [markdown]
# # 🏗️ SimpleDyG Ablation Study — Model Architecture
#
# **Câu hỏi nghiên cứu:** Kiến trúc Transformer ảnh hưởng thế nào đến
# khả năng modeling dynamic graphs?
#
# - Deeper models (nhiều layers hơn) có luôn tốt hơn?
# - Wider models (embedding lớn hơn) giúp gì?
# - Multi-head attention có thực sự cần thiết?
# - Trade-off giữa hiệu suất và chi phí tính toán?

# %% Cell 1: Setup
import subprocess, os, sys, time, json, re, glob, shutil
import pandas as pd
import numpy as np

BASE_DIR = '/kaggle/working/SimpleDyG'

if not os.path.exists(BASE_DIR):
    subprocess.run(['git', 'clone', 'https://github.com/YuxiaWu/SimpleDyG.git', BASE_DIR], check=True)
    subprocess.run(['pip', 'install', '-q',
        'transformers>=4.24.0', 'tokenizers', 'pandas', 'numpy',
        'scikit-learn', 'scipy', 'tensorboard', 'tqdm'], check=True)

os.chdir(BASE_DIR)

import torch
print(f"✅ Setup hoàn thành. GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# %% Cell 2: Hàm training (tái sử dụng, có thêm tracking thời gian & memory)
def run_training(dataset, timestamp, n_layer, n_head, n_embed,
                 batch_size=32, lr=0.0001, seed=0, num_epochs=100,
                 patience=10, node_feat_file=None):
    """Chạy training SimpleDyG cho một cấu hình cụ thể.
    
    Returns:
        tuple: (output_dir, return_code, elapsed_minutes, peak_gpu_mb)
    """
    os.chdir(BASE_DIR)
    
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
    
    if node_feat_file:
        cmd.extend(['--node_feat_file', node_feat_file])
    
    config_str = f"{dataset} | L={n_layer}, H={n_head}, D={n_embed}, seed={seed}"
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING: {config_str}")
    print(f"{'='*70}")
    
    # Reset GPU memory counter
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=BASE_DIR
    )
    
    for line in process.stdout:
        line = line.strip()
        if any(k in line for k in ['Epoch', 'val_loss', 'NDCG', 'jaccard', 'Early', 'cost time', 'ERROR', 'Traceback']):
            print(f"  {line}")
    
    process.wait()
    elapsed = (time.time() - start_time) / 60
    
    # Peak GPU memory
    peak_gpu_mb = 0
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6
    
    if process.returncode == 0:
        print(f"✅ Hoàn thành trong {elapsed:.1f} phút | Peak GPU: {peak_gpu_mb:.0f} MB")
    else:
        print(f"❌ Thất bại (code: {process.returncode})")
    
    return output_dir, process.returncode, elapsed, peak_gpu_mb


# %% Cell 3: Hàm thu thập kết quả
def collect_results(dataset, n_layer=None, n_head=None, n_embed=None):
    """Thu thập kết quả, có thể filter theo config cụ thể."""
    # Tìm trong topk_scores_seed_jac
    result_file = os.path.join(BASE_DIR, 'topk_scores_seed_jac', f'{dataset}_SimpleDyG.csv')
    
    if os.path.exists(result_file):
        return pd.read_csv(result_file)
    
    # Fallback: tìm trong output directories
    if n_layer and n_head and n_embed:
        pattern = os.path.join(
            BASE_DIR,
            f'output/{dataset}/**/{n_layer}_{n_head}_{n_embed}_*/gpt2/**/test_results_epoch.csv'
        )
    else:
        pattern = os.path.join(
            BASE_DIR,
            f'output/{dataset}/**/test_results_epoch.csv'
        )
    
    files = glob.glob(pattern, recursive=True)
    if files:
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return None


def compute_param_count(n_layer, n_head, n_embed, vocab_size=5000):
    """Ước tính số parameters của GPT2 model với config cho trước.
    
    GPT2 architecture:
    - Token embedding: vocab_size * n_embed
    - Position embedding: max_position * n_embed  (max_position=1024)
    - Each transformer block:
        - Layer norm 1: 2 * n_embed
        - Attention (Q,K,V,O): 4 * n_embed^2 + 4 * n_embed
        - Layer norm 2: 2 * n_embed
        - MLP (fc1 + fc2): 8 * n_embed^2 + 5 * n_embed
    - Final layer norm: 2 * n_embed
    - LM head (tied or not): vocab_size * n_embed
    """
    max_position = 1024
    
    # Embeddings
    token_emb = vocab_size * n_embed
    pos_emb = max_position * n_embed
    
    # Per-layer
    attn = 4 * n_embed * n_embed + 4 * n_embed  # Q, K, V, O projections + biases
    mlp = 8 * n_embed * n_embed + 5 * n_embed    # fc1 (4x), fc2 (4x back) + biases
    ln = 4 * n_embed  # 2 layer norms per layer
    per_layer = attn + mlp + ln
    
    # Total
    final_ln = 2 * n_embed
    lm_head = vocab_size * n_embed  # output projection
    
    total = token_emb + pos_emb + (n_layer * per_layer) + final_ln + lm_head
    return total


print("✅ Hàm run_training, collect_results, compute_param_count đã sẵn sàng")


# %% Cell 4: Định nghĩa tất cả cấu hình Ablation
# ==========================================================================
# Tất cả cấu hình ablation, nhóm theo experiment
# ==========================================================================

# ---- Baseline (paper config cho UCI_13) ----
BASELINE = {'n_layer': 6, 'n_head': 8, 'n_embed': 768}

# ---- Ablation 1: Depth (thay đổi n_layer, giữ n_head=8, n_embed=768) ----
DEPTH_CONFIGS = [
    {'n_layer': 2,  'n_head': 8, 'n_embed': 768, 'label': '2 layers'},
    {'n_layer': 4,  'n_head': 8, 'n_embed': 768, 'label': '4 layers'},
    {'n_layer': 6,  'n_head': 8, 'n_embed': 768, 'label': '6 layers (baseline)'},
    {'n_layer': 8,  'n_head': 8, 'n_embed': 768, 'label': '8 layers'},
    {'n_layer': 12, 'n_head': 8, 'n_embed': 768, 'label': '12 layers'},
]

# ---- Ablation 2: Width (thay đổi n_embed, giữ n_layer=6) ----
# Lưu ý: n_embed phải chia hết cho n_head
WIDTH_CONFIGS = [
    {'n_layer': 6, 'n_head': 2, 'n_embed': 128, 'label': 'D=128'},
    {'n_layer': 6, 'n_head': 4, 'n_embed': 256, 'label': 'D=256'},
    {'n_layer': 6, 'n_head': 8, 'n_embed': 512, 'label': 'D=512'},
    {'n_layer': 6, 'n_head': 8, 'n_embed': 768, 'label': 'D=768 (baseline)'},
]

# ---- Ablation 3: Attention Heads (thay đổi n_head, giữ n_layer=6, n_embed=768) ----
HEAD_CONFIGS = [
    {'n_layer': 6, 'n_head': 1,  'n_embed': 768, 'label': '1 head'},
    {'n_layer': 6, 'n_head': 2,  'n_embed': 768, 'label': '2 heads'},
    {'n_layer': 6, 'n_head': 4,  'n_embed': 768, 'label': '4 heads'},
    {'n_layer': 6, 'n_head': 8,  'n_embed': 768, 'label': '8 heads (baseline)'},
    {'n_layer': 6, 'n_head': 12, 'n_embed': 768, 'label': '12 heads'},
]

# ---- Ablation 4: Compact vs Large models ----
SCALE_CONFIGS = [
    {'n_layer': 2, 'n_head': 2, 'n_embed': 128, 'label': 'Tiny (2L-2H-128D)'},
    {'n_layer': 4, 'n_head': 4, 'n_embed': 256, 'label': 'Small (4L-4H-256D)'},
    {'n_layer': 6, 'n_head': 8, 'n_embed': 768, 'label': 'Base (6L-8H-768D)'},
    {'n_layer': 12, 'n_head': 12, 'n_embed': 768, 'label': 'Large (12L-12H-768D)'},
]

# Hiện bảng tóm tắt
print("=" * 70)
print("📋 CÁC CẤU HÌNH ABLATION")
print("=" * 70)

all_configs = {
    'Depth': DEPTH_CONFIGS,
    'Width': WIDTH_CONFIGS,
    'Heads': HEAD_CONFIGS,
    'Scale': SCALE_CONFIGS,
}

for group_name, configs in all_configs.items():
    print(f"\n🔹 {group_name}:")
    for cfg in configs:
        params = compute_param_count(cfg['n_layer'], cfg['n_head'], cfg['n_embed'])
        print(f"   {cfg['label']:30s} → {params/1e6:.1f}M params")


# %% Cell 5: 🚀 Ablation 1 — DEPTH (số layers)
print("\n" + "=" * 70)
print("🏗️ ABLATION 1: MODEL DEPTH (số layers)")
print("=" * 70)
print("Giả thuyết: Deeper models có thể capture longer-range temporal")
print("dependencies, nhưng có thể bị overfitting trên small graphs.")
print("=" * 70)

depth_results = []
for cfg in DEPTH_CONFIGS:
    for seed in [0, 1, 2]:
        output_dir, ret_code, elapsed, peak_gpu = run_training(
            dataset='UCI_13', timestamp='12',
            n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embed=cfg['n_embed'],
            seed=seed, num_epochs=100, patience=10,
        )
        depth_results.append({
            'label': cfg['label'],
            'n_layer': cfg['n_layer'],
            'n_head': cfg['n_head'],
            'n_embed': cfg['n_embed'],
            'seed': seed,
            'elapsed_min': elapsed,
            'peak_gpu_mb': peak_gpu,
            'return_code': ret_code,
        })

print(f"\n✅ Depth ablation: {sum(1 for r in depth_results if r['return_code']==0)}/{len(depth_results)} runs hoàn thành")


# %% Cell 6: 🚀 Ablation 2 — WIDTH (embedding dimension)
print("\n" + "=" * 70)
print("🏗️ ABLATION 2: MODEL WIDTH (embedding dimension)")
print("=" * 70)
print("Giả thuyết: Larger embeddings giúp encode richer node/edge")
print("representations, nhưng small graphs có thể không cần D lớn.")
print("=" * 70)

width_results = []
for cfg in WIDTH_CONFIGS:
    for seed in [0, 1, 2]:
        output_dir, ret_code, elapsed, peak_gpu = run_training(
            dataset='UCI_13', timestamp='12',
            n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embed=cfg['n_embed'],
            seed=seed, num_epochs=100, patience=10,
        )
        width_results.append({
            'label': cfg['label'],
            'n_layer': cfg['n_layer'],
            'n_head': cfg['n_head'],
            'n_embed': cfg['n_embed'],
            'seed': seed,
            'elapsed_min': elapsed,
            'peak_gpu_mb': peak_gpu,
            'return_code': ret_code,
        })

print(f"\n✅ Width ablation: {sum(1 for r in width_results if r['return_code']==0)}/{len(width_results)} runs hoàn thành")


# %% Cell 7: 🚀 Ablation 3 — ATTENTION HEADS
print("\n" + "=" * 70)
print("🏗️ ABLATION 3: ATTENTION HEADS")
print("=" * 70)
print("Giả thuyết: Multi-head attention cho phép model attend")
print("đến nhiều loại quan hệ (temporal, structural) đồng thời.")
print("=" * 70)

head_results = []
for cfg in HEAD_CONFIGS:
    for seed in [0, 1, 2]:
        output_dir, ret_code, elapsed, peak_gpu = run_training(
            dataset='UCI_13', timestamp='12',
            n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embed=cfg['n_embed'],
            seed=seed, num_epochs=100, patience=10,
        )
        head_results.append({
            'label': cfg['label'],
            'n_layer': cfg['n_layer'],
            'n_head': cfg['n_head'],
            'n_embed': cfg['n_embed'],
            'seed': seed,
            'elapsed_min': elapsed,
            'peak_gpu_mb': peak_gpu,
            'return_code': ret_code,
        })

print(f"\n✅ Heads ablation: {sum(1 for r in head_results if r['return_code']==0)}/{len(head_results)} runs hoàn thành")


# %% Cell 8: 🚀 Ablation 4 — SCALING (Tiny → Large)
print("\n" + "=" * 70)
print("🏗️ ABLATION 4: MODEL SCALING (Tiny → Large)")
print("=" * 70)
print("So sánh hiệu suất khi scale toàn bộ model.")
print("Trade-off: accuracy vs compute cost.")
print("=" * 70)

scale_results = []
for cfg in SCALE_CONFIGS:
    for seed in [0, 1, 2]:
        output_dir, ret_code, elapsed, peak_gpu = run_training(
            dataset='UCI_13', timestamp='12',
            n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embed=cfg['n_embed'],
            seed=seed, num_epochs=100, patience=10,
        )
        scale_results.append({
            'label': cfg['label'],
            'n_layer': cfg['n_layer'],
            'n_head': cfg['n_head'],
            'n_embed': cfg['n_embed'],
            'seed': seed,
            'elapsed_min': elapsed,
            'peak_gpu_mb': peak_gpu,
            'return_code': ret_code,
            'params_M': compute_param_count(cfg['n_layer'], cfg['n_head'], cfg['n_embed']) / 1e6,
        })

print(f"\n✅ Scaling ablation: {sum(1 for r in scale_results if r['return_code']==0)}/{len(scale_results)} runs hoàn thành")


# %% Cell 9: 🚀 Ablation 5 — Cross-dataset (Best config trên Hepth)
# Kiểm tra xem config tốt nhất trên UCI có transfer sang Hepth không
print("\n" + "=" * 70)
print("🏗️ ABLATION 5: CROSS-DATASET GENERALIZATION")
print("=" * 70)
print("Chạy subset configs trên Hepth để kiểm tra generalization.")
print("Paper config cho Hepth: n_layer=12, n_head=2, n_embed=256")
print("=" * 70)

# Configs thú vị để test trên Hepth
CROSS_DATASET_CONFIGS = [
    # Paper config for Hepth (baseline)
    {'n_layer': 12, 'n_head': 2, 'n_embed': 256, 'label': 'Hepth paper (12L-2H-256D)'},
    # Paper config for UCI (xem có transfer không)
    {'n_layer': 6,  'n_head': 8, 'n_embed': 768, 'label': 'UCI paper (6L-8H-768D)'},
    # Compact model
    {'n_layer': 4,  'n_head': 4, 'n_embed': 256, 'label': 'Small (4L-4H-256D)'},
    # Large model
    {'n_layer': 12, 'n_head': 8, 'n_embed': 768, 'label': 'Large (12L-8H-768D)'},
]

cross_results = []
for cfg in CROSS_DATASET_CONFIGS:
    for seed in [0, 1, 2]:
        output_dir, ret_code, elapsed, peak_gpu = run_training(
            dataset='hepth', timestamp='11',
            n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embed=cfg['n_embed'],
            seed=seed, num_epochs=100, patience=10,
            node_feat_file='./resources/hepth/node_features.npy',
        )
        cross_results.append({
            'label': cfg['label'],
            'n_layer': cfg['n_layer'],
            'n_head': cfg['n_head'],
            'n_embed': cfg['n_embed'],
            'seed': seed,
            'elapsed_min': elapsed,
            'peak_gpu_mb': peak_gpu,
            'return_code': ret_code,
        })

print(f"\n✅ Cross-dataset ablation: {sum(1 for r in cross_results if r['return_code']==0)}/{len(cross_results)} runs hoàn thành")


# %% Cell 10: 📊 Thu thập & tổng hợp tất cả kết quả
def summarize_ablation_group(results_list, group_name):
    """Tổng hợp kết quả cho một nhóm ablation.
    
    Kết hợp training metadata (time, GPU) với test metrics (NDCG, Jaccard)
    từ output files.
    """
    print(f"\n{'='*70}")
    print(f"📊 {group_name}")
    print(f"{'='*70}")
    
    # Group by config label
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results_list:
        grouped[r['label']].append(r)
    
    summary_rows = []
    for label, runs in grouped.items():
        successful = [r for r in runs if r['return_code'] == 0]
        if not successful:
            print(f"  ⚠️ {label}: Không có run thành công")
            continue
        
        # Training stats
        avg_time = np.mean([r['elapsed_min'] for r in successful])
        avg_gpu = np.mean([r['peak_gpu_mb'] for r in successful])
        
        # Tìm test metrics từ output
        sample = successful[0]
        
        # Scan for result files matching this config
        config_pattern = f"{sample['n_layer']}_{sample['n_head']}_{sample['n_embed']}_"
        result_pattern = os.path.join(
            BASE_DIR, 'output', '**', config_pattern + '*',
            'gpt2', '**', 'test_results_epoch.csv'
        )
        result_files = glob.glob(result_pattern, recursive=True)
        
        row = {
            'Label': label,
            'n_layer': sample['n_layer'],
            'n_head': sample['n_head'],
            'n_embed': sample['n_embed'],
            'n_seeds': len(successful),
            'Avg Time (min)': round(avg_time, 1),
            'Avg GPU (MB)': round(avg_gpu, 0),
            'Params (M)': round(compute_param_count(
                sample['n_layer'], sample['n_head'], sample['n_embed']
            ) / 1e6, 1),
        }
        
        if result_files:
            try:
                dfs = [pd.read_csv(f) for f in result_files]
                df = pd.concat(dfs, ignore_index=True)
                
                for col in df.columns:
                    if 'NDCG' in col or 'jaccard' in col:
                        row[f'{col}_mean'] = round(df[col].mean(), 4)
                        row[f'{col}_std'] = round(df[col].std(), 4)
                        
            except Exception as e:
                print(f"  ⚠️ Lỗi đọc kết quả {label}: {e}")
        
        summary_rows.append(row)
        
        # Print
        metrics_str = ""
        for k, v in row.items():
            if '_mean' in k:
                metric_name = k.replace('_mean', '')
                std_val = row.get(f'{metric_name}_std', 0)
                metrics_str += f" | {metric_name}: {v:.4f}±{std_val:.4f}"
        
        print(f"  {label:35s} → {row['Params (M)']:.1f}M params, "
              f"{row['Avg Time (min)']:.1f} min{metrics_str}")
    
    return pd.DataFrame(summary_rows)


# ---- Tổng hợp ----
print("\n" + "🔬" * 35)
print("📊 TỔNG HỢP KẾT QUẢ ABLATION ARCHITECTURE")
print("🔬" * 35)

all_summaries = {}

if depth_results:
    all_summaries['Depth'] = summarize_ablation_group(depth_results, "ABLATION 1: DEPTH")

if width_results:
    all_summaries['Width'] = summarize_ablation_group(width_results, "ABLATION 2: WIDTH")

if head_results:
    all_summaries['Heads'] = summarize_ablation_group(head_results, "ABLATION 3: ATTENTION HEADS")

if scale_results:
    all_summaries['Scale'] = summarize_ablation_group(scale_results, "ABLATION 4: SCALING")

if cross_results:
    all_summaries['Cross-Dataset'] = summarize_ablation_group(cross_results, "ABLATION 5: CROSS-DATASET")


# %% Cell 11: 📈 Visualization — Ablation Charts
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.facecolor': 'white',
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SimpleDyG Architecture Ablation Study', fontsize=16, fontweight='bold', y=1.02)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    # ---- Plot 1: Depth vs NDCG ----
    ax = axes[0, 0]
    if 'Depth' in all_summaries and not all_summaries['Depth'].empty:
        df = all_summaries['Depth']
        ndcg_col = [c for c in df.columns if 'NDCG' in c and '_mean' in c]
        if ndcg_col:
            col = ndcg_col[0]
            std_col = col.replace('_mean', '_std')
            ax.bar(range(len(df)), df[col], yerr=df.get(std_col, 0),
                   color=colors[:len(df)], capsize=4, alpha=0.85, edgecolor='white', linewidth=0.5)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['n_layer'].astype(str) + 'L', fontsize=10)
            ax.set_ylabel(col.replace('_mean', ''))
    ax.set_title('Depth Ablation (NDCG)')
    ax.grid(axis='y', alpha=0.3)
    
    # ---- Plot 2: Width vs NDCG ----
    ax = axes[0, 1]
    if 'Width' in all_summaries and not all_summaries['Width'].empty:
        df = all_summaries['Width']
        ndcg_col = [c for c in df.columns if 'NDCG' in c and '_mean' in c]
        if ndcg_col:
            col = ndcg_col[0]
            std_col = col.replace('_mean', '_std')
            ax.bar(range(len(df)), df[col], yerr=df.get(std_col, 0),
                   color=colors[:len(df)], capsize=4, alpha=0.85, edgecolor='white', linewidth=0.5)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['n_embed'].astype(str), fontsize=10)
            ax.set_ylabel(col.replace('_mean', ''))
    ax.set_title('Width Ablation (NDCG)')
    ax.grid(axis='y', alpha=0.3)
    
    # ---- Plot 3: Heads vs NDCG ----
    ax = axes[0, 2]
    if 'Heads' in all_summaries and not all_summaries['Heads'].empty:
        df = all_summaries['Heads']
        ndcg_col = [c for c in df.columns if 'NDCG' in c and '_mean' in c]
        if ndcg_col:
            col = ndcg_col[0]
            std_col = col.replace('_mean', '_std')
            ax.bar(range(len(df)), df[col], yerr=df.get(std_col, 0),
                   color=colors[:len(df)], capsize=4, alpha=0.85, edgecolor='white', linewidth=0.5)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['n_head'].astype(str) + 'H', fontsize=10)
            ax.set_ylabel(col.replace('_mean', ''))
    ax.set_title('Heads Ablation (NDCG)')
    ax.grid(axis='y', alpha=0.3)
    
    # ---- Plot 4: Scaling — Params vs NDCG (scatter) ----
    ax = axes[1, 0]
    if 'Scale' in all_summaries and not all_summaries['Scale'].empty:
        df = all_summaries['Scale']
        ndcg_col = [c for c in df.columns if 'NDCG' in c and '_mean' in c]
        if ndcg_col:
            col = ndcg_col[0]
            ax.scatter(df['Params (M)'], df[col], s=120, c=colors[:len(df)],
                       edgecolors='black', linewidth=0.8, zorder=5)
            for i, row in df.iterrows():
                ax.annotate(row['Label'], (row['Params (M)'], row[col]),
                           textcoords="offset points", xytext=(5, 8),
                           fontsize=8, alpha=0.8)
            ax.set_xlabel('Parameters (M)')
            ax.set_ylabel(col.replace('_mean', ''))
    ax.set_title('Scaling: Params vs NDCG')
    ax.grid(alpha=0.3)
    
    # ---- Plot 5: Scaling — Params vs Training Time ----
    ax = axes[1, 1]
    if 'Scale' in all_summaries and not all_summaries['Scale'].empty:
        df = all_summaries['Scale']
        ax.scatter(df['Params (M)'], df['Avg Time (min)'], s=120,
                   c=colors[:len(df)], edgecolors='black', linewidth=0.8, zorder=5)
        for i, row in df.iterrows():
            ax.annotate(row['Label'], (row['Params (M)'], row['Avg Time (min)']),
                       textcoords="offset points", xytext=(5, 8),
                       fontsize=8, alpha=0.8)
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('Training Time (min)')
    ax.set_title('Scaling: Params vs Time')
    ax.grid(alpha=0.3)
    
    # ---- Plot 6: Cross-dataset comparison ----
    ax = axes[1, 2]
    if 'Cross-Dataset' in all_summaries and not all_summaries['Cross-Dataset'].empty:
        df = all_summaries['Cross-Dataset']
        ndcg_col = [c for c in df.columns if 'NDCG' in c and '_mean' in c]
        if ndcg_col:
            col = ndcg_col[0]
            std_col = col.replace('_mean', '_std')
            bars = ax.barh(range(len(df)), df[col], xerr=df.get(std_col, 0),
                          color=colors[:len(df)], capsize=4, alpha=0.85,
                          edgecolor='white', linewidth=0.5)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df['Label'], fontsize=9)
            ax.set_xlabel(col.replace('_mean', ''))
    ax.set_title('Cross-Dataset (Hepth)')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    chart_path = '/kaggle/working/ablation_architecture_charts.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 Charts saved: {chart_path}")

except ImportError:
    print("⚠️ matplotlib không available, skip visualization")


# %% Cell 12: 📊 Efficiency Analysis — FLOPS vs Performance
print("\n" + "=" * 70)
print("⚡ EFFICIENCY ANALYSIS")
print("=" * 70)

if scale_results:
    print("\n  Model configs sorted by efficiency (NDCG per M params):\n")
    
    df_scale = all_summaries.get('Scale', pd.DataFrame())
    if not df_scale.empty:
        ndcg_col = [c for c in df_scale.columns if 'NDCG' in c and '_mean' in c]
        if ndcg_col:
            col = ndcg_col[0]
            df_scale = df_scale.copy()
            df_scale['NDCG_per_param'] = df_scale[col] / df_scale['Params (M)']
            df_scale['NDCG_per_minute'] = df_scale[col] / df_scale['Avg Time (min)'].replace(0, np.nan)
            
            df_sorted = df_scale.sort_values('NDCG_per_param', ascending=False)
            
            print(f"  {'Label':35s} | {'Params':>8s} | {'NDCG':>8s} | {'NDCG/M':>10s} | {'NDCG/min':>10s} | {'Time':>8s}")
            print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
            
            for _, row in df_sorted.iterrows():
                print(f"  {row['Label']:35s} | {row['Params (M)']:7.1f}M | "
                      f"{row[col]:8.4f} | {row['NDCG_per_param']:10.4f} | "
                      f"{row['NDCG_per_minute']:10.4f} | {row['Avg Time (min)']:7.1f}m")
            
            # Best efficiency
            best_eff = df_sorted.iloc[0]
            print(f"\n  🏆 Most efficient: {best_eff['Label']} "
                  f"({best_eff['NDCG_per_param']:.4f} NDCG per M params)")


# %% Cell 13: 💾 Save tất cả kết quả
print("\n" + "=" * 70)
print("💾 LƯU KẾT QUẢ")
print("=" * 70)

save_dir = '/kaggle/working/ablation_architecture'
os.makedirs(save_dir, exist_ok=True)

# Save từng nhóm
for group_name, df in all_summaries.items():
    if not df.empty:
        save_path = os.path.join(save_dir, f'ablation_{group_name.lower().replace("-", "_")}.csv')
        df.to_csv(save_path, index=False)
        print(f"  💾 {group_name}: {save_path}")

# Save raw training logs
raw_data = {
    'depth': depth_results,
    'width': width_results,
    'heads': head_results,
    'scale': scale_results,
    'cross_dataset': cross_results,
}

for name, data in raw_data.items():
    if data:
        raw_df = pd.DataFrame(data)
        raw_path = os.path.join(save_dir, f'raw_{name}.csv')
        raw_df.to_csv(raw_path, index=False)
        print(f"  💾 Raw {name}: {raw_path}")

# Save combined summary
all_combined = []
for group_name, df in all_summaries.items():
    if not df.empty:
        df_copy = df.copy()
        df_copy['Group'] = group_name
        all_combined.append(df_copy)

if all_combined:
    combined_df = pd.concat(all_combined, ignore_index=True)
    combined_path = os.path.join(save_dir, 'ablation_architecture_all.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"\n  💾 Combined: {combined_path}")

print(f"\n📁 Tất cả kết quả tại: {save_dir}")


# %% Cell 14: 📝 Kết luận
print("""
╔══════════════════════════════════════════════════════════════════════╗
║               KẾT LUẬN ABLATION ARCHITECTURE                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Kỳ vọng kết quả:                                                 ║
║                                                                    ║
║  📐 DEPTH (n_layer):                                               ║
║  • Shallow (2L) hoạt động kém — không đủ capacity                 ║
║  • 4-6 layers là sweet spot cho small graphs (UCI)                 ║
║  • 12 layers có thể overfitting trên UCI nhưng cần cho Hepth      ║
║                                                                    ║
║  📏 WIDTH (n_embed):                                               ║
║  • D=128 quá nhỏ — representation bottleneck                      ║
║  • D=256-512 đủ tốt cho small vocab                               ║
║  • D=768 diminishing returns nhưng vẫn best                       ║
║                                                                    ║
║  🔀 HEADS (n_head):                                                ║
║  • Single head hoạt động surprisingly decent                      ║
║  • Multi-head giúp nhưng > 8 heads không cải thiện thêm           ║
║  • → Diversity of attention patterns hữu ích nhưng có giới hạn    ║
║                                                                    ║
║  📈 SCALING:                                                       ║
║  • Model nhỏ (Tiny) vẫn hoạt động khá tốt cho dynamic graphs     ║
║  • → SimpleDyG approach tốt ở mọi model size                     ║
║  • Efficiency (NDCG/param) cao nhất ở mid-size models             ║
║                                                                    ║
║  🌐 CROSS-DATASET:                                                 ║
║  • Optimal architecture KHÁC NHAU giữa UCI vs Hepth               ║
║  • UCI: shallow + wide (6L-8H-768D)                               ║
║  • Hepth: deep + narrow (12L-2H-256D)                             ║
║  • → Cần dataset-specific architecture tuning                     ║
║                                                                    ║
║  💡 INSIGHT CHÍNH:                                                  ║
║  Transformer architecture cho dynamic graphs cần cân bằng giữa    ║
║  depth (temporal range) và width (representation richness).        ║
║  Không có one-size-fits-all; graph characteristics (size, density, ║
║  temporal patterns) quyết định optimal architecture.               ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("=" * 70)
print("🎉 HOÀN THÀNH BỘ 3 NOTEBOOKS ABLATION STUDY!")
print("=" * 70)
print("""
📒 Notebook 1 (01_reproduce.py):    Reproduce kết quả gốc
📒 Notebook 2 (02_ablation_temporal.py):  Ablation temporal alignment
📒 Notebook 3 (03_ablation_architecture.py): Ablation model architecture

Tất cả kết quả được lưu tại /kaggle/working/ để download từ Kaggle.
""")
