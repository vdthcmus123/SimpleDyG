# ============================================================================
# SimpleDyG Ablation Study — Vai trò Temporal Alignment
# ============================================================================
# Notebook 2/3: Kiểm tra vai trò của Temporal Alignment tokens
#
# Thí nghiệm 3 variants:
#   1. Full model (có temporal tokens) — baseline
#   2. No temporal tokens (bỏ <|timeK|>) — ablation chính
#   3. Shuffled temporal tokens — ablation thứ tự thời gian
#
# Hướng dẫn: Chạy notebook 01_reproduce.py trước để verify pipeline
# ============================================================================

# %% [markdown]
# # 🔬 SimpleDyG Ablation Study — Temporal Alignment
# 
# Kiểm tra giả thuyết: **Temporal Alignment tokens (`<|timeK|>`) là yếu tố 
# quan trọng giúp SimpleDyG hoạt động tốt.**

# %% Cell 1: Setup (giống notebook 1, chạy nếu chưa setup)
import subprocess, os, sys, time, json, re, random, shutil, glob
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


# %% Cell 2: Tạo dữ liệu ABLATION — Remove/Shuffle temporal tokens
def create_ablation_data(dataset, timestamp, variant='no_time'):
    """
    Tạo dữ liệu ablation bằng cách xử lý trực tiếp file .link_prediction.
    
    variant:
        'no_time'     — Bỏ hoàn toàn tất cả <|timeK|> tokens
        'shuffle_time' — Giữ time tokens nhưng shuffle thứ tự
        'reverse_time' — Đảo ngược thứ tự time tokens
    """
    src_dir = os.path.join(BASE_DIR, 'resources', dataset, timestamp)
    dst_dir = os.path.join(BASE_DIR, 'resources', f'{dataset}_{variant}', timestamp)
    os.makedirs(dst_dir, exist_ok=True)
    
    # Regex match <|timeN|> tokens
    time_pattern = re.compile(r'<\|time\d+\|>')
    
    files_to_process = [
        'train.link_prediction',
        'val.link_prediction', 'val_gt.link_prediction',
        'test.link_prediction', 'test_gt.link_prediction',
    ]
    
    for fname in files_to_process:
        src_file = os.path.join(src_dir, fname)
        dst_file = os.path.join(dst_dir, fname)
        
        with open(src_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if variant == 'no_time':
                # Bỏ tất cả <|timeK|> tokens
                new_line = time_pattern.sub('', line)
                # Loại bỏ khoảng trắng thừa
                new_line = re.sub(r'\s+', ' ', new_line).strip()
                
            elif variant == 'shuffle_time':
                # Tìm tất cả time tokens trong dòng
                time_tokens = time_pattern.findall(line)
                if len(time_tokens) > 1:
                    # Shuffle (nhưng tránh trường hợp vẫn đúng thứ tự)
                    shuffled = time_tokens.copy()
                    attempts = 0
                    while shuffled == time_tokens and attempts < 10:
                        random.shuffle(shuffled)
                        attempts += 1
                    # Thay thế lần lượt
                    new_line = line
                    for orig, shuf in zip(time_tokens, shuffled):
                        new_line = new_line.replace(orig, f'__PLACEHOLDER_{shuf}__', 1)
                    for shuf in shuffled:
                        new_line = new_line.replace(f'__PLACEHOLDER_{shuf}__', shuf, 1)
                else:
                    new_line = line
                new_line = new_line.strip()
                
            elif variant == 'reverse_time':
                # Đảo ngược thứ tự time tokens
                time_tokens = time_pattern.findall(line)
                if len(time_tokens) > 1:
                    reversed_tokens = time_tokens[::-1]
                    new_line = line
                    for orig, rev in zip(time_tokens, reversed_tokens):
                        new_line = new_line.replace(orig, f'__PLACEHOLDER_{rev}__', 1)
                    for rev in reversed_tokens:
                        new_line = new_line.replace(f'__PLACEHOLDER_{rev}__', rev, 1)
                else:
                    new_line = line
                new_line = new_line.strip()
            
            new_lines.append(new_line + '\n')
        
        with open(dst_file, 'w') as f:
            f.writelines(new_lines)
    
    # Copy vocab (giữ nguyên vì chỉ thay đổi time tokens)
    vocab_src = os.path.join(BASE_DIR, 'vocabs', dataset, timestamp)
    vocab_dst = os.path.join(BASE_DIR, 'vocabs', f'{dataset}_{variant}', timestamp)
    if os.path.exists(vocab_src):
        shutil.copytree(vocab_src, vocab_dst, dirs_exist_ok=True)
    
    # Copy tokenizer dir
    tok_src = os.path.join(BASE_DIR, 'tokenizers', dataset, timestamp)
    tok_dst = os.path.join(BASE_DIR, 'tokenizers', f'{dataset}_{variant}', timestamp)
    if os.path.exists(tok_src):
        shutil.copytree(tok_src, tok_dst, dirs_exist_ok=True)
    
    print(f"✅ Tạo ablation data '{variant}' cho {dataset}: {dst_dir}")
    
    # Hiện ví dụ so sánh
    print(f"\n  📝 So sánh dòng đầu (train):")
    with open(os.path.join(src_dir, 'train.link_prediction')) as f:
        orig = f.readline().strip()
    with open(os.path.join(dst_dir, 'train.link_prediction')) as f:
        modified = f.readline().strip()
    
    orig_tokens = orig.split()
    mod_tokens = modified.split()
    
    print(f"  ORIGINAL ({len(orig_tokens)} tokens): ...{' '.join(orig_tokens[:20])}...")
    print(f"  {variant.upper()} ({len(mod_tokens)} tokens): ...{' '.join(mod_tokens[:20])}...")
    
    return f'{dataset}_{variant}'

# Tạo ablation data cho UCI_13
print("=" * 70)
print("🔧 TẠO DỮ LIỆU ABLATION CHO UCI_13")
print("=" * 70)

dataset_no_time = create_ablation_data('UCI_13', '12', 'no_time')
print()
dataset_shuffle = create_ablation_data('UCI_13', '12', 'shuffle_time')
print()
dataset_reverse = create_ablation_data('UCI_13', '12', 'reverse_time')


# %% Cell 3: Hàm training (tái sử dụng từ notebook 1)
def run_training(dataset, timestamp, n_layer, n_head, n_embed,
                 batch_size=32, lr=0.0001, seed=0, num_epochs=100,
                 patience=10, node_feat_file=None):
    """Chạy training SimpleDyG cho một cấu hình cụ thể."""
    
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
    
    config_str = f"{dataset} | layers={n_layer}, heads={n_head}, embed={n_embed}, seed={seed}"
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING: {config_str}")
    print(f"{'='*70}")
    
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
    
    if process.returncode == 0:
        print(f"✅ Hoàn thành trong {elapsed:.1f} phút")
    else:
        print(f"❌ Thất bại (code: {process.returncode})")
    
    return output_dir, process.returncode


# %% Cell 4: 🚀 Chạy Ablation — Full Model (baseline, nếu chưa chạy từ NB1)
# Nếu đã chạy notebook 01, có thể skip cell này

print("=" * 70)
print("📊 ABLATION STUDY: FULL MODEL (BASELINE)")
print("=" * 70)

for seed in [0, 1, 2]:
    run_training(
        dataset='UCI_13', timestamp='12',
        n_layer=6, n_head=8, n_embed=768,
        seed=seed, num_epochs=100, patience=10,
    )


# %% Cell 5: 🚀 Chạy Ablation — No Temporal Tokens
print("=" * 70)
print("📊 ABLATION STUDY: NO TEMPORAL TOKENS")
print("=" * 70)

for seed in [0, 1, 2]:
    run_training(
        dataset='UCI_13_no_time', timestamp='12',
        n_layer=6, n_head=8, n_embed=768,
        seed=seed, num_epochs=100, patience=10,
    )


# %% Cell 6: 🚀 Chạy Ablation — Shuffled Temporal Tokens
print("=" * 70)
print("📊 ABLATION STUDY: SHUFFLED TEMPORAL TOKENS")
print("=" * 70)

for seed in [0, 1, 2]:
    run_training(
        dataset='UCI_13_shuffle_time', timestamp='12',
        n_layer=6, n_head=8, n_embed=768,
        seed=seed, num_epochs=100, patience=10,
    )


# %% Cell 7: 🚀 Chạy Ablation — Reversed Temporal Tokens
print("=" * 70)
print("📊 ABLATION STUDY: REVERSED TEMPORAL TOKENS")
print("=" * 70)

for seed in [0, 1, 2]:
    run_training(
        dataset='UCI_13_reverse_time', timestamp='12',
        n_layer=6, n_head=8, n_embed=768,
        seed=seed, num_epochs=100, patience=10,
    )


# %% Cell 8: 📊 Thu thập & so sánh tất cả kết quả Ablation
def collect_ablation_results(dataset):
    """Thu thập kết quả ablation."""
    result_file = os.path.join(BASE_DIR, 'topk_scores_seed_jac', f'{dataset}_SimpleDyG.csv')
    
    if os.path.exists(result_file):
        return pd.read_csv(result_file)
    
    # Fallback
    pattern = os.path.join(BASE_DIR, f'output/{dataset}/**/results_seed_jac/test_score/test_results_epoch.csv')
    files = glob.glob(pattern, recursive=True)
    if files:
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return None


print("=" * 70)
print("📊 ABLATION STUDY: KẾT QUẢ TỔNG HỢP")
print("=" * 70)

ablation_variants = {
    'Full Model (w/ temporal)': 'UCI_13',
    'No Temporal Tokens': 'UCI_13_no_time',
    'Shuffled Temporal': 'UCI_13_shuffle_time',
    'Reversed Temporal': 'UCI_13_reverse_time',
}

ablation_summary = []

for variant_name, ds_name in ablation_variants.items():
    df = collect_ablation_results(ds_name)
    if df is not None:
        # Tìm cột NDCG và Jaccard
        ndcg_cols = [c for c in df.columns if 'NDCG' in c]
        jac_cols = [c for c in df.columns if 'jaccard' in c]
        
        row = {'Variant': variant_name}
        for col in ndcg_cols:
            row[f'{col}_mean'] = df[col].mean()
            row[f'{col}_std'] = df[col].std()
        for col in jac_cols:
            row[f'{col}_mean'] = df[col].mean()
            row[f'{col}_std'] = df[col].std()
        row['n_seeds'] = len(df)
        
        ablation_summary.append(row)
        print(f"\n  {variant_name}:")
        for col in ndcg_cols + jac_cols:
            print(f"    {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")
    else:
        print(f"\n  ⚠️ {variant_name}: Chưa có kết quả")

# Tạo bảng tổng hợp
if ablation_summary:
    summary_df = pd.DataFrame(ablation_summary)
    print("\n" + "=" * 70)
    print("📋 BẢNG TỔNG HỢP ABLATION")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    
    # Save
    save_path = '/kaggle/working/ablation_results.csv'
    summary_df.to_csv(save_path, index=False)
    print(f"\n💾 Saved: {save_path}")


# %% Cell 9: 📈 Visualization (optional)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if ablation_summary:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        variants = [r['Variant'] for r in ablation_summary]
        
        # Plot NDCG
        ndcg_means = [r.get('NDCG@5_mean', 0) for r in ablation_summary]
        ndcg_stds = [r.get('NDCG@5_std', 0) for r in ablation_summary]
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        axes[0].bar(range(len(variants)), ndcg_means, yerr=ndcg_stds, 
                    color=colors[:len(variants)], capsize=5, alpha=0.85)
        axes[0].set_xticks(range(len(variants)))
        axes[0].set_xticklabels(variants, rotation=20, ha='right', fontsize=9)
        axes[0].set_ylabel('NDCG@5')
        axes[0].set_title('Ablation Study — NDCG@5')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot Jaccard
        jac_means = [r.get('jaccard@5_mean', 0) for r in ablation_summary]
        jac_stds = [r.get('jaccard@5_std', 0) for r in ablation_summary]
        
        axes[1].bar(range(len(variants)), jac_means, yerr=jac_stds,
                    color=colors[:len(variants)], capsize=5, alpha=0.85)
        axes[1].set_xticks(range(len(variants)))
        axes[1].set_xticklabels(variants, rotation=20, ha='right', fontsize=9)
        axes[1].set_ylabel('Jaccard')
        axes[1].set_title('Ablation Study — Jaccard')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/kaggle/working/ablation_chart.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("💾 Chart saved: /kaggle/working/ablation_chart.png")
    
except ImportError:
    print("⚠️ matplotlib không available, skip visualization")


# %% Cell 10: 📝 Kết luận Ablation
print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    KẾT LUẬN ABLATION STUDY                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Kỳ vọng kết quả:                                                 ║
║  1. Full Model >> No Temporal   → Time tokens quan trọng           ║
║  2. Full Model >  Shuffled      → Thứ tự thời gian quan trọng     ║
║  3. Full Model >  Reversed      → Hướng thời gian quan trọng      ║
║  4. Shuffled ≈ Reversed > No    → Có time token vẫn tốt hơn       ║
║                                    dù sai thứ tự                   ║
║                                                                    ║
║  → Temporal Alignment không chỉ cung cấp thông tin "có thời gian"  ║
║    mà còn encode đúng thứ tự tiến hóa của đồ thị                   ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")
