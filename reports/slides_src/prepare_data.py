"""Unpack the data the slide charts need (run from reports/slides_src).

Sources inside the repo:
  notebooks/SimpleDyG_Previous_Results.zip          -> data/previous_results/ (round 1: histories, predictions, dim search)
  results/round2_uci_permute/*.zip                  -> data/results_uci_permute/
  results/round3_ml_corrected/*.zip, ml_histories.tgz -> data/result_ml/, data/hist_ml/
  notebooks/SimpleDyG_Kaggle_2GPU_Inputs.zip        -> data/steps_UCI.csv, data/steps_ML.csv (interactions per time step)
data/all_runs.csv (per-run scores + collapse census for all 125 runs) is already committed.
"""
import io
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('data')
DATA.mkdir(exist_ok=True)

with zipfile.ZipFile(ROOT / 'notebooks/SimpleDyG_Previous_Results.zip') as z:
    z.extractall(DATA)
for name, folder in [('results_uci_permute', 'round2_uci_permute'), ('result_ml', 'round3_ml_corrected')]:
    for zp in (ROOT / 'results' / folder).glob('*.zip'):
        with zipfile.ZipFile(zp) as z:
            z.extractall(DATA / name / zp.stem)
with tarfile.open(ROOT / 'results/round3_ml_corrected/ml_histories.tgz') as t:
    t.extractall(DATA / 'hist_ml', filter='data')
with zipfile.ZipFile(ROOT / 'notebooks/SimpleDyG_Kaggle_2GPU_Inputs.zip') as z:
    for ds, raw in [('UCI', 'uci'), ('ML', 'movie')]:
        frame = pd.read_csv(io.BytesIO(z.read(f'SimpleDyG/all_data/{raw}/12/ml_{raw}.csv')), usecols=['timestamp'])
        frame.timestamp.value_counts().sort_index().to_csv(DATA / f'steps_{ds}.csv')
print('data ready:', sorted(p.name for p in DATA.iterdir()))
