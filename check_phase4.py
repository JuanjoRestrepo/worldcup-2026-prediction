# Check Phase 4: Dataset Analysis and Preparation for Modeling
# check_phase4.py
import sys

import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv('data/gold/features_dataset.csv')

print('DATASET ANALYSIS - PHASE 4')
print('='*60)
print(f'\nShape: {df.shape}')
print(f'Date range: {df["date"].min()} to {df["date"].max()}')

# 1. Target
if 'target_multiclass' in df.columns:
    print(f'\n1️⃣ TARGET ENCODING:')
    print(f'   Dtype: {df["target_multiclass"].dtype}')
    print(f'   Unique values: {sorted(df["target_multiclass"].unique())}')
    vc = df['target_multiclass'].value_counts().sort_index()
    print(f'   Counts: {dict(vc)}')
    dist = (df['target_multiclass'].value_counts(normalize=True).sort_index() * 100).round(2)
    print(f'   Distribution (%): {dict(dist)}')

# 2. Scaling
print(f'\n2️⃣ SCALING / PREPROCESSING:')
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f'   Numeric columns: {len(numeric_cols)}')
stats = df[numeric_cols].describe().T[['min', 'max']].round(2)
wide_range = stats[(stats['max'] > 100) | (stats['min'] < -100)]
print(f'   Columns with wide range (>100): {len(wide_range)}')
if len(wide_range) > 0:
    print(f'   Raw features (not scaled):')
    for col in wide_range.head(3).index:
        print(f'     - {col}: [{stats.loc[col, "min"]}, {stats.loc[col, "max"]}]')

# 3. Balance
print(f'\n3️⃣ CLASS BALANCE:')
dist_pct = (df['target_multiclass'].value_counts(normalize=True).sort_index() * 100)
min_c = dist_pct.min()
max_c = dist_pct.max()
ratio = max_c / min_c
print(f'   Imbalance ratio (max/min): {ratio:.2f}x')
status = "BALANCED ✅" if ratio < 1.5 else "SLIGHTLY IMBALANCED ⚠️" if ratio < 3 else "IMBALANCED ❌"
print(f'   Status: {status}')

# 4. Quality
print(f'\n4️⃣ DATA QUALITY:')
print(f'   Missing values (total): {df.isna().sum().sum()}')
print(f'   Status: {"✅ CLEAN" if df.isna().sum().sum() == 0 else "⚠️ HAS GAPS"}')

# 5. Features to drop
print(f'\n5️⃣ FEATURE PREPARATION:')
cols_to_drop = ['date', 'homeTeam', 'awayTeam', 'homeGoals', 'awayGoals', 'target_multiclass']
X_cols = [c for c in df.columns if c not in cols_to_drop]
print(f'   Features for X: {len(X_cols)}')
print(f'   Target for y: target_multiclass')
print(f'   Categorical features to handle: {[c for c in df.columns if df[c].dtype == "object"]}')

print('\n' + '='*60)
print('✅ READY FOR PHASE 4 IMPLEMENTATION')
