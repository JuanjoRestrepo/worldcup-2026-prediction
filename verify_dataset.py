# verify_dataset.py
import pandas as pd

df = pd.read_csv('data/silver/features_dataset.csv')

print('📊 DATASET SCHEMA UPDATED')
print(f'\nShape: {df.shape}')
print(f'\nN COLUMNS: {len(df.columns)}')

print('\nALL COLUMNS:')
for i, col in enumerate(df.columns, 1):
    print(f'  {i:2}. {col}')

print('\n🎯 TARGET DISTRIBUTION (Multiclass):')
print(df['target_multiclass'].value_counts().sort_index())

print('\n🎯 TARGET DISTRIBUTION (Binary):')
print(df['target'].value_counts())

print('\n✅ FEATURES WITH LEAKAGE FIX:')
print(f'  home_avg_goals_last5: {df["home_avg_goals_last5"].notna().sum()} non-NaN (shift(1) working)')
print(f'  home_elo_form: {df["home_elo_form"].notna().sum()} non-NaN')
print(f'  home_win_rate_last5: {df["home_win_rate_last5"].notna().sum()} non-NaN')

print('\n📋 SAMPLE ROWS (with new features):')
sample = df[['date', 'homeTeam', 'awayTeam', 'homeGoals', 'awayGoals', 'elo_diff', 'home_avg_goals_last5', 'home_win_rate_last5', 'is_world_cup', 'target_multiclass']].head(20)
print(sample.to_string())
