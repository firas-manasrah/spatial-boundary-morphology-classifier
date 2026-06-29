import pandas as pd
import gzip
from pathlib import Path

OUT = Path('/home/jovyan/scratch/drg_proseg_split')
OUT.mkdir(exist_ok=True)

print("Loading cell ID map...")
with gzip.open(
    '/home/jovyan/scratch/naveed_dataset/'
    'output-XETG00216__0102504__UTD-DN0244__20260403__201340_GROUNDTRUTH/'
    'cell_id_map.csv.gz', 'rt') as f:
    cell_map = pd.read_csv(f)

print("Loading router categories...")
router = pd.read_csv(
    '/home/jovyan/scratch/drg_analysis/merged_drg_fivecategory.csv')
router['category'] = router['category'].str.replace('+', '_').str.replace(' ', '_')

merged_map = cell_map.merge(
    router[['cell_id', 'category']],
    left_on='Xenium Ranger new cell ID',
    right_on='cell_id', how='left')

auto_to_category = dict(zip(
    merged_map['Imported cell ID'].astype(str),
    merged_map['category']))

print("Loading transcripts...")
tx = pd.read_parquet(
    '/home/jovyan/scratch/naveed_dataset/'
    'output-XETG00216__0102504__UTD-DN0244__20260403__201340/'
    'transcripts.parquet')
print(f"Total: {len(tx):,}")

tx['category'] = tx['cell_id'].astype(str).map(auto_to_category)
print(f"Assigned: {tx['category'].notna().sum():,}")
print(tx['category'].value_counts().to_string())

cats = ['Euclidean', 'Fractal', 'Elongated',
        'Complex_elongated', 'Large_neuron']

for cat in cats:
    subset = tx[tx['category']==cat].drop(columns=['category'])
    path = OUT / f'transcripts_{cat}.parquet'
    subset.to_parquet(path)
    print(f"Saved {cat}: {len(subset):,} → {path.name}")

unassigned = tx[tx['category'].isna()].drop(columns=['category'])
unassigned.to_parquet(OUT / 'transcripts_unassigned.parquet')
print(f"Saved unassigned: {len(unassigned):,}")
print(f"\nDone. Files in: {OUT}")
