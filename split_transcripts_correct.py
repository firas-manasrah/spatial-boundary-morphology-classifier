"""
CORRECTED transcript splitting for router v2.
Uses AUTO segmentation cell boundaries for routing — NOT ground truth.
This is the correct workflow:
1. Router classifies AUTO segmentation polygons
2. Each auto-cell gets a category
3. Transcripts are split by which auto-cell they belong to
4. ProSeg re-segments each category subset
"""
import pandas as pd
from pathlib import Path

OUT = Path('/home/jovyan/scratch/drg_proseg_split_v2')
OUT.mkdir(exist_ok=True)

AUTO = Path('/home/jovyan/scratch/naveed_dataset/'
            'output-XETG00216__0102504__UTD-DN0244__20260403__201340')

print("Loading transcripts...")
tx = pd.read_parquet(str(AUTO/'transcripts.parquet'))
print(f"Total: {len(tx):,}")

print("Loading router results on AUTO boundaries...")
router = pd.read_csv('/home/jovyan/scratch/drg_analysis/router_auto.csv')
print(f"Router cells: {len(router):,}")
print(router['route'].value_counts().to_string())

# Map auto cell_id -> category
cat_map = dict(zip(router['cell_id'].astype(str), router['route']))

# Assign category to each transcript based on auto cell assignment
tx['category'] = tx['cell_id'].astype(str).map(cat_map)
print(f"\nAssigned: {tx['category'].notna().sum():,}")
print(f"Unassigned/background: {tx['category'].isna().sum():,}")
print("\nCategory breakdown:")
print(tx['category'].value_counts().to_string())

# Split and save
cats = ['Euclidean','Fractal','Elongated','Complex_elongated','Large_neuron']
for cat in cats:
    subset = tx[tx['category']==cat].drop(columns=['category'])
    path = OUT/f'transcripts_{cat}.parquet'
    subset.to_parquet(path)
    print(f"Saved {cat}: {len(subset):,} -> {path.name}")

unassigned = tx[tx['category'].isna()].drop(columns=['category'])
unassigned.to_parquet(OUT/'transcripts_unassigned.parquet')
print(f"Saved unassigned: {len(unassigned):,}")
print(f"\nDone. Files in: {OUT}")
