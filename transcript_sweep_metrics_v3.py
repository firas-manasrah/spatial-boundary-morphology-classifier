import pandas as pd
import numpy as np
from pathlib import Path

AUTO = Path('/home/jovyan/scratch/naveed_dataset/output-XETG00216__0102504__UTD-DN0244__20260403__201340')
RUNS = Path('/home/jovyan/scratch/validating_proseg/runs')

x1,x2,y1,y2 = 250,450,2500,2700

print('Loading AUTO transcripts...')
tx_auto = pd.read_parquet(str(AUTO/'transcripts.parquet'))
tx_auto = tx_auto[tx_auto['qv']>=20]
r2_auto = tx_auto[
    (tx_auto['x_location']>=x1)&(tx_auto['x_location']<=x2)&
    (tx_auto['y_location']>=y1)&(tx_auto['y_location']<=y2)
].copy()

auto_assigned = set(r2_auto[r2_auto['cell_id']!='UNASSIGNED']['transcript_id'])
auto_unassigned = set(r2_auto[r2_auto['cell_id']=='UNASSIGNED']['transcript_id'])
print(f'AUTO assigned: {len(auto_assigned):,} | unassigned: {len(auto_unassigned):,}')

results = []
compactness_vals = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]

for C in compactness_vals:
    tx_path = RUNS/f'compactness_{C}/proseg-output.zarr/points/transcripts/points.parquet/part.0.parquet'
    if not tx_path.exists():
        print(f'c={C}: not found')
        continue

    tx_ps = pd.read_parquet(str(tx_path),
                            columns=['transcript_id','x','y','assignment','background'])
    r2_ps = tx_ps[
        (tx_ps['x']>=x1)&(tx_ps['x']<=x2)&
        (tx_ps['y']>=y1)&(tx_ps['y']<=y2)
    ].copy()

    ps_assigned   = set(r2_ps[~r2_ps['background']]['transcript_id'])
    ps_unassigned = set(r2_ps[r2_ps['background']]['transcript_id'])

    # Key metrics
    # 1. Recovery: AUTO assigned + ProSeg assigned (kept)
    recovered = len(auto_assigned & ps_assigned)
    # 2. Lost: AUTO assigned + ProSeg background (lost to background)
    lost = len(auto_assigned & ps_unassigned)
    # 3. Gained: AUTO unassigned + ProSeg assigned (rescued from background)
    gained = len(auto_unassigned & ps_assigned)
    # 4. Total AUTO assigned
    n_auto = len(auto_assigned)

    pct_recovered = 100 * recovered / n_auto
    pct_lost      = 100 * lost / n_auto
    pct_gained    = 100 * gained / len(auto_unassigned) if auto_unassigned else 0

    results.append({
        'compactness': C,
        'n_auto_assigned': n_auto,
        'recovered': recovered,
        'lost_to_bg': lost,
        'gained_from_bg': gained,
        'pct_recovered': pct_recovered,
        'pct_lost': pct_lost,
        'pct_gained': pct_gained,
        'pct_background': 100*len(ps_unassigned)/(len(ps_assigned)+len(ps_unassigned))
    })
    print(f'c={C}: recovered={pct_recovered:.1f}% lost={pct_lost:.1f}% gained={pct_gained:.1f}% background={100*len(ps_unassigned)/(len(ps_assigned)+len(ps_unassigned)):.1f}%')

df = pd.DataFrame(results)
print('\n=== SUMMARY ===')
print(df[['compactness','pct_recovered','pct_lost','pct_gained','pct_background']].to_string(index=False))
df.to_csv('/home/jovyan/scratch/monday_meeting/tables/sweep_transcript_metrics.csv',index=False)
print('Saved.')
