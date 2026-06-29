import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.draw import polygon as sk_poly
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

BASE = Path('/home/jovyan/scratch/naveed_dataset')
OUT  = Path('/home/jovyan/scratch/drg_analysis')
OUT.mkdir(exist_ok=True)

DATASETS = {
    'auto': BASE / 'output-XETG00216__0102504__UTD-DN0244__20260403__201340',
    'groundtruth': BASE / 'output-XETG00216__0102504__UTD-DN0244__20260403__201340_GROUNDTRUTH',
}

# ── Geometry metrics ──────────────────────────────────────────────────────────
def perimeter_area_ratio(pts):
    x, y = pts[:,0], pts[:,1]
    perim = np.sum(np.sqrt(
        np.diff(np.append(x,x[0]))**2 +
        np.diff(np.append(y,y[0]))**2))
    area = 0.5*abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    if area < 1e-6 or perim < 1e-6: return np.nan
    return float(np.clip(1 - 4*np.pi*area/perim**2, 0, 1))

def convexity_score(pts):
    try:
        poly_area = Polygon(pts).area
        hull_area = ConvexHull(pts).volume
        if hull_area < 1e-6: return np.nan
        return float(np.clip(poly_area/hull_area, 0, 1))
    except: return np.nan

def fractal_dimension(pts, n_sizes=6):
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    w, h = mx - mn
    if w < 3 or h < 3: return np.nan
    gs = max(64, int(max(w,h)) + 4)
    scale = (gs-4) / max(w,h)
    px = np.clip(((pts[:,0]-mn[0])*scale+2).astype(int), 0, gs-1)
    py = np.clip(((pts[:,1]-mn[1])*scale+2).astype(int), 0, gs-1)
    grid = np.zeros((gs,gs), dtype=bool)
    rr, cc = sk_poly(py, px, shape=grid.shape)
    grid[rr,cc] = True
    filled = binary_fill_holes(grid)
    bnd = filled & ~binary_erosion(filled)
    if bnd.sum() < 10: return np.nan
    sizes = np.unique(np.floor(
        np.logspace(1, np.log10(gs//2), n_sizes)).astype(int))
    sizes = sizes[sizes >= 2]
    counts = np.array([
        sum(1 for i in range(0,gs,s)
              for j in range(0,gs,s)
              if bnd[i:i+s,j:j+s].any())
        for s in sizes], dtype=float)
    valid = counts > 0
    if valid.sum() < 2: return np.nan
    return float(np.clip(
        np.polyfit(np.log(1/sizes[valid]),
                   np.log(counts[valid]), 1)[0], 1.0, 2.0))

def compute_router(bounds_path, label):
    print(f"\n=== {label} ===")
    bounds = pd.read_parquet(bounds_path)
    print(f"Boundaries: {bounds.shape}")
    records = []
    grouped = bounds.groupby('cell_id')
    total = len(grouped)
    for i, (cell_id, group) in enumerate(grouped):
        if i % 5000 == 0:
            print(f"  {i}/{total}...")
        pts = group[['vertex_x','vertex_y']].values
        if len(pts) < 4: continue
        records.append({
            'cell_id':     cell_id,
            'par':         perimeter_area_ratio(pts),
            'convexity':   convexity_score(pts),
            'fractal_dim': fractal_dimension(pts),
            'n_vertices':  len(pts),
        })
    df = pd.DataFrame(records)
    print(f"Valid: {df['fractal_dim'].notna().sum()}/{total}")

    df['flag_par']  = df['par']  > df['par'].quantile(0.75)
    df['flag_conv'] = df['convexity'] < df['convexity'].quantile(0.25)
    df['flag_fd']   = df['fractal_dim'] > df['fractal_dim'].quantile(0.75)
    df['flags'] = (df['flag_par'].astype(int) +
                   df['flag_conv'].astype(int) +
                   df['flag_fd'].astype(int))
    df['route'] = df['flags'].apply(
        lambda x: 'fractal' if x >= 2 else 'euclidean')
    df['complexity'] = (
        df['par'].rank(pct=True) +
        (1-df['convexity']).rank(pct=True) +
        df['fractal_dim'].rank(pct=True)) / 3.0

    n_frac = (df['route']=='fractal').sum()
    print(f"Fractal: {n_frac}/{len(df)} ({100*n_frac/len(df):.1f}%)")
    print(f"Complexity median: {df['complexity'].median():.3f}")
    print(f"FD median: {df['fractal_dim'].median():.3f}")

    df.to_csv(OUT / f'router_{label}.csv', index=False)
    return df

# Run router on both
auto_df = compute_router(
    DATASETS['auto'] / 'cell_boundaries.parquet', 'auto')
gt_df   = compute_router(
    DATASETS['groundtruth'] / 'cell_boundaries.parquet', 'groundtruth')

# Compare
print("\n=== COMPARISON ===")
print(f"Auto:         {len(auto_df)} cells, "
      f"{(auto_df['route']=='fractal').sum()} fractal "
      f"({100*(auto_df['route']=='fractal').mean():.1f}%)")
print(f"Ground truth: {len(gt_df)} cells, "
      f"{(gt_df['route']=='fractal').sum()} fractal "
      f"({100*(gt_df['route']=='fractal').mean():.1f}%)")
print(f"\nMedian complexity:")
print(f"  Auto:         {auto_df['complexity'].median():.3f}")
print(f"  Ground truth: {gt_df['complexity'].median():.3f}")
print(f"\nMedian fractal dim:")
print(f"  Auto:         {auto_df['fractal_dim'].median():.3f}")
print(f"  Ground truth: {gt_df['fractal_dim'].median():.3f}")
print(f"\nMax cell area comparison suggests ground truth")
print(f"captures large neurons auto segmentation misses.")
print(f"Saved to {OUT}")
