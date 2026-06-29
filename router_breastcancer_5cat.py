import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.draw import polygon as sk_poly
from pathlib import Path

OUT = Path('/home/jovyan/scratch/breastcancer_analysis')
OUT.mkdir(exist_ok=True)

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

def elongation(pts):
    max_dist = 0
    p1, p2 = pts[0], pts[1]
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_dist:
                max_dist = d
                p1, p2 = pts[i], pts[j]
    D1 = max_dist
    if D1 < 1e-6: return np.nan
    major_dir = (p2 - p1) / D1
    perp_dir = np.array([-major_dir[1], major_dir[0]])
    proj = pts @ perp_dir
    D2 = proj.max() - proj.min()
    if D2 < 1e-6: return np.nan
    return float(D1 / D2)

print("Loading cell boundaries...")
bounds = pd.read_parquet(
    '/home/jovyan/scratch/SpaceHack2/git_repo_clone/data/outs/'
    'cell_boundaries.parquet')
print(f"Cells: {bounds['cell_id'].nunique():,}")

print("Loading metadata and labels...")
cells = pd.read_parquet(
    '/home/jovyan/scratch/SpaceHack2/git_repo_clone/data/outs/'
    'cells.parquet')
labels = pd.read_csv(
    '/home/jovyan/scratch/SpaceHack2/git_repo_clone/data/'
    'labels.tsv.csv', sep='\t')
labels.index = labels.index + 1
labels.index.name = 'cell_id'
labels = labels.reset_index()

print("Computing five metrics...")
records = []
grouped = bounds.groupby('cell_id')
total = len(grouped)
for i, (cell_id, group) in enumerate(grouped):
    if i % 10000 == 0:
        print(f"  {i}/{total}...")
    pts = group[['vertex_x','vertex_y']].values
    if len(pts) < 4:
        continue
    records.append({
        'cell_id':     cell_id,
        'par':         perimeter_area_ratio(pts),
        'convexity':   convexity_score(pts),
        'fractal_dim': fractal_dimension(pts),
        'elongation':  elongation(pts),
        'n_vertices':  len(pts),
    })

df = pd.DataFrame(records)
cells['cell_id'] = cells['cell_id'].astype(int)
df['cell_id'] = df['cell_id'].astype(int)
df = df.merge(cells[['cell_id','cell_area']], on='cell_id')

# Router thresholds
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

local_median  = df['cell_area'].median()
elong_thresh  = df['elongation'].mean() + 2*df['elongation'].std()
df['is_large']     = df['cell_area'] > local_median * 5.0
df['is_fractal']   = df['route'] == 'fractal'
df['is_elongated'] = df['elongation'] > elong_thresh

def assign_cat(row):
    if row['is_large']:
        return 'Large_cell'
    elif row['is_fractal'] and row['is_elongated']:
        return 'Complex_elongated'
    elif row['is_fractal']:
        return 'Fractal'
    elif row['is_elongated']:
        return 'Elongated'
    else:
        return 'Euclidean'

df['category'] = df.apply(assign_cat, axis=1)

print("\nFive-category breakdown:")
print(df['category'].value_counts().to_string())
print(f"\nElongation threshold: {elong_thresh:.3f}")
print(f"Size threshold: {local_median*5:.1f} um2")

# Merge with labels
labels['cell_id'] = labels['cell_id'].astype(int)
merged = df.merge(labels, on='cell_id', how='left')
merged = merged.merge(
    cells[['cell_id','x_centroid','y_centroid']],
    on='cell_id', how='left')

print("\nComplexity by cell type:")
result = merged.groupby('Cluster')['complexity'].agg(
    ['median','count']).round(3)
result = result.sort_values('median', ascending=False)
print(result.to_string())

merged.to_csv(OUT / 'router_breastcancer_5cat.csv', index=False)
result.to_csv(OUT / 'complexity_breastcancer_5cat.csv')
print(f"\nSaved to {OUT}")
