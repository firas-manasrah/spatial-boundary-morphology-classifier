"""
Five-category geometry-adaptive router for mouse brain (Sergio Salas dataset).
Uses polygons.geojson from ProSeg output.
Adds elongation and size metrics to the existing three-metric router.
Output: ~/scratch/proseg_analysis/router_mousebrain_5cat.csv
"""

import json
import gzip
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, shape
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.draw import polygon as sk_poly
from pathlib import Path

OUT     = Path('/home/jovyan/scratch/proseg_analysis')
GEOJSON = Path('/home/jovyan/scratch/proseg_results/run_default/polygons.geojson')

# ── Metric functions ──────────────────────────────────────────────────────────

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
    except:
        return np.nan

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

def cell_area(pts):
    x, y = pts[:,0], pts[:,1]
    return 0.5*abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# ── Load polygons ─────────────────────────────────────────────────────────────

print("Loading mouse brain polygons...")

# Try gzip first, then plain
try:
    with gzip.open(str(GEOJSON), 'rt') as f:
        gj = json.load(f)
    print("Loaded gzipped GeoJSON")
except:
    with open(str(GEOJSON), 'rt') as f:
        gj = json.load(f)
    print("Loaded plain GeoJSON")

features = gj['features']
print(f"Total polygons: {len(features):,}")

# ── Compute metrics ───────────────────────────────────────────────────────────

print("Computing five metrics...")
records = []

for i, feat in enumerate(features):
    if i % 5000 == 0:
        print(f"  {i}/{len(features)}...")

    cell_id = (feat.get('properties') or {}).get('cell_id', i)
    geom = feat['geometry']

    # Handle MultiPolygon — take largest part
    if geom['type'] == 'MultiPolygon':
        coords = max(geom['coordinates'], key=lambda c: Polygon(c[0]).area)[0]
    elif geom['type'] == 'Polygon':
        coords = geom['coordinates'][0]
    else:
        continue

    pts = np.array(coords)
    if len(pts) < 4:
        continue

    # Remove duplicate closing point if present
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    area = cell_area(pts)

    records.append({
        'cell_id':     str(cell_id),
        'par':         perimeter_area_ratio(pts),
        'convexity':   convexity_score(pts),
        'fractal_dim': fractal_dimension(pts),
        'elongation':  elongation(pts),
        'cell_area':   area,
        'n_vertices':  len(pts),
    })

df = pd.DataFrame(records)
print(f"Computed metrics for {len(df):,} cells")

# ── Router classification ─────────────────────────────────────────────────────

print("Classifying cells...")

# Three-metric flags
df['flag_par']  = df['par']  > df['par'].quantile(0.75)
df['flag_conv'] = df['convexity'] < df['convexity'].quantile(0.25)
df['flag_fd']   = df['fractal_dim'] > df['fractal_dim'].quantile(0.75)
df['flags'] = (df['flag_par'].astype(int) +
               df['flag_conv'].astype(int) +
               df['flag_fd'].astype(int))
df['is_fractal'] = df['flags'] >= 2

# Complexity score
df['complexity'] = (
    df['par'].rank(pct=True) +
    (1-df['convexity']).rank(pct=True) +
    df['fractal_dim'].rank(pct=True)) / 3.0

# Elongation threshold
elong_thresh = df['elongation'].mean() + 2*df['elongation'].std()
df['is_elongated'] = df['elongation'] > elong_thresh

# Size threshold
local_median = df['cell_area'].median()
df['is_large'] = df['cell_area'] > local_median * 5.0

print(f"Elongation threshold: {elong_thresh:.3f}")
print(f"Size threshold: {local_median*5:.1f} um2")

# Five-category assignment
def assign_cat(row):
    if row['is_large']:
        return 'Large_neuron'
    elif row['is_fractal'] and row['is_elongated']:
        return 'Complex_elongated'
    elif row['is_fractal']:
        return 'Fractal'
    elif row['is_elongated']:
        return 'Elongated'
    else:
        return 'Euclidean'

df['category'] = df.apply(assign_cat, axis=1)

# ── Results ───────────────────────────────────────────────────────────────────

print("\n=== Five-category breakdown ===")
cats = df['category'].value_counts()
print(cats.to_string())
print(f"\nTotal cells: {len(df):,}")
print(f"\nComplexity by category:")
print(df.groupby('category')['complexity'].agg(
    ['median','count']).round(3).sort_values('median', ascending=False).to_string())

# ── Merge with Salas annotations ──────────────────────────────────────────────

print("\nMerging with Salas annotations...")
try:
    import anndata as ad
    adata = ad.read_h5ad(
        '/home/jovyan/scratch/SpaceHack2/userfolders/'
        'markrobinsonuzh/SpaceHack2023-main/data/'
        'xenium-mouse-brain-SergioSalas/'
        'adata_multisection_nuclei_r1_with_annotations.h5ad')
    salas = adata.obs[['cell_id','Class','spatial_annotation']].copy()
    salas['cell_id'] = salas['cell_id'].astype(str)
    merged = df.merge(salas, on='cell_id', how='left')
    overlap = merged['Class'].notna().sum()
    print(f"Overlap with Salas annotations: {overlap:,} cells")

    print("\n=== Complexity by cell type (5-cat router) ===")
    result = merged.dropna(subset=['Class']).groupby(
        'Class')['complexity'].agg(['median','count']).round(3)
    result = result.sort_values('median', ascending=False)
    print(result.to_string())

    print("\n=== Category by cell type ===")
    cat_ct = merged.dropna(subset=['Class']).groupby(
        ['Class','category']).size().unstack(fill_value=0)
    print(cat_ct.to_string())

    merged.to_csv(OUT / 'router_mousebrain_5cat_annotated.csv', index=False)
    print(f"\nSaved annotated: router_mousebrain_5cat_annotated.csv")
except Exception as e:
    print(f"Annotation merge failed: {e}")
    print("Saving router results only")

# Save router results
df.to_csv(OUT / 'router_mousebrain_5cat.csv', index=False)
print(f"Saved: router_mousebrain_5cat.csv")
print("Done.")
