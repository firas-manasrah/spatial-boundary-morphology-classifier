"""
spatial-boundary-morphology-classifier
=======================================
Classifies segmentation boundary polygons into morphological groups
using five geometric metrics computed from polygon shape alone.
No gene expression data required.

Compatible with any segmentation tool that produces polygon boundaries
in a parquet file with columns: polygon_id, vertex_x, vertex_y

Author: Firas Manasrah
Developed during research internship at BIH/Charite Berlin, 2026
"""

import argparse
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.draw import polygon as sk_poly
from shapely.geometry import Polygon
from pathlib import Path


# ── Geometric metric functions ────────────────────────────────────────────────

def perimeter_area_ratio(pts):
    """Normalised perimeter-area ratio (0=circle, 1=highly irregular)."""
    x, y = pts[:,0], pts[:,1]
    perim = np.sum(np.sqrt(
        np.diff(np.append(x,x[0]))**2 +
        np.diff(np.append(y,y[0]))**2))
    area = 0.5*abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    if area < 1e-6 or perim < 1e-6:
        return np.nan
    return float(np.clip(1 - 4*np.pi*area/perim**2, 0, 1))

def convexity_score(pts):
    """Ratio of polygon area to convex hull area (1=perfectly convex)."""
    try:
        pa = Polygon(pts).area
        ha = ConvexHull(pts).volume
        if ha < 1e-6:
            return np.nan
        return float(np.clip(pa/ha, 0, 1))
    except:
        return np.nan

def fractal_dimension(pts, n_sizes=6):
    """Box-counting fractal dimension of polygon boundary (1.0-2.0)."""
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    w, h = mx - mn
    if w < 3 or h < 3:
        return np.nan
    gs = max(64, int(max(w,h)) + 4)
    scale = (gs-4) / max(w,h)
    px = np.clip(((pts[:,0]-mn[0])*scale+2).astype(int), 0, gs-1)
    py = np.clip(((pts[:,1]-mn[1])*scale+2).astype(int), 0, gs-1)
    grid = np.zeros((gs,gs), dtype=bool)
    rr, cc = sk_poly(py, px, shape=grid.shape)
    grid[rr, cc] = True
    filled = binary_fill_holes(grid)
    bnd = filled & ~binary_erosion(filled)
    if bnd.sum() < 10:
        return np.nan
    sizes = np.unique(
        np.floor(np.logspace(1, np.log10(gs//2), n_sizes)).astype(int))
    sizes = sizes[sizes >= 2]
    counts = np.array([
        sum(1 for i in range(0,gs,s)
              for j in range(0,gs,s)
              if bnd[i:i+s, j:j+s].any())
        for s in sizes], dtype=float)
    valid = counts > 0
    if valid.sum() < 2:
        return np.nan
    return float(np.clip(
        np.polyfit(np.log(1/sizes[valid]), np.log(counts[valid]), 1)[0],
        1.0, 2.0))

def elongation(pts):
    """Ratio of major to minor axis length (1=round, >1=elongated)."""
    max_dist = 0
    p1, p2 = pts[0], pts[1]
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i]-pts[j])
            if d > max_dist:
                max_dist = d
                p1, p2 = pts[i], pts[j]
    D1 = max_dist
    if D1 < 1e-6:
        return np.nan
    major_dir = (p2-p1)/D1
    perp_dir = np.array([-major_dir[1], major_dir[0]])
    proj = pts @ perp_dir
    D2 = proj.max() - proj.min()
    if D2 < 1e-6:
        return np.nan
    return float(D1/D2)

def polygon_area(pts):
    """Signed area of polygon via shoelace formula."""
    x, y = pts[:,0], pts[:,1]
    return 0.5*abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


# ── Five-category classification ──────────────────────────────────────────────

def classify(df, large_area_multiplier=5.0, elongation_sigma=2.0):
    """
    Classify polygon boundaries into five morphological groups.

    Groups (labels are descriptive, not biologically validated):
      Compact      - round, convex, regular boundaries
      Irregular    - complex, non-convex boundaries
      Elongated    - high aspect ratio boundaries
      Complex      - both irregular and elongated
      Large        - area significantly above median

    Parameters
    ----------
    df : DataFrame with columns par, convexity, fractal_dim, elongation, area
    large_area_multiplier : float
        Polygons with area > median * multiplier are classified as Large
    elongation_sigma : float
        Polygons with elongation > mean + sigma*std are classified as Elongated
    """
    df = df.copy()

    # Complexity flags (top quartile = flagged)
    df['flag_par']   = df['par']         > df['par'].quantile(0.75)
    df['flag_conv']  = df['convexity']   < df['convexity'].quantile(0.25)
    df['flag_fd']    = df['fractal_dim'] > df['fractal_dim'].quantile(0.75)
    df['n_flags']    = (df['flag_par'].astype(int) +
                        df['flag_conv'].astype(int) +
                        df['flag_fd'].astype(int))
    df['is_irregular'] = df['n_flags'] >= 2

    # Complexity score (0-1)
    df['complexity'] = (
        df['par'].rank(pct=True) +
        (1 - df['convexity']).rank(pct=True) +
        df['fractal_dim'].rank(pct=True)
    ) / 3.0

    # Elongation threshold
    elong_thresh = (df['elongation'].mean() +
                    elongation_sigma * df['elongation'].std())
    df['is_elongated'] = df['elongation'] > elong_thresh

    # Size threshold
    size_thresh = df['area'].median() * large_area_multiplier
    df['is_large'] = df['area'] > size_thresh

    def assign(row):
        if row['is_large']:
            return 'Large'
        elif row['is_irregular'] and row['is_elongated']:
            return 'Complex'
        elif row['is_irregular']:
            return 'Irregular'
        elif row['is_elongated']:
            return 'Elongated'
        else:
            return 'Compact'

    df['group'] = df.apply(assign, axis=1)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def compute_metrics(boundaries_parquet, id_col='cell_id',
                    x_col='vertex_x', y_col='vertex_y'):
    """
    Compute five geometric metrics for each polygon in a boundaries file.

    Parameters
    ----------
    boundaries_parquet : str or Path
        Path to parquet file with polygon vertex coordinates
    id_col : str
        Column name for polygon identifier
    x_col, y_col : str
        Column names for vertex coordinates
    """
    print(f"Loading boundaries from {boundaries_parquet}...")
    bounds = pd.read_parquet(str(boundaries_parquet))
    print(f"Loaded {len(bounds):,} vertices across "
          f"{bounds[id_col].nunique():,} polygons")

    records = []
    grouped = bounds.groupby(id_col)
    total = len(grouped)

    for i, (pid, grp) in enumerate(grouped):
        if i % 10000 == 0:
            print(f"  Processing {i}/{total}...")
        pts = grp[[x_col, y_col]].values
        if len(pts) < 4:
            continue
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        records.append({
            'polygon_id':  str(pid),
            'par':         perimeter_area_ratio(pts),
            'convexity':   convexity_score(pts),
            'fractal_dim': fractal_dimension(pts),
            'elongation':  elongation(pts),
            'area':        polygon_area(pts),
            'n_vertices':  len(pts),
        })

    df = pd.DataFrame(records)
    print(f"Computed metrics for {len(df):,} polygons")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Classify segmentation boundary polygons by shape morphology")
    parser.add_argument("input",
        help="Path to parquet file with polygon boundary vertices")
    parser.add_argument("output",
        help="Path for output CSV with metrics and group assignments")
    parser.add_argument("--id-col", default="cell_id",
        help="Column name for polygon ID (default: cell_id)")
    parser.add_argument("--x-col", default="vertex_x",
        help="Column name for x coordinates (default: vertex_x)")
    parser.add_argument("--y-col", default="vertex_y",
        help="Column name for y coordinates (default: vertex_y)")
    parser.add_argument("--large-multiplier", type=float, default=5.0,
        help="Area multiplier for Large group threshold (default: 5.0)")
    parser.add_argument("--elongation-sigma", type=float, default=2.0,
        help="Sigma multiplier for elongation threshold (default: 2.0)")
    args = parser.parse_args()

    df = compute_metrics(args.input, args.id_col, args.x_col, args.y_col)
    df = classify(df, args.large_multiplier, args.elongation_sigma)

    print("\n=== Group breakdown ===")
    print(df['group'].value_counts().to_string())
    print(f"Total: {len(df):,}")

    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
