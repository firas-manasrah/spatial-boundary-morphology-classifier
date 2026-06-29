"""
proseg_analysis.py — FIXED VERSION
Fixes: gzip loading, MultiPolygon coordinate parsing
"""

import json
import gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from pathlib import Path

RESULTS = Path("/home/jovyan/scratch/proseg_results")
OUT     = Path("/home/jovyan/scratch/proseg_analysis")
SIGS    = Path("/home/jovyan/scratch/Sainsc_Xenium_DRG/Xenium/Xenium_signatures.tsv")
OUT.mkdir(exist_ok=True)

RUNS = ["compact", "default", "relaxed"]
COMPACTNESS = {"compact": 2.0, "default": 1.3, "relaxed": 0.8}
RUN_COLORS  = {"compact": "#D85A30", "default": "#1DB87A", "relaxed": "#534AB7"}


# ── 1. Load polygons (FIXED: gzip.open) ──────────────────────────────────────
def load_polygons(run):
    path = RESULTS / f"run_{run}" / "polygons.geojson"
    print(f"[{run}] Loading polygons...")
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        data = json.load(f)
    features = data["features"]
    print(f"[{run}] {len(features)} polygons loaded")
    return features


# ── 2. Extract coords (FIXED: MultiPolygon nesting) ──────────────────────────
def get_coords(feat):
    """
    Extract the largest outer ring from a Polygon or MultiPolygon.
    MultiPolygon structure: coordinates[poly_idx][ring_idx][point_idx]
    Polygon structure:      coordinates[ring_idx][point_idx]
    """
    geom = feat["geometry"]
    gtype = geom["type"]
    if gtype == "Polygon":
        rings = geom["coordinates"]
        # rings[0] is outer ring — list of [x, y]
        return rings[0]
    elif gtype == "MultiPolygon":
        polys = geom["coordinates"]
        # Each poly: [ring0, ring1, ...]; ring0 is outer
        # Pick the polygon with the most outer-ring vertices
        best = max(polys, key=lambda p: len(p[0]))
        return best[0]
    return None


# ── 3. Geometry metrics ───────────────────────────────────────────────────────
def perimeter_area_ratio(coords):
    pts = np.array(coords, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    perim = np.sum(np.sqrt(
        np.diff(np.append(x, x[0]))**2 +
        np.diff(np.append(y, y[0]))**2))
    area = 0.5 * abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if area < 1e-6 or perim < 1e-6:
        return np.nan
    return float(np.clip(1 - 4 * np.pi * area / perim**2, 0, 1))


def convexity_score(coords):
    pts = np.array(coords, dtype=float)
    try:
        poly_area = Polygon(pts).area
        hull_area = ConvexHull(pts).volume
        if hull_area < 1e-6:
            return np.nan
        return float(np.clip(poly_area / hull_area, 0, 1))
    except Exception:
        return np.nan


def fractal_dimension(coords, n_sizes=6):
    pts = np.array(coords, dtype=float)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    w, h = mx - mn
    # FIX: coordinates are in voxel units (~1 unit each)
    # so w=149 means 149 voxels — well above threshold
    if w < 3 or h < 3:
        return np.nan
    gs = max(64, int(max(w, h)) + 4)
    scale = (gs - 4) / max(w, h)
    px = np.clip(((pts[:, 0] - mn[0]) * scale + 2).astype(int), 0, gs - 1)
    py = np.clip(((pts[:, 1] - mn[1]) * scale + 2).astype(int), 0, gs - 1)

    from skimage.draw import polygon as sk_poly
    from scipy.ndimage import binary_fill_holes, binary_erosion  # ← add binary_erosion here
    grid = np.zeros((gs, gs), dtype=bool)
    rr, cc = sk_poly(py, px, shape=grid.shape)
    grid[rr, cc] = True
    filled = binary_fill_holes(grid)
    eroded = binary_erosion(filled)          # ← add this line
    bnd = filled & ~eroded                   # ← replace the old bnd line
    if bnd.sum() < 10:
        return np.nan
        
    sizes = np.unique(
        np.floor(np.logspace(1, np.log10(gs // 2), n_sizes)).astype(int))
    sizes = sizes[sizes >= 2]
    counts = []
    for s in sizes:
        c = sum(1 for i in range(0, gs, s)
                for j in range(0, gs, s)
                if bnd[i:i+s, j:j+s].any())
        counts.append(c)
    counts = np.array(counts, dtype=float)
    valid = counts > 0
    if valid.sum() < 2:
        return np.nan
    return float(np.clip(
        np.polyfit(np.log(1.0 / sizes[valid]),
                   np.log(counts[valid]), 1)[0],
        1.0, 2.0))


# ── 4. Compute all metrics ────────────────────────────────────────────────────
def compute_metrics(features, run):
    records = []
    total = len(features)
    for i, feat in enumerate(features):
        if i % 3000 == 0:
            print(f"  [{run}] {i}/{total}...")
        try:
            props = feat.get("properties", {})
            cell_id = props.get("cell", i)
            coords = get_coords(feat)
            if coords is None or len(coords) < 4:
                continue
            records.append({
                "cell_id":     cell_id,
                "par":         perimeter_area_ratio(coords),
                "convexity":   convexity_score(coords),
                "fractal_dim": fractal_dimension(coords),
                "n_vertices":  len(coords),
            })
        except Exception as e:
            continue

    df = pd.DataFrame(records)
    n_valid = df["fractal_dim"].notna().sum()
    print(f"  [{run}] Complete — {n_valid}/{total} valid fractal dims")
    return df


# ── 5. Router ─────────────────────────────────────────────────────────────────
def assign_route(df):
    df = df.copy()
    df["flag_par"]  = df["par"]  > df["par"].quantile(0.75)
    df["flag_conv"] = df["convexity"] < df["convexity"].quantile(0.25)
    df["flag_fd"]   = df["fractal_dim"] > df["fractal_dim"].quantile(0.75)
    df["flags"] = (df["flag_par"].astype(int) +
                   df["flag_conv"].astype(int) +
                   df["flag_fd"].astype(int))
    df["route"] = df["flags"].apply(
        lambda x: "fractal" if x >= 2 else "euclidean")
    df["complexity"] = (
        df["par"].rank(pct=True) +
        (1 - df["convexity"]).rank(pct=True) +
        df["fractal_dim"].rank(pct=True)) / 3.0
    n = (df["route"] == "fractal").sum()
    print(f"  Router: {n}/{len(df)} ({100*n/len(df):.1f}%) fractal")
    return df


# ── 6. Figures ────────────────────────────────────────────────────────────────
def fig1_metrics(all_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Cell boundary complexity — three ProSeg compactness settings\n"
        "Sergio Salas mouse brain Xenium (31,482 cells)", fontsize=12)
    cols = [
        ("par",         "Perimeter-area ratio\n(0=circle, 1=irregular)"),
        ("convexity",   "Convexity (1=convex, 0=branched)"),
        ("fractal_dim", "Fractal dimension (1.0=smooth, 1.5=complex)"),
    ]
    for ax, (col, label) in zip(axes, cols):
        for run in RUNS:
            vals = all_metrics[run][col].dropna()
            ax.hist(vals, bins=50, alpha=0.55, color=RUN_COLORS[run],
                    label=f"c={COMPACTNESS[run]} med={vals.median():.3f}",
                    density=True)
            ax.axvline(vals.median(), color=RUN_COLORS[run],
                       lw=1.5, ls="--")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = OUT / "fig1_metric_distributions.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def fig2_routing(all_routed):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(
        "Geometry router assignment by compactness\n"
        "Sergio Salas mouse brain Xenium", fontsize=12)
    for ax, run in zip(axes, RUNS):
        df = all_routed[run]
        nf = (df["route"] == "fractal").sum()
        ne = (df["route"] == "euclidean").sum()
        ax.bar(["Euclidean", "Fractal"], [ne, nf],
               color=["#1DB87A", "#534AB7"], alpha=0.85,
               edgecolor="white")
        ax.set_title(
            f"Compactness={COMPACTNESS[run]}\n"
            f"Fractal: {nf:,} ({100*nf/len(df):.1f}%)", fontsize=10)
        ax.set_ylabel("Cells", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = OUT / "fig2_routing_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def fig3_surface_area():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Cell surface area — ProSeg compactness comparison\n"
        "Sergio Salas mouse brain Xenium", fontsize=12)
    for ax, run in zip(axes, RUNS):
        meta = pd.read_csv(RESULTS / f"run_{run}" / "cell_metadata.csv")
        sa = meta["surface_area"].dropna()
        ax.hist(sa, bins=60, color=RUN_COLORS[run],
                alpha=0.85, density=True)
        ax.axvline(sa.median(), color="black", lw=1.5, ls="--")
        ax.set_title(
            f"Compactness={COMPACTNESS[run]}\n"
            f"Median SA={sa.median():.0f}", fontsize=10)
        ax.set_xlabel("Surface area (voxels)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = OUT / "fig3_surface_area.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def fig4_vertices_vs_complexity(all_routed):
    """
    Plot n_vertices vs complexity score — shows that cells
    with more boundary detail have higher complexity scores.
    Also shows routing decision in feature space.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Boundary vertices vs complexity score by route\n"
        "Sergio Salas mouse brain Xenium", fontsize=12)
    for ax, run in zip(axes, RUNS):
        df = all_routed[run].dropna(subset=["fractal_dim", "complexity"])
        sample = df.sample(min(3000, len(df)), random_state=42)
        for route, color in [("euclidean", "#1DB87A"), ("fractal", "#534AB7")]:
            s = sample[sample["route"] == route]
            ax.scatter(s["n_vertices"], s["complexity"],
                       c=color, alpha=0.3, s=5,
                       label=route, rasterized=True)
        ax.set_xlabel("N boundary vertices", fontsize=9)
        ax.set_ylabel("Complexity score", fontsize=9)
        ax.set_title(f"Compactness={COMPACTNESS[run]}", fontsize=10)
        ax.legend(fontsize=8, markerscale=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = OUT / "fig4_vertices_complexity.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def fig5_fractal_vs_convexity(all_routed):
    """
    Key result figure: fractal dimension vs convexity coloured by route.
    Shows the decision space of the geometry router.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Fractal dimension vs convexity — geometry router decision space\n"
        "Sergio Salas mouse brain Xenium", fontsize=12)
    for ax, run in zip(axes, RUNS):
        df = all_routed[run].dropna(subset=["fractal_dim", "convexity"])
        sample = df.sample(min(4000, len(df)), random_state=42)
        for route, color in [("euclidean", "#1DB87A"), ("fractal", "#534AB7")]:
            s = sample[sample["route"] == route]
            ax.scatter(s["fractal_dim"], s["convexity"],
                       c=color, alpha=0.35, s=6,
                       label=route, rasterized=True)
        ax.set_xlabel("Fractal dimension", fontsize=9)
        ax.set_ylabel("Convexity", fontsize=9)
        ax.set_title(f"Compactness={COMPACTNESS[run]}", fontsize=10)
        ax.legend(fontsize=8, markerscale=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = OUT / "fig5_fractal_vs_convexity.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


# ── 7. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ProSeg analysis + geometry router — FIXED VERSION")
    print("Sergio Salas mouse brain Xenium dataset")
    print("=" * 60)

    print("\n--- Fig 3: Surface area (fast) ---")
    fig3_surface_area()

    all_metrics = {}
    all_routed  = {}

    for run in RUNS:
        print(f"\n--- {run} (compactness={COMPACTNESS[run]}) ---")
        features = load_polygons(run)
        df = compute_metrics(features, run)
        df = assign_route(df)
        all_metrics[run] = df
        all_routed[run]  = df
        df.to_csv(OUT / f"router_{run}.csv", index=False)
        print(f"  Saved router_{run}.csv")

    print("\n--- Fig 1: Metric distributions ---")
    fig1_metrics(all_metrics)

    print("\n--- Fig 2: Routing comparison ---")
    fig2_routing(all_routed)

    print("\n--- Fig 4: Vertices vs complexity ---")
    fig4_vertices_vs_complexity(all_routed)

    print("\n--- Fig 5: Fractal vs convexity (key result) ---")
    fig5_fractal_vs_convexity(all_routed)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for run in RUNS:
        df = all_routed[run]
        nf = (df["route"] == "fractal").sum()
        fd = df["fractal_dim"].median()
        cv = df["convexity"].median()
        pa = df["par"].median()
        meta = pd.read_csv(RESULTS / f"run_{run}" / "cell_metadata.csv")
        sa = meta["surface_area"].median()
        print(f"  {run:8s} c={COMPACTNESS[run]}"
              f"  fractal={100*nf/len(df):.1f}%"
              f"  FD={fd:.3f}"
              f"  conv={cv:.3f}"
              f"  PAR={pa:.3f}"
              f"  SA={sa:.0f}")

    print(f"\nAll outputs: {OUT}")
    print("=" * 60)
