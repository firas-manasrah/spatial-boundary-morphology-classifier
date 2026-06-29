# spatial-boundary-morphology-classifier

Classifies segmentation boundary polygons into morphological groups
using five geometric metrics computed from polygon shape alone.
No gene expression data required.

Compatible with 10x Xenium, CosMx, MERSCOPE, Molecular Cartography,
and any tool producing polygon boundaries in parquet format.

## Note

Group labels (Compact, Irregular, Elongated, Complex, Large) are
descriptive geometric categories. Correspondence to biological cell
types has not been validated and requires independent confirmation.

## Metrics

- Perimeter-area ratio (PAR): boundary irregularity (0=circle, 1=irregular)
- Convexity: polygon area / convex hull area (1=perfectly convex)
- Fractal dimension: box-counting boundary complexity (1.0-2.0)
- Elongation: major axis / minor axis ratio (1=round)
- Area: polygon area in coordinate units

## Groups

- Compact: round, convex, regular boundaries
- Irregular: complex, non-convex boundaries
- Elongated: high aspect ratio boundaries
- Complex: both irregular and elongated
- Large: area significantly above tissue median

## Usage

pip install -r requirements.txt

python classifier.py cell_boundaries.parquet output.csv

Options:
  --id-col        polygon ID column (default: cell_id)
  --x-col         x coordinate column (default: vertex_x)
  --y-col         y coordinate column (default: vertex_y)
  --large-multiplier  area threshold multiplier (default: 5.0)
  --elongation-sigma  elongation threshold sigma (default: 2.0)

## Input

Parquet file with one row per vertex: polygon_id, vertex_x, vertex_y
Matches 10x Xenium cell_boundaries.parquet format directly.

## Status

Early-stage research tool. Developed during research internship at
BIH/Charite Berlin, Eils-Ishaque Computational Oncology Group, 2026.

Author: Firas Manasrah — github.com/firas-manasrah

## Group Summary

| Group | Shape | Key metric |
|-------|-------|------------|
| Compact | Round, convex, regular | Low PAR, high convexity |
| Irregular | Complex, non-convex | High fractal dimension |
| Elongated | Long and narrow | High elongation ratio |
| Complex | Irregular and elongated | High fractal + high elongation |
| Large | Much bigger than average | Area > 5x tissue median |
