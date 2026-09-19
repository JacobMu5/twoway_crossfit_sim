"""Generate the heterogeneous-cell GMM illustration data."""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
 
SEED = 2026
N_ROWS = 40
N_COLS = 35
N_FEATURES = 5
THETA_TRUE = 1.0
 
 
def _component(rng: np.random.Generator, row_id: np.ndarray,
               col_id: np.ndarray, cell_index: np.ndarray) -> np.ndarray:
    """Draw a standardized row + column + cell + unit innovation."""
    n_units = len(row_id)
    return (
        rng.normal(size=N_ROWS)[row_id - 1]
        + rng.normal(size=N_COLS)[col_id - 1]
        + rng.normal(size=N_ROWS * N_COLS)[cell_index]
        + rng.normal(size=n_units)
    ) / 2.0
 
 
def generate_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cell_row = np.repeat(np.arange(1, N_ROWS + 1), N_COLS)
    cell_col = np.tile(np.arange(1, N_COLS + 1), N_ROWS)
    cell_size = rng.integers(1, 5, size=N_ROWS * N_COLS)
 
    cell_index = np.repeat(np.arange(N_ROWS * N_COLS), cell_size)
    row_id = cell_row[cell_index]
    col_id = cell_col[cell_index]
    unit_id = np.concatenate([np.arange(1, size + 1) for size in cell_size])
    n_units = len(row_id)
 
    x_row = rng.normal(size=(N_ROWS, N_FEATURES))
    x_col = rng.normal(size=(N_COLS, N_FEATURES))
    x_cell = rng.normal(size=(N_ROWS * N_COLS, N_FEATURES))
    x_unit = rng.normal(size=(n_units, N_FEATURES))
    x = (
        x_row[row_id - 1]
        + x_col[col_id - 1]
        + x_cell[cell_index]
        + x_unit
    ) / 2.0
 
    u = _component(rng, row_id, col_id, cell_index)
    v = 0.60 * u + 0.80 * _component(rng, row_id, col_id, cell_index)
    z1_innovation = _component(rng, row_id, col_id, cell_index)
    z2_innovation = _component(rng, row_id, col_id, cell_index)
 
    x1, x2, x3, x4, x5 = x.T
    m_z1 = 0.40 * x1 - 0.30 * x2 + 0.25 * x3**2 + 0.15 * x1 * x4
    m_z2 = -0.20 * x1 + 0.35 * x4 + 0.20 * x2 * x5 - 0.15 * x3**2
    h_d = 0.45 * x1 + 0.25 * x2 * x3 - 0.20 * x4**2 + 0.10 * x5
    g_y = 0.60 * x1 - 0.40 * x2 + 0.25 * x3**2 + 0.20 * x1 * x2 - 0.15 * x5**2
 
    z1 = m_z1 + z1_innovation
    z2 = m_z2 + z2_innovation
    d = 0.60 * z1 + 0.50 * z2 + h_d + v
    y = THETA_TRUE * d + g_y + u
 
    data = pd.DataFrame({
        "row_id": row_id,
        "col_id": col_id,
        "unit_id": unit_id,
        "cell_size": cell_size[cell_index],
    })
    for index in range(N_FEATURES):
        data[f"x{index + 1}"] = x[:, index]
    data[["z1", "z2", "d", "y"]] = np.column_stack([z1, z2, d, y])
 
    if data.duplicated(["row_id", "col_id", "unit_id"]).any():
        raise RuntimeError("row_id, col_id, and unit_id must identify observations")
    observed_sizes = data.groupby(["row_id", "col_id"]).size().to_numpy()
    if not np.array_equal(observed_sizes, cell_size):
        raise RuntimeError("cell-size bookkeeping failed")
    if data.isna().any().any():
        raise RuntimeError("generated data contain missing values")
    return data
 
 
def main() -> None:
    default_output = Path(__file__).resolve().parent / "synthetic_gmm.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
 
    data = generate_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, float_format="%.12g")
    print(
        f"Wrote {len(data)} units in {N_ROWS * N_COLS} cells "
        f"({N_ROWS} x {N_COLS}; cell sizes 1--4) to {args.output}."
    )
 
 
if __name__ == "__main__":
    main()