"""
PLIV paper reconstruction and seed injection into
dgps/package_pliv + reconstruction to get the true functions.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
 
 
@dataclass(frozen=True)
class PLIVSample:
    x: np.ndarray
    d: np.ndarray
    y: np.ndarray
    z: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    n_rows: int
    n_cols: int
    theta0: float
    # True conditional means, used for oracle estimation and prediction checks.
    mean_y: np.ndarray | None = None
    mean_d: np.ndarray | None = None
    mean_z: np.ndarray | None = None
 
 
class ChenChiangPLIV:
    """Chen--Chiang v4, Section 5, with explicit Gaussian calibration.
 
    Unspecified laws/correlations and my mean shift are added.
    """
 
    def __init__(self, p=3, rho_x=.5, rho_uv=.5, theta0=1., x2_row_mean=0.):
        self.p, self.rho_x, self.rho_uv, self.theta0 = p, rho_x, rho_uv, theta0
        self.x2_row_mean = float(x2_row_mean)
 
    @property
    def true_theta(self):
        return self.theta0
 
    @property
    def name(self):
        base = f'chen_chiang_pliv_p{self.p}'
        return base if self.x2_row_mean == 0 else f'{base}_mu{self.x2_row_mean:g}'
 
    def sample(self, n_rows, n_cols, seed=None):
        rng = np.random.default_rng(seed)
        n = n_rows * n_cols
        rows = np.repeat(np.arange(n_rows), n_cols)
        cols = np.tile(np.arange(n_cols), n_rows)
        def xdraw(size):
            common = rng.standard_normal((size, 1))
            individual = rng.standard_normal((size, self.p))
            return np.sqrt(self.rho_x)*common + np.sqrt(1-self.rho_x)*individual
        x = xdraw(n_rows)[rows] + xdraw(n_cols)[cols] + xdraw(n)
        # Shift X2 in place; no new RNG draws, so mu=0 reproduces the archived calibration.
        x[:, 1] += self.x2_row_mean
        def shocks(size):
            u = rng.standard_normal(size)
            v = self.rho_uv*u + np.sqrt(1-self.rho_uv**2)*rng.standard_normal(size)
            e = rng.standard_normal(size)
            return u, v, e
        row_shocks, col_shocks, cell_shocks = shocks(n_rows), shocks(n_cols), shocks(n)
        u, v, e = [row_shocks[k][rows] + col_shocks[k][cols] + cell_shocks[k]
                   for k in range(3)]
        mean_z = x[:, 1]*x[:, 2]
        mean_d = .5*mean_z + .2*(x[:, 0]+x[:, 1])**2
        g = np.sin(x[:, 0]+x[:, 1]) + .5*x[:, 2:].sum(axis=1)
        mean_y = self.theta0*mean_d + g
        z = mean_z + e
        d = .5*z + .2*(x[:, 0]+x[:, 1])**2 + v
        y = self.theta0*d + g + u
        return PLIVSample(x, d, y, z, rows, cols, n_rows, n_cols, self.theta0,
                          mean_y, mean_d, mean_z)
 
 
class PackagePLIV:
    """Use Z1 from unchanged fullsampleDML-python v1.0.0 data; vary only SEED.
 
    Source: examples/data/generate_synthetic.py (MIT; see PACKAGE_LICENSE).
    """
 
    true_theta = 1.0
    name = "package_pliv_z1"
 
    def sample(self, n_rows, n_cols, seed=None):
        from types import FunctionType
        from dgps import package_pliv as source
 
        if (n_rows, n_cols) != (source.N_ROWS, source.N_COLS):
            raise ValueError("The unchanged package generator requires a 40 x 35 grid")
        # Isolated globals change the seed without editing or mutating the source.
        generate = FunctionType(source.generate_data.__code__,
                                {**source.generate_data.__globals__,
                                 "SEED": source.SEED if seed is None else seed})
        frame = generate()
        x = frame[[f"x{k}" for k in range(1, 6)]].to_numpy()
        x1, x2, x3, x4, x5 = x.T
        mz1 = .40*x1 - .30*x2 + .25*x3**2 + .15*x1*x4
        mz2 = -.20*x1 + .35*x4 + .20*x2*x5 - .15*x3**2
        hd = .45*x1 + .25*x2*x3 - .20*x4**2 + .10*x5
        gy = .60*x1 - .40*x2 + .25*x3**2 + .20*x1*x2 - .15*x5**2
        md = .60*mz1 + .50*mz2 + hd
        return PLIVSample(x, frame.d.to_numpy(), frame.y.to_numpy(),
                          frame.z1.to_numpy(), frame.row_id.to_numpy()-1,
                          frame.col_id.to_numpy()-1, n_rows, n_cols,
                          self.true_theta, self.true_theta*md+gy, md, mz1)