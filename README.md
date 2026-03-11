# Collision Probability Estimation

A Python library for calculating collision probabilities between uncertain objects, supporting both individual state analysis and trajectory-based predictions. Implements analytic, Monte Carlo, and distance-based methods with optional PyTorch acceleration.

## Features

- **State-based collision probability**: Collision probability for individual states using multiple methods
- **Trajectory-based collision probability**: Analyze collision risk along complete trajectories over a time horizon
- NumPy/SciPy core implementation — no GPU required
- Optional PyTorch backend for GPU acceleration
- Orientation-aware collision volume computation
- Easy-to-use API and comprehensive exemplaric benchmark scripts

## Installation

The Code is tested with Python ≥ 3.11 and Python ≤ 3.14. 

Uses [`uv`](https://github.com/astral-sh/uv) as recommended build backend.

### From source

```bash
git clone https://github.com/TUM-AVS/Collision-Probability-Estimation.git
cd collision-probability-estimation
```

### Using `uv` (recommended)

Install `uv` if not already available:
```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

Default install (NumPy/SciPy only):
```bash
uv sync
```

With optional extras:
```bash
# With PyTorch backend
uv sync --extra use_torch

# With plotting support
uv sync --extra plot

# With both
uv sync --extra use_torch --extra plot
```

With a specific Python version:
```bash
uv sync --python 3.12
# or pin permanently
echo "3.12" > .python-version && uv sync
```

### Using `pip`

Default install (NumPy/SciPy only):
```bash
pip install .
```

With optional extras:
```bash
# With PyTorch backend
pip install ".[use_torch]"

# With plotting support
pip install ".[plot]"

# With both
pip install ".[use_torch,plot]"
```

### Summary of optional extras

| Extra | Installs | Required for |
|---|---|---|
| *(none)* | `numpy`, `scipy` | Core NumPy methods |
| `use_torch` | `torch` | PyTorch-accelerated methods, `--use_torch` flag |
| `plot` | `matplotlib` | `--plot` flag in benchmark scripts |

## Quick Start

### Individual State Collision Probability

```python
from collision_probability_estimation.np_impl.coll_state_overlap import point_collvol_analytic

# Object polygons: (n_corners, 3) arrays of (x, y, heading) — counter-clockwise
obs1 = ...   # shape (n_corners, 3)
obs2 = ...   # shape (n_corners, 3)
cov  = ...   # relative covariance matrix (3x3)

probability = point_collvol_analytic(obs1, obs2, cov)
print(f"Collision probability: {probability}")
```

### Trajectory-based Collision Probability

```python
from collision_probability_estimation.np_impl.coll_trajectories import boundary_crossing

# corners: (n_corners, 3)  —  trajectories: (num_steps, 4) with state (x, y, theta, v)
# covariances: (num_steps, 4, 4)

probability, p_cumulative = boundary_crossing(
    corners_obs1=corners1, trajs_obs1=traj1, trajs_obs2=traj2,
    trajs_obs1_covs=cov1,  trajs_obs2_covs=cov2,
    corners_obs2=corners2, dt=0.1,
)
print(f"Trajectory collision probability: {probability}")
```

## Methods

### 1. Individual State Methods (`coll_state_overlap.py`)

| # | Method | Function |
|---|--------|----------|
| 1 | Monte Carlo sampling — polygon-polygon | `poly_poly_mc_sampling` |
| 2 | Inclusion-Exclusion (rect) using bivariate normal CDF | `point_rect_incl_excl` |
| 3 | Monte Carlo sampling — point-in-collision-volume | `point_collvol_mc_sampling` |
| 4 | Analytic solution — point-in-collision-volume | `point_collvol_analytic` |
| 5 | Mahalanobis distance-based collision probability | `mahalanobis_distance` |
| 6 | Corner Mahalanobis (boundary distance) | `mahalanobis_distance(..., use_corners=True)` |

### 2. Trajectory Methods (`coll_trajectories.py`)

| # | Method | Function |
|---|--------|----------|
| 1 | Monte Carlo Sampling — complete trajectories | `point_collvol_mc_traj_sampling` |
| 2 | Boundary crossing — complete trajectories (analytic) | `boundary_crossing` |
| 3 | Monte Carlo sampling — per time step | `point_collvol_mc_state_sampling` |
| 4 | Analytic solution — per time step | `traj_collvol_analytic` |
| 5 | Inclusion-Exclusion (rect) — per time step | `traj_rect_incl_excl` |
| 6 | Mahalanobis distance — per time step | `traj_mahalanobis_distance` |
| 7 | Corner Mahalanobis (boundary distance) — per time step | `traj_corner_mahalanobis_distance` |

All trajectory methods accept `with_orientation: bool` to enable orientation-aware collision volume computation.

## Usage Examples

### Running the Benchmark Scripts

The package exposes two entry-point scripts:

```bash
# State-based collision probability benchmark
uv run state_coll_prob

# Trajectory-based collision probability benchmark
uv run traj_coll_prob

# With optional flags
uv run traj_coll_prob --use_torch            # include PyTorch methods
uv run --extra plot traj_coll_prob --plot    # show probability-over-time plots
uv run --extra plot state_coll_prob --plot   # show overlap visualizations
```

Both scripts print a formatted results table with method name, orientation flag, backend, collision probability, and wall-clock time.



## License

This project is licensed under the [LGPL-3.0 License](LICENSE).

## References

```bibtex
@article{kaufeld2025preciseefficientcollisionprediction,
      title={Precise and Efficient Collision Prediction under Uncertainty in Autonomous Driving}, 
      author={Marc Kaufeld and Johannes Betz},
      year={2025},
      eprint={2510.05729},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.05729}, 
}
```
## Contact

[Marc Kaufeld](mailto:marc.kaufeld@tum.de),
Professorship Autonomous Vehicle Systems,
School of Engineering and Design,
Technical University of Munich,
85748 Garching,
Germany

[Johannes Betz](mailto:johannes.betz@tum.de),
Professorship Autonomous Vehicle Systems,
School of Engineering and Design,
Technical University of Munich,
85748 Garching,
Germany