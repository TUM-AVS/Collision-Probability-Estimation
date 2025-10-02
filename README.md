# Collision Probability Estimation

A Python library for calculating collision probabilities between objects in space, supporting both individual state analysis and trajectory-based predictions.

## Features

- **State-based collision probability**: Calculate overlap probability for individual orbital states
- **Trajectory-based collision probability**: Analyze collision risk along complete orbital trajectories
- Efficient algorithms optimized for space surveillance applications
- Easy-to-use API with comprehensive examples

## Installation

### From source
```bash
git clone https://xxxx/Collision-Probability-Estimation.git
cd Collision-Probability-Estimation
pip install -e .
```


## Quick Start

### Individual State Collision Probability

```python
from coll_state_overlap import point_collvol_analytic

# Example usage for state-based collision probability
state1 = {...}  # Object 1 state vector (x,y,heading) at each corner of polygon
state2 = {...}  # Object 2 state vector (x,y,heading) at each corner of polygon
covariance1 = {...}  # Object 1 covariance matrix (3x3)
covariance2 = {...}  # Object 2 covariance matrix (3x3)

probability = point_collvol_analytic(state1, state2, covariance1, covariance2)
print(f"Collision probability: {probability}")
```

### Trajectory-based Collision Probability

```python
from coll_trajectories import boundary_crossing

# Example usage for trajectory-based analysis
corners1 = {}  # Object 1 polygon (x,y,heading) at each corner 
corners2 = {}  # Object 2 polygon (x,y,heading) at each corner 
trajectory1 = {...}  # Object 1 trajectory nx(x,y,theta,v)
trajectory2 = {...}  # Object 2 trajectory nx(x,y,theta,v)
covariance1 = {...}  # Object 1 covariance matrix (nx4x4)
covariance2 = {...}  # Object 2 covariance matrix (nx4x4)

probability, _ = boundary_crossing(corners1, trajectory1, trajectory2, covariance1, covariance2, corners2)
print(f"Trajectory collision probability: {probability}")
```

## Methods

### 1. Individual State Methods (`coll_state_overlap.py`)

The following collision probability methods are implemented for individual states:

Following methods are implemented:
1) **Monte Carlo sampling** for **polygon-polygon** collision probability.
2) **Inclusion-Exclusion principle** for rectangle (axis-aligned).
3) **Monte Carlo sampling** for **point-in-collision-volume** collision probability.
4) **Analytic solution** for point-in-polygon collision probability.

### 2. Trajectory Methods (`coll_trajectories.py`)

Trajectory-based collision probability methods include:

1) **Monte Carlo Sampling** for **complete trajectories**.
2) **Boundary crossing method** for complete trajectories.
3) **Monte Carlo sampling** for **each time step**.
4) **Analytic solution** for point-in-polygon collision probability at **each time step**.

## Usage Examples

### Running the Scripts

Execute the provided example scripts:

```bash
# Run state-based collision probability examples
python scripts_overlap.py

# Run trajectory-based collision probability examples  
python scripts_coll_traj.py
```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the [LGPL-3.0 License](LICENSE).

## References

xxx TBD after review

## Contact

xxx TBD after review