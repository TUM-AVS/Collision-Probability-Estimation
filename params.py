import numpy as np

# seed for reproducibility
SEED = 42
# quadrature points and weights for heading integration
THETA_NSTD = 4  # number of std to cover
_N_QUAD_THETA = 101
NODES_THETA, WEIGHTS_THETA = np.polynomial.legendre.leggauss(_N_QUAD_THETA)
# quadrature points and weights for xy integration
_N_QUAD_XY = 101
NODES_XY, WEIGHTS_XY = np.polynomial.legendre.leggauss(_N_QUAD_XY)
