import numpy as np



def build_rotation_matrices(theta):
    """Vectorized rotation matrix construction"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rot = np.zeros((*theta.shape, 3, 3), dtype=theta.dtype)
    rot[..., 0, 0] = cos_theta
    rot[..., 0, 1] = sin_theta
    rot[..., 1, 0] = -sin_theta
    rot[..., 1, 1] = cos_theta
    rot[..., 2, 2] = 1.0

    return rot


def compute_max_diagonal(corners):
    """Compute maximum diagonal distance efficiently"""
    diffs = np.roll(corners, -1, axis=-2) - corners
    return np.maximum.reduce(np.hypot(diffs[..., 0], diffs[..., 1]), axis=-1)


# rotate velocity components in 2D:
def transform_velocities_and_covs(trajs, trajs_covs):
    """Helper function to avoid code duplication"""
    theta = trajs[..., 2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    v = trajs[..., -1]

    vx = v * cos_theta
    vy = v * sin_theta
    trajs_new = np.concatenate([trajs[..., :-1], vx[..., None], vy[..., None]], axis=-1)
    assert trajs_new.shape[-1] == 5, (
        f"Expected trajs_new to have last dimension of size 5, got {trajs_new.shape[-1]}, Check difference!"
    )
    # Build transformation matrix H
    H = np.zeros((*trajs_new.shape, 4), dtype=trajs.dtype)
    np.einsum("...ii->...i", H[..., :3, :3])[...] = 1.0
    H[..., 3, 3] = cos_theta
    H[..., 3, 2] = -v * sin_theta
    H[..., 4, 3] = sin_theta
    H[..., 4, 2] = v * cos_theta

    covs_new = H @ trajs_covs @ H.swapaxes(-1, -2)
    return trajs_new, covs_new


def collision_polygon(poly_corners_1: np.ndarray, poly_corners_2: np.ndarray):
    """
    Compute and visualize the collision polygon between two polygons.
    Around polygon1

    Parameters:
        polygon1 (shapely.geometry.Polygon): The first polygon.
        polygon2 (shapely.geometry.Polygon): The second polygon.
    """
    assert poly_corners_1.shape[-1] >= 2 and poly_corners_2.shape[-1] >= 2, (
        "Polygons must be Nx2 arrays of corner coordinates."
    )
    assert poly_corners_2.shape[-2] > 2 or poly_corners_1.shape[-2] > 2, (
        "At least one polygon must have at least 3 corners."
    )

    if poly_corners_2.shape[-2] < 3:
        return poly_corners_1, poly_corners_2
    if poly_corners_1.shape[-2] < 3:
        return poly_corners_2, poly_corners_1

    # Center polygons around their centroids, will be origin for Minkowski sum
    poly1_center = np.mean(poly_corners_1, axis=-2, keepdims=True)
    poly2_center = np.mean(poly_corners_2, axis=-2, keepdims=True)

    poly2_shifted = poly_corners_2 - poly2_center

    # Compute Minkowski sum with polygon1's center preserved
    edges1 = np.roll(poly_corners_1, -1, axis=-2) - poly_corners_1
    edges2 = np.roll(poly2_shifted, -1, axis=-2) - poly2_shifted
    edges = np.concatenate([edges1, edges2], axis=-2)
    angles1 = (np.arctan2(edges1[..., 1], edges1[..., 0]) + 2 * np.pi) % (2 * np.pi)
    angles2 = (np.arctan2(edges2[..., 1], edges2[..., 0]) + 2 * np.pi) % (2 * np.pi)
    angles = np.concatenate([angles1, angles2], axis=-1)

    # Sort edges by angle
    sorted_indices = np.argsort(angles, axis=-1)
    # Find leftmost among vertices with minimum y-coordinate to start the convex hull and move to origin
    min_y1_coord = np.min(poly_corners_1[..., 1], axis=-1, keepdims=True)  # [:,np.newaxis]
    min_y2_coord = np.min(poly2_shifted[..., 1], axis=-1, keepdims=True)
    # Find indices of vertices with minimum y-coordinate for each polygon
    min_y_mask1 = poly_corners_1[..., 1] == min_y1_coord
    min_y_mask2 = poly2_shifted[..., 1] == min_y2_coord

    # Among vertices with min y, find the one with minimum x for each polygon
    # Use masked array approach to handle variable number of bottom vertices
    leftmost_x1 = np.where(min_y_mask1, poly_corners_1[..., 0], np.inf)
    leftmost_x2 = np.where(min_y_mask2, poly2_shifted[..., 0], np.inf)

    # Get indices of leftmost bottom vertices
    leftmost_idx1 = np.argmin(leftmost_x1, axis=-1)
    leftmost_idx2 = np.argmin(leftmost_x2, axis=-1)

    # Extract the leftmost bottom vertices using advanced indexing
    # Create batch indices for all dimensions except the last two
    batch_shape = poly_corners_1.shape[:-2]  # Shape of all batch dimensions
    batch_indices = np.meshgrid(*[np.arange(s) for s in batch_shape], indexing="ij")
    batch_indices = np.broadcast_arrays(*batch_indices)

    # Extract the leftmost bottom vertices using advanced indexing
    leftmost_bottom1 = poly_corners_1[(*batch_indices, leftmost_idx1)]
    leftmost_bottom2 = poly2_shifted[(*batch_indices, leftmost_idx2)]
    minkowski_coords = [leftmost_bottom1 + leftmost_bottom2]
    for idx in sorted_indices[..., :-1].T:
        minkowski_coords.append(minkowski_coords[-1] + edges[(*batch_indices, idx.T)])
    collision_volume = np.stack(minkowski_coords, axis=-2)
    # relative angle between the two polygons
    collision_volume[..., -1] = poly2_center[..., -1] - poly1_center[..., -1]

    return collision_volume, poly2_center


def point_to_polygon_boundary_dist(points, polygon):
    """
    Minimum distance from points to polygon boundary edges.
    Args:
        points:  (..., 2)
        polygon: (..., V, 2)
    Returns:
        min_dist: (...,)
        closest:  (..., 2)  nearest point on boundary
    """
    # edges: from vertex i to vertex i+1 (wrap around)
    v0 = polygon                                        
    v1 = np.roll(polygon, shift=-1, axis=-2)            
    edge = v1 - v0                                     
    edge_len_sq = (edge ** 2).sum(axis=-1)            

    # project point onto each edge, clamp t in [0,1]
    p = points[..., :]                    
    t = ((p - v0) * edge).sum(axis=-1) / (edge_len_sq + 1e-12)  
    t = np.clip(t, 0.0, 1.0)                            

    closest_pts = v0 + t[..., np.newaxis] * edge     
    diff = p - closest_pts                             
    dists = np.linalg.norm(diff, axis=-1)               

    min_idx = dists.argmin(axis=-1)   
    min_dist = diff[np.arange(len(diff)),min_idx]
    return min_dist 

def simple_prediction(
    initial_state: np.ndarray,
    initial_cov: np.ndarray,
    dt: float,
    num_steps: int,
):
    """
    Simple constant velocity model prediction with EKF for position and orientation.
    State: [x, y, theta, v]
    Cov: 4x4 covariance matrix
    Args:
        initial_state: np.array([x, y, theta, v])
        initial_cov: 4x4 np.array, initial covariance matrix
        dt: float, time step
    """

    if initial_state.ndim == 1:
        initial_state = initial_state[np.newaxis, :]
    if initial_cov.ndim == 2:
        initial_cov = initial_cov[np.newaxis, :, :]
    # process noise
    q = 0**2
    sigma_theta = 0.0
    Q = np.zeros((4, 4))
    # Position x block
    Q[0, 0] = 0.25 * dt**4 * q
    Q[0, 3] = 0.5 * dt**3 * q
    Q[3, 0] = 0.5 * dt**3 * q
    Q[3, 3] = dt**2 * q
    # Position y block
    Q[1, 1] = 0.25 * dt**4 * q
    Q[1, 3] = 0.5 * dt**3 * q
    Q[3, 1] = 0.5 * dt**3 * q
    # Q[4, 4] = dt**2 * q
    # Orientation noise (simple random walk)
    Q[2, 2] = sigma_theta**2 * dt

    traj = np.zeros((*initial_state.shape[:-1], num_steps + 1, initial_state.shape[-1]))
    covs = np.zeros(
        (
            *initial_state.shape[:-1],
            num_steps + 1,
            initial_state.shape[-1],
            initial_state.shape[-1],
        )
    )
    traj[..., 0, :] = initial_state
    covs[..., 0, :, :] = initial_cov

    for t in range(1, num_steps + 1):
        state = traj[..., t - 1, :]
        cov = covs[..., t - 1, :, :]
        theta = state[..., 2]
        v = state[..., 3]
        R = np.tile(np.eye(2), (*state.shape[:-1], 1, 1))
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        R[..., 0, 0] = costheta
        R[..., 0, 1] = -sintheta
        R[..., 1, 0] = sintheta
        R[..., 1, 1] = costheta

        zero = np.zeros_like(state[..., 1])
        next_state = state + dt * np.stack([v * costheta, v * sintheta, zero, zero], axis=-1)
        F = np.tile(np.eye(state.shape[-1]), (*state.shape[:-1], 1, 1))
        F[..., 0, 2] = -dt * v * sintheta
        F[..., 0, 3] = dt * costheta
        F[..., 1, 2] = dt * v * costheta
        F[..., 1, 3] = dt * sintheta

        next_cov = np.einsum("...ij,...jk,...lk->...il", F, cov, F) + Q
        traj[..., t, :] = next_state
        covs[..., t, :, :] = next_cov

    return traj, covs


def convolve_distributions(
    mus1: np.ndarray = None,
    devs1: np.ndarray = None,
    mus2: np.ndarray = None,
    devs2: np.ndarray = None,
    dist="Gaussian",
):
    """
    Get the relative distributions between two objects by convolution two Gaussian distributions.

    Parameters:
        mean1 (ndarray): Mean of the first distribution.
        cov1 (ndarray): Covariance of the first distribution.
        mean2 (ndarray): Mean of the second distribution.
        cov2 (ndarray): Covariance of the second distribution.

    Returns:
        tuple: Mean and covariance of the convolved distribution.
    """

    def stable(mu, dev, a, b):
        """
        Stable distribution convolution.
        Gaussian, Cauchy, Levy are stable distributions.
        Parameters:
            mu (list): List of means.
            sig (list): List of standard deviations.
            a (float): Stability parameter (0 < a <= 2).
            b (list): List of skewness parameters (-1 <= b <= 1).
        Returns:
            tuple: Mean and standard deviation of the resulting distribution.
        """
        assert 0 < a <= 2, f"a must be in (0,2], got {a}"
        assert all(-1 <= i <= 1 for i in b), f"all b must be in [-1,1], got {b}"

        rel_mu = sum(mu)
        dev_pow = [d**a for d in dev]
        rel_dev = sum(dev_pow) ** (1 / a)
        return rel_mu, rel_dev

    mus1 = mus1 if mus1 is not None else np.array([0])
    devs1 = devs1 if devs1 is not None else np.array([0])
    mus2 = mus2 if mus2 is not None else np.array([0])
    devs2 = devs2 if devs2 is not None else np.array([0])
    match dist:
        case "Gaussian" | "Normal" | "Cauchy":
            mus = [mus2, -mus1]
            sigs = [devs2, devs1]

            a = 1
            b = [0, 0]
            rel_mean, rel_cov = stable(mus, sigs, a, b)
            mean_batch_shape = rel_mean.shape[:-1]
            cov_batch_shape = rel_cov.shape[:-2]
            batch_shape = np.broadcast_shapes(mean_batch_shape, cov_batch_shape)
            rel_mean = np.broadcast_to(rel_mean, (*batch_shape, rel_mean.shape[-1]))
            rel_cov = np.broadcast_to(rel_cov, (*batch_shape, *rel_cov.shape[-2:]))

            return rel_mean, rel_cov
        case "Levy":
            mu = [mus2, -mus1]
            sig = [devs2, devs1]
            a = 0.5
            b = [1, 1]
            raise NotImplementedError("Levy distribution not implemented yet.")
            rel_mean, rel_cov = stable(mu, sig, a, b)
            return rel_mean, rel_cov
        case _:
            raise NotImplementedError(f"Distribution {dist} not implemented")

