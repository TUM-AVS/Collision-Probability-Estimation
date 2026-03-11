import numpy as np

import torch


DTYPE = torch.float32

def get_device(device=None) -> torch.device:
    """Get the appropriate device (GPU if available, else CPU)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    return device


def to_torch(arr, device: torch.device) -> torch.Tensor:
    """Convert numpy array or tensor to torch tensor on specified device."""
    return arr.to(device) if isinstance(arr, torch.Tensor) else torch.from_numpy(arr).to(device=device, dtype=DTYPE)


def build_rotation_matrices(theta):
    """Vectorized rotation matrix construction"""
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    rot = torch.zeros((*theta.shape, 3, 3), device=theta.device, dtype=theta.dtype)
    rot[..., 0, 0] = cos_theta
    rot[..., 0, 1] = sin_theta
    rot[..., 1, 0] = -sin_theta
    rot[..., 1, 1] = cos_theta
    rot[..., 2, 2] = 1.0

    return rot


def compute_max_diagonal(corners):
    """Compute maximum diagonal distance efficiently"""
    diffs = torch.roll(corners, -1, dims=-2) - corners
    return torch.amax(torch.hypot(diffs[..., 0], diffs[..., 1]), dim=-1)


# rotate velocity components in 2D:
def transform_velocities_and_covs(trajs, trajs_covs):
    """Helper function to avoid code duplication"""
    cos_theta = torch.cos(trajs[..., 2])
    sin_theta = torch.sin(trajs[..., 2])
    v = trajs[..., -1]

    vx = v * cos_theta
    vy = v * sin_theta

    # New trajectory: [x, y, theta, vx, vy]
    trajs_new = torch.cat(
        [
            trajs[..., :-1],  # [x, y, theta]
            vx.unsqueeze(-1),
            vy.unsqueeze(-1),
        ],
        dim=-1,
    )
    assert trajs_new.shape[-1] == 5, (
        f"Expected trajs_new to have last dimension of size 5, got {trajs_new.shape[-1]}, Check difference!"
    )
    # Build transformation matrix H
    H = torch.zeros((*trajs_new.shape, 4), dtype=trajs.dtype, device=trajs.device)
    torch.einsum("...ii->...i", H[..., :3, :3])[...] = 1.0
    H[..., 3, 3] = cos_theta
    H[..., 3, 2] = -v * sin_theta
    H[..., 4, 3] = sin_theta
    H[..., 4, 2] = v * cos_theta

    covs_new = H @ trajs_covs @ H.swapaxes(-1, -2)

    return trajs_new, covs_new


def collision_polygon(poly_corners_1: torch.tensor, poly_corners_2: torch.tensor):
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
    poly1_center = poly_corners_1.mean(dim=-2, keepdim=True)
    poly2_center = poly_corners_2.mean(dim=-2, keepdim=True)

    poly2_shifted = poly_corners_2 - poly2_center

    # Compute Minkowski sum with polygon1's center preserved
    edges1 = torch.roll(poly_corners_1, -1, dims=-2) - poly_corners_1
    edges2 = torch.roll(poly2_shifted, -1, dims=-2) - poly2_shifted
    edges = torch.cat([edges1, edges2], dim=-2)
    angles1 = (torch.atan2(edges1[..., 1], edges1[..., 0]) + 2 * np.pi) % (2 * np.pi)
    angles2 = (torch.atan2(edges2[..., 1], edges2[..., 0]) + 2 * np.pi) % (2 * np.pi)
    angles = torch.cat([angles1, angles2], dim=-1)

    # Sort edges by angle
    sorted_indices = torch.argsort(angles, dim=-1)
    # Find leftmost among vertices with minimum y-coordinate to start the convex hull and move to origin
    min_y1_coord = torch.amin(poly_corners_1[..., 1], dim=-1, keepdim=True)  # [:,np.newaxis]
    min_y2_coord = torch.amin(poly2_shifted[..., 1], dim=-1, keepdim=True)
    # Find indices of vertices with minimum y-coordinate for each polygon
    min_y_mask1 = poly_corners_1[..., 1] == min_y1_coord
    min_y_mask2 = poly2_shifted[..., 1] == min_y2_coord

    # Among vertices with min y, find the one with minimum x for each polygon
    # Use masked array approach to handle variable number of bottom vertices
    leftmost_x1 = torch.where(min_y_mask1, poly_corners_1[..., 0], torch.inf)
    leftmost_x2 = torch.where(min_y_mask2, poly2_shifted[..., 0], torch.inf)

    # Get indices of leftmost bottom vertices
    leftmost_idx1 = leftmost_x1.argmin(dim=-1)
    leftmost_idx2 = leftmost_x2.argmin(dim=-1)

    # Extract the leftmost bottom vertices using advanced indexing
    # Create batch indices for all dimensions except the last two
    batch_shape = poly_corners_1.shape[:-2]  # Shape of all batch dimensions
    if len(batch_shape) > 0:
        # Create batch indices
        batch_indices = torch.meshgrid(
            *[torch.arange(s, device=poly_corners_1.device) for s in batch_shape], indexing="ij"
        )
        batch_indices = torch.broadcast_tensors(*batch_indices)

        leftmost_bottom1 = poly_corners_1[(*batch_indices, leftmost_idx1)]
        leftmost_bottom2 = poly2_shifted[(*batch_indices, leftmost_idx2)]

        # Initialize Minkowski sum
        initial_coord = leftmost_bottom1 + leftmost_bottom2
        n_edges = sorted_indices.shape[-1] - 1
        batch_idx_expanded = [idx.unsqueeze(-1).expand(*batch_shape, n_edges) for idx in batch_indices]

        # Select edges in sorted order (all at once)
        selected_edges = edges[(*batch_idx_expanded, sorted_indices[..., :-1])]

        # Build Minkowski coordinates using cumulative sum
        cumsum_edges = torch.cumsum(selected_edges, dim=-2)
        minkowski_coords = torch.cat([initial_coord.unsqueeze(-2), initial_coord.unsqueeze(-2) + cumsum_edges], dim=-2)

    else:
        # No batch dimensions - simpler indexing
        leftmost_bottom1 = poly_corners_1[leftmost_idx1]
        leftmost_bottom2 = poly2_shifted[leftmost_idx2]
        initial_coord = leftmost_bottom1 + leftmost_bottom2

        # Select edges and compute cumsum
        selected_edges = edges[sorted_indices[:-1]]
        cumsum_edges = torch.cumsum(selected_edges, dim=0)
        minkowski_coords = torch.cat([initial_coord.unsqueeze(0), initial_coord.unsqueeze(0) + cumsum_edges], dim=0)
    # relative angle between the two polygons
    minkowski_coords[..., -1] = poly2_center[..., -1] - poly1_center[..., -1]

    return minkowski_coords, poly2_center

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
    v0 = polygon                                       
    v1 = torch.roll(polygon, shifts=-1, dims=-2)        
    edge = v1 - v0                                     
    edge_len_sq = (edge ** 2).sum(dim=-1)              

    p = points#.unsqueeze(-2)                        
    t = ((p - v0) * edge).sum(dim=-1) / (edge_len_sq + 1e-12)    
    t = t.clamp(0.0, 1.0)                              

    closest_pts = v0 + t.unsqueeze(-1) * edge        
    diff = p - closest_pts                            
    dists = torch.linalg.norm(diff, dim=-1)             

    min_idx = dists.argmin(dim=-1)
    min_dist = diff[np.arange(diff.shape[0]), min_idx]  
    return min_dist

def simple_prediction(
    initial_state: torch.Tensor,
    initial_cov: torch.Tensor = None,
    dt: float = 0.1,
    num_steps: int = 30,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated constant velocity model prediction with EKF.

    State: [x, y, theta, v]

    Args:
        initial_state: (..., state_dim) Initial state
        initial_cov: (..., state_dim, state_dim) Initial covariance
        dt: Time step
        num_steps: Number of prediction steps
        device: PyTorch device

    Returns:
        traj: (..., num_steps+1, state_dim) Predicted trajectory
        covs: (..., num_steps+1, state_dim, state_dim) Predicted covariances
    """
    device = get_device(device)

    if initial_state.ndim == 1:
        initial_state = initial_state.unsqueeze(0)
    if initial_cov is not None and initial_cov.ndim == 2:
        initial_cov = initial_cov.unsqueeze(0)

    batch_shape = initial_state.shape[:-1]
    state_dim = initial_state.shape[-1]

    # Process noise (can be optimized with cached constants)
    q = 0.0  # Process noise strength
    sigma_theta = 0.0

    Q = torch.zeros(state_dim, state_dim, device=device, dtype=torch.float32)
    Q[0, 0] = 0.25 * dt**4 * q
    Q[0, 3] = 0.5 * dt**3 * q
    Q[3, 0] = 0.5 * dt**3 * q
    Q[3, 3] = dt**2 * q
    Q[1, 1] = 0.25 * dt**4 * q
    Q[1, 3] = 0.5 * dt**3 * q
    Q[3, 1] = 0.5 * dt**3 * q
    Q[2, 2] = sigma_theta**2 * dt

    # Initialize trajectory storage
    traj = torch.zeros(*batch_shape, num_steps + 1, state_dim, device=device, dtype=torch.float32)
    covs = torch.zeros(*batch_shape, num_steps + 1, state_dim, state_dim, device=device, dtype=torch.float32)

    traj[..., 0, :] = initial_state
    if initial_cov is not None:
        covs[..., 0, :, :] = initial_cov
    else:
        covs[..., 0, :, :] = torch.eye(state_dim, device=device) * 1e-6

    # Vectorized prediction over all timesteps
    for t in range(1, num_steps + 1):
        state = traj[..., t - 1, :]
        cov = covs[..., t - 1, :, :]

        theta = state[..., 2]
        v = state[..., 3]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # State transition
        dx = v * cos_theta * dt
        dy = v * sin_theta * dt

        next_state = state.clone()
        next_state[..., 0] += dx
        next_state[..., 1] += dy

        # Jacobian F
        F = torch.eye(state_dim, device=device, dtype=torch.float32).unsqueeze(0).expand(*batch_shape, -1, -1).clone()
        F[..., 0, 2] = -dt * v * sin_theta
        F[..., 0, 3] = dt * cos_theta
        F[..., 1, 2] = dt * v * cos_theta
        F[..., 1, 3] = dt * sin_theta

        # Covariance propagation: F @ cov @ F.T + Q
        next_cov = torch.einsum("...ij,...jk,...lk->...il", F, cov, F) + Q

        traj[..., t, :] = next_state
        covs[..., t, :, :] = next_cov

    return traj, covs


def convolve_distributions(
    mus1: torch.Tensor = None,
    devs1: torch.Tensor = None,
    mus2: torch.Tensor = None,
    devs2: torch.Tensor = None,
    dist="Gaussian",
    device="cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the relative distributions between two objects by convolution two Gaussian distributions.

    Parameters:
        mus1 (ndarray): Mean of the first distribution.
        devs1 (ndarray): Covariance of the first distribution.
        mus2 (ndarray): Mean of the second distribution.
        devs2 (ndarray): Covariance of the second distribution.

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

    mus1 = mus1 if mus1 is not None else torch.tensor([0], device=device)
    devs1 = devs1 if devs1 is not None else torch.tensor([0], device=device)
    mus2 = mus2 if mus2 is not None else torch.tensor([0], device=device)
    devs2 = devs2 if devs2 is not None else torch.tensor([0], device=device)
    match dist:
        case "Gaussian" | "Normal" | "Cauchy":
            mus = [mus2, -mus1]
            sigs = [devs2, devs1]

            a = 1
            b = [0, 0]
            rel_mean, rel_cov = stable(mus, sigs, a, b)
            if rel_mean.shape[:-1] != rel_cov.shape[:-2]:
                delta_dim = rel_mean.ndim - rel_cov.ndim+1
                if delta_dim<0:
                    rel_mean = rel_mean + rel_cov[..., 0, 0].unsqueeze(-1) * 0
                else:
                    rel_cov = rel_cov + rel_mean.unsqueeze(-1)* 0

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

def bivariate_normal_cdf(
    mean: torch.Tensor,
    cov: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute bivariate normal CDF Φ₂(x, y) using Drezner's approximation.

    This is a fast approximation accurate to ~1e-7.

    Args:
        mean: Mean vector (2,)
        cov: Covariance matrix (2, 2)
        x, y: Evaluation point
        device: PyTorch device

    Returns:
        CDF value at (x, y)
    """
    # Standardize
    mu_x, mu_y = mean[...,0], mean[...,1]
    sigma_x = torch.sqrt(cov[...,0, 0])
    sigma_y = torch.sqrt(cov[...,1, 1])
    rho = cov[...,0, 1] / (sigma_x * sigma_y + 1e-10)

    h = (x - mu_x) / sigma_x
    k = (y - mu_y) / sigma_y

    # Use Drezner-Wesolowsky approximation
    # This is a fast polynomial approximation
    return bivariate_normal_cdf_standard(h, k, rho, device)


def bivariate_normal_cdf_standard(
    h: torch.Tensor,
    k: torch.Tensor,
    rho: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Standard bivariate normal CDF using Drezner's approximation.

    Approximates Φ(h, k; ρ) for standardized bivariate normal.

    Reference:
    Drezner, Z., & Wesolowsky, G. O. (1990). On the computation of the
    bivariate normal integral. Journal of Statistical Computation and Simulation.
    """
    # device = get_device(device)

    # Constants for Gauss-Legendre quadrature (10-point)
    # These provide ~1e-7 accuracy
    x_nodes = torch.tensor(
        [
            -0.9739065285,
            -0.8650633667,
            -0.6794095683,
            -0.4333953941,
            -0.1488743390,
            0.1488743390,
            0.4333953941,
            0.6794095683,
            0.8650633667,
            0.9739065285,
        ],
        device=device,
        dtype=DTYPE,
    )

    w_nodes = torch.tensor(
        [
            0.0666713443,
            0.1494513492,
            0.2190863625,
            0.2692667193,
            0.2955242247,
            0.2955242247,
            0.2692667193,
            0.2190863625,
            0.1494513492,
            0.0666713443,
        ],
        device=device,
        dtype=DTYPE,
    )

    # Standard normal CDF (using erf)
    def standard_normal_cdf(z):
        return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=device, dtype=DTYPE))))

    # Handle special cases
    if torch.all(torch.abs(rho) < 1e-10):
        # Independent case
        return standard_normal_cdf(h) * standard_normal_cdf(k)

    # Compute using numerical integration
    hk = h * k

    if torch.any(hk >= 0) or (torch.all(h * k == 0) and torch.all(h + k >= 0)):
        # Use direct integration
        result = torch.zeros(h.shape, device=device, dtype=h.dtype)

        for i in range(len(x_nodes)):
            x = x_nodes[i]
            w = w_nodes[i]

            # Gauss-Legendre quadrature
            asin_term = torch.arcsin(rho * (1 + x) / 2)

            exp_term = torch.exp(
                -(h * h + k * k - 2 * rho * h * k * (1 + x) / 2) / (2 * (1 - rho * rho * (1 + x) * (1 + x) / 4) + 1e-10)
            )

            result += w * torch.exp(-asin_term) * exp_term

        result = result / (4 * torch.pi)
        result += standard_normal_cdf(h) * standard_normal_cdf(k)

    else:
        # Use complementary form
        if torch.all(h >= 0):
            result = standard_normal_cdf(h) - bivariate_normal_cdf_standard(h, -k, -rho, device)
        elif torch.all(k >= 0):
            result = standard_normal_cdf(k) - bivariate_normal_cdf_standard(-h, k, -rho, device)
        else:
            result = (
                standard_normal_cdf(h)
                + standard_normal_cdf(k)
                - 1.0
                + bivariate_normal_cdf_standard(-h, -k, rho, device)
            )

    return result.clamp(0.0, 1.0)  # Ensure valid probability


def separating_axis_test_batch(poly1_batch: torch.Tensor, poly2_batch: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Separating Axis Theorem (SAT) test for polygon-polygon collision.

    Args:
        poly1_batch: (B, N1, 2) - Batch of first polygons
        poly2_batch: (B, N2, 2) - Batch of second polygons

    Returns:
        overlap_mask: (B,) - Boolean mask indicating overlaps
    """
    # Get edges for both polygons
    edges1 = torch.roll(poly1_batch, -1, dims=1) - poly1_batch  # (B, N1, 2)
    edges2 = torch.roll(poly2_batch, -1, dims=1) - poly2_batch  # (B, N2, 2)

    # Compute normals (perpendicular to edges)
    normals1 = torch.stack([-edges1[..., 1], edges1[..., 0]], dim=-1)  # (B, N1, 2)
    normals2 = torch.stack([-edges2[..., 1], edges2[..., 0]], dim=-1)  # (B, N2, 2)

    # Normalize
    normals1 = normals1 / (torch.norm(normals1, dim=-1, keepdim=True) + 1e-8)
    normals2 = normals2 / (torch.norm(normals2, dim=-1, keepdim=True) + 1e-8)

    # Combine all axes to test
    all_normals = torch.cat([normals1, normals2], dim=1)  # (B, N1+N2, 2)

    # Project vertices onto each axis)
    proj1 = torch.einsum("bna,bva->bnv", all_normals, poly1_batch)
    proj2 = torch.einsum("bna,bva->bnv", all_normals, poly2_batch)

    # Find min/max projections for each axis
    min1, _ = proj1.min(dim=-1)  # (B, N1+N2)
    max1, _ = proj1.max(dim=-1)
    min2, _ = proj2.min(dim=-1)
    max2, _ = proj2.max(dim=-1)

    # Check for separation on each axis
    # If max1 < min2 or max2 < min1, polygons are separated
    separated = (max1 < min2) | (max2 < min1)  # (B, N1+N2)

    # If ANY axis shows separation, polygons don't overlap
    overlap = ~separated.any(dim=-1)  # (B,)

    return overlap