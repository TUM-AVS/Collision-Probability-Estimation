"""
GPU-accelerated collision probability estimation based on spatial overlap using PyTorch.
Following methods are implemented:
1) Monte Carlo sampling for polygon-polygon collision probability.
2) Inclusion-Exclusion principle for rectangle using bivariate normal CDF.
3) Monte Carlo sampling for point-in-polygon collision probability.
4) Analytic solution for point-in-polygon collision probability.
5) Mahalanobis distance-based collision probability.
6) Corner Mahalanobis distance-based collision probability (boundary distance).
"""

import torch
from torch.distributions import MultivariateNormal
from collision_probability_estimation.params import NODES_XY, WEIGHTS_XY, NODES_THETA, WEIGHTS_THETA, THETA_NSTD

from .utils import (collision_polygon,
                    convolve_distributions,
                    build_rotation_matrices, 
                    to_torch, 
                    get_device, 
                    bivariate_normal_cdf,
                    point_to_polygon_boundary_dist,
                    separating_axis_test_batch,
                    DTYPE)

_CONSTANTS_CACHE = {}


def get_cached_constants(device):
    """Cache frequently used constants on the target device."""
    if device not in _CONSTANTS_CACHE:
        _CONSTANTS_CACHE[device] = {
            "nodes_xy": torch.from_numpy(NODES_XY).to(dtype=DTYPE, device=device),
            "weights_xy": torch.from_numpy(WEIGHTS_XY).to(dtype=DTYPE, device=device),
            "nodes_theta": torch.from_numpy(NODES_THETA).to(dtype=DTYPE, device=device),
            "weights_theta": torch.from_numpy(WEIGHTS_THETA).to(dtype=DTYPE, device=device),
            "sqrt_2": torch.sqrt(torch.tensor(2.0, device=device, dtype=DTYPE)),
            "sqrt_2pi": torch.sqrt(torch.tensor(2.0 * torch.pi, device=device, dtype=DTYPE)),
            "inv_sqrt_8pi": 1.0 / torch.sqrt(torch.tensor(8.0 * torch.pi, device=device, dtype=DTYPE)),
        }
    return _CONSTANTS_CACHE[device]


def poly_poly_mc_sampling(
    obs1: torch.Tensor,
    obs2: torch.Tensor,
    obs1_cov: torch.Tensor = None,
    obs2_cov: torch.Tensor = None,
    num_samples: int = 20000,
    with_orientation: bool = False,
    device: torch.device = None,
) -> tuple[float, dict[str, torch.Tensor]]:
    """
    GPU-accelerated Monte Carlo sampling for polygon-polygon collision probability.

    This function uses vectorized operations and GPU acceleration to compute collision
    probabilities between two polygons using Monte Carlo sampling.

    Args:
        obs1: First polygon vertices (N, 2/3)
        obs2: Second polygon vertices (M, 2/3)
        obs1_cov: Covariance matrix of first polygon (2x2 or 3x3)
        obs2_cov: Covariance matrix of second polygon (2x2 or 3x3)
        num_samples: Number of Monte Carlo samples
        with_orientation: Whether to consider orientation uncertainty
        device: PyTorch device (GPU/CPU)

    Returns:
        coll_prob: Estimated collision probability
        data_dict: Dictionary containing sampled polygons and overlap mask
    """
    device = get_device(device)

    # Convert to torch tensors with EXPLICIT dtype (float32 for GPU efficiency)
    obs1 = to_torch(obs1, device)
    obs2 = to_torch(obs2, device)

    if obs1_cov is not None:
        obs1_cov = to_torch(obs1_cov, device)
    if obs2_cov is not None:
        obs2_cov = to_torch(obs2_cov, device)

    assert obs1.shape[-1] >= 2 and obs2.shape[-1] >= 2, "Polygons must have at least 2D coordinates."
    assert obs1.shape[-2] > 2 and obs2.shape[-2] > 2, "Polygons must have at least 3 corners."

    idx = 3 if with_orientation else 2

    # Compute polygon centroids
    poly1_center = obs1[:, :2].mean(dim=0)
    poly2_center = obs2[:, :2].mean(dim=0)

    # Sample shifts for polygon 1
    n_samples_per_poly = int(torch.sqrt(torch.tensor(num_samples, dtype=DTYPE)).item())

    if obs1_cov is not None:
        obs1_cov_idx = obs1_cov[:idx, :idx]
        # Ensure covariance is float32 and positive definite
        obs1_cov_idx = obs1_cov_idx.float()  # ← Ensure float32

        # Generate samples using PyTorch MultivariateNormal
        dist1 = MultivariateNormal(
            loc=torch.zeros(idx, device=device, dtype=DTYPE),  # ← Add dtype
            covariance_matrix=obs1_cov_idx,
        )
        poly1_shifts = dist1.sample((n_samples_per_poly,))

        if not with_orientation:
            poly1_shifts = torch.cat(
                [
                    poly1_shifts,
                    torch.zeros(n_samples_per_poly, 1, device=device, dtype=DTYPE),  # ← Add dtype
                ],
                dim=-1,
            )
    else:
        poly1_shifts = torch.zeros(1, 3, device=device, dtype=DTYPE)  # ← Add dtype
        n_samples_per_poly = 1

    # Sample shifts for polygon 2
    if obs2_cov is not None:
        obs2_cov_idx = obs2_cov[:idx, :idx]
        obs2_cov_idx = obs2_cov_idx.to(dtype=DTYPE)  # ← Ensure float32

        dist2 = MultivariateNormal(
            loc=torch.zeros(idx, device=device, dtype=DTYPE),  # ← Add dtype
            covariance_matrix=obs2_cov_idx,
        )
        poly2_shifts = dist2.sample((n_samples_per_poly,))

        if not with_orientation:
            poly2_shifts = torch.cat(
                [
                    poly2_shifts,
                    torch.zeros(n_samples_per_poly, 1, device=device, dtype=DTYPE),  # ← Add dtype
                ],
                dim=-1,
            )
    else:
        poly2_shifts = torch.zeros(1, 3, device=device, dtype=DTYPE)

    # Create rotation matrices for all samples
    angles1 = poly1_shifts[:, 2] if poly1_shifts.shape[-1] > 2 else torch.zeros(poly1_shifts.shape[0], device=device)
    angles2 = poly2_shifts[:, 2] if poly2_shifts.shape[-1] > 2 else torch.zeros(poly2_shifts.shape[0], device=device)

    # Expand for all combinations
    angles1_exp = angles1[:, None].expand(len(angles1), len(angles2))
    angles2_exp = angles2[None, :].expand(len(angles1), len(angles2))

    cos1, sin1 = torch.cos(angles1_exp), torch.sin(angles1_exp)
    cos2, sin2 = torch.cos(angles2_exp), torch.sin(angles2_exp)

    # Build rotation matrices
    rot_mat1 = torch.stack([torch.stack([cos1, -sin1], dim=-1), torch.stack([sin1, cos1], dim=-1)], dim=-2)

    rot_mat2 = torch.stack([torch.stack([cos2, -sin2], dim=-1), torch.stack([sin2, cos2], dim=-1)], dim=-2)

    # Apply transformations to polygons (vectorized)
    poly1_centered = obs1[:, :2] - poly1_center
    poly2_centered = obs2[:, :2] - poly2_center

    # Translate and rotate polygon 1
    trans1 = poly1_shifts[:, None, None, :2]  # (n_samples1, 1, 1, 2)
    poly1_transformed = (
        torch.einsum("...ij,...j->...i", rot_mat1[:, :, None, :, :], poly1_centered[None, None, :, :])
        + poly1_center
        + trans1
    )

    # Translate and rotate polygon 2
    trans2 = poly2_shifts[None, :, None, :2]  # (1, n_samples2, 1, 2)
    poly2_transformed = (
        torch.einsum("...ij,...j->...i", rot_mat2[:, :, None, :, :], poly2_centered[None, None, :, :])
        + poly2_center
        + trans2
    )

    # Check for overlaps using Separating Axis Theorem (SAT)
    overlap_mask = separating_axis_test_batch(
        poly1_transformed.reshape(-1, obs1.shape[0], 2), poly2_transformed.reshape(-1, obs2.shape[0], 2)
    )
    overlap_mask = overlap_mask.reshape(len(angles1), len(angles2))

    # Compute collision probability
    coll_prob = overlap_mask.mean(dtype=DTYPE)
    if coll_prob.ndim == 0:
        coll_prob = coll_prob.unsqueeze(0)
    return coll_prob, {
        "poly1": poly1_transformed,
        "poly2": poly2_transformed,
        "overlap_mask": overlap_mask,
    }

def point_rect_incl_excl(
    obs1: torch.Tensor,
    obs2: torch.Tensor,
    obs1_cov: torch.Tensor = None,
    obs2_cov: torch.Tensor = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    GPU-accelerated inclusion-exclusion principle for point-in-rectangle collision probability.

    Calculates the proportion of a 2D Gaussian (mean=point, cov=cov) inside a rectangle
    using the inclusion-exclusion principle.

    Args:
        obs1: Rectangle vertices (4, 2/3) - must have exactly 4 corners
        obs2: Point coordinates (2/3) or (1, 2/3)
        obs1_cov: Covariance matrix of rectangle (2x2 or 3x3)
        obs2_cov: Covariance matrix of point (2x2 or 3x3)
        device: PyTorch device (GPU/CPU)

    Returns:
        coll_prob: Collision probability (scalar or batch)
    """
    device = get_device(device)

    # Convert to torch tensors
    obs1 = to_torch(obs1, device)
    obs2 = to_torch(obs2, device)

    if obs1_cov is not None:
        obs1_cov = to_torch(obs1_cov, device)
    if obs2_cov is not None:
        obs2_cov = to_torch(obs2_cov, device)

    assert obs1.shape[-2] == 4, "Rectangle must have 4 boundary points (closed polygon)."

    # Convolve distributions
    _, rel_cov = convolve_distributions(devs1=obs1_cov, devs2=obs2_cov, dist="Gaussian", device=device)

    # Extract rectangle corners and orientation
    xy = obs1[..., :2]  # (4, 2)
    theta = obs1[...,0,2]  # scalar

    # Ensure obs2 has correct shape
    if obs2.ndim == 1:
        obs2 = obs2.unsqueeze(0)  # (1, 2/3)

    point_coords = obs2.mean(dim=-2)  # (2/3,)

    # Create rotation matrix to align rectangle with axes
    while rel_cov.dim() < 3:
        rel_cov = rel_cov.unsqueeze(0)
    rot = build_rotation_matrices(theta)[...,:2,:2]
 
    # Rotate rectangle to be axis-aligned
    xy_aligned = xy @ rot  # (4, 2)
    x_min, _ = xy_aligned[..., 0].min(dim=-1)
    y_min, _ = xy_aligned[..., 1].min(dim=-1)
    x_max, _ = xy_aligned[..., 0].max(dim=-1)
    y_max, _ = xy_aligned[..., 1].max(dim=-1)

    # Rotate covariance matrix
    cov_aligned = rot.mT @ rel_cov[...,:2, :2] @ rot # (2, 2)

    # Rotate point
    point_aligned = (point_coords[...,None,:2] @ rot).squeeze()  # (2,)

    # Compute CDF using bivariate normal distribution
    # PyTorch doesn't have built-in bivariate CDF, so we implement it
    F1 = bivariate_normal_cdf(mean=point_aligned, cov=cov_aligned, x=x_max, y=y_max, device=device)
    F2 = bivariate_normal_cdf(mean=point_aligned, cov=cov_aligned, x=x_min, y=y_max, device=device)
    F3 = bivariate_normal_cdf(mean=point_aligned, cov=cov_aligned, x=x_max, y=y_min, device=device)
    F4 = bivariate_normal_cdf(mean=point_aligned, cov=cov_aligned, x=x_min, y=y_min, device=device)

    # Inclusion-exclusion principle
    coll_prob = F1 - F2 - F3 + F4

    return coll_prob


def point_collvol_mc_sampling(
    obs1: torch.Tensor,
    obs2: torch.Tensor,
    obs1_cov: torch.Tensor = None,
    obs2_cov: torch.Tensor = None,
    num_samples: int = 20000,
    with_orientation: bool = False,
    device: torch.device = None,
) -> tuple[float, dict[str, torch.Tensor]]:
    """
    GPU-accelerated Monte Carlo sampling for point-in-polygon collision probability.

    Computes the collision probability of a point (Gaussian distributed) inside a polygon
    using vectorized Monte Carlo sampling on GPU.

    Args:
        obs1: First polygon vertices (N, 2/3)
        obs2: Second polygon vertices (M, 2/3) or point coordinates
        obs1_cov: Covariance matrix of first polygon (2x2 or 3x3)
        obs2_cov: Covariance matrix of second polygon/point (2x2 or 3x3)
        num_samples: Number of Monte Carlo samples
        with_orientation: Whether to consider orientation uncertainty
        device: PyTorch device (GPU/CPU)

    Returns:
        coll_prob: Estimated collision probability
        data_dict: Dictionary containing sampled points and overlap mask
    """
    device = get_device(device)

    # Convert to torch tensors with float32
    obs1 = to_torch(obs1, device)
    obs2 = to_torch(obs2, device)
    if obs1_cov is not None:
        obs1_cov = to_torch(obs1_cov, device)
    if obs2_cov is not None:
        obs2_cov = to_torch(obs2_cov, device)

    seg_idx = 3 if with_orientation else 2

    # Handle dimensions
    if obs2.ndim == 1:
        obs2 = obs2.unsqueeze(0)

    # Compute centers
    obs1_center = obs1.mean(dim=-2)
    obs2_center = obs2.mean(dim=-2)

    # Convolve distributions
    rel_dist, rel_cov = convolve_distributions(
        mus1=obs1_center,
        mus2=obs2_center,
        devs1=obs1_cov,
        devs2=obs2_cov,
        dist="Gaussian",
        device=device,
    )

    # Sample random points
    if rel_cov is not None and torch.any(rel_cov != 0):
        dist = MultivariateNormal(loc=obs2_center[..., :seg_idx], covariance_matrix=rel_cov[..., :seg_idx, :seg_idx])
        # Do NOT set seed here - each call should have independent samples
        sampled_points = dist.sample((num_samples,))  # (num_samples, seg_idx)
    else:
        sampled_points = obs2_center[None, ..., :seg_idx]  # (1, seg_idx)

    # Handle orientation
    if with_orientation and rel_cov is not None and rel_cov.shape[-1] >= 3:
        # Compute rotation angles
        delta_theta = sampled_points[..., seg_idx - 1] - obs2_center[..., seg_idx - 1]  # (num_samples,)
        rot_theta = build_rotation_matrices(delta_theta)

        # Rotate obs2 for all samples: (num_samples, M, 3)
        obs2_rot = obs2 @ rot_theta
        # obs2_rot = torch.einsum("ij,njk->nik", obs2, rot_theta)

        # Expand obs1 to match: (num_samples, N, 3)
        obs1_ext = obs1.unsqueeze(0).expand(obs2_rot.shape[0], *obs1.shape)

        # Compute collision polygons
        collvol, _ = collision_polygon(obs1_ext, obs2_rot)

        if collvol.ndim == 3:
            collvol_expand = collvol.unsqueeze(1)  # (num_samples, 1, vertices, 3)
        else: 
            collvol_expand = collvol
    else:
        # No orientation
        collvol, _ = collision_polygon(obs1, obs2)
        if collvol.ndim == 2:
            # Single collision volume: (vertices, 3) -> (1, 1, vertices, 3)
            collvol = collvol.unsqueeze(0)
        if collvol.ndim == 3:
            # Batch collision volumes: (N, vertices, 3) -> (1, N, vertices, 3)
            collvol_expand = collvol.unsqueeze(0)
        else:
            collvol_expand = collvol

        # Now expand along num_samples dimension: (1, N, vertices, 3) -> (num_samples, N, vertices, 3)
        collvol_expand = collvol_expand.expand(sampled_points.shape[0], -1, -1, -1)

    # Prepare points for collision check
    if sampled_points.ndim == 2:
        n_polys = collvol_expand.shape[1]
        points = sampled_points.unsqueeze(1).expand(-1, n_polys, -1)  # (num_samples, n_polys, seg_idx)
    else: 
        points = sampled_points  # (num_samples, n_polys, seg_idx)
    # Vectorized point-in-polygon test
    overlap_mask = point_in_polygon_batch_vectorized(
        points[..., :2],  
        collvol_expand[..., :2],  
    )

    coll_prob = overlap_mask.float().mean(dim=0)  

    return coll_prob, {
        "points": sampled_points, 
        "collvol": collvol,
    }


def point_in_polygon_batch_vectorized(
    points: torch.Tensor,
    polygons: torch.Tensor,
) -> torch.Tensor:
    """
    Fully vectorized point-in-polygon test using ray casting (no loops).

    Args:
        points: (B, N, 2) - Batch of points to test
        polygons: (B, N, V, 2) or (1, N, V, 2) - Batch of polygons
            B = batch size (num_samples)
            N = number of polygons per sample
            V = number of vertices per polygon

    Returns:
        inside: (B, N) - Boolean mask indicating if points are inside polygons
    """

     # Expand points: (B, N, 1, 2) for broadcasting
    points_exp = points.unsqueeze(-2)  # (B, N, 1, 2)
    if points_exp.ndim != polygons.ndim:
         polygons = polygons.unsqueeze(2)
    p1 = polygons
    p2 = torch.roll(polygons, -1, dims=-2)
    # Ray casting: horizontal ray to the right from each point
    # Check if ray crosses each edge

    # Condition 1: Edge crosses horizontal line through point
    y1_above = p1[..., 1] > points_exp[..., 1]  # (B, N, V)
    y2_above = p2[..., 1] > points_exp[..., 1]
    y_crosses = y1_above != y2_above

    # Condition 2: Intersection point is to the right of the point
    dy = p2[..., 1] - p1[..., 1]  # (B, N, V)
    dx = p2[..., 0] - p1[..., 0]

    # Avoid division by zero
    dy_safe = torch.where(torch.abs(dy) < 1e-10, torch.ones_like(dy) * 1e-10, dy)

    x_intersection = p1[..., 0] + (points_exp[..., 1] - p1[..., 1]) * dx / dy_safe
    x_right = points_exp[..., 0] < x_intersection

    # Count intersections per polygon
    intersections = (y_crosses & x_right).sum(dim=-1)  # (B, N)

    # Odd number of intersections = inside
    inside = (intersections % 2) == 1

    return inside.to(dtype=points.dtype)


def point_collvol_analytic(
    obs1: torch.Tensor,
    obs2: torch.Tensor,
    obs1_cov: torch.Tensor = None,
    obs2_cov: torch.Tensor = None,
    with_orientation: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """
    GPU-accelerated analytic solution for point-in-polygon collision probability.

    This function computes the collision probability of a point (Gaussian distributed)
    being inside a polygon using analytic integration.

    Args:
        obs1: First polygon vertices (..., n_corners, 2/3)
        obs2: Second polygon vertices or point (..., n_corners, 2/3)
        obs1_cov: Covariance matrix of first polygon (..., 2/3, 2/3)
        obs2_cov: Covariance matrix of second polygon (..., 2/3, 2/3)
        with_orientation: Whether to consider orientation
        device: Torch device (GPU/CPU)

    Returns:
        overlap_prob: Collision probability (...)
    """
    # device = get_device(device)
    constants = get_cached_constants(device)
    # Convert to torch tensors
    obs1 = to_torch(obs1, device)
    obs2 = to_torch(obs2, device)

    if obs1_cov is not None:
        obs1_cov = to_torch(obs1_cov, device)
    if obs2_cov is not None:
        obs2_cov = to_torch(obs2_cov, device)

    # Handle dimensions
    if obs1.ndim - obs2.ndim == 1:
        obs2 = obs2.unsqueeze(-2)

    obs1_center = obs1.mean(dim=-2)
    obs2_center = obs2.mean(dim=-2)

    # Convolve distributions
    rel_traj, rel_cov = convolve_distributions(
        mus1=obs1_center,
        mus2=obs2_center,
        devs1=obs1_cov,
        devs2=obs2_cov,
        dist="Gaussian",
        device=device,
    )

    if rel_cov is None:
        point_in_polygon_batch_vectorized(obs2[..., :2], obs1[..., :2]).float()
        raise NotImplementedError("Check coall of point_in_polygon")
        #TODO!
        return

    if with_orientation and rel_cov is not None and rel_cov.shape[-1] >= 3:
        theta_mean = rel_traj[..., 2]
        theta_std = torch.sqrt(rel_cov[..., 2, 2])

        delta_theta = (THETA_NSTD * constants["nodes_theta"][:, None] * theta_std).to(dtype=theta_mean.dtype)
        theta_gauss = delta_theta + theta_mean
        theta_jacobian = THETA_NSTD * theta_std

        # Build rotation matrices
        rot_theta = build_rotation_matrices(delta_theta)
        obs2_rot = obs2 @ rot_theta

        ndim_diff = obs2_rot.ndim - obs1.ndim
        obs1_exp = obs1.view((1,) * ndim_diff + obs1.shape).expand(obs2_rot.shape[:-2] + obs1.shape[-2:])
        # obs1_exp = obs1_exp

        collvol, points = collision_polygon(obs1_exp, obs2_rot)
    else:
        collvol, points = collision_polygon(obs1, obs2)
        collvol = collvol.unsqueeze(0)

    # Cholesky decomposition of covariance
    rel_cov_pos = rel_cov[..., :3, :3]
    reg_eye = torch.eye(3, device=device, dtype=rel_cov_pos.dtype) * 1e-8
    rel_cov_reg = rel_cov_pos + reg_eye
    try:
        L = torch.linalg.cholesky(rel_cov_reg)
        W = torch.linalg.solve_triangular(L, torch.eye(3, device=device, dtype=L.dtype).expand_as(L), upper=False)
    except RuntimeError:
        # Add small regularization if Cholesky fails
        rel_cov_reg = rel_cov_pos + reg_eye * 100
        L = torch.linalg.cholesky(rel_cov_reg)
        W = torch.linalg.solve_triangular(L, torch.eye(3, device=device, dtype=L.dtype).expand_as(L), upper=False)

    # Shift polygon to origin and scale with covariance
    poly_shifted = (collvol - points) @ W.mT

    # Compute line segments
    poly_xy = poly_shifted[..., :2]
    poly_xy_next = torch.roll(poly_xy, shifts=-1, dims=-2)
    delta = poly_xy_next - poly_xy

    # Compute line parameters: y = mx + c
    m = delta[..., 1] / (delta[..., 0] + 1e-8)
    c = poly_shifted[..., 1] - m * poly_shifted[..., 0]

    x_low = poly_shifted[..., 0].unsqueeze(-1)
    x_high = torch.roll(x_low, shifts=-1, dims=-2)

    # Gaussian quadrature integration
    nodes_xy_exp = constants["nodes_xy"].unsqueeze(0)
    half_range = (x_high - x_low) * 0.5
    mid_point = (x_high + x_low) * 0.5
    x_quad = half_range * nodes_xy_exp + mid_point
    jacobian = half_range

    # Compute integrand
    inv_sqrt2 = 1.0 / constants["sqrt_2"]
    erf_para_quad = -(m.unsqueeze(-1) * x_quad + c.unsqueeze(-1)) * inv_sqrt2
    exp_para_quad = -0.5 * x_quad * x_quad
    integ_quad = torch.erf(erf_para_quad) * torch.exp(exp_para_quad)

    # Weighted sum over quadrature points
    weights_xy_exp = constants["weights_xy"]  # .unsqueeze(0)
    p_sides = constants["inv_sqrt_8pi"] * torch.einsum(
        "...nq,q,...n->...n", integ_quad, weights_xy_exp, jacobian.squeeze(-1)
    )

    # Sum over all sides
    overlap_prob = p_sides.sum(dim=-1)

    if with_orientation:
        # Compute theta PDF using scipy (need to convert to numpy temporarily)
        z_score = (theta_gauss - theta_mean) / theta_std
        theta_pdf = torch.exp(-0.5 * z_score * z_score) / (theta_std * constants["sqrt_2pi"])

        # overlap_prob = (overlap_prob * theta_pdf * constants["weights_theta"][:, None] * theta_jacobian).sum(dim=0)
        weights_theta = constants["weights_theta"]
        overlap_prob = torch.einsum("n...,n,n...->...", overlap_prob, weights_theta, theta_pdf * theta_jacobian)

    return overlap_prob


def mahalanobis_distance(    
    obs1: torch.Tensor,
    obs2: torch.Tensor,
    obs1_cov: torch.Tensor = None,
    obs2_cov: torch.Tensor = None,
    with_orientation: bool = False,
    use_corners: bool = False,
    device: torch.device = None,
):
    """
    GPU-accelerated Mahalanobis distance between two uncertain observations.

    Computes the Mahalanobis distance between two observations (points or polygons)
    considering their covariance. For polygons, uses the centroids.

    Args:
        obs1: First observation (polygon vertices or point coordinates)
        obs2: Second observation (polygon vertices or point coordinates)
        obs1_cov: Covariance matrix of first observation
        obs2_cov: Covariance matrix of second observation
        with_orientation: Whether to consider orientation in distance
        use_corners: Whether to compute distance to polygon corners instead of centroids
        device: PyTorch device (GPU/CPU)

    Returns:
        mahal_dist: Mahalanobis distance (scalar)
    """
    device = get_device(device)

    # Convert to torch tensors
    obs1 = to_torch(obs1, device)
    obs2 = to_torch(obs2, device)

    if obs1_cov is not None:
        obs1_cov = to_torch(obs1_cov, device)
    if obs2_cov is not None:
        obs2_cov = to_torch(obs2_cov, device)

    idx = 3 if with_orientation else 2

    # Compute centroids
    obs1_center = obs1.mean(dim=-2)  # (B, 2)
    obs2_center = obs2.mean(dim=-2)  # (B, 2)

    # Convolve distributions
    rel_traj, rel_cov = convolve_distributions(
        mus1=obs1_center,
        mus2=obs2_center,
        devs1=obs1_cov,
        devs2=obs2_cov,
        dist="Gaussian",
        device=device,
    )

    if rel_cov is None:
        #TODO !
        raise ValueError("Covariance is required for Mahalanobis distance.")
    
    if not use_corners:
         delta_pos = rel_traj[...,:idx].unsqueeze(-1)
    else:
        collvol, points = collision_polygon(obs1, obs2)
        delta_pos = point_to_polygon_boundary_dist(points, collvol)[...,:idx, None]

    # Extract relevant covariance block
    cov_block = rel_cov[..., :idx, :idx]

    # Compute Mahalanobis distance
    try:
        L = torch.linalg.cholesky(cov_block)
        inv_cov = torch.cholesky_inverse(L)
    except:
        inv_cov = torch.linalg.inv(cov_block + 1e-8 * torch.eye(idx, device=device, dtype=cov_block.dtype))

    mahal_sq = delta_pos.mT @ inv_cov @ delta_pos
    # convert mahalanobis distance to probability
    k = torch.tensor(idx, dtype=mahal_sq.dtype, device=mahal_sq.device)
    cdf = torch.special.gammainc(k / 2, mahal_sq / 2)
    prob = 1-cdf.squeeze()
    return prob
