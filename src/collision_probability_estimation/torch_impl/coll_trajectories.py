"""
GPU-accelerated collision probability estimation along trajectories using PyTorch.
This module provides PyTorch implementations of trajectory-based collision probability
estimation methods that can run on GPU for significant speedup.
Following methods are implemented:
1) Monte Carlo Sampling for complete trajectories (boundary crossing via sampling).
2) Boundary crossing method for complete trajectories (analytic).
3) Monte Carlo sampling for each time step.
4) Analytic solution for point-in-polygon collision probability at each time step.
5) Inclusion-Exclusion (rect) for each time step.
6) Mahalanobis distance-based collision probability for each time step.
7) Corner Mahalanobis distance-based collision probability for each time step.

"""

import torch
from torch.distributions import MultivariateNormal
from collision_probability_estimation.params import NODES_XY, WEIGHTS_XY, NODES_THETA, WEIGHTS_THETA, THETA_NSTD
from .coll_state_overlap import (
    point_collvol_analytic,
    point_collvol_mc_sampling,
    point_in_polygon_batch_vectorized,
    point_rect_incl_excl,
    mahalanobis_distance,
)
from .utils import (
    convolve_distributions,
    collision_polygon,
    simple_prediction,
    build_rotation_matrices,
    transform_velocities_and_covs,
    compute_max_diagonal,
    to_torch,
    get_device,
    DTYPE
)

_CONSTANTS_CACHE = {}

# TODO handle cases where cov is none-> get deterministic intersection (here and np)
# TODO add deterministic, risk averse bounding box coll check

def get_cached_constants(device):
    """Cache frequently used constants on the target device."""
    if device not in _CONSTANTS_CACHE:
        _CONSTANTS_CACHE[device] = {
            "nodes_xy": torch.from_numpy(NODES_XY).to(device=device, dtype=DTYPE),
            "weights_xy": torch.from_numpy(WEIGHTS_XY).to(device=device, dtype=DTYPE),
            "nodes_theta": torch.from_numpy(NODES_THETA).to(device=device, dtype=DTYPE),
            "weights_theta": torch.from_numpy(WEIGHTS_THETA).to(device=device, dtype=DTYPE),
            "sqrt_2": torch.sqrt(torch.tensor(2.0, device=device, dtype=DTYPE)),
            "sqrt_2pi": torch.sqrt(torch.tensor(2.0 * torch.pi, device=device, dtype=DTYPE)),
            "inv_sqrt_8pi": 1.0 / torch.sqrt(torch.tensor(8.0 * torch.pi, device=device, dtype=DTYPE)),
        }
    return _CONSTANTS_CACHE[device]


def point_collvol_mc_traj_sampling(
    corners_obs1: list[torch.Tensor] | torch.Tensor,
    initial_state1: list[torch.Tensor] | torch.Tensor,
    initial_state2: list[torch.Tensor] | torch.Tensor,
    initial_cov1: list[torch.Tensor] | torch.Tensor = None,
    initial_cov2: list[torch.Tensor] | torch.Tensor = None,
    corners_obs2: list[torch.Tensor] | torch.Tensor = None,
    dt: float = 0.1,
    num_traj_steps: int = 30,
    with_orientation: bool = False,
    num_samples: int = 20000,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    GPU-accelerated trajectory collision probability estimation using Monte Carlo sampling.

    Estimates collision probability over trajectories for each combination of trajectories
    using vectorized operations on GPU.

    Args:
        corners_obs1: Corner coordinates of first object's polygon (N, n_corners, 2/3)
        initial_state1: Initial state of first object (N, state_dim) or (N, 1, state_dim)
        initial_state2: Initial state of second object (M, state_dim) or (1, M, state_dim)
        initial_cov1: Initial covariance of first object (N, state_dim, state_dim)
        initial_cov2: Initial covariance of second object (M, state_dim, state_dim)
        corners_obs2: Corner coordinates of second object's polygon (M, n_corners, 2/3)
        dt: Time step for trajectory propagation
        num_traj_steps: Number of time steps to propagate
        with_orientation: Whether to consider orientation in collision checking
        num_samples: Number of Monte Carlo samples
        device: PyTorch device (GPU/CPU)

    Returns:
        coll_probs: Estimated collision probabilities (N, M)
        p_cums: Cumulative collision probabilities over time (N, M, num_traj_steps)
        info: Dictionary with sampled trajectories and overlap mask
    """
    device = get_device(device)

    # Convert inputs to tensors
    corners_obs1 = to_torch(corners_obs1, device)
    initial_state1 = to_torch(initial_state1, device)
    initial_state2 = to_torch(initial_state2, device)

    if initial_cov1 is not None:
        initial_cov1 = to_torch(initial_cov1, device)
    if initial_cov2 is not None:
        initial_cov2 = to_torch(initial_cov2, device)
    if corners_obs2 is not None:
        corners_obs2 = to_torch(corners_obs2, device)

    # Handle dimensions
    if initial_state1.ndim == 1:
        initial_state1 = initial_state1.unsqueeze(0)
        if corners_obs1.ndim == 2:
            corners_obs1 = corners_obs1.unsqueeze(0)

    if initial_state1.ndim == 2:
        initial_state1 = initial_state1.unsqueeze(1)  # (N, 1, state_dim)

    if initial_state2.ndim == 1:
        initial_state2 = initial_state2.unsqueeze(0).unsqueeze(0)
    elif initial_state2.ndim == 2:
        initial_state2 = initial_state2.unsqueeze(0)  # (1, M, state_dim)

    # Handle covariances
    if initial_cov1 is not None:
        if initial_cov1.ndim == 2:
            initial_cov1 = initial_cov1.unsqueeze(0)
        if initial_cov1.ndim == 3:
            initial_cov1 = initial_cov1.unsqueeze(1)  # (N, 1, state_dim, state_dim)

    if initial_cov2 is not None:
        if initial_cov2.ndim == 2:
            initial_cov2 = initial_cov2.unsqueeze(0)
        if initial_cov2.ndim == 3:
            initial_cov2 = initial_cov2.unsqueeze(0)  # (1, M, state_dim, state_dim)

    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2.unsqueeze(0)

    # Suppress orientation uncertainty if not needed
    if not with_orientation:
        if initial_cov1 is not None:
            initial_cov1 = initial_cov1.clone()
            initial_cov1[..., 2, 2] = 1e-8
        if initial_cov2 is not None:
            initial_cov2 = initial_cov2.clone()
            initial_cov2[..., 2, 2] = 1e-8

    # Sample initial states
    if initial_cov2 is not None and torch.any(initial_cov2 != 0):
        dist2 = MultivariateNormal(
            loc=initial_state2,  # (1, M, state_dim)
            covariance_matrix=initial_cov2,  # (1, M, state_dim, state_dim)
        )
        points2 = dist2.sample((num_samples,))  # (num_samples, 1, M, state_dim)
    else:
        points2 = initial_state2.expand(1, -1, -1, -1)  # (1, 1, M, state_dim)

    if initial_cov1 is not None and torch.any(initial_cov1 != 0):
        dist1 = MultivariateNormal(
            loc=initial_state1,  # (N, 1, state_dim)
            covariance_matrix=initial_cov1,  # (N, 1, state_dim, state_dim)
        )
        points1 = dist1.sample((num_samples,))  # (num_samples, N, 1, state_dim)
    else:
        points1 = initial_state1.expand(1, -1, -1, -1)  # (1, N, 1, state_dim)

    # Propagate trajectories
    trajs2, covs2 = simple_prediction(
        points2,
        initial_cov2 if initial_cov2 is not None else None,
        dt=dt,
        num_steps=num_traj_steps,
        device=device,
    )
    trajs1, covs1 = simple_prediction(
        points1,
        initial_cov1 if initial_cov1 is not None else None,
        dt=dt,
        num_steps=num_traj_steps,
        device=device,
    )

    # Convert velocity from (v, theta) to (vx, vy)
    trajs1, covs1 = transform_velocities_and_covs(trajs1, covs1)
    trajs2, covs2 = transform_velocities_and_covs(trajs2, covs2)

    # Convolve distributions (relative motion)
    rel_trajs, rel_covs = convolve_distributions(trajs1, covs1, trajs2, covs2, dist="Gaussian", device=device)

    # Prepare collision volumes
    obs1_centered = corners_obs1 - corners_obs1.mean(dim=-2, keepdim=True)
    if corners_obs1.ndim == 2:
        obs1_ext = obs1_centered.unsqueeze(0).unsqueeze(0).expand(rel_trajs.shape[1], rel_trajs.shape[2], -1, -1)
    else:
        obs1_ext = obs1_centered.unsqueeze(1).expand(-1, rel_trajs.shape[2], -1, -1)

    if corners_obs2 is not None:
        obs2_centered = corners_obs2 - corners_obs2.mean(dim=-2, keepdim=True)
        obs2_ext = obs2_centered.unsqueeze(0).expand(rel_trajs.shape[1], -1, -1, -1)

        if with_orientation:
            # Compute rotation for orientation
            delta_theta = rel_trajs[..., 0, 2] - corners_obs2[..., 0, -1]
            rot2 = build_rotation_matrices(delta_theta)

            # Rotate obs2
            obs2_ext = obs2_ext @ rot2 
            obs1_ext = obs1_ext.unsqueeze(0).expand(obs2_ext.shape[0], -1, -1, -1, -1)

        # Compute collision volumes
        collvol, _ = collision_polygon(obs1_ext, obs2_ext)
    else:
        collvol = corners_obs1

    # Reshape for vectorized point-in-polygon test
    all_trajs = rel_trajs.reshape(num_samples, -1, *rel_trajs.shape[-2:])
    all_collvols = collvol.reshape(collvol.shape[0], -1, *collvol.shape[-2:])

    traj_points = all_trajs[..., :2]  # (num_samples, N*M, num_traj_steps, 2)
    collvol_polygons = all_collvols[..., :2]  # (num_samples, N*M, n_vertices, 2)

    # Use vectorized point-in-polygon
    overlap = point_in_polygon_batch_vectorized(
        traj_points,
        collvol_polygons, 
    )  

    # Reshape back
    overlap = overlap.reshape(num_samples, rel_trajs.shape[1], rel_trajs.shape[2], rel_trajs.shape[-2])

    # Compute entry events (transitions from outside to inside)
    # Prepend False to detect first entry
    overlap_padded = torch.cat([torch.zeros_like(overlap[..., :1], device=device, dtype=torch.bool), overlap], dim=-1)

    entered = torch.diff(overlap_padded.int(), dim=-1) == 1  # (num_samples, N, M, num_traj_steps)

    # Count new entries per timestep
    new_entries_per_timestep = entered.sum(dim=0)  # (N, M, num_traj_steps)

    # Probability of new entries per timestep
    v1 = new_entries_per_timestep / num_samples

    # Total collision probability (integrate over time)
    coll_probs = v1.sum(dim=-1)  # (N, M)

    # Probability rate
    prob_rate = v1 / dt

    # Cumulative probability over time
    p_cums = torch.cumsum(prob_rate * dt, dim=-1).clamp(max=1.0)  # (N, M, num_traj_steps)

    # Inside mask (any timestep inside)
    inside_mask = overlap.any(dim=-1).permute(1, 2, 0)  # .permute(2, 0, 1)  # (num_samples, N, M)

    return (
        coll_probs,
        p_cums,
        {
            "trajs": rel_trajs,
            "overlap_mask": inside_mask,
        },
    )



def boundary_crossing(
    corners_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs2: list[torch.Tensor] | torch.Tensor,
    trajs_obs1_covs: list[torch.Tensor] | torch.Tensor = None,
    trajs_obs2_covs: list[torch.Tensor] | torch.Tensor = None,
    corners_obs2: list[torch.Tensor] | torch.Tensor = None,
    with_orientation: bool = False,
    dt: float = 0.1,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated boundary crossing method for collision probability estimation.

    This method computes collision probability by analyzing boundary crossings of the
    collision volume formed by relative motion, using Gauss-Legendre quadrature for
    numerical integration along polygon edges.

    Args:
        corners_obs1: Corner coordinates of first object's polygon (N, n_corners, 3)
        trajs_obs1: Trajectory of first object (N, num_steps, state_dim)
        trajs_obs2: Trajectory of second object (M, num_steps, state_dim)
        trajs_obs1_covs: Covariance matrices along first trajectory (N, num_steps, state_dim, state_dim)
        trajs_obs2_covs: Covariance matrices along second trajectory (M, num_steps, state_dim, state_dim)
        corners_obs2: Corner coordinates of second object's polygon (M, n_corners, 3)
        with_orientation: Whether to consider orientation uncertainty
        dt: Time step
        device: PyTorch device (GPU/CPU)

    Returns:
        coll_probs_full: Collision probabilities (N, M)
        cep: Collision entry probability rates (N, M, num_steps)
    """
    device = get_device(device)
    constants = get_cached_constants(device)
    # Convert to tensors
    corners_obs1 = to_torch(corners_obs1, device)
    trajs1 = to_torch(trajs_obs1, device)
    trajs2 = to_torch(trajs_obs2, device)
    if trajs_obs1_covs is not None:
        trajs1_covs = to_torch(trajs_obs1_covs, device)
    else:
        trajs1_covs = None

    if trajs_obs2_covs is not None:
        trajs2_covs = to_torch(trajs_obs2_covs, device)
    else:
        trajs2_covs = None

    if corners_obs2 is not None:
        corners_obs2 = to_torch(corners_obs2, device)
    else:
        corners_obs2 = None

    # Handle dimensions
    if trajs1.ndim == 2:
        trajs1 = trajs1.unsqueeze(0)
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1.unsqueeze(0)

    trajs1 = trajs1.unsqueeze(1)  # (N, 1, num_steps, state_dim)

    if trajs2.ndim == 2:
        trajs2 = trajs2.unsqueeze(0)
    if trajs2.ndim == 3:
        trajs2 = trajs2.unsqueeze(0)  # (1, M, num_steps, state_dim)

    # Handle covariances
    if trajs1_covs is not None:
        if trajs1_covs.ndim == 3:
            trajs1_covs = trajs1_covs.unsqueeze(0)
        trajs1_covs = trajs1_covs.unsqueeze(1)  # (N, 1, num_steps, state_dim, state_dim)

    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs.unsqueeze(0)
        trajs2_covs = trajs2_covs.unsqueeze(0)  # (1, M, num_steps, state_dim, state_dim)

    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2.unsqueeze(0).expand(trajs2.shape[1], -1, -1)

    # Suppress orientation uncertainty if not needed
    if not with_orientation:
        if trajs1_covs is not None:
            trajs1_covs = trajs1_covs.clone()
            trajs1_covs[..., 2, 2] = 1e-8
        if trajs2_covs is not None:
            trajs2_covs = trajs2_covs.clone()
            trajs2_covs[..., 2, 2] = 1e-8

    # Transform velocities from (v, theta) to (vx, vy)
    trajs1, trajs1_covs = transform_velocities_and_covs(trajs1, trajs1_covs)
    trajs2, trajs2_covs = transform_velocities_and_covs(trajs2, trajs2_covs)

    # Convolve distributions (relative motion)
    rel_trajs, rel_covs = convolve_distributions(
        trajs1, trajs1_covs, trajs2, trajs2_covs, dist="Gaussian", device=device
    )

    # Early rejection mask
    d1 = compute_max_diagonal(corners_obs1)

    if corners_obs2 is not None and torch.any(corners_obs2 != 0):
        d2 = compute_max_diagonal(corners_obs2)
        d_max = d1.unsqueeze(-1) + d2.unsqueeze(0)
        if d_max.ndim < rel_trajs.ndim - 1:
            d_max = d_max.unsqueeze(-1)
    else:
        d_max = d1.unsqueeze(1).unsqueeze(-1)

    rel_pos_sq = rel_trajs[..., 0] ** 2 + rel_trajs[..., 1] ** 2
    cov_max = rel_covs[..., :2, :2].amax(dim=(-2, -1))
    mask = rel_pos_sq - d_max**2 <= 25 * cov_max

    # Initialize outputs
    coll_probs_full = torch.zeros(rel_trajs.shape[:2], device=device, dtype=rel_trajs.dtype)
    cep = torch.zeros(*rel_trajs.shape[:2], rel_trajs.shape[2], device=device, dtype=rel_trajs.dtype)

    if torch.any(mask):
        # Center polygons
        obs1 = corners_obs1 - corners_obs1.mean(dim=-2, keepdim=True)
        obs1_ext = (
            obs1.unsqueeze(1).unsqueeze(2).expand(rel_trajs.shape[0], rel_trajs.shape[1], rel_trajs.shape[2], -1, -1)
        )

        # Filter by mask (any timestep)
        mask_conc = mask.any(dim=-1)  # (N, M)

        if torch.any(mask_conc):
            filtered_trajs = rel_trajs[mask_conc]  # (num_valid, num_steps, 5)
            filtered_covs = rel_covs[mask_conc]  # (num_valid, num_steps, 5, 5)
            obs1_filtered = obs1_ext[mask_conc]  # (num_valid, num_steps, n_corners, 3)

            theta_mean = filtered_trajs[..., 2]  # (num_valid, num_steps)

            if corners_obs2 is not None:
                # Center second polygon
                obs2 = corners_obs2 - corners_obs2.mean(dim=-2, keepdim=True)
                obs2[..., -1] = corners_obs2[..., -1]  # Preserve orientation

                # Ensure 4D: (M, vertices, 3) -> (M, 1, vertices, 3)
                if obs2.ndim == 3:
                    obs2 = obs2.unsqueeze(1)

                # Broadcast to (N, M, timesteps, vertices, 3)
                obs2_ext = obs2.unsqueeze(0).expand(rel_trajs.shape[0], -1, rel_trajs.shape[2], -1, -1)

                obs2_filtered = obs2_ext[mask_conc]  # (num_valid, num_steps, n_corners, 3)
                ini_obs_theta = obs2_filtered[:, :, 0, -1]  # (num_valid, num_steps)

                # Compute rotation
                delta_theta = theta_mean - ini_obs_theta
                rot2 = build_rotation_matrices(delta_theta)

                # Rotate obs2
                obs_rot2 = obs2_filtered @ rot2
                obs_rot2[..., -1] += delta_theta.unsqueeze(-1)

                # Handle orientation uncertainty with Gauss-Legendre quadrature
                if with_orientation and filtered_covs is not None and filtered_covs.shape[-1] >= 5:
                    theta_std = torch.sqrt(filtered_covs[..., 2, 2])  # (num_valid, num_steps)

                    # Convert nodes to theta range
                    nodes_theta = constants["nodes_theta"]

                    delta_theta_gauss = THETA_NSTD * nodes_theta[:, None, None] * theta_std
                    theta_gauss = delta_theta_gauss + theta_mean
                    theta_jacobian = THETA_NSTD * theta_std

                    # Build rotation matrices for Gauss points
                    rot_gauss = build_rotation_matrices(delta_theta_gauss)

                    # Rotate collision volume
                    obs_rot2 = obs_rot2 @ rot_gauss
                    obs_rot2[..., -1] += delta_theta_gauss.unsqueeze(-1)

                    # Expand obs1
                    obs1_filtered = obs1_filtered.unsqueeze(0).expand(obs_rot2.shape[0], -1, -1, -1, -1)

                # Compute collision volume (Minkowski sum)
                collvol, _ = collision_polygon(obs1_filtered, obs_rot2)
            else:
                # TODO!
                raise NotImplementedError("boundary_crossing requires corners_obs2")

            # Extract position and velocity covariance components
            cov_x = filtered_covs[..., :3, :3]  # Position covariance
            cov_v = filtered_covs[..., 3:, 3:]  # Velocity covariance
            cov_xv = filtered_covs[..., :3, 3:]  # Cross-covariance
            cov_vx = filtered_covs[..., 3:, :3]

            # Kalman gain: K = cov_vx @ inv(cov_x)
            K = torch.linalg.solve(cov_x.transpose(-2, -1), cov_vx.transpose(-2, -1)).transpose(-2, -1)

            # Conditional velocity covariance: cov(v|x) = cov_v - K @ cov_xv
            cov_vifx = cov_v - K @ cov_xv

            # Ensure symmetry and positive definiteness
            cov_vifx = 0.5 * (cov_vifx + cov_vifx.transpose(-2, -1))
            cov_vifx = cov_vifx + 1e-12 * torch.eye(cov_vifx.shape[-1], device=device, dtype=cov_vifx.dtype)

            mu_pos = filtered_trajs[..., :3]  # Position mean
            mu_v = filtered_trajs[..., 3:]  # Velocity mean

            # Shift collision volume by trajectory mean
            delta_pos = collvol - mu_pos.unsqueeze(-2)

            # Compute outward normals of polygon edges
            p1 = delta_pos
            p2 = torch.roll(p1, -1, dims=-2)

            side_vecs = p1[..., :2] - p2[..., :2]
            edge_lengths = torch.sqrt((side_vecs**2).sum(dim=-1))  # torch.hypot(side_vecs[..., 0], side_vecs[..., 1])

            # Outward normal: (-dy, dx) / length
            normals = torch.stack([-side_vecs[..., 1], side_vecs[..., 0]], dim=-1) / edge_lengths.unsqueeze(-1)
            # # normals = normals / edge_lengths.unsqueeze(-1)

            # Gauss-Legendre quadrature
            nodes_xy = constants["nodes_xy"]
            weights_xy = constants["weights_xy"]
            edge_coeff1_exp = 0.5 * (1 - nodes_xy).view(-1, *[1] * (p1.ndim))  # (num_quad, 1, ..., 1, 1)
            edge_coeff2_exp = 0.5 * (1 + nodes_xy).view(-1, *[1] * (p2.ndim))  # (num_quad, 1, ..., 1, 1)

            # Broadcast multiply and add
            edge_points = edge_coeff1_exp * p1 + edge_coeff2_exp * p2

            # Conditional velocity at each edge point
            mu_vifx_edge = mu_v.unsqueeze(0).unsqueeze(-2) + edge_points @ K.mT

            # Velocity component along normal
            v_n_mean = (mu_vifx_edge * normals.unsqueeze(0)).sum(dim=-1)  # (num_nodes, ..., n_edges)

            # Variance of velocity along normal
            v_n_var = ((normals @ cov_vifx) * normals).sum(dim=-1).squeeze(-1)
            v_n_var = torch.clamp(v_n_var, min=1e-8)
            v_n_std = torch.sqrt(v_n_var).unsqueeze(0)

            # Truncated Gaussian expected inward velocity
            sqrt_2 = constants["sqrt_2"]
            sqrt_2pi = constants["sqrt_2pi"]
            a = -v_n_mean / v_n_std
            F_vn = 0.5 * (1 + torch.erf(a / sqrt_2))
            phi = (1 / sqrt_2pi) * torch.exp(-0.5 * a * a)
            E_vn_trunc = -v_n_mean * F_vn + v_n_std * phi

            # PDF of position at each edge point
            delta_pdf = edge_points[..., :2]  # (num_nodes, ..., n_edges, 2)

            # Marginalize orientation: cov(x,y | theta)
            cov_xy = cov_x[..., :2, :2]
            cov_xy_theta = cov_x[..., :2, 2:3]
            cov_theta = cov_x[..., 2:3, 2:3]

            cov_xiftheta = cov_xy + (cov_xy_theta @ cov_xy_theta.mT) / cov_theta
            try:
                L = torch.linalg.cholesky(cov_xiftheta)
                inv_cov_x = torch.cholesky_inverse(L)
                det_cov_x = L.diagonal(dim1=-2, dim2=-1).prod(dim=-1) ** 2
            except:
                inv_cov_x = torch.linalg.inv(cov_xiftheta)
                det_cov_x = torch.linalg.det(cov_xiftheta)

            # Mahalanobis distance
            mahal = ((delta_pdf @ inv_cov_x) * delta_pdf).sum(dim=-1)

            # Position PDF
            p_x = torch.exp(-0.5 * mahal) / (2 * torch.pi * torch.sqrt(det_cov_x.unsqueeze(0).unsqueeze(-1)))

            # Integrand
            integrand = E_vn_trunc * p_x

            # Integrate along edges using Gauss-Legendre quadrature
            weights_expanded = weights_xy.view(-1, *[1] * (integrand.ndim - 1))

            cep_loc = (integrand * weights_expanded).sum(dim=0) * (0.5 * edge_lengths)
            cep_loc = cep_loc.sum(dim=-1)

            # Handle orientation uncertainty
            if with_orientation:
                # Use cached constants for theta PDF
                sqrt_2pi = constants["sqrt_2pi"]
                z_value = (theta_gauss - theta_mean) / theta_std
                theta_pdf = torch.exp(-0.5 * z_value * z_value) / (theta_std * sqrt_2pi)

                weights_theta_expanded = constants["weights_theta"].view(-1, *[1] * (theta_pdf.ndim - 1))

                cep_loc = (cep_loc * theta_pdf * weights_theta_expanded * theta_jacobian).sum(dim=0)

            # Integrate over time
            coll_prob = torch.trapezoid(cep_loc, dx=dt, dim=-1)
            coll_prob = torch.clamp(coll_prob, max=1.0)

            # Assign back to full tensors
            coll_probs_full[mask_conc] = coll_prob
            cep[mask_conc] = cep_loc

    return coll_probs_full, cep


def point_collvol_mc_state_sampling(
    corners_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs2: list[torch.Tensor] | torch.Tensor,
    trajs_obs1_covs: list[torch.Tensor] | torch.Tensor = None,
    trajs_obs2_covs: list[torch.Tensor] | torch.Tensor = None,
    corners_obs2: list[torch.Tensor] | torch.Tensor = None,
    with_orientation: bool = False,
    num_samples: int = 1000,
    dt=None,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated Monte Carlo state sampling for trajectory collision probability estimation.

    Estimates collision probability at each time step along trajectories using vectorized
    Monte Carlo sampling on GPU.

    Args:
        corners_obs1: Corner coordinates of first object's polygon (N, n_corners, 2/3)
        trajs_obs1: Trajectory of first object (N, num_steps, state_dim)
        trajs_obs1_covs: Covariance matrices along first trajectory (N, num_steps, state_dim, state_dim)
        trajs_obs2: Trajectory of second object (M, num_steps, state_dim)
        trajs_obs2_covs: Covariance matrices along second trajectory (M, num_steps, state_dim, state_dim)
        corners_obs2: Corner coordinates of second object's polygon (M, n_corners, 2/3)
        with_orientation: Whether to consider orientation in collision checking
        num_samples: Number of Monte Carlo samples
        dt: Time step (unused, for API consistency)
        device: PyTorch device (GPU/CPU)

    Returns:
        coll_probs_max: Maximum collision probability over time (N, M)
        coll_probs_full: Collision probability at each timestep (N, M, num_steps)
    """
    device = get_device(device)

    # Convert to tensors and marginalize to position + orientation only
    corners_obs1 = to_torch(corners_obs1, device)
    trajs1 = to_torch(trajs_obs1, device)[..., :3]

    if trajs_obs2 is not None:
        trajs2 = to_torch(trajs_obs2, device)[..., :3]
    else:
        trajs2 = None

    if trajs_obs1_covs is not None:
        trajs1_covs = to_torch(trajs_obs1_covs, device)[..., :3, :3]
    else:
        trajs1_covs = None

    if trajs_obs2_covs is not None:
        trajs2_covs = to_torch(trajs_obs2_covs, device)[..., :3, :3]
    else:
        trajs2_covs = None

    if corners_obs2 is not None:
        corners_obs2 = to_torch(corners_obs2, device)
    else:
        corners_obs2 = None

    # Handle dimensions
    if trajs1.ndim == 2:
        trajs1 = trajs1.unsqueeze(0)
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1.unsqueeze(0)

    trajs1 = trajs1.unsqueeze(1)  # (N, 1, num_steps, 3)

    if trajs2 is not None:
        if trajs2.ndim == 2:
            trajs2 = trajs2.unsqueeze(0)
        if trajs2.ndim == 3:
            trajs2 = trajs2.unsqueeze(0)  # (1, M, num_steps, 3)

    # Handle covariances
    if trajs1_covs is not None:
        if trajs1_covs.ndim == 3:
            trajs1_covs = trajs1_covs.unsqueeze(0)
        trajs1_covs = trajs1_covs.unsqueeze(1)  # (N, 1, num_steps, 3, 3)

    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs.unsqueeze(0)
        trajs2_covs = trajs2_covs.unsqueeze(0)  # (1, M, num_steps, 3, 3)

    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2.unsqueeze(0).expand(trajs2.shape[1], -1, -1)

    assert trajs1.ndim == 4, f"trajs1 must be 4D, got {trajs1.shape}"
    if trajs2 is not None:
        assert trajs2.ndim == 4, f"trajs2 must be 4D, got {trajs2.shape}"
        assert trajs1.shape[-2:] == trajs2.shape[-2:], "Trajectory dimensions must match"

    # Convolve distributions (relative motion)
    rel_trajs, rel_covs = convolve_distributions(
        trajs1, trajs1_covs, trajs2, trajs2_covs, dist="Gaussian", device=device
    )

    # Compute maximum diagonal distances for early rejection
    d1 = compute_max_diagonal(corners_obs1)  # (N,)

    if corners_obs2 is not None and torch.any(corners_obs2 != 0):
        d2 = compute_max_diagonal(corners_obs2)  # (M,)
        d_max = d1.unsqueeze(-1) + d2.unsqueeze(0)
        if d_max.ndim < rel_trajs.ndim - 1:
            d_max = d_max.unsqueeze(-1)
    else:
        d_max = d1.unsqueeze(1).unsqueeze(-1)  # (N, 1, 1)

    # Early rejection mask: check if polygons are too far apart
    rel_pos_norm = torch.hypot(rel_trajs[..., 0], rel_trajs[..., 1])  # (N, M, num_steps)
    cov_max = rel_covs[..., :2, :2].amax(dim=(-2, -1))  # (N, M, num_steps)

    # Conservative mask: keep pairs within 5 sigma
    mask = (rel_pos_norm - d_max) <= 5 * torch.sqrt(cov_max)  # (N, M, num_steps)

    # Initialize output
    coll_probs_full = torch.zeros_like(mask, dtype=rel_trajs.dtype)

    # Process only masked (potentially colliding) states
    if torch.any(mask):
        # Center polygons
        obs1 = corners_obs1 - corners_obs1.mean(dim=-2, keepdim=True)  # (N, n_corners, 3)

        # Broadcast to all trajectory combinations
        obs1_ext = obs1.unsqueeze(1).unsqueeze(2).expand(*rel_trajs.shape[:3], -1, -1)

        # Filter based on mask
        obs1_filtered = obs1_ext[mask]  # (num_valid, n_corners, 3)
        filtered_trajs = rel_trajs[mask]  # (num_valid, 3)
        filtered_covs = rel_covs[mask]  # (num_valid, 3, 3)

        theta_mean = filtered_trajs[..., 2]  # (num_valid,)

        if corners_obs2 is not None:
            # Center second polygon
            obs2 = corners_obs2 - corners_obs2.mean(dim=-2, keepdim=True)  # (M, n_corners, 3)
            obs2[..., -1] = corners_obs2[..., -1]  # Preserve original orientation

            if obs2.ndim < 4:
                obs2 = obs2.unsqueeze(1)
            obs2_ext = obs2.unsqueeze(0).expand(*rel_trajs.shape[:3], *obs2.shape[-2:])
            obs2_filtered = obs2_ext[mask]  # (num_valid, n_corners, 3)
            ini_obs_theta = obs2_filtered[:, 0, -1]  # (num_valid,)

            # Compute rotation
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)  # (num_valid, 3, 3)

            # Rotate and translate obs2
            obs2_filtered = obs2_filtered @ rot2
            # Add relative position (only x, y, theta - first 3 components)
            obs2_filtered += filtered_trajs.unsqueeze(-2)[..., :3]
        else:
            obs2_filtered = filtered_trajs

        # Monte Carlo sampling for filtered states
        coll_probs, _ = point_collvol_mc_sampling(
            obs1=obs1_filtered,
            obs2=obs2_filtered,
            obs1_cov=filtered_covs,
            obs2_cov=None,
            with_orientation=with_orientation,
            num_samples=num_samples,
            device=device,
        )

        # Assign back to full tensor
        coll_probs_full[mask] = coll_probs

    # Maximum collision probability over time
    coll_probs_max = coll_probs_full.amax(dim=-1)  # (N, M)

    return coll_probs_max, coll_probs_full


def traj_collvol_analytic(
    corners_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs2: list[torch.Tensor] | torch.Tensor,
    trajs_obs1_covs: list[torch.Tensor] | torch.Tensor = None,
    trajs_obs2_covs: list[torch.Tensor] | torch.Tensor = None,
    corners_obs2: list[torch.Tensor] | torch.Tensor = None,
    with_orientation: bool = False,
    dt: float = 0.1,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated analytic method for trajectory collision probability estimation.

    Computes collision probability at each time step using analytic point-in-polygon
    probability calculation.

    Args:
        corners_obs1: Corner coordinates of first object (..., n_corners, 3)
        trajs_obs1: Trajectory of first object (..., n_steps, 3)
        trajs_obs2: Trajectory of second object (..., n_steps, 3)
        trajs_obs1_covs: Covariance matrices for first trajectory (..., n_steps, 3, 3)
        trajs_obs2_covs: Covariance matrices for second trajectory (..., n_steps, 3, 3)
        corners_obs2: Corner coordinates of second object (..., n_corners, 3)
        with_orientation: Whether to consider orientation
        dt: Time step (unused, for API consistency)
        device: Torch device (GPU/CPU)

    Returns:
        coll_prob: Maximum collision probability over time (..., )
        coll_probs_full: Collision probabilities at each time step (..., n_steps)
    """
    device = get_device(device)

    # Convert to torch tensors
    corners_obs1 = to_torch(corners_obs1, device)
    trajs1 = to_torch(trajs_obs1, device)[..., :3]
    trajs2 = to_torch(trajs_obs2, device)[..., :3]

    if trajs_obs1_covs is not None:
        trajs1_covs = to_torch(trajs_obs1_covs, device)[..., :3, :3]
    else:
        trajs1_covs = torch.zeros(*trajs1.shape[:-1], 3, 3, dtype=trajs1.dtype, device=device)

    if trajs_obs2_covs is not None:
        trajs2_covs = to_torch(trajs_obs2_covs, device)[..., :3, :3]
    else:
        trajs2_covs = None

    if corners_obs2 is not None:
        corners_obs2 = to_torch(corners_obs2, device)

    # Ensure proper dimensionality
    if trajs1.ndim == 2:
        trajs1 = trajs1.unsqueeze(0)
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1.unsqueeze(0)
    trajs1 = trajs1.unsqueeze(1)

    if trajs2.ndim == 2:
        trajs2 = trajs2.unsqueeze(0).unsqueeze(0)
    elif trajs2.ndim == 3:
        trajs2 = trajs2.unsqueeze(0)

    if trajs1_covs.ndim == 3:
        trajs1_covs = trajs1_covs.unsqueeze(0)
    trajs1_covs = trajs1_covs.unsqueeze(1)

    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs.unsqueeze(0)
        trajs2_covs = trajs2_covs.unsqueeze(0)

    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2.unsqueeze(0).expand(trajs2.shape[0], -1, -1)

    if not with_orientation:
        trajs1_covs[..., 2, 2] = 1e-8
        if trajs2_covs is not None:
            trajs2_covs[..., 2, 2] = 1e-8

    # Convolve distributions
    rel_trajs, rel_covs = convolve_distributions(trajs1, trajs1_covs, trajs2, trajs2_covs)

    # Compute maximum diagonals
    d1 = compute_max_diagonal(corners_obs1)
    if corners_obs2 is not None:
        d2 = compute_max_diagonal(corners_obs2)
        d_max = d1.unsqueeze(-1) + d2.unsqueeze(0)
        if d_max.ndim < rel_trajs.ndim - 1:
            d_max = d_max.unsqueeze(-1)
    else:
        d_max = d1.unsqueeze(-1)

    # Compute mask for potential collisions
    rel_pos_norm = torch.hypot(rel_trajs[..., 0], rel_trajs[..., 1])
    cov_max = rel_covs[..., :2, :2].amax(dim=(-2, -1))
    mask = (rel_pos_norm - d_max) <= 5 * torch.sqrt(cov_max)

    coll_probs_full = torch.zeros(mask.shape, dtype=trajs1.dtype, device=device)

    if mask.any():
        # Center polygons
        obs1 = corners_obs1 - corners_obs1.mean(dim=-2, keepdim=True)
        obs1_ext = obs1.view(obs1.shape[0], 1, 1, *obs1.shape[1:]).expand(*rel_trajs.shape[:3], *obs1.shape[1:])

        obs1_filtered = obs1_ext[mask]
        filtered_trajs = rel_trajs[mask]
        filtered_covs = rel_covs[mask]
        theta_mean = filtered_trajs[..., 2]

        if corners_obs2 is not None:
            obs2 = corners_obs2 - corners_obs2.mean(dim=-2, keepdim=True)
            obs2[..., -1] = corners_obs2[..., -1]

            if obs2.ndim < 4:
                obs2 = obs2.unsqueeze(1)
            # obs2_ext = obs2.unsqueeze(2).expand(*rel_trajs.shape[:3], *obs2.shape[-2:])
            obs2_ext = obs2.unsqueeze(0).expand(*rel_trajs.shape[:3], *obs2.shape[-2:])
            obs2_filtered = obs2_ext[mask]

            ini_obs_theta = obs2_filtered[..., 0, -1]
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)
            obs2_filtered = obs2_filtered @ rot2
            obs2_filtered += filtered_trajs.unsqueeze(-2)[..., :3]
        else:
            obs2_filtered = filtered_trajs

        coll_probs = point_collvol_analytic(
            obs1_filtered, obs2_filtered, filtered_covs, with_orientation=with_orientation, device=device
        )

        coll_probs_full[mask] = coll_probs.to(dtype=coll_probs_full.dtype, device=coll_probs_full.device)

    return coll_probs_full.amax(dim=-1), coll_probs_full


def traj_collvol_deterministic( # TODO should work without considering covs -> have look at point_collvol_mc_sampling when cov is none
    corners_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs2: list[torch.Tensor] | torch.Tensor,
    trajs_obs1_covs: list[torch.Tensor] | torch.Tensor = None,
    trajs_obs2_covs: list[torch.Tensor] | torch.Tensor = None,
    corners_obs2: list[torch.Tensor] | torch.Tensor = None,
    with_orientation: bool = False,
    dt: float = 0.1,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("needs revision and integration to numpy!")
    """Fast deterministic collision check when there's no uncertainty."""
    # Simple point-in-polygon test

    # Convert to torch tensors
    corners_obs1 = to_torch(corners_obs1, device)
    trajs1 = to_torch(trajs_obs1, device)[..., :3]
    trajs2 = to_torch(trajs_obs2, device)[..., :3]

    if trajs_obs1_covs is not None:
        trajs1_covs = to_torch(trajs_obs1_covs, device)[..., :3, :3]
    else:
        trajs1_covs = torch.zeros(*trajs1.shape[:-1], 3, 3, dtype=trajs1.dtype, device=device)

    if trajs_obs2_covs is not None:
        trajs2_covs = to_torch(trajs_obs2_covs, device)[..., :3, :3]
    else:
        trajs2_covs = None

    if corners_obs2 is not None:
        corners_obs2 = to_torch(corners_obs2, device)

    # Ensure proper dimensionality
    if trajs1.ndim == 2:
        trajs1 = trajs1.unsqueeze(0)
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1.unsqueeze(0)
    trajs1 = trajs1.unsqueeze(1)

    if trajs2.ndim == 2:
        trajs2 = trajs2.unsqueeze(0).unsqueeze(0)
    elif trajs2.ndim == 3:
        trajs2 = trajs2.unsqueeze(0)

    if trajs1_covs.ndim == 3:
        trajs1_covs = trajs1_covs.unsqueeze(0)
    trajs1_covs = trajs1_covs.unsqueeze(1)

    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs.unsqueeze(0)
        trajs2_covs = trajs2_covs.unsqueeze(0)

    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2.unsqueeze(0).expand(trajs2.shape[0], -1, -1)

    if not with_orientation:
        trajs1_covs[..., 2, 2] = 1e-8
        if trajs2_covs is not None:
            trajs2_covs[..., 2, 2] = 1e-8

    # Convolve distributions
    rel_trajs, rel_covs = convolve_distributions(trajs1, trajs1_covs, trajs2, trajs2_covs)

    # Compute maximum diagonals
    d1 = compute_max_diagonal(corners_obs1)
    if corners_obs2 is not None:
        d2 = compute_max_diagonal(corners_obs2)
        d_max = d1.unsqueeze(-1) + d2.unsqueeze(0)
        if d_max.ndim < rel_trajs.ndim - 1:
            d_max = d_max.unsqueeze(-1)
    else:
        d_max = d1.unsqueeze(-1)

    # Compute mask for potential collisions
    rel_pos_norm = torch.hypot(rel_trajs[..., 0], rel_trajs[..., 1])
    cov_max = rel_covs[..., :2, :2].amax(dim=(-2, -1))
    mask = (rel_pos_norm - d_max) <= 5 * torch.sqrt(cov_max)

    coll_probs_full = torch.zeros(mask.shape, dtype=trajs1.dtype, device=device)

    if mask.any():
        # Center polygons
        obs1 = corners_obs1 - corners_obs1.mean(dim=-2, keepdim=True)
        obs1_ext = obs1.view(obs1.shape[0], 1, 1, *obs1.shape[1:]).expand(*rel_trajs.shape[:3], *obs1.shape[1:])

        obs1_filtered = obs1_ext[mask]
        filtered_trajs = rel_trajs[mask]
        filtered_covs = rel_covs[mask]
        theta_mean = filtered_trajs[..., 2]

        if corners_obs2 is not None:
            obs2 = corners_obs2 - corners_obs2.mean(dim=-2, keepdim=True)
            obs2[..., -1] = corners_obs2[..., -1]

            if obs2.ndim < 4:
                obs2 = obs2.unsqueeze(1)
            # obs2_ext = obs2.unsqueeze(2).expand(*rel_trajs.shape[:3], *obs2.shape[-2:])
            obs2_ext = obs2.unsqueeze(0).expand(*rel_trajs.shape[:3], *obs2.shape[-2:])
            obs2_filtered = obs2_ext[mask]

            ini_obs_theta = obs2_filtered[..., 0, -1]
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)
            obs2_filtered = obs2_filtered @ rot2
            obs2_filtered += filtered_trajs.unsqueeze(-2)[..., :3]
        else:
            obs2_filtered = rel_trajs[mask]

        collvol, points = collision_polygon(obs1_filtered, obs2_filtered)

        # Vectorized point-in-polygon
        inside = point_in_polygon_batch_vectorized(points.squeeze(), collvol[..., :2])

        coll_probs_full[mask] = inside.to(dtype=trajs1.dtype)
        coll_probs_max = coll_probs_full.amax(dim=-1)

    return coll_probs_max, coll_probs_full


def traj_rect_incl_excl(
    corners_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs2: list[torch.Tensor] | torch.Tensor,
    trajs_obs1_covs: list[torch.Tensor] | torch.Tensor = None,
    trajs_obs2_covs: list[torch.Tensor] | torch.Tensor = None,
    corners_obs2: list[torch.Tensor] | torch.Tensor = None,
    with_orientation: bool = False,
    dt = None,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated analytic method for trajectory collision probability estimation using rectangle inclusion-exclusion.    
    
    Computes collision probabilities between two trajectories with rectangular cross-sections by convolving
    their Gaussian distributions and applying inclusion-exclusion principle.
    Args:
        corners_obs1: Corner coordinates of first object (..., n_corners, 3)
        trajs_obs1: Trajectory of first object (..., n_steps, 3)
        trajs_obs2: Trajectory of second object (..., n_steps, 3)
        trajs_obs1_covs: Covariance matrices for first trajectory (..., n_steps, 3, 3)
        trajs_obs2_covs: Covariance matrices for second trajectory (..., n_steps, 3, 3)
        corners_obs2: Corner coordinates of second object (..., n_corners, 3)
        with_orientation: Whether to consider orientation (Not applicable for rectangle inclusion-exclusion, included for interface consistency)
        dt: Time step (unused, for API consistency)
        device: Torch device (GPU/CPU)

    Returns:
        coll_prob: Maximum collision probability over time (..., )
        coll_probs_full: Collision probability at each timestep, shape (..., num_timesteps).
    """

    # Convert to torch tensors
    corners_obs1 = to_torch(corners_obs1, device)
    trajs1 = to_torch(trajs_obs1, device)[..., :3]
    trajs2 = to_torch(trajs_obs2, device)[..., :3]

    if trajs_obs1_covs is not None:
        trajs1_covs = to_torch(trajs_obs1_covs, device)[..., :3, :3]
    else:
        trajs1_covs = torch.zeros(*trajs1.shape[:-1], 3, 3, dtype=trajs1.dtype, device=device)

    if trajs_obs2_covs is not None:
        trajs2_covs = to_torch(trajs_obs2_covs, device)[..., :3, :3]
    else:
        trajs2_covs = None

    if corners_obs2 is not None:
        corners_obs2 = to_torch(corners_obs2, device)

    # Ensure proper dimensionality
    if trajs1.ndim == 2:
        trajs1 = trajs1.unsqueeze(0)
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1.unsqueeze(0)
    trajs1 = trajs1.unsqueeze(1)

    if trajs2.ndim == 2:
        trajs2 = trajs2.unsqueeze(0).unsqueeze(0)
    elif trajs2.ndim == 3:
        trajs2 = trajs2.unsqueeze(0)

    if trajs1_covs.ndim == 3:
        trajs1_covs = trajs1_covs.unsqueeze(0)
    trajs1_covs = trajs1_covs.unsqueeze(1)

    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs.unsqueeze(0)
        trajs2_covs = trajs2_covs.unsqueeze(0)

    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2.unsqueeze(0).expand(trajs2.shape[0], -1, -1)

    if not with_orientation:
        trajs1_covs[..., 2, 2] = 1e-8
        if trajs2_covs is not None:
            trajs2_covs[..., 2, 2] = 1e-8

    # Convolve distributions
    rel_trajs, rel_covs = convolve_distributions(trajs1, trajs1_covs, trajs2, trajs2_covs)

    # Compute maximum diagonals
    d1 = compute_max_diagonal(corners_obs1)
    if corners_obs2 is not None:
        d2 = compute_max_diagonal(corners_obs2)
        d_max = d1.unsqueeze(-1) + d2.unsqueeze(0)
        if d_max.ndim < rel_trajs.ndim - 1:
            d_max = d_max.unsqueeze(-1)
    else:
        d_max = d1.unsqueeze(-1)

    # Compute mask for potential collisions
    rel_pos_norm = torch.hypot(rel_trajs[..., 0], rel_trajs[..., 1])
    cov_max = rel_covs[..., :2, :2].amax(dim=(-2, -1))
    mask = (rel_pos_norm - d_max) <= 5 * torch.sqrt(cov_max)

    coll_probs_full = torch.zeros(mask.shape, dtype=trajs1.dtype, device=device)

    if mask.any():
        # Center polygons
        obs1 = corners_obs1 - corners_obs1.mean(dim=-2, keepdim=True)
        obs1_ext = obs1.view(obs1.shape[0], 1, 1, *obs1.shape[1:]).expand(*rel_trajs.shape[:3], *obs1.shape[1:])

        obs1_filtered = obs1_ext[mask]
        filtered_trajs = rel_trajs[mask]
        filtered_covs = rel_covs[mask]
        theta_mean = filtered_trajs[..., 2]

        if corners_obs2 is not None:
            obs2 = corners_obs2 - corners_obs2.mean(dim=-2, keepdim=True)
            obs2[..., -1] = corners_obs2[..., -1]

            if obs2.ndim < 4:
                obs2 = obs2.unsqueeze(1)
            obs2_ext = obs2.unsqueeze(0).expand(*rel_trajs.shape[:3], *obs2.shape[-2:])
            obs2_filtered = obs2_ext[mask]

            ini_obs_theta = obs2_filtered[..., 0, -1]
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)
            obs2_filtered = obs2_filtered @ rot2
            obs2_filtered += filtered_trajs.unsqueeze(-2)[..., :3]
        else:
            obs2_filtered = filtered_trajs

        coll_prob = point_rect_incl_excl(obs1_filtered, obs2_filtered, filtered_covs)

        coll_probs_full[mask] = coll_prob
    coll_probs_max = coll_probs_full.amax(dim=-1)

    return coll_probs_max, coll_probs_full

def traj_mahalanobis_distance(
    corners_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs2: list[torch.Tensor] | torch.Tensor,
    trajs_obs1_covs: list[torch.Tensor] | torch.Tensor = None,
    trajs_obs2_covs: list[torch.Tensor] | torch.Tensor = None,
    corners_obs2: list[torch.Tensor] | torch.Tensor = None,
    with_orientation: bool = False,
    dt = None,
    use_corners: bool = False,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
        Mahalanobis distance-based collision probability estimation for trajectories.
        
        Computes collision probability based on the Mahalanobis distance between
        trajectory centers, treating objects as ellipses defined by their covariance.
        
        Args:
            corners_obs1: Corner coordinates of first object (N, n_corners, 3)
            trajs_obs1: Trajectory of first object (N, num_steps, 3)
            trajs_obs2: Trajectory of second object (M, num_steps, 3)
            trajs_obs1_covs: Covariance matrices for first trajectory (N, num_steps, 3, 3)
            trajs_obs2_covs: Covariance matrices for second trajectory (M, num_steps, 3, 3)
            corners_obs2: Corner coordinates of second object (M, n_corners, 3)
            with_orientation: Whether to consider orientation
            dt: Time step (unused, for API consistency)
            use_corners: Whether to use corners for distance calculation (if False, uses centers)
            device: PyTorch device (GPU/CPU)
        
        Returns:
            coll_probs_max: Maximum collision probability over time (N, M)
            coll_probs_full: Collision probabilities at each timestep (N, M, num_steps)
        """
    # device = get_device(device)
    
    # Convert to torch tensors
    corners_obs1 = to_torch(corners_obs1, device)
    trajs1 = to_torch(trajs_obs1, device)[..., :3]
    trajs2 = to_torch(trajs_obs2, device)[..., :3]

    if trajs_obs1_covs is not None:
        trajs1_covs = to_torch(trajs_obs1_covs, device)[..., :3, :3]
    else:
        trajs1_covs = torch.zeros(*trajs1.shape[:-1], 3, 3, dtype=trajs1.dtype, device=device)

    if trajs_obs2_covs is not None:
        trajs2_covs = to_torch(trajs_obs2_covs, device)[..., :3, :3]
    else:
        trajs2_covs = None

    if corners_obs2 is not None:
        corners_obs2 = to_torch(corners_obs2, device)

    # Ensure proper dimensionality
    if trajs1.ndim == 2:
        trajs1 = trajs1.unsqueeze(0)
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1.unsqueeze(0)
    trajs1 = trajs1.unsqueeze(1)

    if trajs2.ndim == 2:
        trajs2 = trajs2.unsqueeze(0).unsqueeze(0)
    elif trajs2.ndim == 3:
        trajs2 = trajs2.unsqueeze(0)

    if trajs1_covs.ndim == 3:
        trajs1_covs = trajs1_covs.unsqueeze(0)
    trajs1_covs = trajs1_covs.unsqueeze(1)

    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs.unsqueeze(0)
        trajs2_covs = trajs2_covs.unsqueeze(0)

    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2.unsqueeze(0).expand(trajs2.shape[0], -1, -1)
    
    # Convolve distributions (relative motion)
    rel_trajs, rel_covs = convolve_distributions(trajs1, trajs1_covs, trajs2, trajs2_covs, device=device)
    
    d1 = compute_max_diagonal(corners_obs1)
    if corners_obs2 is not None:
        d2 = compute_max_diagonal(corners_obs2)
        d_max = d1.unsqueeze(-1) + d2.unsqueeze(0)
        if d_max.ndim < rel_trajs.ndim - 1:
            d_max = d_max.unsqueeze(-1)
    else:
        d_max = d1.unsqueeze(-1)

    # Compute mask for potential collisions
    rel_pos_norm = torch.hypot(rel_trajs[..., 0], rel_trajs[..., 1])
    cov_max = rel_covs[..., :2, :2].amax(dim=(-2, -1))
    mask = (rel_pos_norm - d_max) <= 5 * torch.sqrt(cov_max)

    coll_probs_full = torch.zeros(mask.shape, dtype=trajs1.dtype, device=device)

    if mask.any():
        obs1 = corners_obs1 - corners_obs1.mean(dim=-2, keepdim=True)
        obs1_ext = obs1.view(obs1.shape[0], 1, 1, *obs1.shape[1:]).expand(*rel_trajs.shape[:3], *obs1.shape[1:])

        obs1_filtered = obs1_ext[mask]
        filtered_trajs = rel_trajs[mask]
        filtered_covs = rel_covs[mask]
        theta_mean = filtered_trajs[..., 2]

        if corners_obs2 is not None:
            obs2 = corners_obs2 - corners_obs2.mean(dim=-2, keepdim=True)
            obs2[..., -1] = corners_obs2[..., -1]

            if obs2.ndim < 4:
                obs2 = obs2.unsqueeze(1)
            obs2_ext = obs2.unsqueeze(0).expand(*rel_trajs.shape[:3], *obs2.shape[-2:])
            obs2_filtered = obs2_ext[mask]

            ini_obs_theta = obs2_filtered[..., 0, -1]
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)
            obs2_filtered = obs2_filtered @ rot2
            obs2_filtered += filtered_trajs.unsqueeze(-2)[..., :3]
        else:
            obs2_filtered = filtered_trajs
        coll_prob = mahalanobis_distance(obs1_filtered, obs2_filtered, filtered_covs, with_orientation = with_orientation, device=device, use_corners=use_corners)

        coll_probs_full[mask] = coll_prob
    coll_probs_max = coll_probs_full.amax(dim=-1)

    return coll_probs_max, coll_probs_full



def traj_corner_mahalanobis_distance(
    corners_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs1: list[torch.Tensor] | torch.Tensor,
    trajs_obs2: list[torch.Tensor] | torch.Tensor,
    trajs_obs1_covs: list[torch.Tensor] | torch.Tensor = None,
    trajs_obs2_covs: list[torch.Tensor] | torch.Tensor = None,
    corners_obs2: list[torch.Tensor] | torch.Tensor = None,
    with_orientation: bool = False,
    dt = None,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mahalanobis distance-based collision probability estimation for trajectories.
    
    Computes collision probability based on the Mahalanobis distance between
    vehicle boundaries along trajectories.
    
    Args:
        corners_obs1: Corner coordinates of first object (N, n_corners, 3)
        trajs_obs1: Trajectory of first object (N, num_steps, 3)
        trajs_obs2: Trajectory of second object (M, num_steps, 3)
        trajs_obs1_covs: Covariance matrices for first trajectory (N, num_steps, 3, 3)
        trajs_obs2_covs: Covariance matrices for second trajectory (M, num_steps, 3, 3)
        corners_obs2: Corner coordinates of second object (M, n_corners, 3)
        with_orientation: Whether to consider orientation
        dt: Time step (unused, for API consistency)
        device: PyTorch device (GPU/CPU)
    
    Returns:
        coll_probs_max: Maximum collision probability over time (N, M)
        coll_probs_full: Collision probabilities at each timestep (N, M, num_steps)
    """
    return traj_mahalanobis_distance(corners_obs1, trajs_obs1,trajs_obs2,
                                trajs_obs1_covs,trajs_obs2_covs,corners_obs2,
                                with_orientation,dt,use_corners=True, device=device)

