"""
Collision probability estimation along trajectories between two polygons.
Following methods are implemented:
1) Monte Carlo Sampling for complete trajectories (boundary crossing via sampling).
2) Boundary crossing method for complete trajectories (analytic).
3) Monte Carlo sampling for each time step.
4) Analytic solution for point-in-polygon collision probability at each time step.
5) Inclusion-Exclusion (rect) for each time step.
6) Mahalanobis distance-based collision probability for each time step.
7) Corner Mahalanobis distance-based collision probability for each time step.
"""

import numpy as np

from scipy.special import erf
from shapely.geometry import Polygon
from shapely import contains_xy

from collision_probability_estimation.params import SEED, NODES_XY, WEIGHTS_XY, NODES_THETA, WEIGHTS_THETA, THETA_NSTD

from .coll_state_overlap import (
    point_collvol_analytic,
    point_collvol_mc_sampling,
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
)


def point_collvol_mc_traj_sampling(
    corners_obs1: list[np.ndarray] | np.ndarray,
    initial_state1: list[np.ndarray] | np.ndarray,
    initial_state2: list[np.ndarray] | np.ndarray,
    initial_cov1: list[np.ndarray] | np.ndarray = None,
    initial_cov2: list[np.ndarray] | np.ndarray = None,
    corners_obs2: list[np.ndarray] | np.ndarray = None,
    dt=0.1,
    num_traj_steps: int = 30,
    with_orientation: bool = False,
    num_samples: int = 20000,
):
    """
    Estimate collision probability over trajectories for each combinations of trajectories using Monte Carlo sampling.
    Parameters:
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

    Returns:
        coll_probs: Estimated collision probabilities (N, M)
        p_cums: Cumulative collision probabilities over time (N, M, num_traj_steps)
        info: Dictionary with sampled trajectories and overlap mask
    """
    corners_obs1 = np.asarray(corners_obs1)
    initial_state1 = np.asarray(initial_state1)
    initial_state2 = np.asarray(initial_state2)[None]
    initial_cov1 = np.asarray(initial_cov1) if initial_cov1 is not None else None
    initial_cov2 = np.asarray(initial_cov2) if initial_cov2 is not None else None
    corners_obs2 = np.asarray(corners_obs2) if corners_obs2 is not None else None
    if initial_state1.ndim == 1:
        initial_state1 = initial_state1[None, ...]
        corners_obs1 = corners_obs1[None, ...]
    initial_state1 = initial_state1[:, None]
    if initial_state2.ndim == 2:
        initial_state2 = initial_state2[None, ...]
    if initial_cov1 is not None:
        if initial_cov1.ndim == 2:
            initial_cov1 = initial_cov1[None]
        initial_cov1 = initial_cov1[:, None]
    if initial_cov2 is not None:
        if initial_cov2.ndim == 2:
            initial_cov2 = initial_cov2[None]
        initial_cov2 = initial_cov2[None]
    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2[None, ...]

    assert initial_state1.ndim == initial_state2.ndim == 3, (
        f"initial_state1 and initial_state2 must be list of states with shape (N_traj, state_dim), got {initial_state1.shape} and {initial_state2.shape}."
    )
    assert (
        len(corners_obs1) == len(initial_state1)
        and (initial_cov1 is None or initial_cov1.shape[:-2] == initial_state1.shape[:-1])
        and (corners_obs2 is None or len(corners_obs2) == initial_state2.shape[1])
        and (initial_cov2 is None or initial_cov2.shape[:-2] == initial_state2.shape[:-1])
    ), (
        f"inputs poly1 ({len(corners_obs1)}), trajs1 ({len(initial_state1)}), trajs1_covs ({0 if initial_cov1 is None else len(initial_cov1)}) lengths must match, "
        f"as well as trajs2 ({len(initial_state2)}), trajs2 covs ({0 if initial_cov2 is None else len(initial_cov2)}) lengths must match, "
        f"and poly2 ({0 if corners_obs2 is None else len(corners_obs2)}) lengths must match trajs2."
    )
    # rotate velocity components in 2D:
    if not with_orientation:
        initial_cov2[..., 2, 2] = 1e-8
        initial_cov1[..., 2, 2] = 1e-8
    if np.any(initial_cov2):
        L = np.linalg.cholesky(initial_cov2)                          # (..., D, D)
        z = np.random.default_rng(SEED).standard_normal(      # (N, ..., D)
                size=(num_samples, *initial_state2.shape)
            )
        points2 = initial_state2+ (z[..., np.newaxis, :] @ L.swapaxes(-1, -2)).squeeze(-2)
    else:
        points2 = initial_state2[np.newaxis]

    trajs2, covs2 = simple_prediction(points2, initial_cov2, dt=dt, num_steps=num_traj_steps)
    if np.any(initial_cov1):
        L = np.linalg.cholesky(initial_cov1)                          
        z = np.random.default_rng(SEED).standard_normal(      
                size=(num_samples, *initial_state1.shape)
            )
        points1 = initial_state1+ (z[..., np.newaxis, :] @ L.swapaxes(-1, -2)).squeeze(-2)
    else:
        points1 = initial_state1[np.newaxis]

    trajs1, covs1 = simple_prediction(points1, initial_cov1, dt=dt, num_steps=num_traj_steps)

    costheta1 = np.cos(trajs1[..., 2])
    sintheta1 = np.sin(trajs1[..., 2])
    vx1 = trajs1[..., -1] * costheta1
    vy1 = trajs1[..., -1] * sintheta1
    trajs1 = np.concatenate([trajs1[..., :-1], vx1[..., None], vy1[..., None]], axis=-1)

    covvx1 = covs1[..., -1, -1] * costheta1
    covvy1 = covs1[..., -1, -1] * sintheta1
    initial = np.zeros((*covs1.shape[:-2], 5, 5))
    initial[..., :-2, :-2] = covs1[..., :-1, :-1]
    initial[..., -2, -2] = covvx1
    initial[..., -1, -1] = covvy1
    covs1 = initial

    costheta2 = np.cos(trajs2[..., 2])
    sintheta2 = np.sin(trajs2[..., 2])
    vx2 = trajs2[..., -1] * costheta2
    vy2 = trajs2[..., -1] * sintheta2
    trajs2 = np.concatenate([trajs2[..., :-1], vx2[..., None], vy2[..., None]], axis=-1)

    covvx2 = covs2[..., -1, -1] * costheta2
    covvy2 = covs2[..., -1, -1] * sintheta2
    initial = np.zeros((*covs2.shape[:-2], 5, 5))
    initial[..., :-2, :-2] = covs2[..., :-1, :-1]
    initial[..., -2, -2] = covvx2
    initial[..., -1, -1] = covvy2
    covs2 = initial
    rel_trajs, rel_covs = convolve_distributions(trajs1, covs1, trajs2, covs2)

    obs1 = corners_obs1 - np.mean(corners_obs1, axis=-2, keepdims=True)
    obs1_ext = obs1[:, None].repeat(rel_trajs.shape[2], axis=1)
    if corners_obs2 is not None:
        obs2 = corners_obs2 - np.mean(corners_obs2, axis=-2, keepdims=True)
        obs2_ext = obs2[None, :].repeat(rel_trajs.shape[1], axis=0)
        if with_orientation:
            delta_theta = rel_trajs[..., 0, 2] - corners_obs2[..., 0, -1]
            rot2 = build_rotation_matrices(delta_theta)
            obs2_ext = obs2_ext @ rot2  # + rel_trajs[..., None, :3]
            obs1_ext = obs1_ext[None].repeat(obs2_ext.shape[0], axis=0)

        collvol, _ = collision_polygon(obs1_ext, obs2_ext)

    else:
        collvol = corners_obs1

    all_trajs = rel_trajs.reshape(num_samples, -1, *rel_trajs.shape[-2:])
    all_collvols = collvol.reshape(collvol.shape[0], -1, *collvol.shape[-2:])
    overlap = []
    if with_orientation:
        for i in range(all_collvols.shape[1]):
            tmp = []
            for j in range(num_samples):
                poly = Polygon(all_collvols[j, i, :, :2])
                tmp.append(contains_xy(poly, all_trajs[j, i, :, 0], all_trajs[j, i, :, 1]))
            overlap.append(tmp)
    else:
        for i in range(all_collvols.shape[1]):
            poly = Polygon(all_collvols[0, i, :, :2])
            overlap.append(contains_xy(poly, all_trajs[:, i, :, 0], all_trajs[:, i, :, 1]))
    overlap = np.asarray(overlap)
    overlap = overlap.reshape(rel_trajs.shape[1], -1, *overlap.shape[-2:])
    entered = np.diff(overlap.astype(int), prepend=1, axis=-1) == 1
    new_entries_per_timestep = np.sum(entered, axis=-2)
    # Probability of new entries per timestep
    v1 = new_entries_per_timestep / num_samples
    coll_probs = np.sum(v1, axis=-1)  
    prob_rate = v1 / dt
    inside_mask = np.any(overlap, axis=-1)
    p_cums = np.clip(np.cumsum(prob_rate, axis=-1) * dt, None, 1.0)
    return coll_probs, p_cums, {"trajs": rel_trajs, "overlap_mask": inside_mask}


def boundary_crossing(
    corners_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs2: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: list[np.ndarray] | np.ndarray = None,
    trajs_obs2_covs: list[np.ndarray] | np.ndarray = None,
    corners_obs2: list[np.ndarray] | np.ndarray = None,
    with_orientation: bool = False,
    dt=0.1,
):
    """
    Estimate collision probability over trajectories using boundary crossing method.

    This method computes the collision probability by analyzing boundary crossings
    of the collision volume formed by the relative motion of two objects.
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

    Returns:
        coll_probs: Estimated collision probabilities (N, M)
        p_cums: Cumulative collision probabilities over time (N, M, num_traj_steps)
    """
 

    corners_obs1 = np.asarray(corners_obs1)
    trajs1 = np.asarray(trajs_obs1)
    trajs2 = np.asarray(trajs_obs2)[None]
    trajs1_covs = np.asarray(trajs_obs1_covs) if trajs_obs1_covs is not None else None
    trajs2_covs = np.asarray(trajs_obs2_covs) if trajs_obs2_covs is not None else None
    corners_obs2 = np.asarray(corners_obs2) if corners_obs2 is not None else None
    if trajs1.ndim == 2:
        trajs1 = trajs1[None, ...]
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1[None, ...]  
    trajs1 = trajs1[:, None]
    if trajs2.ndim == 3:
        trajs2 = trajs2[None, ...]
    if trajs1_covs is not None:
        if trajs1_covs.ndim == 3:
            trajs1_covs = trajs1_covs[None]
        trajs1_covs = trajs1_covs[:, None]
    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs[None]
        trajs2_covs = trajs2_covs[None]
    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2[None, ...].repeat(trajs2.shape[0], axis=0)

    assert trajs1.ndim == trajs2.ndim == 4 and trajs1.shape[-2:] == trajs2.shape[-2:], (
        f"trajs1 and trajs2 must be list of trajectories with shape (N_traj, N_steps, state_dim), got {trajs1.shape} and {trajs2.shape}."
    )
    assert (
        (trajs1_covs is None or trajs1_covs.shape[:-2] == trajs1.shape[:-1])
        and (corners_obs2 is None or len(corners_obs2) == trajs2.shape[1])
        and (trajs2_covs is None or trajs2_covs.shape[:-2] == trajs2.shape[:-1])
    ), (
        f"inputs poly1 ({len(corners_obs1)}), trajs1 ({len(trajs1)}), trajs1_covs ({0 if trajs1_covs is None else len(trajs1_covs)}) lengths must match, "
        f"as well as trajs2 ({len(trajs2)}), trajs2 covs ({0 if trajs2_covs is None else len(trajs2_covs)}) lengths must match, "
        f"and poly2 ({0 if corners_obs2 is None else len(corners_obs2)}) lengths must match trajs2 ({len(trajs2)})."
    )
    if not with_orientation:
        trajs1_covs[..., 2, 2] = 1e-8
        trajs2_covs[..., 2, 2] = 1e-8

    trajs1, trajs1_covs = transform_velocities_and_covs(trajs1, trajs1_covs)
    trajs2, trajs2_covs = transform_velocities_and_covs(trajs2, trajs2_covs)

    rel_trajs, rel_covs = convolve_distributions(trajs1, trajs1_covs, trajs2, trajs2_covs)

    d1 = compute_max_diagonal(corners_obs1)
    if np.any(corners_obs2):
        d2 = compute_max_diagonal(corners_obs2)
        d_max = d1[:, None] + d2[None, :]
        d_max = np.atleast_3d(d_max)
    else:
        d_max = d1[:, None]

    rel_pos_norm = np.hypot(rel_trajs[..., 0], rel_trajs[..., 1])
    cov_max = rel_covs[..., :2, :2].max(axis=(-2, -1))
    mask = (rel_pos_norm - d_max) <= 5 * np.sqrt(cov_max)

    coll_probs_full = np.zeros(rel_trajs.shape[:2])
    p_cum_full = np.zeros((*rel_trajs.shape[:2], rel_trajs.shape[2]))
    cep = np.zeros((*rel_trajs.shape[:2], rel_trajs.shape[2]))
    if np.any(mask):
        obs1 = corners_obs1 - np.mean(corners_obs1, axis=-2, keepdims=True)
        obs1_ext = np.broadcast_to(obs1, (*rel_trajs.shape[:3], *obs1.shape[1:]))

        mask_conc = mask.any(axis=-1)
        filtered_trajs = rel_trajs[mask_conc]
        filtered_covs = rel_covs[mask_conc]
        obs1_filtered = obs1_ext[mask_conc]

        theta_mean = filtered_trajs[..., 2]
        if corners_obs2 is not None:
            obs2 = corners_obs2 - np.mean(corners_obs2, axis=-2, keepdims=True)
            obs2[..., -1] = corners_obs2[..., -1]
            # Ensure obs2 is 4D before broadcasting
            if obs2.ndim < 4:
                obs2 = obs2[:, None, ...]
            obs2_ext = np.broadcast_to(obs2, (*rel_trajs.shape[:3], *obs2.shape[-2:]))
            obs2_filtered = obs2_ext[mask_conc]
            ini_obs_theta = obs2_filtered[..., 0, -1]
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)
            obs_rot2 = obs2_filtered @ rot2  
            obs_rot2[..., -1] += delta_theta[..., None]

            if with_orientation and filtered_covs is not None and filtered_covs.shape[-1] >= 3:
                theta_std = np.sqrt(filtered_covs[..., 2, 2])
                delta_theta_gauss = THETA_NSTD * NODES_THETA[:, None, None] * theta_std
                theta_gauss = delta_theta_gauss + theta_mean
                theta_jacobian = THETA_NSTD * theta_std
                rot_gauss = build_rotation_matrices(delta_theta_gauss)
                obs_rot2 = obs_rot2 @ rot_gauss
                obs_rot2[..., -1] += delta_theta_gauss[..., None]
                obs1_filtered = obs1_filtered[None].repeat(obs_rot2.shape[0], axis=0)
            collvol, _ = collision_polygon(obs1_filtered, obs_rot2)
        else:
            # TODO!
            raise NotImplementedError

        cov_x = filtered_covs[..., :3, :3]
        cov_v = filtered_covs[..., 3:, 3:]
        cov_xv = filtered_covs[..., :3, 3:]
        cov_vx = filtered_covs[..., 3:, :3]
        K = np.linalg.solve(cov_x.swapaxes(-2, -1), cov_vx.swapaxes(-2, -1)).swapaxes(-2, -1)

        cov_vifx = cov_v - K @ cov_xv
        cov_vifx += cov_vifx.swapaxes(-2, -1)
        cov_vifx *= 0.5
        cov_vifx.ravel()[:: cov_vifx.shape[-1] + 1] += 1e-12  # Add to diagonal

        mu_pos = filtered_trajs[..., :3]
        mu_v = filtered_trajs[..., 3:]

        # --- Shift octagon by trajectory mean ---
        delta_pos = collvol - mu_pos[..., None, :]
        # --- Get outward normals of octagon edges ---
        p1 = delta_pos  # [...,:2]
        p2 = np.roll(p1, -1, axis=-2)  # next corner
        side_vecs = p1[..., :2] - p2[..., :2]
        edge_lengths = np.hypot(side_vecs[..., 0], side_vecs[..., 1])
        normals = np.empty_like(side_vecs)
        # outward normal: (-dy, dx)/length
        normals[..., 0] = -side_vecs[..., 1]
        normals[..., 1] = side_vecs[..., 0]
        normals /= edge_lengths[..., None]

        # --- Gauss-Legendre quadrature ---
        weights_xy_expanded = WEIGHTS_XY[:, *[None] * (p1.ndim - 1)]  # (n_nodes, 1, 1)

        # Pre-compute edge interpolation coefficients
        edge_coeff1 = 0.5 * (1 - NODES_XY)
        edge_coeff2 = 0.5 * (1 + NODES_XY)

        # Use einsum for edge point computation
        edge_points = np.einsum("i,...jk->i...jk", edge_coeff1, p1) + np.einsum("i,...jk->i...jk", edge_coeff2, p2)

        # --- Conditional velocity at each edge point ---
        mu_vifx_edge = mu_v[None, ..., None, :] + edge_points @ K.swapaxes(-1, -2)

        # Velocity along normal
        v_n_mean = (mu_vifx_edge * normals[None, ...]).sum(-1)  # (num_quad,T,1,8)
        v_n_var = np.sum((normals[..., None, :] @ cov_vifx[..., None, :, :]) * normals[..., None, :], axis=-1).squeeze(
            -1
        )
        np.maximum(v_n_var, 1e-8, out=v_n_var)
        v_n_std = np.sqrt(v_n_var)[None, ...]  # broadcast over quadr

        # --- Truncated Gaussian expected inward velocity ---
        a = -v_n_mean / v_n_std
        F_vn = 0.5 * (1 + erf(a / np.sqrt(2)))
        phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * a**2)
        E_vn_trunc = -v_n_mean * F_vn + v_n_std * phi  # (num_quad,T,1,8)

        # --- PDF of position at each edge point ---
        delta_pdf = edge_points[..., :2]  # (num_quad,T,1,8,2)

        cov_xiftheta = cov_x[..., :2, :2] + (cov_x[..., :2, 2:3] @ cov_x[..., 2:3, :2]) / cov_x[..., 2:3, 2:3]
        inv_cov_x = np.linalg.inv(cov_xiftheta)  # (T,3,3)
        det_cov_x = np.linalg.det(cov_xiftheta)  # (T,)
        mahal = np.sum(delta_pdf @ inv_cov_x * delta_pdf, axis=-1)
        p_x = (1 / (2 * np.pi * np.sqrt(det_cov_x[None, ..., None]))) * np.exp(-0.5 * mahal)
        integrand = E_vn_trunc * p_x
        # --- Integrate along edge with Gauss-Legendre quadrature and multiply by edge length ---
        cep_loc = np.sum(
            np.sum(integrand * weights_xy_expanded, axis=0) * 0.5 * edge_lengths,
            axis=-1,
            dtype=cov_vifx.dtype,
        )  # (T,)
        if with_orientation:
            theta_pdf = (1.0 / (theta_std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((theta_gauss - theta_mean) / theta_std) ** 2
            )
            cep_loc = np.sum(
                cep_loc * theta_pdf * WEIGHTS_THETA[:, *[None] * (theta_pdf.ndim - 1)] * theta_jacobian,
                axis=0,
            )
        coll_prob = np.trapezoid(cep_loc, dx=dt, axis=-1)
        coll_prob = np.clip(coll_prob, None, 1)
        p_cum = np.clip(np.cumsum(cep_loc, axis=-1) * dt, None, 1.0)

        coll_probs_full[mask_conc] = coll_prob
        p_cum_full[mask_conc] = p_cum
        cep[mask_conc] = cep_loc

    return coll_probs_full, cep  


def point_collvol_mc_state_sampling(
    corners_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs2: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: list[np.ndarray] | np.ndarray = None,
    trajs_obs2_covs: list[np.ndarray] | np.ndarray = None,
    corners_obs2: list[np.ndarray] | np.ndarray = None,
    with_orientation: bool = False,
    num_samples: int = 20000,
):
    """
    Estimate collision probability over trajectories for each combinations of trajectories using Monte Carlo sampling at each time step.

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
    # marginalize everything except position and orientation
    corners_obs1 = np.asarray(corners_obs1)
    trajs1 = np.asarray(trajs_obs1)[..., :3]
    trajs2 = np.asarray(trajs_obs2)[None, ..., :3] if trajs_obs2 is not None else np.array(None)
    trajs1_covs = np.asarray(trajs_obs1_covs)[..., :3, :3] if trajs_obs1_covs is not None else None
    trajs2_covs = np.asarray(trajs_obs2_covs)[..., :3, :3] if trajs_obs2_covs is not None else None
    corners_obs2 = np.asarray(corners_obs2) if corners_obs2 is not None else None

    if trajs1.ndim == 2:
        trajs1 = trajs1[None, ...]
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1[None, ...]  
    trajs1 = trajs1[:, None]
    if trajs2.ndim == 3:
        trajs2 = trajs2[None, ...]
    if trajs1_covs is not None:
        if trajs1_covs.ndim == 3:
            trajs1_covs = trajs1_covs[None]
        trajs1_covs = trajs1_covs[:, None]
    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs[None]
        trajs2_covs = trajs2_covs[None]
    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2[None, ...].repeat(trajs2.shape[0], axis=0)

    assert trajs1.ndim == trajs2.ndim == 4 and trajs1.shape[-2:] == trajs2.shape[-2:], (
        f"trajs1 and trajs2 must be list of trajectories with shape (N_traj, N_steps, state_dim), got {trajs1.shape} and {trajs2.shape}."
    )
    assert (
        (trajs1_covs is None or trajs1_covs.shape[:-2] == trajs1.shape[:-1])
        and (corners_obs2 is None or len(corners_obs2) == trajs2.shape[1])
        and (trajs2_covs is None or trajs2_covs.shape[:-2] == trajs2.shape[:-1])
    ), (
        f"inputs poly1 ({len(corners_obs1)}), trajs1 ({len(trajs1)}), trajs1_covs ({0 if trajs1_covs is None else len(trajs1_covs)}) lengths must match, "
        f"as well as trajs2 ({len(trajs2)}), trajs2 covs ({0 if trajs2_covs is None else len(trajs2_covs)}) lengths must match, "
        f"and poly2 ({0 if corners_obs2 is None else len(corners_obs2)}) lengths must match trajs2 ({len(trajs2)})."
    )

    rel_trajs, rel_covs = convolve_distributions(trajs1, trajs1_covs, trajs2, trajs2_covs)

    d1 = compute_max_diagonal(corners_obs1)
    if np.any(corners_obs2):
        d2 = compute_max_diagonal(corners_obs2)
        d_max = d1[:, None] + d2[None, :]
        d_max = np.atleast_3d(d_max)
    else:
        d_max = d1[:, None]

    rel_pos_norm = np.hypot(rel_trajs[..., 0], rel_trajs[..., 1])
    cov_max = rel_covs[..., :2, :2].max(axis=(-2, -1))
    mask = (rel_pos_norm - d_max) <= 5 * np.sqrt(cov_max)

    coll_probs_full = np.zeros(mask.shape)
    if np.any(mask):
        obs1 = corners_obs1 - np.mean(corners_obs1, axis=-2, keepdims=True)  # center polygon
        obs1_ext = np.broadcast_to(obs1, (*rel_trajs.shape[:3], *obs1.shape[1:]))

        obs1_filtered = obs1_ext[mask]
        filtered_trajs = rel_trajs[mask]
        filtered_covs = rel_covs[mask]
        theta_mean = filtered_trajs[..., 2]
        if corners_obs2 is not None:
            obs2 = corners_obs2 - np.mean(corners_obs2, axis=-2, keepdims=True)
            obs2[..., -1] = corners_obs2[..., -1]
            # Ensure obs2 is 4D before broadcasting
            if obs2.ndim < 4:
                obs2 = obs2[:, None, ...]
            obs2_ext = np.broadcast_to(obs2, (*rel_trajs.shape[:3], *obs2.shape[-2:]))
            obs2_filtered = obs2_ext[mask]
            ini_obs_theta = obs2_filtered[..., 0, -1]
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)
            obs2_filtered = obs2_filtered @ rot2  # .swapaxes(-1, -2)
            obs2_filtered += filtered_trajs[..., None, :3]

        else:
            obs2_filtered = rel_trajs[mask]

        coll_probs, _ = point_collvol_mc_sampling(
            obs1=obs1_filtered,
            obs2=obs2_filtered,
            obs1_cov=filtered_covs,
            with_orientation=with_orientation,
            num_samples=num_samples,
        )
        coll_probs_full = np.zeros(mask.shape, dtype=coll_probs.dtype)
        coll_probs_full[mask] = coll_probs

    return coll_probs_full.max(axis=-1), coll_probs_full  # shape (N_traj1, N_traj2)


def traj_collvol_analytic(
    corners_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs2: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: list[np.ndarray] | np.ndarray = None,
    trajs_obs2_covs: list[np.ndarray] | np.ndarray = None,
    corners_obs2: list[np.ndarray] | np.ndarray = None,
    with_orientation: bool = False,
):
    """
    Estimate collision probability over trajectories for each combinations of trajectories using analytic method.

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

    # marginalize everything except position and orientation
    corners_obs1 = np.asarray(corners_obs1)
    trajs1 = np.asarray(trajs_obs1)[..., :3]
    trajs2 = np.asarray(trajs_obs2)[None, ..., :3]
    trajs1_covs = np.asarray(trajs_obs1_covs)[..., :3, :3] if trajs_obs1_covs is not None else None
    trajs2_covs = np.asarray(trajs_obs2_covs)[..., :3, :3] if trajs_obs2_covs is not None else None
    corners_obs2 = np.asarray(corners_obs2) if corners_obs2 is not None else None
    if trajs1.ndim == 2:
        trajs1 = trajs1[None, ...]
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1[None, ...]  
    trajs1 = trajs1[:, None]
    if trajs2.ndim == 3:
        trajs2 = trajs2[None, ...]
    if trajs1_covs is not None:
        if trajs1_covs.ndim == 3:
            trajs1_covs = trajs1_covs[None]
        trajs1_covs = trajs1_covs[:, None]
    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs[None]
        trajs2_covs = trajs2_covs[None]
    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = corners_obs2[None, ...].repeat(trajs2.shape[0], axis=0)

    assert trajs1.ndim == trajs2.ndim == 4 and trajs1.shape[-2:] == trajs2.shape[-2:], (
        f"trajs1 and trajs2 must be list of trajectories with shape (N_traj, N_steps, state_dim), got {trajs1.shape} and {trajs2.shape}."
    )
    assert (
        (trajs1_covs is None or trajs1_covs.shape[:-2] == trajs1.shape[:-1])
        and (corners_obs2 is None or len(corners_obs2) == trajs2.shape[1])
        and (trajs2_covs is None or trajs2_covs.shape[:-2] == trajs2.shape[:-1])
    ), (
        f"inputs poly1 ({len(corners_obs1)}), trajs1 ({len(trajs1)}), trajs1_covs ({0 if trajs1_covs is None else len(trajs1_covs)}) lengths must match, "
        f"as well as trajs2 ({len(trajs2)}), trajs2 covs ({0 if trajs2_covs is None else len(trajs2_covs)}) lengths must match, "
        f"and poly2 ({0 if corners_obs2 is None else len(corners_obs2)}) lengths must match trajs2 ({len(trajs2)})."
    )

    if not with_orientation:
        trajs1_covs[..., 2, 2] = 1e-8
        if trajs2_covs is not None:
            trajs2_covs[..., 2, 2] = 1e-8
    rel_trajs, rel_covs = convolve_distributions(trajs1, trajs1_covs, trajs2, trajs2_covs, dist="Gaussian")

    d1 = compute_max_diagonal(corners_obs1)
    if np.any(corners_obs2):
        d2 = compute_max_diagonal(corners_obs2)
        d_max = d1[:, None] + d2[None, :]
        d_max = np.atleast_3d(d_max)
    else:
        d_max = d1[:, None]
    rel_pos_norm = np.hypot(rel_trajs[..., 0], rel_trajs[..., 1])
    cov_max = rel_covs[..., :2, :2].max(axis=(-2, -1))
    mask = (rel_pos_norm - d_max) <= 5 * np.sqrt(cov_max)

    coll_probs_full = np.zeros(mask.shape)
    if np.any(mask):
        obs1 = corners_obs1 - np.mean(corners_obs1, axis=-2, keepdims=True)  

        obs1_ext = np.broadcast_to(obs1, (*rel_trajs.shape[:3], *obs1.shape[1:]))

        obs1_filtered = obs1_ext[mask]
        filtered_trajs = rel_trajs[mask]
        filtered_covs = rel_covs[mask]
        theta_mean = filtered_trajs[..., 2]
        if corners_obs2 is not None:
            obs2 = corners_obs2 - np.mean(corners_obs2, axis=-2, keepdims=True)
            obs2[..., -1] = corners_obs2[..., -1]
            # Ensure obs2 is 4D before broadcasting
            if obs2.ndim < 4:
                obs2 = obs2[:, None, ...]
            obs2_ext = np.broadcast_to(obs2, (*rel_trajs.shape[:3], *obs2.shape[-2:]))
            obs2_filtered = obs2_ext[mask]
            ini_obs_theta = obs2_filtered[..., 0, -1]
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)
            obs2_filtered = obs2_filtered @ rot2  # .swapaxes(-1, -2)
            obs2_filtered += filtered_trajs[..., None, :3]

        else:
            obs2_filtered = rel_trajs[mask]

        coll_probs = point_collvol_analytic(
            obs1_filtered, obs2_filtered, filtered_covs, with_orientation=with_orientation
        )
        coll_probs_full[mask] = coll_probs.squeeze()
    return coll_probs_full.max(axis=-1), coll_probs_full

def traj_rect_incl_excl(
    corners_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs2: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: list[np.ndarray] | np.ndarray = None,
    trajs_obs2_covs: list[np.ndarray] | np.ndarray = None,
    corners_obs2: list[np.ndarray] | np.ndarray = None,
    with_orientation: bool = False,
    dt=None,
) -> tuple[np.ndarray, np.ndarray]:
    """   
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

    # --- Input preparation ---
    corners_obs1 = np.asarray(corners_obs1, dtype=np.float64)
    trajs1 = np.asarray(trajs_obs1, dtype=np.float64)[..., :3]
    trajs2 = np.asarray(trajs_obs2, dtype=np.float64)[..., :3]

    if trajs_obs1_covs is not None:
        trajs1_covs = np.asarray(trajs_obs1_covs, dtype=np.float64)[..., :3, :3]
    else:
        trajs1_covs = np.zeros((*trajs1.shape[:-1], 3, 3), dtype=np.float64)

    trajs2_covs = (
        np.asarray(trajs_obs2_covs, dtype=np.float64)[..., :3, :3]
        if trajs_obs2_covs is not None
        else None
    )

    if corners_obs2 is not None:
        corners_obs2 = np.asarray(corners_obs2, dtype=np.float64)

    # --- Ensure 4D: (n_obs1, n_obs2, T, D) ---
    if trajs1.ndim == 2:
        trajs1 = trajs1[np.newaxis]                   # (1, T, 3)
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1[np.newaxis]       # (1, C, 3)

    trajs1 = trajs1[:, np.newaxis]                    # (n1, 1, T, 3)

    if trajs2.ndim == 2:
        trajs2 = trajs2[np.newaxis, np.newaxis]       # (1, 1, T, 3)
    elif trajs2.ndim == 3:
        trajs2 = trajs2[np.newaxis]                   # (1, n2, T, 3)

    if trajs1_covs.ndim == 3:
        trajs1_covs = trajs1_covs[np.newaxis]
    trajs1_covs = trajs1_covs[:, np.newaxis]          # (n1, 1, T, 3, 3)

    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs[np.newaxis]
        trajs2_covs = trajs2_covs[np.newaxis]         # (1, n2, T, 3, 3)

    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = np.broadcast_to(
            corners_obs2[np.newaxis],
            (trajs2.shape[1], *corners_obs2.shape),
        ).copy()

    # --- Suppress orientation uncertainty if not used ---
    if not with_orientation:
        trajs1_covs[..., 2, 2] = 1e-8
        if trajs2_covs is not None:
            trajs2_covs[..., 2, 2] = 1e-8

    # --- Convolve distributions: (n1, n2, T, 3) and (n1, n2, T, 3, 3) ---
    rel_trajs, rel_covs = convolve_distributions(trajs1, trajs1_covs, trajs2, trajs2_covs)

    # --- Distance mask ---
    d1 = compute_max_diagonal(corners_obs1)            # (n1,)
    if corners_obs2 is not None:
        d2 = compute_max_diagonal(corners_obs2)        # (n2,)
        d_max = d1[:, np.newaxis] + d2[np.newaxis, :]  # (n1, n2)
        d_max = d_max[..., np.newaxis]                 # (n1, n2, 1)
    else:
        d_max = d1[:, np.newaxis, np.newaxis]          # (n1, 1, 1)

    rel_pos_norm = np.hypot(rel_trajs[..., 0], rel_trajs[..., 1])  # (n1, n2, T)
    cov_max = rel_covs[..., :2, :2].reshape(*rel_covs.shape[:-2], -1).max(axis=-1)
    mask = (rel_pos_norm - d_max) <= 5 * np.sqrt(cov_max)         # (n1, n2, T)

    coll_probs_full = np.zeros(mask.shape, dtype=np.float64)

    if mask.any():
        # --- Center polygons ---
        obs1 = corners_obs1 - corners_obs1.mean(axis=-2, keepdims=True)  # (n1, C, 3)
        # broadcast obs1 over (n1, n2, T)
        obs1_ext = np.broadcast_to(
            obs1[:, np.newaxis, np.newaxis],
            (*rel_trajs.shape[:3], *obs1.shape[1:]),
        )
        obs1_filtered = obs1_ext[mask]                # (M, C, 3)
        filtered_trajs = rel_trajs[mask]              # (M, 3)
        filtered_covs = rel_covs[mask]                # (M, 3, 3)
        theta_mean = filtered_trajs[..., 2]           # (M,)

        if corners_obs2 is not None:
            obs2 = corners_obs2 - corners_obs2.mean(axis=-2, keepdims=True)
            obs2[..., -1] = corners_obs2[..., -1]

            if obs2.ndim < 4:
                obs2 = obs2[:, np.newaxis]            # (n2, 1, C, 3)

            obs2_ext = np.broadcast_to(
                obs2[np.newaxis],
                (*rel_trajs.shape[:3], *obs2.shape[-2:]),
            )
            obs2_filtered = obs2_ext[mask].copy()     # (M, C, 3)

            ini_obs_theta = obs2_filtered[..., 0, -1]
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)         # (M, 3, 3)
            obs2_filtered = obs2_filtered @ rot2                # (M, C, 3)
            obs2_filtered += filtered_trajs[:, np.newaxis, :3]  # (M, C, 3)
        else:
            obs2_filtered = filtered_trajs

        coll_prob = point_rect_incl_excl(obs1_filtered, obs2_filtered, filtered_covs)
        coll_probs_full[mask] = coll_prob.ravel()

    coll_probs_max = coll_probs_full.max(axis=-1)
    return coll_probs_max, coll_probs_full

def traj_mahalanobis_distance(
    corners_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs2: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: list[np.ndarray] | np.ndarray = None,
    trajs_obs2_covs: list[np.ndarray] | np.ndarray = None,
    corners_obs2: list[np.ndarray] | np.ndarray = None,
    with_orientation: bool = False,
    dt=None,
    use_corners: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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
    # --- Input preparation ---
    corners_obs1 = np.asarray(corners_obs1, dtype=np.float64)
    trajs1 = np.asarray(trajs_obs1, dtype=np.float64)[..., :3]
    trajs2 = np.asarray(trajs_obs2, dtype=np.float64)[..., :3]

    if trajs_obs1_covs is not None:
        trajs1_covs = np.asarray(trajs_obs1_covs, dtype=np.float64)[..., :3, :3]
    else:
        trajs1_covs = np.zeros((*trajs1.shape[:-1], 3, 3), dtype=np.float64)

    trajs2_covs = (
        np.asarray(trajs_obs2_covs, dtype=np.float64)[..., :3, :3]
        if trajs_obs2_covs is not None
        else None
    )

    if corners_obs2 is not None:
        corners_obs2 = np.asarray(corners_obs2, dtype=np.float64)

    # --- Ensure 4D: (n1, n2, T, D) ---
    if trajs1.ndim == 2:
        trajs1 = trajs1[np.newaxis]                     # (1, T, 3)
    if corners_obs1.ndim == 2:
        corners_obs1 = corners_obs1[np.newaxis]         # (1, C, 3)

    trajs1 = trajs1[:, np.newaxis]                      # (n1, 1, T, 3)

    if trajs2.ndim == 2:
        trajs2 = trajs2[np.newaxis, np.newaxis]         # (1, 1, T, 3)
    elif trajs2.ndim == 3:
        trajs2 = trajs2[np.newaxis]                     # (1, n2, T, 3)

    if trajs1_covs.ndim == 3:
        trajs1_covs = trajs1_covs[np.newaxis]
    trajs1_covs = trajs1_covs[:, np.newaxis]            # (n1, 1, T, 3, 3)

    if trajs2_covs is not None:
        if trajs2_covs.ndim == 3:
            trajs2_covs = trajs2_covs[np.newaxis]
        trajs2_covs = trajs2_covs[np.newaxis]           # (1, n2, T, 3, 3)

    if corners_obs2 is not None and corners_obs2.ndim == 2:
        corners_obs2 = np.broadcast_to(
            corners_obs2[np.newaxis],
            (trajs2.shape[1], *corners_obs2.shape),
        ).copy()

    # --- Convolve distributions ---
    rel_trajs, rel_covs = convolve_distributions(trajs1, trajs1_covs, trajs2, trajs2_covs)

    # --- Distance mask ---
    d1 = compute_max_diagonal(corners_obs1)             # (n1,)
    if corners_obs2 is not None:
        d2 = compute_max_diagonal(corners_obs2)         # (n2,)
        d_max = d1[:, np.newaxis] + d2[np.newaxis, :]  # (n1, n2)
        d_max = d_max[..., np.newaxis]                  # (n1, n2, 1)
    else:
        d_max = d1[:, np.newaxis, np.newaxis]           # (n1, 1, 1)

    rel_pos_norm = np.hypot(rel_trajs[..., 0], rel_trajs[..., 1])       # (n1, n2, T)
    cov_max = rel_covs[..., :2, :2].reshape(*rel_covs.shape[:-2], -1).max(axis=-1)
    mask = (rel_pos_norm - d_max) <= 5 * np.sqrt(cov_max)               # (n1, n2, T)

    coll_probs_full = np.zeros(mask.shape, dtype=np.float64)

    if mask.any():
        # --- Center polygons ---
        obs1 = corners_obs1 - corners_obs1.mean(axis=-2, keepdims=True)  # (n1, C, 3)
        obs1_ext = np.broadcast_to(
            obs1[:, np.newaxis, np.newaxis],
            (*rel_trajs.shape[:3], *obs1.shape[1:]),
        )
        obs1_filtered = obs1_ext[mask]                  # (M, C, 3)
        filtered_trajs = rel_trajs[mask]                # (M, 3)
        filtered_covs = rel_covs[mask]                  # (M, 3, 3)
        theta_mean = filtered_trajs[..., 2]             # (M,)

        if corners_obs2 is not None:
            obs2 = corners_obs2 - corners_obs2.mean(axis=-2, keepdims=True)
            obs2[..., -1] = corners_obs2[..., -1]

            if obs2.ndim < 4:
                obs2 = obs2[:, np.newaxis]              # (n2, 1, C, 3)

            obs2_ext = np.broadcast_to(
                obs2[np.newaxis],
                (*rel_trajs.shape[:3], *obs2.shape[-2:]),
            )
            obs2_filtered = obs2_ext[mask].copy()       # (M, C, 3)

            ini_obs_theta = obs2_filtered[..., 0, -1]
            delta_theta = theta_mean - ini_obs_theta
            rot2 = build_rotation_matrices(delta_theta)             # (M, 3, 3)
            obs2_filtered = obs2_filtered @ rot2                    # (M, C, 3)
            obs2_filtered += filtered_trajs[:, np.newaxis, :3]     # (M, C, 3)
        else:
            obs2_filtered = filtered_trajs

        coll_prob = mahalanobis_distance(
            obs1_filtered, obs2_filtered, filtered_covs,
            with_orientation=with_orientation,
            use_corners=use_corners,
        )
        coll_probs_full[mask] = coll_prob.ravel()

    coll_probs_max = coll_probs_full.max(axis=-1)
    return coll_probs_max, coll_probs_full


def traj_corner_mahalanobis_distance(
    corners_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs2: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: list[np.ndarray] | np.ndarray = None,
    trajs_obs2_covs: list[np.ndarray] | np.ndarray = None,
    corners_obs2: list[np.ndarray] | np.ndarray = None,
    with_orientation: bool = False,
    dt=None,
) -> tuple[np.ndarray, np.ndarray]:
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
    return traj_mahalanobis_distance(
        corners_obs1, trajs_obs1, trajs_obs2,
        trajs_obs1_covs, trajs_obs2_covs, corners_obs2,
        with_orientation, dt, use_corners=True,
    )