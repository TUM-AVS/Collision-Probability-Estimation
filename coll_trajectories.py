from typing import Optional
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import erf
import torch
from torch.distributions import MultivariateNormal
from shapely.geometry import Polygon
from shapely import contains_xy

# from params import SEED
from params import SEED, NODES_XY, WEIGHTS_XY, NODES_THETA, WEIGHTS_THETA, THETA_NSTD
import coll_state_overlap as cpo
from utils import convolve_distributions, collision_polygon, simple_prediction


def boundary_crossing(
    initial_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs2: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: Optional[list[np.ndarray] | np.ndarray] = None,
    trajs_obs2_covs: Optional[list[np.ndarray] | np.ndarray] = None,
    initial_obs2: Optional[list[np.ndarray] | np.ndarray] = None,
    with_orientation: bool = False,
    dt=0.1,
):
    """
    Estimate collision probability over trajectories for each combinations of trajectories using boundary crossing method.


    Parameters:
    trajs1: np.ndarray
        Trajectory of the first object (N x state_dim).
    cov1: np.ndarray
        Covariance matrices along the trajectory (N x state_dim x state_dim).
    obs_traj: np.ndarray
        Trajectory of the second object (M x state_dim).
    obs_cov: np.ndarray
        Covariance matrices along the trajectory (M x state_dim x state_dim).
    poly1_coords: np.ndarray
        Corner coordinates of the first object's polygon in its local frame.
    poly2_coords: np.ndarray
        Corner coordinates of the second object's polygon in its local frame.
    num_samples: int
        Number of Monte Carlo samples for estimation.

    Returns:
    float
        Estimated collision probability over the trajectories.
    """

    initial_obs1 = np.asarray(initial_obs1)
    trajs1 = np.asarray(trajs_obs1)
    trajs2 = np.asarray(trajs_obs2)[None]
    trajs1_covs = np.asarray(trajs_obs1_covs) if trajs_obs1_covs is not None else None
    trajs2_covs = np.asarray(trajs_obs2_covs) if trajs_obs2_covs is not None else None
    initial_obs2 = np.asarray(initial_obs2) if initial_obs2 is not None else None
    if trajs1.ndim == 2:
        trajs1 = trajs1[None, ...]
    if initial_obs1.ndim == 2:
        initial_obs1 = initial_obs1[None, ...]
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
    if initial_obs2 is not None and initial_obs2.ndim == 2:
        initial_obs2 = initial_obs2[None, ...]

    assert (
        trajs1.ndim == trajs2.ndim == 4 and trajs1.shape[-2:] == trajs2.shape[-2:]
    ), f"trajs1 and trajs2 must be list of trajectories with shape (N_traj, N_steps, state_dim), got {trajs1.shape} and {trajs2.shape}."
    assert (
        len(initial_obs1) == len(trajs1)
        and (trajs1_covs is None or trajs1_covs.shape[:-2] == trajs1.shape[:-1])
        and (initial_obs2 is None or len(initial_obs2) == trajs2.shape[1])
        and (trajs2_covs is None or trajs2_covs.shape[:-2] == trajs2.shape[:-1])
    ), (
        f"inputs poly1 ({len(initial_obs1)}), trajs1 ({len(trajs1)}), trajs1_covs ({0 if trajs1_covs is None else len(trajs1_covs)}) lengths must match, "
        f"as well as trajs2 ({len(trajs2)}), trajs2 covs ({0 if trajs2_covs is None else len(trajs2_covs)}) lengths must match, "
        f"and poly2 ({0 if initial_obs2 is None else len(initial_obs2)}) lengths must match trajs2."
    )
    rel_trajs, rel_covs = convolve_distributions(
        trajs1, trajs1_covs, trajs2, trajs2_covs
    )
    if rel_covs is None:
        raise NotImplementedError("Analytic solution not implemented for this case.")

    obs1 = initial_obs1 - np.mean(
        initial_obs1, axis=-2, keepdims=True
    )  # center polygon
    obs1_ext = (
        obs1[:, None, None]
        .repeat(rel_trajs.shape[1], axis=1)
        .repeat(rel_trajs.shape[2], axis=2)
    )  # (N_traj1, N_traj2, 1, N_corners, 3)

    theta_mean = rel_trajs[..., 2]
    if initial_obs2 is not None:
        obs2 = initial_obs2 - np.mean(initial_obs2, axis=-2, keepdims=True)
        obs2[..., -1] = initial_obs2[..., -1]
        delta_theta = theta_mean - initial_obs2[None, ..., 0, -1, None]
        rot2 = np.zeros((*delta_theta.shape, 3, 3))
        costheta = np.cos(delta_theta)
        sintheta = np.sin(delta_theta)
        rot2[..., 0, 0] = costheta
        rot2[..., 0, 1] = sintheta
        rot2[..., 1, 0] = -sintheta
        rot2[..., 1, 1] = costheta
        rot2[..., 2, 2] = 1.0
        obs_rot2 = obs2[None, :, None] @ rot2  # + rel_trajs[..., None, :3]
        obs_rot2[..., -1] += delta_theta[..., None]
        if with_orientation and rel_covs is not None and rel_covs.shape[-1] >= 3:
            theta_std = np.sqrt(rel_covs[..., 2, 2])
            # convert nodes from [-1, 1] to [theta_mean - THETA_NSTD*theta_std, theta_mean + THETA_NSTD*theta_std]
            delta_theta_gauss = (
                THETA_NSTD * NODES_THETA[:, None, None, None] * theta_std
            )
            theta_gauss = delta_theta_gauss + theta_mean
            theta_jacobian = THETA_NSTD * theta_std
            rot_theta = np.zeros((*delta_theta_gauss.shape, 3, 3))
            costheta = np.cos(delta_theta_gauss)
            sintheta = np.sin(delta_theta_gauss)
            rot_theta[..., 0, 0] = costheta
            rot_theta[..., 0, 1] = sintheta
            rot_theta[..., 1, 0] = -sintheta
            rot_theta[..., 1, 1] = costheta
            rot_theta[..., 2, 2] = 1.0
            obs_rot2 = obs_rot2 @ rot_theta
            obs_rot2[..., -1] += delta_theta_gauss[..., None]
            obs1_ext = obs1_ext[None].repeat(obs_rot2.shape[0], axis=0)
        collvol, _ = collision_polygon(obs1_ext, obs_rot2)
        # np.max(points - rel_trajs[...,None, :3])
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # for i in range(collvol.shape[2]):
        #     poly = plt.Polygon(
        #         collvol[0,1,i, :, :2],
        #         closed=True,
        #         edgecolor="#E37222",
        #         facecolor="none",
        #         linewidth=1,
        #         linestyle="--",
        #         alpha=1,
        #         zorder=13,
        #     )
        #     ax.add_patch(poly)
    else:
        # collvol, points = collision_polygon(obs1, obs2)
        if with_orientation:
            raise NotImplementedError
        collvol = initial_obs2[None]
    cov_x = rel_covs[..., :3, :3]  # (T,3,3)
    cov_v = rel_covs[..., 3:, 3:]  # (T,2,2)
    cov_xv = rel_covs[..., :3, 3:]  # (T,3,2)
    cov_vx = rel_covs[..., 3:, :3]  # (T,2,3)
    inv_cov_x = np.linalg.inv(cov_x)
    K = cov_vx @ inv_cov_x
    cov_vifx = cov_v - K @ cov_xv
    cov_vifx = 0.5 * (cov_vifx + cov_vifx.swapaxes(-2, -1)) + 1e-12 * np.eye(
        cov_vifx.shape[-1], dtype=cov_vifx.dtype
    )

    mu_pos = rel_trajs[..., :3]  # (T,3)
    mu_v = rel_trajs[..., 3:]  # (T,2)

    # --- Shift octagon by trajectory mean ---
    delta_pos = collvol - mu_pos[..., None, :]
    # --- Get outward normals of octagon edges ---
    p1 = delta_pos
    p2 = np.roll(p1, -1, axis=-2)  # next corner
    side_vecs = (p1 - p2)[..., :2]
    edge_lengths = np.linalg.norm(side_vecs, axis=-1)
    normals = np.zeros_like(side_vecs)
    # outward normal: (-dy, dx)/length
    normals[..., 0] = -side_vecs[..., 1] / edge_lengths
    normals[..., 1] = side_vecs[..., 0] / edge_lengths

    # --- Gauss-Legendre quadrature ---
    nodes_expanded = NODES_XY[:, *[None] * p1.ndim]  # (n_nodes, 1, 1)
    weights_xy_expanded = WEIGHTS_XY[:, *[None] * (p1.ndim - 1)]  # (n_nodes, 1, 1)
    edge_points = 0.5 * (1 - nodes_expanded) * p1 + 0.5 * (1 + nodes_expanded) * p2
    # neu = edge_points - mu_pos[...,None,:]
    # --- Conditional velocity at each edge point ---
    mu_vifx_edge = mu_v[None, ..., None, :] + edge_points @ K.swapaxes(-1, -2)

    # Velocity along normal
    v_n_mean = (mu_vifx_edge * normals[None, ...]).sum(-1)  # (num_quad,T,1,8)
    v_n_var = (
        (normals[..., None, :] @ cov_vifx[..., None, :, :] @ normals[..., None])
        .squeeze(-1)
        .squeeze(-1)
    )  # (T,1,8)
    v_n_var = np.clip(v_n_var, min=1e-8)
    v_n_std = np.sqrt(v_n_var)[None, ...]  # broadcast over quadr

    # --- Truncated Gaussian expected inward velocity ---
    a = -v_n_mean / v_n_std
    F_vn = 0.5 * (1 + erf(a / np.sqrt(2)))
    phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * a**2)
    E_vn_trunc = -v_n_mean * F_vn + v_n_std * phi  # (num_quad,T,1,8)

    # --- PDF of position at each edge point ---
    delta_pdf = edge_points[..., :2]  # (num_quad,T,1,8,2)
    cov_xiftheta = cov_x[..., :2, :2] + np.einsum(
        "...i,...j->...ij", cov_x[..., :2, 2], cov_x[..., 2, :2]
    ) / (cov_x[..., 2, 2][..., None, None])
    inv_cov_x = np.linalg.inv(cov_xiftheta)  # (T,3,3)
    det_cov_x = np.linalg.det(cov_xiftheta)  # (T,)
    mahal = np.einsum("...di,...ij,...dj->...d", delta_pdf, inv_cov_x, delta_pdf)
    p_x = (1 / (2 * np.pi * np.sqrt(det_cov_x[None, ..., None]))) * np.exp(-0.5 * mahal)
    integrand = E_vn_trunc * p_x
    # --- Integrate along edge with Gauss-Legendre quadrature and multiply by edge length ---
    cep = np.sum(
        np.sum(integrand * weights_xy_expanded, axis=0) * 0.5 * edge_lengths,
        axis=-1,
        dtype=cov_vifx.dtype,
    )  # (T,)
    if with_orientation:
        theta_pdf = norm(loc=theta_mean, scale=theta_std).pdf(theta_gauss)
        cep = np.sum(
            cep
            * theta_pdf
            * WEIGHTS_THETA[:, *[None] * (theta_pdf.ndim - 1)]
            * theta_jacobian,
            axis=0,
        )
    coll_prob = np.trapezoid(cep, dx=dt, axis=-1)
    coll_prob = np.clip(coll_prob, None, 1)
    p_cum = np.clip(np.cumsum(cep, axis=-1) * dt, None, 1.0)
    return coll_prob, p_cum


def point_collvol_mc_state_sampling(
    initial_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: list[np.ndarray] | np.ndarray,
    trajs_obs2: Optional[list[np.ndarray] | np.ndarray] = None,
    trajs_obs2_covs: Optional[list[np.ndarray] | np.ndarray] = None,
    initial_obs2: Optional[list[np.ndarray] | np.ndarray] = None,
    with_orientation: bool = False,
    num_samples: int = 20000,
):

    initial_obs1 = np.asarray(initial_obs1)
    trajs1 = np.asarray(trajs_obs1)
    trajs2 = np.asarray(trajs_obs2)[None]
    trajs1_covs = np.asarray(trajs_obs1_covs) if trajs_obs1_covs is not None else None
    trajs2_covs = np.asarray(trajs_obs2_covs) if trajs_obs2_covs is not None else None
    initial_obs2 = np.asarray(initial_obs2) if initial_obs2 is not None else None
    if trajs1.ndim == 2:
        trajs1 = trajs1[None, ...]
    if initial_obs1.ndim == 2:
        initial_obs1 = initial_obs1[None, ...]
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
    if initial_obs2 is not None and initial_obs2.ndim == 2:
        initial_obs2 = initial_obs2[None, ...]

    assert (
        trajs1.ndim == trajs2.ndim == 4 and trajs1.shape[-2:] == trajs2.shape[-2:]
    ), f"trajs1 and trajs2 must be list of trajectories with shape (N_traj, N_steps, state_dim), got {trajs1.shape} and {trajs2.shape}."
    assert (
        len(initial_obs1) == len(trajs1)
        and (trajs1_covs is None or trajs1_covs.shape[:-2] == trajs1.shape[:-1])
        and (initial_obs2 is None or len(initial_obs2) == trajs2.shape[1])
        and (trajs2_covs is None or trajs2_covs.shape[:-2] == trajs2.shape[:-1])
    ), (
        f"inputs poly1 ({len(initial_obs1)}), trajs1 ({len(trajs1)}), trajs1_covs ({0 if trajs1_covs is None else len(trajs1_covs)}) lengths must match, "
        f"as well as trajs2 ({len(trajs2)}), trajs2 covs ({0 if trajs2_covs is None else len(trajs2_covs)}) lengths must match, "
        f"and poly2 ({0 if initial_obs2 is None else len(initial_obs2)}) lengths must match trajs2."
    )
    rel_trajs, rel_covs = convolve_distributions(
        trajs1, trajs1_covs, trajs2, trajs2_covs
    )

    d1 = np.max(np.linalg.norm(np.diff(initial_obs1, axis=-2), axis=-1), axis=-1)
    d2 = (
        np.max(np.linalg.norm(np.diff(initial_obs2, axis=-2), axis=-1), axis=-1)
        if np.any(initial_obs2)
        else np.array(0)
    )
    d_max = d1[:, None] + d2[None]

    mask = np.linalg.norm(rel_trajs, axis=-1) - d_max[..., None] <= 4 * np.sqrt(
        rel_covs[..., :2, :2].max(axis=-1).max(axis=-1)
    )
    obs1 = initial_obs1 - np.mean(
        initial_obs1, axis=-2, keepdims=True
    )  # center polygon
    obs1_ext = (
        obs1[:, None, None]
        .repeat(rel_trajs.shape[1], axis=1)
        .repeat(rel_trajs.shape[2], axis=2)
    )  # (N_traj1, N_traj2, 1, N_corners, 3)
    if initial_obs2 is not None:
        obs2 = initial_obs2 - np.mean(initial_obs2, axis=-2, keepdims=True)
        delta_theta = rel_trajs[..., 2] - initial_obs2[None, ..., 0, -1, None]
        rot2 = np.zeros((*delta_theta.shape, 3, 3))
        costheta = np.cos(delta_theta)
        sintheta = np.sin(delta_theta)
        rot2[..., 0, 0] = costheta
        rot2[..., 0, 1] = sintheta
        rot2[..., 1, 0] = -sintheta
        rot2[..., 1, 1] = costheta
        rot2[..., 2, 2] = 1.0
        obs_rot2 = obs2[None, :, None] @ rot2 + rel_trajs[..., None, :3]
        states = obs_rot2[mask]
    else:
        # obs_shift = rel_trajs[mask, None, :3] + obs1[None]
        states = rel_trajs[mask]

    coll_probs, _ = cpo.point_collvol_mc_sampling(
        obs1=obs1_ext[mask],
        obs2=states,
        obs1_cov=rel_covs[mask],
        with_orientation=with_orientation,
        num_samples=num_samples,
    )
    coll_probs_full = np.zeros(mask.shape, dtype=coll_probs.dtype)
    coll_probs_full[mask] = coll_probs

    return coll_probs_full.max(axis=-1), coll_probs_full  # shape (N_traj1, N_traj2)


def point_collvol_mc_traj_sampling(
    initial_obs1: list[np.ndarray] | np.ndarray,
    initial_state1: list[np.ndarray] | np.ndarray,
    initial_state2: list[np.ndarray] | np.ndarray,
    initial_cov1: Optional[list[np.ndarray] | np.ndarray] = None,
    initial_cov2: Optional[list[np.ndarray] | np.ndarray] = None,
    initial_obs2: Optional[list[np.ndarray] | np.ndarray] = None,
    dt=0.1,
    num_traj_steps: int = 30,
    with_orientation: bool = False,
    num_samples: int = 20000,
):

    initial_obs1 = np.asarray(initial_obs1)
    initial_state1 = np.asarray(initial_state1)
    initial_state2 = np.asarray(initial_state2)[None]
    initial_cov1 = np.asarray(initial_cov1) if initial_cov1 is not None else None
    initial_cov2 = np.asarray(initial_cov2) if initial_cov2 is not None else None
    initial_obs2 = np.asarray(initial_obs2) if initial_obs2 is not None else None
    if initial_state1.ndim == 1:
        initial_state1 = initial_state1[None, ...]
        initial_obs1 = initial_obs1[None, ...]
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
    if initial_obs2 is not None and initial_obs2.ndim == 2:
        initial_obs2 = initial_obs2[None, ...]

    assert (
        initial_state1.ndim == initial_state2.ndim == 3
    ), f"initial_state1 and initial_state2 must be list of states with shape (N_traj, state_dim), got {initial_state1.shape} and {initial_state2.shape}."
    assert (
        len(initial_obs1) == len(initial_state1)
        and (
            initial_cov1 is None or initial_cov1.shape[:-2] == initial_state1.shape[:-1]
        )
        and (initial_obs2 is None or len(initial_obs2) == initial_state2.shape[1])
        and (
            initial_cov2 is None or initial_cov2.shape[:-2] == initial_state2.shape[:-1]
        )
    ), (
        f"inputs poly1 ({len(initial_obs1)}), trajs1 ({len(initial_state1)}), trajs1_covs ({0 if initial_cov1 is None else len(initial_cov1)}) lengths must match, "
        f"as well as trajs2 ({len(initial_state2)}), trajs2 covs ({0 if initial_cov2 is None else len(initial_cov2)}) lengths must match, "
        f"and poly2 ({0 if initial_obs2 is None else len(initial_obs2)}) lengths must match trajs2."
    )
    rel_points, rel_covs = convolve_distributions(
        initial_state1, initial_cov1, initial_state2, initial_cov2
    )

    # Sample random points, if no covariance provided test whether point is inside polygon
    if not with_orientation:
        rel_covs[..., 2, 2] = 1e-8

    points = (
        MultivariateNormal(
            loc=torch.tensor(initial_state2),
            covariance_matrix=torch.tensor(rel_covs),
        )
        .sample((num_samples,))
        .numpy()
        if np.any(rel_covs)
        else initial_state2[np.newaxis]
    )
    covs = np.repeat(rel_covs[np.newaxis], num_samples, axis=0)
    rel_trajs, rel_cov = simple_prediction(
        points, covs, dt=dt, num_steps=num_traj_steps
    )
    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # for traj in rel_trajs:
    #     ax.plot(traj[0, 1, :, 0], traj[0, 1, :, 1])

    obs1 = initial_obs1 - np.mean(initial_obs1, axis=-2, keepdims=True)
    obs1_ext = obs1[:, None].repeat(rel_trajs.shape[2], axis=1)
    if initial_obs2 is not None:
        obs2 = initial_obs2 - np.mean(initial_obs2, axis=-2, keepdims=True)
        obs2_ext = obs2[None, :].repeat(rel_trajs.shape[1], axis=0)
        if with_orientation:
            delta_theta = rel_trajs[..., 0, 2] - initial_obs2[..., 0, -1]
            # delta_theta = np.sort(delta_theta, axis=0)
            rot2 = np.zeros((*delta_theta.shape, 3, 3))
            costheta = np.cos(delta_theta)
            sintheta = np.sin(delta_theta)
            rot2[..., 0, 0] = costheta
            rot2[..., 0, 1] = sintheta
            rot2[..., 1, 0] = -sintheta
            rot2[..., 1, 1] = costheta
            rot2[..., 2, 2] = 1.0
            obs2_ext = obs2_ext @ rot2  # + rel_trajs[..., None, :3]
            obs1_ext = obs1_ext[None].repeat(obs2_ext.shape[0], axis=0)

        collvol, _ = collision_polygon(obs1_ext, obs2_ext)

        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots()
        # for i in range(collvol.shape[0]):
        #     poly = plt.Polygon(
        #         collvol[i, 0, 1, :, :2],
        #         closed=True,
        #         edgecolor="#E37222",
        #         facecolor="none",
        #         linewidth=1,
        #         linestyle="--",
        #         alpha=1,
        #         zorder=13,
        #     )
        #     ax.add_patch(poly)

    else:
        collvol = initial_obs1

    all_trajs = rel_trajs.reshape(num_samples, -1, *rel_trajs.shape[-2:])
    all_collvols = collvol.reshape(collvol.shape[0], -1, *collvol.shape[-2:])
    overlap = []
    if with_orientation:
        for i in range(all_collvols.shape[1]):
            tmp = []
            for j in range(num_samples):
                poly = Polygon(all_collvols[j, i, :, :2])
                tmp.append(
                    contains_xy(poly, all_trajs[j, i, :, 0], all_trajs[j, i, :, 1])
                )
            overlap.append(tmp)
    else:

        for i in range(all_collvols.shape[1]):
            poly = Polygon(all_collvols[0, i, :, :2])
            overlap.append(
                contains_xy(poly, all_trajs[:, i, :, 0], all_trajs[:, i, :, 1])
            )
    overlap = np.asarray(overlap)
    overlap = overlap.reshape(rel_trajs.shape[1], -1, *overlap.shape[-2:])
    entered = np.diff(overlap.astype(int), prepend=1, axis=-1) == 1
    new_entries_per_timestep = np.sum(entered, axis=-2)
    # Probability of new entries per timestep
    v1 = new_entries_per_timestep / num_samples
    coll_probs = np.sum(v1, axis=-1)  # Approximate integral over time by summation
    prob_rate = v1 / dt
    inside_mask = np.any(overlap, axis=-1)
    p_cums = np.clip(np.cumsum(prob_rate, axis=-1) * dt, None, 1.0)
    return coll_probs, p_cums, {"trajs": rel_trajs, "overlap_mask": inside_mask}


def traj_collvol_analytic(
    initial_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: list[np.ndarray] | np.ndarray,
    trajs_obs2: Optional[list[np.ndarray] | np.ndarray] = None,
    trajs_obs2_covs: Optional[list[np.ndarray] | np.ndarray] = None,
    initial_obs2: Optional[list[np.ndarray] | np.ndarray] = None,
    with_orientation: bool = False,
):
    # assert not with_orientation, "with_orientation=True not implemented yet."

    initial_obs1 = np.asarray(initial_obs1)
    trajs1 = np.asarray(trajs_obs1)
    trajs2 = np.asarray(trajs_obs2)[None]
    trajs1_covs = np.asarray(trajs_obs1_covs) if trajs_obs1_covs is not None else None
    trajs2_covs = np.asarray(trajs_obs2_covs) if trajs_obs2_covs is not None else None
    initial_obs2 = np.asarray(initial_obs2) if initial_obs2 is not None else None
    if trajs1.ndim == 2:
        trajs1 = trajs1[None, ...]
    if initial_obs1.ndim == 2:
        initial_obs1 = initial_obs1[None, ...]
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
    if initial_obs2 is not None and initial_obs2.ndim == 2:
        initial_obs2 = initial_obs2[None, ...]

    assert (
        trajs1.ndim == trajs2.ndim == 4 and trajs1.shape[-2:] == trajs2.shape[-2:]
    ), f"trajs1 and trajs2 must be list of trajectories with shape (N_traj, N_steps, state_dim), got {trajs1.shape} and {trajs2.shape}."
    assert (
        len(initial_obs1) == len(trajs1)
        and (trajs1_covs is None or trajs1_covs.shape[:-2] == trajs1.shape[:-1])
        and (initial_obs2 is None or len(initial_obs2) == trajs2.shape[1])
        and (trajs2_covs is None or trajs2_covs.shape[:-2] == trajs2.shape[:-1])
    ), (
        f"inputs poly1 ({len(initial_obs1)}), trajs1 ({len(trajs1)}), trajs1_covs ({0 if trajs1_covs is None else len(trajs1_covs)}) lengths must match, "
        f"as well as trajs2 ({len(trajs2)}), trajs2 covs ({0 if trajs2_covs is None else len(trajs2_covs)}) lengths must match, "
        f"and poly2 ({0 if initial_obs2 is None else len(initial_obs2)}) lengths must match trajs2."
    )
    rel_trajs, rel_covs = convolve_distributions(
        trajs1, trajs1_covs, trajs2, trajs2_covs
    )

    d1 = np.max(np.linalg.norm(np.diff(initial_obs1, axis=-2), axis=-1), axis=-1)
    d2 = (
        np.max(np.linalg.norm(np.diff(initial_obs2, axis=-2), axis=-1), axis=-1)
        if np.any(initial_obs2)
        else 0
    )
    d_max = d1[:, None] + d2[None]

    mask = np.linalg.norm(rel_trajs, axis=-1) - d_max[..., None] <= 4 * np.sqrt(
        rel_covs[..., :2, :2].max(axis=-1).max(axis=-1)
    )
    obs1 = initial_obs1 - np.mean(
        initial_obs1, axis=-2, keepdims=True
    )  # center polygon
    obs1_ext = (
        obs1[:, None, None]
        .repeat(rel_trajs.shape[1], axis=1)
        .repeat(rel_trajs.shape[2], axis=2)
    )  # (N_traj1, N_traj2, 1, N_corners, 3)
    if initial_obs2 is not None:
        obs2 = initial_obs2 - np.mean(initial_obs2, axis=-2, keepdims=True)
        delta_theta = rel_trajs[..., 2] - initial_obs2[None, ..., 0, -1, None]
        rot2 = np.zeros((*delta_theta.shape, 3, 3))
        costheta = np.cos(delta_theta)
        sintheta = np.sin(delta_theta)
        rot2[..., 0, 0] = costheta
        rot2[..., 0, 1] = sintheta
        rot2[..., 1, 0] = -sintheta
        rot2[..., 1, 1] = costheta
        rot2[..., 2, 2] = 1.0
        obs_rot2 = obs2[None, :, None] @ rot2 + rel_trajs[..., None, :3]
        states = obs_rot2[mask]
    else:
        # obs_shift = rel_trajs[mask, None, :3] + obs1[None]
        states = rel_trajs[mask]
    covs = rel_covs[mask]

    coll_probs = cpo.point_collvol_analytic(
        obs1_ext[mask], states, covs, with_orientation=with_orientation
    )
    coll_probs_full = np.zeros(mask.shape, dtype=coll_probs.dtype)
    coll_probs_full[mask] = coll_probs
    return coll_probs_full.max(axis=-1), coll_probs_full  # shape (N_tr
