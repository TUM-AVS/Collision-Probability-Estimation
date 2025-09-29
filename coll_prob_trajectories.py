__author__ = "Marc Kaufeld"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import re
from typing import Optional
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import erf

from shapely.geometry import Polygon
from shapely import contains_xy

# from params import SEED
from params import SEED, NODES_XY, WEIGHTS_XY, NODES_THETA, WEIGHTS_THETA
import coll_prob_state_overlap as cpo
from utils import convolve_distributions, collision_polygon, simple_prediction


def boundary_crossing(
    poly1_boundary: list[np.ndarray],
    dt: float = 0.1,
    trajs1: Optional[list[np.ndarray]] = None,
    trajs1_covs: Optional[list[np.ndarray]] = None,
    trajs2: Optional[list[np.ndarray]] = None,
    trajs2_covs: Optional[list[np.ndarray]] = None,
    poly2_boundary: Optional[list[np.ndarray]] = None,
    with_orientation: bool = False,
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

    # Validate input shapes and lengths
    if trajs1 is not None and trajs2 is not None:
        assert len(poly1_boundary) == len(trajs1) == len(trajs1_covs) and len(
            trajs2
        ) == len(trajs2_covs), (
            f"inputs poly1 ({len(poly1_boundary)}), trajs1 ({len(trajs1)}), trajs1_covs ({len(trajs1_covs)}) lengths must match, "
            f"as well as trajs2 ({len(trajs2)}), trajs2_covs ({len(trajs2_covs)}) lengths must match."
        )

        rel_traj, rel_cov = convolve_distributions(
            trajs1, trajs1_covs, trajs2, trajs2_covs
        )
    elif trajs1 is not None:
        assert (
            len(poly1_boundary) == len(trajs1) == len(trajs1_covs)
        ), f"inputs poly1 ({len(poly1_boundary)}), trajs1 ({len(trajs1)}), trajs1_covs ({len(trajs1_covs)}) lengths must match."
        # in obs1 local frame
        rel_traj, rel_cov = -np.asarray(trajs1 * len(poly2_boundary)), np.asarray(
            trajs1_covs * len(poly2_boundary)
        )
        rel_traj[..., :3] += np.repeat(
            np.mean(poly2_boundary, axis=-2), len(trajs1), axis=0
        )[:, None]
    elif trajs2 is not None:  # trajs2 is not None
        assert len(trajs2) == len(
            trajs2_covs
        ), f"inputs trajs2 ({len(trajs2)}), trajs2_covs ({len(trajs2_covs)}) lengths must match."
        rel_traj, rel_cov = np.asarray(trajs2 * len(poly1_boundary)), np.asarray(
            trajs2_covs * len(poly1_boundary)
        )
    else:
        raise ValueError("Either trajs1 or trajs2 must be provided.")

    # np.linalg.norm(rel_traj, axis=-1).max()
    # TODO PREFILTERING IF DISTANCE IS LARGE!-> ALSO IN STATE OVERLAP!
    if with_orientation:
        raise NotImplementedError("with_orientation=True is not implemented yet.")
    if poly2_boundary is not None:
        polys2 = np.array(
            [
                poly2_boundary[i]
                for i in range(len(poly2_boundary))
                for j in range(len(poly1_boundary))
            ]
        )
        polys1 = np.array(
            [
                poly1_boundary[j]
                for i in range(len(poly2_boundary))
                for j in range(len(poly1_boundary))
            ]
        )

        delta_theta = rel_traj[..., 2] - polys2[..., 0, -1, None]
        rot_mats = np.zeros((*delta_theta.shape, 3, 3))
        costheta = np.cos(delta_theta)
        sintheta = np.sin(delta_theta)
        rot_mats[..., 0, 0] = costheta
        rot_mats[..., 0, 1] = sintheta
        rot_mats[..., 1, 0] = -sintheta
        rot_mats[..., 1, 1] = costheta
        rot_mats[..., 2, 2] = 1

        rel_polys2 = (
            polys2[:, None] @ rot_mats
        )  # .transpose(0, 2, 1)) #+ rel_traj[:, None, :]
        polys1 = polys1[:, None].repeat(rel_polys2.shape[1], axis=1)
        poly_boundary, _ = collision_polygon(polys1, rel_polys2)
    else:
        poly_boundary = np.array(
            poly1_boundary * int(len(rel_traj) / len(poly1_boundary))
        )[:, None].repeat(rel_traj.shape[1], axis=1)
    poly_boundary = poly_boundary - np.mean(
        poly_boundary, axis=-2, keepdims=True
    )  # center polygon bc of ref_traj
    cov_x = rel_cov[..., :3, :3]  # (T,3,3)
    cov_v = rel_cov[..., 3:, 3:]  # (T,2,2)
    cov_xv = rel_cov[..., :3, 3:]  # (T,3,2)
    cov_vx = rel_cov[..., 3:, :3]  # (T,2,3)
    inv_cov_x = np.linalg.inv(cov_x)
    K = cov_vx @ inv_cov_x
    cov_vifx = cov_v - K @ cov_xv
    cov_vifx = 0.5 * (cov_vifx + cov_vifx.swapaxes(-2, -1)) + 1e-12 * np.eye(
        cov_vifx.shape[-1], dtype=cov_vifx.dtype
    )

    mu_pos = rel_traj[..., :3]  # (T,3)
    mu_v = rel_traj[..., 3:]  # (T,2)

    # --- Shift octagon by trajectory mean ---
    delta_pos = poly_boundary - mu_pos[..., None, :]
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
    nodes_expanded = NODES_XY[:, None, None, None, None]  # (n_nodes, 1, 1)
    weights_xy_expanded = WEIGHTS_XY[:, None, None, None]  # (n_nodes, 1, 1)
    edge_points = 0.5 * (1 - nodes_expanded) * p1 + 0.5 * (1 + nodes_expanded) * p2

    # --- Conditional velocity at each edge point ---
    mu_vifx_edge = mu_v[None, :, :, None] + np.einsum(
        "bcij,...bcnj -> ...bcni", K, edge_points
    )  # (num_quad,T,1,8,2)
    # Velocity along normal
    v_n_mean = (mu_vifx_edge * normals[None, ...]).sum(-1)  # (num_quad,T,1,8)
    v_n_var = (
        (
            normals[..., None, :]
            @ cov_vifx[
                ...,
                None,
                :,
                :,
            ]
            @ normals[..., None]
        )
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
    mahal = np.einsum("...bcdi,bcij,...bcdj->...bcd", delta_pdf, inv_cov_x, delta_pdf)
    p_x = (1 / (2 * np.pi * np.sqrt(det_cov_x[None, ..., None]))) * np.exp(-0.5 * mahal)
    integrand = E_vn_trunc * p_x
    # --- Integrate along edge with Gauss-Legendre quadrature and multiply by edge length ---
    cep = np.sum(
        np.sum(integrand * weights_xy_expanded, axis=0) * 0.5 * edge_lengths,
        axis=-1,
        dtype=cov_vifx.dtype,
    )  # (T,)

    coll_prob = np.trapz(cep, dx=dt, axis=-1)
    coll_prob = np.clip(coll_prob, None, 1)
    return coll_prob


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
    trajs2 = np.asarray(trajs_obs2)
    trajs1_covs = np.asarray(trajs_obs1_covs) if trajs_obs1_covs is not None else None
    trajs2_covs = np.asarray(trajs_obs2_covs) if trajs_obs2_covs is not None else None
    initial_obs2 = np.asarray(initial_obs2) if initial_obs2 is not None else None
    if trajs1.ndim == 2:
        trajs1 = trajs1[None, ...]
        initial_obs1 = initial_obs1[None, ...]

    if trajs2.ndim == 2:
        trajs2 = trajs2[None, ...]
    if trajs1_covs is not None and trajs1_covs.ndim == 3:
        trajs1_covs = trajs1_covs[None, ...]
    if trajs2_covs is not None and trajs2_covs.ndim == 3:
        trajs2_covs = trajs2_covs[None, ...]
    if initial_obs2 is not None and initial_obs2.ndim == 2:
        initial_obs2 = initial_obs2[None, ...]

    assert (
        trajs1.ndim == trajs2.ndim == 3 and trajs1.shape[-2:] == trajs2.shape[-2:]
    ), f"trajs1 and trajs2 must be list of trajectories with shape (N_traj, N_steps, state_dim), got {trajs1.shape} and {trajs2.shape}."
    assert (
        len(initial_obs1) == len(trajs1)
        and (trajs1_covs is None or trajs1_covs.shape[:-2] == trajs1.shape[:-1])
        and (initial_obs2 is None or len(initial_obs2) == len(trajs2))
        and (trajs2_covs is None or trajs2_covs.shape[:-2] == trajs2.shape[:-1])
    ), (
        f"inputs poly1 ({len(initial_obs1)}), trajs1 ({len(trajs1)}), trajs1_covs ({0 if trajs1_covs is None else len(trajs1_covs)}) lengths must match, "
        f"as well as trajs2 ({len(trajs2)}), trajs2 covs ({0 if trajs2_covs is None else len(trajs2_covs)}) lengths must match, "
        f"and poly2 ({0 if initial_obs2 is None else len(initial_obs2)}) lengths must match trajs2."
    )
    rel_trajs, rel_covs = convolve_distributions(
        trajs1[:, None], trajs1_covs[:, None], trajs2[None], trajs2_covs[None]
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

    coll_probs, _ = cpo.point_collvol_mc_sampling(
        obs1=obs1_ext[mask],
        obs2=states,
        obs1_cov=rel_covs[mask],
        with_orientation=with_orientation,
        num_samples=num_samples,
    )
    # coll_probs_full = np.zeros(mask.shape, dtype=coll_probs.dtype)
    # coll_probs_full[mask] = coll_probs

    # coll_probs = []
    # for i, (traj1, initial_poly1) in enumerate(zip(trajs1, initial_poly1_boundary)):
    #     coll_probs_traj1 = []
    #     d1 = np.max(
    #         np.linalg.norm(
    #             np.diff(np.vstack([initial_poly1, initial_poly1[0]]), axis=0), axis=1
    #         )
    #     )
    #     for j, traj2 in enumerate(trajs2):
    #         coll_probs_traj2 = []
    #         initial_poly2 = (
    #             initial_poly2_boundary[j]
    #             if initial_poly2_boundary is not None
    #             else None
    #         )
    #         d2 = (
    #             np.max(
    #                 np.linalg.norm(
    #                     np.diff(np.vstack([initial_poly2, initial_poly2[0]]), axis=0),
    #                     axis=1,
    #                 )
    #             )
    #             if initial_poly2 is not None
    #             else 0
    #         )
    #         d_min = d1 + d2
    #         for t, (state1, state2) in enumerate(zip(traj1, traj2)):
    #             # shift and rotate to current state
    #             delta_theta1 = state1[2] - initial_poly1[0, -1]
    #             rot1 = np.array(
    #                 [
    #                     [np.cos(delta_theta1), -np.sin(delta_theta1), 0],
    #                     [np.sin(delta_theta1), np.cos(delta_theta1), 0],
    #                     [0, 0, 1],
    #                 ]
    #             )
    #             poly1_boundary_t = (
    #                 initial_poly1 - np.mean(initial_poly1, axis=-2, keepdims=True)
    #             ) @ rot1.T + state1[
    #                 :3
    #             ]  # shift and rotate to current state

    #             poly1_cov = trajs1_covs[i, t] if trajs1_covs is not None else None

    #             if initial_poly2 is not None:
    #                 delta_theta2 = state2[2] - initial_poly2[0, -1]
    #                 rot2 = np.array(
    #                     [
    #                         [np.cos(delta_theta2), -np.sin(delta_theta2), 0],
    #                         [np.sin(delta_theta2), np.cos(delta_theta2), 0],
    #                         [0, 0, 1],
    #                     ]
    #                 )
    #                 poly2_boundary = (
    #                     initial_poly2 - np.mean(initial_poly2, axis=-2, keepdims=True)
    #                 ) @ rot2.T + state2[
    #                     :3
    #                 ]  # shift and rotate to current state
    #                 point_coords = None
    #             else:
    #                 poly2_boundary = None
    #                 point_coords = state2
    #             state2_cov = trajs2_covs[j, t] if trajs2_covs is not None else None
    #             poly2_cov = state2_cov if poly2_boundary is not None else None
    #             point_cov = state2_cov if poly2_cov is None else None
    #             if np.linalg.norm(state1[:2] - state2[:2]) - d_min >= 4 * np.sqrt(
    #                 (poly1_cov[0, 0] if poly1_cov is not None else 0)
    #                 + (
    #                     poly2_cov[0, 0]
    #                     if poly2_cov is not None
    #                     else (point_cov[0, 0] if point_cov is not None else 0)
    #                 )
    #             ):
    #                 coll_probs_traj2.append(0.0)
    #             else:
    #                 p, _ = cpo.point_collvol_mc_sampling(
    #                     poly1_boundary=poly1_boundary_t,
    #                     poly1_cov=poly1_cov,
    #                     poly2_boundary=poly2_boundary,
    #                     poly2_cov=poly2_cov,
    #                     point_coords=point_coords if poly2_boundary is None else None,
    #                     point_cov=point_cov,
    #                     with_orientation=with_orientation,
    #                     num_samples=num_samples,
    #                 )
    #                 coll_probs_traj2.append(p)
    #         coll_probs_traj1.append(coll_probs_traj2)
    #     coll_probs.append(coll_probs_traj1)
    # coll_probs = np.asarray(coll_probs)
    return coll_probs.max(axis=-1), coll_probs  # shape (N_traj1, N_traj2)


def point_collvol_mc_traj_sampling(
    initial_poly1_boundary: list[np.ndarray] | np.ndarray,
    initial_state_1: list[np.ndarray] | np.ndarray,
    initial_state_2: list[np.ndarray] | np.ndarray,
    initial_cov_1: Optional[list[np.ndarray] | np.ndarray] = None,
    initial_cov_2: Optional[list[np.ndarray] | np.ndarray] = None,
    initial_poly2_boundary: Optional[list[np.ndarray] | np.ndarray] = None,
    dt=0.1,
    with_orientation: bool = False,
    num_samples: int = 20000,
):

    idx = 3 if with_orientation else 2

    initial_poly1_boundary = np.asarray(initial_poly1_boundary)
    initial_state_1 = np.asarray(initial_state_1)
    initial_state_2 = np.asarray(initial_state_2)
    initial_cov_1 = np.asarray(initial_cov_1) if initial_cov_1 is not None else None
    initial_cov_2 = np.asarray(initial_cov_2) if initial_cov_2 is not None else None
    initial_poly2_boundary = (
        np.asarray(initial_poly2_boundary)
        if initial_poly2_boundary is not None
        else None
    )

    if initial_state_1.ndim == 1:
        trajs1 = trajs1[None, ...]
        initial_poly1_boundary = initial_poly1_boundary[None, ...]

    if initial_state_2.ndim == 1:
        trajs2 = trajs2[None, ...]
    if initial_cov_1 is not None and initial_cov_1.ndim == 2:
        initial_cov_1 = initial_cov_1[None, ...]
    if initial_cov_2 is not None and initial_cov_2.ndim == 2:
        initial_cov_2 = initial_cov_2[None, ...]
    if initial_poly2_boundary is not None and initial_poly2_boundary.ndim == 2:
        initial_poly2_boundary = initial_poly2_boundary[None, ...]

    # if poly1_cov is not None and (point_cov is not None or poly2_cov is not None):
    coll_probs = []
    p_cums = []
    masks = []
    trajs = []
    for i, (poly1, ini_state1) in enumerate(
        zip(initial_poly1_boundary, initial_state_1)
    ):
        poly1 = poly1 - np.mean(poly1, axis=-2)  # center polygon
        coll_probs_traj2 = []
        p_cums_traj2 = []
        masks_traj2 = []
        trajs_traj2 = []
        for j, ini_state2 in enumerate(initial_state_2):

            if initial_cov_2 is not None and initial_cov_1 is not None:
                _, rel_cov = convolve_distributions(
                    0, initial_cov_1[i], 0, initial_cov_2[j], dist="Gaussian"
                )
            elif initial_cov_2 is not None:
                rel_cov = initial_cov_2[j]
            elif initial_cov_1 is not None:
                rel_cov = initial_cov_1[i]
            else:
                rel_cov = None
            rel_state = ini_state2 - ini_state1

            mc_points = multivariate_normal(
                mean=rel_state, cov=rel_cov, seed=SEED, allow_singular=True
            ).rvs(size=num_samples)
            MC_rel_trajs, MC_rel_cov = simple_prediction(mc_points, rel_cov, dt=dt)

            poly2 = (
                initial_poly2_boundary[j]
                if initial_poly2_boundary is not None
                else None
            )
            if poly2 is not None:
                if with_orientation:
                    delta_theta = mc_points[..., idx - 1] - rel_state[..., idx - 1]
                    rot_theta = np.zeros((*delta_theta.shape, 3, 3))
                    costheta = np.cos(delta_theta)
                    sintheta = np.sin(delta_theta)
                    rot_theta[..., 0, 0] = costheta
                    rot_theta[..., 0, 1] = sintheta
                    rot_theta[..., 1, 0] = -sintheta
                    rot_theta[..., 1, 1] = costheta
                    rot_theta[..., 2, 2] = 1.0
                    poly2 = poly2 @ rot_theta
                    poly1 = poly1[None].repeat(poly2.shape[0], axis=0)
                    collvol, _ = collision_polygon(poly1, poly2)
                else:
                    collvol, _ = collision_polygon(poly1, poly2)
                    collvol = collvol[None]
            else:
                if (
                    with_orientation
                    and initial_cov_1 is not None
                    and initial_cov_1[i].shape[-1] >= 3
                ):
                    theta_std = np.sqrt(initial_cov_1[i, 2, 2])
                    delta_theta = norm(scale=theta_std).rvs(
                        size=num_samples, random_state=SEED
                    )
                    rot_theta = np.zeros((*delta_theta.shape, 3, 3))
                    costheta = np.cos(delta_theta)
                    sintheta = np.sin(delta_theta)
                    rot_theta[..., 0, 0] = costheta
                    rot_theta[..., 0, 1] = sintheta
                    rot_theta[..., 1, 0] = -sintheta
                    rot_theta[..., 1, 1] = costheta
                    rot_theta[..., 2, 2] = 1.0

                    collvol = poly1 @ rot_theta
                else:
                    collvol = poly1[None]
            if with_orientation:
                overlap = []
                for i in range(collvol.shape[0]):
                    poly = Polygon(collvol[i, :, :2])
                    overlap.append(
                        contains_xy(poly, MC_rel_trajs[i, :, 0], MC_rel_trajs[i, :, 1])
                    )
                overlap = np.asarray(overlap)  # (num_samples, N_steps)
            else:
                poly = Polygon(collvol[0, :, :2])
                overlap = contains_xy(
                    poly, MC_rel_trajs[:, :, 0], MC_rel_trajs[:, :, 1]
                )  # (num_samples, N_steps)

            entered = np.diff(overlap.astype(int), prepend=1, axis=1) == 1
            new_entries_per_timestep = np.sum(entered, axis=0)
            # print("New entries per timestep:", new_entries_per_timestep)
            # Probability of new entries per timestep
            v1 = new_entries_per_timestep / num_samples
            overlap_prob = np.sum(v1)  # Approximate integral over time by summation
            prob_rate = v1 / dt
            inside_mask = np.any(overlap, axis=1)
            p_cum = np.clip(np.cumsum(prob_rate) * dt, None, 1.0)
            coll_probs_traj2.append(overlap_prob)
            p_cums_traj2.append(p_cum)
            masks_traj2.append(inside_mask)
            trajs_traj2.append(MC_rel_trajs)

        coll_probs.append(coll_probs_traj2)
        p_cums.append(p_cums_traj2)
        masks.append(masks_traj2)
        trajs.append(trajs_traj2)

    coll_probs = np.asarray(coll_probs)
    p_cums = np.asarray(p_cums)
    masks = np.asarray(masks)
    trajs = np.asarray(trajs)
    return coll_probs, p_cums, {"trajs": trajs, "overlap_mask": masks}


def traj_collvol_analytic(
    initial_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1: list[np.ndarray] | np.ndarray,
    trajs_obs1_covs: list[np.ndarray] | np.ndarray,
    trajs_obs2: Optional[list[np.ndarray] | np.ndarray] = None,
    trajs_obs2_covs: Optional[list[np.ndarray] | np.ndarray] = None,
    initial_obs2: Optional[list[np.ndarray] | np.ndarray] = None,
    with_orientation: bool = False,
):

    initial_obs1 = np.asarray(initial_obs1)
    trajs1 = np.asarray(trajs_obs1)
    trajs2 = np.asarray(trajs_obs2)
    trajs1_covs = np.asarray(trajs_obs1_covs) if trajs_obs1_covs is not None else None
    trajs2_covs = np.asarray(trajs_obs2_covs) if trajs_obs2_covs is not None else None
    initial_obs2 = np.asarray(initial_obs2) if initial_obs2 is not None else None
    if trajs1.ndim == 2:
        trajs1 = trajs1[None, ...]
        initial_obs1 = initial_obs1[None, ...]

    if trajs2.ndim == 2:
        trajs2 = trajs2[None, ...]
    if trajs1_covs is not None and trajs1_covs.ndim == 3:
        trajs1_covs = trajs1_covs[None, ...]
    if trajs2_covs is not None and trajs2_covs.ndim == 3:
        trajs2_covs = trajs2_covs[None, ...]
    if initial_obs2 is not None and initial_obs2.ndim == 2:
        initial_obs2 = initial_obs2[None, ...]

    assert (
        trajs1.ndim == trajs2.ndim == 3 and trajs1.shape[-2:] == trajs2.shape[-2:]
    ), f"trajs1 and trajs2 must be list of trajectories with shape (N_traj, N_steps, state_dim), got {trajs1.shape} and {trajs2.shape}."
    assert (
        len(initial_obs1) == len(trajs1)
        and (trajs1_covs is None or trajs1_covs.shape[:-2] == trajs1.shape[:-1])
        and (initial_obs2 is None or len(initial_obs2) == len(trajs2))
        and (trajs2_covs is None or trajs2_covs.shape[:-2] == trajs2.shape[:-1])
    ), (
        f"inputs poly1 ({len(initial_obs1)}), trajs1 ({len(trajs1)}), trajs1_covs ({0 if trajs1_covs is None else len(trajs1_covs)}) lengths must match, "
        f"as well as trajs2 ({len(trajs2)}), trajs2 covs ({0 if trajs2_covs is None else len(trajs2_covs)}) lengths must match, "
        f"and poly2 ({0 if initial_obs2 is None else len(initial_obs2)}) lengths must match trajs2."
    )
    rel_trajs, rel_covs = convolve_distributions(
        trajs1[:, None], trajs1_covs[:, None], trajs2[None], trajs2_covs[None]
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
