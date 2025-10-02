"""
Collision probability estimation based on spatial overlap of polygon and point.
Following methods are implemented:
1) Monte Carlo sampling for polygon-polygon collision probability.
2) Inclusion-Exclusion principle for rectangle (axis-aligned).
3) Monte Carlo sampling for point-in-polygon collision probability.
4) Analytic solution for point-in-polygon collision probability.
"""

from typing import Optional

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal, norm
from shapely.geometry import Polygon
from shapely import contains_xy
from shapely.affinity import translate, rotate
from scipy.special import erf

from params import SEED, NODES_XY, WEIGHTS_XY, NODES_THETA, WEIGHTS_THETA, THETA_NSTD
from utils import collision_polygon, convolve_distributions


def poly_poly_mc_sampling(
    obs1: np.ndarray,
    obs2: np.ndarray = None,
    obs1_cov: Optional[np.ndarray] = None,
    obs2_cov: Optional[np.ndarray] = None,
    num_samples: int = 20000,
    with_orientation: bool = False,
):
    """
    Monte Carlo sampling to compute the collision probability between two polygons.
    Parameters:
        obs1: ndarray: Nx2/3 array of first polygon vertices.
        obs2: ndarray: Mx2/3 array of second polygon vertices, Could also be a single point.
        obs1_cov: ndarray: 2x2 / 3x3 covariance matrix of polygon position uncertainty.
        obs2_cov: ndarray: 2x2 / 3x3 covariance matrix of point position uncertainty.
        with_orientation (bool): Whether to consider orientation uncertainty.
        num_samples (int): Number of random samples.
    Returns:
        float: Estimated collision probability.
        dict: Additional data (sampled points, overlap mask).
    """
    assert (
        obs1.shape[-1] >= 2 and obs2.shape[-1] >= 2
    ), "Polygons must be Nx2 arrays of corner coordinates."
    assert (
        obs1.shape[-2] > 2 and obs2.shape[-2] > 2
    ), "Polygons must have at least 3 corners."

    idx = 3 if with_orientation else 2
    # if with_orientation:
    poly1 = Polygon(obs1[:, :2])
    poly2 = Polygon(obs2[:, :2])
    poly1_rects = [poly1]
    poly2_rects = [poly2]
    if obs1_cov is not None:
        obs1_cov = obs1_cov[..., :idx, :idx]
        poly1_shift = multivariate_normal(cov=obs1_cov).rvs(
            size=int(np.sqrt(num_samples)), random_state=SEED
        )
        poly1_shift = np.column_stack(
            [poly1_shift, np.zeros(poly1_shift.shape[0])]
        )  # append zero theta for rotation if necessary, else not used
        poly1_rects = [
            rotate(
                translate(poly1, xoff=xy[0], yoff=xy[1]),
                angle=xy[2],
                use_radians=True,
                origin="centroid",
            )
            for xy in poly1_shift
        ]
    if obs2_cov is not None:
        obs2_cov = obs2_cov[..., :idx, :idx]
        poly2_shift = multivariate_normal(cov=obs2_cov).rvs(
            size=int(np.sqrt(num_samples)), random_state=SEED
        )
        poly2_shift = np.column_stack([poly2_shift, np.zeros(poly2_shift.shape[0])])
        poly2_rects = [
            rotate(
                translate(poly2, xoff=xy[0], yoff=xy[1]),
                angle=xy[2],
                use_radians=True,
                origin="centroid",
            )
            for xy in poly2_shift
        ]

    overlap_mask = np.array(
        [
            poly_rect1.intersects(poly_rect2)
            for poly_rect1 in poly1_rects
            for poly_rect2 in poly2_rects
        ]
    )
    coll_prob = np.mean(overlap_mask)
    return coll_prob, {
        "poly1_rects": poly1_rects,
        "poly2_rects": poly2_rects,
        "overlap_mask": overlap_mask,
    }


def point_rect_inc_excl(
    obs1: np.ndarray,
    obs2: np.ndarray = None,
    obs1_cov: Optional[np.ndarray] = None,
    obs2_cov: Optional[np.ndarray] = None,
):
    """
    Calculates the proportion of a 2D Gaussian (mean=point, cov=cov) inside a rectangle
    using the inclusion-exclusion principle.

    Returns:
        proportion (float): Estimated probability mass inside the rectangle.
    """
    assert obs1.shape[0] == 4, "Rectangle must have 4 boundary points (closed polygon)."
    _, rel_cov = convolve_distributions(devs1=obs1_cov, devs2=obs2_cov, dist="Gaussian")
    xy = obs1[:, :2]
    theta = obs1[0, 2]
    obs2 = obs2[None]
    point_coords = obs2.mean(axis=-2)
    # Rotation matrix such that rectangle is axis-aligned
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    xy_aligned = xy @ rot  # rotate rectangle
    x_min, y_min = xy_aligned.min(axis=0)
    x_max, y_max = xy_aligned.max(axis=0)
    cov_aligned = rot.T @ rel_cov[:2, :2] @ rot  # rotate covariance
    point_aligned = point_coords[:2] @ rot  # rotate points

    # Create multivariate normal distribution for points
    dist = multivariate_normal(mean=point_aligned, cov=cov_aligned)
    # Inclusion-Exclusion for rectangle: P(x_min < X < x_max, y_min < Y < y_max)
    # = F(x_max, y_max) - F(x_min, y_max) - F(x_max, y_min) + F(x_min, y_min)
    F1 = dist.cdf([x_max, y_max])
    F2 = dist.cdf([x_min, y_max])
    F3 = dist.cdf([x_max, y_min])
    F4 = dist.cdf([x_min, y_min])
    coll_prob = F1 - F2 - F3 + F4
    return coll_prob


def point_collvol_mc_sampling(
    obs1: np.ndarray,
    obs2: np.ndarray = None,
    obs1_cov: Optional[np.ndarray] = None,
    obs2_cov: Optional[np.ndarray] = None,
    num_samples: int = 20000,
    with_orientation: bool = False,
):
    """
    Monte Carlo sampling to compute the collision probability of a point (Gaussian distributed) inside a polygon.
    Parameters:
        obs1: ndarray: Nx2/3 array of first polygon vertices.
        obs2: ndarray: Mx2/3 array of second polygon vertices, Could also be a single point.
        obs1_cov: ndarray: 2x2 / 3x3 covariance matrix of polygon position uncertainty.
        obs2_cov: ndarray: 2x2 / 3x3 covariance matrix of point position uncertainty.
        with_orientation (bool): Whether to consider orientation uncertainty.
        num_samples (int): Number of random samples.
    Returns:
        float: Estimated collision probability.
        dict: Additional data (sampled points, overlap mask).
    """
    seg_idx = 3 if with_orientation else 2
    if obs2.ndim == 1:
        obs2 = obs2[np.newaxis, :]
    obs1_center = np.mean(obs1, axis=-2)
    obs2_center = np.mean(obs2, axis=-2)

    rel_dist, rel_cov = convolve_distributions(
        mus1=obs1_center,
        mus2=obs2_center,
        devs1=obs1_cov,
        devs2=obs2_cov,
        dist="Gaussian",
    )

    # Sample random points, if no covariance provided test whether point is inside polygon
    points = (
        MultivariateNormal(
            loc=torch.tensor(obs2_center[..., :seg_idx]),
            covariance_matrix=torch.tensor(rel_cov[..., :seg_idx, :seg_idx]),
        )
        .sample((num_samples,))
        .numpy()
        if np.any(rel_cov)
        else obs2_center[np.newaxis, ..., :seg_idx]
    )

    if with_orientation and rel_cov is not None and rel_cov.shape[-1] >= 3:
        delta_theta = points[..., seg_idx - 1] - obs2_center[..., seg_idx - 1]
        rot_theta = np.zeros((*delta_theta.shape, 3, 3))
        costheta = np.cos(delta_theta)
        sintheta = np.sin(delta_theta)
        rot_theta[..., 0, 0] = costheta
        rot_theta[..., 0, 1] = sintheta
        rot_theta[..., 1, 0] = -sintheta
        rot_theta[..., 1, 1] = costheta
        rot_theta[..., 2, 2] = 1.0
        obs2_rot = obs2 @ rot_theta
        obs1_ext = obs1[None].repeat(obs2_rot.shape[0], axis=0)
        collvol, _ = collision_polygon(obs1_ext, obs2_rot)
        if collvol.ndim == 3:
            collvol = collvol[:, None]
    else:
        collvol, _ = collision_polygon(obs1, obs2)
        if collvol.ndim == 2:
            collvol = collvol[None]
    # Check if points are inside polygon
    if points.ndim == 2:
        points = points[:, np.newaxis, :].repeat(collvol.shape[1], axis=1)
    overlap_mask = []
    if with_orientation:
        for i in range(collvol.shape[1]):
            tmp = []
            for j in range(num_samples):
                poly = Polygon(collvol[j, i, :, :2])
                tmp.append(contains_xy(poly, points[j, i, 0], points[j, i, 1]))
            overlap_mask.append(tmp)
    else:
        for idx, poly in enumerate(collvol):
            poly = Polygon(poly[:, :2])
            overlap_mask.append(contains_xy(poly, points[:, idx, 0], points[:, idx, 1]))
    overlap_mask = np.array(overlap_mask)  # (num_samples, N_polygons)
    coll_prob = np.mean(overlap_mask, axis=-1).squeeze()
    return coll_prob, {"points": points, "overlap_mask": overlap_mask}


def point_collvol_analytic(
    obs1: np.ndarray,
    obs2: np.ndarray = None,
    obs1_cov: Optional[np.ndarray] = None,
    obs2_cov: Optional[np.ndarray] = None,
    with_orientation: bool = False,
):
    """
    Analytic solution to compute the collision probability of a point (Gaussian distributed) inside a polygon.
    Parameters:
        obs1: ndarray: Nx2/3 array of first polygon vertices.
        obs2: ndarray: Mx2/3 array of second polygon vertices, Could also be a single point.
        obs1_cov: ndarray: 2x2 / 3x3 covariance matrix of polygon position uncertainty.
        obs2_cov: ndarray: 2x2 / 3x3 covariance matrix of point position uncertainty.
        with_orientation (bool): Whether to consider orientation uncertainty.
    Returns:
        float: Estimated collision probability.
    """

    if obs2.ndim == 1:
        obs2 = obs2[np.newaxis, :]
    obs1_center = np.mean(obs1, axis=-2)
    obs2_center = np.mean(obs2, axis=-2)

    rel_traj, rel_cov = convolve_distributions(
        mus1=obs1_center,
        mus2=obs2_center,
        devs1=obs1_cov,
        devs2=obs2_cov,
        dist="Gaussian",
    )
    if rel_cov is None:
        raise NotImplementedError("Analytic solution not implemented for this case.")

    if with_orientation and rel_cov is not None and rel_cov.shape[-1] >= 3:

        theta_mean = rel_traj[..., 2]
        theta_std = np.sqrt(rel_cov[..., 2, 2])
        # convert nodes from [-1, 1] to [theta_mean - THETA_NSTD*theta_std, theta_mean + THETA_NSTD*theta_std]
        delta_theta = THETA_NSTD * NODES_THETA[:, None] * theta_std
        theta_gauss = delta_theta + theta_mean
        theta_jacobian = THETA_NSTD * theta_std
        rot_theta = np.zeros((*delta_theta.shape, 3, 3))
        costheta = np.cos(delta_theta)
        sintheta = np.sin(delta_theta)
        rot_theta[..., 0, 0] = costheta
        rot_theta[..., 0, 1] = sintheta
        rot_theta[..., 1, 0] = -sintheta
        rot_theta[..., 1, 1] = costheta
        rot_theta[..., 2, 2] = 1.0
        obs2_rot = obs2 @ rot_theta
        obs1_ext = np.broadcast_to(
            obs1[*[None] * (obs2_rot.ndim - obs1.ndim)],
            (obs2_rot.shape[:-2] + obs1.shape[-2:]),
        )

        collvol, points = collision_polygon(obs1_ext, obs2_rot)
    else:
        collvol, points = collision_polygon(obs1, obs2)
        collvol = collvol[None]

    W = np.linalg.inv(np.linalg.cholesky(rel_cov))
    poly_shifted = (collvol - points) @ W[..., :3, :3].swapaxes(
        -1, -2
    )  # shift polygon to origin of point dist and scale with cov
    delta = np.diff(poly_shifted[..., :2], axis=-2, append=poly_shifted[..., 0:1, :2])
    m = delta[..., 1] / (delta[..., 0] + 1e-8)
    c = poly_shifted[..., 1] - m * poly_shifted[..., 0]
    x_low = poly_shifted[..., 0, np.newaxis]
    x_high = np.roll(x_low, shift=-1, axis=-2)

    # Linear transformation: x = (xh - xl)/2 * node + (xh + xl)/2
    x_quad = (x_high - x_low) / 2 * NODES_XY[np.newaxis, :] + (x_high + x_low) / 2
    jacobian = (x_high - x_low) / 2

    erf_para_quad = -(m[..., np.newaxis] * x_quad + c[..., np.newaxis]) / np.sqrt(2)
    exp_para_quad = -(x_quad**2) / 2
    integ_quad = erf(erf_para_quad) * np.exp(exp_para_quad)
    p_sides = (
        1 / (np.sqrt(8 * np.pi)) * np.sum(integ_quad * WEIGHTS_XY * jacobian, axis=-1)
    )
    overlap_prob = np.sum(p_sides, axis=-1)

    if with_orientation:
        theta_pdf = norm(loc=theta_mean, scale=theta_std).pdf(theta_gauss)
        overlap_prob = np.sum(
            overlap_prob * theta_pdf * WEIGHTS_THETA[:, None] * theta_jacobian, axis=0
        )

    return overlap_prob.squeeze()
