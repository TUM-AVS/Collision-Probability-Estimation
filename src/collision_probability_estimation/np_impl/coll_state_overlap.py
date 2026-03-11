"""
Collision probability estimation based on spatial overlap of polygon and point.
Following methods are implemented:
1) Monte Carlo sampling for polygon-polygon collision probability.
2) Inclusion-Exclusion principle for rectangle using bivariate normal CDF.
3) Monte Carlo sampling for point-in-polygon collision probability.
4) Analytic solution for point-in-polygon collision probability.
5) Mahalanobis distance-based collision probability.
6) Corner Mahalanobis distance-based collision probability (boundary distance).
"""


import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import gammainc
from shapely.geometry import Polygon
from shapely import contains_xy
from shapely.affinity import translate, rotate
from scipy.special import erf

from collision_probability_estimation.params import SEED, NODES_XY, WEIGHTS_XY, NODES_THETA, WEIGHTS_THETA, THETA_NSTD
from .utils import (collision_polygon, 
                    convolve_distributions, 
                    point_to_polygon_boundary_dist, 
                    build_rotation_matrices)


def poly_poly_mc_sampling(
    obs1: np.ndarray,
    obs2: np.ndarray,
    obs1_cov: np.ndarray = None,
    obs2_cov: np.ndarray = None,
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
    assert obs1.shape[-1] >= 2 and obs2.shape[-1] >= 2, "Polygons must be Nx2 arrays of corner coordinates."
    assert obs1.shape[-2] > 2 and obs2.shape[-2] > 2, "Polygons must have at least 3 corners."

    idx = 3 if with_orientation else 2
    # if with_orientation:
    poly1 = Polygon(obs1[:, :2])
    poly2 = Polygon(obs2[:, :2])
    poly1_rects = [poly1]
    poly2_rects = [poly2]
    if obs1_cov is not None:
        obs1_cov = obs1_cov[..., :idx, :idx]
        poly1_shift = multivariate_normal(cov=obs1_cov).rvs(size=int(np.sqrt(num_samples)), random_state=SEED)
        poly1_shift = np.column_stack(
            [poly1_shift, np.zeros(poly1_shift.shape[0])]
        ) 
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
        poly2_shift = multivariate_normal(cov=obs2_cov).rvs(size=int(np.sqrt(num_samples)), random_state=SEED)
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
        [[poly_rect1.intersects(poly_rect2) for poly_rect1 in poly1_rects] for poly_rect2 in poly2_rects]
    )
    coll_prob = np.mean(overlap_mask)

    return np.atleast_1d(coll_prob), {
        "poly1": [poly1_rects]*len(poly2_rects),
        "poly2": [[i]*len(poly1_rects) for i in poly2_rects],
        "overlap_mask": overlap_mask,
    }


def point_rect_incl_excl(
    obs1: np.ndarray,
    obs2: np.ndarray = None,
    obs1_cov: np.ndarray = None,
    obs2_cov: np.ndarray = None,
):
    """
    Calculates the proportion of a 2D Gaussian (mean=point, cov=cov) inside a rectangle
    using the inclusion-exclusion principle.

    Returns:
        proportion (float): Estimated probability mass inside the rectangle.
    """
    assert obs1.shape[-2] == 4, "Rectangle must have 4 boundary points (closed polygon)."
    _, rel_cov = convolve_distributions(devs1=obs1_cov, devs2=obs2_cov, dist="Gaussian")
    xy = obs1[..., :2]
    theta = obs1[..., 0, 2]
    if obs2.ndim == 1:
        obs2 = obs2[None]
    point_coords = obs2.mean(axis=-2)
    # Rotation matrix such that rectangle is axis-aligned
    while rel_cov.ndim < 3:
        rel_cov = rel_cov[None]
    rot = build_rotation_matrices(theta)[...,:2,:2]
    xy_aligned = xy @ rot  # rotate rectangle
    x_min = xy_aligned[...,0].min(axis=-1)
    x_max = xy_aligned[...,0].max(axis=-1)
    y_min = xy_aligned[...,1].min(axis=-1)
    y_max = xy_aligned[...,1].max(axis=-1)
    cov_aligned = rot.swapaxes(-1, -2) @ rel_cov[..., :2, :2] @ rot  
    point_aligned = (point_coords[...,None, :2] @ rot).squeeze()  

    def bivariate_cdf(upper_x, upper_y, means, covs):
        """Vectorised bivariate normal CDF evaluated at (upper_x, upper_y)."""
        # standardise
        sig_x = np.sqrt(covs[..., 0, 0])
        sig_y = np.sqrt(covs[..., 1, 1])

        # conditional: P(X < x, Y < y) = P(X < x) * P(Y < y | X < x)
        # Use the standard formula via scipy looped over batch
        batch_shape = means.shape[:-1]
        result = np.empty(batch_shape)
        flat_means = means.reshape(-1, 2)
        flat_covs  = covs.reshape(-1, 2, 2)
        flat_ux    = upper_x.ravel()
        flat_uy    = upper_y.ravel()
        for i in range(flat_means.shape[0]):
            result.ravel()[i] = multivariate_normal(
                mean=flat_means[i], cov=flat_covs[i]
            ).cdf([flat_ux[i], flat_uy[i]])
        return result

    means2d = point_aligned[..., :2]               # (N, 2)
    covs2d  = cov_aligned[..., :2, :2]             # (N, 2, 2)

    F1 = bivariate_cdf(x_max, y_max, means2d, covs2d)
    F2 = bivariate_cdf(x_min, y_max, means2d, covs2d)
    F3 = bivariate_cdf(x_max, y_min, means2d, covs2d)
    F4 = bivariate_cdf(x_min, y_min, means2d, covs2d)

    coll_prob = F1 - F2 - F3 + F4
    return np.atleast_1d(coll_prob)


def point_collvol_mc_sampling(
    obs1: np.ndarray,
    obs2: np.ndarray,
    obs1_cov: np.ndarray = None,
    obs2_cov: np.ndarray = None,
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
    if np.any(rel_cov):
        L = np.linalg.cholesky(rel_cov[..., :seg_idx, :seg_idx])                     
        z = np.random.default_rng(SEED).standard_normal(     
            size=(num_samples, *obs2_center[..., :seg_idx].shape)
        )
        sampled_points = obs2_center[..., :seg_idx] + (z[..., np.newaxis, :] @ L.swapaxes(-1, -2)).squeeze(-2)
    else:
        sampled_points = obs2_center[np.newaxis, ..., :seg_idx]

    if with_orientation and rel_cov is not None and rel_cov.shape[-1] >= 3:
        delta_theta = sampled_points[..., seg_idx - 1] - obs2_center[..., seg_idx - 1]
        rot_theta = build_rotation_matrices(delta_theta)
        obs2_rot = obs2 @ rot_theta
        obs2_rot[...,-1] += delta_theta[..., np.newaxis]
        obs1_ext = obs1[None].repeat(obs2_rot.shape[0], axis=0)
        collvol, _ = collision_polygon(obs1_ext, obs2_rot)
        if collvol.ndim == 3:
            collvol = collvol[:, None]
    else:
        collvol, _ = collision_polygon(obs1, obs2)
        if collvol.ndim == 2:
            collvol = collvol[None]
    # Check if points are inside polygon
    if sampled_points.ndim == 2:
        points = sampled_points[:, np.newaxis, :].repeat(collvol.shape[1], axis=1)
    else:
        points = sampled_points
    overlap_mask = []
    if with_orientation:
        for i in range(collvol.shape[1]):
            oriented_mask = []
            for j in range(num_samples):
                poly = Polygon(collvol[j, i, :, :2])
                oriented_mask.append(contains_xy(poly, points[j, i, 0], points[j, i, 1]))
            overlap_mask.append(oriented_mask)
    else:
        for idx, poly in enumerate(collvol):
            poly = Polygon(poly[:, :2])
            overlap_mask.append(contains_xy(poly, points[:, idx, 0], points[:, idx, 1]))
    overlap_mask = np.array(overlap_mask).squeeze() 
    coll_prob = np.mean(overlap_mask, axis=-1)
    if collvol.ndim >3:
        collvol = collvol.squeeze()
    return np.atleast_1d(coll_prob), {"points": sampled_points, "overlap_mask": overlap_mask, "collvol": collvol}


def point_collvol_analytic(
    obs1: np.ndarray,
    obs2: np.ndarray,
    obs1_cov: np.ndarray = None,
    obs2_cov: np.ndarray = None,
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

    if obs1.ndim - obs2.ndim == 1:
        obs2 = obs2[..., np.newaxis, :]
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
        delta_theta = THETA_NSTD * NODES_THETA[:, None] * theta_std
        theta_gauss = delta_theta + theta_mean
        theta_jacobian = THETA_NSTD * theta_std
        rot_theta = build_rotation_matrices(delta_theta)
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

    x_quad = (x_high - x_low) / 2 * NODES_XY[np.newaxis, :] + (x_high + x_low) / 2
    jacobian = (x_high - x_low) / 2

    erf_para_quad = -(m[..., np.newaxis] * x_quad + c[..., np.newaxis]) / np.sqrt(2)
    exp_para_quad = -(x_quad**2) / 2
    integ_quad = erf(erf_para_quad) * np.exp(exp_para_quad)
    p_sides = 1 / (np.sqrt(8 * np.pi)) * np.sum(integ_quad * WEIGHTS_XY * jacobian, axis=-1)
    overlap_prob = np.sum(p_sides, axis=-1)

    if with_orientation:
        theta_pdf = norm(loc=theta_mean, scale=theta_std).pdf(theta_gauss)
        overlap_prob = np.sum(overlap_prob * theta_pdf * WEIGHTS_THETA[:, None] * theta_jacobian, axis=0)

    return overlap_prob



def mahalanobis_distance(
    obs1: np.ndarray,
    obs2: np.ndarray,
    obs1_cov: np.ndarray = None,
    obs2_cov: np.ndarray = None,
    with_orientation: bool = False,
    use_corners: bool = False,
) -> np.ndarray:
    """
    Mahalanobis distance between two uncertain observations.

    Computes the Mahalanobis distance between two observations (points or polygons)
    considering their covariance. For polygons, uses the centroids.

    Args:
        obs1: First observation (polygon vertices or point coordinates)
        obs2: Second observation (polygon vertices or point coordinates)
        obs1_cov: Covariance matrix of first observation
        obs2_cov: Covariance matrix of second observation
        with_orientation: Whether to consider orientation in distance
        use_corners: Whether to use closest collision volume corner instead of centroid

    Returns:
        prob: Collision probability derived from Mahalanobis distance
    """
    obs1 = np.asarray(obs1, dtype=np.float64)
    obs2 = np.asarray(obs2, dtype=np.float64)

    if obs1_cov is not None:
        obs1_cov = np.asarray(obs1_cov, dtype=np.float64)
    if obs2_cov is not None:
        obs2_cov = np.asarray(obs2_cov, dtype=np.float64)

    idx = 3 if with_orientation else 2

    # Compute centroids
    obs1_center = obs1.mean(axis=-2)   # (..., 3)
    obs2_center = obs2.mean(axis=-2)   # (..., 3)

    # Convolve distributions
    rel_traj, rel_cov = convolve_distributions(
        mus1=obs1_center,
        mus2=obs2_center,
        devs1=obs1_cov,
        devs2=obs2_cov,
    )

    if rel_cov is None:
        raise ValueError("Covariance is required for Mahalanobis distance.")

    if not use_corners:
        delta_pos = rel_traj[..., :idx, np.newaxis]         
    else:
        collvol, points = collision_polygon(obs1, obs2)
        delta_pos = point_to_polygon_boundary_dist(points, collvol)[...,:idx, np.newaxis] 
    # Extract relevant covariance block
    cov_block = rel_cov[..., :idx, :idx]                   

    # Compute inverse covariance (Cholesky for stability)
    try:
        inv_cov = np.linalg.inv(cov_block)                  
    except np.linalg.LinAlgError:
        reg = 1e-8 * np.eye(idx, dtype=cov_block.dtype)
        inv_cov = np.linalg.inv(cov_block + reg)

    mahal_sq = (delta_pos.swapaxes(-1, -2) @ inv_cov @ delta_pos).squeeze((-1, -2))  # (...)

    # Convert to probability via chi² CDF with `idx` degrees of freedom
    prob = 1.0 - gammainc(idx / 2, mahal_sq / 2)

    return np.atleast_1d(prob)


