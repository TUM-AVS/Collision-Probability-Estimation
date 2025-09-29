__author__ = "Marc Kaufeld"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


def simple_prediction(
    initial_state: np.ndarray,
    initial_cov: np.ndarray,
    dt: float = 0.1,
    num_steps: int = 30,
):
    """
    Simple constant velocity model prediction with EKF for position and orientation.
    State: [x, y, theta, vx, vy]
    Cov: 5x5 covariance matrix
    Args:
        initial_state: np.array([x, y, theta, vx, vy])
        initial_cov: 5x5 np.array, initial covariance matrix
        dt: float, time step
    """

    if initial_state.ndim == 1:
        initial_state = initial_state[np.newaxis, :]
    if initial_cov.ndim == 2:
        initial_cov = initial_cov[np.newaxis, :, :]
    # process noise
    q = 0**2
    sigma_theta = 0.0
    Q = np.zeros((5, 5))
    # Position x block
    Q[0, 0] = 0.25 * dt**4 * q
    Q[0, 3] = 0.5 * dt**3 * q
    Q[3, 0] = 0.5 * dt**3 * q
    Q[3, 3] = dt**2 * q
    # Position y block
    Q[1, 1] = 0.25 * dt**4 * q
    Q[1, 4] = 0.5 * dt**3 * q
    Q[4, 1] = 0.5 * dt**3 * q
    Q[4, 4] = dt**2 * q
    # Orientation noise (simple random walk)
    Q[2, 2] = sigma_theta**2 * dt

    traj = np.zeros((initial_state.shape[0], num_steps + 1, initial_state.shape[-1]))
    covs = np.zeros(
        (
            initial_state.shape[0],
            num_steps + 1,
            initial_state.shape[-1],
            initial_state.shape[-1],
        )
    )
    traj[:, 0] = initial_state
    covs[:, 0] = initial_cov

    R_loc_trans = np.array(
        [
            [
                np.cos(np.mean(initial_state[:, 2])),
                np.sin(np.mean(initial_state[:, 2])),
            ],
            [
                -np.sin(np.mean(initial_state[:, 2])),
                np.cos(np.mean(initial_state[:, 2])),
            ],
        ]
    )

    for t in range(1, num_steps + 1):
        state = traj[:, t - 1]
        cov = covs[:, t - 1]
        xy = state[:, :2]
        theta = state[:, 2]
        v = state[:, 3:]
        R = np.tile(np.eye(2), (state.shape[0], 1, 1))
        R[:, 0, 0] = np.cos(theta)
        R[:, 0, 1] = -np.sin(theta)
        R[:, 1, 0] = np.sin(theta)
        R[:, 1, 1] = np.cos(theta)

        # TODO is R_loc_trans =R? -> for single traj yes!
        v_loc = R_loc_trans @ v[..., :, np.newaxis]
        # Nonlinear motion model: x, y, theta, v
        # new_xy = xy + self.dt * v #
        new_xy = xy + dt * (R @ v_loc).squeeze(-1)
        next_state = np.hstack([new_xy, state[:, 2:]])

        # Jacobian of motion model w.r.t state
        F = np.tile(np.eye(5), (state.shape[0], 1, 1))
        vtheta = dt * R
        J = np.array([[0, -1], [1, 0]])  # 90 degree rotation matrix
        xytheta = dt * np.einsum("bij,jk, bk->bi", R, J, v_loc.squeeze(-1))
        F[:, 0, 2] = xytheta[:, 0]
        F[:, 0, 3:] = vtheta[:, 0]
        F[:, 1, 2] = xytheta[:, 1]
        F[:, 1, 3:] = vtheta[:, 1]
        next_cov = np.einsum("bij,bjk,blk->bil", F, cov, F) + Q
        traj[:, t] = next_state
        covs[:, t] = next_cov

    return traj.squeeze(), covs.squeeze()


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
        # if np.ndim(dev[0]) == 2:
        #     assert all(np.all(np.linalg.eigvals(s) > 0) for s in dev), f"sig must be positive definite,  got {dev}"
        # else:
        #     assert all(s > 0 for s in dev), f"sig must be > 0, got {dev}"

        rel_mu = sum(mu)
        dev_pow = [d**a for d in dev]
        rel_dev = sum(dev_pow) ** (1 / a)
        return rel_mu, rel_dev

    mus1 = mus1 if mus1 is not None else 0
    devs1 = devs1 if devs1 is not None else 0
    mus2 = mus2 if mus2 is not None else 0
    devs2 = devs2 if devs2 is not None else 0
    match dist:
        case "Gaussian" | "Normal" | "Cauchy":
            # Create all combinations of obs_mu and ego_mu
            # mus2_expanded = np.array([mus2[i] for i in range(len(mus2)) for j in range(len(mus1))])
            # mus1_expanded = np.array([mus1[j] for i in range(len(mus2)) for j in range(len(mus1))])

            # mus = [mus2_expanded, -mus1_expanded]

            mus = [mus2, -mus1]
            # Similarly expand the covariance matrices if they exist
            # devs2_expanded = np.array([devs2[i] for i in range(len(devs2)) for j in range(len(devs1))])
            # devs1_expanded = np.array([devs1[j] for i in range(len(devs2)) for j in range(len(devs1))])
            # sigs = [devs2_expanded, devs1_expanded]
            sigs = [devs2, devs1]
            # else:
            #     sig = [obs_dev, ego_dev]
            # mu = [obs_mu, -ego_mu]
            # sig = [obs_dev, ego_dev]
            a = 1
            b = [0, 0]
            rel_mean, rel_cov = stable(mus, sigs, a, b)
            return rel_mean, rel_cov
        case "Levy":
            mu = [obs_mu, -ego_mu]
            sig = [obs_dev, ego_dev]
            a = 0.5
            b = [1, 1]
            raise NotImplementedError("Levy distribution not implemented yet.")
            rel_mean, rel_cov = stable(mu, sig, a, b)
            return rel_mean, rel_cov
        case _:
            raise NotImplementedError(f"Distribution {dist} not implemented")


def visualize_overlap_point_poly(poly_corners, sampled_points, inside_mask):
    """
    Visualize the overlap probability between a polygon and sampled points.

    Parameters:
        polygon (shapely.geometry.Polygon): The polygon to visualize.
        sampled_points (ndarray): Nx2 array of sampled points.
        inside_mask (ndarray): Boolean array indicating which points are inside the polygon.
    """

    fig, ax = plt.subplots()
    # xy = np.array(polygon.boundary.coords)[:, :2]
    poly = plt.Polygon(
        poly_corners[:, :2],
        closed=True,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        alpha=0.1,
    )
    ax.add_patch(poly)
    # ax.plot(xy[:, 0], xy[:, 1], color='blue', label='Polygon Boundary')

    # Plot points
    ax.scatter(
        sampled_points[~inside_mask, 0],
        sampled_points[~inside_mask, 1],
        color="#ACFFAC",
        s=1,
        label="Outside Points",
    )
    ax.scatter(
        sampled_points[inside_mask, 0],
        sampled_points[inside_mask, 1],
        color="#FF8484",
        s=1,
        label="Inside Points",
    )

    mean = np.mean(sampled_points[:, :2], axis=0)
    cov = np.cov(sampled_points[:, :2], rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    for nsig, color in zip([1, 2, 3], ["red", "orange", "green"]):
        width, height = 2 * nsig * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        ellipse = Ellipse(
            mean,
            width,
            height,
            angle=angle,
            edgecolor=color,
            facecolor="none",
            linewidth=2,
            label=rf"{nsig}$\sigma$ Ellipse",
        )
        ax.add_patch(ellipse)

    ax.set_aspect("equal", "box")
    ax.legend()
    plt.title("Overlap Probability Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def visualize_overlap_poly_poly(poly1_corners, poly2_corners, overlap_mask):
    """
    Visualize the overlap probability between two polygons.

    Parameters:
        polygon1 (shapely.geometry.Polygon): The first polygon.
        polygon2 (shapely.geometry.Polygon): The second polygon.
        overlap_mask (ndarray): Boolean array indicating which polygon pairs overlap.
    """

    fig, ax = plt.subplots()
    for i, (p1, p2, overlap) in enumerate(
        zip(poly1_corners, poly2_corners, overlap_mask)
    ):
        # if overlap:
        if i % 50 == 0:
            color = "#FF8484" if overlap else "#ACFFAC"
            poly1 = plt.Polygon(
                p1.exterior.coords[:],
                closed=True,
                edgecolor=color,
                facecolor="none",
                linewidth=1,
                alpha=0.5,
            )
            ax.add_patch(poly1)
            poly2 = plt.Polygon(
                p2.exterior.coords[:],
                closed=True,
                edgecolor=color,
                facecolor="none",
                linewidth=1,
                alpha=0.5,
            )
            ax.add_patch(poly2)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    # ax.legend()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def visualize_traj_poly(
    poly_corners1: np.ndarray,
    traj: np.ndarray,
    poly_corners2: Optional[np.ndarray] = None,
):
    """
    Visualize the trajectory of a polygon.

    Parameters:
        traj (ndarray): NxM array of trajectory points.
        poly_corners (ndarray): Kx2 array of polygon corners.
    """

    fig, ax = plt.subplots()
    # Plot trajectory
    ax.plot(traj[:, 0], traj[:, 1], color="blue", label="Trajectory")
    poly = plt.Polygon(
        poly_corners1[:, :2],
        closed=True,
        edgecolor="red",
        facecolor="none",
        linewidth=1,
        alpha=0.5,
    )
    ax.add_patch(poly)
    # Plot polygon at each trajectory point
    if poly_corners2 is not None:
        for i in range(0, len(traj), max(1, len(traj) // 20)):
            poly = plt.Polygon(
                poly_corners2[:, :2] + traj[i, :2] - traj[0, :2],
                closed=True,
                edgecolor="red",
                facecolor="none",
                linewidth=1,
                alpha=0.5,
            )
            ax.add_patch(poly)

    ax.set_aspect("equal", "box")
    ax.legend()
    plt.title("Trajectory with Polygon Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def collision_polygon(poly_corners_1: np.ndarray, poly_corners_2: np.ndarray):
    """
    Compute and visualize the collision polygon between two polygons.
    Around polygon1

    Parameters:
        polygon1 (shapely.geometry.Polygon): The first polygon.
        polygon2 (shapely.geometry.Polygon): The second polygon.
    """
    assert (
        poly_corners_1.shape[-1] >= 2 and poly_corners_2.shape[-1] >= 2
    ), "Polygons must be Nx2 arrays of corner coordinates."
    assert (
        poly_corners_2.shape[-2] > 2 or poly_corners_1.shape[-2] > 2
    ), "At least one polygon must have at least 3 corners."

    if poly_corners_2.shape[-2] < 3:
        return poly_corners_1, poly_corners_2
    if poly_corners_1.shape[-2] < 3:
        return poly_corners_2, poly_corners_1

    # poly1_coords = np.array(polygon1.exterior.coords)
    # poly2_coords = np.array(polygon2.exterior.coords)
    # Ensure arrays can be broadcasted together
    # max_dims = poly_corners_1.ndim + poly_corners_2.ndim - 2
    # if poly_corners_1.ndim == 2 and poly_corners_2.ndim == 2:
    #     poly_corners_1 = poly_corners_1[np.newaxis, ...]
    #     poly_corners_2 = poly_corners_2[np.newaxis, ...]

    # # Make arrays broadcastable by expanding dimensions as needed
    # # shape1 = poly_corners_1.shape
    # # shape2 = poly_corners_2.shape
    # # max(len(shape1) - 2, len(shape2) - 2)

    # # Expand poly_corners_1 to match broadcasting requirements
    # i=0
    # while poly_corners_1.ndim < max_dims:
    #     i+=1
    #     poly_corners_1 = poly_corners_1[np.newaxis, ...].repeat(poly_corners_2.shape[-2-i], axis=0)

    # # Expand poly_corners_2 to match broadcasting requirements
    # i=0
    # while poly_corners_2.ndim < max_dims:
    #     i+=1

    #     poly_corners_2 = poly_corners_2[...,np.newaxis, :,:].repeat(poly_corners_2.shape[-2-i], axis=-2-i)
    #     raise NotImplementedError("TBT Order is reversed?.")
    # Center polygons around their centroids, will be origin for Minkowski sum
    poly1_center = np.mean(poly_corners_1, axis=-2, keepdims=True)
    # poly_corners_1[..., :2] -= poly1_center[..., :2]
    poly2_center = np.mean(poly_corners_2, axis=-2, keepdims=True)

    poly2_shifted = poly_corners_2 - poly2_center

    # # Compute Minkowski sum with polygon1's center preserved
    # # poly1_center = np.mean(poly1_coords[:-1], axis=0)
    # # poly2_center = np.mean(poly2_coords[:-1], axis=0)
    # poly2_center = polygon2.centroid.coords[0]
    edges1 = np.diff(poly_corners_1, axis=-2, append=poly_corners_1[..., 0:1, :])
    edges2 = np.diff(poly2_shifted, axis=-2, append=poly2_shifted[..., 0:1, :])
    edges = np.concatenate([edges1, edges2], axis=-2)
    angles1 = (np.arctan2(edges1[..., 1], edges1[..., 0]) + 2 * np.pi) % (2 * np.pi)
    angles2 = (np.arctan2(edges2[..., 1], edges2[..., 0]) + 2 * np.pi) % (2 * np.pi)
    angles = np.concatenate([angles1, angles2], axis=-1)

    # Sort edges by angle
    sorted_indices = np.argsort(angles, axis=-1)
    # Find leftmost among vertices with minimum y-coordinate to start the convex hull and move to origin
    min_y1_coord = np.min(
        poly_corners_1[..., 1], axis=-1, keepdims=True
    )  # [:,np.newaxis]
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
    # batch_indices = np.arange(poly_corners_1.shape[0])
    # leftmost_bottom1 = poly_corners_1[batch_indices, leftmost_idx1]
    # leftmost_bottom2 = poly2_shifted[batch_indices, leftmost_idx2]

    # batch_indices = [np.arange(i) for i in poly_corners_1.shape[:-2]]
    # import itertools
    # batch_indices = np.array(list(itertools.product(*batch_indices)))
    # batch_indices = np.arange( np.prod(poly_corners_1.shape[:-2]))
    # leftmost_bottom1 = poly_corners_1[batch_indices, leftmost_idx1]
    # leftmost_bottom2 = poly2_shifted[batch_indices, leftmost_idx2]

    # bottom1 = poly_corners_1[poly_corners_1[..., 1] == min_y1]
    # bottom2 = poly2_shifted[poly2_shifted[..., 1:2] == min_y2]
    # bottom1_min_x = np.min(bottom1[..., 0], axis=-1, keepdims=True)
    # bottom2_min_x = np.min(bottom2[..., 0], axis=-1, keepdims=True)
    # # leftmost_bottom1 = bottom1[bottom1[..., 0] ==
    # leftmost_bottom1 = poly_corners_1[poly_corners_1[..., 1] == min_y1][np.argmin(poly_corners_1[poly_corners_1[..., 1] == min_y1, 0], axis=-1, keepdims=True)]
    # leftmost_bottom2 = poly2_shifted[poly2_shifted[..., 1] == min_y2][np.argmin(poly2_shifted[poly2_shifted[..., 1] == min_y2, 0], axis=-1, keepdims=True)]

    # bottom_poly1 = poly_corners_1[np.argmin(poly_corners_1[:, 1])]
    # bottom_poly2 = poly_corners_2[np.argmin(poly_corners_2[:, 1])]
    # bottommost_combination = bottom_poly1 + bottom_poly2
    minkowski_coords = [leftmost_bottom1 + leftmost_bottom2]
    for idx in sorted_indices[..., :-1].T:
        minkowski_coords.append(minkowski_coords[-1] + edges[(*batch_indices, idx.T)])
    collision_volume = np.stack(minkowski_coords, axis=-2)
    # relative angle between the two polygons
    collision_volume[..., -1] = poly2_center[..., -1] - poly1_center[..., -1]

    # rel_angle = np.array([-np.pi/2])
    # octa_corners = np.array([
    #     poly1_coords[0] + poly2_coords[...,0,:]*(rel_angle[:] >= 0)[:,np.newaxis] + poly2_coords[...,1,:]*(rel_angle[:] < 0)[:,np.newaxis],  # 0
    #     poly1_coords[1] + poly2_coords[...,0,:]*(rel_angle[:] >= 0)[:,np.newaxis] + poly2_coords[...,1,:]*(rel_angle[:] < 0)[:,np.newaxis],  # 1
    #     poly1_coords[1] + poly2_coords[...,1,:]*(rel_angle[:] >= 0)[:,np.newaxis] + poly2_coords[...,2,:]*(rel_angle[:] < 0)[:,np.newaxis],  # 2
    #     poly1_coords[2] + poly2_coords[...,1,:]*(rel_angle[:] >= 0)[:,np.newaxis] + poly2_coords[...,2,:]*(rel_angle[:] < 0)[:,np.newaxis],  # 3
    #     poly1_coords[2] + poly2_coords[...,2,:]*(rel_angle[:] >= 0)[:,np.newaxis] + poly2_coords[...,3,:]*(rel_angle[:] < 0)[:,np.newaxis],  # 4
    #     poly1_coords[3] + poly2_coords[...,2,:]*(rel_angle[:] >= 0)[:,np.newaxis] + poly2_coords[...,3,:]*(rel_angle[:] < 0)[:,np.newaxis],  # 5
    #     poly1_coords[3] + poly2_coords[...,3,:]*(rel_angle[:] >= 0)[:,np.newaxis] + poly2_coords[...,0,:]*(rel_angle[:] < 0)[:,np.newaxis],  # 6
    #     poly1_coords[0] + poly2_coords[...,3,:]*(rel_angle[:] >= 0)[:,np.newaxis] + poly2_coords[...,0,:]*(rel_angle[:] < 0)[:,np.newaxis],  # 7
    # ])
    # octa_corners = np.transpose(octa_corners, (1, 0, 2))

    # fig, ax = plt.subplots()
    # # xy = np.array(collision_poly.exterior.coords)[:, :2]
    # for i in range(0,collision_volume.shape[0]):
    #     for j in range(0,poly_corners_1.shape[1]):
    #         poly = plt.Polygon(collision_volume[i,j, :, :2], closed=True, edgecolor='blue', facecolor='red', linewidth=2,
    #                         alpha=0.5)
    #         ax.add_patch(poly)
    # poly = plt.Polygon(poly_corners_1[0, :, :2], closed=True, edgecolor='green', facecolor='green', linewidth=2,
    #                    alpha=0.5)
    # ax.add_patch(poly)

    # poly = plt.Polygon(poly_corners_2[0, :, :2], closed=True, edgecolor='green', facecolor='green', linewidth=2,
    #                    alpha=0.5)
    # ax.add_patch(poly)
    # for corner in minkowski_coords:
    #     translated_poly2 = poly2_shifted + corner
    #     poly = plt.Polygon(translated_poly2[:, :2], closed=True, edgecolor='orange', facecolor='none', linewidth=1,
    #                        alpha=0.7)
    #     ax.add_patch(poly)
    # # ax.set_aspect('equal')
    # ax.set_xlim(-8, 8)
    # ax.set_ylim(-8, 8)
    # # plt.tight_layout()
    # plt.title('Collision Polygon Visualization')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.grid(True)
    # plt.show()
    return collision_volume, poly2_center
