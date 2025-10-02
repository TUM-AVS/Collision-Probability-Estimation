from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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

    # Center polygons around their centroids, will be origin for Minkowski sum
    poly1_center = np.mean(poly_corners_1, axis=-2, keepdims=True)
    # poly_corners_1[..., :2] -= poly1_center[..., :2]
    poly2_center = np.mean(poly_corners_2, axis=-2, keepdims=True)

    poly2_shifted = poly_corners_2 - poly2_center

    # Compute Minkowski sum with polygon1's center preserved
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
    minkowski_coords = [leftmost_bottom1 + leftmost_bottom2]
    for idx in sorted_indices[..., :-1].T:
        minkowski_coords.append(minkowski_coords[-1] + edges[(*batch_indices, idx.T)])
    collision_volume = np.stack(minkowski_coords, axis=-2)
    # relative angle between the two polygons
    collision_volume[..., -1] = poly2_center[..., -1] - poly1_center[..., -1]

    return collision_volume, poly2_center


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
        next_state = state + dt * np.stack(
            [v * costheta, v * sintheta, zero, zero], axis=-1
        )
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

    mus1 = mus1 if mus1 is not None else 0
    devs1 = devs1 if devs1 is not None else 0
    mus2 = mus2 if mus2 is not None else 0
    devs2 = devs2 if devs2 is not None else 0
    match dist:
        case "Gaussian" | "Normal" | "Cauchy":

            mus = [mus2, -mus1]
            sigs = [devs2, devs1]

            a = 1
            b = [0, 0]
            rel_mean, rel_cov = stable(mus, sigs, a, b)
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
    mask: Optional[np.ndarray] = None,
    poly_corners2: Optional[np.ndarray] = None,
):
    """
    Visualize the trajectory of a polygon.

    Parameters:
        traj (ndarray): NxM array of trajectory points.
        poly_corners (ndarray): Kx2 array of polygon corners.
    """
    if mask is None:
        mask = np.zeros(traj.shape[0], dtype=bool)
    fig, ax = plt.subplots()

    # collvol
    collvol, _ = (
        collision_polygon(poly_corners1, poly_corners2)
        if poly_corners2 is not None
        else poly_corners1
    )
    poly = plt.Polygon(
        collvol[:, :2],
        closed=True,
        edgecolor="#E37222",
        facecolor="none",
        linewidth=1,
        linestyle="--",
        alpha=1,
        zorder=13,
    )
    ax.add_patch(poly)

    # Plot polygons
    poly = plt.Polygon(
        poly_corners1[:, :2],
        edgecolor="#E37222",
        facecolor="None",
        alpha=0.7,
        linewidth=1,
        zorder=4,
        label="Ego Rectangle",
    )
    ax.add_patch(poly)
    if poly_corners2 is not None:
        # for i in range(0, len(traj), max(1, len(traj) // 20)):
        poly = plt.Polygon(
            poly_corners2[:, :2],
            edgecolor="#2266e3",
            facecolor="None",
            alpha=0.7,
            linewidth=1,
            zorder=4,
            label="Obs Rectangle",
        )
        ax.add_patch(poly)

    # Plot trajectory
    ax.plot(
        traj[mask, :, 0].T,
        traj[mask, :, 1].T,
        color="lightgray",
        alpha=0.5,
        label="Inside Rectangle",
        zorder=5,
    )
    ax.plot(
        traj[~mask, :, 0].T,
        traj[~mask, :, 1].T,
        color="#6B8E23",
        alpha=0.5,
        label="Inside Rectangle",
        zorder=5,
    )

    ax.scatter(
        traj[mask, 0, 0],
        traj[mask, 0, 1],
        color="darkgray",
        s=4,
        label="Inside Rectangle",
        zorder=5,
    )
    ax.scatter(
        traj[~mask, 0, 0],
        traj[~mask, 0, 1],
        color="darkgreen",
        s=4,
        label="Outside Rectangle",
        zorder=5,
    )
    means = traj.mean(axis=0)
    # ax.scatter(means[0], means[1], color='black', s=50, marker='x', label='Obs Center')
    ax.plot(
        means[:, 0],
        means[:, 1],
        color="black",
        linestyle="-",
        label="Mean Trajectory",
        zorder=6,
    )

    ax.set_aspect("equal")
    plt.title("Trajectory with Polygon Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
