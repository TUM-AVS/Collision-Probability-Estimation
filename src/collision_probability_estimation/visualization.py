import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from .np_impl.utils import collision_polygon


def visualize_overlap_point_poly(poly_corners, sampled_points, inside_mask, txt=""):
    """
    Visualize the overlap probability between a polygon and sampled points.

    Parameters:
        polygon (shapely.geometry.Polygon): The polygon to visualize.
        sampled_points (ndarray): Nx2 array of sampled points.
        inside_mask (ndarray): Boolean array indicating which points are inside the polygon.
    """

    fig, ax = plt.subplots()
    # xy = np.array(polygon.boundary.coords)[:, :2]
    poly_corners = np.atleast_3d(np.asarray(poly_corners))
    inside_mask = np.asarray(inside_mask, dtype=bool)
    sampled_points = np.asarray(sampled_points)
    
    for i, poly in enumerate(poly_corners):
        if i % 20 == 0:
            poly = plt.Polygon(
                poly[:, :2],
                closed=True,
            edgecolor="red",
            facecolor="gray",
            linewidth=2,
            alpha=0.4,
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
    plt.title(f"Overlap Probability {txt}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    return



def visualize_overlap_poly_poly(poly1_corners, poly2_corners, overlap_mask, txt= ""):
    """
    Visualize the overlap probability between two polygons.

    Parameters:
        polygon1 (shapely.geometry.Polygon): The first polygon.
        polygon2 (shapely.geometry.Polygon): The second polygon.
        overlap_mask (ndarray): Boolean array indicating which polygon pairs overlap.
    """
    use_torch=True
    if isinstance(overlap_mask, np.ndarray):
        use_torch=False
    fig, ax = plt.subplots()
    count=0
    for i, (polys1, polys2) in enumerate(zip(poly1_corners, poly2_corners)):
        for j, (poly1, poly2) in enumerate(zip(polys1, polys2)):
            if (i+j) % 20 == 0:
                count+=1
                overlap = overlap_mask[i, j]
                color = "#FF8484" if overlap else "#ACFFAC"
                poly1_patch = plt.Polygon(
                    poly1.exterior.coords[:] if not use_torch else poly1[:,:2],
                    closed=True,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=1,
                    alpha=0.5,
                )
                ax.add_patch(poly1_patch)
                poly2_patch = plt.Polygon(
                    poly2.exterior.coords[:] if not use_torch else poly2[:,:2],
                    closed=True,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=1,
                    alpha=0.5,
                )
                ax.add_patch(poly2_patch)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    # ax.legend()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.title(f"Overlap Probability {txt}")
    return


def visualize_traj_poly(
    poly_corners1: np.ndarray,
    traj: np.ndarray,
    mask: np.ndarray = None,
    poly_corners2: np.ndarray = None,
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
    collvol, _ = collision_polygon(poly_corners1, poly_corners2) if poly_corners2 is not None else poly_corners1
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
