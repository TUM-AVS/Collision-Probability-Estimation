import time
from tabulate import tabulate
import numpy as np
from shapely.geometry import Polygon
from utils import (
    convolve_distributions,
    collision_polygon,
    visualize_overlap_point_poly,
    visualize_overlap_poly_poly,
)
import coll_prob_state_overlap as cpo


# mc sampling
num_samples = 20000

# EGO ############
ego_center = np.array([1.5, 0])
ego_angle = 0 * np.pi / 8
ego_center = np.concatenate([ego_center, [ego_angle]])

ego_rot = np.array(
    [
        [np.cos(ego_angle), np.sin(ego_angle), 0],
        [-np.sin(ego_angle), np.cos(ego_angle), 0],
        [0, 0, 1],
    ]
)

ego_length = 5
ego_width = 2
ego_poly_corners = np.array(
    [
        [-ego_length / 2, -ego_width / 2, 0],
        [ego_length / 2, -ego_width / 2, 0],
        [ego_length / 2, ego_width / 2, 0],
        [-ego_length / 2, ego_width / 2, 0],
    ]
)
ego_poly_corners = ego_poly_corners @ ego_rot + ego_center
# ego covariance
ego_sigma_x = 0.03  # in ego direction
ego_sigma_y = 0.01  # in ego direction
ego_sigma_theta = 1e-5  # in rad
ego_cov_xy = 0.0

ego_cov = np.array(
    [
        [ego_sigma_x**2, ego_cov_xy, 0],
        [ego_cov_xy, ego_sigma_y**2, 0],
        [0, 0, ego_sigma_theta**2],
    ]
)
ego_cov = ego_rot.T @ ego_cov @ ego_rot  # rotate to world frame

# OBS ############
obs_center = np.array([0.3, -1.7])
obs_angle = np.pi / 4  # in radians
obs_center = np.concatenate([obs_center, [obs_angle]])
obs_rot = np.array(
    [
        [np.cos(obs_angle), np.sin(obs_angle), 0],
        [-np.sin(obs_angle), np.cos(obs_angle), 0],
        [0, 0, 1],
    ]
)
obs_length = 5
obs_width = 2
obs_poly_corners = np.array(
    [
        [-obs_length / 2, -obs_width / 2, 0],
        [obs_length / 2, -obs_width / 2, 0],
        [obs_length / 2, obs_width / 2, 0],
        [-obs_length / 2, obs_width / 2, 0],
    ]
)
obs_poly_corners = obs_poly_corners @ obs_rot + obs_center
# obs covariance
obs_sigma_x = 0.3  # in obs direction
obs_sigma_y = 0.1  # in obs direction
obs_cov_xy = 0.0
obs_sigma_theta = 0.1  # in rad

obs_cov = np.array(
    [
        [obs_sigma_x**2, obs_cov_xy, 0],
        [obs_cov_xy, obs_sigma_y**2, 0],
        [0, 0, obs_sigma_theta**2],
    ]
)
obs_cov = obs_rot.T @ obs_cov @ obs_rot  # rotate to world frame


# COLLISION EGO - POINT WITHOUT ORIENTATION ############
rel_pos, rel_cov = convolve_distributions(
    ego_center, ego_cov, obs_center, obs_cov, dist="Gaussian"
)
# MC Sampling point-poly
coll_prob, visu_dict = cpo.point_collvol_mc_sampling(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_center,
    obs2_cov=obs_cov,
    num_samples=num_samples,
    with_orientation=False,
)

# num_samples, rel_pos, rel_cov,
# ego_poly_corners)
print(f"MC Sampling coll prob (Point-Rect): {coll_prob:.4f}")

# visualize_overlap_point_poly(ego_poly_corners, visu_dict["points"], visu_dict["overlap_mask"])

# Inc-Excl: Only valid for point-rectangle
coll_prob = cpo.point_rect_inc_excl(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_center,
    obs2_cov=obs_cov,
)

print(f"Inc-Excl coll prob (Point-Rect): {coll_prob:.4f}")

coll_prob = cpo.point_collvol_analytic(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_center,
    obs2_cov=obs_cov,
    with_orientation=False,
)
print(f"Analytic coll prob (Point-Rect): {coll_prob:.4f}")

# COLLISION VOLUME WITHOUT ORIENTATION ############
# MC Sampling poly-poly
coll_prob, visu_dict = cpo.poly_poly_mc_sampling(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_poly_corners,
    obs2_cov=obs_cov,
    num_samples=num_samples,
    with_orientation=False,
)

print(f"MC Sampling coll prob (Poly-Poly): {coll_prob:.4f}")
# visualize_overlap_poly_poly(visu_dict["ego_rects"], visu_dict["obs_rects"], visu_dict["overlap_mask"])
# MC Sampling poly-poly with ego fixed at origin
coll_prob, visu_dict = cpo.poly_poly_mc_sampling(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_poly_corners,
    obs2_cov=obs_cov,
    num_samples=num_samples,
    with_orientation=False,
)
print(f"MC Sampling coll prob (Poly-Poly ego fixed at origin): {coll_prob:.4f}")
# visualize_overlap_poly_poly(visu_dict["ego_rects"], visu_dict["obs_rects"], visu_dict["overlap_mask"])

# coll_vol_corners = collision_polygon(ego_poly_corners, obs_poly_corners)

# MC Sampling coll_vol-point
coll_prob, visu_dict_coll_vol = cpo.point_collvol_mc_sampling(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_poly_corners,
    obs2_cov=obs_cov,
    num_samples=num_samples,
    with_orientation=False,
)
print(f"MC Sampling coll prob (Point-CollVol): {coll_prob:.4f}")

# visualize_overlap_point_poly(coll_vol_corners, visu_dict_coll_vol["points"], visu_dict_coll_vol["overlap_mask"])
# data.append(["Poly MC Sampling", f"{coll_prob:.4f}", f"{t*1e3:.1f}"])

coll_prob = cpo.point_collvol_analytic(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_poly_corners,
    obs2_cov=obs_cov,
    with_orientation=False,
)
print(f"Analytic coll prob (Poly-Poly): {coll_prob:.4f}")
# print(tabulate(data, headers=headers, tablefmt="simple"))  # fancy_grid pipe grid

# COLLISION EGO - POINT WITH ORIENTATION ############
# MC Sampling point-poly
coll_prob, visu_dict = cpo.point_collvol_mc_sampling(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_center,
    obs2_cov=obs_cov,
    num_samples=num_samples,
    with_orientation=True,
)
print(f"MC Sampling coll prob (Point-Rect with orientation): {coll_prob:.4f}")

coll_prob = cpo.point_collvol_analytic(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_center,
    obs2_cov=obs_cov,
    with_orientation=True,
)
print(f"Analytic coll prob (Point-Rect with orientation): {coll_prob:.4f}")

# COLLISION VOLUME WITH ORIENTATION ############
# MC Sampling poly-poly with orientation
coll_prob, visu_dict = cpo.poly_poly_mc_sampling(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_poly_corners,
    obs2_cov=obs_cov,
    num_samples=num_samples,
    with_orientation=True,
)
print(f"MC Sampling coll prob (Poly-Poly with orientation): {coll_prob:.4f}")
# visualize_overlap_poly_poly(visu_dict["ego_rects"], visu_dict["obs_rects"], visu_dict["overlap_mask"])

# MC Sampling coll_vol-point with orientation
coll_prob, visu_dict_coll_vol = cpo.point_collvol_mc_sampling(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_poly_corners,
    obs2_cov=obs_cov,
    num_samples=num_samples,
    with_orientation=True,
)
print(f"MC Sampling coll prob (Point-CollVol Orient): {coll_prob:.4f}")

# Analytic poly-poly with orientation
coll_prob = cpo.point_collvol_analytic(
    obs1=ego_poly_corners,
    obs1_cov=ego_cov,
    obs2=obs_poly_corners,
    obs2_cov=obs_cov,
    with_orientation=True,
)
print(f"Analytic coll prob (Poly-Poly with orientation): {coll_prob:.4f}")
