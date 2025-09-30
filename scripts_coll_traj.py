import numpy as np
from scipy.stats import multivariate_normal
from utils import simple_prediction, visualize_traj_poly
import coll_trajectories as cpt
from params import SEED

# mc sampling
num_samples = 25000

# EGO ############
ego_center = [1, 1]
ego_angle = 0 * np.pi / 8
ego_v = np.array([1, 1])  # , [1,1]]
ego_initial_state = np.concatenate([ego_center, [ego_angle], ego_v])
ego_length = 5
ego_width = 2

ego_rot = np.array(
    [
        [np.cos(ego_angle), np.sin(ego_angle), 0],
        [-np.sin(ego_angle), np.cos(ego_angle), 0],
        [0, 0, 1],
    ]
)
ego_poly_corners = np.array(
    [
        [-ego_length / 2, -ego_width / 2, 0],
        [ego_length / 2, -ego_width / 2, 0],
        [ego_length / 2, ego_width / 2, 0],
        [-ego_length / 2, ego_width / 2, 0],
    ]
)

ego_poly_corners = ego_poly_corners @ ego_rot + ego_initial_state[:3]

# ego covariance
ego_sigma_x = 0.1  # in ego direction
ego_sigma_y = 0.1  # in ego direction
ego_sigma_theta = 1e-9  # in rad
ego_cov_xy = 0.0
ego_sigma_v = 1e-9  # sigma_vx, sigma_vy
ego_cov_vxy = 0.0  # 0.1-1e-9

ego_cov = np.array(
    [
        [ego_sigma_x**2, ego_cov_xy, 0, 0, 0],
        [ego_cov_xy, ego_sigma_y**2, 0, 0, 0],
        [0, 0, ego_sigma_theta**2, 0, 0],
        [0, 0, 0, ego_sigma_v**2, ego_cov_vxy],
        [0, 0, 0, ego_cov_vxy, ego_sigma_v**2],
    ]
)
# rotate to world frame
ego_cov[0:3, 0:3] = ego_rot.T @ ego_cov[0:3, 0:3] @ ego_rot  # rotate to world frame
ego_cov[3:5, 3:5] = (
    ego_rot[0:2, 0:2].T @ ego_cov[3:5, 3:5] @ ego_rot[0:2, 0:2]
)  # rotate to world frame

# OBS ############
obs_center = np.array([5.5, 5.5])  # , np.array([-1.5, 6])]
obs_angle = np.pi / 4  # , 3*np.pi / 8]
obs_v = np.array([-1, -1])  # , [-1.2426,-3]]
obs_initial_state = np.concatenate([obs_center, [obs_angle], obs_v])

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
obs_poly_corners = obs_poly_corners @ obs_rot + obs_initial_state[:3]

# obs covariance
obs_sigma_x = 0.3  # in obs direction
obs_sigma_y = 0.1  # in obs direction
obs_cov_xy = 0.0
obs_sigma_theta = 0.1  # in rad
obs_sigma_v = 0.1  # sigma_vx, sigma_vy
obs_cov_vxy = 0  # 0.1-1e-9

obs_cov = np.array(
    [
        [obs_sigma_x**2, obs_cov_xy, 0, 0, 0],
        [obs_cov_xy, obs_sigma_y**2, 0, 0, 0],
        [0, 0, obs_sigma_theta**2, 0, 0],
        [0, 0, 0, obs_sigma_v**2, obs_cov_vxy],
        [0, 0, 0, obs_cov_vxy, obs_sigma_v**2],
    ]
)
# rotate to world frame
obs_cov[0:3, 0:3] = obs_rot.T @ obs_cov[0:3, 0:3] @ obs_rot  # rotate to world frame
obs_cov[3:5, 3:5] = (
    obs_rot[0:2, 0:2].T @ obs_cov[3:5, 3:5] @ obs_rot[0:2, 0:2]
)  # rotate to world frame

# COLLISION VOLUME WITHOUT ORIENTATION ############
# MC Sampling traj collvol
# coll_prob, p_cum, _ = cpt.point_collvol_mc_traj_sampling(
#     initial_poly1_boundary=[ego_poly_corners],
#     initial_state_1=[ego_initial_state],
#     initial_cov_1=[ego_cov],
#     initial_state_2=[obs_initial_state, obs_initial_state],
#     initial_cov_2=[obs_cov, obs_cov],
#     initial_poly2_boundary=[obs_poly_corners, obs_poly_corners],
#     with_orientation=False,
#     num_samples=num_samples,
# )
# print(f"MC Traj coll prob (no orient): {coll_prob}")

# # MC Sampling point collvol
ego_traj, ego_cov1 = simple_prediction(ego_initial_state, ego_cov)
obs_traj, obs_cov1 = simple_prediction(obs_initial_state, obs_cov)
ego_traj2, ego_cov2 = simple_prediction(
    ego_initial_state - np.array([1, 1, 0, 1, 1]), ego_cov
)
obs_traj2, obs_cov2 = simple_prediction(
    obs_initial_state - np.array([7, 7, 0.2, -2, -2]), obs_cov
)

# # visualize_traj_poly(ego_poly_corners, obs_traj, obs_poly_corners)
# # visualize_traj_poly(ego_poly_corners, obs_traj2, obs_poly_corners-np.array([7,7,0]))
# # visualize_traj_poly(ego_poly_corners*2, obs_traj, obs_poly_corners)
# # visualize_traj_poly(ego_poly_corners*2, obs_traj2, obs_poly_corners- np.array([7,7,0]))
coll_prob, p_states = cpt.point_collvol_mc_state_sampling(
    initial_obs1=[ego_poly_corners, ego_poly_corners - np.array([1, 1, 0])],
    trajs_obs1=[ego_traj, ego_traj2],
    trajs_obs1_covs=[ego_cov1, ego_cov2],
    trajs_obs2=[obs_traj, obs_traj2],
    trajs_obs2_covs=[obs_cov1, obs_cov2],
    initial_obs2=[obs_poly_corners, obs_poly_corners - np.array([7, 7, 0.2])],
    with_orientation=False,
    num_samples=num_samples,
)
print(f"MC State coll prob (no orient): {coll_prob}")

coll_prob, _ = cpt.traj_collvol_analytic(
    initial_obs1=[ego_poly_corners, ego_poly_corners - np.array([1, 1, 0])],
    trajs_obs1=[ego_traj, ego_traj2],
    trajs_obs1_covs=[ego_cov1, ego_cov2],
    trajs_obs2=[obs_traj, obs_traj2],
    trajs_obs2_covs=[obs_cov1, obs_cov2],
    initial_obs2=[obs_poly_corners, obs_poly_corners - np.array([7, 7, 0.2])],
    with_orientation=False,
)
print(f"Analytic coll prob (no orient): {coll_prob}")

# Boundary Crossing
coll_prob = cpt.boundary_crossing(
    poly1_boundary=[ego_poly_corners, ego_poly_corners - np.array([1, 1, 0])],
    trajs1=[ego_traj, ego_traj2],
    trajs1_covs=[ego_cov1, ego_cov2],
    trajs2=[obs_traj, obs_traj2],
    trajs2_covs=[obs_cov1, obs_cov2],
    poly2_boundary=[obs_poly_corners, obs_poly_corners - np.array([7, 7, 0])],
    with_orientation=False,
)
print(f"Boundary Crossing coll prob (no orient): {coll_prob}")


# COLLISION VOLUME WITH ORIENTATION ############

# MC Sampling traj collvol
coll_prob, p_cum, _ = cpt.point_collvol_mc_traj_sampling(
    initial_poly1_boundary=[ego_poly_corners],
    initial_state_1=[ego_initial_state],
    initial_cov_1=[ego_cov],
    initial_state_2=[obs_initial_state, obs_initial_state],
    initial_cov_2=[obs_cov, obs_cov],
    initial_poly2_boundary=[obs_poly_corners, obs_poly_corners],
    with_orientation=True,
    num_samples=num_samples,
)
print(f"MC Traj coll prob (with orient): {coll_prob}")

# MC Sampling point collvol
coll_prob, _ = cpt.point_collvol_mc_state_sampling(
    initial_poly1_boundary=[
        ego_poly_corners,
        ego_poly_corners * 2 - np.array([1, 1, 0]),
    ],
    trajs1=[ego_traj, ego_traj2],
    trajs1_covs=[ego_cov1, ego_cov2],
    trajs2=[obs_traj, obs_traj2],
    trajs2_covs=[obs_cov1, obs_cov2],
    initial_poly2_boundary=[obs_poly_corners, obs_poly_corners - np.array([7, 7, 0])],
    with_orientation=True,
    num_samples=num_samples,
)
print(f"MC State coll prob (with orient): {coll_prob}")
