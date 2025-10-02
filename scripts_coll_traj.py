import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Ellipse
from utils import simple_prediction, visualize_traj_poly
import coll_trajectories as cpt
from params import SEED

# mc sampling
with_orientation = True
num_samples = 1000
# trajectory steps
horizon = 3.0
dt = 0.1
num_traj_steps = int(horizon / dt)

visu_data_with = None
visu_data_without = None
# EGO Static
ego_center = [0, 0]
ego_angle = 0
ego_v = np.array([0, 0])
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


ego_static_corners = np.array(
    [
        [-ego_length / 2, -ego_width / 2, 0],
        [ego_length / 2, -ego_width / 2, 0],
        [ego_length / 2, ego_width / 2, 0],
        [-ego_length / 2, ego_width / 2, 0],
    ]
)
ego_poly_corners = ego_static_corners @ ego_rot + ego_initial_state[:3]

# # ego covariance
ego_sigma_x = 0.1  # in ego direction
ego_sigma_y = 0.05  # in ego direction
ego_sigma_theta = 1e-9  # in rad
ego_cov_xy = 0.0
ego_sigma_vx = 1e-9  # in ego direction
ego_sigma_vy = 1e-9  # in ego direction
ego_cov_vxy = 0.0  # 0.1-1e-9

ego_initial_cov = np.array(
    [
        [ego_sigma_x**2, ego_cov_xy, 0, 0, 0],
        [ego_cov_xy, ego_sigma_y**2, 0, 0, 0],
        [0, 0, ego_sigma_theta**2, 0, 0],
        [0, 0, 0, ego_sigma_vx**2, ego_cov_vxy],
        [0, 0, 0, ego_cov_vxy, ego_sigma_vy**2],
    ]
)
# rotate to world frame
ego_initial_cov[0:3, 0:3] = (
    ego_rot.T @ ego_initial_cov[0:3, 0:3] @ ego_rot
)  # rotate to world frame
ego_initial_cov[3:5, 3:5] = (
    ego_rot[0:2, 0:2].T @ ego_initial_cov[3:5, 3:5] @ ego_rot[0:2, 0:2]
)  # rotate to world frame

# ego_traj_cov = ego_initial_cov[None, ...].repeat(num_traj_steps + 1, axis=0)
# ego_traj = ego_initial_state[None, ...].repeat(num_traj_steps + 1, axis=0)
# ego_traj, ego_traj_cov = simple_prediction(
#     ego_initial_state, ego_initial_cov, dt=dt, num_steps=num_traj_steps
# )


# OBS ############
obs_center = np.array([[5.5, 5.5], [-1.5, 6]])
obs_angle = [np.pi / 4, 3 * np.pi / 8]
obs_v = np.array([[-1, -1], [-1.2426, -3]])
obs_initial_states = np.vstack([obs_center.T, obs_angle, obs_v.T]).T

obs_rot = np.zeros((len(obs_center), 3, 3))
cos = np.cos(obs_angle)
sin = np.sin(obs_angle)
obs_rot[..., 0, 0] = cos
obs_rot[..., 0, 1] = sin
obs_rot[..., 1, 0] = -sin
obs_rot[..., 1, 1] = cos
obs_rot[..., 2, 2] = 1

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
obs_poly_corners = obs_poly_corners @ obs_rot + obs_initial_states[..., None, :3]

# obs covariance
obs_sigma_x = [0.2, 0.2]  # in obs direction
obs_sigma_y = [0.1, 0.1]  # in obs direction
obs_cov_xy = [0.0, 0.0]
obs_sigma_theta = [0.1, 0.1]  # in rad
obs_sigma_vx = [0.1, 0.1]  # in obs direction
obs_sigma_vy = [0.1, 0.1]  # in obs direction
obs_cov_vxy = [(0.1 - 1e-9) ** 2, (0.1 - 1e-9) ** 2]


obs_initial_covs = np.zeros((len(obs_center), 5, 5))
for i in range(len(obs_center)):
    obs_initial_covs[i] = np.array(
        [
            [obs_sigma_x[i] ** 2, obs_cov_xy[i], 0, 0, 0],
            [obs_cov_xy[i], obs_sigma_y[i] ** 2, 0, 0, 0],
            [0, 0, obs_sigma_theta[i] ** 2, 0, 0],
            [0, 0, 0, obs_sigma_vx[i] ** 2, obs_cov_vxy[i]],
            [0, 0, 0, obs_cov_vxy[i], obs_sigma_vy[i] ** 2],
        ]
    )
# rotate to world frame
obs_initial_covs[..., 0:3, 0:3] = (
    obs_rot.swapaxes(-1, -2) @ obs_initial_covs[..., 0:3, 0:3] @ obs_rot
)
obs_initial_covs[..., 3:5, 3:5] = (
    obs_rot[..., 0:2, 0:2].swapaxes(-1, -2)
    @ obs_initial_covs[..., 3:5, 3:5]
    @ obs_rot[..., 0:2, 0:2]
)

# obs_initial_covs[..., 3:5, 3:5] = 0
# obs_initial_covs[..., -2, -2] = 1e-8
# obs_initial_covs[..., -1, -1] = 0.02
# obs_initial_covs += ego_initial_cov


data = {"with": {}, "without": {}}
print("Ego initial state:", ego_initial_state)
print("Obs initial states:", obs_initial_states)
# COLLISION VOLUME WITHOUT ORIENTATION ############
ego_ini_cov = ego_initial_cov.copy()
ego_ini_cov[..., 2, 2] = 1e-8
ego_traj, ego_traj_cov = simple_prediction(
    ego_initial_state, ego_ini_cov, dt=dt, num_steps=num_traj_steps
)
obs_ini_cov = obs_initial_covs.copy()
obs_ini_cov[..., 2, 2] = 1e-8
obs_trajs, obs_covs = simple_prediction(
    obs_initial_states, obs_ini_cov, dt=dt, num_steps=num_traj_steps
)
# MC Sampling point collvol
coll_prob, p_states = cpt.point_collvol_mc_state_sampling(
    initial_obs1=ego_static_corners,
    trajs_obs1=ego_traj,
    trajs_obs1_covs=ego_traj_cov,
    trajs_obs2=obs_trajs,
    trajs_obs2_covs=obs_covs,
    initial_obs2=obs_poly_corners,
    with_orientation=False,
    num_samples=num_samples,
)
print(f"MC State coll prob (no orient): {coll_prob}")
data["without"]["MC State"] = p_states

# Analytic point collvol
coll_prob, p_states = cpt.traj_collvol_analytic(
    initial_obs1=ego_static_corners,
    trajs_obs1=ego_traj,
    trajs_obs1_covs=ego_traj_cov,
    trajs_obs2=obs_trajs,
    trajs_obs2_covs=obs_covs,
    initial_obs2=obs_poly_corners,
    with_orientation=False,
)
print(f"Analytic coll prob (no orient): {coll_prob}")
data["without"]["Analytic"] = p_states
data["without"]["Independent"] = 1 - np.cumprod(1 - p_states, axis=-1)

# MC Sampling traj collvol
coll_prob, p_cum, visu_data_without = cpt.point_collvol_mc_traj_sampling(
    initial_obs1=ego_static_corners,
    initial_state1=ego_initial_state,
    initial_cov1=ego_initial_cov,
    initial_state2=obs_initial_states,
    initial_cov2=obs_initial_covs,
    initial_obs2=obs_poly_corners,
    with_orientation=False,
    num_samples=num_samples,
    num_traj_steps=num_traj_steps,
    dt=dt,
)
print(f"MC Traj coll prob (no orient): {coll_prob}")
data["without"]["MC Traj"] = p_cum

# Boundary Crossing
coll_prob, p_cum = cpt.boundary_crossing(
    initial_obs1=ego_static_corners,
    trajs_obs1=ego_traj,
    trajs_obs1_covs=ego_traj_cov,
    trajs_obs2=obs_trajs,
    trajs_obs2_covs=obs_covs,
    initial_obs2=obs_poly_corners,
    with_orientation=False,
    dt=dt,
)
print(f"Boundary Crossing coll prob (no orient): {coll_prob}")
data["without"]["Boundary Crossing"] = p_cum

# COLLISION VOLUME WITH ORIENTATION ############
if with_orientation:
    ego_traj, ego_traj_cov = simple_prediction(
        ego_initial_state, ego_initial_cov, dt=dt, num_steps=num_traj_steps
    )
    obs_poly_corners = obs_poly_corners[1]
    obs_initial_states = obs_initial_states[1]
    obs_initial_covs = obs_initial_covs[1]
    obs_trajs, obs_covs = simple_prediction(
        obs_initial_states, obs_initial_covs, dt=dt, num_steps=num_traj_steps
    )

    # MC Sampling point collvol
    coll_prob, p_states = cpt.point_collvol_mc_state_sampling(
        initial_obs1=ego_static_corners,
        trajs_obs1=ego_traj,
        trajs_obs1_covs=ego_traj_cov,
        trajs_obs2=obs_trajs,
        trajs_obs2_covs=obs_covs,
        initial_obs2=obs_poly_corners,
        with_orientation=True,
        num_samples=num_samples,
    )
    print(f"MC State coll prob (with orient): {coll_prob}")
    data["with"]["MC State"] = p_states

    # Analytic point collvol
    coll_prob, p_states = cpt.traj_collvol_analytic(
        initial_obs1=ego_static_corners,
        trajs_obs1=ego_traj,
        trajs_obs1_covs=ego_traj_cov,
        trajs_obs2=obs_trajs,
        trajs_obs2_covs=obs_covs,
        initial_obs2=obs_poly_corners,
        with_orientation=True,
    )
    print(f"Analytic coll prob (with orient): {coll_prob}")
    data["with"]["Analytic"] = p_states
    data["with"]["Independent"] = 1 - np.cumprod(1 - p_states, axis=-1)

    # MC Sampling traj collvol
    coll_prob, p_cum, visu_data_with = cpt.point_collvol_mc_traj_sampling(
        initial_obs1=ego_static_corners,
        initial_state1=ego_initial_state,
        initial_cov1=ego_initial_cov,
        initial_state2=obs_initial_states,
        initial_cov2=obs_initial_covs,
        initial_obs2=obs_poly_corners,
        with_orientation=True,
        num_samples=num_samples,
        num_traj_steps=num_traj_steps,
        dt=dt,
    )
    print(f"MC Traj coll prob (with orient): {coll_prob}")
    data["with"]["MC Traj"] = p_cum

    # Boundary Crossing
    coll_prob, p_cum = cpt.boundary_crossing(
        initial_obs1=ego_static_corners,
        trajs_obs1=ego_traj,
        trajs_obs1_covs=ego_traj_cov,
        trajs_obs2=obs_trajs,
        trajs_obs2_covs=obs_covs,
        initial_obs2=obs_poly_corners,
        with_orientation=True,
        dt=dt,
    )
    print(f"Boundary Crossing coll prob (with orient): {coll_prob}")
    data["with"]["Boundary Crossing"] = p_cum


# PLOTTING ############


# for visu_data in [visu_data_without, visu_data_with]:
#     if visu_data is None:
#         continue
#     trajs, mask = visu_data.values()
#     ego_poly_corners = (
#         ego_poly_corners[None] if ego_poly_corners.ndim == 2 else ego_poly_corners
#     )
#     obs_poly_corners = (
#         obs_poly_corners[None] if obs_poly_corners.ndim == 2 else obs_poly_corners
#     )
#     for i in range(trajs.shape[1]):
#         for j in range(trajs.shape[2]):
#             visualize_traj_poly(
#                 ego_poly_corners[i], trajs[:, i, j], mask[i, j], obs_poly_corners[j]
#             )


# colors = mpl.color_sequences["tab20"]
# linestyle = ["-", "--", "-.", ":"]
# for orient, results in data.items():
#     if results:
#         plt.figure(figsize=(10, 6))
#         for i, (method, p) in enumerate(results.items()):
#             p = p.reshape(-1, p.shape[-1])
#             for j in range(p.shape[0]):
#                 plt.plot(
#                     np.linspace(0, horizon, len(p[j])),
#                     p[j],
#                     label=method,
#                     color=colors[i],
#                     linestyle=linestyle[j % len(linestyle)],
#                 )
#         plt.ylim(-0.05, 1.05)
#         plt.xlim(0, horizon)
#         plt.xlabel("Time [s]")
#         plt.ylabel("Collision Probability")
#         plt.title(f"Collision Probability over Time ({orient} orientation)")
#         plt.legend()
#         plt.grid()
#         plt.tight_layout()
# plt.show()
