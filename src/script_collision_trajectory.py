"""
Script to compare different collision probability estimation methods over a trajectory.
"""

import argparse
import time

import numpy as np
from collision_probability_estimation.np_impl.utils import simple_prediction
import collision_probability_estimation.np_impl.coll_trajectories as cpt

np.set_printoptions(precision=4, suppress=True)


def main():
    parser = argparse.ArgumentParser(description="Collision probability estimation over trajectories")
    parser.add_argument(
        "--use_torch",
        action="store_true",
        default=False,
        help="Include torch-based methods in the comparison",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Plot the collision probability over time using matplotlib",
    )
    # parser.add_argument(
    #     "--with_orientation",
    #     action="store_true",
    #     default=False,
    #     help="Also run methods that include orientation uncertainty",
    # )
    args = parser.parse_args()
    use_torch = args.use_torch
    use_plot = args.plot
    with_orientation = True

    if use_torch:
        try:
            import collision_probability_estimation.torch_impl.coll_trajectories as cpt_torch
        except Exception as e:
            raise ModuleNotFoundError(
                "torch is required for --use_torch flag. "
                "Install it with: uv sync --extra torch"
            ) from e

    if use_plot:
        try:
            import matplotlib as mpl
            mpl.use("TkAgg")
            import matplotlib.pyplot as plt
            from collision_probability_estimation.visualization import visualize_traj_poly
        except Exception as e:
            raise ModuleNotFoundError(
                "matplotlib is required for --plot flag. "
                "Install it with: uv sync --extra plot"
            ) from e

    #  Scenario setup 
    num_samples = 5000
    horizon = 3.0
    dt = 0.1
    num_traj_steps = int(horizon / dt)

    # EGO 
    ego_center = [0, 0]
    ego_angle = 0
    ego_v = [0]
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

    ego_sigma_x = 0.1
    ego_sigma_y = 0.05
    ego_sigma_theta = 0.1
    ego_cov_xy = 0.0
    ego_sigma_v = 0.1
    ego_initial_cov = np.array(
        [
            [ego_sigma_x**2, ego_cov_xy, 0, 0],
            [ego_cov_xy, ego_sigma_y**2, 0, 0],
            [0, 0, ego_sigma_theta**2, 0],
            [0, 0, 0, ego_sigma_v**2],
        ]
    )
    ego_initial_cov[0:3, 0:3] = ego_rot.T @ ego_initial_cov[0:3, 0:3] @ ego_rot

    # OBS 
    obs_center = np.array([[5.5, 5.5], [-1.5, 6]])
    obs_angle = [np.pi / 4, 3 * np.pi / 8]
    obs_v = [-np.sqrt(2), -3 / np.sin(3 * np.pi / 8)]
    obs_initial_states = np.vstack([obs_center.T, obs_angle, obs_v]).T

    obs_rot = np.zeros((len(obs_center), 3, 3))
    cos_a = np.cos(obs_angle)
    sin_a = np.sin(obs_angle)
    obs_rot[..., 0, 0] = cos_a
    obs_rot[..., 0, 1] = sin_a
    obs_rot[..., 1, 0] = -sin_a
    obs_rot[..., 1, 1] = cos_a
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

    obs_sigma_x = [0.2, 0.2]
    obs_sigma_y = [0.1, 0.1]
    obs_cov_xy = [0.0, 0.0]
    obs_sigma_theta = [0.1, 0.1]
    obs_sigma_v = [0.1, 0.1]
    obs_initial_covs = np.zeros((len(obs_center), 4, 4))
    for i in range(len(obs_center)):
        obs_initial_covs[i] = np.array(
            [
                [obs_sigma_x[i] ** 2, obs_cov_xy[i], 0, 0],
                [obs_cov_xy[i], obs_sigma_y[i] ** 2, 0, 0],
                [0, 0, obs_sigma_theta[i] ** 2, 0],
                [0, 0, 0, obs_sigma_v[i] ** 2],
            ]
        )
    obs_initial_covs[..., 0:3, 0:3] = (
        obs_rot.swapaxes(-1, -2) @ obs_initial_covs[..., 0:3, 0:3] @ obs_rot
    )

    print("Ego initial state:", ego_initial_state)
    print("Obs initial states:", obs_initial_states)

    # Results collection 
    results = []  # (method, orientation, backend, coll_prob, duration_s)
    plot_data = {"without": {}, "with": {}}   # time-series for plotting
    visu_data = {}                             # trajectory samples for visualisation

    def add(method, orientation, backend, coll_prob, duration_s: float, p_series=None):
        val = coll_prob
        if hasattr(val, "detach") and hasattr(val, "cpu"):
            val = val.detach().cpu().numpy()
        # if hasattr(val, "item"):
            # val = val.item()
        results.append((method, "yes" if orientation else "no", backend, val, float(duration_s)))
        if p_series is not None:
            key = "with" if orientation else "without"
            if hasattr(p_series, "detach"):
                p_series = p_series.detach().cpu().numpy()
            plot_data[key][f"{method} ({backend})"] = p_series

    #  WITHOUT ORIENTATION ########
    ego_ini_cov_wo = ego_initial_cov.copy()
    ego_ini_cov_wo[..., 2, 2] = 1e-8
    obs_ini_cov_wo = obs_initial_covs.copy()
    obs_ini_cov_wo[..., 2, 2] = 1e-8

    ego_traj_wo, ego_traj_cov_wo = simple_prediction(
        ego_initial_state, ego_ini_cov_wo, dt=dt, num_steps=num_traj_steps
    )
    obs_trajs_wo, obs_covs_wo = simple_prediction(
        obs_initial_states, obs_ini_cov_wo, dt=dt, num_steps=num_traj_steps
    )

    # MC State sampling
    t0 = time.perf_counter()
    coll_prob, p_states = cpt.point_collvol_mc_state_sampling(
        corners_obs1=ego_poly_corners,
        trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
        trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
        corners_obs2=obs_poly_corners,
        with_orientation=False, num_samples=num_samples,
    )
    add("MC State", False, "numpy", coll_prob, time.perf_counter() - t0, p_states)

    if use_torch:
        t0 = time.perf_counter()
        coll_prob, p_states = cpt_torch.point_collvol_mc_state_sampling(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
            trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
            corners_obs2=obs_poly_corners,
            with_orientation=False, num_samples=num_samples,
        )
        add("MC State", False, "torch", coll_prob, time.perf_counter() - t0, p_states)

    # Analytic
    t0 = time.perf_counter()
    coll_prob, p_states = cpt.traj_collvol_analytic(
        corners_obs1=ego_poly_corners,
        trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
        trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
        corners_obs2=obs_poly_corners, with_orientation=False,
    )
    add("Analytic", False, "numpy", coll_prob, time.perf_counter() - t0, p_states)
    plot_data["without"]["Independent (numpy)"] = 1 - np.cumprod(1 - p_states, axis=-1)

    if use_torch:
        t0 = time.perf_counter()
        coll_prob, p_states = cpt_torch.traj_collvol_analytic(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
            trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
            corners_obs2=obs_poly_corners, with_orientation=False,
        )
        if hasattr(p_states, "detach"):
            p_states = p_states.detach().cpu().numpy()
        add("Analytic", False, "torch", coll_prob, time.perf_counter() - t0, p_states)
        plot_data["without"]["Independent (torch)"] = 1 - np.cumprod(1 - p_states, axis=-1)

    # MC Traj sampling
    t0 = time.perf_counter()
    coll_prob, p_cum, visu_wo = cpt.point_collvol_mc_traj_sampling(
        corners_obs1=ego_poly_corners,
        initial_state1=ego_initial_state, initial_cov1=ego_ini_cov_wo,
        initial_state2=obs_initial_states, initial_cov2=obs_ini_cov_wo,
        corners_obs2=obs_poly_corners,
        with_orientation=False, num_samples=num_samples,
        num_traj_steps=num_traj_steps, dt=dt,
    )
    add("MC Traj", False, "numpy", coll_prob, time.perf_counter() - t0, p_cum)
    visu_data["without_numpy"] = visu_wo

    if use_torch:
        t0 = time.perf_counter()
        coll_prob, p_cum, visu_wo_torch = cpt_torch.point_collvol_mc_traj_sampling(
            corners_obs1=ego_poly_corners,
            initial_state1=ego_initial_state, initial_cov1=ego_ini_cov_wo,
            initial_state2=obs_initial_states, initial_cov2=obs_ini_cov_wo,
            corners_obs2=obs_poly_corners,
            with_orientation=False, num_samples=num_samples,
            num_traj_steps=num_traj_steps, dt=dt,
        )
        add("MC Traj", False, "torch", coll_prob, time.perf_counter() - t0, p_cum)
        visu_data["without_torch"] = visu_wo_torch

    # Boundary Crossing
    t0 = time.perf_counter()
    coll_prob, cep = cpt.boundary_crossing(
        corners_obs1=ego_poly_corners,
        trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
        trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
        corners_obs2=obs_poly_corners, with_orientation=False, dt=dt,
    )
    add("Boundary Crossing", False, "numpy", coll_prob, time.perf_counter() - t0, cep)

    if use_torch:
        t0 = time.perf_counter()
        coll_prob, cep = cpt_torch.boundary_crossing(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
            trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
            corners_obs2=obs_poly_corners, with_orientation=False, dt=dt,
        )
        add("Boundary Crossing", False, "torch", coll_prob, time.perf_counter() - t0, cep)

    # Incl-Excl
    t0 = time.perf_counter()
    coll_prob, p_states = cpt.traj_rect_incl_excl(
        corners_obs1=ego_poly_corners,
        trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
        trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
        corners_obs2=obs_poly_corners, with_orientation=False, dt=dt,
    )
    add("Incl-Excl", False, "numpy", coll_prob, time.perf_counter() - t0, p_states)

    if use_torch:
        t0 = time.perf_counter()
        coll_prob, p_states = cpt_torch.traj_rect_incl_excl(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
            trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
            corners_obs2=obs_poly_corners, with_orientation=False, dt=dt,
        )
        add("Incl-Excl", False, "torch", coll_prob, time.perf_counter() - t0, p_states)

    # Mahalanobis Distance
    t0 = time.perf_counter()
    coll_prob, p_states = cpt.traj_mahalanobis_distance(
        corners_obs1=ego_poly_corners,
        trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
        trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
        corners_obs2=obs_poly_corners, with_orientation=False, dt=dt,
    )
    add("Mahalanobis", False, "numpy", coll_prob, time.perf_counter() - t0, p_states)

    if use_torch:
        t0 = time.perf_counter()
        coll_prob, p_states = cpt_torch.traj_mahalanobis_distance(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
            trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
            corners_obs2=obs_poly_corners, with_orientation=False, dt=dt,
        )
        add("Mahalanobis", False, "torch", coll_prob, time.perf_counter() - t0, p_states)

    # Corner Mahalanobis Distance
    t0 = time.perf_counter()
    coll_prob, p_states = cpt.traj_corner_mahalanobis_distance(
        corners_obs1=ego_poly_corners,
        trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
        trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
        corners_obs2=obs_poly_corners, with_orientation=False, dt=dt,
    )
    add("Corner Mahalanobis", False, "numpy", coll_prob, time.perf_counter() - t0, p_states)

    if use_torch:
        t0 = time.perf_counter()
        coll_prob, p_states = cpt_torch.traj_corner_mahalanobis_distance(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_wo, trajs_obs1_covs=ego_traj_cov_wo,
            trajs_obs2=obs_trajs_wo, trajs_obs2_covs=obs_covs_wo,
            corners_obs2=obs_poly_corners, with_orientation=False, dt=dt,
        )
        add("Corner Mahalanobis", False, "torch", coll_prob, time.perf_counter() - t0, p_states)


    # ── WITH ORIENTATION #########
    if with_orientation:
        ego_traj_w, ego_traj_cov_w = simple_prediction(
            ego_initial_state, ego_initial_cov, dt=dt, num_steps=num_traj_steps
        )
        obs_trajs_w, obs_covs_w = simple_prediction(
            obs_initial_states, obs_initial_covs, dt=dt, num_steps=num_traj_steps
        )

        # MC State sampling
        t0 = time.perf_counter()
        coll_prob, p_states = cpt.point_collvol_mc_state_sampling(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
            trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
            corners_obs2=obs_poly_corners,
            with_orientation=True, num_samples=num_samples,
        )
        add("MC State", True, "numpy", coll_prob, time.perf_counter() - t0, p_states)

        if use_torch:
            t0 = time.perf_counter()
            coll_prob, p_states = cpt_torch.point_collvol_mc_state_sampling(
                corners_obs1=ego_poly_corners,
                trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
                trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
                corners_obs2=obs_poly_corners,
                with_orientation=True, num_samples=num_samples,
            )
            add("MC State", True, "torch", coll_prob, time.perf_counter() - t0, p_states)

        # Analytic
        t0 = time.perf_counter()
        coll_prob, p_states = cpt.traj_collvol_analytic(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
            trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
            corners_obs2=obs_poly_corners, with_orientation=True,
        )
        add("Analytic", True, "numpy", coll_prob, time.perf_counter() - t0, p_states)
        plot_data["with"]["Independent (numpy)"] = 1 - np.cumprod(1 - p_states, axis=-1)

        if use_torch:
            t0 = time.perf_counter()
            coll_prob, p_states = cpt_torch.traj_collvol_analytic(
                corners_obs1=ego_poly_corners,
                trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
                trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
                corners_obs2=obs_poly_corners, with_orientation=True,
            )
            if hasattr(p_states, "detach"):
                p_states = p_states.detach().cpu().numpy()
            add("Analytic", True, "torch", coll_prob, time.perf_counter() - t0, p_states)
            plot_data["with"]["Independent (torch)"] = 1 - np.cumprod(1 - p_states, axis=-1)

        # MC Traj sampling
        t0 = time.perf_counter()
        coll_prob, p_cum, visu_w = cpt.point_collvol_mc_traj_sampling(
            corners_obs1=ego_poly_corners,
            initial_state1=ego_initial_state, initial_cov1=ego_initial_cov,
            initial_state2=obs_initial_states, initial_cov2=obs_initial_covs,
            corners_obs2=obs_poly_corners,
            with_orientation=True, num_samples=num_samples,
            num_traj_steps=num_traj_steps, dt=dt,
        )
        add("MC Traj", True, "numpy", coll_prob, time.perf_counter() - t0, p_cum)
        visu_data["with_numpy"] = visu_w

        if use_torch:
            t0 = time.perf_counter()
            coll_prob, p_cum, visu_w_torch = cpt_torch.point_collvol_mc_traj_sampling(
                corners_obs1=ego_poly_corners,
                initial_state1=ego_initial_state, initial_cov1=ego_initial_cov,
                initial_state2=obs_initial_states, initial_cov2=obs_initial_covs,
                corners_obs2=obs_poly_corners,
                with_orientation=True, num_samples=num_samples,
                num_traj_steps=num_traj_steps, dt=dt,
            )
            add("MC Traj", True, "torch", coll_prob, time.perf_counter() - t0, p_cum)
            visu_data["with_torch"] = visu_w_torch

        # Boundary Crossing
        t0 = time.perf_counter()
        coll_prob, p_cum = cpt.boundary_crossing(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
            trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
            corners_obs2=obs_poly_corners, with_orientation=True, dt=dt,
        )
        add("Boundary Crossing", True, "numpy", coll_prob, time.perf_counter() - t0, p_cum)

        if use_torch:
            t0 = time.perf_counter()
            coll_prob, p_cum = cpt_torch.boundary_crossing(
                corners_obs1=ego_poly_corners,
                trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
                trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
                corners_obs2=obs_poly_corners, with_orientation=True, dt=dt,
            )
            add("Boundary Crossing", True, "torch", coll_prob, time.perf_counter() - t0, p_cum)

        # Mahalanobis Distance
        t0 = time.perf_counter()
        coll_prob, p_states = cpt.traj_mahalanobis_distance(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
            trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
            corners_obs2=obs_poly_corners, with_orientation=True, dt=dt,
        )
        add("Mahalanobis", True, "numpy", coll_prob, time.perf_counter() - t0, p_states)

        if use_torch:
            t0 = time.perf_counter()
            coll_prob, p_states = cpt_torch.traj_mahalanobis_distance(
                corners_obs1=ego_poly_corners,
                trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
                trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
                corners_obs2=obs_poly_corners, with_orientation=True, dt=dt,
            )
            add("Mahalanobis", True, "torch", coll_prob, time.perf_counter() - t0, p_states)

        # Corner Mahalanobis Distance
        t0 = time.perf_counter()
        coll_prob, p_states = cpt.traj_corner_mahalanobis_distance(
            corners_obs1=ego_poly_corners,
            trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
            trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
            corners_obs2=obs_poly_corners, with_orientation=True, dt=dt,
        )
        add("Corner Mahalanobis", True, "numpy", coll_prob, time.perf_counter() - t0, p_states)

        if use_torch:
            t0 = time.perf_counter()
            coll_prob, p_states = cpt_torch.traj_corner_mahalanobis_distance(
                corners_obs1=ego_poly_corners,
                trajs_obs1=ego_traj_w, trajs_obs1_covs=ego_traj_cov_w,
                trajs_obs2=obs_trajs_w, trajs_obs2_covs=obs_covs_w,
                corners_obs2=obs_poly_corners, with_orientation=True, dt=dt,
            )
            add("Corner Mahalanobis", True, "torch", coll_prob, time.perf_counter() - t0, p_states)



    #  Print results table 
    col_w = [20, 11, 8, 18, 10]
    headers = ["Method", "Orientation", "Backend", "Coll. Prob.", "Time (s)"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    row_fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(sep)
    print(row_fmt.format(*headers))
    print(sep.replace("-", "="))
    prev_group = None
    for method, orientation, backend, value, duration in results:
        group = (method, orientation)
        if prev_group is not None and group != prev_group:
            print(sep)
        print(row_fmt.format(method, orientation, backend, f"{value}", f"{duration:.4f}"))
        prev_group = group
    print(sep)

    # Optional plotting 
    if use_plot:
        colors = mpl.color_sequences["tab20"]
        linestyle = ["-", "--", "-.", ":"]

        for orient_key, results_dict in plot_data.items():
            if not results_dict:
                continue
            plt.figure(figsize=(10, 6))
            for i, (method, p) in enumerate(results_dict.items()):
                if "Boundary Crossing" in method:
                    p = np.clip(np.cumsum(p, axis=-1) * dt, None, 1.0)
                p = p.reshape(-1, p.shape[-1])
                for j in range(p.shape[0]):
                    plt.plot(
                        np.linspace(0, horizon, p.shape[-1]),
                        p[j],
                        label=method if j == 0 else "_nolegend_",
                        color=colors[i % len(colors)],
                        linestyle=linestyle[j % len(linestyle)],
                    )
            plt.ylim(-0.05, 1.05)
            plt.xlim(0, horizon)
            plt.xlabel("Time [s]")
            plt.ylabel("Collision Probability")
            plt.title(f"Collision Probability over Time ({orient_key} orientation)")
            plt.legend()
            plt.grid()
            plt.tight_layout()

        for visu_key, visu in visu_data.items():
            trajs, mask = visu["trajs"], visu["overlap_mask"]
            if hasattr(trajs, "cpu"):
                trajs = trajs.cpu().numpy()
                mask = mask.cpu().numpy()
            ego_corners = ego_poly_corners[None] if ego_poly_corners.ndim == 2 else ego_poly_corners
            obs_corners = obs_poly_corners[None] if obs_poly_corners.ndim == 2 else obs_poly_corners
            for i in range(trajs.shape[1]):
                for j in range(trajs.shape[2]):
                    visualize_traj_poly(ego_corners[i], trajs[:, i, j], mask[i, j], obs_corners[j])

        plt.show(block=True)


if __name__ == "__main__":
    main()
