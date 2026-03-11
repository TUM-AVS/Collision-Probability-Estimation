"""
Script to test collision probability estimation methods for two uncertain polygons.
"""

import argparse

import numpy as np
import time

import collision_probability_estimation.np_impl.coll_state_overlap as cpo

np.set_printoptions(precision=4, suppress=True)

def main():
    parser = argparse.ArgumentParser(description="Collision probability estimation")
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
        help="Plot the results using matplotlib",
    )
    args = parser.parse_args()
    use_torch = args.use_torch
    use_plot = args.plot
    if use_torch:
        try:
            import collision_probability_estimation.torch_impl.coll_state_overlap as cpo_torch
        except Exception as e:
                raise ModuleNotFoundError(
                    r"torch is required for --use_torch flag. "
                    r"Install it with: uv sync --extra torch or run it directly with uv run --extra torch state_coll_prob"
                ) from e
    if use_plot:
        try:
            import matplotlib as mpl
            mpl.use("TkAgg")  
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            from collision_probability_estimation.visualization import (
                visualize_overlap_point_poly,
                visualize_overlap_poly_poly,
            )
            
        except Exception as e:
            raise ModuleNotFoundError(
                r"matplotlib is required for --plot flag. "
                r"Install it with: uv sync --extra plot or run it directly with uv run --extra plot state_coll_prob"
            ) from e

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


    results = []  # list of (method, geometry, orientation, backend, value, duration_s)

    def add(method, geometry, orientation, backend, value, duration_s: float):
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            val = value.detach().cpu().numpy()
        else:
            val = value
        results.append((method, geometry, "yes" if orientation else "no", backend, val, duration_s))

    # COLLISION EGO - POINT WITHOUT ORIENTATION ############
    # MC Sampling point-poly
    t0 = time.perf_counter()
    coll_prob, visu_point_wo = cpo.point_collvol_mc_sampling(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
        num_samples=num_samples, with_orientation=False,
    )
    dt = time.perf_counter() - t0
    add("MC Sampling", "Point-Rect", False, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob, visu_point_torch_wo = cpo_torch.point_collvol_mc_sampling(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
            num_samples=num_samples, with_orientation=False,
        )
        dt = time.perf_counter() - t0
        add("MC Sampling", "Point-Rect", False, "torch", coll_prob, dt)

    

    # Inc-Excl: Only valid for point-rectangle
    t0 = time.perf_counter()
    coll_prob = cpo.point_rect_incl_excl(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
    )
    dt = time.perf_counter() - t0
    add("Inc-Excl", "Point-Rect", False, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob = cpo_torch.point_rect_incl_excl(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
        )
        dt = time.perf_counter() - t0
        add("Inc-Excl", "Point-Rect", False, "torch", coll_prob, dt)

    t0 = time.perf_counter()
    coll_prob = cpo.point_collvol_analytic(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
        with_orientation=False,
    )
    dt = time.perf_counter() - t0
    add("Analytic", "Point-Rect", False, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob = cpo_torch.point_collvol_analytic(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
            with_orientation=False,
        )
        dt = time.perf_counter() - t0
        add("Analytic", "Point-Rect", False, "torch", coll_prob, dt)

    # COLLISION VOLUME WITHOUT ORIENTATION ############
    t0 = time.perf_counter()
    coll_prob, visu_polypoly_wo = cpo.poly_poly_mc_sampling(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
        num_samples=num_samples, with_orientation=False,
    )
    dt = time.perf_counter() - t0
    add("MC Sampling", "Poly-Poly", False, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob, visu_polypoly_wo_torch = cpo_torch.poly_poly_mc_sampling(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
            num_samples=num_samples, with_orientation=False,
        )
        dt = time.perf_counter() - t0
        add("MC Sampling", "Poly-Poly", False, "torch", coll_prob, dt)

    t0 = time.perf_counter()
    coll_prob, visu_point_collvol_wo = cpo.point_collvol_mc_sampling(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
        num_samples=num_samples, with_orientation=False,
    )
    dt = time.perf_counter() - t0
    add("MC Sampling", "Point-CollVol", False, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob, visu_point_collvol_wo_torch = cpo_torch.point_collvol_mc_sampling(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
            num_samples=num_samples, with_orientation=False,
        )
        dt = time.perf_counter() - t0
        add("MC Sampling", "Point-CollVol", False, "torch", coll_prob, dt)

    t0 = time.perf_counter()
    coll_prob = cpo.point_collvol_analytic(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
        with_orientation=False,
    )
    dt = time.perf_counter() - t0
    add("Analytic", "Point-CollVol", False, "numpy", coll_prob, dt)

    if use_torch:
        t0 = time.perf_counter()
        coll_prob = cpo_torch.point_collvol_analytic(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
            with_orientation=False,
        )
        dt = time.perf_counter() - t0
        add("Analytic", "Point-CollVol", False, "torch", coll_prob, dt)

    # COLLISION EGO - POINT WITH ORIENTATION ############
    t0 = time.perf_counter()
    coll_prob, visu_point_with = cpo.point_collvol_mc_sampling(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
        num_samples=num_samples, with_orientation=True,
    )
    dt = time.perf_counter() - t0
    add("MC Sampling", "Point-Rect", True, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob, visu_point_with_torch = cpo_torch.point_collvol_mc_sampling(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
            num_samples=num_samples, with_orientation=True,
        )
        dt = time.perf_counter() - t0
        add("MC Sampling", "Point-Rect", True, "torch", coll_prob, dt)

    t0 = time.perf_counter()
    coll_prob = cpo.point_collvol_analytic(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
        with_orientation=True,
    )
    dt = time.perf_counter() - t0
    add("Analytic", "Point-Rect", True, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob = cpo_torch.point_collvol_analytic(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_center, obs2_cov=obs_cov,
            with_orientation=True,
        )
        dt = time.perf_counter() - t0
        add("Analytic", "Point-Rect", True, "torch", coll_prob, dt)

    # COLLISION VOLUME WITH ORIENTATION ############
    t0 = time.perf_counter()
    coll_prob, visu_polypoly_with = cpo.poly_poly_mc_sampling(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
        num_samples=num_samples, with_orientation=True,
    )
    dt = time.perf_counter() - t0
    add("MC Sampling", "Poly-Poly", True, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob, visu_polypoly_with_torch = cpo_torch.poly_poly_mc_sampling(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
            num_samples=num_samples, with_orientation=True,
        )
        dt = time.perf_counter() - t0
        add("MC Sampling", "Poly-Poly", True, "torch", coll_prob, dt)

    t0 = time.perf_counter()
    coll_prob, visu_point_collvol_with = cpo.point_collvol_mc_sampling(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
        num_samples=num_samples, with_orientation=True,
    )
    dt = time.perf_counter() - t0
    add("MC Sampling", "Point-CollVol", True, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob, visu_point_collvol_with_torch = cpo_torch.point_collvol_mc_sampling(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
            num_samples=num_samples, with_orientation=True,
        )
        dt = time.perf_counter() - t0
        add("MC Sampling", "Point-CollVol", True, "torch", coll_prob, dt)

    t0 = time.perf_counter()
    coll_prob = cpo.point_collvol_analytic(
        obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
        with_orientation=True,
    )
    dt = time.perf_counter() - t0
    add("Analytic", "Point-CollVol", True, "numpy", coll_prob, dt)
    if use_torch:
        t0 = time.perf_counter()
        coll_prob = cpo_torch.point_collvol_analytic(
            obs1=ego_poly_corners, obs1_cov=ego_cov, obs2=obs_poly_corners, obs2_cov=obs_cov,
            with_orientation=True,
        )
        dt = time.perf_counter() - t0
        add("Analytic", "Point-CollVol", True, "torch", coll_prob, dt)

    # Print results table
    col_w = [16, 16, 11, 8, 12, 10]
    headers = ["Method", "Geometry", "Orientation", "Backend", "Coll. Prob.", "Time (s)"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    row_fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(sep)
    print(row_fmt.format(*headers))
    print(sep.replace("-", "="))
    prev_group = None
    for method, geometry, orientation, backend, value, duration in results:
        group = (method, geometry, orientation)
        if prev_group is not None and group != prev_group:
            print(sep)
        print(row_fmt.format(method, geometry, orientation, backend, f"{value}", f"{duration:.4f}"))
        prev_group = group
    print(sep)

    if use_plot:
        visualize_overlap_point_poly(visu_point_wo["collvol"], visu_point_wo["points"], visu_point_wo["overlap_mask"], "(Point-Rect) without orientation")
        visualize_overlap_poly_poly(visu_polypoly_wo["poly1"], visu_polypoly_wo["poly2"], visu_polypoly_wo["overlap_mask"], "(Poly-Poly) without orientation")
        visualize_overlap_point_poly(visu_point_collvol_wo["collvol"], visu_point_collvol_wo["points"], visu_point_collvol_wo["overlap_mask"], "(Point-CollVol) without orientation")
        visualize_overlap_point_poly(visu_point_with["collvol"], visu_point_with["points"], visu_point_with["overlap_mask"], "(Point-Rect) with orientation")
        visualize_overlap_poly_poly(visu_polypoly_with["poly1"], visu_polypoly_with["poly2"], visu_polypoly_with["overlap_mask"], "(Poly-Poly) with orientation")
        visualize_overlap_point_poly(visu_point_collvol_with["collvol"], visu_point_collvol_with["points"], visu_point_collvol_with["overlap_mask"], "(Point-CollVol) with orientation")
        plt.show(block=True)


if __name__ == "__main__":
    main()