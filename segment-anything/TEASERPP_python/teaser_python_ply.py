# coding:utf-8
#
# pypcd_python_ply.py.py
#  Created on: 2023/5/16
#      Author: Tex Yan Liu
# description: 可以直接运行，测试后直接Pass，效果太差了

import copy
import time
import numpy as np
import open3d as o3d
import teaserpp_python
import matplotlib.pyplot as plt

NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10


def drawing(src_1, src_2, index):
    # print("s_x", s_x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # print(s_x)
    # ax.plot_surface(s_x, s_z, s_y, rstride = 1, cstride = 1, cmap=cm.RdYlGn, linewidth=0, alpha=0.4, antialiased=False)
    ax.scatter(src_1[0], src_1[1], src_1[2], s=0.01, c="r", marker="x", label="source")
    ax.scatter(src_2[0], src_2[1], src_2[2], s=0.01, c="b", marker="x", label="target")

    ax.set_xlabel('X Label')
    # ax.set_xlim(40, 160)
    ax.set_ylabel('Z Label')
    # ax.set_ylim(0, 1000)
    ax.set_zlabel('Y Label')
    # ax.set_zlim(150, -25)
    # ax.set_box_aspect()
    plt.savefig("rebuild k={}.svg".format(index), format="svg")  # 耗费时间19秒
    plt.show()
    return 1

def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));


if __name__ == "__main__":
    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    # Load bunny ply file
    src_cloud = o3d.io.read_point_cloud("src.ply")
    target_cloud = o3d.io.read_point_cloud("target.ply")
    src_cloud = src_cloud.uniform_down_sample(1)    # 每()个点采样一次
    target_cloud = target_cloud.uniform_down_sample(2)  # 每()个点采样一次
    src = np.transpose(np.asarray(src_cloud.points))
    target = np.transpose(np.asarray(target_cloud.points))  # target的点数不能和src差太多，否则就会崩溃
    print("src", src.shape)
    print("target", target.shape)
    N = src.shape[1]

    # # Apply arbitrary scale, translation and rotation
    # T = np.array(
    #     [[9.96926560e-01, 6.68735757e-02, -4.06664421e-02, -1.15576939e-01],
    #      [-6.61289946e-02, 9.97617877e-01, 1.94008687e-02, -3.87705398e-02],
    #      [4.18675510e-02, -1.66517807e-02, 9.98977765e-01, 1.14874890e-01],
    #      [0, 0, 0, 1]])
    #
    # dst_cloud = copy.deepcopy(src_cloud)
    # dst_cloud.transform(T)
    # dst = np.transpose(np.asarray(dst_cloud.points))
    #
    # # Add some noise
    # dst += (np.random.rand(3, N) - 0.5) * 2 * NOISE_BOUND
    #
    # # Add some outliers
    # outlier_indices = np.random.randint(N_OUTLIERS, size=N_OUTLIERS)
    # for i in range(outlier_indices.size):
    #     shift = OUTLIER_TRANSLATION_LB + np.random.rand(3, 1) * (OUTLIER_TRANSLATION_UB - OUTLIER_TRANSLATION_LB)
    #     dst[:, outlier_indices[i]] += shift.squeeze()

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    # print("dst", dst.shape)
    solver.solve(src, target)
    end = time.time()

    solution = solver.getSolution()

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")

    print("Estimated scale: ")
    print(solution.scale)

    # print("Expected rotation: ")
    # print(T[:3, :3])
    print("Estimated rotation: ")
    print(solution.rotation)
    # print("Error (rad): ")
    # print(get_angular_error(T[:3,:3], solution.rotation))

    # print("Expected translation: ")
    # print(T[:3, 3])
    print("Estimated translation: ")
    print(solution.translation)
    # print("Error (m): ")
    # print(np.linalg.norm(T[:3, 3] - solution.translation))

    print("Number of correspondences: ", N)
    print("Number of outliers: ", N_OUTLIERS)
    print("Time taken (s): ", end - start)

    src = np.array(solution.scale) * np.matmul(np.array(solution.rotation), src) + np.array(solution.translation).reshape(3,1)
    print(src.shape)
    print(target.shape)
    drawing(src, target, 2)  # 检验一下变换后的形态

