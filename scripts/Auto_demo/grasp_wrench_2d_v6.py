# coding:utf-8
#
# grasp_wrench_2d_v6.py
#
#  Created on: 2024/04/15
#      Author: Tex Yan Liu
#
# description: 是grasp_wrench_2d 第五代的修改，用于改进凹面物体，但是还没有改透

import numpy as np
import argparse
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

parser = argparse.ArgumentParser()

# config = {
#             "font.family": 'serif',
#             "font.size": 12,# 相当于小四大小
#             "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
#             "font.serif": ['SimSun'],#宋体
#             'axes.unicode_minus': False # 处理负号，即-号
#          }
# rcParams.update(config)


def plot_3d_picture(convex_hull_vertex, min_distance, num=0):  # 这个是画抓取质量图的

    fig3 = plt.figure(num=3)
    ax3 = fig3.add_subplot(111, projection="3d")

    theta = np.linspace(0, 2 * np.pi, 100)  # 用参数方程画图
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)  # 生成网格点坐标矩阵
    # print("theta", theta)
    s_x = min_distance * np.cos(theta) * np.cos(phi)
    s_y = min_distance * np.cos(theta) * np.sin(phi)
    s_z = min_distance * np.sin(theta)
    ax3.plot_surface(s_x, s_y, s_z, rstride=1, cstride=1, cmap=cm.rainbow, linewidth=0, alpha=0.7, antialiased=False)

    ax3.scatter(convex_hull_vertex.T[0], convex_hull_vertex.T[1], convex_hull_vertex.T[2])
    collection = Poly3DCollection(convex_hull_vertex, cmap=cm.Spectral, alpha=0.2)
    face_color = [0.1, 0.1, 0.1]
    collection.set_facecolor(face_color)
    collection.set_edgecolor("w")
    ax3.add_collection3d(collection)
    ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # red, green, blue, alpha透明度 [0到1不是到255] 将轴交面背景改为白色
    ax3.set_xlabel("f_x")
    ax3.set_xlim()
    ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3.set_ylabel("f_y")
    ax3.set_ylim()
    ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3.set_zlabel("tau")
    ax3.set_zlim()
    # ax3.set_box_aspect((2, 0.2, 80))
    if num == 9:
        plt.savefig('figs/best_wrench_quality.pdf')
    elif num == 1:
        plt.savefig('figs/normal_wrench_quality_{}.pdf'.format(num))
        plt.close(fig3)


def plot_quality_picture(circle_slice, straight_slice, quality_q):  # 绘制质量分布图

    fig4 = plt.figure(num=4)
    ax4 = fig4.add_subplot(111, projection="3d")
    circle_index = np.linspace(0, np.pi, circle_slice)  # 圆序号
    straight_index = np.linspace(0, straight_slice, straight_slice + 1)  # 线序号
    circle_index_1, straight_index_1 = np.meshgrid(circle_index, straight_index)  # 生成网格点坐标矩阵
    quality_q_1 = quality_q.reshape((201, 400))
    # ax.plot_surface(circle_index_1, straight_index_1, quality_q_1, rstride=1, cstride=1, cmap=cm.Spectral, linewidth=0, alpha=0.7, antialiased=False)
    ax4.scatter3D(circle_index_1, straight_index_1, quality_q_1, c=quality_q_1, marker=".", linewidths=0.5)
    ax4.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # red, green, blue, alpha透明度 [0到1不是到255] 将轴交面背景改为白色
    ax4.set_xlabel("circle_index")
    ax4.set_xlim()
    ax4.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax4.set_ylabel("straight_index")
    ax4.set_ylim()
    ax4.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax4.set_zlabel("quality_q")
    ax4.set_zlim()
    plt.savefig('figs/quality_q.pdf')
    # plt.show()


def calculate_qij_in_Wij(w_1, w_2, w_3, w_4, property=0):  # property 0 是普通的，不用出图， 1是随意挑选的，需要出图，9是最好的，需要出图

    convex_hull_vertex = np.array([[w_1, w_2, w_3], [w_1, w_2, w_4], [w_1, w_3, w_4], [w_2, w_3, w_4]])
    convex_hull_plane = np.zeros((4, 4))   # 初始化凸壳表面
    origin_point = np.array([0, 0, 0, 1])
    convex_hull_min_distance = 0
    for l in range(len(convex_hull_vertex)):
        plane_normal = np.cross(convex_hull_vertex[l][1] - convex_hull_vertex[l][0], convex_hull_vertex[l][2] - convex_hull_vertex[l][0])
        d = - (plane_normal[0] * convex_hull_vertex[l][0][0]) - (plane_normal[1] * convex_hull_vertex[l][0][1]) - (plane_normal[2] * convex_hull_vertex[l][0][2])
        if d < 0:
            convex_hull_plane[l] = -np.append(plane_normal, d)  # 构造凸壳表面
        else:
            convex_hull_plane[l] = np.append(plane_normal, d)  # 构造凸壳表面
        # print("convex_hull_plane", convex_hull_plane)

    for m in range(len(convex_hull_plane)):
        distance_origin = np.dot(convex_hull_plane[m], origin_point)/np.linalg.norm(convex_hull_plane[m][0:3])
        # print("distance_origin", distance_origin)
        if m == 0:
            convex_hull_min_distance = distance_origin
        elif distance_origin < convex_hull_min_distance:
            convex_hull_min_distance = distance_origin

    # if property == 1 or property == 9:
    #     plot_3d_picture(convex_hull_vertex, convex_hull_min_distance, property)

    return convex_hull_min_distance


# object_2d_contour_P：获取到的边界点集 [n,2]  object_2d_center_c：获取到的中心点[n]
def grasp_wrench_2d(object_2d_contour_P, object_2d_center_c):
    # 由于像素边线具有锯齿性，所以点筛选为一半消除锯齿
    object_2d_contour_P = np.array([p for p_i, p in enumerate(object_2d_contour_P) if p_i % 4 == 0])
    circle_lambda = 0.005
    line_sigma = 0.005
    gripper_delta_max = 100
    gripper_delta_min = 0
    miu = 0.1  # 预定义摩擦系数
    circle_lambda_num = int(1 / circle_lambda)
    print("旋转切片为：", circle_lambda_num, "次")
    line_sigma_num = int(1 / line_sigma)
    print("垂直切片为：", line_sigma_num, "次")
    quality_Q = []
    wrench_space_W = []
    grasp_points_S = []
    normal_set = []
    F_1 = [1, 0]  # 左手指的力
    F_2 = [-1, 0]  # 右手指的力
    for i in tqdm(range(circle_lambda_num)):
        # print("i", i)
        # 将点集的质心平移至原点y轴方向对齐，旋转，生成新的二维点集，逆时针旋转
        P_i = np.matmul((object_2d_contour_P - object_2d_center_c), np.array(
            [[np.cos(np.pi * i / circle_lambda_num), np.sin(np.pi * i / circle_lambda_num)],
             [-np.sin(np.pi * i / circle_lambda_num), np.cos(np.pi * i / circle_lambda_num)]]))
        tck, myu = splprep(P_i.T, s=10)  # 求解B样条曲线
        u, v = splev(myu, tck, der=1)  # 获得每个点的切线方向 u是x方向 v是y方向 法线方向（-v, u） 切线方向（u, v）
        max_y_value = np.max(P_i, axis=0)[1]   # 总高度，用于计算切片个数
        min_y_value = np.min(P_i, axis=0)[1]
        unit_slice_y = (max_y_value - min_y_value) * line_sigma

        # fig = plt.figure()
        # plt.scatter(P_i.T[0], P_i.T[1])
        # plt.quiver(P_i.T[0], P_i.T[1], v, -u, color='r')  # v, -u是法线方向
        # plt.quiver(P_i.T[0], P_i.T[1], u, v, color='g')  # u, v是切线方向
        # plt.show()

        for j in range(line_sigma_num + 1):
            threshold_y = j * unit_slice_y  # 计算当前切片的y值
            # print("min_y_value", min_y_value)
            # print("max_y_value", max_y_value)
            # print("min_y_value + threshold_y", min_y_value + threshold_y)
            min_approach_zero_y = np.absolute(P_i.T[1] - (min_y_value + threshold_y))  # 所有点的y值减去当前的y值，距离0最近的值就是对趾的坐标值
            min_approach_zero_y_index = np.argsort(min_approach_zero_y)[0:20]  # 对大小值进行升序排序后，选择其索引
            # print("min_approach_zero_y_index", min_approach_zero_y_index)
            '''
            for k in range(20):
                print("P_i_min", P_i[min_approach_zero_y_index[k]])
            '''
            p_i_j_1 = []
            p_i_j_2 = []
            min_approach_left_y_index = 0
            min_approach_right_y_index = 0
            # 对这20个点进行筛选，选出左右的两个点
            for k in range(len(min_approach_zero_y_index)):
                # 判断最小的两个序号是否挨着的方式不太好，依旧容易出现误差，补充一个选择法向量相反的点，但是当出现正方形边缘的大平线时候，20个点也不足以产生对趾点
                if np.absolute(min_approach_zero_y_index[0] - min_approach_zero_y_index[k+1]) > 1\
                        and v[min_approach_zero_y_index[0]] * v[min_approach_zero_y_index[k+1]] < 0:
                    if P_i[min_approach_zero_y_index[0]][0] < P_i[min_approach_zero_y_index[k + 1]][0]:
                        p_i_j_1 = P_i[min_approach_zero_y_index[0]]  # 左侧接触点
                        p_i_j_2 = P_i[min_approach_zero_y_index[k + 1]]  # 右侧接触点
                        min_approach_left_y_index = min_approach_zero_y_index[0]
                        min_approach_right_y_index = min_approach_zero_y_index[k + 1]
                    else:
                        p_i_j_1 = P_i[min_approach_zero_y_index[k + 1]]  # 左侧接触点
                        p_i_j_2 = P_i[min_approach_zero_y_index[0]]  # 右侧接触点
                        min_approach_left_y_index = min_approach_zero_y_index[k + 1]
                        min_approach_right_y_index = min_approach_zero_y_index[0]
                    break
                else:
                    p_i_j_1 = 10000
                    p_i_j_2 = 0
                    break
            # 选完
            if gripper_delta_min < np.linalg.norm(p_i_j_1 - p_i_j_2) < gripper_delta_max:  # 当两个点的毫米距离小于70毫米时候才能进行计算，如果大于就超出夹爪范围了
                n_1_i_j = np.array([v[min_approach_left_y_index], -u[min_approach_left_y_index]])  # 当前左侧点的法向量
                normalized_n_1_i_j = n_1_i_j / np.linalg.norm(n_1_i_j, 2)   # 左侧单位法向量
                t_1_i_j = np.array([u[min_approach_left_y_index], v[min_approach_left_y_index]])  # 当前左侧点的切向量
                normalized_t_1_i_j = t_1_i_j / np.linalg.norm(t_1_i_j, 2)   # 左侧单位切向量
                n_2_i_j = np.array([v[min_approach_right_y_index], -u[min_approach_right_y_index]])  # 当前右侧点的法向量
                normalized_n_2_i_j = n_2_i_j / np.linalg.norm(n_2_i_j, 2)   # 右侧单位法向量
                t_2_i_j = np.array([u[min_approach_right_y_index], v[min_approach_right_y_index]])  # 当前右侧点的切向量
                normalized_t_2_i_j = t_2_i_j / np.linalg.norm(t_2_i_j, 2)   # 右侧单位切向量 如果超90度投影会自动为负值用于调整方向
                if np.dot(F_1, normalized_n_1_i_j) > 0 and np.dot(F_2, normalized_n_2_i_j) > 0:
                    n_1_F_i_j = np.dot(F_1, normalized_n_1_i_j) * normalized_n_1_i_j   # 左侧法向力
                    t_1_F_i_j = np.dot(F_1, normalized_t_1_i_j) * normalized_t_1_i_j   # 左侧切向力
                    n_2_F_i_j = np.dot(F_2, normalized_n_2_i_j) * normalized_n_2_i_j   # 右侧法向力
                    t_2_F_i_j = np.dot(F_2, normalized_t_2_i_j) * normalized_t_2_i_j   # 右侧切向力
                    f_1_i_j = miu * np.linalg.norm(n_1_F_i_j, 2) * normalized_t_1_i_j   # 左侧最大摩擦力
                    f_2_i_j = miu * np.linalg.norm(n_2_F_i_j, 2) * normalized_t_2_i_j   # 右侧最大摩擦力
                    F_left_1_i_j = n_1_F_i_j + t_1_F_i_j + f_1_i_j   # 左侧点合力1
                    F_left_2_i_j = n_1_F_i_j + t_1_F_i_j - f_1_i_j  # 左侧点合力2
                    F_right_1_i_j = n_2_F_i_j + t_2_F_i_j + f_2_i_j  # 右侧点合力1
                    F_right_2_i_j = n_2_F_i_j + t_2_F_i_j - f_2_i_j   # 右侧点合力2
                    tau_left_1_i_j = np.cross(p_i_j_1, F_left_1_i_j)   # 求旋转力矩1.1
                    tau_left_2_i_j = np.cross(p_i_j_1, F_left_2_i_j)  # 求旋转力矩1.1
                    tau_right_1_i_j = np.cross(p_i_j_2, F_right_1_i_j)  # 求旋转力矩2.1
                    tau_right_2_i_j = np.cross(p_i_j_2, F_right_2_i_j)  # 求旋转力矩2.2
                    w_left_1_i_j = np.array([F_left_1_i_j[0], F_left_1_i_j[1], tau_left_1_i_j])  # 计算力旋量空间1.1
                    w_left_2_i_j = np.array([F_left_2_i_j[0], F_left_2_i_j[1], tau_left_2_i_j])  # 计算力旋量空间1.2
                    w_right_1_i_j = np.array([F_right_1_i_j[0], F_right_1_i_j[1], tau_right_1_i_j])  # 计算力旋量空间2.1
                    w_right_2_i_j = np.array([F_right_2_i_j[0], F_right_2_i_j[1], tau_right_2_i_j])  # 计算力旋量空间2.2
                    if i == 0 and j == 50:
                        q_i_j = calculate_qij_in_Wij(w_left_1_i_j, w_left_2_i_j, w_right_1_i_j, w_right_2_i_j, 1)
                    # Calculate q_i_j in W_i_j
                    q_i_j = calculate_qij_in_Wij(w_left_1_i_j, w_left_2_i_j, w_right_1_i_j, w_right_2_i_j)
                    # 对一些指标进行存储，首先将一些指标拉平在进行存储
                    # 将扳手空间w的left_1, left_2, right_1, right_2拉平后放在一个里边
                    quality_Q = np.append(quality_Q, q_i_j)   # 第i*j个质量参数，别的空间数组按照这个空间的0，1，2，3....排序
                    wrench_space_W = np.append(wrench_space_W, np.concatenate([w_left_1_i_j, w_left_2_i_j, w_right_1_i_j, w_right_2_i_j]))  # 第 i*j个扳手空间，每个空间有12个，所以0->0, 12->1
                    grasp_points_S = np.append(grasp_points_S, np.concatenate([[i], [j], p_i_j_1, p_i_j_2]))  # 第 i*j个对趾抓取点，每个空间6个，所以0->0, 6->

                # # 扳手空间的展示示例图
                # if i == 0 and j == 50:
                #     fig2 = plt.figure(num=2)
                #
                #     plt.plot(P_i.T[0], -P_i.T[1])
                #     plt.scatter(0, 0, c="r", marker="o")
                #     # 获得当前的axis
                #     ax2 = plt.gca()
                #     # 设置图像的上边、右边axis为无色
                #     ax2.spines['right'].set_color('none')
                #     ax2.spines['top'].set_color('none')
                #     # 设置x轴坐标在下部
                #     ax2.xaxis.set_ticks_position('bottom')
                #     # 设置x轴位于图像y=0处
                #     ax2.spines['bottom'].set_position(('data', 0))
                #     # 设置y轴坐标在左部
                #     ax2.yaxis.set_ticks_position('left')
                #     # 设置y轴位于图像x=0处
                #     ax2.spines['left'].set_position(('data', 0))
                #     # plt.scatter(-30, 40, c="black", marker="o", linewidths=12)
                #     # plt.scatter(30, 40, c="black", marker="o", linewidths=12)
                #     plt.quiver(p_i_j_1[0], -p_i_j_1[1], F_left_1_i_j[0], -F_left_1_i_j[1], scale=7, width=0.003, color='pink')  # v, -u是法线方向
                #     plt.quiver(p_i_j_1[0], -p_i_j_1[1], F_left_2_i_j[0], -F_left_2_i_j[1], scale=7, width=0.003, color='pink')  # u, v是切线方向
                #     plt.quiver(p_i_j_2[0], -p_i_j_2[1], F_right_1_i_j[0], -F_right_1_i_j[1], scale=7, width=0.003, color='pink')  # v, -u是法线方向
                #     plt.quiver(p_i_j_2[0], -p_i_j_2[1], F_right_2_i_j[0], -F_right_2_i_j[1], scale=7, width=0.003, color='pink')  # u, v是切线方向
                #
                #     plt.savefig('figs/wrench_space.pdf')
                #     plt.show()
            else:
                quality_Q = np.append(quality_Q, np.zeros(1))
                wrench_space_W = np.append(wrench_space_W, np.zeros(12))  # 第 i*j个扳手空间，每个空间有12个，所以0->0, 12->1
                grasp_points_S = np.append(grasp_points_S, np.zeros(6))   # 第 i*j个对趾抓取点，每个空间6个，所以0->0, 6->1

    print("抓取质量的结果", quality_Q)
    print("抓取质量的数量:", len(quality_Q))
    print("扳手空间的数量:", len(wrench_space_W))
    print("抓取对趾点的数量:", len(grasp_points_S))  # 写在论文里面引出遍历的数量
    # plot_quality_picture(circle_lambda_num, line_sigma_num, quality_Q)  # 绘制二维质量图

    # 寻找最大的抓取质量 返回抓取质量，扳手空间，抓取点
    q_max = np.max(quality_Q)   # 找到最大的质量值
    q_max_index = np.argmax(quality_Q)  # 找到质量值最大的索引
    W_max = wrench_space_W[12*q_max_index: 12*q_max_index+12].reshape((4, 3))  # 找到相关索引的扳手空间
    calculate_qij_in_Wij(W_max[0], W_max[1], W_max[2], W_max[3], 9)
    S_rotate_index = grasp_points_S[6*q_max_index]  # 找到旋转角度的索引
    S_max = grasp_points_S[6*q_max_index + 2: 6*q_max_index + 6].reshape((2, 2))  # 找到两个接触点
    print("变换前的最优夹持接触点", S_max)
    S_optimal = (np.matmul(S_max, np.array(
        [[np.cos(np.pi * S_rotate_index / circle_lambda_num), -np.sin(np.pi * S_rotate_index / circle_lambda_num)],
         [np.sin(np.pi * S_rotate_index / circle_lambda_num), np.cos(np.pi * S_rotate_index / circle_lambda_num)]]))
                 + object_2d_center_c)
    Force_direction = np.matmul(np.array([F_1, F_2]), np.array(
        [[np.cos(np.pi * S_rotate_index / circle_lambda_num), -np.sin(np.pi * S_rotate_index / circle_lambda_num)],
         [np.sin(np.pi * S_rotate_index / circle_lambda_num), np.cos(np.pi * S_rotate_index / circle_lambda_num)]]))

    print("q_max", q_max)
    print("q_max_index", q_max_index)
    print("W_max", W_max)
    print("S_rotate_index", S_rotate_index)
    print("最优夹持接触点", S_optimal)
    print("夹持方向", Force_direction)

    # fig = plt.figure(num=5)
    # # plt.scatter(object_2d_contour_P.T[0], object_2d_contour_P.T[1])
    # plt.scatter(S_optimal.T[0], -S_optimal.T[1], s=20, c="r")
    # plt.quiver(S_optimal.T[0], -S_optimal.T[1], Force_direction.T[0], -Force_direction.T[1], scale=7, width=0.006, color='r', pivot='tip')
    # plt.plot(object_2d_contour_P.T[0], -object_2d_contour_P.T[1])
    # plt.grid(True, linestyle="--", alpha=0.5)
    # # 获得当前的axis
    # ax5 = plt.gca()
    # # 设置图像的上边、右边axis为无色
    # ax5.spines['right'].set_color('none')
    # ax5.spines['bottom'].set_color('none')
    # # 设置x轴坐标在下部
    # ax5.xaxis.set_ticks_position('top')
    # # plt.quiver(object_2d_contour_P.T[0], object_2d_contour_P.T[1], v, -u, color='r')  # -v, u是法线方向
    # # plt.quiver(object_2d_contour_P.T[0], object_2d_contour_P.T[1], u, v, color='g')  # u, v是切线方向
    # plt.savefig('figs/grasp_force.pdf')
    # plt.show()

    # print()
    return q_max, W_max, S_optimal, Force_direction


if __name__ == '__main__':
    # 输入物品小鲨鱼玩具的边界点和边界中心
    shark_toy_boundary = np.array([[762, 357], [761, 358], [757, 358], [756, 359], [755, 359], [753, 361], [752, 361], [749, 364], [748, 364], [745, 367], [745, 368], [743, 370], [742, 370], [741, 371], [741, 372], [740, 373], [740, 375], [738, 377], [738, 378], [736, 380], [736, 381], [735, 382], [735, 383], [734, 384], [734, 385], [733, 386], [733, 387], [732, 388], [732, 389], [731, 390], [731, 392], [730, 393], [730, 395], [729, 396], [729, 399], [728, 400], [728, 402], [727, 403], [727, 405], [726, 406], [726, 410], [725, 411], [725, 422], [724, 423], [724, 432], [723, 433], [723, 437], [722, 438], [722, 440], [719, 443], [718, 443], [716, 445], [716, 446], [713, 449], [713, 450], [711, 452], [711, 453], [710, 454], [710, 456], [709, 457], [709, 460], [708, 461], [708, 467], [709, 468], [709, 469], [710, 470], [710, 471], [711, 471], [713, 473], [715, 473], [716, 474], [721, 474], [722, 475], [724, 475], [725, 474], [726, 474], [729, 477], [729, 479], [730, 480], [730, 482], [731, 483], [731, 485], [732, 486], [732, 487], [734, 489], [734, 490], [736, 492], [736, 493], [737, 494], [737, 495], [739, 497], [739, 498], [740, 499], [740, 500], [741, 501], [741, 503], [742, 504], [742, 505], [743, 506], [743, 507], [745, 509], [745, 510], [746, 511], [746, 512], [747, 513], [747, 514], [748, 515], [748, 516], [749, 517], [749, 518], [750, 519], [750, 521], [751, 522], [751, 524], [752, 525], [752, 527], [753, 528], [753, 531], [754, 532], [754, 538], [755, 539], [755, 543], [756, 544], [756, 547], [757, 548], [757, 549], [759, 551], [759, 552], [760, 553], [760, 554], [761, 555], [761, 556], [763, 558], [764, 558], [765, 559], [766, 559], [767, 560], [768, 560], [769, 561], [772, 561], [773, 562], [786, 562], [787, 561], [789, 561], [790, 560], [792, 560], [793, 559], [794, 559], [796, 557], [797, 557], [802, 552], [802, 551], [803, 550], [803, 548], [804, 547], [804, 531], [803, 530], [803, 510], [804, 509], [804, 505], [805, 504], [805, 502], [806, 501], [806, 498], [807, 497], [807, 495], [808, 494], [808, 492], [809, 491], [809, 489], [810, 488], [810, 484], [811, 483], [811, 478], [812, 477], [812, 475], [813, 474], [813, 470], [814, 469], [814, 468], [816, 466], [818, 466], [819, 465], [822, 465], [823, 466], [830, 466], [832, 464], [832, 461], [833, 460], [833, 456], [832, 455], [832, 452], [831, 451], [831, 450], [830, 449], [830, 447], [828, 445], [828, 444], [821, 437], [820, 437], [818, 435], [818, 434], [817, 433], [817, 432], [816, 431], [816, 429], [815, 428], [815, 425], [814, 424], [814, 422], [813, 421], [813, 416], [812, 415], [812, 413], [811, 412], [811, 409], [810, 408], [810, 405], [809, 404], [809, 403], [808, 402], [808, 401], [807, 400], [807, 399], [806, 398], [806, 397], [805, 396], [805, 394], [804, 393], [804, 391], [803, 390], [803, 388], [802, 387], [802, 386], [795, 379], [795, 378], [791, 374], [791, 373], [786, 368], [786, 367], [783, 364], [782, 364], [781, 363], [780, 363], [778, 361], [777, 361], [776, 360], [775, 360], [774, 359], [773, 359], [772, 358], [769, 358], [768, 357]])
    shark_toy_centroid = np.array([770, 455])
    usb_box_boundary = np.array([[860, 391], [859, 392], [855, 392], [854, 393], [851, 393], [850, 394], [845, 394], [844, 395], [837, 395], [836, 396], [832, 396], [831, 397], [826, 397], [825, 398], [821, 398], [819, 400], [817, 400], [816, 401], [805, 401], [804, 402], [791, 402], [790, 403], [783, 403], [782, 404], [781, 404], [778, 407], [778, 409], [777, 410], [777, 413], [778, 414], [778, 417], [779, 418], [779, 420], [780, 421], [780, 424], [781, 425], [781, 429], [782, 430], [782, 433], [783, 434], [783, 435], [784, 436], [784, 437], [785, 438], [785, 440], [786, 441], [786, 444], [787, 445], [787, 447], [788, 448], [788, 449], [789, 450], [789, 452], [790, 453], [790, 456], [791, 457], [791, 461], [792, 462], [792, 464], [793, 465], [793, 466], [794, 467], [794, 470], [795, 471], [795, 474], [796, 475], [796, 478], [797, 479], [797, 480], [798, 481], [798, 482], [799, 483], [799, 484], [800, 485], [800, 487], [801, 488], [801, 490], [802, 491], [802, 492], [803, 493], [803, 495], [804, 496], [804, 499], [805, 500], [805, 502], [806, 503], [806, 505], [807, 506], [807, 509], [808, 510], [808, 514], [809, 515], [809, 519], [810, 520], [810, 523], [811, 524], [811, 527], [812, 528], [812, 529], [813, 530], [813, 532], [817, 536], [819, 536], [820, 535], [827, 535], [828, 534], [835, 534], [836, 533], [848, 533], [849, 532], [854, 532], [855, 531], [856, 531], [857, 530], [861, 530], [862, 529], [866, 529], [867, 528], [874, 528], [875, 527], [879, 527], [880, 526], [882, 526], [883, 525], [886, 525], [887, 524], [892, 524], [893, 523], [900, 523], [901, 522], [905, 522], [906, 521], [910, 521], [911, 520], [913, 520], [914, 519], [915, 519], [917, 517], [917, 516], [918, 515], [918, 508], [917, 507], [917, 506], [916, 505], [916, 504], [915, 503], [915, 502], [914, 501], [914, 499], [913, 498], [913, 496], [912, 495], [912, 493], [911, 492], [911, 491], [910, 490], [910, 489], [909, 488], [909, 486], [908, 485], [908, 483], [907, 482], [907, 481], [906, 480], [906, 479], [905, 478], [905, 476], [904, 475], [904, 473], [903, 472], [903, 470], [902, 469], [902, 467], [901, 466], [901, 465], [900, 464], [900, 463], [899, 462], [899, 461], [898, 460], [898, 459], [897, 458], [897, 456], [896, 455], [896, 454], [895, 453], [895, 452], [894, 451], [894, 449], [893, 448], [893, 447], [892, 446], [892, 445], [891, 444], [891, 443], [890, 442], [890, 439], [889, 438], [889, 434], [888, 433], [888, 431], [886, 429], [886, 428], [885, 427], [885, 426], [884, 425], [884, 423], [883, 422], [883, 421], [882, 420], [882, 419], [879, 416], [879, 412], [878, 411], [878, 407], [877, 406], [877, 404], [876, 403], [876, 402], [874, 400], [874, 396], [873, 395], [873, 394], [872, 393], [872, 392], [871, 392], [870, 391]])
    usb_box_centroid = np.array([845, 464])
    cub_boundary = np.array([[853, 434], [852, 435], [851, 435], [848, 438], [848, 439], [846, 441], [846, 442], [845, 443], [845, 445], [844, 446], [844, 447], [843, 448], [843, 450], [842, 451], [842, 453], [841, 454], [841, 456], [840, 457], [840, 459], [839, 460], [839, 461], [838, 462], [838, 463], [837, 464], [837, 466], [836, 467], [836, 469], [835, 470], [835, 472], [834, 473], [834, 475], [833, 476], [833, 477], [832, 478], [832, 480], [831, 481], [831, 483], [830, 484], [830, 486], [829, 487], [829, 490], [828, 491], [828, 494], [827, 495], [827, 496], [826, 497], [826, 500], [825, 501], [825, 504], [824, 505], [824, 508], [823, 509], [823, 511], [822, 512], [822, 515], [821, 516], [821, 519], [820, 520], [820, 524], [819, 525], [819, 530], [818, 531], [818, 536], [817, 537], [817, 539], [818, 540], [818, 548], [813, 553], [812, 553], [810, 555], [810, 556], [809, 557], [809, 560], [808, 561], [808, 564], [807, 565], [807, 576], [808, 577], [808, 578], [809, 579], [809, 580], [810, 581], [810, 582], [811, 583], [811, 584], [815, 588], [816, 588], [818, 590], [819, 590], [822, 593], [823, 593], [825, 595], [826, 595], [827, 596], [828, 596], [829, 597], [830, 597], [831, 598], [832, 598], [833, 599], [834, 599], [835, 600], [836, 600], [837, 601], [838, 601], [841, 604], [842, 604], [844, 606], [845, 606], [847, 608], [848, 608], [849, 609], [850, 609], [851, 610], [852, 610], [853, 611], [854, 611], [856, 613], [857, 613], [860, 616], [861, 616], [862, 617], [863, 617], [864, 618], [865, 618], [866, 619], [867, 619], [871, 623], [872, 623], [873, 624], [874, 624], [875, 625], [877, 625], [878, 626], [885, 626], [886, 627], [888, 627], [889, 626], [893, 626], [894, 625], [895, 625], [898, 622], [899, 622], [901, 620], [902, 620], [904, 618], [905, 618], [907, 616], [908, 616], [914, 610], [915, 610], [916, 609], [917, 609], [918, 608], [920, 608], [921, 607], [933, 607], [933, 606], [932, 605], [932, 604], [931, 603], [931, 602], [928, 599], [928, 593], [929, 592], [929, 590], [930, 589], [930, 586], [931, 585], [931, 583], [932, 582], [932, 581], [933, 580], [945, 580], [946, 579], [948, 579], [954, 573], [954, 572], [956, 570], [956, 569], [957, 568], [957, 567], [960, 564], [960, 563], [961, 562], [961, 561], [962, 560], [962, 559], [963, 558], [963, 557], [964, 556], [964, 554], [965, 553], [965, 552], [966, 551], [966, 549], [967, 548], [967, 547], [968, 546], [968, 545], [969, 544], [969, 542], [970, 541], [970, 540], [971, 539], [971, 537], [972, 536], [972, 532], [973, 531], [973, 529], [974, 528], [974, 526], [975, 525], [975, 522], [976, 521], [976, 517], [977, 516], [977, 514], [978, 513], [978, 510], [979, 509], [979, 507], [980, 506], [980, 499], [981, 498], [981, 488], [980, 487], [980, 484], [979, 483], [979, 481], [974, 476], [973, 476], [972, 475], [971, 475], [970, 474], [968, 474], [967, 473], [965, 473], [964, 472], [963, 472], [962, 471], [961, 471], [960, 470], [958, 470], [957, 469], [954, 469], [953, 468], [951, 468], [950, 467], [948, 467], [947, 466], [946, 466], [945, 465], [943, 465], [942, 464], [940, 464], [939, 463], [938, 463], [937, 462], [935, 462], [934, 461], [933, 461], [932, 460], [930, 460], [929, 459], [928, 459], [927, 458], [925, 458], [923, 456], [920, 456], [919, 455], [916, 455], [915, 454], [913, 454], [912, 453], [910, 453], [909, 452], [908, 452], [907, 451], [905, 451], [904, 450], [901, 450], [900, 449], [898, 449], [897, 448], [895, 448], [894, 447], [893, 447], [892, 446], [890, 446], [889, 445], [888, 445], [887, 444], [884, 444], [883, 443], [881, 443], [880, 442], [878, 442], [877, 441], [876, 441], [875, 440], [874, 440], [873, 439], [870, 439], [869, 438], [868, 438], [867, 437], [865, 437], [864, 436], [863, 436], [862, 435], [858, 435], [857, 434]])
    cub_centroid = np.array([891, 529])
    cup_boundary = np.array([[409, 181], [408, 182], [401, 182], [400, 183], [398, 183], [397, 184], [395, 184], [394, 185], [392, 185], [391, 186], [390, 186], [388, 188], [386, 188], [385, 189], [384, 189], [383, 190], [382, 190], [381, 191], [380, 191], [379, 192], [378, 192], [377, 193], [376, 193], [375, 194], [374, 194], [373, 195], [372, 195], [371, 196], [370, 196], [368, 198], [367, 198], [366, 199], [365, 199], [363, 201], [361, 201], [360, 202], [359, 202], [358, 203], [356, 203], [355, 204], [354, 204], [352, 206], [351, 206], [345, 212], [342, 212], [341, 213], [339, 213], [338, 214], [337, 214], [336, 215], [335, 215], [335, 216], [333, 218], [333, 219], [332, 220], [332, 223], [331, 224], [331, 230], [332, 231], [332, 237], [333, 238], [333, 240], [334, 241], [334, 245], [335, 246], [335, 252], [336, 253], [336, 255], [337, 256], [337, 258], [338, 259], [338, 261], [339, 262], [339, 263], [341, 265], [342, 265], [344, 267], [345, 267], [346, 268], [350, 268], [351, 269], [355, 269], [356, 270], [359, 270], [363, 274], [364, 274], [366, 276], [369, 273], [369, 272], [370, 271], [371, 271], [372, 270], [375, 270], [376, 271], [377, 271], [378, 272], [379, 272], [380, 273], [389, 273], [390, 272], [392, 272], [393, 271], [394, 271], [395, 270], [397, 270], [398, 269], [400, 269], [401, 268], [402, 268], [403, 267], [404, 267], [405, 266], [406, 266], [407, 265], [408, 265], [409, 264], [410, 264], [411, 263], [412, 263], [413, 262], [414, 262], [415, 261], [416, 261], [417, 260], [418, 260], [419, 259], [420, 259], [421, 258], [423, 258], [424, 257], [425, 257], [426, 256], [427, 256], [428, 255], [429, 255], [430, 254], [431, 254], [432, 253], [433, 253], [436, 250], [437, 250], [437, 245], [436, 244], [436, 242], [435, 241], [435, 239], [434, 238], [434, 237], [433, 236], [433, 234], [432, 233], [432, 232], [431, 231], [431, 230], [430, 229], [430, 227], [429, 226], [429, 224], [428, 223], [428, 222], [427, 221], [427, 220], [426, 219], [426, 218], [425, 217], [425, 215], [424, 214], [424, 212], [423, 211], [423, 210], [422, 209], [422, 207], [421, 206], [421, 205], [420, 204], [420, 202], [419, 201], [419, 199], [418, 198], [418, 197], [417, 196], [417, 194], [416, 193], [416, 191], [415, 190], [415, 188], [414, 187], [414, 185], [413, 184], [413, 183], [412, 182], [411, 182], [410, 181]])
    cup_centroid = np.array([383, 231])

    boundary = np.loadtxt("txt/boundary.txt")
    center = np.loadtxt("txt/center.txt")

    input_boundary = boundary
    input_centroid = center

    # fig0 = plt.figure(num=0)
    # plt.plot(input_boundary.T[0], input_boundary.T[1])
    # plt.show()

    input_boundary = np.array([p for p_i, p in enumerate(input_boundary) if p_i % 2 == 0])
    input_boundary = input_boundary * np.array([1, -1])
    input_centroid = input_centroid * np.array([1, -1])

    # 用于初始化图像的收集
    fig1 = plt.figure(num=1)
    num = 3
    for n in range(num):
        input_boundary_1 = np.matmul((input_boundary - input_centroid), np.array(
            [[np.cos(np.pi * n / num), np.sin(np.pi * n / num)],
             [-np.sin(np.pi * n / num), np.cos(np.pi * n / num)]])) + input_centroid
        plt.plot(input_boundary_1.T[0], input_boundary_1.T[1])

    # plt.axis("off")
    # plt.xticks([])
    # plt.yticks([])
    plt.grid(True, linestyle="--", alpha=0.5)
    # 获得当前的axis
    ax1 = plt.gca()
    # 设置图像的上边、右边axis为无色
    ax1.spines['right'].set_color('none')
    ax1.spines['bottom'].set_color('none')
    # 设置x轴坐标在下部
    ax1.xaxis.set_ticks_position('top')
    plt.savefig('figs/bianjie.pdf')
    # plt.show()

    input_boundary = input_boundary * np.array([1, -1])
    input_centroid = input_centroid * np.array([1, -1])

    # 对输入的进行一下简单的角度变换测试鲁棒性
    # input_boundary = np.matmul((input_boundary - input_centroid), np.array(
    #     [[np.cos(np.pi * 255 / 180), np.sin(np.pi * 255 / 180)],
    #      [-np.sin(np.pi * 255 / 180), np.cos(np.pi * 255 / 180)]])) + input_centroid
    print(len(input_boundary))
    print(input_centroid)
    grasp_wrench_2d(input_boundary, input_centroid)
