#!/usr/bin/env python3
# coding:utf-8
#
# house_rebuild_teaserpp.py
#
#  Created on: 2023年3月6日
#      Author: Tex Yan Liu
# description: 测试Teaserpp算法的效果


import pcl
from pcl import pcl_visualization
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import math
from pycpd import RigidRegistration
from functools import partial
from pyinstrument import Profiler

address = "/home/wmra/pointcloud_restruct"


def filter_points(tem_points):

    fil0 = tem_points.make_passthrough_filter()
    fil0.set_filter_field_name("z")
    fil0.set_filter_limits(0.2, 1.1)
    fil_pass = fil0.filter()

    fil1 = fil_pass.make_statistical_outlier_filter()
    fil1.set_mean_k(10)
    fil1.set_std_dev_mul_thresh(1.0)
    fil_statis = fil1.filter()

    fil2 = fil_statis.make_voxel_grid_filter()
    fil2.set_leaf_size(0.015, 0.015, 0.015)
    fil_voxel = fil2.filter()

    return fil_voxel


def drawing(src_1, src_2, index):

    # print("s_x", s_x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #print(s_x)
    #ax.plot_surface(s_x, s_z, s_y, rstride = 1, cstride = 1, cmap=cm.RdYlGn, linewidth=0, alpha=0.4, antialiased=False)
    ax.scatter(src_1[0],src_1[1],src_1[2], s=0.01, c="r", marker = "x", label="source")
    ax.scatter(src_2[0],src_2[1],src_2[2], s=0.01, c="b", marker = "x", label="target")
    
    ax.set_xlabel('X Label')
    #ax.set_xlim(40, 160)
    ax.set_ylabel('Z Label')
    #ax.set_ylim(0, 1000)
    ax.set_zlabel('Y Label')
    #ax.set_zlim(150, -25)
    #ax.set_box_aspect()
    plt.savefig("rebuild k={}.svg".format(index), format="svg")  # 耗费时间19秒
    plt.show()
    return 1


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def coherent_point_drift(source_point, target_point):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)
    reg = RigidRegistration(**{'X': target_point, 'Y': source_point})
    TY, (scale_matrix, rotate_matrix, translation_matrix) = reg.register(callback)
    plt.show()
    print("scale_matrix",scale_matrix)
    print("rotate_matrix",rotate_matrix)
    print("translation_matrix",translation_matrix)
    return scale_matrix, rotate_matrix, translation_matrix


def running():

    print("===========================================")
    print("   CPD Point registration solver .......   ")
    print("===========================================")

    profiler = Profiler() # 计时分析
    profiler.start()

    src_cloud = pcl.load(address + "/test_1.pcd")
    target_cloud = pcl.load(address + "/test_2.pcd")

    src_cloud = filter_points(src_cloud)
    target_cloud = filter_points(target_cloud)

    src_cloud = src_cloud.to_array()  # CPD算法要n*3的格式
    target_cloud = target_cloud.to_array()

    # 测试
    #scale = 1.5
    #translation = np.array([[1], [0], [-1]])
    #rotation = np.array([[0.98370992, 0.17903344, -0.01618098],
    #                     [-0.04165862, 0.13947877, -0.98934839],
    #                     [-0.17486954, 0.9739059, 0.14466493]])

    #target_cloud = scale * np.matmul(rotation, src_cloud) + translation

    drawing(src_cloud.T, target_cloud.T,1)  # 检验一下变换前的形态

    scale, rotation, translation = coherent_point_drift(src_cloud, target_cloud)  # 经过cpd算法输出变换系数

    src_cloud =  scale * np.matmul(src_cloud, rotation) + translation
    drawing(src_cloud.T, target_cloud.T, 2)  # 检验一下变换后的形态
    profiler.stop()
    profiler.print()


if __name__ == "__main__":

    running()
