# coding:utf-8
#
# gmm.py
#
#  Created on: 2024/3/29
#      Author: Tex Yan Liu
#
# description:  用于对激光点进行期望筛选的算法

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.stats as stats

def gmm(data):
    font_size = 10

    # 设置字体样式
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = font_size

    # 创建围绕圆心的高斯分布样本
    # np.random.seed(0)
    # radius = 5
    # theta = np.random.uniform(0, 2*np.pi, 1000)
    # random_radius = np.random.normal(0, 1, 1000)
    #
    # # x = radius * random_radius * np.cos(theta)
    # # y = radius * random_radius * np.sin(theta)
    # x = np.random.uniform(274, 322, 30)
    # y = np.random.uniform(261, 313, 30)

    # print("x:", x)
    # data = np.column_stack((x, y))
    data = np.array(data)
    print("数据集data是：", data)
    print("数据集data的类型是：", data.shape)

    # 使用Gaussian Mixture模型进行拟合
    gmm = GaussianMixture(n_components=1, covariance_type='full')
    gmm.fit(data)

    # 提取拟合结果的均值和协方差矩阵
    mean = gmm.means_[0]
    print("均值", mean)
    covariance = gmm.covariances_[0]
    print("方差", covariance)

    # 生成拟合的二维高斯分布
    x, y = np.meshgrid(np.linspace(250, 350, 100), np.linspace(210, 350, 100))
    pos = np.dstack((x, y))
    pos = np.column_stack((x.flatten(), y.flatten()))  # 将x和y合并成二维数组
    # print("pos", pos)
    z = gmm.score_samples(pos)

    z2 = gmm.score_samples([[mean[0] - np.sqrt(covariance[0][0]), mean[1] - np.sqrt(covariance[1][1])]])
    z3 = gmm.score_samples([[mean[0] - 2*np.sqrt(covariance[0][0]), mean[1] - 2*np.sqrt(covariance[1][1])]])
    z4 = gmm.score_samples([[mean[0] - 3*np.sqrt(covariance[0][0]), mean[1] - 3*np.sqrt(covariance[1][1])]])
    # print("一个标准差等高值", z2)
    # print("二个标准差等高值", z3)
    # print("三个标准差等高值", z4)
    contour_level = np.sort(np.array([z2, z3]), axis=0).flatten()  # axis=0等于按列排序
    # print("排序后的值", contour_level)

    z = z.reshape(x.shape)
    # 可视化拟合结果
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, marker="*")
    plt.scatter(mean[0], mean[1], color="brown", alpha=0.5, marker="*")
    colors = ['orange', 'purple', 'brown', 'red', 'blue']
    # C = plt.contour(x, y, z, contour_level, cmap=matplotlib.cm.jet, linestyles="--")  # 等高线图表示概率密度函数
    C = plt.contour(x, y, z, contour_level, colors=colors, linestyles="--")  # 等高线图表示概率密度函数
    # C = plt.contour(x, y, z, levels=10)  # 设置等高线数量为10条


    plt.xlabel('pixel x', fontsize=font_size)
    # 设置X轴刻度
    plt.xlim(0, 640)
    plt.ylabel('pixel y', fontsize=font_size)
    # 设置X轴刻度
    plt.ylim(480, 0)
    plt.title('Fitting 2D Gaussian Model to Data', fontsize=font_size)
    # plt.axis('equal')

    # 添加等高线参数的注释框
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # textstr = f'One Std Dev: {z2[0]:.2f}\nTwo Std Dev: {z3[0]:.2f}\nThree Std Dev: {z4[0]:.2f}'
    # plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=props, fontsize=font_size)

    # textstr = "One Std Dev: Yellow Dashed Line"
    # plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=props, fontsize=8)

    # 根据等高线样式画图例
    legend_handles1 = [plt.Line2D([0], [0], color='purple', linestyle='--', lw=2)]
    legend_labels1 = ['One Std Dev']

    # 根据等高线样式画图例
    legend_handles2 = [plt.Line2D([0], [0], color='orange', linestyle='--', lw=2)]
    legend_labels2 = ['Two Std Dev']

    legend_handles = legend_handles1 + legend_handles2
    # print("legend_handles", legend_handles[0])
    legend_labels = legend_labels1 + legend_labels2

    plt.legend(legend_handles, legend_labels, loc='upper right', fontsize=font_size, fancybox=True, edgecolor='black')
    plt.savefig('experiment_picture/duoyuangaosi.png', dpi=600)
    # plt.show()
    return mean

