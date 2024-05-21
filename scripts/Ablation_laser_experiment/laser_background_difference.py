# coding:utf-8
#
# laser_background_difference.py.py
#
#  Created on: 2024/5/1
#      Author: Tex Yan Liu
#
# description:

# coding:utf-8
#
# laser_SAM.py.py
#
#  Created on: 2023/5/24
#      Author: Tex Yan Liu
#
# description: 包括检测和分割

import argparse  # python的命令行解析的标准模块  可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import os
import shutil
import time
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块
import cv2    # 惊天大BUG，cv2必须在torch上面，否则会RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
import torch
import torch.backends.cudnn as cudnn  # cuda模块
from numpy import random
import numpy as np
import pyrealsense2 as rs  # 导入realsense的sdk模块
import sys
import json  # 作用未知
import threading
import quaternion
import open3d as o3d
from pyinstrument import Profiler
import copy
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.plots import plot_one_box, plot_new_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox
from segment_anything import sam_model_registry, SamPredictor  # 被点图片识别
from segment_anything import build_sam, SamAutomaticMaskGenerator  # 整张图片识别
from scipy.spatial.distance import cdist  # 样本和中心的距离
from scipy.cluster.vq import kmeans, vq  # 聚类

sys.path.insert(0, r"/home/ly/catkin_ws/src/visual_affordance_ros/scripts/yolov5")  # 导入yolov5的路径
sys.path.insert(0, r"/home/ly/catkin_ws/src/visual_affordance_ros/scripts/grasp_stability")  # 导入yolov5的路径

# 定义realsense相机的API
pipeline = rs.pipeline()  # 创建管道-这是流媒体和处理帧的顶级API 该管道简化了用户与设备和计算机视觉处理模块的交互。
config = rs.config()  # 该配置允许管道用户为管道流以及设备选择和配置请求过滤器。
pipeline_wrapper = rs.pipeline_wrapper(pipeline)  # 管道握手函数
pipeline_profile = config.resolve(pipeline_wrapper)  # 管道配置
rgbd_device = pipeline_profile.get_device()  # 获取设备
print(rgbd_device)
found_rgb = False
for s in rgbd_device.sensors:  # 验证相机信息是否拉取完全
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The Manual_demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置深度图像
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置彩色图像
pipeline.start(config)  # 启动相关配置的API

# 定义一个MyThread.py线程类，构造多线程，用于SAM计算分割
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.result = None
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None


# 定义鼠标点击事件
def mouse_click(event, mouse_x, mouse_y, flags, param):  # 鼠标点击事件
    # 全局变量，输入点，响应信号
    global seg_input_point_mouse, seg_input_label_mouse, seg_input_stop_mouse, center_x_mouse, center_y_mouse
    if not seg_input_stop_mouse:  # 判定标志是否停止输入响应了！
        if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键
            seg_input_point_mouse.append([mouse_x, mouse_y])
            seg_input_label_mouse.append(1)  # 1表示前景点
            seg_input_stop_mouse = True  # 定义语义分割的输入状态
            center_x_mouse = mouse_x
            center_y_mouse = mouse_y
        elif event == cv2.EVENT_RBUTTONDOWN:  # 鼠标右键
            seg_input_point_mouse.append([mouse_x, mouse_y])
            seg_input_label_mouse.append(0)  # 0表示后景点
            seg_input_stop_mouse = True
            center_x_mouse = mouse_x
            center_y_mouse = mouse_y
        elif event == cv2.EVENT_FLAG_LBUTTON:  # 鼠标左键长按 重置
            seg_input_point_mouse = []
            seg_input_label_mouse = []
            seg_input_stop_mouse = True
            center_x_mouse = mouse_x
            center_y_mouse = mouse_y


# realsense图像对齐函数
def get_aligned_images():
    # 创建对齐对象与color流对齐
    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    d435_aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    d435_aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    # 将images转为numpy arrays
    d435_img_color = np.asanyarray(d435_aligned_color_frame.get_data())  # BGR图
    d435_img_depth = np.asanyarray(d435_aligned_depth_frame.get_data())  # 深度图

    # 获取相机参数
    d435_depth_intrin = d435_aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    d435_color_intrin = d435_aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    #print("深度内参", d435_depth_intrin)
    #print("彩色内参", d435_color_intrin)  # 目前看来，他和深度内参的一样
    d435_depth_to_color_extrin = d435_aligned_depth_frame.profile.get_extrinsics_to(d435_aligned_color_frame.profile)
    d435_color_to_depth_extrin = d435_aligned_color_frame.profile.get_extrinsics_to(d435_aligned_depth_frame.profile)

    d435_depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(d435_img_depth, alpha=0.03), cv2.COLORMAP_JET)

    # return 1相机内参，2深度参数，3BRG图，4深度图，5深度彩色映射图，6对齐的彩色帧，7对齐的深度帧
    return d435_color_intrin, d435_depth_intrin, d435_img_color, d435_img_depth, d435_depth_mapped_image, \
        d435_aligned_color_frame, d435_aligned_depth_frame, d435_depth_to_color_extrin, d435_color_to_depth_extrin


# 开运算，去红色噪音
def open_mor(src):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=3)  # iterations进行3次操作
    return opening


def laser_point_process(laser_tmp_color_rs):  # 激光点 参考于：https://blog.csdn.net/Kang14789/article/details/124049626
    laser_tmp_state = 0  # 设置激光响应状态，如果未获得激光为0，获得激光为1
    laser_tmp_center_x = 0  # 初始花激光中心点 x
    laser_tmp_center_y = 0  # 初始化激光中心点 y
    laser_tmp_color_rgb = cv2.cvtColor(laser_tmp_color_rs, cv2.COLOR_BGR2RGB)  # 转为rgb图像
    laser_tmp_color_hsv = cv2.cvtColor(laser_tmp_color_rgb, cv2.COLOR_RGB2HSV)  # 转为HSV图像

    # 设置红色范围,在HSV中，红色有两个范围
    red_point_lower1 = np.array([0, 43, 46])
    red_point_upper1 = np.array([10, 255, 255])
    red_point_mask1 = cv2.inRange(laser_tmp_color_hsv, red_point_lower1, red_point_upper1)  # red_point_mask1 为二值图像
    # 位运算，return的是激光点扫描识别区的图像, 仅显示有红色激光点的图像
    # res1 = cv2.bitwise_and(laser_tmp_color_rgb, laser_tmp_color_rgb, mask=red_point_mask1)
    # cv2.imshow("res1", res1)

    red_point_lower2 = np.array([156, 43, 46])
    red_point_upper2 = np.array([180, 255, 255])
    red_point_mask2 = cv2.inRange(laser_tmp_color_hsv, red_point_lower2, red_point_upper2)
    # res2 = cv2.bitwise_and(laser_tmp_color_rgb, laser_tmp_color_rgb, mask=red_point_mask2)
    # cv2.imshow("res2", res2)

    red_point_mask3 = red_point_mask1 + red_point_mask2  # red_point_mask3 是两个掩码的总和
    # print("red_point_mask3",red_point_mask3)
    red_point_mask3_open = open_mor(red_point_mask3)  # 开运算，去噪音
    # print("red_point_mask3_open", red_point_mask3_open)
    # h1, w1 = red_point_mask3_open.shape
    # 寻找边缘
    laser_tmp_contours, cnt = cv2.findContours(red_point_mask3_open.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(laser_tmp_contours) > 0:  # 轮廓总数超过0
        for laser_tmp_contour in laser_tmp_contours:
            moment_laser_tmp_contours = cv2.moments(laser_tmp_contour)
            laser_tmp_center_x = int(moment_laser_tmp_contours["m10"] / moment_laser_tmp_contours["m00"])
            laser_tmp_center_y = int(moment_laser_tmp_contours["m01"] / moment_laser_tmp_contours["m00"])
            cv2.drawContours(laser_tmp_color_rs, laser_tmp_contour, 0, (0, 255, 0), 2)  # 绘制轮廓，填充（图像，轮廓，轮廓序号，颜色，轮廓线粗细）
            cv2.circle(laser_tmp_color_rs, (laser_tmp_center_x, laser_tmp_center_y), 2, (0, 0, 255), -1)  # 绘制中心点
            text = "laser (mm)"
            cv2.putText(laser_tmp_color_rs, text, (laser_tmp_center_x - 30, laser_tmp_center_y - 20),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (180, 105, 255), 1)

        # if len(laser_tmp_contours) == 1:
        #     moment_laser_tmp_contours = cv2.moments(laser_tmp_contours[0])  # 计算第一条轮廓的各阶矩,字典形式，根据自己的激光点效果寻找轮廓
        # else:
        #     moment_laser_tmp_contours = cv2.moments(laser_tmp_contours[1])  # 计算第二条轮廓的各阶矩,字典形式，根据自己的激光点效果寻找轮廓
        # laser_tmp_center_x = int(moment_laser_tmp_contours["m10"] / moment_laser_tmp_contours["m00"])
        # laser_tmp_center_y = int(moment_laser_tmp_contours["m01"] / moment_laser_tmp_contours["m00"])
        # # mask_open = np.zeros([h1, w1], dtype=mask_open.dtype)   # 应该是用于复制掩码图像的
        laser_tmp_state = 1  # 设置状态，用于后续确认

    # return laser_tmp_state, laser_tmp_contours, laser_tmp_center_x, laser_tmp_center_y
    return laser_tmp_state


def get_6d_pointnetplusplus_npy(open3dpointcloud_for_6dpointnet, tmp_num):
    tmp_points = np.asarray(open3dpointcloud_for_6dpointnet.points)  # 如果想看点的数据需要加上这句话
    tmp_points_colors = np.asarray(open3dpointcloud_for_6dpointnet.colors)  # 点云对应的RGB信息
    k_instrin_in = [[605.471, 0.0, 327.157], [0.0, 604.936, 247.894], [0.0, 0.0, 1.0]]

    info_npy = {
        'smoothed_object_pc': tmp_points.tolist(),
        'image': tmp_points_colors.tolist(),
        'intrinsics_matrix': k_instrin_in
    }
    # print("infoNpy", json.dumps(infoNpy))
    np.save('pre_pointcloud/open3dpointcloud_for_6dpointnet_{}.npy'.format(tmp_num), json.dumps(info_npy))

    return True


def kmeancluster_bounding_box(kmeancluster_tmp_pointcloud):
    kmeancluster_tmp_pointcloud.estimate_normals()   # 利用法向量
    #print("当前点云的法线", np.asarray(kmeancluster_tmp_pointcloud.normals))
    centroids, distortion = kmeans(kmeancluster_tmp_pointcloud.normals, 6)  # 获得了三个聚类向量，但是暂时依旧无法根据这三个聚类向量获得
    #print("聚类中心", centroids)  # 聚类中心理论上应该就是我们想要的方向
    #print("点云的点", np.asarray(kmeancluster_tmp_pointcloud.points))
    #print("聚类中心的数据类型", centroids.shape)
    labels, _ = vq(kmeancluster_tmp_pointcloud.normals, centroids)
    #print("样本分类结果：", labels)
    sum_1 = np.sum(np.where(labels, 0, 1))
    #print("类别0总数", sum_1)  # 计数类别0
    sum_2 = np.sum(np.where(labels, 1, 1))
    #print("类别1总数", sum_2)  # 计数类别1
    max_support_vector_1 = []
    if sum_2 < sum_1:
        max_support_vector_1 = centroids[0]
        tmp_max_support_vector_2 = centroids[1]
    else:
        max_support_vector_1 = centroids[1]
        tmp_max_support_vector_2 = centroids[0]

    max_support_vector_2 = np.cross(max_support_vector_1, tmp_max_support_vector_2)  # 首先通过来两个最优方向叉乘计算出另一个垂直方向
    max_support_vector_3 = np.cross(max_support_vector_1, max_support_vector_2)  # 在根据这个新的垂直方向计算出最后一个垂直方向
    #print("点云的点的数据类型", np.asarray(kmeancluster_tmp_pointcloud.points).shape)
    max_min_value1 = np.matmul(np.asarray(kmeancluster_tmp_pointcloud.points),
                               max_support_vector_1.T)  # (n,3)*(3,3).T 得到三个方向的所有投影值
    max_min_value2 = np.matmul(np.asarray(kmeancluster_tmp_pointcloud.points),
                               max_support_vector_2.T)  # (n,3)*(3,3).T 得到三个方向的所有投影值
    max_min_value3 = np.matmul(np.asarray(kmeancluster_tmp_pointcloud.points),
                               max_support_vector_3.T)  # (n,3)*(3,3).T 得到三个方向的所有投影值
    max_index1 = np.argmax(max_min_value1)  # 索引1
    min_index1 = np.argmin(max_min_value1)
    max_points1 = kmeancluster_tmp_pointcloud.points[max_index1]
    min_points1 = kmeancluster_tmp_pointcloud.points[min_index1]

    max_index2 = np.argmax(max_min_value2)  # 索引2
    min_index2 = np.argmin(max_min_value2)
    max_points2 = kmeancluster_tmp_pointcloud.points[max_index2]
    min_points2 = kmeancluster_tmp_pointcloud.points[min_index2]

    max_index3 = np.argmax(max_min_value3)  # 索引3
    min_index3 = np.argmin(max_min_value3)
    max_points3 = kmeancluster_tmp_pointcloud.points[max_index3]
    min_points3 = kmeancluster_tmp_pointcloud.points[min_index3]

    a_front_face = np.append(max_support_vector_1, -np.matmul(max_support_vector_1, max_points1.T))
    a_back_face = np.append(-max_support_vector_1, -np.matmul(-max_support_vector_1, min_points1.T))
    a_face = np.array([a_front_face, a_back_face])  # a_face 0为正面 1为背面
    b_front_face = np.append(max_support_vector_2, -np.matmul(max_support_vector_2, max_points2.T))
    b_back_face = np.append(-max_support_vector_2, -np.matmul(-max_support_vector_2, min_points2.T))
    b_face = np.array([b_front_face, b_back_face])  # b_face 0为正面 1为背面
    c_front_face = np.append(max_support_vector_3, -np.matmul(max_support_vector_3, max_points3.T))
    c_back_face = np.append(-max_support_vector_3, -np.matmul(-max_support_vector_3, min_points3.T))
    c_face = np.array([c_front_face, c_back_face])  # b_face 0为正面 1为背面
    vertices_set = []
    for i in range(2):   # 获得长方体的8个点
        for j in range(2):
            for k in range(2):
                Mat_array_A = np.array([a_face[i, 0:3], b_face[j, 0:3], c_face[k, 0:3]])
                Mat_array_B = -np.array([a_face[i, 3], b_face[j, 3], c_face[k, 3]])
                if i+j+k == 0:
                    vertices_set = np.linalg.solve(Mat_array_A, Mat_array_B)
                else:
                    vertices_set = np.row_stack((vertices_set, np.linalg.solve(Mat_array_A, Mat_array_B)))

    #print("bounding box的8个顶点", vertices_set)
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    line_pcd = o3d.geometry.LineSet()
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(len(lines))])
    line_pcd.points = o3d.utility.Vector3dVector(vertices_set)

    return line_pcd, centroids


def get_3d_camera_coordinate(get_coord_tmp_depth_pixel, get_coord_tmp_aligned_depth_frame, get_coord_tmp_depth_intrin):
    x = get_coord_tmp_depth_pixel[0]
    y = get_coord_tmp_depth_pixel[1]
    get_coord_tmp_distance = get_coord_tmp_aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    get_coord_tmp_camera_coordinate = rs.rs2_deproject_pixel_to_point(get_coord_tmp_depth_intrin,
                                                                      get_coord_tmp_depth_pixel,
                                                                      get_coord_tmp_distance)
    # print ('camera_coordinate: ',camera_coordinate)
    return get_coord_tmp_distance, get_coord_tmp_camera_coordinate


def SAM_wrapper(frame, seg_input_point, seg_input_label, seg_input_stop):
    sam_checkpoint = "segment-anything/checkpoints/sam_vit_h_4b8939.pth"  # sam的权重文件
    sam_model_type = "vit_h"  # 模型类型
    sam_device = "cuda"  # 应用 cuda
    print("start processing segment")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=sam_device)
    sam_predictor = SamPredictor(sam)
    sam_predictor.set_image(frame)
    choosen_point = np.array(seg_input_point)
    # print("choosen_point", choosen_point)
    choosen_label = np.array(seg_input_label)  # 标签， 1是前景点用于选中，2是背景点用于排斥
    # print("choosen_label", choosen_label)
    sam_tmp_masks, sam_tmp_scores, sam_tmp_logits = sam_predictor.predict(
        point_coords=choosen_point,
        point_labels=choosen_label,
        multimask_output=True,
    )
    # sam_all_mask_tmp_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="checkpoints/sam_vit_h_4b8939.pth"))
    # sam_all_tmp_masks = sam_all_mask_tmp_generator.generate(frame)
    return sam_tmp_masks, sam_tmp_scores, sam_tmp_logits  # , sam_all_tmp_masks


# 定义眼在手外(eye-to-hand)的外参矩阵，眼在手 Tpoint_to_baselink = Tcamtobase * Tpoint_to_cam
def eye_to_hand_matrix(camera_tmp_coordinate):
    camera_tmp_coordinate = np.array([camera_tmp_coordinate])
    # print("camera_tmp_coordinate", camera_tmp_coordinate)
    # 手眼平移[x, y, z] 和 手眼四元数*[qw,qx,qy,qz]*（重点） 标定于catkin_ws中的handeye_calibrate
    eye_to_hand_t_matrix = np.array([[0.591117, -0.046708, 0.754738]])
    eye_to_hand_q_matrix = np.quaternion(0.10097695253979255, -0.23429649363885538,
                                         -0.8882804846195733, 0.38192484701617724)
    # 四元数转旋转矩阵
    eye_to_hand_r_matrix = quaternion.as_rotation_matrix(eye_to_hand_q_matrix)
    base_link_tmp_coordinate = np.matmul(eye_to_hand_r_matrix, camera_tmp_coordinate.T) + eye_to_hand_t_matrix.T

    return base_link_tmp_coordinate  # 返回两个，一个是相对于手抓末端的坐标，一个是相对于基座，也是后续用于执行的坐标


# 定义眼在手上(eye-on-hand)的外参矩阵，眼在手 Tpoint_to_baselink = Tendtobase * Tcam_to_end * Tpoint_to_cam
def eye_on_hand_matrix(camera_tmp_coordinate):
    camera_tmp_coordinate = np.array([camera_tmp_coordinate])
    # print("camera_tmp_coordinate", camera_tmp_coordinate)
    # 手眼平移[x, y, z] 和 手眼四元数*[qw,qx,qy,qz]*（重点） 标定于catkin_ws中的handeye_calibrate
    eye_on_hand_t_matrix = np.array([[0.0548432, 0.0871482, -0.146589]])
    eye_on_hand_q_matrix = np.quaternion(0.12934569723181133, 0.01040190109549791,
                                         0.0054398078540793815, 0.9915300800034894)
    # 四元数转旋转矩阵
    eye_on_hand_r_matrix = quaternion.as_rotation_matrix(eye_on_hand_q_matrix)
    end_effector_tmp_coordinate = np.matmul(eye_on_hand_r_matrix, camera_tmp_coordinate.T) + eye_on_hand_t_matrix.T
    # 手到基座平移矩阵，需要等待从KINOVA_API中回调
    end_effector_to_base_link_t_matrix = np.array([[-0.013422133401036263, -0.2824711799621582, 0.508553147315979]])
    # 手到基座四元数，需要等待从KINOVA_API中回调
    end_effector_to_base_link_q_matrix = np.quaternion(0.2496097981929779, 0.8791682124137878,
                                                       0.35243546962738037, 0.20136378705501556)
    # 手到基座旋转矩阵，自己转换就行
    end_effector_to_base_link_r_matrix = quaternion.as_rotation_matrix(end_effector_to_base_link_q_matrix)
    # 所点选物品基于基座的坐标
    base_link_tmp_coordinate = np.matmul(end_effector_to_base_link_r_matrix, end_effector_tmp_coordinate) + end_effector_to_base_link_t_matrix.T
    # print("base_link_tmp_coordinate", base_link_tmp_coordinate)
    return end_effector_tmp_coordinate, base_link_tmp_coordinate  # 返回两个，一个是相对于手抓末端的坐标，一个是相对于基座，也是后续用于执行的坐标


def apply_color_mask(apply_color_tmp_image, apply_color_tmp_mask, apply_color_tmp_color):  # 对掩码进行赋予颜色
    color_dark = 0.6
    # print("apply_color_tmp_mask", apply_color_tmp_mask)
    for c in range(3):
        # np.where(condition, x, y)  满足condition取x,不满足取y
        apply_color_tmp_image[:, :, c] = np.where(
            apply_color_tmp_mask == 1,
            apply_color_tmp_image[:, :, c] * (1 - color_dark) + color_dark * apply_color_tmp_color[c],
            apply_color_tmp_image[:, :, c])
    return apply_color_tmp_image


def apply_white_mask(apply_color_tmp_image, apply_color_tmp_mask, apply_color_tmp_color):  # 对掩码进行赋予颜色
    color_dark = 0
    # print("apply_color_tmp_mask", apply_color_tmp_mask)
    for c in range(3):
        # np.where(condition, x, y)  满足condition取x,不满足取y
        apply_color_tmp_image[:, :, c] = np.where(
            apply_color_tmp_mask == 1,
            apply_color_tmp_image[:, :, c] * (1 - color_dark) + color_dark * apply_color_tmp_color[c], 255)
    return apply_color_tmp_image


def apply_pointcloud_mask(depth_image, mask):
    return np.where(mask == 1, depth_image, 0)


def quaternion_to_rotation_matrix(qx_o_, qy_o_, qz_o_, qw_o_):  # 四元数转化为欧拉矩阵，留着验证用
    return np.array([[1-2*qy_o_*qy_o_-2*qz_o_*qz_o_,   2*qx_o_*qy_o_-2*qz_o_*qw_o_,   2*qx_o_*qz_o_+2*qy_o_*qw_o_],
                     [  2*qx_o_*qy_o_+2*qz_o_*qw_o_, 1-2*qx_o_*qx_o_-2*qz_o_*qz_o_,   2*qy_o_*qz_o_-2*qx_o_*qw_o_],
                     [  2*qx_o_*qz_o_-2*qy_o_*qw_o_,   2*qy_o_*qz_o_+2*qx_o_*qw_o_, 1-2*qx_o_*qx_o_-2*qy_o_*qy_o_]])


# 这个可供性设计主要用于判断场景中目标物品和周围物体的空间可供性，从而识别障碍物
def grasp_affordance(gripper_width=115, gripper_min_depth=50, gripper_max_depth=86):

    return 1


def get_pointcloud(pointcloud_tmp_color_rs, pointcloud_tmp_depth_rs):  # 对齐后的彩色图和深度图作为输入，不是彩色帧的数组和深度帧的数组
    # 因为open3d处理的是RGB的，而realsense出来的是BGR的，需要在此处转换以下颜色通道
    pointcloud_tmp_color_rs = cv2.cvtColor(pointcloud_tmp_color_rs, cv2.COLOR_BGR2RGB)
    pointcloud_tmp_color_rs = o3d.geometry.Image(pointcloud_tmp_color_rs)
    pointcloud_tmp_depth_rs = o3d.geometry.Image(pointcloud_tmp_depth_rs)
    # print("pointcloud_tmp_color_rs", pointcloud_tmp_color_rs)
    # print("pointcloud_tmp_depth_rs", pointcloud_tmp_depth_rs)
    pointcloud_tmp_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(pointcloud_tmp_color_rs,
                                                                                   pointcloud_tmp_depth_rs,
                                                                                   depth_scale=1000.0,
                                                                                   depth_trunc=3.0,
                                                                                   convert_rgb_to_intensity=False)
    # print(np.asarray(rgbd_image.depth))  # 看点的深度
    # 通过彩色图和深度图的点云合成函数
    open3d_process_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        pointcloud_tmp_rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            640, 480, 605.470703125,  604.9356689453125, 327.1570129394531, 247.89410400390625))  # 这个地方默认是识别不了的，需要改成相机D435的内参
    open3d_process_pointcloud.estimate_normals()
    return open3d_process_pointcloud


def open3d_eye_to_base_trans(trans_pointcloud):  # eye to hand 眼在手外，所以可以直接眼睛到基座 当云台的位姿R yaw:-30, pitch:45
    # 计算相对于机械臂末端的点云 qw, qx, qy, qz
    end_rotation = trans_pointcloud.get_rotation_matrix_from_quaternion((0.10097695253979255,
                                                                         -0.23429649363885538,
                                                                         -0.8882804846195733,
                                                                         0.38192484701617724))
    trans_pointcloud_end = copy.deepcopy(trans_pointcloud)
    trans_pointcloud_end = trans_pointcloud_end.rotate(end_rotation, center=(0, 0, 0))
    trans_pointcloud_end = trans_pointcloud_end.translate((0.591117,
                                                           -0.046708,
                                                           0.754738))

    return trans_pointcloud_end


# -0.0378797  -0.0363309    0.0272868    -0.0437703    -0.00384224    0.0171487 云台 x,y,z,rx,ry,rz(rad)
# -0.0378797  -0.0363309    0.0272868    -0.0437703  0.9997222945779027 -0.021866090020998537 -0.0021082214761615438 0.008530136768776517 x y z qw qx qy qz


def open3d_eye_to_end_trans(trans_pointcloud):  # eye on hand 眼在手上，所以需要眼到手，手再到基座
    # 计算相对于机械臂末端的点云 qw, qx, qy, qz
    end_rotation = trans_pointcloud.get_rotation_matrix_from_quaternion((0.12934569723181133,
                                                                         0.01040190109549791,
                                                                         0.0054398078540793815,
                                                                         0.9915300800034894))
    trans_pointcloud_end = copy.deepcopy(trans_pointcloud)
    trans_pointcloud_end = trans_pointcloud_end.rotate(end_rotation, center=(0, 0, 0))
    trans_pointcloud_end = trans_pointcloud_end.translate((0.0548432,
                                                           0.0871482,
                                                           -0.146589))
    return trans_pointcloud_end


def open3d_end_to_base_trans(trans_pointcloud_end):  # eye on hand 眼在手上，所以需要眼到手，手再到基座
    # 计算相对于机械臂基座的点云，该坐标变换需要根据KINOVA_API_tool_pose调用而定
    base_rotation = trans_pointcloud_end.get_rotation_matrix_from_quaternion((0.2496097981929779,
                                                                              0.8791682124137878,
                                                                              0.35243546962738037,
                                                                              0.20136378705501556))
    trans_pointcloud_base = copy.deepcopy(trans_pointcloud_end)
    trans_pointcloud_base = trans_pointcloud_base.rotate(base_rotation, center=(0, 0, 0))
    trans_pointcloud_base = trans_pointcloud_base.translate((-0.013422133401036263,
                                                             -0.2824711799621582,
                                                             0.508553147315979))

    return trans_pointcloud_base

def global_pointcloud_process(global_tmp_pointcloud, pointcloud_tmp_num, kmean_lines_in_global, global_tmp_pointcloud_bounding_box_min_obb, tmp_a, tmp_b):

    passthrough_bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-2, -2, 0),
                                                                   max_bound=(2, 2, 2.5))  # 直通滤波全局点云
    global_tmp_pointcloud = global_tmp_pointcloud.crop(passthrough_bounding_box)
    # o3d.visualization.draw_geometries([open3d_process_pointcloud])
    pointcloud_tmp_name = "global"
    o3d.io.write_point_cloud("pre_pointcloud/{}_camera_{}.ply".format(pointcloud_tmp_name, pointcloud_tmp_num),
                             global_tmp_pointcloud)
    global_tmp_pointcloud_end = open3d_eye_to_end_trans(global_tmp_pointcloud)
    o3d.io.write_point_cloud("pre_pointcloud/{}_end_{}.ply".format(pointcloud_tmp_name, pointcloud_tmp_num),
                             global_tmp_pointcloud_end)
    global_tmp_pointcloud_base = open3d_end_to_base_trans(global_tmp_pointcloud_end)
    o3d.io.write_point_cloud("pre_pointcloud/{}_base_{}.ply".format(pointcloud_tmp_name, pointcloud_tmp_num),
                             global_tmp_pointcloud_base)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])  # 坐标轴
    #print("kmean_lines_in_global", kmean_lines_in_global)
    #print("global_tmp_pointcloud", global_tmp_pointcloud)   # append 失败了， 明天在改
    kmean_lines_in_global.append(global_tmp_pointcloud)
    kmean_lines_in_global.append(axis_pcd)
    kmean_lines_in_global.append(global_tmp_pointcloud_bounding_box_min_obb)
    kmean_lines_in_global.append(tmp_a)
    kmean_lines_in_global.append(tmp_b)

    # o3d.visualization.draw_geometries(kmean_lines_in_global)  # , point_show_normal=True)
    return True


def local_pointcloud_process(local_tmp_pointcloud, pointcloud_tmp_num):

    cl1, ind1 = local_tmp_pointcloud.remove_statistical_outlier(nb_neighbors=80, std_ratio=0.5)  # remove_statistical_outlier 统计异常值去除 会删除距离其邻居较远的点。
    local_tmp_pointcloud = local_tmp_pointcloud.select_by_index(ind1)
    cl2, ind2 = local_tmp_pointcloud.remove_radius_outlier(nb_points=60, radius=1.5)
    local_tmp_pointcloud = local_tmp_pointcloud.select_by_index(ind2)
    local_tmp_pointcloud = local_tmp_pointcloud.uniform_down_sample(5)  # 均匀下采样，在之后的实验中可以通过调节点云下采样的密度进行实验
    # --------在这里需要再补充一个聚类分割，提取更大块目标点云
    local_tmp_pointcloud_bounding_box_min_obb = local_tmp_pointcloud.get_minimal_oriented_bounding_box(robust=True)
    local_tmp_pointcloud_bounding_box_min_obb.color = (0, 0, 1)  # 蓝色
    local_tmp_pointcloud_bounding_box_obb = local_tmp_pointcloud.get_oriented_bounding_box(robust=True)
    local_tmp_pointcloud_bounding_box_obb.color = (1, 0, 0)  # 红色
    local_tmp_pointcloud_bounding_box = local_tmp_pointcloud.get_axis_aligned_bounding_box()
    local_tmp_pointcloud_bounding_box.color = (0, 1, 0)  # 绿色

    line_pcd, centroids = kmeancluster_bounding_box(local_tmp_pointcloud)

    centroids = np.row_stack((centroids, [0,0,0]))
    lines1 = [[0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [5, 6]]
    line_pcd_1 = o3d.geometry.LineSet()
    line_pcd_1.lines = o3d.utility.Vector2iVector(lines1)
    line_pcd_1.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for i in range(len(lines1))])
    line_pcd_1.points = o3d.utility.Vector3dVector(centroids)

    # print("最小OBB的中心", local_tmp_pointcloud_bounding_box_min_obb.get_center())
    # print("OBB的中心", local_tmp_pointcloud_bounding_box_obb.get_center())
    # print("AABB的中心", local_tmp_pointcloud_bounding_box.get_center())
    # print("AABB的最大边", local_tmp_pointcloud_bounding_box.get_max_bound())
    # print("AABB的最短边", local_tmp_pointcloud_bounding_box.get_min_bound())

    # mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.01,
    #                                                 height=0.02,
    #                                                 depth=0.03,
    #                                                 create_uv_map=True, map_texture_to_each_face=False)
    # mesh_box.compute_vertex_normals()
    # mesh_box.paint_uniform_color([1, 0, 0])
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])  # 坐标轴
    # o3d.visualization.draw_geometries([local_tmp_pointcloud,
    #                                    local_tmp_pointcloud_bounding_box,
    #                                    local_tmp_pointcloud_bounding_box_obb,
    #                                    local_tmp_pointcloud_bounding_box_min_obb,
    #                                    axis_pcd,
    #                                    line_pcd], point_show_normal=False)
    # o3d.visualization.draw_geometries([open3d_process_pointcloud_base, axis_pcd, line_pcd, line_pcd_1], point_show_normal=True)
    get_6d_pointnetplusplus_npy(local_tmp_pointcloud, pointcloud_tmp_num)  # 储存被选中点云的信息，return完成True状态
    pointcloud_tmp_name = "local"
    o3d.io.write_point_cloud("pre_pointcloud/{}_camera_{}.ply".format(pointcloud_tmp_name, pointcloud_tmp_num),
                             local_tmp_pointcloud)
    local_tmp_pointcloud_end = open3d_eye_to_end_trans(local_tmp_pointcloud)
    o3d.io.write_point_cloud("pre_pointcloud/{}_end_{}.ply".format(pointcloud_tmp_name, pointcloud_tmp_num),
                             local_tmp_pointcloud_end)
    local_tmp_pointcloud_base = open3d_end_to_base_trans(local_tmp_pointcloud_end)
    o3d.io.write_point_cloud("pre_pointcloud/{}_base_{}.ply".format(pointcloud_tmp_name, pointcloud_tmp_num),
                             local_tmp_pointcloud_base)

    # o3d.visualization.draw_geometries([local_tmp_pointcloud,
    #                                    local_tmp_pointcloud_end,
    #                                    local_tmp_pointcloud_base])  # 点云在open3d中的可视化

    return line_pcd, local_tmp_pointcloud_bounding_box_min_obb, local_tmp_pointcloud_bounding_box, local_tmp_pointcloud_bounding_box_obb


def yolov5_detect(img_color, save_img=False):

    yolov5_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(yolov5_names))]
    mask = np.zeros([img_color.shape[0], img_color.shape[1]], dtype=np.uint8)
    mask[0:480, 320:640] = 255
    # 对RGB的img进行处理，送入预测模型
    yolov5_imgs = [None]
    yolov5_imgs[0] = img_color
    yolov5_im0s = yolov5_imgs.copy()  # img0s: 原尺寸的图片
    yolov5_img = [letterbox(x, new_shape=yolov5_imgsz)[0] for x in yolov5_im0s]  # img: 进行resize + pad之后的图片
    yolov5_img = np.stack(yolov5_img, 0)  # 沿着0dim进行堆叠
    yolov5_img = yolov5_img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416, uint8 to float32
    yolov5_img = np.ascontiguousarray(yolov5_img, dtype=np.float16 if yolov5_half else np.float32)
    # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    yolov5_img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # 处理每一张图片的数据格式
    yolov5_img = torch.from_numpy(yolov5_img).to(yolov5_device)  # 将numpy转为pytorch的tensor,并转移到运算设备上计算
    # 如果图片是3维(RGB) 就在前面添加一个维度1当中batch_size=1
    # 因为输入网络的图片需要是4为的 [batch_size, channel, w, h]
    if yolov5_img.ndimension() == 3:
        yolov5_img = yolov5_img.unsqueeze(0)  # 在dim0位置添加维度1，[channel, w, h] -> [batch_size, channel, w, h]
    # 对每张图片/视频进行前向推理
    yolov5_pred = yolov5_model(yolov5_img, augment=opt.augment)[0]

    # 进行NMS
    # conf_thres: 置信度阈值
    # iou_thres: iou阈值
    # classes: 是否只保留特定的类别 默认为None
    # agnostic_nms: 进行nms是否也去除不同类别之间的框 默认False
    # max_det: 每张图片的最大目标个数 默认1000
    # pred: [num_obj, 6] = [5, 6]   这里的预测信息pred还是相对于 img_size(640) 的
    yolov5_pred = non_max_suppression(yolov5_pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # 后续保存或者打印预测信息
    for v5_i, v5_det in enumerate(yolov5_pred):  # detections per image
        v5_s, yolov5_im0 = '%g: ' % v5_i, yolov5_im0s[v5_i].copy()
        v5_s += '%gx%g ' % yolov5_img.shape[2:]  # print string
        gn = torch.tensor(yolov5_im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if v5_det is not None and len(v5_det):
            # Rescale boxes from img_size to im0 size
            v5_det[:, :4] = scale_coords(yolov5_img.shape[2:], v5_det[:, :4], yolov5_im0.shape).round()

            # Print results
            for c in v5_det[:, -1].unique():
                n = (v5_det[:, -1] == c).sum()  # detections per class
                v5_s += '%g %ss, ' % (n, yolov5_names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(v5_det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化为 xywh
                line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                # print("yolov5_names[int(cls)]", yolov5_names[int(cls)])
                yolov5_label = 'name: %s' % (yolov5_names[int(cls)])  # 最后取平均值为目标深度，深度已经删除了，只有类别的名字
                plot_new_box(xyxy, img_color, label=yolov5_label, color=yolov5_colors[int(cls)], line_thickness=2)  # 将结果框打印回原图
                if yolov5_names[int(cls)] == "cup":
                    tl = 2
                    tf = max(tl - 1, 1)  # font thickness
                    function_label_1 = "function1: pour"
                    function_label_2 = "function2: be_poured"
                    function_label_3 = "function3: be_pushed"
                    function_label_4 = "function4: move"

                    t_size = cv2.getTextSize(function_label_1, 0, fontScale=tl / 3, thickness=tf)[0]
                    cv2.putText(img_color, function_label_1, (int(xyxy[0]), int(xyxy[1])-t_size[1]-2), 0, tl / 3, yolov5_colors[int(cls)], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.putText(img_color, function_label_2, (int(xyxy[0]), int(xyxy[1])-t_size[1]-2-t_size[1]-2), 0, tl / 3, yolov5_colors[int(cls)], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.putText(img_color, function_label_3, (int(xyxy[0]), int(xyxy[1])-t_size[1]-2-t_size[1]-2-t_size[1]-2), 0, tl / 3, yolov5_colors[int(cls)], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.putText(img_color, function_label_4, (int(xyxy[0]), int(xyxy[1])-t_size[1]-2-t_size[1]-2-t_size[1]-2-t_size[1]-2), 0, tl / 3, yolov5_colors[int(cls)], thickness=tf, lineType=cv2.LINE_AA)
                elif yolov5_names[int(cls)] == "bowl":
                    tl = 2
                    tf = max(tl - 1, 1)  # font thickness
                    function_label_1 = "function1: pour"
                    function_label_2 = "function2: be_poured"
                    function_label_3 = "function3: be_pushed"
                    function_label_4 = "function4: move"
                    function_label_5 = "function5: be_stirred"
                    t_size = cv2.getTextSize(function_label_1, 0, fontScale=tl / 3, thickness=tf)[0]
                    cv2.putText(img_color, function_label_1, (int(xyxy[0]), int(xyxy[1])-t_size[1]-2), 0, tl / 3, yolov5_colors[int(cls)], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.putText(img_color, function_label_2, (int(xyxy[0]), int(xyxy[1])-t_size[1]-2-t_size[1]-2), 0, tl / 3, yolov5_colors[int(cls)], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.putText(img_color, function_label_3, (int(xyxy[0]), int(xyxy[1])-t_size[1]-2-t_size[1]-2-t_size[1]-2), 0, tl / 3, yolov5_colors[int(cls)], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.putText(img_color, function_label_4, (int(xyxy[0]), int(xyxy[1])-t_size[1]-2-t_size[1]-2-t_size[1]-2-t_size[1]-2), 0, tl / 3, yolov5_colors[int(cls)], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.putText(img_color, function_label_5, (int(xyxy[0]), int(xyxy[1])-t_size[1]-2-t_size[1]-2-t_size[1]-2-t_size[1]-2-t_size[1]-2), 0, tl / 3, yolov5_colors[int(cls)], thickness=tf, lineType=cv2.LINE_AA)


def two_dimensions_grasp_wrench(two_dimensoions_grasp_mask):
    # print("two_dimensoions_grasp_mask", two_dimensoions_grasp_mask)
    two_dimensoions_grasp_contours, two_dimensoions_grasp_cnt = cv2.findContours(two_dimensoions_grasp_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    moment_2d_grasp_contours = cv2.moments(two_dimensoions_grasp_contours[0])  # 计算第一条轮廓的各阶矩,字典形式，根据自己的激光点效果寻找轮廓
    moment_2d_grasp_center_x = int(moment_2d_grasp_contours["m10"] / moment_2d_grasp_contours["m00"])
    moment_2d_grasp_center_y = int(moment_2d_grasp_contours["m01"] / moment_2d_grasp_contours["m00"])

    return two_dimensoions_grasp_contours, moment_2d_grasp_center_x, moment_2d_grasp_center_y


if __name__ == '__main__':
    # 定义一些全局变量
    seg_input_point_laser = []  # 定义语义分割的输入点（激光的）
    seg_input_label_laser = []  # 定义语义分割的输入标签，包括前景点、背景点
    seg_input_stop_laser = False  # 定义语义分割的输入状态
    seg_input_point_laser_tmp = []  # 定义临时变量
    seg_input_label_laser_tmp = []
    seg_input_point_mouse = []  # 定义语义分割的输入点（鼠标的）
    seg_input_label_mouse = []  # 定义语义分割的输入标签，包括前景点、背景点
    seg_input_stop_mouse = False  # 定义语义分割的输入状态
    camera_coordinate = np.zeros(3)
    center_x_mouse, center_y_mouse = 0.0, 0.0
    cv2.namedWindow("Scene", cv2.WINDOW_NORMAL)  # 初始化界面
    cv2.resizeWindow("Scene", 640, 480)  # 调整界面尺寸
    cv2.setMouseCallback("Scene", mouse_click)  # 调用鼠标点击
    # cap = cv2.VideoCapture(4)  注释了这段对话，因为不用cv导入图像，改为realsense的sdk导入图像
    seg_tmp_masks = []  # 定义一个空mask用于存储被选中target目标的segment mask
    profiler = Profiler()  # 时间规划器、记录整体运行时间
    profiler.start()
    k = 0  # 定义循环次数
    structure_kmean_lines = []
    # k_instrin_in = [[604.645, 0.0, 327.829], [0.0, 604.012, 247.008], [0.0, 0.0, 1.0]]  # 相机内参
    k_instrin_in = [[605.471, 0.0, 327.157], [0.0, 604.936, 247.894], [0.0, 0.0, 1.0]]  # 相机内参

    # yolov5的参数设置
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='../yolov5/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../yolov5/inference/images',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='../yolov5/inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    # 加载参数
    yolov5_out, yolov5_source, yolov5_weights, yolov5_view_img, yolov5_save_txt, yolov5_imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = yolov5_source == '0' or yolov5_source.startswith(
        ('rtsp://', 'rtmp://', 'http://')) or yolov5_source.endswith('.txt')
    # 初始化
    set_logging()  # 生成日志
    yolov5_device = select_device(opt.device)  # 获取当前主机可用的设备
    if os.path.exists(yolov5_out):  # output dir
        shutil.rmtree(yolov5_out)  # delete dir
    os.makedirs(yolov5_out)  # make new dir
    # 如果设配是GPU 就使用half(float16)  包括模型半精度和输入图片半精度
    yolov5_half = yolov5_device.type != 'cpu'  # half precision only supported on CUDA

    # 载入模型和模型参数并调整模型
    yolov5_model = attempt_load(yolov5_weights, map_location=yolov5_device)  # 加载Float32模型
    yolov5_imgsz = check_img_size(yolov5_imgsz,
                                  s=yolov5_model.stride.max())  # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
    if yolov5_half:  # 是否将模型从float32 -> float16  加速推理
        yolov5_model.half()  # to FP16

    # 加载推理数据
    vid_path, vid_writer = None, None
    # 采用webcam数据源
    yolov5_view_img = True
    cudnn.benchmark = True  # 加快常量图像大小推断
    # dataset = LoadStreams(source, img_size=imgsz)  #load 文件夹中视频流

    # 获取每个类别的名字和随机生成类别颜色
    yolov5_names = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    yolov5_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(yolov5_names))]

    # 初始化一个全0tensor进行一次正向推理
    yolov5_img = torch.zeros((1, 3, yolov5_imgsz, yolov5_imgsz), device=yolov5_device)
    _ = yolov5_model(
        yolov5_img.half() if yolov5_half else yolov5_img) if yolov5_device.type != 'cpu' else None  # run once

    try:
        while True:
            # _1, frame = cap.read()  # cv2读取图像
            color_intrin, depth_intrin, img_color, img_depth, depth_mapped_image, aligned_color_frame, \
                aligned_depth_frame, d_to_c_extrin, c_to_d_extrin = get_aligned_images()   # return 1相机内参，2深度参数，3BRG图，4深度图，5深度彩色映射图，6对齐的彩色帧，7对齐的深度帧
            # with torch.no_grad():
            #     yolov5_detect(img_color)

            # laser_state, contours, center_x_laser, center_y_laser = laser_point_process(img_color)  # 激光点处理
            laser_state = laser_point_process(img_color)  # 激光点处理
            # laser_state = False  # 先把系统中的laser 交互断掉
            if laser_state:
                print("laser detection")
                # if len(contours) == 1:
                #     cv2.drawContours(img_color, contours, 0, (0, 255, 0), 2)  # 绘制轮廓，填充（图像，轮廓，轮廓序号，颜色，轮廓线粗细）
                # else:
                #     cv2.drawContours(img_color, contours, 1, (0, 255, 0), 2)  # 绘制轮廓，填充（图像，轮廓，轮廓序号，颜色，轮廓线粗细）
                # cv2.circle(img_color, (center_x_laser, center_y_laser), 2, (0, 0, 255), -1)  # 绘制中心点
                # dis, camera_coordinate = get_3d_camera_coordinate([center_x_laser, center_y_laser],
                #                                                   aligned_depth_frame,
                #                                                   depth_intrin)

                # text = "laser (mm): {:.0f},{:.0f},{:.0f}".format(camera_coordinate[0] * 1000,
                #                                                  camera_coordinate[1] * 1000,
                #                                                  camera_coordinate[2] * 1000)
                # text = "laser (mm)"
                # cv2.putText(img_color, text, (center_x_laser - 30, center_y_laser - 20),
                #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (180, 105, 255), 1)
                # seg_input_point_laser_tmp.append([center_x_laser, center_y_laser])
                # seg_input_label_laser_tmp.append(1)  # 1表示前景点

            cv2.imshow("Scene", img_color)  # 展示图像
            # mix = cv2.addWeighted(img_color, 0.8, depth_mapped_image, 0.2, 0)
            # cv2.imshow("Scene", mix)  # 展示图像
            # ori_frame = img_color  # 储存上一帧
            k = k + 1
            key = cv2.waitKey(1)

            # del seg_input_point[:]
            # del seg_input_label[:]

            if key == 27 or (key & 0XFF == ord("q")):
                cv2.destroyAllWindows()
                break
    finally:
        # destroy the instance
        # cap.release()
        pipeline.stop()

    profiler.stop()
    profiler.print()