# coding:utf-8
#
# Interactive_prompt.py
#
#  Created on: 2024/4/7
#      Author: Tex Yan Liu
#
# description: Ros node for the module of interactive_prompt

import argparse  # python的命令行解析的标准模块  可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import os
import shutil
import rospy
import time
import cv2  # 惊天大BUG，cv2必须在torch上面，否则会RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
import torch
import torch.backends.cudnn as cudnn  # cuda模块

import numpy as np
import pyrealsense2 as rs  # 导入realsense的sdk模块
import threading
import open3d as o3d
from pyinstrument import Profiler
import copy
import sys
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.plots import plot_one_box, plot_new_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox
from numpy import random

from Anything_Grasping.msg import prompt_info
from Anything_Grasping.msg import grasp_info
from segment_anything import sam_model_registry, SamPredictor  # 被点图片识别
from grasp_wrench_2d_v6 import grasp_wrench_2d
from std_msgs.msg import String

from gmm import gmm

rospy.init_node('interactive_prompt', anonymous=True)
prompt_pub = rospy.Publisher('auto_grasp/interactive_prompt', prompt_info, queue_size=1)
grasp_pub = rospy.Publisher('auto_grasp/grasp_execution', grasp_info, queue_size=1)
laser_pub = rospy.Publisher('auto_grasp/laser_waiter', String, queue_size=1)
grasp_info = grasp_info()
prompt_info = prompt_info()

laser_callback_status = 'laser_back_is_waiting'
pose_adjustment_status = ''
grasp_status_info = ''

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

sam_checkpoint = "../../segment-anything/checkpoints/sam_vit_h_4b8939.pth"  # sam的权重文件
sam_model_type = "vit_h"  # 模型类型
sam_device = "cuda"  # 应用 cuda
print("start processing segment")
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
sam.to(device=sam_device)
sam_predictor = SamPredictor(sam)

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


def SAM_wrapper(frame, seg_input_point, seg_input_label, seg_input_stop):

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
    # print("深度内参", d435_depth_intrin)
    # print("彩色内参", d435_color_intrin)  # 目前看来，他和深度内参的一样
    d435_depth_to_color_extrin = d435_aligned_depth_frame.profile.get_extrinsics_to(d435_aligned_color_frame.profile)
    d435_color_to_depth_extrin = d435_aligned_color_frame.profile.get_extrinsics_to(d435_aligned_depth_frame.profile)

    d435_depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(d435_img_depth, alpha=0.03), cv2.COLORMAP_JET)

    # return 1相机内参，2深度参数，3BRG图，4深度图，5深度彩色映射图，6对齐的彩色帧，7对齐的深度帧
    return d435_color_intrin, d435_depth_intrin, d435_img_color, d435_img_depth, d435_depth_mapped_image, \
        d435_aligned_color_frame, d435_aligned_depth_frame, d435_depth_to_color_extrin, d435_color_to_depth_extrin


def get_3d_camera_coordinate(get_coord_tmp_depth_pixel, get_coord_tmp_aligned_depth_frame, get_coord_tmp_depth_intrin):
    x = get_coord_tmp_depth_pixel[0]
    y = get_coord_tmp_depth_pixel[1]
    get_coord_tmp_distance = get_coord_tmp_aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    get_coord_tmp_camera_coordinate = rs.rs2_deproject_pixel_to_point(get_coord_tmp_depth_intrin,
                                                                      get_coord_tmp_depth_pixel,
                                                                      get_coord_tmp_distance)
    # print ('camera_coordinate: ',camera_coordinate)
    get_coord_tmp_camera_coordinate_round = [round(num, 5) for num in get_coord_tmp_camera_coordinate]
    return get_coord_tmp_distance, get_coord_tmp_camera_coordinate_round


def get_pointcloud(pointcloud_tmp_color_rs, pointcloud_tmp_depth_rs):  # 对齐后的彩色图和深度图作为输入，不是彩色帧的数组和深度帧的数组
    # 因为open3d处理的是RGB的，而realsense出来的是BGR的，需要在此处转换以下颜色通道
    pointcloud_tmp_color_rs = cv2.cvtColor(pointcloud_tmp_color_rs, cv2.COLOR_BGR2RGB)
    pointcloud_tmp_color_rs = o3d.geometry.Image(pointcloud_tmp_color_rs)
    pointcloud_tmp_depth_rs = o3d.geometry.Image(pointcloud_tmp_depth_rs)

    pointcloud_tmp_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(pointcloud_tmp_color_rs,
                                                                                   pointcloud_tmp_depth_rs,
                                                                                   depth_scale=1000.0,
                                                                                   depth_trunc=3.0,
                                                                                   convert_rgb_to_intensity=False)

    open3d_process_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        pointcloud_tmp_rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            640, 480, 605.471, 604.936, 327.157, 247.894))  # 这个地方默认是识别不了的，需要改成相机D435的内参
    open3d_process_pointcloud.estimate_normals()

    passthrough_bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, 0),
                                                                   max_bound=(1, 1, 1.5))  # 直通滤波全局点云  # 蓝色z 红x 绿y
    pcl_fil = open3d_process_pointcloud.crop(passthrough_bounding_box)

    return pcl_fil


def apply_pointcloud_mask(depth_image, mask):
    return np.where(mask == 1, depth_image, 0)


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


def show_pose_o3d_pcl(lin_shi_ceshi_o3d_vs_pc_tmp, coordinate, attention_normal):
    """
    求解机器人的预调整姿态
    """
    attention_t = coordinate  # 提示点相对于相机坐标系原点的坐标 （圆心）
    new_axis_z = attention_normal  # 提示点处的法线方向，该法线方向为原坐标系的Z轴方向
    # print("new_axis_z的模", np.linalg.norm(new_axis_z))
    attention_mol = np.linalg.norm(attention_t)  # 求解距离范数
    attention_t_2 = attention_t - attention_mol * attention_normal  # 求解新的相机坐标原点 （圆弧末端点） （0，0，0 圆弧起始点）
    # print("相机的新的原点attention_t_2", attention_t_2)
    axis_pcd_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2,
                                                                   origin=attention_t_2)  # 坐标轴1
    axis_pcd_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2,
                                                                   origin=attention_t_2)  # 坐标轴2
    # print("attention_q_old:", attention_q_old)
    new_axis_x = np.zeros(3)
    new_axis_x = np.cross(attention_t / attention_mol, new_axis_z)
    new_axis_x = new_axis_x / np.linalg.norm(new_axis_x)
    new_axis_y = np.zeros(3)
    new_axis_y = -np.cross(new_axis_x, new_axis_z)
    new_axis_y = new_axis_y / np.linalg.norm(new_axis_y)
    # print("new_axis_y的模", np.linalg.norm(new_axis_y))

    rotation_matrix = np.array([[new_axis_x[0], new_axis_y[0], new_axis_z[0]],
                                [new_axis_x[1], new_axis_y[1], new_axis_z[1]],
                                [new_axis_x[2], new_axis_y[2], new_axis_z[2]]])
    # print("预调整姿态的旋转矩阵: ", rotation_matrix)
    axis_pcd_2.rotate(rotation_matrix, center=attention_t_2)

    # 创建 Open3D 的 LineSet 对象来表示点和方向向量
    attention_normal_line_set = o3d.geometry.LineSet()
    attention_normal_line_set.points = o3d.utility.Vector3dVector(
        np.array([camera_coordinate, camera_coordinate - 0.5 * attention_normal]))  # +号是冲外，-号是冲里
    attention_normal_line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    o3d.visualization.draw_geometries(
        [lin_shi_ceshi_o3d_vs_pc_tmp, attention_normal_line_set, axis_pcd, axis_pcd_1,
         axis_pcd_2])  # , point_show_normal=True)


def two_dimensions_grasp_wrench(two_dimensoions_grasp_mask):
    # print("two_dimensoions_grasp_mask", two_dimensoions_grasp_mask)
    two_dimensoions_grasp_contours, two_dimensoions_grasp_cnt = cv2.findContours(two_dimensoions_grasp_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    moment_2d_grasp_contours = cv2.moments(two_dimensoions_grasp_contours[0])  # 计算第一条轮廓的各阶矩,字典形式，根据自己的激光点效果寻找轮廓
    moment_2d_grasp_center_x = int(moment_2d_grasp_contours["m10"] / moment_2d_grasp_contours["m00"])
    moment_2d_grasp_center_y = int(moment_2d_grasp_contours["m01"] / moment_2d_grasp_contours["m00"])

    return two_dimensoions_grasp_contours, moment_2d_grasp_center_x, moment_2d_grasp_center_y


def pose_status_callback(status):
    global pose_adjustment_status
    print("status", status.data)
    pose_adjustment_status = status.data


def grasp_status_callback(status):
    global grasp_status_info
    print("status", status.data)
    grasp_status_info = status.data


def laser_waiter_status_callback(status):
    global laser_callback_status
    print("激光等待完发送回信儿了")
    laser_callback_status = status.data


def yolov5_detect(img_color, save_img=False):
    laser_tmp_state = 0  # 设置激光响应状态，如果未获得激光为0，获得激光为1
    laser_tmp_center = []  # 初始花激光中心点

    # yolov5_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(yolov5_names))]
    yolov5_colors = [[180, 105, 255]]   # 浅粉色 RGB=(255,182,193) 桃粉色 RGB=(255,192,203) 亮粉色 RGB=(255,105,180) 桃红色 RGB=(255,139,254) 嫩粉色 RGB=(244,198,200)
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
                # yolov5_label = 'name: %s' % (yolov5_names[int(cls)])  # 最后取平均值为目标深度，深度已经删除了，只有类别的名字
                yolov5_label = '%s' % (yolov5_names[int(cls)])  # 最后取平均值为目标深度，深度已经删除了，只有类别的名字
                if laser_callback_status == "laser_back_is_waiting" :
                    plot_one_box(xyxy, img_color, label=yolov5_label, color=yolov5_colors[int(cls)], line_thickness=2)  # 将结果框打印回原图
                # c1:x[0]x[1]-------c4:x[2]x[1]
                #        |              |
                # c3:x[0]x[3]-------c2:x[2]x[3]
                if yolov5_names[int(cls)] == "laser":
                    laser_tmp_center.append([int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2)])
                    laser_tmp_state = 1

    return laser_tmp_state, laser_tmp_center   # 返回激光识别信息和激光中心点信息


if __name__ == '__main__':

    # 姿态预调整状态的订阅信息，抓取结束，返回ok状态执行下一步工作
    pose_adjust_status = rospy.Subscriber('/auto_grasp/pose_adjustment', String, pose_status_callback)
    # 抓取状态的订阅信息，抓取结束，返回ok状态执行下一步工作
    grasp_execute_status = rospy.Subscriber('/auto_grasp/grasp_status', String, grasp_status_callback)
    laser_back_wait_status = rospy.Subscriber('/auto_grasp/laser_waiter_back', String, laser_waiter_status_callback)

    # 定义一些激光的全局变量
    seg_input_point_laser = []  # 定义语义分割的输入点（激光的）
    seg_input_label_laser = []  # 定义语义分割的输入标签，包括前景点、背景点
    seg_input_stop_laser = False  # 定义语义分割的输入状态
    seg_input_point_laser_tmp = []  # 定义临时变量
    seg_input_label_laser_tmp = []

    # 定义一些全局变量
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
    S_optimal = []
    profiler = Profiler()  # 时间规划器、记录整体运行时间
    profiler.start()
    k = 0  # 定义循环次数
    k_instrin_in = [[605.471, 0.0, 327.157], [0.0, 604.936, 247.894], [0.0, 0.0, 1.0]]  # 相机内参

    # yolov5的参数设置
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='../yolov5/yolov5s_laser2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../yolov5/inference/images',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='object confidence threshold')
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
    yolov5_imgsz = check_img_size(yolov5_imgsz, s=yolov5_model.stride.max())  # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
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
            # return 1相机内参，2深度参数，3BRG图，4深度图，5深度彩色映射图，6对齐的彩色帧，7对齐的深度帧
            color_intrin, depth_intrin, img_color, img_depth, depth_mapped_image, aligned_color_frame, \
                aligned_depth_frame, d_to_c_extrin, c_to_d_extrin = get_aligned_images()

            # img_color_222 = copy.deepcopy(img_color)
            with torch.no_grad():   # yolov5_detect() 启动yolov5检测，并提取出激光点出现的状态和信息
                laser_status, laser_center = yolov5_detect(img_color)
                if laser_status:
                    laser_status = False
                    for center in laser_center:
                        # print("激光交互中")
                        seg_input_point_laser_tmp.append(center)   # 收集激光点，生成数组
                        seg_input_label_laser_tmp.append(1)  # 1表示前景点
            # 如果没有状态启动,并且激光点收集大于100次, 通过GMM对激光点期望值进行计算，求取最优激光点
            # print("len(np.array(seg_input_label_laser_tmp))", len(np.array(seg_input_label_laser_tmp)))
            """
            stop信息初始化为False, 当收集超过100个激光点，进入if处理，对收集的100个激光点清空保留center,并将处理以后的信息状态发布出去，
            此时，stop状态为True,if即使收集超过100个激光点也进不来，并进入else进行清理。当激光信号等待回来，进入下一阶段姿态预处理，stop
            重新变为false,
            """
            seg_input_stop_laser = True
            if not seg_input_stop_laser and len(np.array(seg_input_label_laser_tmp)) > 300:
                print("激光交互模块")
                # 添加GMM信号处理方法
                center_mean = gmm(seg_input_point_laser_tmp)
                print("seg_input_point_laser_tmp", seg_input_point_laser_tmp)
                # center_mean_1 = np.mean(np.array(seg_input_point_laser_tmp), axis=0)
                center_mean_1 = np.array(seg_input_point_laser_tmp)[299]
                print("GMM预测后的结果是：", center_mean)
                print("没有GMM预测后的结果是：", center_mean_1)
                center_x_mouse = int(center_mean[0])
                center_y_mouse = int(center_mean[1])
                seg_input_stop_laser = True    # 重置激光点处理状态
                seg_input_label_mouse.append(1)
                seg_input_stop_mouse = False    # 进入下一个基于点的姿态处理过程
                del seg_input_point_laser_tmp[:]    # 重置激光信息收集状态
                del seg_input_label_laser_tmp[:]
                if laser_callback_status == "laser_back_is_waiting":   # 等待发布时候,准备发布
                    print("激光信息收集完毕进行发布")
                    laser_topic_status = String()
                    laser_topic_status.data = "laser_is_ok"
                    laser_pub.publish(laser_topic_status)
                    rospy.loginfo("laser_topic_status is published: %s", laser_topic_status)
            elif laser_callback_status == "laser_back_is_working":   # 进入工作后，进来多少激光点都删掉
                # print("模块2")
                del seg_input_point_laser_tmp[:]    # 重置激光信息收集状态
                del seg_input_label_laser_tmp[:]

            # print("图像运行")

            # 2.鼠标交互, 鼠标中点左键就是前景点1，鼠标中点右键就是后景点标签0。
            if (seg_input_stop_mouse and len(np.array(seg_input_label_mouse)) > 0) or (laser_callback_status == "laser_back_is_ok"):
                print("模块3")
                if seg_input_stop_laser:
                    print("激光交互信号接入")
                    seg_input_stop_laser = False  # 重置激光点处理状态
                    laser_callback_status = "laser_back_is_working"
                else:
                    print("屏幕交互信号接入")
                    laser_callback_status = "laser_back_is_working"

                decision = input("是否为想要选择的对象，想要输入:y,想删除输入:n:")
                if decision == "y":
                    print("关注点为：", [center_x_mouse, center_y_mouse])
                    seg_input_stop_mouse = False  # 重置进入该判断的状态
                    lin_shi_ceshi_o3d_vs_pc = get_pointcloud(img_color, img_depth)  # 获取并保存局部点云图像
                    # o3d.visualization.draw_geometries([lin_shi_ceshi_o3d_vs_pc])
                    # lin_shi_ceshi_o3d_vs_pc.paint_uniform_color([0,1,0])
                    pc = rs.pointcloud()
                    pc.map_to(aligned_color_frame)
                    points = pc.calculate(aligned_depth_frame)
                    vtx = np.asanyarray(points.get_vertices())
                    npy_vtx = np.zeros((len(vtx), 3), float)
                    for i in range(len(vtx)):
                        npy_vtx[i][0] = np.float64(vtx[i][0])
                        npy_vtx[i][1] = np.float64(vtx[i][1])
                        npy_vtx[i][2] = np.float64(vtx[i][2])

                    pcd_pc_vs_o3d = o3d.geometry.PointCloud()
                    pcd_pc_vs_o3d.points = o3d.utility.Vector3dVector(npy_vtx)
                    # points.paint_uniform_color([1,0,0])
                    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])  # 坐标轴
                    # o3d.visualization.draw_geometries([pcd_pc_vs_o3d, lin_shi_ceshi_o3d_vs_pc, axis_pcd])
                    # 设置法线长度
                    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                    lin_shi_ceshi_o3d_vs_pc.estimate_normals(search_param=search_param)
                    # print(seg_input_point)
                    # print("center_x_mouse, center_y_mouse", center_x_mouse, center_y_mouse)

                    dis, camera_coordinat = get_3d_camera_coordinate([center_x_mouse, center_y_mouse],
                                                                     aligned_depth_frame,
                                                                     depth_intrin)
                    camera_coordinate = np.array(camera_coordinat)
                    # print("鼠标x，y", [center_x_mouse, center_y_mouse])
                    normals_set = np.asarray(lin_shi_ceshi_o3d_vs_pc.normals)
                    points_set = np.asarray(lin_shi_ceshi_o3d_vs_pc.points)
                    # print("法线的个数", len(np.asarray(lin_shi_ceshi_o3d_vs_pc.normals)))
                    # print("点的坐标", np.asarray(lin_shi_ceshi_o3d_vs_pc.points))
                    # 使用 np.where() 来找到一维数组在二维数组中的位置
                    matches = np.where((points_set == camera_coordinate).all(axis=1))  # 找到关注点的对象的法线
                    matches_index = 0
                    if len(matches[0]) > 0:
                        print(f"一维数组在二维数组中的索引为：{matches[0][0]}")
                    else:
                        print("无法直接找到法线的方向，需要进一步处理")
                        new_points_set = points_set - camera_coordinate
                        # print("新点集是：", new_points_set)
                        new_points_set_norm = np.linalg.norm(new_points_set, axis=1)
                        # print("求范数后的新点集", new_points_set_norm)
                        matches_index = np.argmin(new_points_set_norm)
                        matches_min = np.min(new_points_set_norm)
                        # print(f"最小值是：{matches_min}")
                        # print(f"一维数组在二维数组中的索引为：{matches_index}")

                    attention_normals = np.array(normals_set[matches_index])  # 关注点处的法线是这个
                    print("关注点法线的方向三维向量", attention_normals)
                    print("关注点的三维坐标_相对于相机坐标系", camera_coordinate)
                    prompt_info.normal = attention_normals
                    prompt_info.coordinate = camera_coordinate
                    print(prompt_info)
                    prompt_pub.publish(prompt_info)
                    rospy.loginfo("Prompt is published: %s", prompt_info)
                    # show_pose_o3d_pcl(lin_shi_ceshi_o3d_vs_pc, camera_coordinate, attention_normals)    # 展示调整后的坐标系姿态

                    """
                    计算相机坐标系变换后，激光点相对应的新像素点用于图像分割
                    """
                    new_camera_frame_coord = camera_coordinate
                    new_camera_frame_normal = attention_normals
                    new_axis_z = new_camera_frame_normal  # 提示点处的法线方向，该法线方向为原坐标系的Z轴方向
                    attention_mol = np.linalg.norm(new_camera_frame_coord)  # 求解距离范数
                    attention_t_2 = new_camera_frame_coord - attention_mol * new_camera_frame_normal  # 求解新的相机坐标原点 （圆弧末端点） （0，0，0 圆弧起始点）
                    # attention_q_old = vector_to_quaternion(new_camera_frame_normal)  # 求解法线的四元数
                    # print("attention_q_old:", attention_q_old)
                    new_axis_x = np.zeros(3)
                    new_axis_x = np.cross(new_camera_frame_coord / attention_mol, new_axis_z)
                    new_axis_x = new_axis_x / np.linalg.norm(new_axis_x)
                    new_axis_y = np.zeros(3)
                    new_axis_y = -np.cross(new_axis_x, new_axis_z)
                    new_axis_y = new_axis_y / np.linalg.norm(new_axis_y)

                    attention_r_matrix = np.array([[new_axis_x[0], new_axis_y[0], new_axis_z[0]],
                                                   [new_axis_x[1], new_axis_y[1], new_axis_z[1]],
                                                   [new_axis_x[2], new_axis_y[2], new_axis_z[2]]])
                    # print("预调整姿态的旋转矩阵: ", attention_r_matrix)
                    attention_matrix = np.eye(4)  # 这个矩阵存在很大的问题20240301！回答：经测试不存在问题20240319
                    # print("矩阵的转置", attention_r_matrix.T)
                    # print("矩阵的逆矩", np.linalg.inv(attention_r_matrix))
                    attention_matrix[:3, :3] = attention_r_matrix  # 2024/03/01 现在这个代码好像是对的
                    attention_matrix[:3, 3] = np.array(attention_t_2)  # 得加个括号，双括号才行, 好像不加也行

                    attention_matrix_inv = np.eye(4)
                    attention_matrix_inv[:3, :3] = attention_r_matrix.T
                    attention_matrix_inv[:3, 3] = -1 * np.matmul(attention_r_matrix.T, np.array(attention_t_2))
                    camera_coordinate_with_1 = np.ones(4)
                    camera_coordinate_with_1[0:3] = camera_coordinate
                    print("camera_coordinate_with_1", camera_coordinate_with_1)
                    camera_coordinate_base_on_new_frame = np.matmul(attention_matrix_inv, camera_coordinate_with_1)
                    print("原始相机坐标系的像素点", [center_x_mouse, center_y_mouse])
                    new_attention_pixel = rs.rs2_project_point_to_pixel(color_intrin, camera_coordinate_base_on_new_frame[:3])
                    print("变换后相机坐标系像素点", new_attention_pixel)
                    color_by_intrinsic = np.matmul(k_instrin_in, np.array(camera_coordinate).T)
                    # print("color_by_intrinsic", color_by_intrinsic)
                    color_point = rs.rs2_transform_point_to_point(d_to_c_extrin, camera_coordinate)  #
                    color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
                    # print("三维转回来后的点：", color_pixel)

                else:
                    del seg_input_point_mouse[:]
                    del seg_input_label_mouse[:]
                    center_x_mouse = 0.0
                    center_y_mouse = 0.0
                    seg_input_stop_mouse = False  # 定义语义分割的输入状态
                    laser_callback_status = "laser_back_is_waiting"

            if pose_adjustment_status == "ok":    # ！！！！！需要加一些信息，看看center在这个过程中的保持程度
                # print("等1s，让延迟的图像进来")
                time.sleep(1)
                print("进入SAM分割阶段")
                input_point = [[int(new_attention_pixel[0]), int(new_attention_pixel[1])]]
                print("变换后的输入点", input_point)
                input_label = [1]
                input_bool = True  # 定义语义分割的输入状态
                thread_sam = MyThread(SAM_wrapper, (img_color,
                                                    input_point,
                                                    input_label,
                                                    input_bool))
                # print("seg_input_label_mouse", seg_input_label_mouse)
                thread_sam.start()
                thread_sam.join()
                sam_masks, sam_scores, sam_logits = thread_sam.get_result()
                seg_tmp_masks = sam_masks[0]    # 0,1,2分割的逐渐变大
                local_depth = apply_pointcloud_mask(img_depth, seg_tmp_masks)
                local_pcl_points = get_pointcloud(img_color, local_depth)  # 获取并保存局部点云图像
                # o3d.visualization.draw_geometries([local_pcl_points, attention_normal_line_set, axis_pcd])
                o3d.visualization.draw_geometries([local_pcl_points, axis_pcd])

                del seg_input_point_mouse[:]
                del seg_input_label_mouse[:]

                seg_input_stop_mouse = False  # 定义语义分割的输入状态
                print("分割完毕")

            if len(seg_tmp_masks) != 0:
                # print("目标图像掩码上色")
                # color = tuple(np.array([255, 255, 255]).tolist())
                color = tuple(np.random.randint(0, 256, 3).tolist())  # 设置颜色随机
                img_color = apply_color_mask(img_color, seg_tmp_masks, color)  # 为实例分割物品赋予阴影
                # img_white_color = apply_white_mask(img_color, seg_tmp_masks, color)   # 为实例分割物品赋予阴影
                seg_tmp_masks = np.where(seg_tmp_masks == True, 1, 0)  # 改成黑白图， 有阴影部分为白1，无阴影部分为黑0
                seg_tmp_masks = np.array(seg_tmp_masks, np.uint8)  # 改为int8
                grasp_2d_wrench_contours, grasp_2d_wrench_contours_center_x, grasp_2d_wrench_contours_center_y = two_dimensions_grasp_wrench(
                    seg_tmp_masks)  # 求解阴影的边界和圆心
                grasp_2d_wrench_contours_reshape = np.reshape(grasp_2d_wrench_contours[0], (-1, 2))
                grasp_2d_wrench_contours_center = np.array(
                    [grasp_2d_wrench_contours_center_x, grasp_2d_wrench_contours_center_y])
                # np.savetxt("grasp_stability/txt/boundary.txt", grasp_2d_wrench_contours_reshape, fmt='%s')
                # np.savetxt("grasp_stability/txt/center.txt", grasp_2d_wrench_contours_center, fmt='%s')
                # print(grasp_2d_wrench_contours_reshape)
                # print(grasp_2d_wrench_contours_center)

                cv2.drawContours(img_color, grasp_2d_wrench_contours, 0, color, 2)  # 绘制轮廓，填充（图像，轮廓，轮廓序号，颜色，轮廓线粗细）

                # 将这两个if全部注释可以隐藏抓取
                if pose_adjustment_status == "ok" and len(S_optimal) == 0:
                    print("力闭合分析")
                    q_max, W_max, S_optimal, S_rotate_degree = grasp_wrench_2d(grasp_2d_wrench_contours_reshape,
                                                                               grasp_2d_wrench_contours_center)
                    S_optimal = S_optimal.astype(np.int32)

                if len(S_optimal) != 0:
                    color_2 = tuple(np.random.randint(0, 256, 3).tolist())  # 设置颜色随机
                    cv2.line(img_color, S_optimal[0], S_optimal[1], color_2, 1, cv2.LINE_AA)
                    cv2.circle(img_color, S_optimal[0], 8, color_2, -1)
                    cv2.circle(img_color, S_optimal[1], 8, color_2, -1)

                if pose_adjustment_status == "ok" and len(S_optimal) != 0:
                    print("计算抓取点")
                    print("左侧接触像素点：", S_optimal[1])    # 旋转回来后，S_optimal[1]变成了左侧接触点，转过来了
                    print("右侧接触像素点：", S_optimal[0])
                    # cv2.arrowedLine(img_color, S_optimal[0] + S_rotate_degree, S_optimal[0], color=(0, 255, 0), thickness=2, tipLength=0.3)
                    dis_left, touch_left = get_3d_camera_coordinate([S_optimal[0][0] - 18, S_optimal[0][1]],
                                                                    # 先加后减能让左点更靠里
                                                                    aligned_depth_frame,
                                                                    depth_intrin)
                    # print("dis_left", dis_left, "m")
                    print("左侧接触实际三维点", touch_left, "m")
                    dis_right, touch_right = get_3d_camera_coordinate([S_optimal[1][0] + 18, S_optimal[1][1]],
                                                                      aligned_depth_frame,
                                                                      depth_intrin)
                    # print("dis_right", dis_right, "m")
                    print("右侧接触实际三维点", touch_right, "m")
                    camera_xyz = np.array(touch_left + touch_right) / 2  # 收集相机坐标系下中心坐标点的xyz坐标

                    angle_vector = S_optimal[1] - S_optimal[0]  # 求解旋转角度的向量
                    angle = 0
                    # 求解需要旋转的角度
                    if angle_vector[0] == 0 and angle_vector[1] > 0:  # 角度为正无穷
                        angle = np.pi / 2
                    elif angle_vector[0] == 0 and angle_vector[1] < 0:  # 角度为负无穷
                        angle = -np.pi / 2
                    elif angle_vector[0] == 0 and angle_vector[1] == 0:  # 这意味着两个点重合了
                        angle = 0
                    elif angle_vector[0] != 0:
                        oriented_vector = np.array([angle_vector[1] / angle_vector[0]])
                        angle = np.arctan(oriented_vector)  # 如果求解结果为负，机械手末端应该逆时针旋转。如果求解结果为正，机械手末端应该顺时针旋转。
                        print("机械臂在抓取之前应该旋转的角度为：", angle)

                    touch_left.append(1)
                    touch_right.append(1)
                    if (touch_left[2] or touch_right[2]) > 1.5:
                        print("touch_left[2]", touch_left[2])
                        print("touch_right[2]", touch_right[2])
                        break

                    print("订阅之前的grasp_info", grasp_info)
                    grasp_info.left_touch = touch_left
                    grasp_info.right_touch = touch_right
                    grasp_info.angle_touch = angle
                    print("发布抓取信息")
                    print("发布之后的grasp_info", grasp_info)
                    time.sleep(1)
                    grasp_pub.publish(grasp_info)
                    # rospy.loginfo("grasp is published: %s %s %s",
                    #               grasp_info.left_touch, grasp_info.right_touch, grasp_info.angle_touch)
                    pose_adjustment_status = ""
                    cv2.imwrite("history/grasp_{}.png".format(time_time), img_color)

            if grasp_status_info == "ok":
                print("grasp status is ok")
                seg_tmp_masks = []
                laser_callback_status = "laser_back_is_waiting"
                center_x_mouse = 0.0
                center_y_mouse = 0.0
                grasp_status_info = ""
                S_optimal = []

            # if move_status == "ok":
            #     print("ok", ok)

            time_time = time.time()
            # cv2.imshow("Scene", np.hstack((img_color, img_color_222)))  # 展示彩色图像和深度图像
            cv2.imshow("Scene", img_color)  # 展示图像
            # cv2.imwrite("laser/edge_{}.png".format(time_time), img_color)
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
