# coding:utf-8
#
# laser_SAM.py.py
#
#  Created on: 2023/5/24
#      Author: Tex Yan Liu
#
# description:

import cv2
import json  # 作用未知
import threading
import numpy as np
import quaternion
from segment_anything import sam_model_registry, SamPredictor  # 被点图片识别
import pyrealsense2 as rs
import open3d as o3d
from pyinstrument import Profiler
import copy
from scipy.cluster.vq import kmeans, vq  # 聚类

# 定义realsense相机的API
pipeline = rs.pipeline()  # 创建管道-这是流媒体和处理帧的顶级API 该管道简化了用户与设备和计算机视觉处理模块的交互。
config = rs.config()  # 该配置允许管道用户为管道流以及设备选择和配置请求过滤器。
pipeline_wrapper = rs.pipeline_wrapper(pipeline)  # 管道握手函数
pipeline_profile = config.resolve(pipeline_wrapper)  # 管道配置
rgbd_device = pipeline_profile.get_device()  # 获取设备
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
    global seg_input_point_mouse, seg_input_label_mouse, seg_input_stop_mouse, center_x_mouse, center_y_mouse  # 全局变量，输入点，响应信号
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
    #print("d435_depth_intrin", d435_depth_intrin)
    #print("d435_color_intrin", d435_color_intrin)

    d435_depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(d435_img_depth, alpha=0.03), cv2.COLORMAP_JET)

    # return 1相机内参，2深度参数，3BRG图，4深度图，5深度彩色映射图，6对齐的彩色帧，7对齐的深度帧
    return d435_color_intrin, d435_depth_intrin, d435_img_color, d435_img_depth, d435_depth_mapped_image, \
        d435_aligned_color_frame, d435_aligned_depth_frame


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
    # h1, w1 = red_point_mask3_open.shape
    # 寻找边缘
    laser_tmp_contours, cnt = cv2.findContours(red_point_mask3_open.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(laser_tmp_contours) > 0:  # 轮廓总数超过0
        if len(laser_tmp_contours) == 1:
            moment_laser_tmp_contours = cv2.moments(laser_tmp_contours[0])  # 计算第一条轮廓的各阶矩,字典形式，根据自己的激光点效果寻找轮廓
        else:
            moment_laser_tmp_contours = cv2.moments(laser_tmp_contours[1])  # 计算第一条轮廓的各阶矩,字典形式，根据自己的激光点效果寻找轮廓
        laser_tmp_center_x = int(moment_laser_tmp_contours["m10"] / moment_laser_tmp_contours["m00"])
        laser_tmp_center_y = int(moment_laser_tmp_contours["m01"] / moment_laser_tmp_contours["m00"])
        # mask_open = np.zeros([h1, w1], dtype=mask_open.dtype)   # 应该是用于复制掩码图像的
        laser_tmp_state = 1  # 设置状态，用于后续确认

    return laser_tmp_state, laser_tmp_contours, laser_tmp_center_x, laser_tmp_center_y


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
    kmeancluster_tmp_pointcloud.estimate_normals()
    print("当前点云的法线", np.asarray(kmeancluster_tmp_pointcloud.normals))
    centroids, distortion = kmeans(kmeancluster_tmp_pointcloud.normals, 6)  # 获得了三个聚类向量，但是暂时依旧无法根据这三个聚类向量获得
    print("聚类中心", centroids)  # 聚类中心理论上应该就是我们想要的方向
    print("点云的点", np.asarray(kmeancluster_tmp_pointcloud.points))
    print("聚类中心的数据类型", centroids.shape)
    labels, _ = vq(kmeancluster_tmp_pointcloud.normals, centroids)
    print("样本分类结果：", labels)
    sum_1 = np.sum(np.where(labels, 0, 1))
    print("类别0总数", sum_1)  # 计数类别0
    sum_2 = np.sum(np.where(labels, 1, 1))
    print("类别1总数", sum_2)  # 计数类别1
    max_support_vector_1 = []
    if sum_2 < sum_1:
        max_support_vector_1 = centroids[0]
        tmp_max_support_vector_2 = centroids[1]
    else:
        max_support_vector_1 = centroids[1]
        tmp_max_support_vector_2 = centroids[0]

    max_support_vector_2 = np.cross(max_support_vector_1, tmp_max_support_vector_2)  # 首先通过来两个最优方向叉乘计算出另一个垂直方向
    max_support_vector_3 = np.cross(max_support_vector_1, max_support_vector_2)  # 在根据这个新的垂直方向计算出最后一个垂直方向
    print("点云的点的数据类型", np.asarray(kmeancluster_tmp_pointcloud.points).shape)
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

    lines = [[0,1],[0,2],[1,3],[2,3],[0,4],[1,5],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]]
    line_pcd = o3d.geometry.LineSet()
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.colors = o3d.utility.Vector3dVector([[0,0,0] for i in range(len(lines))])
    line_pcd.points = o3d.utility.Vector3dVector(vertices_set)

    return line_pcd, centroids

def get_pointcloud(pointcloud_tmp_color_rs, pointcloud_tmp_depth_rs, pointcloud_tmp_name,
                   pointcloud_tmp_num):  # 对齐后的彩色图和深度图作为输入，不是彩色帧的数组和深度帧的数组
    # 因为open3d处理的是RGB的，而realsense出来的是BGR的，需要在此处转换以下颜色通道
    pointcloud_tmp_color_rs = cv2.cvtColor(pointcloud_tmp_color_rs, cv2.COLOR_BGR2RGB)
    pointcloud_tmp_color_rs = o3d.geometry.Image(pointcloud_tmp_color_rs)
    pointcloud_tmp_depth_rs = o3d.geometry.Image(pointcloud_tmp_depth_rs)
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
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    if pointcloud_tmp_name == "global_pcl":
        passthrough_bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1.3, -1.3, 0),
                                                                       max_bound=(1.3, 1.3, 1.3))    # 直通滤波全局点云
        open3d_process_pointcloud = open3d_process_pointcloud.crop(passthrough_bounding_box)
        # o3d.visualization.draw_geometries([open3d_process_pointcloud])

    if pointcloud_tmp_name == "local_pcl_by_mouse":  # 来到这里之前先在Mask上腐蚀一下，
        cl1, ind1 = open3d_process_pointcloud.remove_statistical_outlier(nb_neighbors=80, std_ratio=0.5)
        open3d_process_pointcloud = open3d_process_pointcloud.select_by_index(ind1)
        cl2, ind2 = open3d_process_pointcloud.remove_radius_outlier(nb_points=60, radius=1.5)
        open3d_process_pointcloud = open3d_process_pointcloud.select_by_index(ind2)
        open3d_process_pointcloud = open3d_process_pointcloud.uniform_down_sample(5)  # 均匀下采样，在之后的实验中可以通过调节点云下采样的密度进行实验
        #--------在这里需要再补充一个聚类分割，提取更大块目标点云

    # 计算相对于机械臂末端的点云
    end_rotation = open3d_process_pointcloud.get_rotation_matrix_from_quaternion((0.12934569723181133,
                                                                                  0.01040190109549791,
                                                                                  0.0054398078540793815,
                                                                                  0.9915300800034894))

    open3d_process_pointcloud_end = copy.deepcopy(open3d_process_pointcloud)
    open3d_process_pointcloud_end = open3d_process_pointcloud_end.rotate(end_rotation, center=(0, 0, 0))
    open3d_process_pointcloud_end = open3d_process_pointcloud_end.translate((0.0548432,
                                                                             0.0871482,
                                                                             -0.146589))
    # 计算相对于机械臂基座的点云，该坐标变换需要根据KINOVA_API_tool_pose调用而定
    base_rotation = open3d_process_pointcloud_end.get_rotation_matrix_from_quaternion((0.2496097981929779,
                                                                                       0.8791682124137878,
                                                                                       0.35243546962738037,
                                                                                       0.20136378705501556))
    open3d_process_pointcloud_base = copy.deepcopy(open3d_process_pointcloud_end)
    open3d_process_pointcloud_base = open3d_process_pointcloud_base.rotate(base_rotation, center=(0, 0, 0))
    open3d_process_pointcloud_base = open3d_process_pointcloud_base.translate((-0.013422133401036263,
                                                                               -0.2824711799621582,
                                                                               0.508553147315979))

    if pointcloud_tmp_name == "local_pcl_by_mouse":  # 来到这里之前先在Mask上腐蚀一下，
        open3d_process_pointcloud_bounding_box_min_obb = open3d_process_pointcloud_base.get_minimal_oriented_bounding_box(robust=True)
        open3d_process_pointcloud_bounding_box_min_obb.color = (0, 0, 1)
        open3d_process_pointcloud_bounding_box_obb = open3d_process_pointcloud_base.get_oriented_bounding_box(robust=True)
        open3d_process_pointcloud_bounding_box_obb.color = (1, 0, 0)
        open3d_process_pointcloud_bounding_box = open3d_process_pointcloud_base.get_axis_aligned_bounding_box()
        open3d_process_pointcloud_bounding_box.color = (0, 1, 0)

        line_pcd, centroids = kmeancluster_bounding_box(open3d_process_pointcloud_base)

        centroids = np.row_stack((centroids, [0,0,0]))
        lines1 = [[0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [5, 6]]
        line_pcd_1 = o3d.geometry.LineSet()
        line_pcd_1.lines = o3d.utility.Vector2iVector(lines1)
        line_pcd_1.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for i in range(len(lines1))])
        line_pcd_1.points = o3d.utility.Vector3dVector(centroids)

        print("open3d_process_pointcloud_bounding_box", open3d_process_pointcloud_bounding_box)
        print("open3d_process_pointcloud_bounding_box_obb.get_center()",
              open3d_process_pointcloud_bounding_box_min_obb.get_center())
        print("open3d_process_pointcloud_bounding_box.get_center()",
              open3d_process_pointcloud_bounding_box.get_center())
        print("open3d_process_pointcloud_bounding_box.get_max_bound()",
              open3d_process_pointcloud_bounding_box.get_max_bound())
        print("open3d_process_pointcloud_bounding_box.get_min_bound()",
              open3d_process_pointcloud_bounding_box.get_min_bound())

        mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.01,
                                                        height=0.02,
                                                        depth=0.03,
                                                        create_uv_map=True, map_texture_to_each_face=False)
        mesh_box.compute_vertex_normals()
        mesh_box.paint_uniform_color([1, 0, 0])
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])  # 坐标轴
        o3d.visualization.draw_geometries([open3d_process_pointcloud_base, open3d_process_pointcloud_bounding_box, open3d_process_pointcloud_bounding_box_obb, open3d_process_pointcloud_bounding_box_min_obb, axis_pcd, line_pcd], point_show_normal=True)
        # o3d.visualization.draw_geometries([open3d_process_pointcloud_base, axis_pcd, line_pcd, line_pcd_1], point_show_normal=True)

    get_npy_state = get_6d_pointnetplusplus_npy(open3d_process_pointcloud_base, pointcloud_tmp_num)  # 储存被选中点云的信息，return完成True状态

    o3d.io.write_point_cloud("pre_pointcloud/{}_camera_{}.ply".format(pointcloud_tmp_name, pointcloud_tmp_num),
                             open3d_process_pointcloud)
    o3d.io.write_point_cloud("pre_pointcloud/{}_end_{}.ply".format(pointcloud_tmp_name, pointcloud_tmp_num),
                             open3d_process_pointcloud_end)
    o3d.io.write_point_cloud("pre_pointcloud/{}_base_{}.ply".format(pointcloud_tmp_name, pointcloud_tmp_num),
                             open3d_process_pointcloud_base)

    # o3d.visualization.draw_geometries([open3d_process_pointcloud,
    #                                    open3d_process_pointcloud_end,
    #                                    open3d_process_pointcloud_base])  # 点云在open3d中的可视化

    return open3d_process_pointcloud, open3d_process_pointcloud_end, open3d_process_pointcloud_base


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
    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"  # sam的权重文件
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
    # print("eye_on_hand_r_matrix", eye_on_hand_r_matrix)
    # print("camera_tmp_coordinate", camera_tmp_coordinate)
    # # 坐标变换 np.matmul(a,b) 如果b是一维向量，可以自动调整，如果b是一维，就要调整到正常形式
    # print("np.matmul(eye_on_hand_r_matrix, camera_tmp_coordinate", np.matmul(eye_on_hand_r_matrix, camera_tmp_coordinate.T))
    # print("eye_on_hand_t_matrix.T", eye_on_hand_t_matrix.transpose())
    end_effector_tmp_coordinate = np.matmul(eye_on_hand_r_matrix, camera_tmp_coordinate.T) + eye_on_hand_t_matrix.T
    # print("end_effector_tmp_coordinate", end_effector_tmp_coordinate)
    # 手到基座平移矩阵，需要等待从KINOVA_API中回调
    end_effector_to_base_link_t_matrix = np.array([[-0.013422133401036263, -0.2824711799621582, 0.508553147315979]])
    # 手到基座四元数，需要等待从KINOVA_API中回调
    end_effector_to_base_link_q_matrix = np.quaternion(0.2496097981929779, 0.8791682124137878,
                                                       0.35243546962738037, 0.20136378705501556)
    # 手到基座旋转矩阵，自己转换就行
    end_effector_to_base_link_r_matrix = quaternion.as_rotation_matrix(end_effector_to_base_link_q_matrix)
    # 所点选物品基于基座的坐标
    base_link_tmp_coordinate = np.matmul(end_effector_to_base_link_r_matrix, end_effector_tmp_coordinate) + end_effector_to_base_link_t_matrix.T
    print("base_link_tmp_coordinate", base_link_tmp_coordinate)
    return end_effector_tmp_coordinate, base_link_tmp_coordinate  # 返回两个，一个是相对于手抓末端的坐标，一个是相对于基座，也是后续用于执行的坐标


def apply_color_mask(apply_color_tmp_image, apply_color_tmp_mask, apply_color_tmp_color):  # 对掩码进行赋予颜色
    color_dark = 0.4
    for c in range(3):
        # np.where(condition, x, y)  满足condition取x,不满足取y
        apply_color_tmp_image[:, :, c] = np.where(
            apply_color_tmp_mask == 1,
            apply_color_tmp_image[:, :, c] * (1 - color_dark) + color_dark * apply_color_tmp_color[c],
            apply_color_tmp_image[:, :, c])
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
    cv2.resizeWindow("Scene", 1280, 480)  # 调整界面尺寸
    cv2.setMouseCallback("Scene", mouse_click)  # 调用鼠标点击
    # cap = cv2.VideoCapture(4)  注释了这段对话，因为不用cv导入图像，改为realsense的sdk导入图像
    seg_tmp_masks = []  # 定义一个空mask用于存储被选中target目标的segment mask
    profiler = Profiler()  # 时间规划器、记录整体运行时间
    profiler.start()
    k = 0  # 定义循环次数
    try:
        while True:
            # _1, frame = cap.read()  # cv2读取图像
            color_intrin, depth_intrin, img_color, img_depth, depth_mapped_image, aligned_color_frame, \
                aligned_depth_frame = get_aligned_images()  # return 1相机内参，2深度参数，3BRG图，4深度图，5深度彩色映射图，6对齐的彩色帧，7对齐的深度帧
            # print("color_intrin", color_intrin)
            # print("depth_intrin", depth_intrin)
            # print("img_color", img_color)
            laser_state, contours, center_x_laser, center_y_laser = laser_point_process(img_color)  # 激光点处理
            laser_state = False  # 先把系统中的laser 交互断掉
            if laser_state:
                print("laser detection")
                if len(contours) == 1:
                    cv2.drawContours(img_color, contours, 0, (0, 255, 0), 2)  # 绘制轮廓，填充（图像，轮廓，轮廓序号，颜色，轮廓线粗细）
                else:
                    cv2.drawContours(img_color, contours, 1, (0, 255, 0), 2)  # 绘制轮廓，填充（图像，轮廓，轮廓序号，颜色，轮廓线粗细）
                cv2.circle(img_color, (center_x_laser, center_y_laser), 2, (0, 0, 255), -1)  # 绘制中心点
                dis, camera_coordinate = get_3d_camera_coordinate([center_x_laser, center_y_laser],
                                                                  aligned_depth_frame,
                                                                  depth_intrin)

                text = "laser (mm): {:.0f},{:.0f},{:.0f}".format(camera_coordinate[0] * 1000,
                                                                 camera_coordinate[1] * 1000,
                                                                 camera_coordinate[2] * 1000)
                cv2.putText(img_color, text, (center_x_laser - 30, center_y_laser - 20),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)
                seg_input_point_laser_tmp.append([center_x_laser, center_y_laser])
                seg_input_label_laser_tmp.append(1)  # 1表示前景点

            # 如果获得了激光语义超过70或者鼠标点击超过1次，开始对点击场景进行分割，生成蓝色
            # 1.激光交互，选择y就是前景点，选择n就是后景点，以后有了语义分类，在改成和鼠标一样的内容
            if not seg_input_stop_laser and len(np.array(seg_input_label_laser_tmp)) > 100:
                decision = input("激光交互：是否为想要选择的对象，想要输入:y,想删除输入:n,都不要输入任意键并回车:")
                if decision == "y" or decision == "n":
                    del seg_input_point_laser_tmp[0:70]
                    del seg_input_label_laser_tmp[0:70]
                    laser_label_dictionary = {"y": 1, "n": 0}
                    # del seg_tmp_masks[:]
                    seg_input_point_laser.append(seg_input_point_laser_tmp[0])
                    seg_input_label_laser.append(laser_label_dictionary.get(decision))
                    # print("24595059")
                    # print("seg_input_label_laser", seg_input_label_laser)
                    thread_sam = MyThread(SAM_wrapper, (img_color,
                                                        seg_input_point_laser,
                                                        seg_input_label_laser,
                                                        seg_input_stop_laser))
                    thread_sam.start()
                    thread_sam.join()
                    sam_masks, sam_scores, sam_logits = thread_sam.get_result()
                    # sam_masks, sam_scores, sam_logits, sam_all_masks = thread_sam.get_result()
                    # print("mask[2]", masks[2])
                    seg_tmp_masks = sam_masks[2]
                    del seg_input_point_laser_tmp[:]
                    del seg_input_label_laser_tmp[:]
                    local_depth = apply_pointcloud_mask(img_depth, seg_tmp_masks)  # 获取局部点云
                    local_pcl_points = get_pointcloud(img_color, local_depth, "local_pcl_by_laser", k)  # 保存局部点云
                    global_pcl_points = get_pointcloud(img_color, img_depth, "global_pcl", k)  # 获取全局点云

                else:
                    del seg_input_point_laser_tmp[:]
                    del seg_input_label_laser_tmp[:]
                    del seg_input_point_laser[:]
                    del seg_input_label_laser[:]
                    seg_tmp_masks = []
            # 2.鼠标交互, 鼠标中点左键就是前景点1，鼠标中点右键就是后景点标签0。
            elif seg_input_stop_mouse and len(np.array(seg_input_label_mouse)) > 0:
                decision = input("鼠标交互：是否为想要选择的对象，想要输入:y,想删除输入:n:")
                if decision == "y":
                    # print(seg_input_point)
                    # print("center_x_mouse, center_y_mouse", center_x_mouse, center_y_mouse)
                    dis, camera_coordinate = get_3d_camera_coordinate([center_x_mouse, center_y_mouse],
                                                                      aligned_depth_frame,
                                                                      depth_intrin)
                    end_effector_coordinate, base_link_coordinate = eye_on_hand_matrix(camera_coordinate)
                    thread_sam = MyThread(SAM_wrapper, (img_color,
                                                        seg_input_point_mouse,
                                                        seg_input_label_mouse,
                                                        seg_input_stop_mouse))
                    # print("seg_input_label_mouse", seg_input_label_mouse)
                    thread_sam.start()
                    thread_sam.join()
                    sam_masks, sam_scores, sam_logits = thread_sam.get_result()
                    # sam_masks, sam_scores, sam_logits, sam_all_masks = thread_sam.get_result()
                    # print("sam_all_masks", sam_all_masks)
                    # print("mask[2]", masks[2])
                    seg_tmp_masks = sam_masks[2]
                    #  print("seg_tmp_masks", seg_tmp_masks)
                    # seg_tmp_masks = open_mor(np.array(seg_tmp_masks))  # 开闭运算滤波还没有调整好，Mask是True false格式和函数是0，255格式
                    seg_input_stop_mouse = False  # 定义语义分割的输入状态
                    local_depth = apply_pointcloud_mask(img_depth, seg_tmp_masks)
                    local_pcl_points = get_pointcloud(img_color, local_depth, "local_pcl_by_mouse", k)  # 获取并保存局部点云图像
                    global_pcl_points = get_pointcloud(img_color, img_depth, "global_pcl", k)  # 获取并保存全局点云图像
                    tmp_afford = grasp_affordance()

                    # -----局部点云后续需要添加滤波 （调参问题，不着急）
                    # -----相机下坐标转换为机械臂基座坐标，便于后续判断上下左右空间关系（完成坐标变换）（上下空间问题根据相对手臂末端的坐标进行计算）
                    # -----沿着Target物体分为14个区域（8个象限，6个轴区域）（基于上条）
                    # -----基于Seg的二维像素点选择特征点，计算紧挨着的物品和target物品的距离，判断其空间距离，根据该点和target物品的点结合上述坐标系计算方位，并结合手抓结构尺寸判断是否为障碍物。（还没有思路）
                    # -----遍历剩余物品，计算空间距离和方位，根据该点和target物品的点结合上述坐标系计算方位，并结合手抓结构尺寸判断是否为障碍物。（还没有思路）
                    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
                    # -----基于目标物品计算无障碍方位。（基于上述）
                    # -----基于目标物品为圆心旋转至无障碍方位。（基于上述）
                    # -----确定该方位为最优方位，存在阈值。（阈值问题还没有思路）
                    # -----计算力闭合抓取位姿，这个地方基本就没什么太复杂的东西。（还不会计算）（现在有个凑合的6DPointnet++）
                    # -----前进抓取，原路返回。（比较容易）

                    print("Everything processed")

                else:
                    del seg_input_point_mouse[:]
                    del seg_input_label_mouse[:]
                    seg_tmp_masks = []
                    seg_input_stop_mouse = False  # 定义语义分割的输入状态

            if len(seg_tmp_masks) != 0:
                # color = tuple(np.array([234, 152, 15]).tolist())
                color = tuple(np.random.randint(0, 256, 3).tolist())  # 设置颜色随机
                img_color = apply_color_mask(img_color, seg_tmp_masks, color)
                # img_color = apply_color_mask(img_color, seg_tmp_masks, color)

            if k != 0:  # 如果是第一次，则不需要局部点云配准，否则需要进行局部配准
                A = 1

            cv2.imshow("Scene", np.hstack((img_color, depth_mapped_image)))  # 展示图像
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
