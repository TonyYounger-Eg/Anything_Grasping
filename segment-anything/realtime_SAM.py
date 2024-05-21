# coding:utf-8
#
# realtime_SAM.py
#  Created on: 2023/5/13
#      Author: Tex Yan Liu
# description: 用于SAM实时检测测试

import cv2
import threading
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import pyrealsense2 as rs
import open3d as o3d
from pyinstrument import Profiler

sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"  # sam的全中文件
model_type = "vit_h"  # 模型类型
device = "cuda"  # 应用 cuda
input_point = []
input_label = []
input_stop = False
pipeline = rs.pipeline()  # 创建管道-这是流媒体和处理帧的顶级API 该管道简化了用户与设备和计算机视觉处理模块的交互。
config = rs.config()  # 该配置允许管道用户为管道流以及设备选择和配置请求过滤器。
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
rgbd_device = pipeline_profile.get_device()
found_rgb = False
for s in rgbd_device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The Manual_demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

pipeline.start(config)

# 定义一个MyThread.py线程类
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
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


def mouse_click(event, x, y, flags, param):  # 鼠标点击事件
    global input_point, input_label, input_stop, input_start  # 全局变量，输入点，响应信号
    if not input_stop:  # 判定标志是否停止输入响应了！
        if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键
            input_point.append([x, y])
            input_label.append(1)  # 1表示前景点
        elif event == cv2.EVENT_RBUTTONDOWN:  # 鼠标右键
            input_point.append([x, y])
            input_label.append(0)  # 0表示背景点
        elif event == cv2.EVENT_FLAG_LBUTTON:  # 鼠标左键长按 重置
            input_point = []
            input_label = []


def get_aligned_images():
    # 创建对齐对象与color流对齐
    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    # 将images转为numpy arrays
    img_color = np.asanyarray(aligned_color_frame.get_data())  # BGR图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图

    # 获取相机参数
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)

    return color_intrin, depth_intrin, img_color, img_depth, depth_mapped_image, aligned_color_frame, aligned_depth_frame


def pcl_process(points):

    return


def get_pointcloud(color_rs, depth_rs, name, num):  # 对齐后的彩色图和深度图作为输入，不是彩色帧的数组和深度帧的数组
    # 声明点云对象
    # pc = rs.pointcloud()
    # points = rs.points()  # 使用其他与点云相关的属性和函数扩展框架类。
    # pc.map_to(color_rs)
    # points_rs = pc.calculate(depth_rs)
    # points = np.asanyarray(points_rs.get_vertices())
    # colorful = np.asanyarray(color_rs.get_data())
    # print(points.shape, f"640*480 = {640 * 480}")
    # points = np.reshape(points, (640, 480, -1))
    #print("color_rs", color_rs.shape)
    color_rs = cv2.cvtColor(color_rs, cv2.COLOR_BGR2RGB)
    color_rs = o3d.geometry.Image(color_rs)
    depth_rs = o3d.geometry.Image(depth_rs)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_rs, depth_rs, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
    #print(np.asarray(rgbd_image.depth))
    tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    #tmp_points = np.asarray(tmp.points)  # 如果想看点的数据需要加上这句话
    #print("tmp_points", tmp_points)
    o3d.io.write_point_cloud("pre_pointcloud/{}{}.ply".format(name, num), tmp)
    # o3d.visualization.draw_geometries([tmp]) # 点云在open3d中的可视化

    return tmp


def SAM_wrapper(frame):
    global input_point, input_label, input_stop  # 全局变量，输入点，
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(frame)
    choosen_point = np.array(input_point)
    print("choosen_point", choosen_point)
    choosen_label = np.array(input_label)  # 标签， 1是前景点用于选中，2是背景点用于排斥
    print("choosen_label", choosen_label)
    masks, scores, logits = predictor.predict(
        point_coords=choosen_point,
        point_labels=choosen_label,
        multimask_output=True,
    )
    return masks, scores, logits


def apply_color_mask(image, mask, color, color_dark=0.4):  # 对掩体进行赋予颜色
    for c in range(3):
        # print("mask", mask)
        # np.where(condition, x, y)  满足condition取x,不满足取y
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - color_dark) + color_dark * color[c], image[:, :, c])
    return image


def apply_pointcloud_mask(color_image, depth_image, mask):
    return np.where(mask == 1, depth_image, 0)


if __name__ == '__main__':
    #global input_point, input_label, input_stop  # 全局变量，输入点，

    cv2.namedWindow("Scene", cv2.WINDOW_NORMAL)  # 初始化界面
    cv2.resizeWindow("Scene", 1280, 480)  # 调整界面尺寸
    cv2.setMouseCallback("Scene", mouse_click)
    # cap = cv2.VideoCapture(4)
    ori_masks = None  # 定义一个空mask
    ori_frame = None  # 定义一个空ori_frame
    k = 0
    profiler = Profiler()
    profiler.start()
    try:
        while True:
            # _1, frame = cap.read()  # 读取图像
            #data = np.linspace(0, 150, 151)
            #laser_pointer = np.random.choice(data, size=5)
            color_intrin, depth_intrin, img_color, img_depth, depth_mapped_image, aligned_color_frame, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数
            #print("img_depth", img_depth)
            pcl_points = get_pointcloud(img_color, img_depth, "global_pcl", k)
            print("input_stop", input_stop)
            print("input_point", input_point)
            print("input_label", input_label)
            if not input_stop and np.array(input_point).size != 0:  # 如果获得了激光语义，开始对点击场景进行分割，生成蓝色
                print("we have a interaction")
                thread_sam = MyThread(SAM_wrapper, (img_color,))
                thread_sam.start()
                thread_sam.join()
                masks, scores, logits = thread_sam.get_result()
                color = tuple(np.array([255, 0, 0]).tolist())
                # color = tuple(np.random.randint(0, 256, 3).tolist())
                #print("color", color)
                print("img_color1", img_color)
                img_color = apply_color_mask(img_color, masks[2], color)
                print("img_color2", img_color)
                local_depth = apply_pointcloud_mask(img_color, img_depth, masks[2])
                print("img_color3", img_color)
                local_pclpoints = get_pointcloud(img_color, local_depth, "local_pcl", k)
                # ori_masks = masks[2]

            if k != 0: # 如果是第一次，则不需要局部点云配准，否则需要进行局部配准
                A = 1

            # 如果获取语义后，场景没有发生变化，且语义与之前一致，那么继续使用上次的mask,节省时间。
            # if k != 0:
            #     image_subtraction = cv2.subtract(frame, ori_frame, dst=None, mask=None, dtype=None)
            #     Gray_image = cv2.cvtColor(image_subtraction, cv2.COLOR_BGR2GRAY)
            #     _2, thresh_image = cv2.threshold(Gray_image, 1, 1, cv2.THRESH_BINARY)
            #     print("thresh_image", thresh_image)
            #     cv2.imshow("image_subtraction", image_subtraction)  # 展示图像

            # if image_subtraction
            #     color = tuple(np.array([0,0,255]).tolist())
            #     frame = apply_color_mask(frame, ori_masks, color)

            cv2.imshow("Scene", np.hstack((img_color, depth_mapped_image)))  # 展示图像
            # ori_frame = img_color  # 储存上一帧
            k = k + 1
            key = cv2.waitKey(1)
            if key == 27 or (key & 0XFF == ord("q")):
                cv2.destroyAllWindows()
                break
    finally:
        # destroy the instance
        # cap.release()
        pipeline.stop()

    profiler.end()