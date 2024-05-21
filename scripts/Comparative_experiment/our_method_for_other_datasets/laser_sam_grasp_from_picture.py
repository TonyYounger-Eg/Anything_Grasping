# coding:utf-8
#
# laser_SAM.py.py
#
#  Created on: 2024/1/24
#      Author: Tex Yan Liu
#
# description: 包括检测和分割,用于2023/02/04提交的代码

import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from grasp_wrench_2d_v6 import grasp_wrench_2d
import copy
import torch
import os

sam_checkpoint = "../../../segment-anything/checkpoints/sam_vit_h_4b8939.pth"  # sam的权重文件
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


def apply_white_mask(apply_color_tmp_image, apply_color_tmp_mask, apply_color_tmp_color):  # 对掩码进行赋予颜色
    color_dark = 0
    # print("apply_color_tmp_mask", apply_color_tmp_mask)
    for c in range(3):
        # np.where(condition, x, y)  满足condition取x,不满足取y
        apply_color_tmp_image[:, :, c] = np.where(
            apply_color_tmp_mask == 1,
            apply_color_tmp_image[:, :, c] * (1 - color_dark) + color_dark * apply_color_tmp_color[c], 255)
    return apply_color_tmp_image


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


def two_dimensions_grasp_wrench(two_dimensoions_grasp_mask):


    print("two_dimensoions_grasp_mask.copy()", two_dimensoions_grasp_mask.copy())
    two_dimensoions_grasp_contours, two_dimensoions_grasp_cnt = cv2.findContours(two_dimensoions_grasp_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("轮廓总数", len(two_dimensoions_grasp_contours))
    moment_2d_grasp_contours = cv2.moments(two_dimensoions_grasp_contours[0])  # 计算第一条轮廓的各阶矩,字典形式，根据自己的激光点效果寻找轮廓
    print("moment_2d_grasp_contours", moment_2d_grasp_contours)
    print("moment_2d_grasp_contours[m00]", moment_2d_grasp_contours["m00"])
    moment_2d_grasp_center_x = int(moment_2d_grasp_contours["m10"] / moment_2d_grasp_contours["m00"])
    moment_2d_grasp_center_y = int(moment_2d_grasp_contours["m01"] / moment_2d_grasp_contours["m00"])

    return two_dimensoions_grasp_contours, moment_2d_grasp_center_x, moment_2d_grasp_center_y


if __name__ == '__main__':
    sub_folder = '11/error/error'
    folder_path = "Jacquard_dataset/{}".format(sub_folder)
    files = os.listdir(folder_path)
    for file in files:
        print("处理文件中...,当前处理{}".format(file))
        name, extension = os.path.splitext(file)
        S_optimal = []
        img_color = cv2.imread('Jacquard_dataset/{}/{}'.format(sub_folder, file))
        # cv2.namedWindow("scene3", cv2.WINDOW_NORMAL)
        # cv2.imshow("scene3", img_color)
        # key = cv2.waitKey(0)

        img_color_2 = copy.deepcopy(img_color)
        image = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        boxes = np.array([[254,131,909,854]])   # 10: 186, 296, 329, 398 error: 203 325 320 412 error in error: 193, 317, 253, 380 11:282,159,749,794
        input_boxes = torch.tensor(boxes, device=predictor.device)

        predictor.set_image(image)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        tmp_masks = masks[0].cpu().numpy()    # 设置分割等级
        seg_tmp_masks = tmp_masks[0]
        print("seg_tmp_masks", seg_tmp_masks)
        color = tuple(np.random.randint(0, 256, 3).tolist())  # 设置颜色随机
        # color = tuple(np.array([200, 86, 253]))  # 设置颜色随机
        print("color is :", color)

        # img_white_color = apply_white_mask(img_color, seg_tmp_masks, color)   # 为实例分割物品赋予阴影
        seg_tmp_masks = np.where(seg_tmp_masks == True, 1, 0)  # 改成黑白图， 有阴影部分为白1，无阴影部分为黑0
        seg_tmp_masks = np.array(seg_tmp_masks, np.uint8)  # 改为int8
        try:
            grasp_2d_wrench_contours, grasp_2d_wrench_contours_center_x, grasp_2d_wrench_contours_center_y = two_dimensions_grasp_wrench(
                seg_tmp_masks)  # 求解阴影的边界和圆心
            grasp_2d_wrench_contours_reshape = np.reshape(grasp_2d_wrench_contours[0], (-1, 2))
            grasp_2d_wrench_contours_center = np.array([grasp_2d_wrench_contours_center_x, grasp_2d_wrench_contours_center_y])
            # np.savetxt("grasp_stability/txt/boundary.txt", grasp_2d_wrench_contours_reshape, fmt='%s')
            # np.savetxt("grasp_stability/txt/center.txt", grasp_2d_wrench_contours_center, fmt='%s')
            cv2.drawContours(img_color, grasp_2d_wrench_contours, 0, color, 2)  # 绘制轮廓，填充（图像，轮廓，轮廓序号，颜色，轮廓线粗细）
            cv2.circle(img_color, (grasp_2d_wrench_contours_center_x, grasp_2d_wrench_contours_center_y), 2,
                       (0, 0, 255), -1)  # 绘制中心点
            # 将这两个if全部注释可以隐藏抓取
            if len(S_optimal) == 0:
                q_max, W_max, S_optimal, S_rotate_degree = grasp_wrench_2d(grasp_2d_wrench_contours_reshape, grasp_2d_wrench_contours_center)
                S_optimal = S_optimal.astype(np.int32)

            if len(S_optimal) != 0:
                color_2 = tuple(np.random.randint(0, 256, 3).tolist())  # 设置颜色随机
                cv2.drawContours(img_color_2, grasp_2d_wrench_contours, 0, color, 2)  # 绘制轮廓，填充（图像，轮廓，轮廓序号，颜色，轮廓线粗细）
                cv2.circle(img_color_2, (grasp_2d_wrench_contours_center_x, grasp_2d_wrench_contours_center_y), 2,
                           (0, 0, 255), -1)  # 绘制中心点
                cv2.line(img_color_2, S_optimal[0], S_optimal[1], color_2, 1, cv2.LINE_AA)
                cv2.circle(img_color_2, S_optimal[0], 8, color_2, -1)
                cv2.circle(img_color_2, S_optimal[1], 8, color_2, -1)
        except ZeroDivisionError:
            cv2.imwrite("picture/{}/ZeroDivisionError_{}_grasp.png".format(sub_folder, name), img_color_2)
            img_color = apply_color_mask(img_color, seg_tmp_masks, color)   # 为实例分割物品赋予阴影
            cv2.imwrite("picture/{}/ZeroDivisionError_{}_boundary.png".format(sub_folder, name), img_color)
            continue
        except TypeError:
            cv2.imwrite("picture/{}/TypeError_{}_grasp.png".format(sub_folder, name), img_color_2)
            img_color = apply_color_mask(img_color, seg_tmp_masks, color)   # 为实例分割物品赋予阴影
            cv2.imwrite("picture/{}/TypeError_{}_boundary.png".format(sub_folder, name), img_color)
            continue
        except ValueError:
            cv2.imwrite("picture/{}/ValueError_{}_grasp.png".format(sub_folder, name), img_color_2)
            img_color = apply_color_mask(img_color, seg_tmp_masks, color)  # 为实例分割物品赋予阴影
            cv2.imwrite("picture/{}/ValueError_{}_boundary.png".format(sub_folder, name), img_color)
            continue

        # cv2.imshow("Scene2", img_color_2)  # 展示图像
        cv2.imwrite("picture/{}/{}_grasp.png".format(sub_folder, name), img_color_2)
        # img_color = apply_white_mask(img_color, seg_tmp_masks, color)   # 为实例分割物品赋予阴影
        img_color = apply_color_mask(img_color, seg_tmp_masks, color)   # 为实例分割物品赋予阴影
        # cv2.imshow("Scene", img_color)  # 展示图像
        cv2.imwrite("picture/{}/{}_boundary.png".format(sub_folder, name), img_color)

        key = cv2.waitKey(0)