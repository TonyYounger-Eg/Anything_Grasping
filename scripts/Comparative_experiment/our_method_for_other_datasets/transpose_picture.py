# coding:utf-8
#
# transpose_picture.py
#
#  Created on: 2024/4/15
#      Author: Tex Yan Liu
#
# description: 遍历文件夹中的所有照片

import os
import cv2

folder_path = "Cornell_dataset"

if __name__ == '__main__':
    files = os.listdir(folder_path)
    for file in files:
        # print(file)
        # print(len(files))
        name, extension = os.path.splitext(file)
        img_color = cv2.imread('Cornell_dataset/{}'.format(file))
        cv2.imwrite("picture/{}_grasp.png".format(name), img_color)
        key = cv2.waitKey(0)

