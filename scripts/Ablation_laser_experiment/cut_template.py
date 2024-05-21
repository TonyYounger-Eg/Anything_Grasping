# coding:utf-8
#
# cut_template.py
#
#  Created on: 2024/5/1
#      Author: Tex Yan Liu
#
# description: 裁减一个模板出来

import cv2

# 读取图片
image = cv2.imread('template_laser_origin.png')

# 指定裁剪区域，这里用矩形定义：(x, y, width, height)
x, y, w, h = 320, 373, 40, 30

# 裁剪图片
cropped_image = image[y:y + h, x:x + w]

# 显示裁剪后的图片
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)

# 保存裁剪后的图片
cv2.imwrite('template_laser_cut.png', cropped_image)

# 关闭所有窗口
cv2.destroyAllWindows()