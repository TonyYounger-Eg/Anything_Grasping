# coding:utf-8
#
# laser_wait_time.py
#
#  Created on: 2024/4/10
#      Author: Tex Yan Liu
#
# description: 这个用于延时信息，当带有激光的图片用于被提取深度信息以及生成点云的时候，带有激光的区域会丢失信息

# coding:utf-8
#
# Grasp_execution.py
#
#  Created on: 2024/3/29
#      Author: Tex Yan Liu
#
# description:  用于配合抓取点进行实际抓取的grasp文件
import numpy as np
# dis_left 0.5320000052452087 m
# touch_left [0.01744, 0.08804, 0.532] m
# dis_right 0.5200000405311584 m
# touch_right [0.0763, 0.08605, 0.52] m

import rospy
from std_msgs.msg import String
import time

laser_wait_back = rospy.Publisher("auto_grasp/laser_waiter_back", String, queue_size=1)


def laser_wait_callback(laser_waiter):
    print("已经接收到信息")
    if laser_waiter.data == "laser_is_ok":
        time.sleep(10)
        laser_wait_back_pub = String()
        laser_wait_back_pub.data = "laser_back_is_ok"
        laser_wait_back.publish(laser_wait_back_pub)
        print("已经反馈状态")


if __name__ == '__main__':
    global subscriber
    rospy.init_node('object_grasp_pose_pre_adjust', anonymous=True)  # 初始化本节点
    subscriber = rospy.Subscriber("/auto_grasp/laser_waiter", String, laser_wait_callback)  # 订阅KINOVA当前姿态的轨迹。
    rospy.spin()

