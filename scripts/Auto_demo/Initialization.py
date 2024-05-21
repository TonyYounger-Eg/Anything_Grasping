# coding:utf-8
#
# Initialization.py
#
#  Created on: 2024/3/12
#      Author: Tex Yan Liu
#
# description: 为了调试方便，不用来回跑，特意做得姿态自动调整文件

import roslib
import numpy as np
from robot_control_modules import *  # 调用现有的机器人控制模块
roslib.load_manifest('kinova_demo')  # 调用库

rospy.init_node('pre_pose', anonymous=True)  # 初始化本节点
# final_position = np.array([0.14506925642490387, -0.3463810086250305, 0.5038034319877625])
# final_quaternion = np.array([0.8126999139785767,  0.2672758102416992, 0.24132144451141357, 0.4580899477005005])

final_position = np.array([0.19737038016319275, -0.34376224875450134, 0.5017629265785217])
final_quaternion = np.array([0.7541885375976562, -0.06125084310770035, 0.03138843923807144, 0.6530411243438721])


# final_position = np.array([0.08116522431373596, -0.2921654284000397, 0.3214850127696991])
# final_quaternion = np.array([0.764767050743103,  0.2899439036846161, 0.31415873765945435, 0.4820457696914673])
prefix = 'j2n6s300_'
setFingerPos = np.zeros(3)
setFingerPos[0] = 0
setFingerPos[1] = 0
setFingerPos[2] = 0
result = gripper_client(setFingerPos,prefix)
result = cartesian_pose_client(final_position, final_quaternion, prefix)
