# coding:utf-8
#
# Grasp_execution.py
#
#  Created on: 2024/3/29
#      Author: Tex Yan Liu
#
# description:  用于配合抓取点进行实际抓取的grasp文件
import numpy as np
import roslib
from robot_control_modules import *  # 调用现有的机器人控制模块
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import quaternion
from Anything_Grasping.msg import grasp_info

roslib.load_manifest('kinova_demo')  # 调用库
prefix = 'j2n6s300_'
jaco_matrix = np.eye(4)
grasp_status = rospy.Publisher('auto_grasp/grasp_status', String, queue_size=1)


def matrix_to_quaternion(matrix):
    trace = np.trace(matrix)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2 * np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2 * np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def grasp_callback(grasp_info):
    global jaco_matrix
    print("订阅到抓取消息：", grasp_info)
    touch_left = np.array(grasp_info.left_touch)
    touch_right = np.array(grasp_info.right_touch)
    touch_angle = grasp_info.angle_touch
    touch_center = np.array(touch_left + touch_right) / 2  # 收集相机坐标系下中心坐标点的xyz坐标

    eye_on_hand_t_matrix = np.array([0.0373408, 0.100744, -0.15574])  # 2024年3月1日
    eye_on_hand_q_matrix = np.quaternion(0.07655784608114653, 0.014197669585180192,
                                         0.009373324380869241, 0.9969199883500266)

    eye_on_hand_r_matrix = quaternion.as_rotation_matrix(eye_on_hand_q_matrix)
    eye_on_hand_matrix = np.eye(4)
    eye_on_hand_matrix[:3, :3] = eye_on_hand_r_matrix
    eye_on_hand_matrix[:3, 3] = eye_on_hand_t_matrix
    touch_center_after_eyeonhand = np.matmul(eye_on_hand_matrix, touch_center) + np.array([0, 0, 0 + 0.04, 0])
    # hand_to_finger = np.eye(4)
    # eye_on_hand_matrix[:3, 3] = np.array([])
    setFingerPos = np.zeros(3)
    setFingerPos[0] = 0
    setFingerPos[1] = 0
    setFingerPos[2] = 0
    result_finger_open = gripper_client(setFingerPos, prefix)
    print("手指张开状态结果：", result_finger_open)
    touch_matrix = np.eye(4)  # 最终抓取的姿态矩阵初始化
    touch_matrix[:3, :3] = np.array([
        [np.cos(touch_angle), -np.sin(touch_angle), 0],
        [np.sin(touch_angle), np.cos(touch_angle), 0],
        [0, 0, 1]
    ])
    touch_matrix[:3, 3] = touch_center_after_eyeonhand[:3]
    # touch_matrix[:3, 3] = np.zeros(3)

    print("手上位置的姿态矩阵：", touch_matrix)

    center_after_jaco = np.matmul(jaco_matrix, touch_matrix)
    final_position = center_after_jaco[:3, 3]
    tmp_quaternion = matrix_to_quaternion(center_after_jaco[:3, :3])
    final_quaternion = np.array([tmp_quaternion[1], tmp_quaternion[2], tmp_quaternion[3], tmp_quaternion[0]])  # x,y,z,w
    result_robot = cartesian_pose_client(final_position, final_quaternion, prefix)
    print("机器人运动状态结果：", result_robot)

    setFingerPos[0] = 6000
    setFingerPos[1] = 6000
    setFingerPos[2] = 0
    result_finger_close = gripper_client(setFingerPos, prefix)
    print("手指闭合状态结果：", result_finger_close)
    grasp_status_info = String()
    grasp_status_info.data = "ok"
    grasp_status.publish(grasp_status_info)
    rospy.loginfo("grasp_status is published: %s", grasp_status_info.data)


def kinovagrasp_callback(pose_data):
    global jaco_matrix
    jaco_position = np.array([pose_data.pose.position.x,
                              pose_data.pose.position.y,
                              pose_data.pose.position.z])
    jaco_orientation = np.quaternion(pose_data.pose.orientation.w,
                                     pose_data.pose.orientation.x,
                                     pose_data.pose.orientation.y,
                                     pose_data.pose.orientation.z)  # w, x, y, z
    jaco_r_matrix = quaternion.as_rotation_matrix(jaco_orientation)
    jaco_matrix[:3, :3] = jaco_r_matrix
    jaco_matrix[:3, 3] = jaco_position


if __name__ == '__main__':
    global subscriber
    rospy.init_node('object_grasp_pose_pre_adjust', anonymous=True)  # 初始化本节点

    subscriber_1 = rospy.Subscriber("/j2n6s300_driver/out/tool_pose", PoseStamped, kinovagrasp_callback)  # 订阅KINOVA当前姿态的轨迹。
    subscriber_2 = rospy.Subscriber("/auto_grasp/grasp_execution", grasp_info, grasp_callback)

    rospy.spin()
