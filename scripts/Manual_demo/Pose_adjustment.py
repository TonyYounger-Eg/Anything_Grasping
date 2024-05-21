# coding:utf-8
#
# Pose_adjustment.py
#
#  Created on: 2023/02/29
#      Author: Tex Yan Liu
#
# description: 此脚本将根据提示点的法线预调整机器人的姿态

import roslib
from robot_control_modules import *  # 调用现有的机器人控制模块
from std_msgs.msg import String
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation

roslib.load_manifest('kinova_demo')  # 调用库

prefix = 'j2n6s300_'

def QuaternionNorm(Q_raw):  # 正则化四元数
    qx_temp, qy_temp, qz_temp, qw_temp = Q_raw[0:4]
    qnorm = math.sqrt(qx_temp * qx_temp + qy_temp * qy_temp + qz_temp * qz_temp + qw_temp * qw_temp)
    if qnorm != 0:
        qx_ = qx_temp / qnorm
        qy_ = qy_temp / qnorm
        qz_ = qz_temp / qnorm
        qw_ = qw_temp / qnorm
        Q_normed_ = [qx_, qy_, qz_, qw_]

    return Q_normed_


def quaternion_to_rotation_matrix(quaternion):
    q0, q1, q2, q3 = quaternion
    rotation_matrix = np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ])
    return rotation_matrix


def markerpose_callback(marker_data):
    global transformation_marker_after
    marker_orientation = [0.0, 0.0, 0.0, 0.0]
    marker_orientation[0] = marker_data.pose.orientation.w
    marker_orientation[1] = marker_data.pose.orientation.x
    marker_orientation[2] = marker_data.pose.orientation.y
    marker_orientation[3] = marker_data.pose.orientation.z

    Q_normed = QuaternionNorm(
        marker_orientation)  # 四元数到欧拉的变换函数 posestamped.pose.position:机械臂的笛卡尔xyz posestamped.pose.orientation:机械臂姿态四元数
    rotation_matrix = quaternion_to_rotation_matrix(Q_normed)
    transformation_marker = np.eye(4)
    transformation_marker[:3, :3] = rotation_matrix
    x_rotato_180_degree = np.eye(4)
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    x_rotato_180_degree[:3, :3] = rotation_matrix_x
    print("标定板变换前的姿态: \n", transformation_marker)
    transformation_marker_tmp = np.matmul(x_rotato_180_degree, transformation_marker)
    z_rotato_180_degree = np.eye(4)
    rotation_matrix_z = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    z_rotato_180_degree[:3, :3] = rotation_matrix_z
    transformation_marker_after = np.matmul(z_rotato_180_degree, transformation_marker_tmp)

    transformation_marker_after[:3, 3] = np.array(
        [marker_data.pose.position.x, marker_data.pose.position.y, marker_data.pose.position.z])
    print("标定板绕x轴变换180度的姿态: \n", transformation_marker_after)


# 计算四元数的部分值
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

def vector_to_quaternion(rotation_vector):
    # Normalize the rotation vector
    normalized_vector = rotation_vector / np.linalg.norm(rotation_vector)
    print("规范化向量:", normalized_vector)
    # Create a rotation object from the normalized vector
    rotation = Rotation.from_rotvec(normalized_vector)
    print("规范化向量:", normalized_vector)

    # Get the quaternion representation
    quaternion = rotation.as_quat()

    return quaternion



def vector_to_quaternion(v):
    # 计算旋转角度
    theta = np.linalg.norm(v)

    # 单位化向量
    v_unit = v / np.linalg.norm(v)

    # 计算四元数的实部和虚部
    q0 = np.cos(theta / 2.0)
    q1 = np.sin(theta / 2.0) * v_unit[0]
    q2 = np.sin(theta / 2.0) * v_unit[1]
    q3 = np.sin(theta / 2.0) * v_unit[2]

    return np.array([q0, q1, q2, q3])


def kinovapose_callback(kinova_data):
    global subscriber

    # global transformation_marker_after
    attention_t = np.array([0.03639,0.05347,0.425])    # 提示点相对于相机坐标系原点的坐标
    attention_normals = np.array([0.29246,-0.31052,0.90446])    # 提示点处的法线方向

    new_axis_z = attention_normals  # 提示点处的法线方向，该法线方向为原坐标系的Z轴方向
    attention_mol = np.linalg.norm(attention_t)  # 求解距离范数
    attention_t_2 = attention_t - attention_mol * attention_normals  # 求解新的相机坐标原点 （圆弧末端点） （0，0，0 圆弧起始点）
    print("求解新的相机坐标原点", attention_t_2)

    # attention_q_old = vector_to_quaternion(attention_normals)  # 求解法线的四元数
    # print("attention_q_old:", attention_q_old)
    new_axis_x = np.zeros(3)
    new_axis_x = np.cross(attention_t/attention_mol, new_axis_z)
    new_axis_x = new_axis_x / np.linalg.norm(new_axis_x)
    new_axis_y = np.zeros(3)
    new_axis_y = -np.cross(new_axis_x, new_axis_z)
    new_axis_y = new_axis_y / np.linalg.norm(new_axis_y)

    attention_r_matrix = np.array([[new_axis_x[0], new_axis_y[0], new_axis_z[0]],
                                   [new_axis_x[1], new_axis_y[1], new_axis_z[1]],
                                   [new_axis_x[2], new_axis_y[2], new_axis_z[2]]])
    print("预调整姿态的旋转矩阵: ", attention_r_matrix)
    attention_matrix = np.eye(4)     # 这个矩阵存在很大的问题20240301！回答：经测试不存在问题20240319
    print("矩阵的转置", attention_r_matrix.T)
    print("矩阵的逆矩", np.linalg.inv(attention_r_matrix))
    attention_matrix[:3, :3] = attention_r_matrix   # 2024/03/01 现在这个代码好像是对的

    # x_rotato_180_degree = np.eye(4)    # 由于相机坐标系和机械臂坐标系的原因，需要将姿态结果先绕X轴旋转180度，再绕Z轴旋转180度
    # rotation_matrix_x = np.array([
    #     [1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, -1]
    # ])
    # x_rotato_180_degree[:3, :3] = rotation_matrix_x
    # print("绕x轴转换180度之前的姿态: \n", attention_matrix)
    # attention_r_matrix_tmp = np.matmul(x_rotato_180_degree, attention_matrix)
    # print("绕x轴转换180度之后的姿态: \n", attention_matrix)
    # z_rotato_180_degree = np.eye(4)
    # rotation_matrix_z = np.array([
    #     [-1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, 1]
    # ])
    # z_rotato_180_degree[:3, :3] = rotation_matrix_z
    # attention_r_matrix_after = np.matmul(z_rotato_180_degree, attention_r_matrix_tmp)

    attention_matrix[:3, 3] = np.array(attention_t_2)  # 得加个括号，双括号才行, 好像不加也行

    # 眼在手上矩阵
    eye_on_hand_t_matrix = np.array([0.0373408, 0.100744, -0.15574])     # 2024年3月1日
    eye_on_hand_q_matrix = np.quaternion(0.07655784608114653, 0.014197669585180192,
                                         0.009373324380869241, 0.9969199883500266)
    eye_on_hand_r_matrix = quaternion.as_rotation_matrix(eye_on_hand_q_matrix)
    eye_on_hand_matrix = np.eye(4)
    eye_on_hand_matrix[:3, :3] = eye_on_hand_r_matrix
    eye_on_hand_matrix[:3, 3] = eye_on_hand_t_matrix
    # marker_after_eyeonhand = np.matmul(eye_on_hand_matrix, transformation_marker_after)
    eye_on_hand_matrix_inv = np.eye(4)
    eye_on_hand_matrix_inv[:3, :3] = eye_on_hand_r_matrix.T
    eye_on_hand_matrix_inv[:3, 3] = -1 * np.matmul(eye_on_hand_r_matrix.T, eye_on_hand_t_matrix)
    eyeonhand_after_attention = np.matmul(attention_matrix, eye_on_hand_matrix_inv)
    attention_after_eyeonhand = np.matmul(eye_on_hand_matrix, eyeonhand_after_attention)
    # 眼在手矩阵调整后的姿态

    # Jaco矩阵
    # 机械臂当前回调的姿态和坐标
    jaco_position = np.array([kinova_data.pose.position.x,
                              kinova_data.pose.position.y,
                              kinova_data.pose.position.z])
    jaco_orientation = np.quaternion(kinova_data.pose.orientation.w,
                                     kinova_data.pose.orientation.x,
                                     kinova_data.pose.orientation.y,
                                     kinova_data.pose.orientation.z)  # w, x, y, z
    jaco_r_matrix = quaternion.as_rotation_matrix(jaco_orientation)
    jaco_matrix = np.eye(4)
    jaco_matrix[:3, :3] = jaco_r_matrix
    jaco_matrix[:3, 3] = jaco_position
    marker_after_jaco = np.matmul(jaco_matrix, attention_after_eyeonhand)
    print("Jaco机械臂当前的姿态 \n", jaco_matrix)
    print("估计预调整姿态 \n", marker_after_jaco)
    final_position = marker_after_jaco[:3, 3]
    tmp_quaternion = matrix_to_quaternion(marker_after_jaco[:3, :3])
    final_quaternion = np.array([tmp_quaternion[1], tmp_quaternion[2], tmp_quaternion[3], tmp_quaternion[0]])
    print("位置", final_position)
    print("姿态", final_quaternion)
    result = cartesian_pose_client(final_position, final_quaternion, prefix)
    # 取消订阅
    subscriber.unregister()

if __name__ == '__main__':
    global subscriber

    # [   0.03683    -0.01251       0.476]
    # [   0.047061    -0.43715     0.89816]
    rospy.init_node('object_grasp_pose_pre_adjust', anonymous=True)  # 初始化本节点

    # rospy.Subscriber("/aruco_single/pose", PoseStamped, markerpose_callback)  # 获取标定版姿态的节点
    subscriber = rospy.Subscriber("/j2n6s300_driver/out/tool_pose", PoseStamped, kinovapose_callback)  # 订阅KINOVA当前姿态的轨迹。

    rospy.spin()
