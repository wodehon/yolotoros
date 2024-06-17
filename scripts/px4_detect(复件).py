#! /home/duan/anaconda3/envs/ros2yolo/bin/python3
# -*- coding: utf-8 -*-

import cv2
import detect_test
import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool, Int8, Float32MultiArray
import yaml
import math

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

from cv_bridge import CvBridge

from cv_tool import px2xy

# from ament_index_python.packages import get_package_share_directory
# package_share_directory = get_package_share_directory('yolotoros2')

class yolo_detect():
    def __init__(self):
        self.bridge = CvBridge() #OpenCV与ROS的消息转换类
        self.yolo = detect_test.detectapi(weights = '/home/duan/Code/test/src/yolotoros/scripts/best.pt')
        # self.im_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.detectimg)

        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.pose_sub = message_filters.Subscriber('/mavros/local_position/pose', PoseStamped)
        # ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
        ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.pose_sub], 1, 0.1, allow_headerless=True) # allow_ 可以不适用时间戳
        ts.registerCallback(self.detectimg)
        # self.color_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.detectimg)

        # 默认从camera_info中读取参数,如果可以从话题接收到参数则覆盖文件中的参数
        # self.declare_parameter("camera_info_file", f"/home/duan/Code/test/src/yolotoros/config/camera_info.yaml")

        # camera_info_topic = self.get_parameter('camera_info_topic').value
        # get camera info
        self.camera_info = {}
        with open("/home/duan/Code/test/src/yolotoros/config/camera_info.yaml") as f:
            self.camera_info = yaml.full_load(f.read())
            print(self.camera_info['K'], self.camera_info['D'])

        self.camera_info_sub = rospy.Subscriber(
            '/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        self.img_pub = rospy.Publisher('/yolov5/detect' , Image, queue_size = 1)
        self.res_pub = rospy.Publisher('/yolov5/result', Detection2DArray, queue_size = 1)
        self.result_msg = Detection2DArray()
        self.pos_pub = rospy.Publisher('/yolov5/pose', Pose, queue_size = 1)
        self.pos_msg = Pose()

        # rospy.Timer(rospy.Duration(0.1), self.posetcb)
        # change 
        # rospy.Timer(rospy.Duration(0.05), self.posetcb)

        self.pose_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size = 1)
        self.pose_msg = PoseStamped() # 发送到planning
        self.trigger = False
        self.tri_pub = rospy.Publisher('/insulator_pub_flag', Bool, queue_size = 1)
        self.tri_msg = Bool()
        self.tri_msg.data = False # false表示tower pub，true表示insulator pub
        self.tuning_pub = rospy.Publisher('/yolov5/tuning', Int8, queue_size = 1)
        self.tuning_msg = Int8()
        self.tuning_msg.data = 0
        self.error_pub = rospy.Publisher('/yolov5/error', Float32MultiArray, queue_size = 1)
        self.error = Float32MultiArray()
        self.error.data = [0,0] 
        self.discover_pub = rospy.Publisher('/discover', Bool, queue_size = 1)
        self.hover_pub = rospy.Publisher('/reach', Bool, queue_size = 1)

        self.MAX_Z = 10
        self.depth_min = 10 # 初始值
        self.count = 0
        self.triggertest = False

        self.dicover = False
        self.reach = False

    def posetcb(self, event):
        # if self.trigger:
            # print("-------pub success!------")
            # # print("------------")
            # self.pose_pub.publish(self.pose_msg)
        self.tri_pub.publish(self.tri_msg)


    def camera_info_callback(self, msg):
        """
        通过回调函数获取到相机的参数信息
        """
        self.camera_info['K'] = msg.K
        self.camera_info['P'] = msg.P
        self.camera_info['D'] = msg.D
        self.camera_info['R'] = msg.R
        self.camera_info['roi'] = msg.roi
        

        # self.camera_info_sub.destroy()

    # def detectimg(self, img):
    #     frame = self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
    #     result, names = self.a.detect([frame])
    #     image_detect = result[0][0]
    #     # print(result[0][1])
    #     self.img_pub.publish(self.bridge.cv2_to_imgmsg(image_detect, "bgr8"))

    def detectimg(self, img, img_depth, cam_pose):
        print("-----------")
        time1 = rospy.Time.now()
        frame = self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
        y,x,z = frame.shape[0:3]
        # print(x,y,z)
        frame_depth = self.bridge.imgmsg_to_cv2(img_depth, desired_encoding='16UC1')
        # print(frame_depth.shape)
        result, names = self.yolo.detect([frame])
        image_detect = result[0][0]
        planning = []

        trans = [0,0,0]
        trans[0] = cam_pose.pose.position.x
        trans[1] = cam_pose.pose.position.y
        trans[2] = cam_pose.pose.position.z
        quat = [0,0,0,0]
        quat[0] = cam_pose.pose.orientation.w
        quat[1] = cam_pose.pose.orientation.x
        quat[2] = cam_pose.pose.orientation.y
        quat[3] = cam_pose.pose.orientation.z
        
        # print(result[0][1]) 直接打印box
        # 发送到result
        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = rospy.Time.now()

        
        self.flag = False # Ajust:False往左，True往右
        self.wh_prev = 0 # 宽高比
        self.wh_real = 4
        self.distance = 1 # 安全距离

        if not result[0][1]:
            self.dicover = True
        else:
            self.dicover = False
        
        # self.pos_msg.position.clear()
        for i in result[0][1]:
            # self.pose_msg.pose.position.x = 0
            # self.pose_msg.pose.position.y = 0
            # self.pose_msg.pose.position.z = 0
            # self.result_msg.header.stamp = rospy.Time.now()

            name = names[int(i[0])]
            detection2d = Detection2D()
            # detection2d.is_tracking = True
            # detection2d.tracking_id = name
            x1, y1, x2, y2 = i[1]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            center_x = 0
            center_y = 0
            center_x = (x1+x2)/2.0
            center_y = (y1+y2)/2.0
            # print(center_x,center_y)

            detection2d.bbox.center.x = center_x
            detection2d.bbox.center.y = center_y

            detection2d.bbox.size_x = float(x2-x1)
            detection2d.bbox.size_y = float(y2-y1)

            # 边检测边规划部分
            wh_ = float(x2-x1) / float(y2-y1)
            if self.wh_prev == 0:
                self.wh_prev = wh_

            obj_pose = ObjectHypothesisWithPose()
            obj_pose.id = int(i[0])
            # # test 
            # print("======test name======")
            # print(name)
            # print(type(obj_pose.id))
            # print(type(name))
            # print("======test name======")
            obj_pose.score = float(i[2])

            # print(self.camera_info["K"])

            # px2xy
            point = [center_x, center_y]
            world_x = 0
            world_y = 0
            world_x, world_y = px2xy(point, self.camera_info["K"], self.camera_info["D"], 1)
            depth_ = frame_depth[int(center_y),int(center_x)]
            if depth_ == 0: # 无效深度(设置最大值)
                depth_ = self.MAX_Z
            # if id == "tower": # 如果是电塔，则以最远为目标点
            #     depth_ = self.MAX_Z
            scale = 1    # 真实相机为0.001
            obj_pose.pose.pose.position.x = world_x * depth_ * scale
            obj_pose.pose.pose.position.y = world_y * depth_ * scale
            obj_pose.pose.pose.position.z =  depth_ * scale   #2D相机则显示,归一化后的结果,用户用时自行乘上深度z获取正确xy
            # print(obj_pose)

            planning.append([name,float(i[2]),center_x,center_y,float(x2-x1),float(y2-y1),depth_, world_x, world_y])
            
            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)
            # print(detection2d)
            # print("-----------")
            # print(self.result_msg.detections)
            # for i in self.result_msg.detections:
            #     item = i.results
            #     print("------------")
            #     print(len(item))


            pointxyz = [obj_pose.pose.pose.position.x, obj_pose.pose.pose.position.y, obj_pose.pose.pose.position.z]
            print("-----------")
            print(pointxyz)
            if name == "insulator":
                print("-----------")
            #     print(point, depth_, self.depth_min)
            

            # 在每次yolo的过程中进行规划
            if name == "tower" and obj_pose.score > 0.6 and (not self.trigger):
                self.error.data[0] = (x/2 - center_x) / x
                self.discover_pub.publish(self.dicover and self.triggertest)
                time2 = rospy.Time.now()
                self.error.data[1] = time2.to_sec() - time1.to_sec()
                self.error_pub.publish(self.error)
                if not self.triggertest:
                    if center_x < x/2 and self.tuning_msg.data != -1:
                        self.tuning_msg.data = 1
                        # print("1")
                    elif center_x > x/2 and self.tuning_msg.data != 1:
                        self.tuning_msg.data = -1
                        # print("2")
                    else:
                        self.tuning_msg.data = 0
                        # print("3")
                    self.count+=1
                if self.tuning_msg.data == 0 and self.count > 100:
                    self.triggertest = True
                    depth_ = self.MAX_Z
                    obj_pose.pose.pose.position.x = world_x * depth_ * scale
                    obj_pose.pose.pose.position.y = world_y * depth_ * scale
                    obj_pose.pose.pose.position.z =  depth_ * scale
                    pointxyz = [obj_pose.pose.pose.position.x, obj_pose.pose.pose.position.y, obj_pose.pose.pose.position.z]
                    # print(point,[world_x,world_y],pointxyz)
                    point_ = cal(pointxyz, trans, quat)
                    if(abs(point_[0]) > 5 or abs(point_[1]) > 5 or abs(point_[2]) > 3):
                        point_ = trunc(point_, trans)
                    self.pos_msg.position.x = point_[0]
                    self.pos_msg.position.y = point_[1]
                    self.pos_msg.position.z = point_[2]
                    self.pose_msg.pose.position.x = point_[0]
                    self.pose_msg.pose.position.y = point_[1]
                    self.pose_msg.pose.position.z = point_[2]
                    yaw = math.atan(point_[1]/point_[0])
                    self.pose_msg.pose.orientation.w = yaw

                    self.pose_pub.publish(self.pose_msg)
            elif name == "insulator" and obj_pose.score > 0.6 and depth_ > 1 and depth_ < 5 and depth_ < self.depth_min: # 6m开始切换adjust,且以最近的那个为目标
                self.trigger = True
                self.tri_msg.data = self.trigger
                self.depth_min = depth_
                print("==================")
                print("===",depth_,depth_<8,self.depth_min,self.trigger)

                obj_pose.pose.pose.position.x = world_x * depth_ * scale
                obj_pose.pose.pose.position.y = world_y * depth_ * scale
                obj_pose.pose.pose.position.z = depth_ * scale
                pointxyz = [obj_pose.pose.pose.position.x, obj_pose.pose.pose.position.y, obj_pose.pose.pose.position.z]
                point_ = cal(pointxyz, trans, quat)
                print("-----------")
                print(point_)
                yaw = math.atan((point_[1]-trans[1])/(point_[0]-trans[0]))
                self.pose_msg.pose.orientation.w = yaw
                self.pose_msg.pose.orientation.x = depth_

                # 调整阶段偏转
                cos_ = wh_ / self.wh_real
                sin_ = math.sqrt(1 - cos_ * cos_)
                # 左右偏转
                # if wh_ < self.wh_prev and not self.flag:
                #     self.flag = True
                # if wh_ < self.wh_prev and self.flag:
                #     self.flag = False
                # if not self.flag: # 往左
                #     pointxyz[0] = pointxyz[0] - sin_ * self.distance
                # else : # 往右
                #     pointxyz[0] = pointxyz[0] + sin_ * self.distance
                pointxyz[0] = pointxyz[0] - sin_ * max((depth_ -3),self.distance)
                pointxyz[2] = pointxyz[2] + cos_ * max((depth_),self.distance)
                point_ = cal(pointxyz, trans, quat)
                self.pos_msg.position.x = point_[0]
                self.pos_msg.position.y = point_[1] 
                self.pos_msg.position.z = point_[2] - obj_pose.pose.pose.position.y
                self.pose_msg.pose.position.x = point_[0]
                self.pose_msg.pose.position.y = point_[1]
                self.pose_msg.pose.position.z = point_[2]
                if depth_ < 4 :
                    self.pose_msg.pose.position.x = -1.105
                    self.pose_msg.pose.position.y = 0.801
                    self.pose_msg.pose.position.z = 1.04
                    self.reach = True
                self.error.data[0] = (x/2 - center_x) / x
                self.discover_pub.publish(self.dicover and self.triggertest)
                time2 = rospy.Time.now()
                self.error.data[1] = time2.to_sec() - time1.to_sec()
                self.error_pub.publish(self.error)
            if self.reach:
                self.hover_pub.publish(self.reach)   

        if self.trigger:
            print(self.pose_msg.pose.position)
            print(self.pose_msg.pose.orientation)
            self.pose_pub.publish(self.pose_msg)
            self.tri_pub.publish(self.tri_msg)


        
        

        time2 = rospy.Time.now()
        print("------coss " + str(time2.to_sec() - time1.to_sec()) + " s-------")
        # print("trigger: " + str(self.trigger))

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(image_detect, "bgr8"))
        self.res_pub.publish(self.result_msg)
        # self.pos_pub.publish(self.pos_msg)
        self.tuning_pub.publish(self.tuning_msg)
        
        # if self.trigger:
        #     # print(self.pose_msg.pose.position)
        #     # self.pose_pub.publish(self.pose_msg)
        #     self.tri_pub.publish(self.tri_msg)


def quattorot(quat_):
    # return tf.transformations.quaternion_matrix(quat_)
    q = quat_.copy()
    q /= np.linalg.norm(q)
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        print("error")
        return np.identity(3)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    # rot_matrix = np.array(
    # [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
    #  [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
    #  [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
    #  [0.0, 0.0, 0.0, 1.0]],
    # dtype=q.dtype)
    rot_matrix = np.array(
    [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0]],
    [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0]],
    [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2]]],
    dtype=q.dtype)
    return rot_matrix


def quat_rot(quat_):
    q = np.quaternoion


def cal(point, trans_, quat_):
    # get_tf = False
    # while not get_tf:
    # (trans,quat) = listener.lookupTransform('/turtle2', '/turtle1', rospy.Time(0))
    rot = quattorot(np.array(quat_))
    rot_ = np.linalg.inv(rot) # 逆变换
    # print(rot,rot_)
    # print(rot.shape)
    # point_ = np.dot(point, rot) + np.array(trans_)
    point_ = camtobody(point)
    point_ = np.dot(point_, np.transpose(rot_)) + np.array(trans_) # 使用行向量，所以转置
    
    return point_


def trunc(point, trans_):
    range_x = 10.0
    range_y = 10.0
    range_z = 3.0 - 0.1

    i_ = point[0] - trans_[0]
    j_ = point[1] - trans_[1]
    k_ = point[2] - trans_[2]
    if i_ != 0:
        ti = (np.sign(i_) * range_x - trans_[0]) / i_
    elif i_ == 0:
        ti = float('inf')
    if j_ != 0:
        tj = (np.sign(j_) * range_y - trans_[1]) / j_
    elif j_ == 0:
        tj = float('inf')
    if k_ > 0:
        tk = (range_z - trans_[2]) / k_
    elif k_ <0:
        tk = (0.0 - trans_[2]) / k_
    elif k_ == 0:
        tk = float('inf')
    t_ = [ti,tj,tk]
    t = np.min(t_)
    x_ = trans_[0] + t * i_
    y_ = trans_[1] + t * j_
    z_ = trans_[2] + t * k_

    return [x_,y_,z_]

# camera to FRD
def camtobody(point_):
    point = point_
    # 1.FRD（就是px4）
    # cam2body_ = np.array(
    # [[0,0,1],
    #  [1,0,0],
    #  [0,1,0]],
    # dtype=3)
    # 2.FLU（就是ROS的，或者MAVROS的）
    cam2body_ = np.array(
    [[0,0,1],
    [-1,0,0],
    [0,-1,0]])
    point = np.dot(point, np.transpose(cam2body_)) + np.array([-0.02,0,0])# 使用的是行向量，所以用了转置

    return point


if __name__ == '__main__':
    rospy.init_node("yolo_detect")
    rospy.loginfo("yolo_detect node started")
    # listener = tf.TransformListener()
    yolo_detect()
    rospy.spin()

# while True:
#     rec, img = cap.read()
#     result, names = a.detect([img])
#     img = result[0][0]
#     '''
#     for cls,(x1,y1,x2,y2),conf in result[0][1]: #第一张图片的处理结果标签。
#         print(cls,x1,y1,x2,y2,conf)
#         cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
#         cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
#     '''
#     cv2.imshow("vedio", img)
#     if cv2.waitKey(1)==ord('q'):
#         break
