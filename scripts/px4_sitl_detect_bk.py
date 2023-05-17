#! /home/duan/anaconda3/envs/ros2yolo/bin/python3
# -*- coding: utf-8 -*-

import cv2
import detect_test
import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from geometry_msgs.msg import Pose
# import tf
import yaml

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
        self.yolo = detect_test.detectapi(weights = '/home/duan/Code/my_yolo5/runs/train/exp27/weights/best.pt')
        # self.im_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.detectimg)
        
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        # ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
        ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 1, 0.1, allow_headerless=True) # allow_ 可以不适用时间戳
        ts.registerCallback(self.detectimg)

        # 默认从camera_info中读取参数,如果可以从话题接收到参数则覆盖文件中的参数
        # self.declare_parameter("camera_info_file", f"/home/duan/Code/test/src/yolotoros/config/camera_info.yaml")

        # camera_info_topic = self.get_parameter('camera_info_topic').value
        # get camera info
        with open("/home/duan/Code/test/src/yolotoros/config/camera_info.yaml") as f:
            self.camera_info = yaml.full_load(f.read())
            print(self.camera_info['k'], self.camera_info['d'])

        self.camera_info_sub = rospy.Subscriber(
            '/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        self.camera_info = {}
        self.img_pub = rospy.Publisher('/yolov5/detect' , Image, queue_size = 1)
        self.res_pub = rospy.Publisher('/yolov5/result', Detection2DArray, queue_size = 1)
        self.result_msg = Detection2DArray()
        self.pos_pub = rospy.Publisher('/yolov5/pose', Pose, queue_size = 1)
        self.pos_msg = Pose()
        
        self.bridge = CvBridge() #OpenCV与ROS的消息转换类

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

    def detectimg(self, img, img_depth):
        frame = self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
        frame_depth = self.bridge.imgmsg_to_cv2(img_depth, desired_encoding='16UC1')
        # print(frame_depth.shape)
        result, names = self.yolo.detect([frame])
        image_detect = result[0][0]

        # print(result[0][1]) 直接打印box
        # 发送到result
        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = rospy.Time.now()
        # self.pos_msg.position.clear()
        for i in result[0][1]:
            name = names[int(i[0])]
            detection2d = Detection2D()
            # detection2d.is_tracking = True
            # detection2d.tracking_id = name
            x1, y1, x2, y2 = i[1]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            center_x = (x1+x2)/2.0
            center_y = (y1+y2)/2.0

            detection2d.bbox.center.x = center_x
            detection2d.bbox.center.y = center_y

            detection2d.bbox.size_x = float(x2-x1)
            detection2d.bbox.size_y = float(y2-y1)

            obj_pose = ObjectHypothesisWithPose()
            obj_pose.id = name
            obj_pose.score = float(i[2])

            # print(self.camera_info["k"])

            # px2xy
            point = [center_x, center_y]
            world_x, world_y = px2xy(point, self.camera_info["K"], self.camera_info["D"], 1)
            depth_ = frame_depth[int(center_y),int(center_x)]
            obj_pose.pose.pose.position.x = world_x * depth_ * 0.001
            obj_pose.pose.pose.position.y = world_y * depth_ * 0.001
            obj_pose.pose.pose.position.z =  depth_ * 0.001   #2D相机则显示,归一化后的结果,用户用时自行乘上深度z获取正确xy
            if depth_ == 0: # 无效深度
                obj_pose.pose.pose.position.z = -1 * 1.0
            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)
            self.pos_msg.position.x = world_x * depth_ * 0.001
            self.pos_msg.position.y = world_y * depth_ * 0.001
            self.pos_msg.position.z = obj_pose.pose.pose.position.z


        self.img_pub.publish(self.bridge.cv2_to_imgmsg(image_detect, "bgr8"))
        self.res_pub.publish(self.result_msg)
        self.pos_pub.publish(self.pos_msg)

        

            

        
if __name__ == '__main__':
    rospy.init_node("yolo_detect")
    rospy.loginfo("yolo_detect node started")
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
