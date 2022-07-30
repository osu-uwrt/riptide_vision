import argparse
from distutils.log import Log, debug
from logging import Logger
import math
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
#from sensor_msgs_py import point_cloud2 as pc2

import torch
from torch import hypot
import torch.backends.cudnn as cudnn

from yolov5_ros.models.common import DetectMultiBackend
from yolov5_ros.utils.datasets import IMG_FORMATS, VID_FORMATS
from yolov5_ros.utils.general import (LOGGER, check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5_ros.utils.plots import Annotator, colors
from yolov5_ros.utils.torch_utils import select_device, time_sync

from yolov5_ros.utils.datasets import letterbox

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import Quaternion, Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import image_geometry
from transforms3d import euler

import ctypes
import struct
from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

bridge = CvBridge()

def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output

class yolov5_demo():
    def __init__(self,  weights,
                        data,
                        imagez_height,
                        imagez_width,
                        conf_thres,
                        iou_thres,
                        max_det,
                        device,
                        view_img,
                        classes,
                        agnostic_nms,
                        line_thickness,
                        half,
                        dnn
                        ):
        self.weights = weights
        self.data = data
        self.imagez_height = imagez_height
        self.imagez_width = imagez_width
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness
        self.half = half
        self.dnn = dnn

        self.s = str()

        self.load_model()

    def load_model(self):
        imgsz = (self.imagez_height, self.imagez_width)

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        source = 0
        # Dataloader
        webcam = False
        if webcam:
            view_img = check_imshow()
        cudnn.benchmark = True
        bs = 1
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

    # callback ==========================================================================

    # return ---------------------------------------
    # 1. class (str)                                +
    # 2. confidence (float)                         +
    # 3. x_min, y_min, x_max, y_max (float)         +
    # ----------------------------------------------
    def image_callback(self, image_raw):
        class_list = []
        class_id = []
        confidence_list = []
        x_max_list = []
        x_min_list = []
        y_min_list = []
        y_max_list = []
        zed_obj = []
        bounding_box_2d = []

        image_raw = bridge.imgmsg_to_cv2(image_raw, "bgr8")
        #LOGGER.info("Converted new image")
        # im is  NDArray[_SCT@ascontiguousarray
        # im = im.transpose(2, 0, 1)
        self.stride = 32  # stride
        self.img_size = 1920
        img = letterbox(image_raw, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        save_dir = "runs/detect/exp7"
        path = ['0']

        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        self.dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):
            im0 = image_raw
            self.s += f'{i}: '

            # p = Path(str(p))  # to Path
            self.s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    save_conf = False
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                     # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # print(xyxy, label)
                    class_id.append(c)
                    class_list.append(self.names[c])
                    confidence_list.append(conf)
                    # tensor to float
                    x_min_list.append(xyxy[0].item())
                    y_min_list.append(xyxy[1].item())
                    x_max_list.append(xyxy[2].item())
                    y_max_list.append(xyxy[3].item())
                     # Creating ingestable objects for the ZED SDK
                    # obj = sl.CustomBoxObjectData()
                    bounding_box_2d.append(xywh2abcd(xywh, im0.shape))
                    # obj.label = cls
                    # obj.probability = conf
                    # obj.is_grounded = False
                    

            # Stream results
            im0 = annotator.result()
            #LOGGER.info("Got Detections")
            #if self.view_img:
                #cv2.imshow("yolov5", im0)
                #cv2.waitKey(1)  # 1 millisecond
            #LOGGER.info("Returned image")
            
            return class_id, class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list, bounding_box_2d

class yolov5_ros(Node):
    def __init__(self):
        super().__init__('yolov5_ros')

        self.bridge = CvBridge()

        self.pub_bbox = self.create_publisher(BoundingBoxes, 'yolov5/bounding_boxes', 10)
        self.pub_image = self.create_publisher(Image, 'yolov5/image_raw', 10)
        self.pub_detection = self.create_publisher(Image, 'yolov5/image_detection', 10)
        self.pub_det3d = self.create_publisher(Detection3DArray, 'dope/detected_objects', 10)
        # self.sub_image = self.create_subscription(Image, '/tempest/stereo/left_raw/image_raw_color', self.image_callback,1)
        # self.right_sub_image = self.create_subscription(Image, '/tempest/stereo/right_raw/image_raw_color', self.empty_callback,1)
        #self.sub_depth = self.create_subscription(Image, '/tempest/stereo/depth/depth_registered', self.depth_callback, 1)
        # self.left_infosub = self.create_subscription(CameraInfo, '/tempest/stereo/left_raw/camera_info', self.left_info_callback, 1)
        # self.right_infosub = self.create_subscription(CameraInfo, '/tempest/stereo/right_raw/camera_info', self.right_info_callback, 1)
        # self.sub_disparity = self.create_subscription(DisparityImage, '/tempest/stereo/disparity/disparity_image', self.disparity_callback, 1)
        self.image_pub = self.create_publisher(Image, '/tempest/stereo/left_raw/image_raw_color', 5)
        self.camera_model = None
        self.left_info = None
        self.right_info = None
        self.disparity_latest = None
        self.pc2_latest = None
        # parameter
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

        self.declare_parameter('weights', str(ROOT) + '/config/yolov5s.pt')
        self.declare_parameter('data', str(ROOT) + '/data/coco128.yaml')
        self.declare_parameter('imagez_height', 1080)
        self.declare_parameter('imagez_width', 1920)
        self.declare_parameter('conf_thres', 0.65)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 1000)
        self.declare_parameter('device', '')
        self.declare_parameter('view_img', False)
        self.declare_parameter('classes', None)
        self.declare_parameter('agnostic_nms', False)
        self.declare_parameter('line_thickness', 3)
        self.declare_parameter('half', False)
        self.declare_parameter('dnn', False)

        self.weights = self.get_parameter('weights').value
        self.data = self.get_parameter('data').value
        self.imagez_height = self.get_parameter('imagez_height').value
        self.imagez_width = self.get_parameter('imagez_width').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.device = self.get_parameter('device').value
        self.view_img = self.get_parameter('view_img').value
        self.classes = self.get_parameter('classes').value
        self.agnostic_nms = self.get_parameter('agnostic_nms').value
        self.line_thickness = self.get_parameter('line_thickness').value
        self.half = self.get_parameter('half').value
        self.dnn = self.get_parameter('dnn').value

        self.zed = sl.Camera()
        
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_maximum_distance = 50
        stream = sl.StreamingParameters()
        stream.codec = sl.STREAMING_CODEC.H265 # Can be H264 or H265

        self.runtime_params = sl.RuntimeParameters()
        status = self.zed.open(init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            LOGGER.error(repr(status))
            exit()

        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
        # positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(positional_tracking_parameters)

        detection_parameters = sl.ObjectDetectionParameters()
        detection_parameters.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        detection_parameters.enable_tracking = True
        detection_parameters.enable_mask_output = True # Outputs 2D masks over detected objects
        err = self.zed.enable_object_detection(detection_parameters)
        if err != sl.ERROR_CODE.SUCCESS :
            LOGGER.error(repr(err))
            self.zed.close()
            exit(1)

        self.objects = sl.Objects()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        self.obj_runtime_param.detection_confidence_threshold = 20

        self.object_ids = self.data.names
        self.rolledIds = self.data.rolledCaseIds
        if len(self.object_ids) == 0:
            LOGGER.fatal("Id lookup did not import from yaml!")
            

        self.yolov5 = yolov5_demo(self.weights,
                                self.data,
                                self.imagez_height,
                                self.imagez_width,
                                self.conf_thres,
                                self.iou_thres,
                                self.max_det,
                                self.device,
                                self.view_img,
                                self.classes,
                                self.agnostic_nms,
                                self.line_thickness,
                                self.half,
                                self.dnn)
        LOGGER.info("Loaded Model")

        t = threading.Thread(target=self.publish_camera)
        t.start()
    
    def empty_callback(self, img):
        pass

    def left_info_callback(self, data):
        # Get a camera model object using image_geometry and the camera_info topic
        #LOGGER.info("LeftCB")
        self.left_info = data
        if self.right_info is not None:
            self.camera_model = image_geometry.StereoCameraModel()
            self.camera_model.fromCameraInfo(data, self.right_info)
        #self.destroy_subscription(self.left_infosub)#Only subscribe once
        
    def right_info_callback(self, data):
        #LOGGER.info("RightCB")
        self.right_info = data
        if self.left_info is not None:            
            self.camera_model = image_geometry.StereoCameraModel()
            self.camera_model.fromCameraInfo(self.left_info, data)
        #self.destroy_subscription(self.right_infosub) #Only subscribe once

    def yolovFive2bboxes_msgs(self, bboxes:list, scores:list, cls:list, img_header:Header):
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = img_header
        print(bboxes)
        # print(bbox[0][0])
        i = 0
        for score in scores:
            one_box = BoundingBox()
            one_box.xmin = int(bboxes[0][i])
            one_box.ymin = int(bboxes[1][i])
            one_box.xmax = int(bboxes[2][i])
            one_box.ymax = int(bboxes[3][i])
            one_box.probability = float(score)
            one_box.class_id = cls[i]
            bboxes_msg.bounding_boxes.append(one_box)
            i = i+1
        
        return bboxes_msg

    def disparity_callback(self,msg):
        self.disparity_latest = msg.image

    def pc2_callback(self,msg):
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        #self.lock.acquire()
        gen = pc2.read_points(msg, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            test = x[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
            # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
            rgb = np.append(rgb,[[r,g,b]], axis = 0)
        

        self.pc2_latest = xyz

    def get_obj_msg(self, bbox, header, ids, confidences, img):
        detmsg = Detection3DArray()
        detmsg.header = header
        detmsg.detections = []
       # LOGGER.info("About to start loop")

        if self.disparity_latest:
            for idx, x_min in enumerate(bbox[0]):        
                        
                det3d = Detection3D()   

                #get world frame position 
                x = (bbox[2][idx] + bbox[0][idx]) /2
                y = (bbox[3][idx] + bbox[1][idx]) /2  
                #get disparity image 
                disp = self.bridge.imgmsg_to_cv2(self.disparity_latest, "passthrough")
                LOGGER.info(f"Y: {int(y)} , X: {int(x)}")
                #LOGGER.info(f"bbox: X: {bbox[0][idx]}, {bbox[2][idx]}, Y: {bbox[1][idx]} , {bbox[3][idx]}")
                LOGGER.info(ids[idx])
                #log center of circle
                img = cv2.circle(img, (int(x),int(y)), radius=8, color=(0, 0, 255), thickness=-1)

                pixelFrameCenterPoint = (int(x), int(y))
                #order matters here
                #top left corner
                innerratio = 7

                lUx = bbox[0][idx] + ((bbox[2][idx] - bbox[0][idx])/innerratio)
                lUy = bbox[1][idx] + ((bbox[3][idx] - bbox[1][idx])/innerratio)
                lu = (lUx, lUy)
                
                #bottom left corner
                lDx = bbox[0][idx] + ((bbox[2][idx] - bbox[0][idx])/innerratio)
                lDy = bbox[3][idx] - ((bbox[3][idx] - bbox[1][idx])/innerratio)
                ld = (lDx, lDy)
                
                #bottom right corner
                rDx = bbox[2][idx] - ((bbox[2][idx] - bbox[0][idx])/innerratio)
                rDy = bbox[3][idx] - ((bbox[3][idx] - bbox[1][idx])/innerratio)
                rd = (rDx, rDy)

                #top right corner
                rUx = bbox[2][idx] - ((bbox[2][idx] - bbox[0][idx])/innerratio)
                rUy = bbox[1][idx] + ((bbox[3][idx] - bbox[1][idx])/innerratio)
                ru = (rUx, rUy)

                
                #get all pixel positions in camera frame
                pixel_points = [lu, ld, rd ,ru]

                camera_frame_points = []
                for p in pixel_points:
                    tX, tY = p
                    #print on image
                    img = cv2.circle(img, (int(tX),int(tY)), radius=4, color=(0, 0, 255), thickness=-1)
                    #compute cam frame positions
                    
                    wx,wy,wz = self.camera_model.projectPixelTo3d((tX,tY), disp[int(tY),int(tX)]) 
                    #LOGGER.info(f" Points: {tX} , {tY}, 3D: {wx} , {wy}, {wz}, {disp.shape}, {disp[int(tY),int(tX)]}")               
                    wx = -wx
                    wy = -wy
                    wz = -wz
                    if not math.isnan(disp[int(tY),int(tX)]):
                        camera_frame_points.append([wx,wy,wz])
                
                
                LOGGER.info(camera_frame_points)

                #center point
                x,y,z = self.camera_model.projectPixelTo3d((x,y), disp[int(y),int(x)])
                
                x = -x
                y = -y
                z = -z

                center = [x,y,z]

                candidate_points = []
                for up, vp in zip(camera_frame_points[:-1],camera_frame_points[1:]):
                    if (vp is not None) and (not math.isnan(center[0])):
                        #LOGGER.info(f"U: {up} , V: {vp}")
                        u = np.subtract(up, center)
                        v = np.subtract(vp, center)
                        #LOGGER.info(f"u: {u}")
                        #LOGGER.info(f"v: {v}")
                        #cross product
                        norm = np.cross(v,u)
                        #LOGGER.info(norm)
                        #lets print it here
                        norm = self.normalize(norm)
                        #LOGGER.info(norm)
                        point = np.subtract(norm, center)
                        pixelFrameNormPoint = self.camera_model.project3dToPixel(point)[0]
                        #LOGGER.info(f"Arrays: {pixelFrameNormPoint}, {pixelFrameCenterPoint} | Tuples: {tuple(pixelFrameNormPoint)}, {tuple(pixelFrameCenterPoint)}")
                        if not math.isnan(pixelFrameNormPoint[0]):
                            pixelFrameNormPoint = (int(pixelFrameNormPoint[0]), int(pixelFrameNormPoint[1])) 
                            candidate_points.append(point)
                            #debug lines of candidates                   
                            img = cv2.line(img, tuple(pixelFrameCenterPoint), tuple(pixelFrameNormPoint), color=(0, 0, 255), thickness=2)


                #TODO quaternion calcs
                #loop through clockwise and subtract from center to get vectors
                #Cross Multiply!
                #print that vector onto image
                #get quaternion from where v1 is (1,0,0)
                '''
                Quaternion q;
                vector a = crossproduct(v1, v2);
                q.xyz = a;
                q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
                '''
                candidate_orientations = []
                v1 = [0,-1,0]
                v3 = [1,0,0]
                #LOGGER.info(f'candidate_points: {candidate_points}')
                average_quaternion = None
                candidate_points_filtered = filter(lambda point: not math.isnan(point[0]) and not math.isnan(point[1]) and not math.isnan(point[2]), candidate_points)
                # candidate_points_filtered = filter(lambda point: not math.isnan(point), candidate_points)
                candidate_points_filtered = list(candidate_points_filtered)
                if len(candidate_points_filtered) > 0:
                    average_orientation = np.mean(candidate_points_filtered, axis = 0)
                    v2 = self.normalize(average_orientation)
                    LOGGER.info(f'average_orientation:{average_orientation} ')

                    vectProj = (np.dot(v2, v1)) * np.array(v1)
                    planeProj = v1 - vectProj

                    yaw = math.acos((np.dot(planeProj, v3)/(np.linalg.norm(planeProj))))
                    r = 0
                    p = 0
                    y = yaw

                    LOGGER.info(f"Yaw: {yaw}")

                    w,x,y,z = euler.euler2quat(r,p,y,axes='sxyz')
                    average_quaternion = [x,y,z,w]
                    
                    # Roll, Pitch, and Yaw configuration
                    # #angle = np.dot(average_orientation,point)/(np.linalg.norm(average_orientation)*np.linalg.norm(point))
                    # axis = np.cross(v1, average_orientation)
                    # sinA = np.linalg.norm(axis) / (np.linalg.norm(v1) * np.linalg.norm(average_orientation))
                    # cosA = (np.dot(v1, average_orientation)) / (np.linalg.norm(v1) * np.linalg.norm(average_orientation))
                    # average_quaternion = self.normalize(np.array([axis[0] * sinA, axis[1] * sinA ,axis[2] * sinA,cosA]))


                    

                # for point in candidate_points:

                #     #q = Quaternion()
                #     axis = np.cross(v1, point)
                #     angle = np.dot(v1,point)/(np.linalg.norm(v1)*np.linalg.norm(point))
                #     s = math.sin(angle/2)
                #     point[0] = axis[0] * s
                #     point[1] = axis[1] * s
                #     point[2] = axis[2] * s
                    

                #     point = self.normalize(np.array([point[0], point[1],point[2],math.cos(angle/2)]))
                #     # q.x = point[0]
                #     # q.y = point[1]
                #     # q.z = point[2]
                #     # q.w = point[3]
                #     candidate_orientations.append(point)


                #LOGGER.info(candidate_orientations[0])
                
                    
                # weights = np.ones(len(candidate_orientations))

                #Default is pointing towards camera
                orientation = Quaternion()
                orientation.x = 0.0
                orientation.y = 0.707
                orientation.z = 0.0
                orientation.w = 0.707

                if average_quaternion is not None:                                     
                    
                    orientation.x = average_quaternion[0]
                    orientation.y = average_quaternion[1]
                    orientation.z = average_quaternion[2]
                    orientation.w = average_quaternion[3]
                    LOGGER.info(f"Orientation: {orientation}")

                
    
                det3d.bbox.center.position.x = x
                det3d.bbox.center.position.y = y
                det3d.bbox.center.position.z = z
                det3d.results = []
                #LOGGER.info("setbbox msg")
                #LOGGER.info(det3d.bbox)
                hyp = ObjectHypothesisWithPose()
                #LOGGER.info("Made object hypothesis")
                #LOGGER.info(hyp)
                hyp.pose.pose = det3d.bbox.center
                #hyp.pose.pose.orientation = orientation
                hyp.hypothesis.class_id = ids[idx]
                hyp.hypothesis.score = float(confidences[idx])
                #LOGGER.info("Filled in Hypothesis")
                #LOGGER.info(hyp)
                det3d.results.append(hyp)
                detmsg.detections.append(det3d)
            
            #LOGGER.info(detmsg)
        return detmsg, img

    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    # def image_callback(self, image:Image):
    #     if self.camera_model:       
    #         image_raw = image
    #         # return (class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
    #         class_id, class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list, im0, zed_obj = self.yolov5.image_callback(image_raw)
    #         bboxes=[x_min_list, y_min_list, x_max_list, y_max_list]
    #         msg = self.yolovFive2bboxes_msgs(bboxes=[x_min_list, y_min_list, x_max_list, y_max_list], scores=confidence_list, cls=class_list, img_header=image.header)
    #         self.pub_bbox.publish(msg)
    #         #LOGGER.info(zed_obj)
            

    #         # self.zed.ingest_custom_box_objects(zed_obj)
    #         # LOGGER.info("Ingested Detections")
    #         # self.zed.retrieve_objects(self.objects, self.obj_runtime_param)
    #         # LOGGER.info("Retrieved object data")
            
    #         objmsg, im0= self.get_obj_msg(bboxes ,image.header, class_list, confidence_list, im0)
    #         self.pub_image.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8")) 
    #         self.pub_det3d.publish(objmsg)
            

    #         print("start ==================")
    #         print(class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
    #         print("end ====================")
    
    def publish_camera(self):
        image_left_tmp = sl.Mat()
        graberr = self.zed.grab(self.runtime_params) 
        while graberr == sl.ERROR_CODE.SUCCESS:
            
            self.zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            #LOGGER.info("FUCK")
            image = image_left_tmp.get_data()
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            ros_image = self.bridge.cv2_to_imgmsg(image,'bgr8')
            #LOGGER.info("Got data")
            self.image_pub.publish(ros_image)
            class_id, class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list, bounding_box_2d = self.yolov5.image_callback(ros_image)

            if not (len(class_id) == len(confidence_list)):
                LOGGER.warn(f"The number of ids returned, {len(class_id)}, is not equal to the number of detections, {len(confidence_list)}! ")

            classIds = []
            boudningRects = []
            objects_in = []
            # The "detections" variable contains your custom 2D detections
            for i in range(len(class_id)):
                tmp = sl.CustomBoxObjectData()
                # Fill the detections into the correct SDK format
                tmp.unique_object_id = sl.generate_unique_id()
                tmp.probability = confidence_list[i]
                tmp.label = class_id[i]
                classIds.append(class_id[i])
                tmp.bounding_box_2d = bounding_box_2d[i]
                boudningRects.append(bounding_box_2d[i])
                # tmp.bounding_box_2d = [[x_min_list[i],[x_max_list[i]]],[y_min_list[i], y_max_list[i]]]
                tmp.is_grounded = False # objects are moving on the floor plane and tracked in 2D only
                objects_in.append(tmp)
            self.zed.ingest_custom_box_objects(objects_in)

            objects = sl.Objects() # Structure containing all the detected objects
            self.zed.retrieve_objects(objects, self.obj_runtime_param) # Retrieve the 3D tracked objects

            detections = Detection3DArray()
            detections.detections = []
            
            counter = 0 #use for label lookup
            for obj in objects.object_list:
                if (counter < len(classIds)):
                    object_id = obj.id # Get the object id
                    object_position = obj.position # Get the object position
                    object_tracking_state = obj.tracking_state # Get the tracking state of the object
                    # if object_tracking_state == sl.OBJECT_TRACK_STATE.OK :
                    #     print("Object {0} is tracked\n".format(object_id))
                    detection = Detection3D()
                    detection.results = []
                    object_hypothesis = ObjectHypothesisWithPose()
                    object_hypothesis.hypothesis.class_id = 'test'
                    position = Point()
                    object_hypothesis.pose.pose.position = position
                    # print(obj.position)
                    LOGGER.info(obj.position)
                    LOGGER.info(classIds[counter])
                    detection.hypothese.class_id = self.object_ids[classIds[counter]]

                    # draw cv rect
                    rect = boudningRects[counter]
                    x = rect[3][0]
                    y = rect[3][1]
                    w = rect[1][0] - x
                    h = rect[1][1] - y
                    cv2.rectangle(image, (x, y), (x+w, y+h),(0, 250, 0), 2)

                    ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
                    self.pub_detection.publish(ros_image)

                    threeBoundingBox = obj.bounding_box
            
                    #determine the orientation of the object
                    object_orientation = Quaternion()
                    if classIds[counter] in self.rolledIds:
                        # if robot is rolled forward

                        #from the back plane to the front plane -- accounting for roll forward
                        centerFrontPlane = {(threeBoundingBox[4][0] + threeBoundingBox[6][0]) / 2, (threeBoundingBox[4][1] + threeBoundingBox[6][1]) / 2, (threeBoundingBox[4][2] + threeBoundingBox[6][2]) / 2}
                        centerBackPlane = {(threeBoundingBox[0][0] + threeBoundingBox[2][0]) / 2, (threeBoundingBox[0][1] + threeBoundingBox[2][1]) / 2, (threeBoundingBox[0][2] + threeBoundingBox[2][2]) / 2}
                        vector = {centerFrontPlane[0] - centerBackPlane[0], centerFrontPlane[1] - centerBackPlane[1], centerFrontPlane[2] - centerBackPlane[2]}
                        #vector = {x, y, z}

                        imageYaw = 0
                        if not vector[1] == 0:
                            #stops a nan error

                            #this is the way the image is facing - not the orientation of the camer
                            imageYaw = math.atan(vector[0] / vector[1])

                        # we dont care about x,z and w
                        object_orientation.x = 0
                        object_orientation.y = 0 
                        object_orientation.z = imageYaw
                        object_orientation.w = 0

                    else:
                        #if robot not rolled forward

                        #from the back plane to the front plane
                        centerFrontPlane = {(threeBoundingBox[0][0] + threeBoundingBox[7][0]) / 2, (threeBoundingBox[0][1] + threeBoundingBox[7][1]) / 2, (threeBoundingBox[0][2] + threeBoundingBox[7][2]) / 2}
                        centerBackPlane = {(threeBoundingBox[1][0] + threeBoundingBox[6][0]) / 2, (threeBoundingBox[1][1] + threeBoundingBox[6][1]) / 2, (threeBoundingBox[1][2] + threeBoundingBox[6][2]) / 2}
                        vector = {centerFrontPlane[0] - centerBackPlane[0], centerFrontPlane[1] - centerBackPlane[1], centerFrontPlane[2] - centerBackPlane[2]}
                        #vector = {x, y, z}

                        imageYaw = 0
                        if not vector[2] == 0:
                            #stops a nan error

                            #this is the way the image is facing - not the orientation of the camer
                            imageYaw = math.atan(vector[0] / vector[2])

                        # we dont care about x,z and w
                        object_orientation.x = 0
                        object_orientation.y = 0 
                        object_orientation.z = imageYaw
                        object_orientation.w = 0

                    object_hypothesis.pose.pose.orientation = object_orientation
                    
                    #returns score between 0 and 100 -> score wants between 0 and 1
                    object_hypothesis.score = obj.confidence / 100
                    
                    # hypothesis =           ObjectHypothesis()imageRelativeYaw
                    # hyp.hypothesis.class_id = ids[idx]
                    # hyp.hypothesis.score = float(confidences[idx])
                    # hypothesis.id = 1
                    # object_hypothesis.hypothesis
                    # hypothesis.class_id = 1
                    # hypothesis.id = 'this is the dope int id'


                    #Mapping will reject in two objects in one place
                    detection.results.append(object_hypothesis)
                    detections.detections.append(detection)
                    
                    counter += 1

                LOGGER.warn("Threw out {} detections.", len(objects.object_list) - counter)

            
            self.pub_det3d.publish(detections)


            graberr = self.zed.grab(self.runtime_params) 

        LOGGER.error(repr(graberr))
        return

def ros_main(args=None):
    rclpy.init(args=args)
    yolov5_node = yolov5_ros()
    
    rclpy.spin(yolov5_node)
    yolov5_ros.zed.close()
    yolov5_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    ros_main()