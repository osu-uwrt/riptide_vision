import math
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

import torch
import torch.backends.cudnn as cudnn

from yolov5_ros.models.common import DetectMultiBackend
from yolov5_ros.utils.general import (LOGGER, check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5_ros.utils.plots import Annotator, colors
from yolov5_ros.utils.torch_utils import select_device, time_sync

from yolov5_ros.utils.datasets import letterbox

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from bboxes_ex_msgs.msg import BoundingBoxes
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import Quaternion, Point
from cv_bridge import CvBridge


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5 * xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5 * xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5 * xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5 * xywh[3]) * im_shape[0]

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


class yolov5_ros(Node):
    def __init__(self):
        super().__init__('yolov5_ros')

        self.bridge = CvBridge()

        self.pub_bbox = self.create_publisher(BoundingBoxes, 'yolov5/bounding_boxes', 10)
        self.pub_image = self.create_publisher(Image, 'yolov5/image_raw', 10)
        self.pub_detection = self.create_publisher(Image, '/tempest/yolov5/image_detection', 10)
        self.pub_det3d = self.create_publisher(Detection3DArray, '{}/yolo/detected_objects'.format(self.get_namespace()), 10)
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

        # from yolov5_demo
        self.s = str()
        self.load_model()

        self.zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_maximum_distance = 50
        stream = sl.StreamingParameters()
        stream.codec = sl.STREAMING_CODEC.H265  # Can be H264 or H265

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
        detection_parameters.enable_mask_output = True  # Outputs 2D masks over detected objects
        err = self.zed.enable_object_detection(detection_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            LOGGER.error(repr(err))
            self.zed.close()
            exit(1)

        self.objects = sl.Objects()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        self.obj_runtime_param.detection_confidence_threshold = 20

        # if the names are changed in pool.yaml, changes the IDs and lookup below
        self.rolledCaseIds = {0, 1, 4, 9}
        self.object_ids = {
            0: "BinBarrel",
            1: "BinPhone",
            2: "TommyGun",
            3: "gman",
            4: "axe",
            5: "torpedoGman",
            6: "badge",
            7: "torpedoBootlegger",
            8: "bootlegger",
            9: "cash"
        }

        LOGGER.info("Loaded Model")

        t = threading.Thread(target=self.publish_camera)
        t.start()

    # from yolov5_demo
    def image_callback(self, image_raw):
        class_list = []
        class_id = []
        confidence_list = []
        x_max_list = []
        x_min_list = []
        y_min_list = []
        y_max_list = []
        bounding_box_2d = []

        image_raw = self.bridge.imgmsg_to_cv2(image_raw, "bgr8")

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

            self.s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
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
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    class_id.append(c)
                    class_list.append(self.names[c])
                    confidence_list.append(conf)
                    # tensor to float
                    x_min_list.append(xyxy[0].item())
                    y_min_list.append(xyxy[1].item())
                    x_max_list.append(xyxy[2].item())
                    y_max_list.append(xyxy[3].item())
                    # Creating ingestable objects for the ZED SDK
                    bounding_box_2d.append(xywh2abcd(xywh, im0.shape))

            return class_id, class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list, bounding_box_2d

    # callback ==========================================================================

    # return ---------------------------------------
    # 1. class (str)                                +
    # 2. confidence (float)                         +
    # 3. x_min, y_min, x_max, y_max (float)         +
    # ----------------------------------------------

    # from yolov5_demo
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

        # Dataloader

        cudnn.benchmark = True
        bs = 1
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

    def publish_camera(self):
        image_left_tmp = sl.Mat()
        graberr = self.zed.grab(self.runtime_params)
        while graberr == sl.ERROR_CODE.SUCCESS:

            self.zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image = image_left_tmp.get_data()
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            ros_image = self.bridge.cv2_to_imgmsg(image, 'bgr8')
            ros_image.header.frame_id = "/tempest/stereo/left_optical"
            self.image_pub.publish(ros_image)
            class_id, class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list, bounding_box_2d = self.image_callback(ros_image)

            if not (len(class_id) == len(confidence_list)):
                LOGGER.warn(f"The number of ids returned, {len(class_id)}, is not equal to the number of detections, {len(confidence_list)}! ")

            classIds = []
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
                tmp.is_grounded = False  # objects are moving on the floor plane and tracked in 2D only
                objects_in.append(tmp)

            self.zed.ingest_custom_box_objects(objects_in)

            objects = sl.Objects()  # Structure containing all the detected objects
            self.zed.retrieve_objects(objects, self.obj_runtime_param)  # Retrieve the 3D tracked objects

            detections = Detection3DArray()
            detections.header.stamp.nanosec = self.get_clock().now().seconds_nanoseconds()[1]
            detections.header.stamp.sec = self.get_clock().now().seconds_nanoseconds()[0]

            detections.detections = []

            counter = 0  # use for label lookup
            for obj in objects.object_list:
                if counter < len(classIds):
                    detection = Detection3D()
                    detection.results = []
                    object_hypothesis = ObjectHypothesisWithPose()
                    object_hypothesis.hypothesis.class_id = 'test'
                    position = Point()

                    # flip coordinates
                    position.x = -obj.position[0]
                    position.y = -obj.position[1]
                    position.z = -obj.position[2]

                    object_hypothesis.pose.pose.position = position
                    LOGGER.info(f"Adjusted Position {position}")
                    LOGGER.info(f"Class Ids{self.object_ids[classIds[counter]]}")
                    object_hypothesis.hypothesis.class_id = self.object_ids[classIds[counter]]

                    ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
                    self.pub_detection.publish(ros_image)

                    # cannot determine orentation without 3Dbox
                    threeBoundingBox = obj.bounding_box

                    if len(threeBoundingBox) == 8:
                        flippedThreeBoundingBox = []
                        for point in threeBoundingBox:
                            flippedPoint = []
                            for coordinate in point:
                                flippedPoint.append(-coordinate)
                            flippedThreeBoundingBox.append(flippedPoint)

                        LOGGER.info(f"Bound: {flippedThreeBoundingBox}")

                        # determine the orientation of the object
                        object_orientation = Quaternion()
                        if classIds[counter] in self.rolledCaseIds:
                            # if robot is rolled forward

                            # from the back plane to the front plane -- accounting for roll forward
                            centerFrontPlane = [(flippedThreeBoundingBox[4][0] + flippedThreeBoundingBox[6][0]) / 2,
                                                (flippedThreeBoundingBox[4][1] + flippedThreeBoundingBox[6][1]) / 2,
                                                -(flippedThreeBoundingBox[4][2] + flippedThreeBoundingBox[6][2]) / 2]
                            centerBackPlane = [(flippedThreeBoundingBox[0][0] + flippedThreeBoundingBox[2][0]) / 2,
                                               (flippedThreeBoundingBox[0][1] + flippedThreeBoundingBox[2][1]) / 2,
                                               -(flippedThreeBoundingBox[0][2] + flippedThreeBoundingBox[2][2]) / 2]
                            arrowVector = [centerFrontPlane[0] - centerBackPlane[0], centerFrontPlane[1] - centerBackPlane[1], centerFrontPlane[2] - centerBackPlane[2]]
                            # vector = {x, y, z}

                            imageYaw = 0
                            if not arrowVector[1] == 0:
                                # stops a nan error

                                # this is the way the image is facing - not the orientation of the camera
                                imageYaw = math.atan(arrowVector[0] / arrowVector[1])

                            # we don't care about x,z and w
                            object_orientation.x = 0.0
                            object_orientation.y = 0.0
                            object_orientation.z = imageYaw
                            object_orientation.w = 0.0

                        else:
                            # if robot not rolled forward

                            # from the back plane to the front plane
                            centerFrontPlane = [(flippedThreeBoundingBox[0][0] + flippedThreeBoundingBox[7][0]) / 2,
                                                (flippedThreeBoundingBox[0][1] + flippedThreeBoundingBox[7][1]) / 2,
                                                -(flippedThreeBoundingBox[0][2] + flippedThreeBoundingBox[7][2]) / 2]
                            centerBackPlane = [(flippedThreeBoundingBox[1][0] + flippedThreeBoundingBox[6][0]) / 2,
                                               (flippedThreeBoundingBox[1][1] + flippedThreeBoundingBox[6][1]) / 2,
                                               -(flippedThreeBoundingBox[1][2] + flippedThreeBoundingBox[6][2]) / 2]
                            arrowVector = [centerFrontPlane[0] - centerBackPlane[0], centerFrontPlane[1] - centerBackPlane[1], centerFrontPlane[2] - centerBackPlane[2]]
                            # vector = {x, y, z}

                            imageYaw = 0
                            if not arrowVector[2] == 0:
                                # stops a nan error

                                # this is the way the image is facing - not the orientation of the camer
                                imageYaw = math.atan(arrowVector[0] / arrowVector[2])

                            # we dont care about x,z and w
                            object_orientation.x = 0.0
                            object_orientation.y = 0.0
                            object_orientation.z = imageYaw
                            object_orientation.w = 0.0

                        LOGGER.info(f"Yaw: {imageYaw}")

                        object_hypothesis.pose.pose.orientation = object_orientation

                        # returns score between 0 and 100 -> score wants between 0 and 1
                        object_hypothesis.hypothesis.score = obj.confidence / 100
                        LOGGER.info(obj.confidence)

                        # Mapping will reject in two objects in one place
                        detection.results.append(object_hypothesis)
                        detections.detections.append(detection)

                        counter += 1

                LOGGER.warn(f"Threw out {len(objects.object_list) - counter} detections.")

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
