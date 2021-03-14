#! /usr/bin/env python

import rospy

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time

from geometry_msgs.msg import Quaternion, PoseStamped, Point, PoseWithCovarianceStamped
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import ObjectHypothesisWithPose, Detection3D


last_time = 0

def imgCB(msg):
    global bridge
    global last_time

    if time.time() - last_time < .5:
        return

    try:
        cv_image = bridge.imgmsg_to_cv2(msg.image)
    except CvBridgeError as e:
        print(e)

    (rows,cols) = cv_image.shape

    # Add up all the pixels in each column
    integral = cv2.integral(cv_image)
    score_img = np.zeros((1,cols,1), np.float32)
    x = 0
    maxScore = 0
    for c in range(cols):
        score = (integral[rows, c+1] - integral[rows,c])/rows
        if (score > maxScore):
            maxScore = score
            x = c
        elif score < 0:
            score = 0
        score_img[0,c] = score

    column_pub.publish(bridge.cv2_to_imgmsg(score_img))

    # Threshold only the pixels more than 2.5 std deviations above the mean
    [mean, std] = cv2.meanStdDev(score_img)
    _, thresh = cv2.threshold(score_img, int(mean+2.5*std), 500000, cv2.THRESH_TOZERO)

    # Fill in the gaps
    kernel = np.ones((1,15),np.float32)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = thresh / thresh.max() * 255
    thresh = thresh.astype(np.uint8)

    thresh_pub.publish(bridge.cv2_to_imgmsg(thresh))

    # Find each blob in the threshold image
    _, contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    original_pub.publish(msg.image)

    # If we have at least one contour in our threshold image
    if len(contours) != 0:
        # Find the biggest area. Turns out all contours have an area of 0 so it picks a random one.
        # It works so I'm not changing it
        c = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)
        
        if w > 15:

            # Select the sample region of original image to determine distance
            sample_region = cv_image[rows//4:rows*3//4, x:x+w].reshape((-1,1))
            
            # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            # Set flags (Just to avoid line break in the code)
            flags = cv2.KMEANS_RANDOM_CENTERS

            # Apply KMeans
            _,labels,centers = cv2.kmeans(sample_region,4,None,criteria,10,flags)

            # Find which disparity is the most common in the sample region
            labels = [l[0] for l in labels]
            maxLabel = max(set(labels), key=labels.count)
            disparity = centers[maxLabel][0]

            # If negative, look for next most often
            if disparity < 0:
                labels = [l for l in labels if l != maxLabel]
                maxLabel = max(set(labels), key=labels.count)
                disparity = centers[maxLabel][0]

            # Find the distance from camera in meters
            z = msg.f * msg.T / disparity

            # Extract camera matrix parameters
            fx = k[0, 0]
            fy = k[1, 1]
            cx = k[0, 2]
            cy = k[1, 2]

            # Compute the camera-frame coordinates. https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
            start_vector = np.array([x + w/2, 0, 1, 1/z]).T

            k_inv_eff = np.array([[1/fx, 0, -cx/fx, 0],
                                  [0, 1/fy, -cy/fy, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

            coordinates = z * np.dot(k_inv_eff, start_vector)

            rospy.loginfo_throttle(.5, str(coordinates))

            hypothesis = ObjectHypothesisWithPose()

            hypothesis.pose.pose.position = Point(*coordinates[:3])
            hypothesis.pose.pose.orientation = Quaternion(.5, -.5, .5, .5)

            covariance = np.zeros((6,6))
            covariance[0,0] = 0.1**2
            covariance[1,1] = 10000
            covariance[2,2] = 0.1**2
            covariance[3,3] = 0
            covariance[4,4] = 0
            covariance[5,5] = 10000

            hypothesis.pose.covariance = covariance.ravel()

            hypothesis.id = 0
            hypothesis.score = 0

            msg = Detection3D()
            msg.results.append(hypothesis)
            msg.header.frame_id = "puddles/stereo/left_optical"	
            msg.header.stamp = rospy.get_rostime()


            pose = PoseWithCovarianceStamped()

            pose.pose = hypothesis.pose
            pose.header = msg.header

            detection_pub.publish(msg)
            pose_pub.publish(pose)

            last_time = time.time()
    
    

def cam_info_cb(msg):
    global k
    k = np.array(msg.K).reshape((3,3))
    

rospy.init_node("pole_processor")
rospy.Subscriber("stereo/disparity", DisparityImage, imgCB)
rospy.Subscriber("stereo/left/camera_info", CameraInfo, cam_info_cb)


column_pub = rospy.Publisher("debug/column", Image, queue_size=5)
original_pub = rospy.Publisher("debug/original", Image, queue_size=5)
thresh_pub = rospy.Publisher("debug/thresh", Image, queue_size=5)
detection_pub = rospy.Publisher("pole_detection", Detection3D, queue_size=5)
pose_pub = rospy.Publisher("pole_pose", PoseWithCovarianceStamped, queue_size=5)
bridge = CvBridge()
rospy.spin()

