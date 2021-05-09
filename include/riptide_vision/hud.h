#ifndef HUD_H
#define HUD_H

#include "ros/ros.h"
#include "cmath"
#include "std_msgs/Header.h"
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Accel.h"
#include "sensor_msgs/image_encodings.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
using namespace cv;
using namespace std;

class HUD
{
 private:
  ros::NodeHandle nh;
  ros::Subscriber imu_sub, depth_sub, stereo_img_sub, down_img_sub, darknet_img_sub, reset_sub;
  ros::Subscriber cmd_roll_sub, cmd_pitch_sub, cmd_yaw_sub, cmd_depth_sub, cmd_x_sub, cmd_y_sub, cmd_z_sub, object_sub;
  image_transport::Publisher stereo_img_pub, down_img_pub, darknet_img_pub;

  geometry_msgs::Vector3 euler_rpy, cmd_euler_rpy, linear_accel;
  double depth, cmd_depth, cmd_x, cmd_y, cmd_z;
  bool reset = false;

  int width, height, top_margin, num_rows, offset, text_start[4];
  Scalar margin_color, text_color;

 public:
  HUD();
  void InitMsgs();
  void StereoImgCB(const sensor_msgs::ImageConstPtr& msg);
  void DownwardImgCB(const sensor_msgs::ImageConstPtr& msg);
  void DarknetImgCB(const sensor_msgs::ImageConstPtr& msg);
  Mat CreateHUD(Mat &img);

  void OdomCB(const nav_msgs::Odometry::ConstPtr& odom_msg);
  void ForceXCB(const std_msgs::Float64::ConstPtr& msg);
  void ForceYCB(const std_msgs::Float64::ConstPtr& msg);
  void ForceZCB(const std_msgs::Float64::ConstPtr& msg);
  void Loop();
};

#endif
