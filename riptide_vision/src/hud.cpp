#include "riptide_vision/hud.h"
#include "image_transport/image_transport.h"

#define GRAVITY 9.81 // [m/s^2]
#define WATER_DENSITY 1000 // [kg/m^3]

int main(int argc, char** argv)
{
  ros::init(argc, argv, "hud");
  HUD hud;
  hud.Loop();
}

// Heads Up Display
HUD::HUD() : nh("hud") {
  stereo_img_sub = nh.subscribe<sensor_msgs::Image>("/stereo/left/image_rect_color", 1, &HUD::StereoImgCB, this);
  down_img_sub = nh.subscribe<sensor_msgs::Image>("/downward/image_rect_color", 1, &HUD::DownwardImgCB, this);
  darknet_img_sub = nh.subscribe<sensor_msgs::Image>("/darknet_ros/detection_image", 1, &HUD::DarknetImgCB, this);
  imu_sub = nh.subscribe<riptide_msgs::Imu>("/state/imu", 1, &HUD::ImuCB, this);
  depth_sub = nh.subscribe<riptide_msgs::Depth>("/state/depth", 1, &HUD::DepthCB, this);
  object_sub = nh.subscribe<riptide_msgs::Object>("/state/object", 1, &HUD::ObjectCB, this);

  cmd_roll_sub = nh.subscribe<riptide_msgs::AttitudeCommand>("/command/roll", 1, &HUD::CmdRollCB, this);
  cmd_pitch_sub = nh.subscribe<riptide_msgs::AttitudeCommand>("/command/pitch", 1, &HUD::CmdPitchCB, this);
  cmd_yaw_sub = nh.subscribe<riptide_msgs::AttitudeCommand>("/command/yaw", 1, &HUD::CmdYawCB, this);
  cmd_depth_sub = nh.subscribe<riptide_msgs::DepthCommand>("/command/depth", 1, &HUD::CmdDepthCB, this);
  cmd_x_sub = nh.subscribe<std_msgs::Float64>("/command/force_x", 1, &HUD::ForceXCB, this);
  cmd_y_sub = nh.subscribe<std_msgs::Float64>("/command/force_y", 1, &HUD::ForceYCB, this);
  cmd_z_sub = nh.subscribe<std_msgs::Float64>("/command/force_z", 1, &HUD::ForceZCB, this);
  reset_sub = nh.subscribe<riptide_msgs::ResetControls>("/controls/reset", 1, &HUD::ResetCB, this);

  // Outputs
  image_transport::ImageTransport it(nh);
  stereo_img_pub = it.advertise("/stereo/left/image_hud", 1);
  down_img_pub = it.advertise("/downward/image_hud", 1);
  darknet_img_pub = it.advertise("/darknet_ros/image_hud", 1);

  top_margin = 120;
  num_rows = 4;
  offset = top_margin/15;
  for(int i = 0; i < num_rows; i++)
    text_start[i] = (i+1.0)/num_rows*top_margin - offset;
  /*text_start[0] = (1/num_rows)*top_margin - offset;
  text_start[1] = 0.5*top_margin - offset;
  text_start[2] = 0.75*top_margin - offset;
  text_start[3] = 1.0*top_margin - offset;*/

  margin_color = Scalar(255, 255, 255); // White
  text_color = Scalar(0, 0, 0); // Black

  HUD::InitMsgs();
}

void HUD::InitMsgs() {
  euler_rpy.x = 0;
  euler_rpy.y = 0;
  euler_rpy.z = 0;
  linear_accel.x = 0;
  linear_accel.y = 0;
  linear_accel.z = 0;
  depth = 0;

  cmd_euler_rpy.x = 0;
  cmd_euler_rpy.y = 0;
  cmd_euler_rpy.z = 0;
  cmd_x = 0;
  cmd_y = 0;
  cmd_z = 0;
  cmd_depth = 0;
}

void HUD::StereoImgCB(const sensor_msgs::ImageConstPtr& msg) {
  if (stereo_img_pub.getNumSubscribers() > 0) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      // Use the BGR8 image_encoding for proper color encoding
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e ){
      ROS_ERROR("cv_bridge exception:  %s", e.what());
      return;
    }
    /*width = msg->width;
    height = msg->height;

    if((ros::Time::now() - object.header.stamp).toSec() < .1) {
      int w = object.bbox_width;
      int h = object.bbox_height;
      cv::Rect rect(object.pos.y - w/2 + width/2, object.pos.z - h/2 + height/2, w, h);
      cv::rectangle(cv_ptr->image, rect, cv::Scalar(0, 255, 0));
    }*/

    Mat img = HUD::CreateHUD(cv_ptr->image);
    sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    stereo_img_pub.publish(out_msg);
  }
}

void HUD::ObjectCB(const riptide_msgs::Object::ConstPtr& msg) {
  object = *msg;
}

void HUD::DownwardImgCB(const sensor_msgs::ImageConstPtr& msg) {
  if (down_img_pub.getNumSubscribers() > 0) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      // Use the BGR8 image_encoding for proper color encoding
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e ){
      ROS_ERROR("cv_bridge exception:  %s", e.what());
      return;
    }

    Mat img = HUD::CreateHUD(cv_ptr->image);
    sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    down_img_pub.publish(out_msg);
  }
}

void HUD::DarknetImgCB(const sensor_msgs::ImageConstPtr& msg){
  if (darknet_img_pub.getNumSubscribers() > 0) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      // Use the BGR8 image_encoding for proper color encoding
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e ){
      ROS_ERROR("cv_bridge exception:  %s", e.what());
      return;
    }

    Mat img = HUD::CreateHUD(cv_ptr->image);
    sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    darknet_img_pub.publish(out_msg);
  }
}

// Add top margin and display key states/cmmands
Mat HUD::CreateHUD(Mat &img) {
  width = img.size().width;
  height = img.size().height;
  Mat hud;
  double font_scale = 1;
  int thickness = 2;

  // Add margin to top and draw divider
  copyMakeBorder(img, hud, top_margin, 0, 0, 0, BORDER_CONSTANT, margin_color);
  line(hud, Point(0, top_margin/2), Point(width, top_margin/2), text_color, 2);

  char state_rpyd[100], cmd_rpyd[100], state_accel[100], cmd_accel[100];

  sprintf(state_rpyd, "STATE: R: %.2f, P: %.2f, Y: %.2f, D: %.2f", euler_rpy.x, euler_rpy.y, euler_rpy.z, depth);
  sprintf(cmd_rpyd, "CMD: R: %.2f, P: %.2f, Y: %.2f, D: %.2f", cmd_euler_rpy.x, cmd_euler_rpy.y, cmd_euler_rpy.z, cmd_depth);
  sprintf(state_accel, "STATE: Ax: %.3f, Ay: %.3f, Az: %.3f", linear_accel.x, linear_accel.y, linear_accel.z);
  sprintf(cmd_accel, "CMD: Ax: %.3f, Ay: %.3f, Az: %.3f", cmd_x, cmd_y, cmd_z);

  putText(hud, string(state_rpyd), Point(5, text_start[0]), FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color, thickness);
  putText(hud, string(cmd_rpyd), Point(5, text_start[1]), FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color, thickness);
  putText(hud, string(state_accel), Point(5, text_start[2]), FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color, thickness);
  putText(hud, string(cmd_accel), Point(5, text_start[3]), FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color, thickness);

  if(reset)
    putText(hud, "RESET", Point(width/3, height/3), FONT_HERSHEY_COMPLEX_SMALL, 3, text_color, thickness);

  return hud;
}

// Get current orientation and linear accel
void HUD::ImuCB(const riptide_msgs::Imu::ConstPtr &imu_msg) {
  euler_rpy.x = imu_msg->rpy_deg.x;
  euler_rpy.y = imu_msg->rpy_deg.y;
  euler_rpy.z = imu_msg->rpy_deg.z;

  linear_accel.x = imu_msg->linear_accel.x;
  linear_accel.y = imu_msg->linear_accel.y;
  linear_accel.z = imu_msg->linear_accel.z;
}

// Get current depth
void HUD::ResetCB(const riptide_msgs::ResetControls::ConstPtr &reset_msg) {
  reset = reset_msg->reset_pwm;
}

// Get current depth
void HUD::DepthCB(const riptide_msgs::Depth::ConstPtr &depth_msg) {
  depth = depth_msg->depth;
}

void HUD::CmdRollCB(const riptide_msgs::AttitudeCommand::ConstPtr& cmd_msg) {
  if (cmd_msg->mode == cmd_msg->POSITION)
    cmd_euler_rpy.x = cmd_msg->value;
}

void HUD::CmdPitchCB(const riptide_msgs::AttitudeCommand::ConstPtr& cmd_msg) {
  if (cmd_msg->mode == cmd_msg->POSITION)
    cmd_euler_rpy.y = cmd_msg->value;
}

void HUD::CmdYawCB(const riptide_msgs::AttitudeCommand::ConstPtr& cmd_msg) {
  if (cmd_msg->mode == cmd_msg->POSITION)
    cmd_euler_rpy.z = cmd_msg->value;
}

// Get command depth
void HUD::CmdDepthCB(const riptide_msgs::DepthCommand::ConstPtr& cmd_msg) {
  cmd_depth = cmd_msg->depth;
}

void HUD::ForceXCB(const std_msgs::Float64::ConstPtr& msg){
  cmd_x = msg->data;
}

void HUD::ForceYCB(const std_msgs::Float64::ConstPtr& msg){
  cmd_y = msg->data;
}

void HUD::ForceZCB(const std_msgs::Float64::ConstPtr& msg){
  cmd_z = msg->data;
}

void HUD::Loop()
{
  ros::Rate rate(50);
  while(ros::ok())
  {
    ros::spinOnce();
    rate.sleep();
  }
}
