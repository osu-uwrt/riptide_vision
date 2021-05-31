cmake_minimum_required(VERSION 3.0.2)
project(riptide_vision)

find_package(catkin REQUIRED
    COMPONENTS
    rospy
    roscpp
    roslint
    sensor_msgs
    cv_bridge
    image_transport
    compressed_image_transport
    tf2_geometry_msgs
)

catkin_package()