cmake_minimum_required(VERSION 2.8.3)
project(riptide_vision)
#set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
add_compile_options(-std=c++11)

# NOTE: Most dependencies can be taken care of with $(catkin_EXPORTED_TARGETS).
# There is no need to add mention the specific dependency, such as
# <package_name>_gencpp, or <package_name>_gencfg

# NOTE: To include OpenCV, must do find_package(OpenCV REQUIRED), followed by
# include_directories(...)

find_package(catkin REQUIRED
    COMPONENTS
    rospy
    roscpp
    roslint
    riptide_msgs
    darknet_ros_msgs
    sensor_msgs
    cv_bridge
    image_transport
    compressed_image_transport
    tf2_geometry_msgs
)

catkin_package(INCLUDE_DIRS include)
find_package(OpenCV REQUIRED) # To include, OpenCV, this MUST go BEFORE include_directories()
include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
roslint_cpp()

cmake_host_system_information(RESULT _host_name QUERY HOSTNAME)


add_executable(hud src/hud.cpp)
target_link_libraries(hud ${catkin_LIBRARIES})
add_dependencies(hud ${catkin_EXPORTED_TARGETS} ${OpenCV_LIBRARIES})