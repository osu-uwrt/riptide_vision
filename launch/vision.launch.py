import launch
import os
import launch_ros.actions
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument 
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import PushRosNamespace
from launch.launch_description_sources import AnyLaunchDescriptionSource

robot_name = "tempest"

def generate_launch_description():
    riptide_vision_share_dir = get_package_share_directory('riptide_vision2')

    yoloV5_launch_file_path = os.path.join(get_package_share_directory('yolov5_ros'), "launch", "yolov5s_simple.launch.py")


    riptide_vision = launch_ros.actions.Node(
        package="riptide_vision2", executable="vision",
        parameters=[
                       {"weights":os.path.join(riptide_vision_share_dir,"weights/last.pt")},
                       {"data":os.path.join(riptide_vision_share_dir,"config/pool.yaml")}
                   ],
    )
    static_transform = launch_ros.actions.Node(
            name="odom_to_world_broadcaster",
            package="tf2_ros",
            executable="static_transform_publisher",
            # arguments=["-0.242", "0.283", "-0.066", "1.57079", "3.14159", "1.5707", "tempest/origin", "tempest/stereo/left_optical"]
            arguments=["-0.242", "0.283", "-0.066", "-1.5707", "0", "0", "tempest/origin", "tempest/stereo/left_optical"]
    )

    

    return launch.LaunchDescription([
        # riptide_vision,
        static_transform,

        DeclareLaunchArgument('robot', default_value=robot_name, description="Name of the vehicle"),

        #push the the namespace
        PushRosNamespace(robot_name), #only push names

        #launch yolo
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(yoloV5_launch_file_path),
        ),
    ])