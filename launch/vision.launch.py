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

    return launch.LaunchDescription([
        DeclareLaunchArgument('robot', default_value=robot_name, description="Name of the vehicle"),

        #push the the namespace
        PushRosNamespace(robot_name), #only push names

        #launch yolo
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(yoloV5_launch_file_path),
        ),
    ])