import launch
import os
import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    riptide_vision_share_dir = get_package_share_directory('riptide_vision2')

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
            arguments=["-0.242", "0.283", "-0.066", "0", "0", "0", "tempest/base_link", "tempest/stereo/left_optical"]
    )

    return launch.LaunchDescription([
        riptide_vision,
        static_transform
    ])