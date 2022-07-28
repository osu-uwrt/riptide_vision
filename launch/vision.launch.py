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

    return launch.LaunchDescription([
        riptide_vision,
    ])