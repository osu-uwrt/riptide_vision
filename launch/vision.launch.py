import launch
import launch.actions
from ament_index_python.packages import get_package_share_directory
import launch_ros.actions
import os
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    # declare the launch args to read for this file
    config = os.path.join(
        get_package_share_directory('riptide_vision2'),
        'config',
        'config_pose.yaml'
        )

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            "log_level", 
            default_value="INFO",
            description="log level to use",
        ),

        # create the nodes    
        launch_ros.actions.Node(
            package='riptide_vision2',
            executable='vision',
            name='riptide_vision2',
            respawn=True,
            output='screen',
            
            # use the parameters on the node
            parameters = [
                config
            ]
        )
    ])