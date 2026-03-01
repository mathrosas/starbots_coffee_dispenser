from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    holder_offset_x = LaunchConfiguration('holder_offset_x')
    holder_offset_y = LaunchConfiguration('holder_offset_y')
    holder_offset_z = LaunchConfiguration('holder_offset_z')

    declare_holder_offset_x = DeclareLaunchArgument(
        'holder_offset_x',
        default_value='0.0',
        description='Cup-holder detection X offset in base_link (meters)'
    )
    declare_holder_offset_y = DeclareLaunchArgument(
        'holder_offset_y',
        default_value='0.0',
        description='Cup-holder detection Y offset in base_link (meters)'
    )
    declare_holder_offset_z = DeclareLaunchArgument(
        'holder_offset_z',
        default_value='0.0',
        description='Cup-holder detection Z offset in base_link (meters)'
    )

    object_detection = Node(
        package='object_detection',
        executable='object_detection',
        name='object_detection',
        output='screen',
        parameters=[{
            'holder_offset_x': holder_offset_x,
            'holder_offset_y': holder_offset_y,
            'holder_offset_z': holder_offset_z,
        }],
        arguments=["--ros-args", "--log-level", "info"]
    )

    return LaunchDescription([
        declare_holder_offset_x,
        declare_holder_offset_y,
        declare_holder_offset_z,
        object_detection
    ])
