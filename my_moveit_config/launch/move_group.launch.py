from moveit_configs_utils import MoveItConfigsBuilder

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("ur3e", package_name="my_moveit_config")
        .robot_description(file_path="config/ur3e.urdf.xacro")
        .robot_description_semantic(file_path="config/ur3e.srdf")
        .to_moveit_configs()
    )
    
    # Move Group Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"trajectory_execution.allowed_execution_duration_scaling": 5.0,},
            {"publish_robot_description_semantic": True},
            {"use_sim_time": False},
        ],
    )

    # Include the scene-adding node
    add_scene_node = Node(
        package='object_manipulation', 
        executable='add_coffee_scene',
        name='add_coffee_scene',
        output='screen',
    )

    return LaunchDescription(
        [move_group_node, add_scene_node]
    )
