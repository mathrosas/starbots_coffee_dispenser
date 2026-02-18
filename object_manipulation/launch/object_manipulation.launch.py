import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory("object_detection"),
        "rviz",
        "config.rviz",
    )

    moveit_config = (
        MoveItConfigsBuilder("name", package_name="my_moveit_config")
        .robot_description(file_path="config/name.urdf.xacro")
        .robot_description_semantic(file_path="config/name.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"]
        )
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .to_moveit_configs()
    )

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("ch", default_value="ch_1"),
        DeclareLaunchArgument("repeat", default_value="1"),
        DeclareLaunchArgument("holder_dx", default_value="0.0"),
        DeclareLaunchArgument("holder_dy", default_value="0.0"),
        DeclareLaunchArgument("holder_dz", default_value="0.0"),
        DeclareLaunchArgument("cup_dx", default_value="0.0"),
        DeclareLaunchArgument("cup_dy", default_value="0.0"),
        DeclareLaunchArgument("cup_dz", default_value="0.0"),
        DeclareLaunchArgument("place_hover_z_offset", default_value="0.20"),
        DeclareLaunchArgument("place_insert_z_delta", default_value="-0.23"),
        DeclareLaunchArgument("place_hover_min_z", default_value="-0.30"),
        DeclareLaunchArgument("adaptive_insert_to_place", default_value="true"),
    ]

    runtime_params = {
        "use_sim_time": ParameterValue(
            LaunchConfiguration("use_sim_time"), value_type=bool
        ),
        "ch": LaunchConfiguration("ch"),
        "repeat": ParameterValue(LaunchConfiguration("repeat"), value_type=int),
        "holder_dx": ParameterValue(
            LaunchConfiguration("holder_dx"), value_type=float
        ),
        "holder_dy": ParameterValue(
            LaunchConfiguration("holder_dy"), value_type=float
        ),
        "holder_dz": ParameterValue(
            LaunchConfiguration("holder_dz"), value_type=float
        ),
        "cup_dx": ParameterValue(LaunchConfiguration("cup_dx"), value_type=float),
        "cup_dy": ParameterValue(LaunchConfiguration("cup_dy"), value_type=float),
        "cup_dz": ParameterValue(LaunchConfiguration("cup_dz"), value_type=float),
        "place_hover_z_offset": ParameterValue(
            LaunchConfiguration("place_hover_z_offset"), value_type=float
        ),
        "place_insert_z_delta": ParameterValue(
            LaunchConfiguration("place_insert_z_delta"), value_type=float
        ),
        "place_hover_min_z": ParameterValue(
            LaunchConfiguration("place_hover_min_z"), value_type=float
        ),
        "adaptive_insert_to_place": ParameterValue(
            LaunchConfiguration("adaptive_insert_to_place"), value_type=bool
        ),
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        arguments=["--ros-args", "--log-level", "warn"],
        parameters=[
            moveit_config.to_dict(),
            {"trajectory_execution.allowed_execution_duration_scaling": 5.0},
            {"publish_robot_description_semantic": True},
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                )
            },
        ],
    )

    add_scene_node = Node(
        package="object_manipulation",
        executable="add_coffee_scene",
        name="add_coffee_scene",
        output="screen",
        parameters=[
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                )
            },
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                )
            },
        ],
    )

    manipulation_node = Node(
        name="object_manipulation",
        package="object_manipulation",
        executable="object_manipulation",
        output="screen",
        parameters=[moveit_config.to_dict(), runtime_params],
        arguments=["--ros-args", "--log-level", "info"],
    )

    delayed_manipulation_node = TimerAction(
        period=3.0,
        actions=[manipulation_node],
    )

    ld = LaunchDescription(args)
    ld.add_action(move_group_node)
    ld.add_action(add_scene_node)
    ld.add_action(rviz_node)
    ld.add_action(delayed_manipulation_node)
    return ld
