import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share_dir = get_package_share_directory("aydin_v2")
    params_file = os.path.join(package_share_dir, "config", "params.yaml")

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/camera/color/image_raw",
        description="Input camera image topic.",
    )
    debug_image_topic_arg = DeclareLaunchArgument(
        "debug_image_topic",
        default_value="/aydin_v2/white_patches/debug_image",
        description="Annotated debug image output topic.",
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/camera/color/camera_info",
        description="Input camera calibration topic.",
    )
    camera_frame_arg = DeclareLaunchArgument(
        "camera_frame",
        default_value="camera_color_optical_frame",
        description="Camera frame used for ArUco pose and TF outputs.",
    )
    enable_tuning_window_arg = DeclareLaunchArgument(
        "enable_tuning_window",
        default_value="true",
        description="Open the interactive white-detection tuning window.",
    )
    tuning_config_path_arg = DeclareLaunchArgument(
        "tuning_config_path",
        default_value="white_detection_params.json",
        description="JSON file used to load and save white-detection tuning.",
    )
    launch_camera_arg = DeclareLaunchArgument(
        "launch_camera",
        default_value="true",
        description="Start the RealSense camera driver.",
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        condition=IfCondition(LaunchConfiguration("launch_camera")),
    )

    white_patch_detector = Node(
        package="aydin_v2",
        executable="white_patch_detector_node",
        name="white_patch_detector_node",
        output="screen",
        parameters=[
            params_file,
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "debug_image_topic": LaunchConfiguration("debug_image_topic"),
                "camera_frame": LaunchConfiguration("camera_frame"),
                "enable_tuning_window": ParameterValue(
                    LaunchConfiguration("enable_tuning_window"),
                    value_type=bool,
                ),
                "tuning_config_path": LaunchConfiguration("tuning_config_path"),
            },
        ],
    )

    return LaunchDescription(
        [
            image_topic_arg,
            debug_image_topic_arg,
            camera_info_topic_arg,
            camera_frame_arg,
            enable_tuning_window_arg,
            tuning_config_path_arg,
            launch_camera_arg,
            realsense_launch,
            white_patch_detector,
        ]
    )
