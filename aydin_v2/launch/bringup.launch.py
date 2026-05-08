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
        default_value="/aydin_v2/green_nubs/debug_image_v2",
        description="Annotated green nub debug image output topic.",
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/camera/color/camera_info",
        description="Input camera calibration topic.",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/camera/aligned_depth_to_color/image_raw",
        description="Aligned depth image topic.",
    )
    camera_frame_arg = DeclareLaunchArgument(
        "camera_frame",
        default_value="camera_color_optical_frame",
        description="Camera frame used for ArUco pose and TF outputs.",
    )
    target_frame_arg = DeclareLaunchArgument(
        "target_frame",
        default_value="base_link",
        description="Frame used for live and snapshot nub outputs.",
    )
    camera_tf_parent_frame_arg = DeclareLaunchArgument(
        "camera_tf_parent_frame",
        default_value="wrist_3_link",
        description="Robot frame that the camera is mounted to.",
    )
    camera_tf_child_frame_arg = DeclareLaunchArgument(
        "camera_tf_child_frame",
        default_value="camera_color_optical_frame",
        description="Camera optical frame connected to the camera mount frame.",
    )
    camera_tf_x_arg = DeclareLaunchArgument(
        "camera_tf_x",
        default_value="-0.025",
        description="Static camera transform x offset in meters.",
    )
    camera_tf_y_arg = DeclareLaunchArgument(
        "camera_tf_y",
        default_value="0.13",
        description="Static camera transform y offset in meters.",
    )
    camera_tf_z_arg = DeclareLaunchArgument(
        "camera_tf_z",
        default_value="0.0",
        description="Static camera transform z offset in meters.",
    )
    camera_tf_roll_arg = DeclareLaunchArgument(
        "camera_tf_roll",
        default_value="0.0",
        description="Static camera transform roll in radians.",
    )
    camera_tf_pitch_arg = DeclareLaunchArgument(
        "camera_tf_pitch",
        default_value="0.0",
        description="Static camera transform pitch in radians.",
    )
    camera_tf_yaw_arg = DeclareLaunchArgument(
        "camera_tf_yaw",
        default_value="0.0",
        description="Static camera transform yaw in radians.",
    )
    snapshot_duration_arg = DeclareLaunchArgument(
        "snapshot_duration",
        default_value="3.0",
        description="Seconds of green nub observations to cluster.",
    )
    enable_tuning_window_arg = DeclareLaunchArgument(
        "enable_tuning_window",
        default_value="true",
        description="Open the interactive green nub tuning window.",
    )
    tuning_config_path_arg = DeclareLaunchArgument(
        "tuning_config_path",
        default_value="green_detection_params_v2.json",
        description="JSON file used to load and save green nub tuning.",
    )
    launch_camera_arg = DeclareLaunchArgument(
        "launch_camera",
        default_value="true",
        description="Start the RealSense camera driver.",
    )
    launch_path_executor_arg = DeclareLaunchArgument(
        "launch_path_executor",
        default_value="true",
        description="Start the MoveIt-backed green nub path executor.",
    )
    hover_z_offset_arg = DeclareLaunchArgument(
        "hover_z_offset",
        default_value="0.185",
        description="Meters above each snapshot nub used for path execution.",
    )
    shutdown_path_executor_after_execution_arg = DeclareLaunchArgument(
        "shutdown_path_executor_after_execution",
        default_value="false",
        description="Shutdown the path executor after it completes one path.",
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
        launch_arguments={
            "align_depth.enable": "true",
            "publish_tf": "false",
        }.items(),
    )

    camera_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_optical_static_tf_v2",
        output="screen",
        arguments=[
            "--x",
            LaunchConfiguration("camera_tf_x"),
            "--y",
            LaunchConfiguration("camera_tf_y"),
            "--z",
            LaunchConfiguration("camera_tf_z"),
            "--roll",
            LaunchConfiguration("camera_tf_roll"),
            "--pitch",
            LaunchConfiguration("camera_tf_pitch"),
            "--yaw",
            LaunchConfiguration("camera_tf_yaw"),
            "--frame-id",
            LaunchConfiguration("camera_tf_parent_frame"),
            "--child-frame-id",
            LaunchConfiguration("camera_tf_child_frame"),
        ],
    )

    green_nub_detector = Node(
        package="aydin_v2",
        executable="green_nub_detector_node_v2",
        name="green_nub_detector_node_v2",
        output="screen",
        parameters=[
            params_file,
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "debug_image_topic": LaunchConfiguration("debug_image_topic"),
                "camera_frame": LaunchConfiguration("camera_frame"),
                "target_frame": LaunchConfiguration("target_frame"),
                "enable_tuning_window": ParameterValue(
                    LaunchConfiguration("enable_tuning_window"),
                    value_type=bool,
                ),
                "tuning_config_path": LaunchConfiguration("tuning_config_path"),
            },
        ],
    )

    nub_snapshot = Node(
        package="aydin_v2",
        executable="nub_snapshot_node_v2",
        name="nub_snapshot_node_v2",
        output="screen",
        parameters=[
            params_file,
            {
                "debug_image_topic": LaunchConfiguration("debug_image_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "snapshot_duration": ParameterValue(
                    LaunchConfiguration("snapshot_duration"),
                    value_type=float,
                ),
            },
        ],
    )

    nub_path_executor = Node(
        package="aydin_v2",
        executable="nub_path_executor_node_v2",
        name="nub_path_executor_node_v2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_path_executor")),
        parameters=[
            params_file,
            {
                "target_frame": LaunchConfiguration("target_frame"),
                "hover_z_offset": ParameterValue(
                    LaunchConfiguration("hover_z_offset"),
                    value_type=float,
                ),
                "shutdown_after_execution": ParameterValue(
                    LaunchConfiguration("shutdown_path_executor_after_execution"),
                    value_type=bool,
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            image_topic_arg,
            debug_image_topic_arg,
            camera_info_topic_arg,
            depth_topic_arg,
            camera_frame_arg,
            target_frame_arg,
            camera_tf_parent_frame_arg,
            camera_tf_child_frame_arg,
            camera_tf_x_arg,
            camera_tf_y_arg,
            camera_tf_z_arg,
            camera_tf_roll_arg,
            camera_tf_pitch_arg,
            camera_tf_yaw_arg,
            snapshot_duration_arg,
            enable_tuning_window_arg,
            tuning_config_path_arg,
            launch_camera_arg,
            launch_path_executor_arg,
            hover_z_offset_arg,
            shutdown_path_executor_after_execution_arg,
            realsense_launch,
            camera_static_tf,
            green_nub_detector,
            nub_snapshot,
            nub_path_executor,
        ]
    )
