from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="aydin_v2",
                executable="example_node",
                name="example_node",
                output="screen",
            ),
        ]
    )
