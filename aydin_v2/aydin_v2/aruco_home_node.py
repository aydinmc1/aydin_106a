import math
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener


class ArucoHomeNode(Node):
    def __init__(self):
        super().__init__("aruco_home_node_v2")

        self.declare_parameter("home_service_name", "/home_to_aruco_v2")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("cartesian_path_service_name", "/compute_cartesian_path")
        self.declare_parameter("execute_trajectory_action_name", "/execute_trajectory")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("aruco_marker_id", 1)
        self.declare_parameter("aruco_marker_frame", "")
        self.declare_parameter("home_offset_x", 0.0)
        self.declare_parameter("home_offset_y", 0.0)
        self.declare_parameter("home_offset_z", 0.0)
        self.declare_parameter("home_offset_frame", "marker")
        self.declare_parameter("move_group_name", "ur_manipulator")
        self.declare_parameter("planning_link_name", "wrist_3_link")
        self.declare_parameter("end_effector_frame", "wrist_3_link")
        self.declare_parameter("cartesian_max_step", 0.01)
        self.declare_parameter("cartesian_jump_threshold", 0.0)
        self.declare_parameter("minimum_cartesian_fraction", 0.95)
        self.declare_parameter("max_joint_speed", 0.35)
        self.declare_parameter("min_waypoint_dt", 0.25)
        self.declare_parameter("client_timeout_sec", 10.0)
        self.declare_parameter("avoid_collisions", True)
        self.declare_parameter("use_current_end_effector_orientation", True)
        self.declare_parameter("home_orientation_x", 0.0)
        self.declare_parameter("home_orientation_y", 1.0)
        self.declare_parameter("home_orientation_z", 0.0)
        self.declare_parameter("home_orientation_w", 0.0)

        self.home_service_name = self.get_parameter("home_service_name").value
        self.joint_states_topic = self.get_parameter("joint_states_topic").value
        self.cartesian_path_service_name = self.get_parameter(
            "cartesian_path_service_name"
        ).value
        self.execute_trajectory_action_name = self.get_parameter(
            "execute_trajectory_action_name"
        ).value
        self.target_frame = self.get_parameter("target_frame").value
        self.aruco_marker_id = int(self.get_parameter("aruco_marker_id").value)
        self.aruco_marker_frame = self.get_parameter("aruco_marker_frame").value
        if not self.aruco_marker_frame:
            self.aruco_marker_frame = f"ar_marker_v2_{self.aruco_marker_id}"
        self.home_offset = np.array(
            [
                float(self.get_parameter("home_offset_x").value),
                float(self.get_parameter("home_offset_y").value),
                float(self.get_parameter("home_offset_z").value),
            ],
            dtype=float,
        )
        self.home_offset_frame = self.get_parameter("home_offset_frame").value
        self.move_group_name = self.get_parameter("move_group_name").value
        self.planning_link_name = self.get_parameter("planning_link_name").value
        self.end_effector_frame = self.get_parameter("end_effector_frame").value
        self.cartesian_max_step = float(
            self.get_parameter("cartesian_max_step").value
        )
        self.cartesian_jump_threshold = float(
            self.get_parameter("cartesian_jump_threshold").value
        )
        self.minimum_cartesian_fraction = float(
            self.get_parameter("minimum_cartesian_fraction").value
        )
        self.max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self.min_waypoint_dt = float(self.get_parameter("min_waypoint_dt").value)
        self.client_timeout_sec = float(self.get_parameter("client_timeout_sec").value)
        self.avoid_collisions = bool(self.get_parameter("avoid_collisions").value)
        self.use_current_end_effector_orientation = bool(
            self.get_parameter("use_current_end_effector_orientation").value
        )
        self.home_orientation = [
            float(self.get_parameter("home_orientation_x").value),
            float(self.get_parameter("home_orientation_y").value),
            float(self.get_parameter("home_orientation_z").value),
            float(self.get_parameter("home_orientation_w").value),
        ]

        self.latest_joint_state = None
        self.busy = False
        self.state_lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            self.cartesian_path_service_name,
        )
        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            self.execute_trajectory_action_name,
        )

        self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            10,
        )
        self.home_service = self.create_service(
            Trigger,
            self.home_service_name,
            self.home_callback,
        )

        self.get_logger().info(
            f"Aruco home service: {self.home_service_name}; "
            f"targeting {self.planning_link_name} from {self.aruco_marker_frame}"
        )

    def joint_state_callback(self, msg):
        with self.state_lock:
            self.latest_joint_state = msg

    def home_callback(self, request, response):
        del request
        with self.state_lock:
            if self.busy:
                response.success = False
                response.message = "ArUco home move is already running"
                return response

            if self.latest_joint_state is None:
                response.success = False
                response.message = "No joint state is available yet"
                return response

            self.busy = True

        worker = threading.Thread(target=self.home_worker, daemon=True)
        worker.start()

        response.success = True
        response.message = "Started ArUco home move"
        return response

    def home_worker(self):
        try:
            if self.plan_and_execute_home():
                self.get_logger().info("ArUco home move complete")
        except Exception as exc:
            self.get_logger().error(f"ArUco home move failed: {exc}")
        finally:
            with self.state_lock:
                self.busy = False

    def plan_and_execute_home(self):
        target_pose = self.make_home_pose()
        robot_trajectory = self.compute_cartesian_path([target_pose])
        if robot_trajectory is None:
            return False

        self.ensure_trajectory_timing(robot_trajectory)
        return self.execute_moveit_trajectory(robot_trajectory)

    def make_home_pose(self):
        marker_transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            self.aruco_marker_frame,
            Time(),
            timeout=Duration(seconds=self.client_timeout_sec),
        )
        marker_translation = marker_transform.transform.translation
        marker_rotation = marker_transform.transform.rotation

        marker_position = np.array(
            [marker_translation.x, marker_translation.y, marker_translation.z],
            dtype=float,
        )
        if self.home_offset_frame == "marker":
            offset = self.rotate_vector_by_quaternion(self.home_offset, marker_rotation)
        elif self.home_offset_frame == "target":
            offset = self.home_offset
        else:
            self.get_logger().warn(
                f"Unknown home_offset_frame '{self.home_offset_frame}', using marker"
            )
            offset = self.rotate_vector_by_quaternion(self.home_offset, marker_rotation)

        position = marker_position + offset
        if self.use_current_end_effector_orientation:
            orientation = self.lookup_end_effector_orientation()
        else:
            orientation = self.home_orientation

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])
        self.get_logger().info(
            f"Planning home pose in {self.target_frame}: "
            f"({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
        )
        return pose

    def lookup_end_effector_orientation(self):
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            self.end_effector_frame,
            Time(),
            timeout=Duration(seconds=self.client_timeout_sec),
        )
        rotation = transform.transform.rotation
        return [rotation.x, rotation.y, rotation.z, rotation.w]

    def copy_latest_joint_state(self):
        with self.state_lock:
            joint_state = self.latest_joint_state

        joint_copy = JointState()
        joint_copy.header = joint_state.header
        joint_copy.name = list(joint_state.name)
        joint_copy.position = list(joint_state.position)
        joint_copy.velocity = list(joint_state.velocity)
        joint_copy.effort = list(joint_state.effort)
        return joint_copy

    def compute_cartesian_path(self, waypoints):
        if not self.cartesian_path_client.wait_for_service(
            timeout_sec=self.client_timeout_sec
        ):
            self.get_logger().error(
                f"MoveIt Cartesian path service unavailable: "
                f"{self.cartesian_path_service_name}"
            )
            return None

        request = GetCartesianPath.Request()
        request.header.frame_id = self.target_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.start_state = RobotState()
        request.start_state.joint_state = self.copy_latest_joint_state()
        request.start_state.is_diff = True
        request.group_name = self.move_group_name
        request.link_name = self.planning_link_name
        request.waypoints = waypoints
        request.max_step = self.cartesian_max_step
        if hasattr(request, "jump_threshold"):
            request.jump_threshold = self.cartesian_jump_threshold
        request.avoid_collisions = self.avoid_collisions

        future = self.cartesian_path_client.call_async(request)
        response = self.wait_for_future(future, self.client_timeout_sec)
        if response is None:
            self.get_logger().error("Timed out waiting for Cartesian path response")
            return None

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"MoveIt Cartesian path failed with code {response.error_code.val}"
            )
            return None

        if response.fraction < self.minimum_cartesian_fraction:
            self.get_logger().error(
                f"MoveIt planned {response.fraction:.2f} of the requested "
                f"Cartesian path; required {self.minimum_cartesian_fraction:.2f}"
            )
            return None

        if not response.solution.joint_trajectory.points:
            self.get_logger().error("MoveIt returned an empty trajectory")
            return None

        return response.solution

    def execute_moveit_trajectory(self, robot_trajectory):
        if not self.execute_trajectory_client.wait_for_server(
            timeout_sec=self.client_timeout_sec
        ):
            self.get_logger().error(
                f"MoveIt execute trajectory action unavailable: "
                f"{self.execute_trajectory_action_name}"
            )
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_trajectory

        send_future = self.execute_trajectory_client.send_goal_async(goal)
        goal_handle = self.wait_for_future(send_future, self.client_timeout_sec)
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveIt execute trajectory goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        result = self.wait_for_future(result_future, self.client_timeout_sec + 60.0)
        if result is None:
            self.get_logger().error("Timed out waiting for MoveIt execution result")
            return False

        error_code = result.result.error_code
        if error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"MoveIt execution failed with code {error_code.val}"
            )
            return False

        return True

    def ensure_trajectory_timing(self, robot_trajectory):
        points = robot_trajectory.joint_trajectory.points
        if not points:
            return

        has_timing = any(
            self.duration_to_seconds(point.time_from_start) > 0.0
            for point in points
        )
        if has_timing:
            return

        elapsed = 0.0
        previous_positions = None
        for point in points:
            if previous_positions is None:
                elapsed = max(self.min_waypoint_dt, 0.1)
            else:
                max_delta = max(
                    abs(current - previous)
                    for current, previous in zip(point.positions, previous_positions)
                )
                elapsed += max(
                    self.min_waypoint_dt,
                    max_delta / max(self.max_joint_speed, 1e-6),
                )

            point.time_from_start = self.seconds_to_duration(elapsed)
            previous_positions = list(point.positions)

        self.fill_joint_velocities(robot_trajectory)

    def fill_joint_velocities(self, robot_trajectory):
        points = robot_trajectory.joint_trajectory.points
        joint_count = len(robot_trajectory.joint_trajectory.joint_names)
        for index, point in enumerate(points):
            if len(points) < 2:
                point.velocities = [0.0] * joint_count
                continue

            if index == 0:
                previous_point = points[index]
                next_point = points[index + 1]
            elif index == len(points) - 1:
                previous_point = points[index - 1]
                next_point = points[index]
            else:
                previous_point = points[index - 1]
                next_point = points[index + 1]

            dt = self.duration_to_seconds(next_point.time_from_start) - (
                self.duration_to_seconds(previous_point.time_from_start)
            )
            if dt <= 0:
                point.velocities = [0.0] * joint_count
                continue

            point.velocities = (
                (
                    np.array(next_point.positions)
                    - np.array(previous_point.positions)
                )
                / dt
            ).tolist()

    def wait_for_future(self, future, timeout_sec):
        event = threading.Event()
        future.add_done_callback(lambda done_future: event.set())
        if not event.wait(timeout_sec):
            return None

        try:
            return future.result()
        except Exception as exc:
            self.get_logger().error(f"Async ROS request failed: {exc}")
            return None

    def seconds_to_duration(self, seconds):
        seconds = max(0.0, float(seconds))
        whole_seconds = int(math.floor(seconds))
        nanoseconds = int((seconds - whole_seconds) * 1e9)
        return Duration(
            seconds=whole_seconds,
            nanoseconds=nanoseconds,
        ).to_msg()

    def duration_to_seconds(self, duration_msg):
        return duration_msg.sec + duration_msg.nanosec * 1e-9

    def rotate_vector_by_quaternion(self, vector, quaternion):
        q = np.array(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            dtype=float,
        )
        norm = np.linalg.norm(q)
        if norm <= 1e-12:
            return np.array(vector, dtype=float)
        q = q / norm
        q_vec = q[:3]
        q_w = q[3]
        vector = np.array(vector, dtype=float)
        return (
            vector
            + 2.0 * q_w * np.cross(q_vec, vector)
            + 2.0 * np.cross(q_vec, np.cross(q_vec, vector))
        )


def main(args=None):
    rclpy.init(args=args)
    node = ArucoHomeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
