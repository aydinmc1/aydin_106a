import math
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rcl_interfaces.msg import SetParametersResult
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetCartesianPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener



class NubPathExecutorNode(Node):
    def __init__(self):
        super().__init__("nub_path_executor_node_v2")

        self.declare_parameter("snapshot_pose_topic", "/aydin_v2/nub_snapshot_v2/poses")
        self.declare_parameter("path_topic", "/aydin_v2/nub_path_v2/path")
        self.declare_parameter("execute_service_name", "/plan_execute_nub_path_v2")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("cartesian_path_service_name", "/compute_cartesian_path")
        self.declare_parameter("execute_trajectory_action_name", "/execute_trajectory")
        self.declare_parameter("gripper_pi_url", "http://192.168.1.202:5000")
        self.declare_parameter("gripper_http_timeout_sec", 1.0)
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("end_effector_frame", "wrist_3_link")
        self.declare_parameter("move_group_name", "ur_manipulator")
        self.declare_parameter("planning_link_name", "wrist_3_link")
        self.declare_parameter("hover_z_offset", .33025)
        self.declare_parameter("grasp_z_offset", 0.279)
        self.declare_parameter("hover_grasp_x_adjustment", 0.0)
        self.declare_parameter("hover_grasp_y_adjustment", 0.0)
        self.declare_parameter("hover_grasp_z_adjustment", 0.0)
        self.declare_parameter("release_x_offset", 0.03)
        self.declare_parameter("curve_samples_per_segment", 8)
        self.declare_parameter("cartesian_max_step", 0.01)
        self.declare_parameter("cartesian_jump_threshold", 0.0)
        self.declare_parameter("minimum_cartesian_fraction", 0.95)
        self.declare_parameter("max_joint_speed", 0.45)
        self.declare_parameter("min_waypoint_dt", 0.20)
        self.declare_parameter("client_timeout_sec", 10.0)
        self.declare_parameter("gripper_toggle_delay_sec", 0.5)
        self.declare_parameter("gripper_cycles_per_nub", 1)
        self.declare_parameter("post_segment_settle_sec", 0.2)
        self.declare_parameter("shutdown_after_execution", True)
        self.declare_parameter("avoid_collisions", True)
        self.declare_parameter("use_current_end_effector_orientation", True)
        self.declare_parameter("tool_orientation_x", 0.0)
        self.declare_parameter("tool_orientation_y", 1.0)
        self.declare_parameter("tool_orientation_z", 0.0)
        self.declare_parameter("tool_orientation_w", 0.0)

        self.snapshot_pose_topic = self.get_parameter("snapshot_pose_topic").value
        self.path_topic = self.get_parameter("path_topic").value
        self.execute_service_name = self.get_parameter("execute_service_name").value
        self.joint_states_topic = self.get_parameter("joint_states_topic").value
        self.cartesian_path_service_name = self.get_parameter(
            "cartesian_path_service_name"
        ).value
        self.execute_trajectory_action_name = self.get_parameter(
            "execute_trajectory_action_name"
        ).value
        self.gripper_pi_url = self.get_parameter("gripper_pi_url").value.rstrip("/")
        self.gripper_http_timeout_sec = float(
            self.get_parameter("gripper_http_timeout_sec").value
        )
        self.target_frame = self.get_parameter("target_frame").value
        self.end_effector_frame = self.get_parameter("end_effector_frame").value
        self.move_group_name = self.get_parameter("move_group_name").value
        self.planning_link_name = self.get_parameter("planning_link_name").value
        self.hover_z_offset = float(self.get_parameter("hover_z_offset").value)
        self.grasp_z_offset = float(self.get_parameter("grasp_z_offset").value)
        self.hover_grasp_x_adjustment = float(
            self.get_parameter("hover_grasp_x_adjustment").value
        )
        self.hover_grasp_y_adjustment = float(
            self.get_parameter("hover_grasp_y_adjustment").value
        )
        self.hover_grasp_z_adjustment = float(
            self.get_parameter("hover_grasp_z_adjustment").value
        )
        self.release_x_offset = float(self.get_parameter("release_x_offset").value)
        self.curve_samples_per_segment = max(
            1,
            int(self.get_parameter("curve_samples_per_segment").value),
        )
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
        self.gripper_toggle_delay_sec = float(
            self.get_parameter("gripper_toggle_delay_sec").value
        )
        self.gripper_cycles_per_nub = int(
            self.get_parameter("gripper_cycles_per_nub").value
        )
        self.post_segment_settle_sec = float(
            self.get_parameter("post_segment_settle_sec").value
        )
        self.shutdown_after_execution = bool(
            self.get_parameter("shutdown_after_execution").value
        )
        self.avoid_collisions = bool(self.get_parameter("avoid_collisions").value)
        self.use_current_end_effector_orientation = bool(
            self.get_parameter("use_current_end_effector_orientation").value
        )
        self.tool_orientation = [
            float(self.get_parameter("tool_orientation_x").value),
            float(self.get_parameter("tool_orientation_y").value),
            float(self.get_parameter("tool_orientation_z").value),
            float(self.get_parameter("tool_orientation_w").value),
        ]
        self.path_orientation = list(self.tool_orientation)
        self.add_on_set_parameters_callback(self.dynamic_parameter_callback)

        self.latest_snapshot = None
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
            PoseArray,
            self.snapshot_pose_topic,
            self.snapshot_callback,
            10,
        )
        self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            10,
        )
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.execute_service = self.create_service(
            Trigger,
            self.execute_service_name,
            self.execute_path_callback,
        )

        self.get_logger().info(f"Listening for snapshots on {self.snapshot_pose_topic}")
        self.get_logger().info(f"Path execution service: {self.execute_service_name}")
        self.get_logger().info(f"Publishing planned path to {self.path_topic}")

    def snapshot_callback(self, msg):
        with self.state_lock:
            self.latest_snapshot = msg

    def joint_state_callback(self, msg):
        with self.state_lock:
            self.latest_joint_state = msg

    def execute_path_callback(self, request, response):
        del request
        with self.state_lock:
            if self.busy:
                response.success = False
                response.message = "Nub path execution is already running"
                return response

            if self.latest_snapshot is None or not self.latest_snapshot.poses:
                response.success = False
                response.message = "No nub snapshot poses are available yet"
                return response

            if self.latest_joint_state is None:
                response.success = False
                response.message = "No joint state is available yet"
                return response

            self.busy = True

        worker = threading.Thread(target=self.execute_path_worker, daemon=True)
        worker.start()

        response.success = True
        response.message = "Started nub path planning and execution"
        return response

    def execute_path_worker(self):
        try:
            self.plan_and_execute_path()
        except Exception as exc:
            self.get_logger().error(f"Nub path execution failed: {exc}")
        finally:
            with self.state_lock:
                self.busy = False

    def plan_and_execute_path(self):
        snapshot = self.copy_latest_snapshot()
        current_position, current_orientation = self.lookup_end_effector_pose()
        if self.use_current_end_effector_orientation:
            self.path_orientation = current_orientation
        else:
            self.path_orientation = list(self.tool_orientation)

        grasp_positions = self.snapshot_to_grasp_targets(snapshot)
        if not grasp_positions:
            self.get_logger().error("Snapshot did not contain any nub targets")
            return

        hover_positions = self.grasp_targets_to_hover_targets(grasp_positions)
        ordered_hover_targets, ordered_indices = self.order_targets(
            current_position,
            hover_positions,
        )
        ordered_grasp_targets = [grasp_positions[index] for index in ordered_indices]
        self.get_logger().info(f"Visiting nub order: {ordered_indices}")

        current_path_position = np.array(current_position, dtype=float)
        self.publish_path_visualization([current_path_position] + ordered_hover_targets)

        for segment_index, target_index in enumerate(ordered_indices):
            remaining_hover_targets = ordered_hover_targets[segment_index:]
            control_points = [current_path_position] + remaining_hover_targets
            segment_points = self.sample_curve_segment(control_points, 0)
            waypoints = [self.position_to_pose(point) for point in segment_points]
            self.get_logger().info(
                f"Moving to hover above nub {target_index} "
                f"({segment_index + 1}/{len(ordered_indices)})"
            )
            if not self.plan_and_execute_waypoints(
                waypoints,
                f"hover above nub {target_index}",
            ):
                return

            time.sleep(self.post_segment_settle_sec)
            hover_target = ordered_hover_targets[segment_index]
            grasp_target = ordered_grasp_targets[segment_index]
            approach_distance = hover_target[2] - grasp_target[2]
            self.get_logger().info(
                f"Lowering {approach_distance:.3f} m to nub {target_index}"
            )
            if not self.plan_and_execute_waypoints(
                [self.position_to_pose(grasp_target)],
                f"grasp height for nub {target_index}",
            ):
                return

            time.sleep(self.post_segment_settle_sec)
            if not self.set_gripper_enabled(True, target_index):
                return

            self.get_logger().info(f"Retreating to hover above nub {target_index}")
            if not self.plan_and_execute_waypoints(
                [self.position_to_pose(hover_target)],
                f"hover retreat for nub {target_index}",
            ):
                return
            time.sleep(self.post_segment_settle_sec)

            release_target = self.hover_target_to_release_target(hover_target)
            release_distance = np.linalg.norm(release_target - hover_target)
            if release_distance > 1e-6:
                self.get_logger().info(
                    f"Moving {self.release_x_offset:.3f} m in x before releasing nub "
                    f"{target_index}"
                )
                if not self.plan_and_execute_waypoints(
                    [self.position_to_pose(release_target)],
                    f"release offset for nub {target_index}",
                ):
                    return

                time.sleep(self.post_segment_settle_sec)

            if not self.set_gripper_enabled(False, target_index):
                return
            current_path_position = release_target
            self.publish_path_visualization(
                [current_path_position] + ordered_hover_targets[segment_index + 1 :]
            )

        self.get_logger().info("Completed nub path execution")
        if self.shutdown_after_execution:
            self.get_logger().info("Path executor node is shutting down")
            rclpy.shutdown()

    def copy_latest_snapshot(self):
        with self.state_lock:
            snapshot = self.latest_snapshot

        snapshot_copy = PoseArray()
        snapshot_copy.header = snapshot.header
        snapshot_copy.poses = list(snapshot.poses)
        return snapshot_copy

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

    def lookup_end_effector_pose(self):
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            self.end_effector_frame,
            Time(),
            timeout=Duration(seconds=self.client_timeout_sec),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            np.array([translation.x, translation.y, translation.z], dtype=float),
            [rotation.x, rotation.y, rotation.z, rotation.w],
        )

    def snapshot_to_grasp_targets(self, snapshot):
        if snapshot.header.frame_id and snapshot.header.frame_id != self.target_frame:
            self.get_logger().warn(
                f"Snapshot frame is {snapshot.header.frame_id}, "
                f"but path executor is using {self.target_frame}"
            )

        targets = []
        for pose in snapshot.poses:
            targets.append(
                np.array(
                    [
                        pose.position.x + self.hover_grasp_x_adjustment,
                        pose.position.y + self.hover_grasp_y_adjustment,
                        pose.position.z
                        + self.grasp_z_offset
                        + self.hover_grasp_z_adjustment,
                    ],
                    dtype=float,
                )
            )
        return targets

    def dynamic_parameter_callback(self, parameters):
        adjustable_parameters = {
            "hover_z_offset": "hover_z_offset",
            "grasp_z_offset": "grasp_z_offset",
            "hover_grasp_x_adjustment": "hover_grasp_x_adjustment",
            "hover_grasp_y_adjustment": "hover_grasp_y_adjustment",
            "hover_grasp_z_adjustment": "hover_grasp_z_adjustment",
        }

        updates = {}
        for parameter in parameters:
            attribute = adjustable_parameters.get(parameter.name)
            if attribute is None:
                continue

            try:
                value = float(parameter.value)
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False,
                    reason=f"{parameter.name} must be a number",
                )

            if not math.isfinite(value):
                return SetParametersResult(
                    successful=False,
                    reason=f"{parameter.name} must be finite",
                )

            updates[attribute] = value

        for attribute, value in updates.items():
            setattr(self, attribute, value)
            self.get_logger().info(f"Updated {attribute} to {value:.4f} m")

        return SetParametersResult(successful=True)

    def grasp_targets_to_hover_targets(self, grasp_positions):
        targets = []
        for position in grasp_positions:
            hover_position = np.array(position, dtype=float)
            hover_position[2] += self.hover_z_offset - self.grasp_z_offset
            targets.append(hover_position)
        return targets

    def hover_target_to_release_target(self, hover_position):
        release_position = np.array(hover_position, dtype=float)
        release_position[0] += self.release_x_offset
        return release_position

    def order_targets(self, start_position, target_positions):
        closest_index = min(
            range(len(target_positions)),
            key=lambda index: np.linalg.norm(target_positions[index] - start_position),
        )
        if len(target_positions) == 1:
            return [target_positions[closest_index]], [closest_index]

        remaining_indices = [
            index for index in range(len(target_positions)) if index != closest_index
        ]
        remaining_positions = [
            target_positions[index] for index in remaining_indices
        ]
        remainder_start = target_positions[closest_index]

        if len(remaining_positions) > 12:
            _, remaining_order = self.nearest_neighbor_order(
                remainder_start,
                remaining_positions,
            )
        else:
            _, remaining_order = self.shortest_open_tsp_order(
                remainder_start,
                remaining_positions,
            )

        ordered_indices = [closest_index] + [
            remaining_indices[index] for index in remaining_order
        ]
        return [target_positions[index] for index in ordered_indices], ordered_indices

    def nearest_neighbor_order(self, start_position, target_positions):
        remaining = set(range(len(target_positions)))
        ordered_indices = []
        cursor = start_position

        while remaining:
            next_index = min(
                remaining,
                key=lambda index: np.linalg.norm(target_positions[index] - cursor),
            )
            ordered_indices.append(next_index)
            remaining.remove(next_index)
            cursor = target_positions[next_index]

        return [target_positions[index] for index in ordered_indices], ordered_indices

    def shortest_open_tsp_order(self, start_position, target_positions):
        target_count = len(target_positions)
        if target_count == 1:
            return [target_positions[0]], [0]

        start_distances = [
            np.linalg.norm(target_positions[index] - start_position)
            for index in range(target_count)
        ]
        pair_distances = np.zeros((target_count, target_count))
        for row in range(target_count):
            for col in range(target_count):
                pair_distances[row, col] = np.linalg.norm(
                    target_positions[row] - target_positions[col]
                )

        costs = {}
        parents = {}
        for index in range(target_count):
            mask = 1 << index
            costs[(mask, index)] = start_distances[index]
            parents[(mask, index)] = None

        for mask in range(1, 1 << target_count):
            for last in range(target_count):
                if not mask & (1 << last) or (mask, last) not in costs:
                    continue

                current_cost = costs[(mask, last)]
                for nxt in range(target_count):
                    if mask & (1 << nxt):
                        continue

                    next_mask = mask | (1 << nxt)
                    next_cost = current_cost + pair_distances[last, nxt]
                    key = (next_mask, nxt)
                    if key not in costs or next_cost < costs[key]:
                        costs[key] = next_cost
                        parents[key] = last

        full_mask = (1 << target_count) - 1
        last_index = min(
            range(target_count),
            key=lambda index: costs[(full_mask, index)],
        )

        ordered_indices = []
        mask = full_mask
        while last_index is not None:
            ordered_indices.append(last_index)
            parent = parents[(mask, last_index)]
            mask &= ~(1 << last_index)
            last_index = parent

        ordered_indices.reverse()
        return [target_positions[index] for index in ordered_indices], ordered_indices

    def sample_curve_segment(self, control_points, segment_index):
        p1 = control_points[segment_index]
        p2 = control_points[segment_index + 1]
        p0 = control_points[max(segment_index - 1, 0)]
        p3 = control_points[min(segment_index + 2, len(control_points) - 1)]

        samples = []
        for sample_index in range(1, self.curve_samples_per_segment + 1):
            t = sample_index / self.curve_samples_per_segment
            samples.append(self.catmull_rom(p0, p1, p2, p3, t))
        return samples

    def catmull_rom(self, p0, p1, p2, p3, t):
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * (t**2)
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * (t**3)
        )

    def position_to_pose(self, point):
        pose = Pose()
        pose.position.x = float(point[0])
        pose.position.y = float(point[1])
        pose.position.z = float(point[2])
        pose.orientation.x = self.path_orientation[0]
        pose.orientation.y = self.path_orientation[1]
        pose.orientation.z = self.path_orientation[2]
        pose.orientation.w = self.path_orientation[3]
        return pose

    def publish_path_visualization(self, control_points):
        path = Path()
        path.header.frame_id = self.target_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for segment_index in range(len(control_points) - 1):
            for point in self.sample_curve_segment(control_points, segment_index):
                pose = PoseStamped()
                pose.header = path.header
                pose.pose = self.position_to_pose(point)
                path.poses.append(pose)

        self.path_pub.publish(path)

    def plan_and_execute_waypoints(self, waypoints, description):
        robot_trajectory = self.compute_cartesian_path(waypoints)
        if robot_trajectory is None:
            self.get_logger().error(f"Could not plan to {description}")
            return False

        self.ensure_trajectory_timing(robot_trajectory)
        if not self.execute_moveit_trajectory(robot_trajectory):
            self.get_logger().error(f"Trajectory execution failed to {description}")
            return False

        return True

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

    def ensure_trajectory_timing(self, robot_trajectory):
        points = robot_trajectory.joint_trajectory.points
        if not points:
            return

        has_timing = any(self.duration_to_seconds(point.time_from_start) > 0.0 for point in points)
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

    def set_gripper_enabled(self, enabled, target_index):
        if self.gripper_cycles_per_nub <= 0:
            return True

        action = "grip" if enabled else "release"
        endpoint = "on" if enabled else "off"
        url = f"{self.gripper_pi_url}/{endpoint}"
        self.get_logger().info(f"Calling Pi gripper to {action} nub {target_index}")

        try:
            request = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(
                request,
                timeout=self.gripper_http_timeout_sec,
            ) as response:
                if response.status >= 400:
                    self.get_logger().error(
                        f"Pi gripper HTTP {response.status} for {url}"
                    )
                    return False
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.get_logger().error(f"Pi gripper request failed for {url}: {exc}")
            return False

        time.sleep(self.gripper_toggle_delay_sec)

        return True



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


def main(args=None):
    rclpy.init(args=args)
    node = NubPathExecutorNode()

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
