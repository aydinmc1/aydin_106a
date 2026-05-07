import json
import math
import os

import cv2
import message_filters
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Pose, PoseArray, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from aydin_v2.nub_detection import (
    GreenNubDetector,
    NUB_DETECTION_FIELD_COUNT,
    make_nub_detection_layout,
    pack_nub_detection,
)


def quaternion_from_matrix(matrix):
    q = np.empty((4,), dtype=np.float64)
    m = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    trace = np.trace(m)
    if trace > m[3, 3]:
        q[3] = trace
        q[2] = m[1, 0] - m[0, 1]
        q[1] = m[0, 2] - m[2, 0]
        q[0] = m[2, 1] - m[1, 2]
    else:
        i, j, k = 0, 1, 2
        if m[1, 1] > m[0, 0]:
            i, j, k = 1, 2, 0
        if m[2, 2] > m[i, i]:
            i, j, k = 2, 0, 1
        trace = m[i, i] - (m[j, j] + m[k, k]) + m[3, 3]
        q[i] = trace
        q[j] = m[i, j] + m[j, i]
        q[k] = m[k, i] + m[i, k]
        q[3] = m[k, j] - m[j, k]
    q *= 0.5 / math.sqrt(trace * m[3, 3])
    return q


class WhitePatchDetectorNode(Node):
    def __init__(self):
        super().__init__("white_patch_detector_node_v2")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter(
            "debug_image_topic", "/aydin_v2/white_patches/debug_image_v2"
        )
        self.declare_parameter("aruco_dictionary_id", "DICT_5X5_250")
        self.declare_parameter("aruco_marker_id", 1)
        self.declare_parameter("aruco_marker_size", 0.15)
        self.declare_parameter("aruco_axis_length", 0.075)
        self.declare_parameter("aruco_pose_topic", "/aydin_v2/aruco_poses_v2")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter(
            "live_nub_pose_topic", "/aydin_v2/live_nubs_v2/poses"
        )
        self.declare_parameter(
            "live_nub_detection_topic", "/aydin_v2/live_nubs_v2/detections"
        )
        self.declare_parameter("snapshot_service_name", "/take_nub_snapshot_v2")
        self.declare_parameter("white_min_value", 200)
        self.declare_parameter("white_max_saturation", 45)
        self.declare_parameter("min_patch_area", 150.0)
        self.declare_parameter("max_patch_area", 0.0)
        self.declare_parameter("blur_kernel_size", 5)
        self.declare_parameter("morph_kernel_size", 5)
        self.declare_parameter("green_h_min", 35)
        self.declare_parameter("green_h_max", 85)
        self.declare_parameter("green_s_min", 80)
        self.declare_parameter("green_s_max", 255)
        self.declare_parameter("green_v_min", 50)
        self.declare_parameter("green_v_max", 255)
        self.declare_parameter("green_min_area", 20.0)
        self.declare_parameter("green_max_area", 800.0)
        self.declare_parameter("green_min_circularity", 0.35)
        self.declare_parameter("green_blur_kernel_size", 3)
        self.declare_parameter("green_morph_kernel_size", 3)
        self.declare_parameter("enable_tuning_window", True)
        self.declare_parameter(
            "tuning_config_path", "white_detection_params_v2.json"
        )

        self.bridge = CvBridge()
        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.aruco_marker_id = int(self.get_parameter("aruco_marker_id").value)
        self.aruco_marker_size = float(self.get_parameter("aruco_marker_size").value)
        self.aruco_axis_length = float(self.get_parameter("aruco_axis_length").value)
        self.aruco_pose_topic = self.get_parameter("aruco_pose_topic").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.live_nub_pose_topic = self.get_parameter("live_nub_pose_topic").value
        self.live_nub_detection_topic = self.get_parameter(
            "live_nub_detection_topic"
        ).value
        self.snapshot_service_name = self.get_parameter("snapshot_service_name").value
        self.info_msg = None
        self.camera_matrix = None
        self.distortion = None
        self.enable_tuning_window = bool(
            self.get_parameter("enable_tuning_window").value
        )
        self.tuning_config_path = os.path.expanduser(
            self.get_parameter("tuning_config_path").value
        )
        self.tuning_window_name = "aydin_v2 white detection tuning"
        self.tuning_window_ready = False
        self.save_requested = False
        self.save_trackbar_armed = False
        self.snapshot_trackbar_armed = False
        self.white_settings = self.load_white_settings()
        self.green_settings = self.load_green_settings()
        self.green_nub_detector = GreenNubDetector(self.green_settings)
        self.snapshot_request_pending = False
        self.last_snapshot_status = "Snapshot service not called"

        dictionary_name = self.get_parameter("aruco_dictionary_id").value
        self.aruco_dictionary = self.get_aruco_dictionary(dictionary_name)
        self.aruco_parameters = self.create_aruco_parameters()

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.aruco_poses_pub = self.create_publisher(
            PoseArray, self.aruco_pose_topic, 10
        )
        self.live_nub_poses_pub = self.create_publisher(
            PoseArray, self.live_nub_pose_topic, 10
        )
        self.live_nub_detections_pub = self.create_publisher(
            Float32MultiArray, self.live_nub_detection_topic, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.rgb_sub = message_filters.Subscriber(
            self,
            Image,
            self.image_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.synced_callback)
        self.snapshot_client = self.create_client(Trigger, self.snapshot_service_name)

        self.get_logger().info(f"Subscribing to {self.image_topic}")
        self.get_logger().info(f"Subscribing to {self.depth_topic}")
        self.get_logger().info(f"Subscribing to {self.camera_info_topic}")
        self.get_logger().info(f"Publishing debug images to {self.debug_image_topic}")
        self.get_logger().info(f"Publishing ArUco poses to {self.aruco_pose_topic}")
        self.get_logger().info(
            f"Publishing live nub poses to {self.live_nub_pose_topic}"
        )
        self.get_logger().info(
            f"Publishing live nub detections to {self.live_nub_detection_topic}"
        )
        self.get_logger().info(f"Snapshot service: {self.snapshot_service_name}")
        self.get_logger().info(f"Drawing axes for ArUco marker {self.aruco_marker_id}")
        if self.enable_tuning_window:
            self.get_logger().info(
                f"Loading/saving white tuning at {self.tuning_config_path}"
            )

    def camera_info_callback(self, msg):
        self.info_msg = msg
        self.camera_matrix = np.reshape(np.array(msg.k), (3, 3))
        self.distortion = np.array(msg.d)
        self.destroy_subscription(self.camera_info_sub)

    def process_frame(self, msg, depth_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")
            return

        if self.enable_tuning_window:
            try:
                self.ensure_tuning_window()
                self.read_tuning_window()
            except cv2.error as exc:
                self.get_logger().error(f"Disabling tuning window: {exc}")
                self.enable_tuning_window = False

        mask = self.find_white_mask(cv_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_image = cv_image.copy()
        patch_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = float(self.white_settings["min_patch_area"])
            max_area = float(self.white_settings["max_patch_area"])
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            if self.camera_matrix is not None:
                pt = self.get_3d_position(center_x, center_y, depth_image)
                if pt is not None:
                    self.get_logger().debug(
                        f"Piece at {self.target_frame}: "
                        f"x={pt.x:.3f} y={pt.y:.3f} z={pt.z:.3f}"
                    )
                    cv2.putText(
                        debug_image,
                        f"{pt.x:.2f},{pt.y:.2f},{pt.z:.2f}",
                        (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                    self.publish_point_transform(
                        f"piece_v2_{patch_count}",
                        pt,
                        msg.header.stamp,
                    )

            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)
            cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
            patch_count += 1

        green_candidates = self.green_nub_detector.detect(cv_image)
        nub_count = 0
        live_nub_poses = PoseArray()
        live_nub_poses.header = msg.header
        live_nub_poses.header.frame_id = self.target_frame
        live_detection_values = []

        for candidate in green_candidates:
            cx = candidate["u"]
            cy = candidate["v"]
            contour = candidate["contour"]

            cv2.circle(debug_image, (cx, cy), 6, (0, 255, 0), -1)
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)

            if self.camera_matrix is not None:
                pt = self.get_3d_position(cx, cy, depth_image)
                if pt is not None:
                    cv2.putText(
                        debug_image,
                        f"NUB {pt.x:.2f},{pt.y:.2f},{pt.z:.2f}",
                        (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    self.publish_point_transform(
                        f"nub_v2_{nub_count}",
                        pt,
                        msg.header.stamp,
                    )

                    pose = Pose()
                    pose.position.x = float(pt.x)
                    pose.position.y = float(pt.y)
                    pose.position.z = float(pt.z)
                    pose.orientation.w = 1.0
                    live_nub_poses.poses.append(pose)
                    live_detection_values.extend(
                        pack_nub_detection(
                            pt,
                            cx,
                            cy,
                            candidate["area"],
                            candidate["circularity"],
                        )
                    )
            nub_count += 1

        self.live_nub_poses_pub.publish(live_nub_poses)
        self.publish_live_nub_detection_array(live_detection_values)

        self.process_aruco_marker(cv_image, debug_image, msg.header.stamp)

        if self.enable_tuning_window:
            self.show_tuning_window(debug_image, mask)
            if self.save_requested:
                self.save_white_settings()
                self.save_requested = False

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_image_pub.publish(debug_msg)

        self.get_logger().debug(f"Detected {patch_count} white patches")

    def find_white_mask(self, cv_image):
        blur_kernel_size = self.get_odd_kernel_size(
            int(self.white_settings["blur_kernel_size"])
        )
        morph_kernel_size = self.get_odd_kernel_size(
            int(self.white_settings["morph_kernel_size"])
        )

        working_image = cv_image
        if blur_kernel_size > 1:
            working_image = cv2.GaussianBlur(
                working_image, (blur_kernel_size, blur_kernel_size), 0
            )

        hsv_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array(
            [
                int(self.white_settings["h_min"]),
                int(self.white_settings["s_min"]),
                int(self.white_settings["v_min"]),
            ],
            dtype=np.uint8,
        )
        upper_white = np.array(
            [
                int(self.white_settings["h_max"]),
                int(self.white_settings["s_max"]),
                int(self.white_settings["v_max"]),
            ],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv_image, lower_white, upper_white)

        if morph_kernel_size > 1:
            kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def load_green_settings(self):
        return {
            "green_h_min": int(self.get_parameter("green_h_min").value),
            "green_h_max": int(self.get_parameter("green_h_max").value),
            "green_s_min": int(self.get_parameter("green_s_min").value),
            "green_s_max": int(self.get_parameter("green_s_max").value),
            "green_v_min": int(self.get_parameter("green_v_min").value),
            "green_v_max": int(self.get_parameter("green_v_max").value),
            "green_min_area": float(self.get_parameter("green_min_area").value),
            "green_max_area": float(self.get_parameter("green_max_area").value),
            "green_min_circularity": float(
                self.get_parameter("green_min_circularity").value
            ),
            "green_blur_kernel_size": int(
                self.get_parameter("green_blur_kernel_size").value
            ),
            "green_morph_kernel_size": int(
                self.get_parameter("green_morph_kernel_size").value
            ),
        }

    def load_white_settings(self):
        defaults = {
            "h_min": 0,
            "h_max": 179,
            "s_min": 0,
            "s_max": int(self.get_parameter("white_max_saturation").value),
            "v_min": int(self.get_parameter("white_min_value").value),
            "v_max": 255,
            "min_patch_area": int(self.get_parameter("min_patch_area").value),
            "max_patch_area": int(self.get_parameter("max_patch_area").value),
            "blur_kernel_size": int(self.get_parameter("blur_kernel_size").value),
            "morph_kernel_size": int(self.get_parameter("morph_kernel_size").value),
        }

        if not os.path.exists(self.tuning_config_path):
            return defaults

        try:
            with open(self.tuning_config_path, "r", encoding="utf-8") as config_file:
                loaded = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warn(
                f"Could not load {self.tuning_config_path}: {exc}; using defaults"
            )
            return defaults

        settings = defaults.copy()
        for key in settings:
            if key in loaded:
                settings[key] = int(loaded[key])

        self.get_logger().info(f"Loaded white tuning from {self.tuning_config_path}")
        return self.clamp_white_settings(settings)

    def save_white_settings(self):
        config_dir = os.path.dirname(os.path.abspath(self.tuning_config_path))
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

        try:
            with open(self.tuning_config_path, "w", encoding="utf-8") as config_file:
                json.dump(
                    self.clamp_white_settings(self.white_settings),
                    config_file,
                    indent=2,
                )
                config_file.write("\n")
        except OSError as exc:
            self.get_logger().error(
                f"Could not save white tuning to {self.tuning_config_path}: {exc}"
            )
            return

        self.get_logger().info(f"Saved white tuning to {self.tuning_config_path}")

    def clamp_white_settings(self, settings):
        limits = {
            "h_min": (0, 179),
            "h_max": (0, 179),
            "s_min": (0, 255),
            "s_max": (0, 255),
            "v_min": (0, 255),
            "v_max": (0, 255),
            "min_patch_area": (0, 200000),
            "max_patch_area": (0, 200000),
            "blur_kernel_size": (0, 31),
            "morph_kernel_size": (0, 31),
        }

        clamped = {}
        for key, value in settings.items():
            low, high = limits[key]
            clamped[key] = max(low, min(high, int(value)))

        if clamped["h_min"] > clamped["h_max"]:
            clamped["h_min"], clamped["h_max"] = clamped["h_max"], clamped["h_min"]
        if clamped["s_min"] > clamped["s_max"]:
            clamped["s_min"], clamped["s_max"] = clamped["s_max"], clamped["s_min"]
        if clamped["v_min"] > clamped["v_max"]:
            clamped["v_min"], clamped["v_max"] = clamped["v_max"], clamped["v_min"]
        return clamped

    def ensure_tuning_window(self):
        if self.tuning_window_ready:
            return

        cv2.namedWindow(self.tuning_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.tuning_window_name, 1280, 720)

        trackbars = {
            "H min": ("h_min", 179),
            "H max": ("h_max", 179),
            "S min": ("s_min", 255),
            "S max": ("s_max", 255),
            "V min": ("v_min", 255),
            "V max": ("v_max", 255),
            "Min area": ("min_patch_area", 200000),
            "Max area": ("max_patch_area", 200000),
            "Blur": ("blur_kernel_size", 31),
            "Morph": ("morph_kernel_size", 31),
        }

        for label, (key, maximum) in trackbars.items():
            cv2.createTrackbar(
                label,
                self.tuning_window_name,
                int(self.white_settings[key]),
                maximum,
                self.noop_trackbar_callback,
            )

        try:
            cv2.createButton(
                "Save white_detection_params_v2.json",
                self.tuning_button_callback,
                None,
                cv2.QT_PUSH_BUTTON,
                0,
            )
        except (AttributeError, cv2.error):
            cv2.createTrackbar(
                "Save settings",
                self.tuning_window_name,
                0,
                1,
                self.noop_trackbar_callback,
            )
            self.save_trackbar_armed = True

        try:
            cv2.createButton(
                "Take nub snapshot",
                self.snapshot_button_callback,
                None,
                cv2.QT_PUSH_BUTTON,
                0,
            )
        except (AttributeError, cv2.error):
            cv2.createTrackbar(
                "Take snapshot",
                self.tuning_window_name,
                0,
                1,
                self.noop_trackbar_callback,
            )
            self.snapshot_trackbar_armed = True

        self.tuning_window_ready = True

    def read_tuning_window(self):
        trackbars = {
            "H min": "h_min",
            "H max": "h_max",
            "S min": "s_min",
            "S max": "s_max",
            "V min": "v_min",
            "V max": "v_max",
            "Min area": "min_patch_area",
            "Max area": "max_patch_area",
            "Blur": "blur_kernel_size",
            "Morph": "morph_kernel_size",
        }

        for label, key in trackbars.items():
            self.white_settings[key] = cv2.getTrackbarPos(
                label, self.tuning_window_name
            )

        self.white_settings = self.clamp_white_settings(self.white_settings)

        if self.save_trackbar_armed:
            save_value = cv2.getTrackbarPos("Save settings", self.tuning_window_name)
            if save_value == 1:
                self.save_requested = True
                cv2.setTrackbarPos("Save settings", self.tuning_window_name, 0)

        if self.snapshot_trackbar_armed:
            snapshot_value = cv2.getTrackbarPos("Take snapshot", self.tuning_window_name)
            if snapshot_value == 1:
                self.call_snapshot_service()
                cv2.setTrackbarPos("Take snapshot", self.tuning_window_name, 0)

    def show_tuning_window(self, debug_image, mask):
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if self.snapshot_request_pending:
            status = "Snapshot request pending"
        elif self.snapshot_client.service_is_ready():
            status = self.last_snapshot_status
        else:
            status = "Snapshot service not ready"

        cv2.putText(
            debug_image,
            status,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            mask_bgr,
            "Mask preview",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        preview = np.hstack((debug_image, mask_bgr))
        cv2.imshow(self.tuning_window_name, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            self.save_requested = True

    def tuning_button_callback(self, *args):
        self.save_requested = True

    def snapshot_button_callback(self, *args):
        self.call_snapshot_service()

    def noop_trackbar_callback(self, value):
        pass

    def call_snapshot_service(self):
        if self.snapshot_request_pending:
            return

        if not self.snapshot_client.service_is_ready():
            self.last_snapshot_status = "Snapshot service not ready"
            self.get_logger().warn(
                f"Snapshot service {self.snapshot_service_name} is not ready"
            )
            return

        self.snapshot_request_pending = True
        self.last_snapshot_status = "Snapshot requested"
        future = self.snapshot_client.call_async(Trigger.Request())
        future.add_done_callback(self.snapshot_response_callback)

    def snapshot_response_callback(self, future):
        self.snapshot_request_pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.last_snapshot_status = "Snapshot service failed"
            self.get_logger().error(f"Snapshot service call failed: {exc}")
            return

        self.last_snapshot_status = response.message
        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().warn(response.message)

    def publish_live_nub_detection_array(self, values):
        detections_msg = Float32MultiArray()
        detection_count = len(values) // NUB_DETECTION_FIELD_COUNT
        detections_msg.layout.dim = make_nub_detection_layout(detection_count)
        detections_msg.data = values
        self.live_nub_detections_pub.publish(detections_msg)

    def publish_point_transform(self, child_frame_id, point, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.target_frame
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = point.x
        transform.transform.translation.y = point.y
        transform.transform.translation.z = point.z
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)

    def process_aruco_marker(self, source_image, debug_image, stamp):
        if (
            self.info_msg is None
            or self.camera_matrix is None
            or self.distortion is None
        ):
            self.get_logger().debug("No camera info has been received")
            return

        gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray_image,
            self.aruco_dictionary,
            parameters=self.aruco_parameters,
        )

        if marker_ids is None:
            return

        cv2.aruco.drawDetectedMarkers(debug_image, corners, marker_ids)
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.get_aruco_frame_id()

        for index, marker_id in enumerate(marker_ids.flatten()):
            if marker_id != self.aruco_marker_id:
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[index]],
                self.aruco_marker_size,
                self.camera_matrix,
                self.distortion,
            )
            pose = self.pose_from_rvec_tvec(rvecs[0], tvecs[0])
            pose_array.poses.append(pose)
            self.aruco_poses_pub.publish(pose_array)
            self.publish_aruco_transform(pose, stamp)
            self.draw_frame_axes(debug_image, rvecs[0], tvecs[0])
            return

    def pose_from_rvec_tvec(self, rvec, tvec):
        pose = Pose()
        pose.position.x = float(tvec[0][0])
        pose.position.y = float(tvec[0][1])
        pose.position.z = float(tvec[0][2])

        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvec[0]))[0]
        quat = quaternion_from_matrix(rotation_matrix)

        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose

    def publish_aruco_transform(self, pose, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.get_aruco_frame_id()
        transform.child_frame_id = f"ar_marker_v2_{self.aruco_marker_id}"
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w
        self.tf_broadcaster.sendTransform(transform)

    def get_aruco_frame_id(self):
        if self.camera_frame:
            return self.camera_frame
        return self.info_msg.header.frame_id

    def draw_frame_axes(self, image, rvec, tvec):
        if hasattr(cv2, "drawFrameAxes"):
            cv2.drawFrameAxes(
                image,
                self.camera_matrix,
                self.distortion,
                rvec,
                tvec,
                self.aruco_axis_length,
            )
        else:
            cv2.aruco.drawAxis(
                image,
                self.camera_matrix,
                self.distortion,
                rvec,
                tvec,
                self.aruco_axis_length,
            )

    def get_aruco_dictionary(self, dictionary_name):
        try:
            dictionary_id = getattr(cv2.aruco, dictionary_name)
        except AttributeError:
            self.get_logger().error(f"Invalid aruco_dictionary_id: {dictionary_name}")
            dictionary_id = cv2.aruco.DICT_5X5_250

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            return cv2.aruco.getPredefinedDictionary(dictionary_id)
        return cv2.aruco.Dictionary_get(dictionary_id)

    def create_aruco_parameters(self):
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            return cv2.aruco.DetectorParameters_create()
        return cv2.aruco.DetectorParameters()

    def get_odd_kernel_size(self, kernel_size):
        if kernel_size < 1:
            return 1
        if kernel_size % 2 == 0:
            return kernel_size + 1
        return kernel_size

    def get_3d_position(self, u, v, depth_image):
        # sample 7x7 neighborhood
        h, w = depth_image.shape
        u, v = int(u), int(v)
        u1, u2 = max(0, u - 3), min(w, u + 4)
        v1, v2 = max(0, v - 3), min(h, v + 4)
        patch = depth_image[v1:v2, u1:u2]
        nonzero = patch[patch > 0]
        if len(nonzero) == 0:
            return None
        z = float(np.median(nonzero)) / 1000.0
        if z <= 0 or z > 1.5:
            return None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        camera_cx = self.camera_matrix[0, 2]
        camera_cy = self.camera_matrix[1, 2]

        x = (u - camera_cx) * z / fx
        y = (v - camera_cy) * z / fy

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                Time(),
            )
            pt = PointStamped()
            pt.header.frame_id = self.camera_frame
            pt.point.x = x
            pt.point.y = y
            pt.point.z = z
            pt_base = tf2_geometry_msgs.do_transform_point(pt, transform)
            return pt_base.point
        except Exception as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}")
            return None

    def synced_callback(self, rgb_msg, depth_msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
        except Exception as exc:
            self.get_logger().error(f"Depth conversion failed: {exc}")
            return
        self.process_frame(rgb_msg, depth_image)


def main(args=None):
    rclpy.init(args=args)
    node = WhitePatchDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
