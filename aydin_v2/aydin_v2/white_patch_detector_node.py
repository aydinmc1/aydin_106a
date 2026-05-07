import json
import math
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster


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
        super().__init__("white_patch_detector_node")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter(
            "debug_image_topic", "/aydin_v2/white_patches/debug_image"
        )
        self.declare_parameter("aruco_dictionary_id", "DICT_5X5_250")
        self.declare_parameter("aruco_marker_id", 1)
        self.declare_parameter("aruco_marker_size", 0.15)
        self.declare_parameter("aruco_axis_length", 0.075)
        self.declare_parameter("aruco_pose_topic", "aruco_poses")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("white_min_value", 200)
        self.declare_parameter("white_max_saturation", 45)
        self.declare_parameter("min_patch_area", 150.0)
        self.declare_parameter("max_patch_area", 0.0)
        self.declare_parameter("blur_kernel_size", 5)
        self.declare_parameter("morph_kernel_size", 5)
        self.declare_parameter("enable_tuning_window", True)
        self.declare_parameter(
            "tuning_config_path", "white_detection_params.json"
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
        self.white_settings = self.load_white_settings()

        dictionary_name = self.get_parameter("aruco_dictionary_id").value
        self.aruco_dictionary = self.get_aruco_dictionary(dictionary_name)
        self.aruco_parameters = self.create_aruco_parameters()

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
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
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(f"Subscribing to {self.image_topic}")
        self.get_logger().info(f"Subscribing to {self.camera_info_topic}")
        self.get_logger().info(f"Publishing debug images to {self.debug_image_topic}")
        self.get_logger().info(f"Publishing ArUco poses to {self.aruco_pose_topic}")
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

    def image_callback(self, msg):
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

            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)
            cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
            patch_count += 1

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
                "Save white_detection_params.json",
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

    def show_tuning_window(self, debug_image, mask):
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
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

    def noop_trackbar_callback(self, value):
        pass

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
        transform.child_frame_id = f"ar_marker_{self.aruco_marker_id}"
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
