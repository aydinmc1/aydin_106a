import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


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
        self.declare_parameter("white_min_value", 200)
        self.declare_parameter("white_max_saturation", 45)
        self.declare_parameter("min_patch_area", 150.0)
        self.declare_parameter("blur_kernel_size", 5)
        self.declare_parameter("morph_kernel_size", 5)

        self.bridge = CvBridge()
        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.aruco_marker_id = int(self.get_parameter("aruco_marker_id").value)
        self.aruco_marker_size = float(self.get_parameter("aruco_marker_size").value)
        self.aruco_axis_length = float(self.get_parameter("aruco_axis_length").value)
        self.camera_matrix = None
        self.distortion = None

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

        self.get_logger().info(f"Subscribing to {self.image_topic}")
        self.get_logger().info(f"Subscribing to {self.camera_info_topic}")
        self.get_logger().info(f"Publishing debug images to {self.debug_image_topic}")
        self.get_logger().info(f"Drawing axes for ArUco marker {self.aruco_marker_id}")

    def camera_info_callback(self, msg):
        self.camera_matrix = np.reshape(np.array(msg.k), (3, 3))
        self.distortion = np.array(msg.d)
        self.destroy_subscription(self.camera_info_sub)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")
            return

        mask = self.find_white_mask(cv_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_image = cv_image.copy()
        patch_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = float(self.get_parameter("min_patch_area").value)
            if area < min_area:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)
            cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
            patch_count += 1

        self.draw_aruco_axes(debug_image)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_image_pub.publish(debug_msg)

        self.get_logger().debug(f"Detected {patch_count} white patches")

    def find_white_mask(self, cv_image):
        blur_kernel_size = self.get_odd_kernel_size("blur_kernel_size")
        morph_kernel_size = self.get_odd_kernel_size("morph_kernel_size")
        white_min_value = int(self.get_parameter("white_min_value").value)
        white_max_saturation = int(self.get_parameter("white_max_saturation").value)

        working_image = cv_image
        if blur_kernel_size > 1:
            working_image = cv2.GaussianBlur(
                working_image, (blur_kernel_size, blur_kernel_size), 0
            )

        hsv_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, white_min_value], dtype=np.uint8)
        upper_white = np.array([179, white_max_saturation, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_white, upper_white)

        if morph_kernel_size > 1:
            kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def draw_aruco_axes(self, debug_image):
        if self.camera_matrix is None or self.distortion is None:
            self.get_logger().debug("No camera info has been received")
            return

        gray_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2GRAY)
        corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray_image,
            self.aruco_dictionary,
            parameters=self.aruco_parameters,
        )

        if marker_ids is None:
            return

        cv2.aruco.drawDetectedMarkers(debug_image, corners, marker_ids)
        for index, marker_id in enumerate(marker_ids.flatten()):
            if marker_id != self.aruco_marker_id:
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[index]],
                self.aruco_marker_size,
                self.camera_matrix,
                self.distortion,
            )
            self.draw_frame_axes(debug_image, rvecs[0], tvecs[0])
            return

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

    def get_odd_kernel_size(self, parameter_name):
        kernel_size = int(self.get_parameter(parameter_name).value)
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
