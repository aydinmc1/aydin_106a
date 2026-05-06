import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class WhitePatchDetectorNode(Node):
    def __init__(self):
        super().__init__("white_patch_detector_node")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter(
            "debug_image_topic", "/aydin_v2/white_patches/debug_image"
        )
        self.declare_parameter("white_min_value", 200)
        self.declare_parameter("white_max_saturation", 45)
        self.declare_parameter("min_patch_area", 150.0)
        self.declare_parameter("blur_kernel_size", 5)
        self.declare_parameter("morph_kernel_size", 5)

        self.bridge = CvBridge()
        self.image_topic = self.get_parameter("image_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )
        self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self.get_logger().info(f"Subscribing to {self.image_topic}")
        self.get_logger().info(f"Publishing debug images to {self.debug_image_topic}")

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
