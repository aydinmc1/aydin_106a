import rclpy
from rclpy.node import Node


class ExampleNode(Node):
    def __init__(self):
        super().__init__("example_node")
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info("aydin_v2 example node is running")


def main(args=None):
    rclpy.init(args=args)
    node = ExampleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
