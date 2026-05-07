import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Pose, PoseArray, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from aydin_v2.nub_detection import iter_nub_detection_rows


class NubSnapshotNode(Node):
    def __init__(self):
        super().__init__("nub_snapshot_node_v2")

        self.declare_parameter(
            "live_nub_detection_topic", "/aydin_v2/live_nubs_v2/detections"
        )
        self.declare_parameter(
            "debug_image_topic", "/aydin_v2/green_nubs/debug_image_v2"
        )
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("snapshot_duration", 3.0)
        self.declare_parameter("snapshot_expected_nubs", 11)
        self.declare_parameter("snapshot_cluster_radius", 0.025)
        self.declare_parameter("snapshot_min_samples", 5)
        self.declare_parameter(
            "snapshot_pose_topic", "/aydin_v2/nub_snapshot_v2/poses"
        )
        self.declare_parameter(
            "snapshot_point_topic", "/aydin_v2/nub_snapshot_v2/points"
        )
        self.declare_parameter("snapshot_frame_prefix", "snapshot_nub_v2")
        self.declare_parameter("snapshot_service_name", "/take_nub_snapshot_v2")
        self.declare_parameter("enable_snapshot_popup", True)

        self.live_nub_detection_topic = self.get_parameter(
            "live_nub_detection_topic"
        ).value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.snapshot_duration = float(self.get_parameter("snapshot_duration").value)
        self.snapshot_expected_nubs = int(
            self.get_parameter("snapshot_expected_nubs").value
        )
        self.snapshot_cluster_radius = float(
            self.get_parameter("snapshot_cluster_radius").value
        )
        self.snapshot_min_samples = int(self.get_parameter("snapshot_min_samples").value)
        self.snapshot_pose_topic = self.get_parameter("snapshot_pose_topic").value
        self.snapshot_point_topic = self.get_parameter("snapshot_point_topic").value
        self.snapshot_frame_prefix = self.get_parameter("snapshot_frame_prefix").value
        self.snapshot_service_name = self.get_parameter("snapshot_service_name").value
        self.enable_snapshot_popup = bool(
            self.get_parameter("enable_snapshot_popup").value
        )

        self.bridge = CvBridge()
        self.latest_debug_image = None
        self.snapshot_active = False
        self.snapshot_start_time = None
        self.snapshot_samples = []
        self.snapshot_id = 0
        self.latest_snapshot_clusters = []
        self.latest_pose_array = None
        self.latest_point_stamps = []

        self.detections_sub = self.create_subscription(
            Float32MultiArray,
            self.live_nub_detection_topic,
            self.live_detections_callback,
            10,
        )
        self.debug_image_sub = self.create_subscription(
            Image,
            self.debug_image_topic,
            self.debug_image_callback,
            10,
        )
        self.snapshot_poses_pub = self.create_publisher(
            PoseArray, self.snapshot_pose_topic, 10
        )
        self.snapshot_points_pub = self.create_publisher(
            PointStamped, self.snapshot_point_topic, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.snapshot_service = self.create_service(
            Trigger, self.snapshot_service_name, self.take_snapshot_callback
        )
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f"Listening for live nubs on {self.live_nub_detection_topic}"
        )
        self.get_logger().info(f"Snapshot service: {self.snapshot_service_name}")
        self.get_logger().info(f"Publishing snapshot poses to {self.snapshot_pose_topic}")
        self.get_logger().info(
            f"Publishing snapshot points to {self.snapshot_point_topic}"
        )

    def live_detections_callback(self, msg):
        if not self.snapshot_active:
            return

        for fields in iter_nub_detection_rows(list(msg.data)):
            self.snapshot_samples.append(
                {
                    "xyz": [float(fields[0]), float(fields[1]), float(fields[2])],
                    "uv": [float(fields[3]), float(fields[4])],
                    "area": float(fields[5]),
                    "circularity": float(fields[6]),
                }
            )

    def debug_image_callback(self, msg):
        try:
            self.latest_debug_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            ).copy()
        except Exception as exc:
            self.get_logger().warn(f"Could not convert debug image: {exc}")

    def take_snapshot_callback(self, request, response):
        del request
        if self.snapshot_active:
            response.success = False
            response.message = (
                f"Snapshot {self.snapshot_id} is already running; "
                "wait for it to finish before starting another"
            )
            return response

        self.start_snapshot()
        response.success = True
        response.message = (
            f"Started {self.snapshot_duration:.1f}s green nub snapshot "
            f"{self.snapshot_id}"
        )
        return response

    def start_snapshot(self):
        self.snapshot_active = True
        self.snapshot_start_time = self.get_clock().now()
        self.snapshot_samples = []
        self.latest_snapshot_clusters = []
        self.snapshot_id += 1
        self.clear_snapshot_outputs()
        self.get_logger().info(
            f"Started snapshot {self.snapshot_id}: {self.snapshot_duration:.1f}s, "
            f"expecting {self.snapshot_expected_nubs} nubs"
        )

    def timer_callback(self):
        if self.snapshot_active:
            elapsed = (
                self.get_clock().now() - self.snapshot_start_time
            ).nanoseconds * 1e-9
            if elapsed >= self.snapshot_duration:
                self.snapshot_active = False
                self.finalize_snapshot()

        self.publish_latest_snapshot_transforms()
        self.publish_latest_snapshot_messages()

    def finalize_snapshot(self):
        clusters = self.cluster_snapshot_samples()
        self.latest_snapshot_clusters = clusters
        self.publish_snapshot_outputs(clusters)
        self.show_snapshot_popup(clusters)
        self.get_logger().info(
            f"Finished snapshot {self.snapshot_id}: "
            f"{len(self.snapshot_samples)} samples -> {len(clusters)} nubs"
        )

    def cluster_snapshot_samples(self):
        if not self.snapshot_samples:
            return []

        points = np.array([sample["xyz"] for sample in self.snapshot_samples])
        labels = self.dbscan_xy(points, self.snapshot_cluster_radius)
        clusters = []

        for label in sorted(set(labels)):
            if label == -1:
                continue

            indexes = np.where(labels == label)[0]
            if len(indexes) < self.snapshot_min_samples:
                continue

            cluster_points = points[indexes]
            cluster_uvs = np.array(
                [self.snapshot_samples[index]["uv"] for index in indexes]
            )
            median_xyz = np.median(cluster_points, axis=0)
            median_uv = np.median(cluster_uvs, axis=0)
            xy_spread = float(
                np.max(np.linalg.norm(cluster_points[:, :2] - median_xyz[:2], axis=1))
            )
            pose = Pose()
            pose.position.x = float(median_xyz[0])
            pose.position.y = float(median_xyz[1])
            pose.position.z = float(median_xyz[2])
            pose.orientation.w = 1.0
            clusters.append(
                {
                    "pose": pose,
                    "uv": [int(round(median_uv[0])), int(round(median_uv[1]))],
                    "samples": int(len(indexes)),
                    "spread": xy_spread,
                }
            )

        clusters.sort(key=lambda cluster: cluster["samples"], reverse=True)
        return clusters[: self.snapshot_expected_nubs]

    def dbscan_xy(self, points, eps):
        labels = np.full(len(points), -2, dtype=int)
        cluster_id = 0

        for point_index in range(len(points)):
            if labels[point_index] != -2:
                continue

            neighbors = self.region_query_xy(points, point_index, eps)
            if len(neighbors) < self.snapshot_min_samples:
                labels[point_index] = -1
                continue

            labels[point_index] = cluster_id
            seeds = list(neighbors)
            seed_cursor = 0
            while seed_cursor < len(seeds):
                neighbor_index = seeds[seed_cursor]
                if labels[neighbor_index] == -1:
                    labels[neighbor_index] = cluster_id
                if labels[neighbor_index] != -2:
                    seed_cursor += 1
                    continue

                labels[neighbor_index] = cluster_id
                expanded_neighbors = self.region_query_xy(points, neighbor_index, eps)
                if len(expanded_neighbors) >= self.snapshot_min_samples:
                    for expanded_index in expanded_neighbors:
                        if expanded_index not in seeds:
                            seeds.append(expanded_index)
                seed_cursor += 1

            cluster_id += 1

        return labels

    def region_query_xy(self, points, point_index, eps):
        distances = np.linalg.norm(points[:, :2] - points[point_index, :2], axis=1)
        return np.where(distances <= eps)[0].tolist()

    def clear_snapshot_outputs(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.target_frame
        self.latest_pose_array = pose_array
        self.latest_point_stamps = []
        self.snapshot_poses_pub.publish(pose_array)

    def publish_snapshot_outputs(self, clusters):
        stamp = self.get_clock().now().to_msg()

        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.target_frame
        pose_array.poses = [cluster["pose"] for cluster in clusters]
        self.latest_pose_array = pose_array
        self.snapshot_poses_pub.publish(pose_array)

        point_stamps = []
        for index, cluster in enumerate(clusters):
            pose = cluster["pose"]
            frame_name = f"{self.snapshot_frame_prefix}_{index}"
            self.publish_snapshot_transform(frame_name, pose, stamp)

            point_stamp = self.point_stamp_from_pose(pose, stamp)
            point_stamps.append(point_stamp)
            self.snapshot_points_pub.publish(point_stamp)

        self.latest_point_stamps = point_stamps

    def point_stamp_from_pose(self, pose, stamp):
        point_stamp = PointStamped()
        point_stamp.header.stamp = stamp
        point_stamp.header.frame_id = self.target_frame
        point_stamp.point.x = pose.position.x
        point_stamp.point.y = pose.position.y
        point_stamp.point.z = pose.position.z
        return point_stamp

    def publish_snapshot_transform(self, frame_name, pose, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.target_frame
        transform.child_frame_id = frame_name
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w
        self.tf_broadcaster.sendTransform(transform)

    def publish_latest_snapshot_transforms(self):
        if not self.latest_snapshot_clusters:
            return

        stamp = self.get_clock().now().to_msg()
        for index, cluster in enumerate(self.latest_snapshot_clusters):
            self.publish_snapshot_transform(
                f"{self.snapshot_frame_prefix}_{index}",
                cluster["pose"],
                stamp,
            )

    def publish_latest_snapshot_messages(self):
        stamp = self.get_clock().now().to_msg()

        if self.latest_pose_array is not None:
            self.latest_pose_array.header.stamp = stamp
            self.snapshot_poses_pub.publish(self.latest_pose_array)

        for point_stamp in self.latest_point_stamps:
            point_stamp.header.stamp = stamp
            self.snapshot_points_pub.publish(point_stamp)

    def show_snapshot_popup(self, clusters):
        if not self.enable_snapshot_popup or self.latest_debug_image is None:
            return

        snapshot_image = self.latest_debug_image.copy()
        for index, cluster in enumerate(clusters):
            u, v = cluster["uv"]
            cv2.circle(snapshot_image, (u, v), 8, (0, 0, 255), -1)
            cv2.putText(
                snapshot_image,
                str(index),
                (u + 10, v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        try:
            cv2.imshow("aydin_v2 nub snapshot result", snapshot_image)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.get_logger().warn(f"Could not show snapshot popup: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = NubSnapshotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyWindow("aydin_v2 nub snapshot result")
        except cv2.error:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
