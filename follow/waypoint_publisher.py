#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('waypoint_publisher_node')

        # SET YOUR WAYPOINT COORDINATES (X, Y) HERE
        self.waypoints = [
            (0.0, 0.0),
            (0.3, 0.0),
            (0.6, 0.0),
            (0.9, 0.0),
            (1.0, 0.0)
        ]

        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.path_publisher = self.create_publisher(
            Path,
            '/path',
            qos_profile
        )

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

        for point in self.waypoints:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'

            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0

            pose.pose.orientation.w = 1.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0

            self.path_msg.poses.append(pose)

        self.get_logger().info(f'Created a path with {len(self.path_msg.poses)} waypoints.')
        self.timer = self.create_timer(1.0, self.publish_path)

    def publish_path(self):
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_publisher.publish(self.path_msg)
        self.get_logger().info("Publishing path to '/path' topic...")


def main(args=None):
    rclpy.init(args=args)
    waypoint_publisher_node = WaypointPublisher()
    
    try:
        rclpy.spin(waypoint_publisher_node)
    except KeyboardInterrupt:
        pass
    
    waypoint_publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()