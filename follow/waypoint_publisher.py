#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion

class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('waypoint_publisher_node')

        # Helper function to convert yaw angle (in radians) to a quaternion
        def create_quaternion_from_yaw(yaw):
            return Quaternion(
                x=0.0,
                y=0.0,
                z=math.sin(yaw / 2.0),
                w=math.cos(yaw / 2.0)
            )

        # SET YOUR WAYPOINT COORDINATES (X, Y, YAW in radians) HERE
        # Yaw: 0 is forward (X-axis), pi/2 is left (Y-axis), pi is backward (-X axis)
        self.waypoints = [
            # Start at (0,0) facing forward (0 rad yaw)
            (0.0, 0.0, 0.0),
            # Waypoint at (1.0, 0.0), but set orientation for the *next* segment (90-deg left turn)
            (1.0, 0.0, math.pi / 2.0),
            # Waypoint at (1.0, 0.5), set orientation for the *next* segment (another 90-deg left turn)
            (1.0, 0.5, math.pi),
            # Final waypoint at (0.0, 0.5), facing backward
            (0.0, 0.5, math.pi)
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

            pose.pose.orientation = create_quaternion_from_yaw(point[2])

            self.path_msg.poses.append(pose)

        self.get_logger().info(f'Created a path with {len(self.path_msg.poses)} waypoints.')
        self.timer = self.create_timer(1.0, self.publish_path)

    def publish_path(self):
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_publisher.publish(self.path_msg)
        self.get_logger().info("Publishing U-shaped path to '/path' topic...")


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
