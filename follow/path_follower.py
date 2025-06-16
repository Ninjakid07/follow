#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower_node')

        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_navigating = False

        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.get_logger().info('Path follower node has been initialized.')
        
        self.get_logger().info('Waiting for NavigateToPose action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Action server is available.')

        self.path_subscriber = self.create_subscription(
            Path,
            '/path',
            self.path_callback,
            10
        )
        self.get_logger().info('Waiting for a path on the /path topic...')


    def path_callback(self, msg: Path):
        if self.is_navigating:
            self.get_logger().warn('Currently navigating, ignoring new path.')
            return

        if not msg.poses:
            self.get_logger().info('Received an empty path. Nothing to do.')
            return

        self.get_logger().info(f'Received a path with {len(msg.poses)} waypoints. Starting navigation.')
        
        self.waypoints = msg.poses
        self.current_waypoint_index = 0
        self.is_navigating = True
        
        self.navigate_to_next_waypoint()

    def navigate_to_next_waypoint(self):
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info('All waypoints have been processed. Navigation complete!')
            self.is_navigating = False
            self.waypoints = []
            return

        target_pose = self.waypoints[self.current_waypoint_index]

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        
        if not self._action_client.wait_for_server(timeout_sec=1.0):
             self.get_logger().error('NavigateToPose action server disappeared. Aborting.')
             self.is_navigating = False
             return

        self.get_logger().info(f'Sending waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} as goal.')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the server.')
            self.current_waypoint_index += 1
            self.navigate_to_next_waypoint()
            return

        self.get_logger().info('Goal accepted. Waiting for result...')
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Successfully reached waypoint {self.current_waypoint_index + 1}.')
        else:
            self.get_logger().warn(f'Failed to reach waypoint {self.current_waypoint_index + 1}. Status: {status}')

        self.current_waypoint_index += 1
        self.navigate_to_next_waypoint()

    def feedback_callback(self, feedback_msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    path_follower_node = PathFollower()
    
    try:
        rclpy.spin(path_follower_node)
    except KeyboardInterrupt:
        pass
        
    path_follower_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
