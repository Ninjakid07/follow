#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class Nav2MissionPlanner(Node):
    def __init__(self):
        super().__init__('nav2_mission_planner')

        # --- Waypoints: (x, y, yaw_degrees) ---
        self.waypoints: List[Tuple[float, float, float]] = [
            # (1.2, 0.0,   0),
            # (1.2, 0.6,  90),
            # (0.0, 0.6, 180),

            (1.2, 0.0,   0),  # P1
            (1.2, 0.6,  90),  # P2
            (0.0, 0.6, 90),   # P3
            (0.0, 1.2, -90),  # P4 
            (1.2, 1.2, -90),  # P5
            (0.0, 0.0, -90),  # P6 (HOME) 
            
            # (1.8, 0.0,   0),  # P1
            # (1.8, 0.6,  90),  # P2
            # (0.0, 0.6, 90),   # P3
            # (0.0, 1.8, -90),  # P4 
            # (1.8, 1.8, -90),  # P5
        ]

        self.nav_action = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_idx = 0
        self.is_sending = False

        self.timer = self.create_timer(1.0, self._tick)

    def _pose_from_xyyaw(self, x: float, y: float, yaw_deg: float) -> PoseStamped:
        yaw_rad = math.radians(yaw_deg)
        q = Quaternion()
        q.z = math.sin(yaw_rad / 2.0)
        q.w = math.cos(yaw_rad / 2.0)

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = q
        return pose

    def _tick(self):
        if not self.nav_action.server_is_ready():
            self.get_logger().info('Waiting for Nav2 action server...')
            return
        if not self.is_sending:
            self._send_next_goal()

    def _send_next_goal(self):
        if self.current_idx >= len(self.waypoints):
            self.get_logger().info('🎉 Mission complete – all waypoints reached!')
            self.destroy_timer(self.timer)
            return

        x, y, yaw = self.waypoints[self.current_idx]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._pose_from_xyyaw(x, y, yaw)

        self.get_logger().info(
            f'🚩 Sending waypoint {self.current_idx + 1}/{len(self.waypoints)}: '
            f'({x:.2f}, {y:.2f}, {yaw:.1f}°)')
        self.is_sending = True
        self._send_future = self.nav_action.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_cb)
        self._send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('❌ Goal rejected by Nav2')
            self.is_sending = False
            return

        self.get_logger().info('✔️ Goal accepted, navigating...')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().debug(
            f'Distance to goal: {fb.distance_remaining:.2f} m')

    def _result_cb(self, future):
        # result = future.result().result
        # if result.error_code == 0:
        #     self.get_logger().info('✅ Goal reached successfully')
        # else:
        #     self.get_logger().warn(f'⚠️ Goal failed with error code {result.error_code}')
        # self.current_idx += 1
        # self.is_sending = False
        status = future.result().status          # <‑‑ USE STATUS
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('✅ Goal reached successfully')
        else:
            self.get_logger().warn(f'⚠️ Goal ended with status {status}')
        self.current_idx += 1
        self.is_sending = False        


def main(args=None):
    rclpy.init(args=args)
    node = Nav2MissionPlanner()
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
