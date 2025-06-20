# horizontal_cylinder_detector_final_robust_fix.py

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
# ... (all other imports are the same) ...
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped, Point, Pose, Vector3
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

# ... (All TUNING PARAMETERS are the same) ...
LOWER_RED_1 = np.array([0, 120, 70]); UPPER_RED_1 = np.array([10, 255, 255]); LOWER_RED_2 = np.array([165, 120, 70]); UPPER_RED_2 = np.array([180, 255, 255])
MIN_EDGE_CONTOUR_LENGTH = 30; MAX_HORIZONTAL_DEVIATION_PX = 40; MAX_DEPTH_DIFFERENCE_MM = 150; MIN_VERTICAL_SEPARATION_PX = 50
ASSOCIATION_THRESHOLD_METERS = 0.4; INITIALIZATION_DELAY_SEC = 5.0; MIN_DETECTION_RANGE_M = 0.69; MAX_DETECTION_RANGE_M = 1.5

class HorizontalCylinderDetector(Node):
    def __init__(self):
        super().__init__('horizontal_cylinder_detector_final_robust')
        # ... (__init__ is the same) ...
        if not self.has_parameter('use_sim_time'): self.declare_parameter('use_sim_time', True)
        sim_time_is_enabled = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        log_msg = 'Node is using simulation time.' if sim_time_is_enabled else 'Node is using system (wall) time.'
        self.get_logger().info(log_msg)
        self.start_time = self.get_clock().now(); self.queue_size = 30; self.bridge = CvBridge()
        self.latest_depth_image = None; self.camera_intrinsics = None; self.intrinsics_received = False
        self.known_cylinders = []; self.next_marker_id = 0
        self.get_logger().info('Setting TF buffer duration to 10.0 seconds.')
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0)); self.tf_listener = TransformListener(self.tf_buffer, self)
        self.source_frame, self.target_frame = 'oak_camera_rgb_camera_optical_frame', 'map'
        self.get_logger().info(f'Source frame: {self.source_frame}, Target frame: {self.target_frame}')
        self.image_sub = self.create_subscription(CompressedImage, '/oak/rgb/image_rect/compressed', self.image_callback, self.queue_size)
        self.depth_sub = self.create_subscription(Image, '/oak/stereo/image_raw', self.depth_callback, self.queue_size)
        self.info_sub = self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.info_callback, self.queue_size)
        self.marker_publisher = self.create_publisher(Marker, '~/permanent_cylinders', self.queue_size)
        self.get_logger().info('Cylinder Detector (Final Robust Fix) has started.')

    def depth_callback(self, msg):
        try: self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e: self.get_logger().error(f'Failed to convert depth image: {e}')

    def info_callback(self, msg):
        if not self.intrinsics_received:
            self.camera_intrinsics, self.intrinsics_received = msg, True; self.destroy_subscription(self.info_sub)
            self.get_logger().info('Camera intrinsics received.')

    def image_callback(self, msg):
        if self.latest_depth_image is None or not self.intrinsics_received: return
        try: cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e: self.get_logger().error(f'Failed to convert color image: {e}'); return
            
        # ... (CV detection logic is identical) ...
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV); mask1, mask2 = cv2.inRange(hsv_image, LOWER_RED_1, UPPER_RED_1), cv2.inRange(hsv_image, LOWER_RED_2, UPPER_RED_2)
        red_mask = cv2.bitwise_or(mask1, mask2); red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)); red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        h, w = red_mask.shape; top_edge_image, bottom_edge_image = np.zeros_like(red_mask), np.zeros_like(red_mask)
        cols_with_red = np.where(red_mask.max(axis=0) > 0)[0]
        if cols_with_red.size > 0:
            top_indices, bottom_indices = np.argmax(red_mask, axis=0)[cols_with_red], h - 1 - np.argmax(np.flipud(red_mask), axis=0)[cols_with_red]
            top_edge_image[top_indices, cols_with_red], bottom_edge_image[bottom_indices, cols_with_red] = 255, 255
        top_contours, _ = cv2.findContours(top_edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); bottom_contours, _ = cv2.findContours(bottom_edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_edges, bottom_edges = [c for c in top_contours if cv2.arcLength(c, False) > MIN_EDGE_CONTOUR_LENGTH], [c for c in bottom_contours if cv2.arcLength(c, False) > MIN_EDGE_CONTOUR_LENGTH]
        detected_cylinders = []; available_bottom_edges = list(bottom_edges) 
        for top_c in top_edges:
            x_top, y_top, w_top, h_top = cv2.boundingRect(top_c); cx_top, cy_top = x_top + w_top // 2, y_top + h_top // 2
            top_mask = np.zeros_like(red_mask); cv2.drawContours(top_mask, [top_c], -1, 255, thickness=5)
            top_depths = self.latest_depth_image[top_mask == 255]; top_depths = top_depths[top_depths > 0]
            if top_depths.size == 0: continue
            avg_depth_top = np.mean(top_depths)
            best_match, best_match_idx, best_match_score = None, -1, float('inf')
            for i, bottom_c in enumerate(available_bottom_edges):
                x_bot, y_bot, w_bot, h_bot = cv2.boundingRect(bottom_c); cx_bottom, cy_bottom = x_bot + w_bot // 2, y_bot + h_bot // 2
                if cy_bottom <= cy_top + MIN_VERTICAL_SEPARATION_PX: continue
                bottom_mask = np.zeros_like(red_mask); cv2.drawContours(bottom_mask, [bottom_c], -1, 255, thickness=5)
                bottom_depths = self.latest_depth_image[bottom_mask == 255]; bottom_depths = bottom_depths[bottom_depths > 0]
                if bottom_depths.size == 0: continue
                avg_depth_bottom = np.mean(bottom_depths)
                depth_diff, horizontal_diff = abs(avg_depth_top - avg_depth_bottom), abs(cx_top - cx_bottom)
                if depth_diff > MAX_DEPTH_DIFFERENCE_MM or horizontal_diff >= MAX_HORIZONTAL_DEVIATION_PX: continue
                if horizontal_diff < best_match_score: best_match_score, best_match, best_match_idx = horizontal_diff, (bottom_c, avg_depth_bottom), i
            if best_match is not None:
                matched_bottom_c, avg_depth_bottom = best_match; x_bot_match, y_bot_match, w_bot_match, h_bot_match = cv2.boundingRect(matched_bottom_c)
                x_full, y_full, w_full, h_full = min(x_top, x_bot_match), y_top, max(x_top + w_top, x_bot_match + w_bot_match) - min(x_top, x_bot_match), (y_bot_match + h_bot_match) - y_top
                avg_depth_cylinder = (avg_depth_top + avg_depth_bottom) / 2.0
                detected_cylinders.append({'rect': (x_full, y_full, w_full, h_full), 'depth': avg_depth_cylinder, 'timestamp': msg.header.stamp})

        for cylinder in detected_cylinders:
            Z = cylinder['depth'] / 1000.0
            if not (MIN_DETECTION_RANGE_M <= Z <= MAX_DETECTION_RANGE_M): continue

            # THE FINAL FIX: Smartly look up the transform
            try:
                # First, try to get the transform at the exact time the image was taken
                transform_time = cylinder['timestamp']
                transform = self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, transform_time, timeout=Duration(seconds=0.05))
            except tf2_ros.ExtrapolationException as e:
                # If that fails, it's likely a clock mismatch. Fall back to getting the latest transform.
                self.get_logger().warn(f'Exact timestamp lookup failed: {e}. Falling back to latest transform.', throttle_duration_sec=5.0)
                try:
                    transform_time = rclpy.time.Time()
                    transform = self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, transform_time, timeout=Duration(seconds=0.1))
                except Exception as e_inner:
                    self.get_logger().error(f'Could not get latest transform either: {e_inner}'); continue
            except Exception as e:
                self.get_logger().error(f'Failed to get any transform: {e}'); continue
            
            # If we successfully got a transform, proceed...
            x, y, w_box, h_box = cylinder['rect']
            fx, fy, cx, cy = self.camera_intrinsics.k[0], self.camera_intrinsics.k[4], self.camera_intrinsics.k[2], self.camera_intrinsics.k[5]
            u, v = x + w_box // 2, y + h_box // 2
            X, Y = (u - cx) * Z / fx, (v - cy) * Z / fy
            point_in_camera = PointStamped(); point_in_camera.header.stamp = cylinder['timestamp']; point_in_camera.header.frame_id = self.source_frame
            point_in_camera.point.x, point_in_camera.point.y, point_in_camera.point.z = X, Y, Z
            
            newly_detected_point = tf2_geometry_msgs.do_transform_point(point_in_camera, transform).point

            # ... (Data association and marker publishing logic is the same) ...
            matched_known_cylinder = None
            for known_cylinder in self.known_cylinders:
                dx, dy = newly_detected_point.x - known_cylinder['position'].x, newly_detected_point.y - known_cylinder['position'].y
                if math.sqrt(dx*dx + dy*dy) < ASSOCIATION_THRESHOLD_METERS:
                    matched_known_cylinder = known_cylinder; break
            if matched_known_cylinder:
                p = matched_known_cylinder['position']
                p.x, p.y, p.z = p.x * 0.95 + newly_detected_point.x * 0.05, p.y * 0.95 + newly_detected_point.y * 0.05, p.z * 0.95 + newly_detected_point.z * 0.05
                self.publish_permanent_marker(p, matched_known_cylinder['id'])
            else:
                elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                if elapsed_time > INITIALIZATION_DELAY_SEC:
                    self.get_logger().info(f"*** DISCOVERED A NEW CYLINDER! Total count: {self.next_marker_id + 1} ***")
                    new_cylinder_data = {'id': self.next_marker_id, 'position': newly_detected_point}
                    self.known_cylinders.append(new_cylinder_data)
                    self.publish_permanent_marker(new_cylinder_data['position'], new_cylinder_data['id'])
                    self.next_marker_id += 1
                else:
                    self.get_logger().warn("Ignoring new cylinder during initial stabilization period.", throttle_duration_sec=1.0)

    def publish_permanent_marker(self, position: Point, marker_id: int):
        # ... (This function is the same) ...
        marker = Marker(); marker.header.frame_id = self.target_frame; marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "permanent_cylinders"; marker.id = marker_id; marker.type = Marker.CYLINDER; marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = position.x, position.y, position.z
        marker.pose.orientation.w = 1.0; marker.scale.x, marker.scale.y, marker.scale.z = 0.15, 0.15, 0.30
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.3, 1.0, 0.9
        marker.lifetime = Duration(seconds=300).to_msg(); self.marker_publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args); detector_node = HorizontalCylinderDetector()
    try: rclpy.spin(detector_node)
    except KeyboardInterrupt: pass
    finally: detector_node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
