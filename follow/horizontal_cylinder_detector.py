import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration

# New: Import the Marker message type for RViz visualization
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

# ... (Tuning Parameters are unchanged) ...
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([165, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])
MIN_BLOB_AREA = 2000
DEPTH_SAMPLE_PERCENT = 0.5
MIN_DEFECT_DEPTH = 20
MAX_DEFECT_ANGLE_DEG = 90

class HorizontalCylinderDetector(Node):
    def __init__(self):
        super().__init__('horizontal_cylinder_detector')

        # Robustly handle sim time parameter
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        sim_time_is_enabled = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        log_msg = 'Node is using simulation time.' if sim_time_is_enabled else 'Node is using system (wall) time.'
        self.get_logger().info(log_msg)

        self.queue_size = 30
        self.bridge = CvBridge()

        # Cached data
        self.latest_depth_image = None
        self.latest_image_msg = None
        self.camera_intrinsics = None
        self.intrinsics_received = False

        # TF setup
        self.get_logger().info('Setting TF buffer duration to 10.0 seconds.')
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.source_frame = 'oak_camera_rgb_camera_optical_frame'
        self.target_frame = 'map'
        self.get_logger().info(f'Source frame: {self.source_frame}, Target frame: {self.target_frame}')

        # Subscribers
        self.image_sub = self.create_subscription(CompressedImage, '/oak/rgb/image_rect/compressed', self.image_callback, self.queue_size)
        self.depth_sub = self.create_subscription(Image, '/oak/stereo/image_raw', self.depth_callback, self.queue_size)
        self.info_sub = self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.info_callback, self.queue_size)
        
        # Publishers
        self.pos_publisher = self.create_publisher(PointStamped, 'tree_pos', self.queue_size)
        # NEW: Publisher for the RViz marker
        self.marker_publisher = self.create_publisher(Marker, '~/detected_cylinders', self.queue_size)

        # CHANGE 1: Increase processing rate to 15 Hz
        self.get_logger().info('Starting 15Hz processing loop.')
        self.processing_timer = self.create_timer(1.0 / 15.0, self.processing_callback)

    def image_callback(self, msg): self.latest_image_msg = msg
    def depth_callback(self, msg): self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def info_callback(self, msg):
        if not self.intrinsics_received:
            self.camera_intrinsics = msg
            self.intrinsics_received = True
            self.destroy_subscription(self.info_sub)
            self.get_logger().info('Camera intrinsics received.')

    def processing_callback(self):
        if self.latest_image_msg is None or self.latest_depth_image is None or not self.intrinsics_received:
            return

        # Core detection logic ... (unchanged)
        try: cv_image = self.bridge.compressed_imgmsg_to_cv2(self.latest_image_msg, desired_encoding='bgr8')
        except Exception as e: self.get_logger().error(f'Failed to convert color image: {e}'); return
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv2.inRange(hsv_image, LOWER_RED_2, UPPER_RED_2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_bounding_boxes = []
        for c in contours:
            if cv2.contourArea(c) < MIN_BLOB_AREA: continue
            x, y, w, h = cv2.boundingRect(c)
            final_bounding_boxes.append((x,y,w,h))

        for i, (x, y, w, h) in enumerate(final_bounding_boxes):
            sample_w = int(w * DEPTH_SAMPLE_PERCENT)
            sample_h = int(h * DEPTH_SAMPLE_PERCENT)
            sample_x = x + (w - sample_w) // 2
            sample_y = y + (h - sample_h) // 2
            
            if sample_y + sample_h > self.latest_depth_image.shape[0] or sample_x + sample_w > self.latest_depth_image.shape[1]: continue
            depth_roi = self.latest_depth_image[sample_y : sample_y + sample_h, sample_x : sample_x + sample_w]
            valid_depths = depth_roi[depth_roi > 0]

            if valid_depths.size > 0:
                Z = np.median(valid_depths) / 1000.0
                if Z <= 0: continue
                fx, fy, cx, cy = self.camera_intrinsics.k[0], self.camera_intrinsics.k[4], self.camera_intrinsics.k[2], self.camera_intrinsics.k[5]
                u, v = sample_x + (sample_w // 2), y + (sample_h // 2)
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

                point_in_camera = PointStamped()
                point_in_camera.header.stamp = self.latest_image_msg.header.stamp
                point_in_camera.header.frame_id = self.source_frame
                point_in_camera.point.x, point_in_camera.point.y, point_in_camera.point.z = X, Y, Z
                
                try:
                    transform = self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, rclpy.time.Time())
                    transformed_point_stamped = tf2_geometry_msgs.do_transform_point(point_in_camera, transform)
                    
                    # Publish the raw coordinate
                    self.pos_publisher.publish(transformed_point_stamped)
                    
                    # NEW: Create and publish the RViz marker
                    self.publish_cylinder_marker(transformed_point_stamped.point, i)
                    
                    self.get_logger().info(f'Published cylinder {i+1} position and marker.', throttle_duration_sec=1.0)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    self.get_logger().warn(f'Transform failed: {e}', throttle_duration_sec=1.0)

    def publish_cylinder_marker(self, position: Point, marker_id: int):
        """Creates and publishes a CYLINDER marker at the given position."""
        marker = Marker()
        marker.header.frame_id = self.target_frame # The marker is in the 'map' frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detected_cylinders" # Namespace for our markers
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Set the pose of the marker. This is a full 6DOF pose relative to the frame_id.
        marker.pose.position.x = position.x
        marker.pose.position.y = position.y
        marker.pose.position.z = position.z # Center the marker vertically
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set the scale of the marker
        marker.scale.x = 0.15 # Diameter in meters
        marker.scale.y = 0.15 # Diameter in meters
        marker.scale.z = 0.30 # Height in meters

        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8 # Make it slightly transparent

        # Set the lifetime. The marker will automatically disappear after this duration.
        marker.lifetime = Duration(seconds=0.5).to_msg()

        self.marker_publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    detector_node = HorizontalCylinderDetector()
    try: rclpy.spin(detector_node)
    except KeyboardInterrupt: pass
    finally:
        detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
