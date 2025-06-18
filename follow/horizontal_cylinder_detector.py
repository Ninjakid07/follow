import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped  # New: For publishing the 3D position
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# New: Imports for TF2
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs # Important for transforming geometry_msgs

# =====================================================================================
# --- TUNING PARAMETERS (Unchanged) ---
# =====================================================================================
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([165, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])
MIN_BLOB_AREA = 2000
DEPTH_SAMPLE_PERCENT = 0.5
MIN_DEFECT_DEPTH = 20
MAX_DEFECT_ANGLE_DEG = 90
# =====================================================================================

class HorizontalCylinderDetector(Node):
    def __init__(self):
        super().__init__('horizontal_cylinder_detector')
        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.camera_intrinsics = None
        self.intrinsics_received = False
        self.warned_about_intrinsics = False

        # --- TF2 Listener Setup ---
        # The Buffer stores received transforms, and the Listener populates it.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Define Coordinate Frames ---
        # The frame our calculated coordinates are in (from the camera's perspective)
        self.source_frame = 'oak_camera_rgb_camera_optical_frame'
        # The global frame we want to transform into
        self.target_frame = 'map'

        # --- Subscriptions (Unchanged) ---
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/oak/rgb/image_rect/compressed',
            self.image_callback,
            10)

        self.depth_sub = self.create_subscription(
            Image,
            '/oak/stereo/image_raw',
            self.depth_callback,
            10)

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/oak/rgb/camera_info',
            self.info_callback,
            10)

        # --- NEW: Publisher for the transformed position ---
        self.pos_publisher = self.create_publisher(
            PointStamped,
            'tree_pos', # The topic name you requested
            10)

        self.get_logger().info('Cylinder Detector has started.')
        self.get_logger().info('Waiting for camera intrinsics...')

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    def info_callback(self, msg):
        if not self.intrinsics_received:
            self.camera_intrinsics = msg
            self.intrinsics_received = True
            # We can update the source frame if the CameraInfo header provides it
            if msg.header.frame_id:
                self.source_frame = msg.header.frame_id
            self.destroy_subscription(self.info_sub)
            self.get_logger().info(f'Camera intrinsics received. Using source frame: {self.source_frame}')

    def image_callback(self, msg):
        if self.latest_depth_image is None or not self.intrinsics_received:
            if not self.warned_about_intrinsics:
                self.get_logger().warn('Camera intrinsics or depth image not yet received.')
                self.warned_about_intrinsics = True
            return

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert color image: {e}')
            return

        # Core detection logic remains the same, but without creating a debug_image
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
            hull = cv2.convexHull(c, returnPoints=False)
            defects = cv2.convexityDefects(c, hull) if len(hull) > 3 and len(c) > 3 else None
            split_points = []
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start, end, far = tuple(c[s][0]), tuple(c[e][0]), tuple(c[f][0])
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c_side = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    if a == 0 or b == 0 or c_side == 0: continue
                    angle = math.acos((b**2 + c_side**2 - a**2) / (2 * b * c_side))
                    if d > MIN_DEFECT_DEPTH * 256 and angle < math.radians(MAX_DEFECT_ANGLE_DEG):
                        split_points.append(far)
            x, y, w, h = cv2.boundingRect(c)
            boundary_xs = sorted(list(set([x] + [pt[0] for pt in split_points] + [x + w])))
            for i in range(len(boundary_xs) - 1):
                final_bounding_boxes.append((boundary_xs[i], y, boundary_xs[i+1] - boundary_xs[i], h))

        # --- Process, Transform, and Publish ---
        for i, (x, y, w, h) in enumerate(final_bounding_boxes):
            sample_w, sample_h = int(w * DEPTH_SAMPLE_PERCENT), int(h * DEPTH_SAMPLE_PERCENT)
            sample_x, sample_y = x + (w - sample_w) // 2, y + (h - sample_h) // 2
            
            depth_roi = self.latest_depth_image[sample_y : sample_y + sample_h, sample_x : sample_x + sample_w]
            valid_depths = depth_roi[depth_roi > 0]

            if valid_depths.size > 0:
                median_depth_mm = np.median(valid_depths)
                Z = median_depth_mm / 1000.0
                
                if Z > 0:
                    fx, fy, cx, cy = self.camera_intrinsics.k[0], self.camera_intrinsics.k[4], self.camera_intrinsics.k[2], self.camera_intrinsics.k[5]
                    u, v = sample_x + (sample_w // 2), sample_y + (sample_h // 2)
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy

                    # At this point, (X, Y, Z) are in the self.source_frame

                    # --- NEW: Create a PointStamped message in the source frame ---
                    point_in_camera_frame = PointStamped()
                    point_in_camera_frame.header.stamp = self.get_clock().now().to_msg()
                    point_in_camera_frame.header.frame_id = self.source_frame
                    point_in_camera_frame.point.x = X
                    point_in_camera_frame.point.y = Y
                    point_in_camera_frame.point.z = Z
                    
                    # Log the original coordinates
                    self.get_logger().info(f'Cylinder {i+1} found at X={X:.2f}, Y={Y:.2f}, Z={Z:.2f} in "{self.source_frame}"')

                    # --- NEW: Transform the point to the target frame ---
                    try:
                        # Use the buffer to transform the point
                        point_in_map_frame = self.tf_buffer.transform(
                            point_in_camera_frame,
                            self.target_frame,
                            timeout=rclpy.duration.Duration(seconds=0.1) # Add a timeout
                        )

                        # --- NEW: Publish the transformed point ---
                        self.pos_publisher.publish(point_in_map_frame)
                        self.get_logger().info(f'--> Published transformed position to /tree_pos in "{self.target_frame}" frame.')

                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        self.get_logger().error(f'Could not transform from "{self.source_frame}" to "{self.target_frame}": {e}')

def main(args=None):
    rclpy.init(args=args)
    detector_node = HorizontalCylinderDetector()
    try:
        rclpy.spin(detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        detector_node.destroy_node()
        rclpy.shutdown()
        # cv2.destroyAllWindows() is no longer needed as we are not showing images

if __name__ == '__main__':
    main()
