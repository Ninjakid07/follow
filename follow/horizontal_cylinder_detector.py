# horizontal_cylinder_detector.py (FINAL - With Transform Wait)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs

# =====================================================================================
# --- TUNING PARAMETERS (Unchanged) ---
LOWER_RED_1 = np.array([0, 120, 70]); UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([165, 120, 70]); UPPER_RED_2 = np.array([180, 255, 255])
MIN_EDGE_CONTOUR_LENGTH = 30; MAX_HORIZONTAL_DEVIATION_PX = 40
MAX_DEPTH_DIFFERENCE_MM = 150; MIN_VERTICAL_SEPARATION_PX = 50
# =====================================================================================

class CylinderDetector(Node):
    def __init__(self):
        super().__init__('cylinder_detector')
        # ... (all __init__ contents are the same as before) ...
        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.camera_intrinsics = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.image_sub = self.create_subscription(
            CompressedImage, '/oak/rgb/image_rect/compressed', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/oak/stereo/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/oak/rgb/camera_info', self.info_callback, 10)
        self.tree_pos_pub = self.create_publisher(PointStamped, '/tree_pos', 10)
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)
        self.get_logger().info('Cylinder Detector (Publish Nearest) has started.')

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info('Camera intrinsics received.')
            self.destroy_subscription(self.info_sub)

    def depth_callback(self, msg):
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def image_callback(self, msg):
        if self.latest_depth_image is None or self.camera_intrinsics is None: return

        # --- Detection logic (unchanged) ---
        # ... (The entire, long block of detection code is identical) ...
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        if cv_image.shape[:2] != self.latest_depth_image.shape[:2]: return
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, LOWER_RED_1, UPPER_RED_1); mask2 = cv2.inRange(hsv_image, LOWER_RED_2, UPPER_RED_2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)); red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        h, w = red_mask.shape
        top_edge_image = np.zeros_like(red_mask); bottom_edge_image = np.zeros_like(red_mask)
        cols_with_red = np.where(red_mask.max(axis=0) > 0)[0]
        if cols_with_red.size == 0: self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')); return
        top_indices = np.argmax(red_mask, axis=0)[cols_with_red]; bottom_indices = h - 1 - np.argmax(np.flipud(red_mask), axis=0)[cols_with_red]
        top_edge_image[top_indices, cols_with_red] = 255; bottom_edge_image[bottom_indices, cols_with_red] = 255
        top_contours, _ = cv2.findContours(top_edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bottom_contours, _ = cv2.findContours(bottom_edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_edges = [c for c in top_contours if cv2.arcLength(c, False) > MIN_EDGE_CONTOUR_LENGTH]
        bottom_edges = [c for c in bottom_contours if cv2.arcLength(c, False) > MIN_EDGE_CONTOUR_LENGTH]
        detected_cylinders = []; available_bottom_edges = list(bottom_edges)
        for top_c in top_edges:
            x_top, y_top, w_top, h_top = cv2.boundingRect(top_c)
            cx_top = x_top + w_top // 2; cy_top = y_top + h_top // 2
            top_mask = np.zeros_like(red_mask); cv2.drawContours(top_mask, [top_c], -1, 255, thickness=5)
            top_depths = self.latest_depth_image[top_mask == 255]; top_depths = top_depths[top_depths > 0]
            if top_depths.size == 0: continue
            avg_depth_top = np.mean(top_depths)
            best_match = None; best_match_idx = -1; best_match_score = float('inf')
            for i, bottom_c in enumerate(available_bottom_edges):
                x_bot, y_bot, w_bot, h_bot = cv2.boundingRect(bottom_c)
                cx_bottom = x_bot + w_bot // 2; cy_bottom = y_bot + h_bot // 2
                if cy_bottom <= cy_top + MIN_VERTICAL_SEPARATION_PX: continue
                bottom_mask = np.zeros_like(red_mask); cv2.drawContours(bottom_mask, [bottom_c], -1, 255, thickness=5)
                bottom_depths = self.latest_depth_image[bottom_mask == 255]; bottom_depths = bottom_depths[bottom_depths > 0]
                if bottom_depths.size == 0: continue
                avg_depth_bottom = np.mean(bottom_depths)
                if abs(avg_depth_top - avg_depth_bottom) > MAX_DEPTH_DIFFERENCE_MM: continue
                horizontal_diff = abs(cx_top - cx_bottom)
                if horizontal_diff >= MAX_HORIZONTAL_DEVIATION_PX: continue
                if horizontal_diff < best_match_score:
                    best_match_score = horizontal_diff; best_match = (bottom_c, avg_depth_bottom); best_match_idx = i
            if best_match is not None:
                matched_bottom_c, avg_depth_bottom = best_match
                x_bot_match, y_bot_match, w_bot_match, h_bot_match = cv2.boundingRect(matched_bottom_c)
                x_full = min(x_top, x_bot_match); y_full = y_top
                w_full = max(x_top + w_top, x_bot_match + w_bot_match) - x_full
                h_full = (y_bot_match + h_bot_match) - y_top
                avg_depth_cylinder = (avg_depth_top + avg_depth_bottom) / 2.0
                detected_cylinders.append({'rect': (x_full, y_full, w_full, h_full), 'depth': avg_depth_cylinder})
                available_bottom_edges.pop(best_match_idx)
        
        if not detected_cylinders: return
        nearest_cylinder = min(detected_cylinders, key=lambda c: c['depth'])

        # --- 3D Calculation and Publishing (with THE FIX) ---
        target_frame = 'map'
        source_frame = 'oak_rgb_camera_optical_frame'
        
        # --- NEW: Wait for the transform to be available ---
        # This check prevents the warning by making sure the transform exists before we ask for it.
        try:
            when = rclpy.time.Time()
            if not self.tf_buffer.can_transform(target_frame, source_frame, when, timeout=rclpy.duration.Duration(seconds=0.1)):
                self.get_logger().warn(f"Waiting for transform from '{source_frame}' to '{target_frame}'...", throttle_duration_sec=1.0)
                return
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"can_transform failed: {e}")
            return
        
        # Now we know the transform exists, so we can safely calculate and publish
        rect = nearest_cylinder['rect']; depth_mm = nearest_cylinder['depth']
        fx = self.camera_intrinsics.k[0]; fy = self.camera_intrinsics.k[4]
        cx = self.camera_intrinsics.k[2]; cy = self.camera_intrinsics.k[5]
        pixel_x = rect[0] + rect[2] / 2; pixel_y = rect[1] + rect[3] / 2
        depth_m = depth_mm / 1000.0
        x_cam = (pixel_x - cx) * depth_m / fx
        y_cam = (pixel_y - cy) * depth_m / fy
        z_cam = depth_m

        point_in_camera_frame = PointStamped()
        point_in_camera_frame.header.stamp = msg.header.stamp
        point_in_camera_frame.header.frame_id = source_frame
        point_in_camera_frame.point = Point(x=x_cam, y=y_cam, z=z_cam)

        try:
            point_in_map_frame = self.tf_buffer.transform(point_in_camera_frame, target_frame)
            self.tree_pos_pub.publish(point_in_map_frame)
            self.get_logger().info(f'Published nearest cylinder at map coordinates: x={point_in_map_frame.point.x:.2f}, y={point_in_map_frame.point.y:.2f}')
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"Could not transform point: {e}")
        
        # --- Visualization ---
        debug_image = cv_image.copy()
        x, y, w, h = nearest_cylinder['rect']
        avg_depth_m = nearest_cylinder['depth'] / 1000.0
        label = f"Nearest @ {avg_depth_m:.2f}m"
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (36, 255, 12), 3)
        cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))
        cv2.imshow("Detection Result", debug_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    detector_node = CylinderDetector()
    try: rclpy.spin(detector_node)
    except KeyboardInterrupt: pass
    finally:
        detector_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
