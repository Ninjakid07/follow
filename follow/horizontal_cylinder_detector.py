# horizontal_cylinder_detector.py (Contour Splitting with Convexity Defects)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# TF2 imports
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

# =====================================================================================
# --- TUNING PARAMETERS ---
# =====================================================================================
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([165, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

# --- Blob Filtering ---
MIN_BLOB_AREA = 2000 # A contour must have at least this many pixels to be a cylinder.

# --- Depth Sampling ---
DEPTH_SAMPLE_PERCENT = 0.5 

# --- Convexity Defect / Splitting Logic ---
# The "valley" between cylinders must be at least this deep (in pixels) to be considered a valid split point.
MIN_DEFECT_DEPTH = 20 
# The angle at the "valley" point must be sharp (less than 90 degrees) to be a split point.
MAX_DEFECT_ANGLE_DEG = 90
# =====================================================================================

class HorizontalCylinderDetector(Node):
    def __init__(self):
        super().__init__('horizontal_cylinder_detector')
        self.bridge = CvBridge()
        self.latest_depth_image = None
        
        # Camera intrinsics storage
        self.camera_intrinsics = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Frame names
        self.camera_frame = 'oak_camera_rgb_camera_optical_frame'
        self.map_frame = 'map'

        # Subscribers
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
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/oak/rgb/camera_info',
            self.camera_info_callback,
            10)
        
        # Publishers
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)
        
        # NEW: Publisher for tree positions in map frame
        self.tree_pos_pub = self.create_publisher(PointStamped, 'tree_pos', 10)
        
        self.get_logger().info('Cylinder Detector (Contour Splitting) has started.')
        self.get_logger().info('Waiting for camera intrinsics...')
        self.get_logger().info(f'Will transform from {self.camera_frame} to {self.map_frame}')

    def camera_info_callback(self, msg):
        """Extract camera intrinsics from CameraInfo message"""
        if self.camera_intrinsics is None:
            # Extract K matrix (3x3 intrinsic matrix stored as 1D array)
            # K = [fx  0  cx]
            #     [ 0  fy cy]
            #     [ 0  0  1 ]
            K = np.array(msg.k).reshape(3, 3)
            
            self.fx = K[0, 0]  # Focal length in x
            self.fy = K[1, 1]  # Focal length in y
            self.cx = K[0, 2]  # Principal point x
            self.cy = K[1, 2]  # Principal point y
            
            self.camera_intrinsics = K
            
            self.get_logger().info(f'Camera intrinsics received:')
            self.get_logger().info(f'  fx: {self.fx:.2f}, fy: {self.fy:.2f}')
            self.get_logger().info(f'  cx: {self.cx:.2f}, cy: {self.cy:.2f}')

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    def unproject_pixel_to_3d(self, u, v, depth_meters):
        """
        Convert 2D pixel coordinates to 3D camera coordinates
        
        Args:
            u: pixel x-coordinate
            v: pixel y-coordinate
            depth_meters: depth value in meters
            
        Returns:
            (X, Y, Z) in camera frame (meters)
        """
        # Apply the unprojection formula
        X = (u - self.cx) / self.fx * depth_meters
        Y = (v - self.cy) / self.fy * depth_meters
        Z = depth_meters
        
        return X, Y, Z
    
    def transform_point_to_map(self, x_cam, y_cam, z_cam, timestamp):
        """
        Transform a 3D point from camera frame to map frame
        
        Args:
            x_cam, y_cam, z_cam: 3D coordinates in camera frame
            timestamp: ROS timestamp for the transformation
            
        Returns:
            PointStamped in map frame, or None if transformation fails
        """
        # Create PointStamped in camera frame
        point_cam = PointStamped()
        point_cam.header.frame_id = self.camera_frame
        point_cam.header.stamp = timestamp
        point_cam.point.x = x_cam
        point_cam.point.y = y_cam
        point_cam.point.z = z_cam
        
        try:
            # Transform to map frame
            # Using a small timeout to get the latest available transform
            point_map = self.tf_buffer.transform(point_cam, self.map_frame, timeout=rclpy.duration.Duration(seconds=0.1))
            return point_map
            
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform from {self.camera_frame} to {self.map_frame}: {ex}')
            return None

    def image_callback(self, msg):
        # Check if we have both depth image and camera intrinsics
        if self.latest_depth_image is None:
            return
        
        if self.camera_intrinsics is None:
            self.get_logger().warn_once('Camera intrinsics not yet received. Cannot compute 3D coordinates.')
            # Continue with detection but without 3D coordinates
            
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert color image: {e}')
            return
        debug_image = cv_image.copy()
        
        # Get timestamp for TF lookups
        timestamp = msg.header.stamp

        # Step 1: Create a clean mask of the red areas
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv2.inRange(hsv_image, LOWER_RED_2, UPPER_RED_2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))

        # Step 2: Find all the distinct blobs (contours) in the mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_bounding_boxes = []

        for c in contours:
            # Step 2a: Filter out small, noisy blobs
            if cv2.contourArea(c) < MIN_BLOB_AREA:
                continue

            # Step 3: Calculate the Convex Hull and Convexity Defects
            # The hull is the "rubber band" stretched around the blob.
            # Defects are the "valleys" where the blob deviates inwards from the hull.
            hull = cv2.convexHull(c, returnPoints=False)
            if len(hull) > 3 and len(c) > 3: # Need enough points to calculate defects
                defects = cv2.convexityDefects(c, hull)
            else:
                defects = None

            # Visualize the contour and hull for debugging
            cv2.drawContours(debug_image, [c], -1, (0, 0, 255), 2) # Red contour
            hull_points = [c[i][0] for i in hull]
            hull_points = np.array(hull_points, dtype=np.int32)
            cv2.drawContours(debug_image, [hull_points], -1, (255, 0, 0), 2) # Blue hull

            split_points = []
            if defects is not None:
                # Step 4: Analyze the defects to find valid split points
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(c[s][0])
                    end = tuple(c[e][0])
                    far = tuple(c[f][0]) # This is the deepest point of the "valley"

                    # Calculate the angle of the defect
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c_side = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = math.acos((b**2 + c_side**2 - a**2) / (2 * b * c_side))

                    # If the valley is deep enough and the angle is sharp enough, it's a split point
                    if d > MIN_DEFECT_DEPTH * 256 and angle < math.radians(MAX_DEFECT_ANGLE_DEG):
                        split_points.append(far)
                        cv2.circle(debug_image, far, 5, [0, 255, 0], -1) # Green circle on split point

            # Step 5: Split the main bounding box based on the found split points
            x, y, w, h = cv2.boundingRect(c)
            
            # Get the x-coordinates of the blob's left edge, right edge, and all split points
            boundary_xs = [x] + [pt[0] for pt in split_points] + [x + w]
            boundary_xs = sorted(list(set(boundary_xs))) # Sort and remove duplicates

            # Create new bounding boxes between each boundary point
            for i in range(len(boundary_xs) - 1):
                x1 = boundary_xs[i]
                x2 = boundary_xs[i+1]
                # A new bounding box is the full height of the original blob,
                # but sliced vertically at the split points.
                new_box = (x1, y, x2 - x1, h)
                final_bounding_boxes.append(new_box)
        
        self.get_logger().info(f"Detected {len(final_bounding_boxes)} cylinders.")

        # Step 6: Process the final list of bounding boxes (calculate depth, visualize, and 3D coordinates)
        for i, (x, y, w, h) in enumerate(final_bounding_boxes):
            # Define a central sampling area for robust depth
            sample_w = int(w * DEPTH_SAMPLE_PERCENT)
            sample_h = int(h * DEPTH_SAMPLE_PERCENT)
            sample_x = x + (w - sample_w) // 2
            sample_y = y + (h - sample_h) // 2
            
            # Get all depth pixels from the sampling area
            depth_roi = self.latest_depth_image[sample_y : sample_y + sample_h, sample_x : sample_x + sample_w]
            valid_depths = depth_roi[depth_roi > 0]

            if valid_depths.size == 0:
                depth_str = "Depth: N/A"
                self.get_logger().warn(f"Cylinder {i+1}: No valid depth data")
            else:
                median_depth_mm = np.median(valid_depths)
                depth_meters = median_depth_mm / 1000.0
                depth_str = f"Depth: {depth_meters:.2f}m"
                
                # Calculate 3D coordinates if we have camera intrinsics
                if self.camera_intrinsics is not None:
                    # Use the center of the sampling region as the measurement point
                    u = sample_x + sample_w // 2
                    v = sample_y + sample_h // 2
                    
                    # Unproject to 3D in camera frame
                    X_cam, Y_cam, Z_cam = self.unproject_pixel_to_3d(u, v, depth_meters)
                    
                    # Log the camera frame coordinates
                    self.get_logger().info(
                        f"Cylinder {i+1} in camera frame: "
                        f"X={X_cam:.3f}m, Y={Y_cam:.3f}m, Z={Z_cam:.3f}m"
                    )
                    
                    # Transform to map frame and publish
                    point_map = self.transform_point_to_map(X_cam, Y_cam, Z_cam, timestamp)
                    
                    if point_map is not None:
                        # Publish to tree_pos topic
                        self.tree_pos_pub.publish(point_map)
                        
                        # Log the map frame coordinates
                        self.get_logger().info(
                            f"Cylinder {i+1} in map frame: "
                            f"X={point_map.point.x:.3f}m, Y={point_map.point.y:.3f}m, Z={point_map.point.z:.3f}m"
                        )
                        
                        # Add map frame info to visualization
                        map_coord_str = f"Map: ({point_map.point.x:.2f}, {point_map.point.y:.2f}, {point_map.point.z:.2f})m"
                        cv2.putText(debug_image, map_coord_str, (x, y + h + 75), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
                    
                    # Add camera frame 3D info to the visualization
                    cam_coord_str = f"Cam: ({X_cam:.2f}, {Y_cam:.2f}, {Z_cam:.2f})m"
                    cv2.putText(debug_image, cam_coord_str, (x, y + h + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

            # Draw the final bounding box and label
            label = f"Cylinder {i+1}"
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            cv2.putText(debug_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            cv2.putText(debug_image, depth_str, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        
        # Publish the debug image
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))
        
        # Only show window if display is available
        try:
            cv2.imshow("Detection Result", debug_image)
            cv2.waitKey(1)
        except cv2.error as e:
            # Silently ignore if no display is available
            pass


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
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == '__main__':
    main()
