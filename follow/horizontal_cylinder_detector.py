import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# =====================================================================================
# --- TUNING PARAMETERS ---
# (These are unchanged as per your request)
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
MIN_DEFECT_DEPTH = 20 
MAX_DEFECT_ANGLE_DEG = 90
# =====================================================================================

class HorizontalCylinderDetector(Node):
    def __init__(self):
        super().__init__('horizontal_cylinder_detector')
        self.bridge = CvBridge()
        self.latest_depth_image = None
        
        # --- NEW: Variables for Camera Intrinsics ---
        self.camera_intrinsics = None
        self.intrinsics_received = False
        
        # --- MODIFIED: Subscriptions now include CameraInfo ---
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
            
        # --- NEW: Subscription for Camera Intrinsics ---
        # This subscribes to the camera's intrinsic parameters
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/oak/rgb/camera_info',
            self.info_callback,
            10)
        
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)
        
        self.get_logger().info('Cylinder Detector (Contour Splitting) has started.')
        self.get_logger().info('Waiting for camera intrinsics...')

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    # --- NEW: Callback for Camera Intrinsics ---
    def info_callback(self, msg):
        """
        Receives the camera's intrinsic parameters and stores them.
        This is typically a one-time operation.
        """
        if not self.intrinsics_received:
            self.camera_intrinsics = msg
            self.intrinsics_received = True
            # We can destroy the subscription now because intrinsics are static
            self.destroy_subscription(self.info_sub)
            self.get_logger().info('Camera intrinsics received and stored.')

    def image_callback(self, msg):
        # --- MODIFIED: Guard clause to ensure all data is present ---
        # Before processing, we need the depth image AND the camera intrinsics.
        if self.latest_depth_image is None or not self.intrinsics_received:
            return

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert color image: {e}')
            return
        debug_image = cv_image.copy()

        # Step 1 & 2: Masking and Contour finding (Unchanged)
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv2.inRange(hsv_image, LOWER_RED_2, UPPER_RED_2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_bounding_boxes = []

        # Steps 3, 4, 5: Hull, Defect, and Splitting Logic (Unchanged)
        for c in contours:
            if cv2.contourArea(c) < MIN_BLOB_AREA:
                continue
            hull = cv2.convexHull(c, returnPoints=False)
            if len(hull) > 3 and len(c) > 3:
                defects = cv2.convexityDefects(c, hull)
            else:
                defects = None
            
            # (Debug visualization for hull is removed for clarity, but logic is the same)
            # cv2.drawContours(debug_image, [c], -1, (0, 0, 255), 2)
            # hull_points = [c[i][0] for i in hull]
            # hull_points = np.array(hull_points, dtype=np.int32)
            # cv2.drawContours(debug_image, [hull_points], -1, (255, 0, 0), 2)
            
            split_points = []
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(c[s][0])
                    end = tuple(c[e][0])
                    far = tuple(c[f][0])
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c_side = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = math.acos((b**2 + c_side**2 - a**2) / (2 * b * c_side))
                    if d > MIN_DEFECT_DEPTH * 256 and angle < math.radians(MAX_DEFECT_ANGLE_DEG):
                        split_points.append(far)
            
            x, y, w, h = cv2.boundingRect(c)
            boundary_xs = [x] + [pt[0] for pt in split_points] + [x + w]
            boundary_xs = sorted(list(set(boundary_xs)))
            for i in range(len(boundary_xs) - 1):
                x1 = boundary_xs[i]
                x2 = boundary_xs[i+1]
                new_box = (x1, y, x2 - x1, h)
                final_bounding_boxes.append(new_box)
        
        # self.get_logger().info(f"Detected {len(final_bounding_boxes)} cylinders.")

        # Step 6: Process final boxes, calculate depth, AND CALCULATE 3D COORDINATES
        for i, (x, y, w, h) in enumerate(final_bounding_boxes):
            sample_w = int(w * DEPTH_SAMPLE_PERCENT)
            sample_h = int(h * DEPTH_SAMPLE_PERCENT)
            sample_x = x + (w - sample_w) // 2
            sample_y = y + (h - sample_h) // 2
            
            depth_roi = self.latest_depth_image[sample_y : sample_y + sample_h, sample_x : sample_x + sample_w]
            valid_depths = depth_roi[depth_roi > 0]

            if valid_depths.size == 0:
                depth_str = "Depth: N/A"
            else:
                median_depth_mm = np.median(valid_depths)
                depth_str = f"Depth: {median_depth_mm/1000.0:.2f}m"

                # --- NEW: 3D Coordinate Calculation Logic ---
                # Unproject the 2D pixel to a 3D point in the camera frame
                
                # 1. Get the Z coordinate (depth) in meters
                Z = median_depth_mm / 1000.0
                
                if Z > 0:
                    # 2. Get the camera intrinsic parameters from the stored message
                    # K is a 3x3 matrix: [fx, 0, cx], [0, fy, cy], [0, 0, 1]
                    fx = self.camera_intrinsics.k[0]
                    fy = self.camera_intrinsics.k[4]
                    cx = self.camera_intrinsics.k[2]
                    cy = self.camera_intrinsics.k[5]

                    # 3. Define the 2D "measurement spot" (u,v) as the center of the sampling box
                    u = sample_x + (sample_w // 2)
                    v = sample_y + (sample_h // 2)

                    # 4. Apply the unprojection formulas
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy

                    # 5. Announce the 3D discovery in the terminal
                    self.get_logger().info(
                        f'Cylinder {i+1} Coordinates (Camera Frame): X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m'
                    )

            # Draw the final bounding box and label (Unchanged)
            label = f"Cylinder {i+1}"
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            cv2.putText(debug_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            cv2.putText(debug_image, depth_str, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))
        cv2.imshow("Detection Result", debug_image)
        cv2.waitKey(1)

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
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
