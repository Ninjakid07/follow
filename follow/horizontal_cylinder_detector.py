# horizontal_cylinder_detector.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# =====================================================================================
# --- TUNING PARAMETERS ---
# You will need to adjust these values for your specific environment and camera.
# =====================================================================================

# --- 1. Color Tuning ---
# HSV Color range for RED. This is the MOST IMPORTANT setting.
# Use a color picker tool to find the exact HSV range for the red on your cylinder
# under the robot's lighting conditions.
# Red wraps around 180, so we check two ranges.
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([165, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

# --- 2. Stripe Shape Filtering ---
# Filters out contours that are not shaped like the horizontal stripes.
MIN_STRIPE_AREA = 400         # Minimum pixel area to be considered a potential stripe. Filters out small red noise.
MIN_ASPECT_RATIO = 2.0        # A stripe must be at least 2.0 times wider than it is tall.
                              # This is key for identifying the horizontal bands.

# --- 3. Cylinder Grouping Logic ---
# Defines what constitutes a valid cylinder from a stack of stripes.
MAX_HORIZONTAL_DEVIATION_PX = 50 # Max horizontal pixel distance between the centers of stacked stripes.
                                 # Allows for some wiggle room.
MAX_VERTICAL_GAP_RATIO = 2.0     # The gap between stripes can't be larger than 2.0x the height of a stripe.
                                 # This allows for the white stripes between the red ones.
MIN_STRIPES_IN_GROUP = 3         # A valid cylinder must have at least 3 red stripes.
                                 # Your image shows 6, so 3 is a robust minimum for partial views.

# =====================================================================================

class HorizontalCylinderDetector(Node):
    def __init__(self):
        super().__init__('horizontal_cylinder_detector')
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)

        self.get_logger().info('Horizontal Cylinder Detector has started.')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Step 1: Isolate the red color
        mask1 = cv2.inRange(hsv_image, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv2.inRange(hsv_image, LOWER_RED_2, UPPER_RED_2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        # Step 2: Find all contours of the red areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 3: Filter contours to find only valid "stripes"
        valid_stripes = []
        for c in contours:
            if cv2.contourArea(c) < MIN_STRIPE_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)
            # Avoid division by zero
            if h == 0: continue
            
            aspect_ratio = w / float(h)

            if aspect_ratio > MIN_ASPECT_RATIO:
                # Store stripe info for easier processing
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00']) if M['m00'] != 0 else x + w/2
                valid_stripes.append({'contour': c, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': cx})
        
        # Sort stripes from top to bottom to make grouping easier
        valid_stripes.sort(key=lambda s: s['y'])

        # Step 4: Group stripes into cylinders
        detected_cylinders = []
        while len(valid_stripes) > 0:
            # Start a new potential cylinder group with the top-most remaining stripe
            base_stripe = valid_stripes.pop(0)
            current_group = [base_stripe]
            
            # This list will hold stripes that don't belong to the current group
            other_stripes = []
            
            for stripe in valid_stripes:
                # Check for horizontal alignment and vertical stacking relative to the base
                is_aligned = abs(stripe['cx'] - base_stripe['cx']) < MAX_HORIZONTAL_DEVIATION_PX
                
                # Check if the gap between the last stripe in the group and the new one is reasonable
                last_in_group = current_group[-1]
                gap = stripe['y'] - (last_in_group['y'] + last_in_group['h'])
                is_stacked = 0 < gap < (MAX_VERTICAL_GAP_RATIO * last_in_group['h'])

                if is_aligned and is_stacked:
                    current_group.append(stripe)
                else:
                    other_stripes.append(stripe)
            
            valid_stripes = other_stripes

            if len(current_group) >= MIN_STRIPES_IN_GROUP:
                detected_cylinders.append(current_group)
        
        # Step 5: Log, count, and visualize the results
        cylinder_count = len(detected_cylinders)
        self.get_logger().info(f'Detected {cylinder_count} cylinders.')

        debug_image = cv_image.copy()
        for i, cylinder_group in enumerate(detected_cylinders):
            all_points = np.vstack([s['contour'] for s in cylinder_group])
            x, y, w, h = cv2.boundingRect(all_points)
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (36, 255, 12), 3) # Green box
            label = f"Cylinder {i+1}"
            cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Publish and display debug images
        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, 'bgr8'))
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')

        cv2.imshow("Detection Result", debug_image)
        cv2.imshow("Red Mask", mask)
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
