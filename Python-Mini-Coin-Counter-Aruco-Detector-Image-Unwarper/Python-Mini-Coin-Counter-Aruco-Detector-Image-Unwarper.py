import argparse
import cv2
import numpy as np
from cv2 import aruco
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ComputerVision constants #
# Miscellaneous constants #
MISC_IMAGE_PATH_PARAMETER_PATH = "Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper/Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper-images/good/good1.jpg"  # Change this to your desired image path
MISC_USE_IMAGE_PATH_PARAMETER_IMAGE_SELECTION = False  # Set this to True if you want to use the image path parameter
MISC_USE_WEBCAM_SNAPSHOT_IMAGE_SELECTION = False  # Set this to True if you want to use a snapshot from your webcam instead of an image file
MISC_COMMAND_LINE_ARGUMENT_IMAGE_SELECTION = False  # Set this to True if you want to use command line argument image selection

# Argparse constants #
ARGPARSE_IMAGE_PATH_ARG_NAME = "image_path"  # Name of the image path argument
ARGPARSE_IMAGE_PATH_ARG_HELP = "Path to image you would like to process"  # Help text for the image path argument
ARGPARSE_DEFAULT_IMAGE_PATH = "Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper/Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper-images/good/good1.jpg"  # Change this to your desired default image path

# UI constants #
UI_WINDOW_TITLE = "Python-Mini-Coin-Counter-Aruco-Detector-Image-Unwarper - Samuel Bullard"  # Title of the UI window
UI_MAX_IMAGE_WIDTH = 500  # Maximum width of the image in the UI
UI_MAX_IMAGE_HEIGHT = 750  # Maximum height of the image in the UI
UI_RESIZABLE = False  # Whether the UI window is resizable or not
UI_TOPMOST = True  # Whether the UI window is topmost or not

UI_COIN_LABEL_TEXT = "Coins: "  # Text of the coin label
UI_COIN_LABEL_TEXT_COLOR = "black"  # Color of the coin label text
UI_COIN_LABEL_FONT = "Helvetica 16 bold"  # Font of the coin label text
UI_COIN_LABEL_FONT_SIZE = 16  # Font size of the coin label text
UI_COIN_LABEL_COLOR = "white"  # Color of the coin label

UI_DETECT_BUTTON_TEXT = "Detect"  # Text of the detect button
UI_DETECT_BUTTON_TEXT_COLOR = "black"  # Color of the detect button text
UI_DETECT_BUTTON_FONT = "Helvetica 16 bold"  # Font of the detect button text
UI_DETECT_BUTTON_FONT_SIZE = 16  # Font size of the detect button text
UI_DETECT_BUTTON_COLOR = "white"  # Color of the detect button

# Unwarp image constants #
# Preprocessing constants
UNWARP_GAUSSIAN_BLUR_KERNEL_SIZE = 7  # Kernel size for gaussian blur operation
UNWARP_CANNY_LOWER_THRESHOLD = 50  # Lower threshold of the two passed to the Canny edge detector (Values below this are considered non-edges)
UNWARP_CANNY_UPPER_THRESHOLD = 100  # Upper threshold of the two passed to the Canny edge detector (Values above this are considered strong edges)
UNWARP_CANNY_APERTURE_KERNEL_SIZE = 3  # Aperture Kernel size for Canny edge detection
UNWARP_CANNY_L2_GRADIENT = True  # Whether to use the L2 norm for gradient calculation (L2 = Euclidean distance, L1 = Manhattan distance)
UNWARP_ERODE_KERNEL_SIZE = (3, 3)  # Kernel size for erode operation
UNWARP_DILATE_KERNEL_SIZE = (3, 3)  # Kernel size for dilate operation
UNWARP_DILATE_ITERATIONS = 3  # Number of iterations for dilate operation
UNWARP_ERODE_ITERATIONS = 1  # Number of iterations for erode operation

# Paper contour display constants #
# Contour Outline
CONTOUR_FILTER_END_RETAIN_AMOUNT = 5  # Amount of contours to keep
CONTOUR_OUTLINE_COLOR = (0, 0, 255)  # Color of the contour outline
CONTOUR_OUTLINE_THICKNESS = 8  # Width of the contour outline
CONTOUR_HEURISTIC_APPROXIMATION_EPSILON = 0.02  # Maximum distance between the original curve and its approximation

# Coin detection constants #
# Preprocessing constants
COIN_DETECTION_MEDIAN_BLUR_KERNEL_SIZE = 5  # Kernel size for median blur preprocessing
# Adaptive threshold
COIN_DETECTION_ADAPTIVE_THRESHOLD_ENABLED = False  # Enables adaptive thresholding
COIN_DETECTION_ADAPTIVE_THRESHOLD_BLOCK_SIZE = 5  # Size of a pixel neighborhood block that is used to calculate a threshold value for the pixel
COIN_DETECTION_ADAPTIVE_THRESHOLD_C = 4  # Constant subtracted from the mean or weighted mean of the neighborhood pixels
# Erode and dilate
COIN_DETECTION_ERODE_KERNEL_SIZE = (5, 5)  # Kernel size for erode preprocessing
COIN_DETECTION_DILATE_KERNEL_SIZE = (5, 5)  # Kernel size for dilate preprocessing
COIN_DETECTION_ERODE_ITERATIONS = 2  # Number of times erosion is applied
COIN_DETECTION_DILATE_ITERATIONS = 2  # Number of times dilation is applied

# HoughCircles
HOUGH_CIRCLES_MIN_DIST = 30  # Minimum distance between the centers of the detected circles.
HOUGH_CIRCLES_MIN_RADIUS = 10  # Minimum circle radius
HOUGH_CIRCLES_MAX_RADIUS = 40  # Maximum circle radius
HOUGH_CIRCLES_CANNY_UPPER_THRESHOLD_COIN = 80  # Top Threshold of the two passed to the Canny edge detector (Values higher than this are considered strong edges)
# Values between the two thresholds are considered weak edges and are included in the output only if they are connected to strong edges
HOUGH_CIRCLES_CANNY_LOWER_THRESHOLD = 20  # Lower threshold of the two passed to the Canny edge detector (Values smaller than this are considered non-edges
HOUGH_CIRCLES_DP = 1  # Inverse ratio of the accumulator resolution to the image resolution (2 would half the accumulator resolution, effectively downsampling the image for processing)

# Circles display constants
# Circle Outline
CIRCLE_DISPLAY_OUTLINE_COLOR = (0, 255, 0)  # Color of the circle outline
CIRCLE_DISPLAY_OUTLINE_THICKNESS = 2  # Width of the circle outline

# Circle Text
CIRCLE_DISPLAY_TEXT_COLOR = (255, 255, 255)  # Color of the circle text
CIRCLE_DISPLAY_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font of the circle text
CIRCLE_DISPLAY_TEXT_SCALE = 0.5  # Scale of the circle text
CIRCLE_DISPLAY_TEXT_THICKNESS = 1  # Thickness of the circle text
CIRCLE_DISPLAY_TEXT_X_OFFSET = 20  # X offset of the circle text
CIRCLE_DISPLAY_TEXT_Y_OFFSET = 0  # Y offset of the circle text

# Circle Center Point
CIRCLE_DISPLAY_CENTER_POINT_RADIUS = 5  # Radius of the circle center point
CIRCLE_DISPLAY_CENTER_POINT_LINE_COLOR = (0, 255, 0)  # Color of the circle center line
CIRCLE_DISPLAY_CENTER_POINT_LINE_THICKNESS = -1  # Width of the circle center line

# Aruco detection constants #
ARUCO_MARKER_REALWORLD_SIZE_MM = 60  # Real world size of the aruco marker in meters
ARUCO_RECT_COLOR = (0, 255, 0)  # Color of the aruco rectangle
ARUCO_RECT_THICKNESS = 2  # Width of the aruco rectangle
ARUCO_CIRCLE_COLOR = (0, 0, 255)  # Color of the aruco circle
ARUCO_CIRCLE_RADIUS = 30  # Radius of the aruco circle
ARUCO_CIRCLE_THICKNESS = 2  # Width of the aruco circle

# Coin value detection constants #
# Keys in the dict are the coins diameter in mm
COIN_VALUE_DETECTION_COIN_VALUE_DICT = {25.75: "2€",
                                        23.25: "1€",
                                        24.25: "50 cent",
                                        22.25: "20 cent",
                                        19.75: "10 cent",
                                        21.25: "5 cent",
                                        18.75: "2 cent",
                                        16.25: "1 cent"}


# Class definition
class ComputerVision:
    # Constructor
    def __init__(self, image_path=None, use_image_path_parameter_image_selection=False,
                 use_webcam_snapshot_image_selection=False, use_command_line_image_selection=False):
        # Loads image from parameter path
        if image_path and use_image_path_parameter_image_selection:
            self.image = self.load_image_from_path(image_path)
        # Loads image from command line argument path
        elif use_command_line_image_selection:
            self.image = self.load_image_from_path(self.parse_commandline_arguments().image_path)
        # Loads image from webcam snapshot
        elif use_webcam_snapshot_image_selection:
            self.image = self.capture_webcam_snapshot()
        # Loads image from file explorer selection
        else:
            self.image = self.select_image_via_file_explorer()

        # Copies the original image to a new variable, which is displayed in the right column of the UI
        self.coin_counter_label = None
        self.processed_image = self.image.copy()
        self.root = None

    def detect(self):
        processed_image = self.return_original_image()  # Retrieves the original image to process
        processed_image = self.full_pil_to_np_cv2(processed_image)  # Makes the image compatible with OpenCV

        processed_image = self.unwarp_using_contours(processed_image)   # Unwarps the image
        processed_image, coin_diameters = self.detect_coins(processed_image)    # Detects the coins in the image
        processed_image, aruco_marker_top_corner_coords = self.detect_aruco_markers_custom_outline_top_corner_return(
            processed_image, ARUCO_RECT_COLOR,
            ARUCO_RECT_THICKNESS,
            ARUCO_CIRCLE_COLOR, ARUCO_CIRCLE_RADIUS,
            ARUCO_CIRCLE_THICKNESS) # Detects the aruco marker in the image

        pixel_size_in_mm = self.get_pixel_size_in_mm(aruco_marker_top_corner_coords) # Calculates the pixel size in mm
        coin_values = self.detect_coin_values(COIN_VALUE_DETECTION_COIN_VALUE_DICT, coin_diameters, pixel_size_in_mm)
        print(coin_values)  # Detects the coin values

        processed_image = self.full_np_cv2_to_pil(processed_image)  # Makes the image compatible with PIL again
        processed_image = self.resize_pil_image(processed_image, UI_MAX_IMAGE_WIDTH) # Resizes the image to fit the UI
        self.update_processed_image_ui(processed_image)     # Updates the UI with the processed image

    # Unwarps the image using contour detection
    def unwarp_using_contours(self, image):
        processed_output_image = image.copy()
        greyscale_np_cv2_image = self.np_bgr_image_to_np_greyscale(image)
        # Applies a Gaussian blur to the image
        greyscale_np_cv2_image = cv2.GaussianBlur(greyscale_np_cv2_image,
                                                  (UNWARP_GAUSSIAN_BLUR_KERNEL_SIZE, UNWARP_GAUSSIAN_BLUR_KERNEL_SIZE),
                                                  0)

        # Applies Canny edge detection to the image
        greyscale_np_cv2_image = cv2.Canny(greyscale_np_cv2_image, UNWARP_CANNY_LOWER_THRESHOLD,
                                           UNWARP_CANNY_UPPER_THRESHOLD,
                                           apertureSize=UNWARP_CANNY_APERTURE_KERNEL_SIZE,
                                           L2gradient=UNWARP_CANNY_L2_GRADIENT)

        # Dilates and erodes the image
        greyscale_np_cv2_image = cv2.dilate(greyscale_np_cv2_image, UNWARP_DILATE_KERNEL_SIZE,
                                            iterations=UNWARP_DILATE_ITERATIONS)
        greyscale_np_cv2_image = cv2.erode(greyscale_np_cv2_image, UNWARP_ERODE_KERNEL_SIZE,
                                           iterations=UNWARP_ERODE_ITERATIONS)

        # Finds the contours in the image
        contours, _ = cv2.findContours(greyscale_np_cv2_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sorts the contours by area in descending order (largest to smallest)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Keeps only the 5 largest contours
        contours = contours[0:CONTOUR_FILTER_END_RETAIN_AMOUNT]
        # Draws the contours on the image

        # Showcase of the exact contours
        image_copy = image.copy()
        cv2.drawContours(image_copy, contours, -1, CONTOUR_OUTLINE_COLOR, CONTOUR_OUTLINE_THICKNESS)
        self.custom_window_display("Exact Contours", image_copy, UI_MAX_IMAGE_WIDTH)

        # Loops through the contours and finds the one with 4 corners
        corner_count = None
        for c in contours:
            # Approximates the contour (reduces the number of points in the contour to an optimal minimum)
            perimeter = cv2.arcLength(c, True)
            # Douglas-Peucker algorithm
            approx = cv2.approxPolyDP(c, CONTOUR_HEURISTIC_APPROXIMATION_EPSILON * perimeter, True)
            # If 4 corners are detected it's the correct contour for the sheet of paper
            if len(approx) == 4:
                corner_count = approx
                cv2.drawContours(processed_output_image, [approx], -1, CONTOUR_OUTLINE_COLOR, CONTOUR_OUTLINE_THICKNESS)
                break

        if corner_count is None:
            return greyscale_np_cv2_image

        # Rearranges the corners in corner_count in a consistent order comparing the sum and difference of the x and
        # y values
        rect = np.zeros((4, 2), dtype="float32")
        s = corner_count.sum(axis=2)
        rect[0] = corner_count[np.argmin(s)]
        rect[2] = corner_count[np.argmax(s)]
        diff = np.diff(corner_count, axis=2)
        rect[1] = corner_count[np.argmin(diff)]
        rect[3] = corner_count[np.argmax(diff)]

        # Calculates the width and height of the detected rectangle using the Euclidean distance between the corners
        width_a = np.linalg.norm(rect[2] - rect[3])
        width_b = np.linalg.norm(rect[1] - rect[0])
        height_a = np.linalg.norm(rect[0] - rect[3])
        height_b = np.linalg.norm(rect[1] - rect[2])

        # Computes the actual maximum width and height of the rectangle
        max_width = max(int(width_a), int(width_b))
        max_height = max(int(height_a), int(height_b))

        # Defines the points for a perfect rectangle based on the computed dimensions
        # Calculates the scaling factors for width and height
        width_scale = UI_MAX_IMAGE_WIDTH / float(max_width)
        height_scale = UI_MAX_IMAGE_HEIGHT / float(max_height)

        # Uses the minimum scale to ensure the unwarped image fits within the bounds
        scale = min(width_scale, height_scale)
        max_width_scaled = int(max_width * scale)
        max_height_scaled = int(max_height * scale)

        # Defines the points for a perfect rectangle based on the scaled dimensions
        dst_pts = np.array([
            [0, 0],
            [max_width_scaled - 1, 0],
            [max_width_scaled - 1, max_height_scaled - 1],
            [0, max_height_scaled - 1]
        ], dtype="float32")

        # Gets the perspective transformation matrix
        unwarp_transformation_matrix = cv2.getPerspectiveTransform(rect, dst_pts)

        # Applies the unwarping transformation matrix to the image
        unwarped_image = cv2.warpPerspective(processed_output_image, unwarp_transformation_matrix,
                                             (max_width_scaled, max_height_scaled))
        return unwarped_image

    # Detects the coins in the image
    def detect_coins(self, image):
        processed_output_image = image.copy()
        cropped_out_coins = []
        coin_diameters = []

        # Converts the image to greyscale
        greyscale_np_cv2_image = self.np_bgr_image_to_np_greyscale(image)

        # Blurs the image to reduce noise
        greyscale_np_cv2_image = cv2.medianBlur(greyscale_np_cv2_image, COIN_DETECTION_MEDIAN_BLUR_KERNEL_SIZE)

        # Adaptive thresholding
        if COIN_DETECTION_ADAPTIVE_THRESHOLD_ENABLED:
            greyscale_np_cv2_image = cv2.adaptiveThreshold(greyscale_np_cv2_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                           cv2.THRESH_BINARY,
                                                           COIN_DETECTION_ADAPTIVE_THRESHOLD_BLOCK_SIZE,
                                                           COIN_DETECTION_ADAPTIVE_THRESHOLD_C)
            self.custom_window_display("Adaptive Thresholding", greyscale_np_cv2_image, UI_MAX_IMAGE_WIDTH)

        # Erosion and dilation
        greyscale_np_cv2_image = cv2.erode(greyscale_np_cv2_image, COIN_DETECTION_ERODE_KERNEL_SIZE,
                                           iterations=COIN_DETECTION_ERODE_ITERATIONS)
        greyscale_np_cv2_image = cv2.dilate(greyscale_np_cv2_image, COIN_DETECTION_DILATE_KERNEL_SIZE,
                                            iterations=COIN_DETECTION_DILATE_ITERATIONS)

        # Detects circles using HoughCircles Algorithm
        circles = cv2.HoughCircles(greyscale_np_cv2_image, cv2.HOUGH_GRADIENT, HOUGH_CIRCLES_DP,
                                   minDist=HOUGH_CIRCLES_MIN_DIST,
                                   param1=HOUGH_CIRCLES_CANNY_UPPER_THRESHOLD_COIN,
                                   param2=HOUGH_CIRCLES_CANNY_LOWER_THRESHOLD,
                                   minRadius=HOUGH_CIRCLES_MIN_RADIUS, maxRadius=HOUGH_CIRCLES_MAX_RADIUS)

        # Check to see if there is any detection
        if circles is not None:
            # If there are some detections, convert radius and x,y(center) coordinates to integer
            circles = np.round(circles[0, :]).astype("int")
        else:
            print("No coins detected.")
            return processed_output_image

        for (x, y, radius) in circles:
            # Draw the circle
            cv2.circle(processed_output_image, (x, y), radius, CIRCLE_DISPLAY_OUTLINE_COLOR,
                       CIRCLE_DISPLAY_OUTLINE_THICKNESS)
            # Draw the center of the circle
            cv2.circle(processed_output_image, (x, y), CIRCLE_DISPLAY_CENTER_POINT_RADIUS,
                       CIRCLE_DISPLAY_CENTER_POINT_LINE_COLOR,
                       CIRCLE_DISPLAY_CENTER_POINT_LINE_THICKNESS)

            cv2.putText(img=processed_output_image,
                        text=str(radius),
                        org=(x + CIRCLE_DISPLAY_TEXT_X_OFFSET, y + CIRCLE_DISPLAY_TEXT_Y_OFFSET),
                        fontFace=CIRCLE_DISPLAY_TEXT_FONT,
                        fontScale=CIRCLE_DISPLAY_TEXT_SCALE,
                        color=CIRCLE_DISPLAY_TEXT_COLOR,
                        thickness=CIRCLE_DISPLAY_TEXT_THICKNESS)

            coin = image[y - radius:y + radius, x - radius:x + radius]
            cropped_out_coins.append(coin)
            coin_diameters.append(radius * 2)

            # Displays the number of coins detected
            self.coin_counter_label.config(text=UI_COIN_LABEL_TEXT + str(len(circles)))

        return processed_output_image, coin_diameters

    # Detects the aruco markers in the image
    def detect_aruco_markers(self, image):
        # Creates the aruco dictionary
        dic = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

        # Converts the image from BGR to greyscale
        greyscale_np_image = self.np_bgr_image_to_np_greyscale(image)

        # Detects the aruco markers in the image
        corners, marker_ids, rejected_points = aruco.detectMarkers(greyscale_np_image, dic)

        print("Detected marker corners:", corners)
        print("Detected marker IDs:", marker_ids)

        # Draws the detected markers on the image
        cv2.aruco.drawDetectedMarkers(image, corners, marker_ids)

        return image

    # Detects the aruco markers in the image, offers customization, returns the top aruco corners
    def detect_aruco_markers_custom_outline_top_corner_return(self, image, rect_color_rgb, rect_thickness,
                                                              circle_color_rgb, circle_radius,
                                                              circle_thickness):
        # Creates the aruco dictionary
        dic = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

        # Converts the image from BGR to greyscale
        greyscale_np_image = self.np_bgr_image_to_np_greyscale(image)

        # Detects the aruco markers in the image
        corners, marker_ids, rejected_points = aruco.detectMarkers(greyscale_np_image, dic)

        print("Detected marker corners:", corners)
        print("Detected marker IDs:", marker_ids)

        # Draws the detected markers on the image
        cv2.aruco.drawDetectedMarkers(image, corners, marker_ids)

        top_left_corner = (int(corners[0][0][0][0]), int(corners[0][0][0][1]))
        top_right_corner = (int(corners[0][0][1][0]), int(corners[0][0][1][1]))
        bottom_right_corner = (int(corners[0][0][2][0]), int(corners[0][0][2][1]))
        bottom_left_corner = (int(corners[0][0][3][0]), int(corners[0][0][3][1]))

        aruco_marker_top_corner_coords = [top_left_corner, top_right_corner]

        # Draws a rectangle where the marker is located and circles on the top corners of the marker
        cv2.rectangle(image, top_left_corner, bottom_right_corner, rect_color_rgb, rect_thickness)
        cv2.circle(image, top_left_corner, circle_radius, circle_color_rgb, circle_thickness)
        cv2.circle(image, top_right_corner, circle_radius, circle_color_rgb, circle_thickness)

        return image, aruco_marker_top_corner_coords

    # Gets the size of a pixel in the image in mm
    def get_pixel_size_in_mm(self, aruco_marker_corner_coords):
        top_left_corner = aruco_marker_corner_coords[0]
        top_right_corner = aruco_marker_corner_coords[1]

        # Calculate the euclidian distance between the top left and top right corner in px
        dist = np.linalg.norm(np.array(top_left_corner) - np.array(top_right_corner))
        print("Width of Marker in image: {} px".format(dist))

        pixel_size_in_mm = ARUCO_MARKER_REALWORLD_SIZE_MM / dist
        print("=> 1 Pixel in the image equals ~{}mm".format(round(pixel_size_in_mm, 3)))

        return float(format(round(pixel_size_in_mm, 4)))

    # Determines the value of each coin
    def detect_coin_values(self, coin_values_dict, coin_diameters, pixel_size_in_mm):

        coin_values = []
        # Checks the diameter of each coin and returns the value of the coin
        for i, diameter_px in enumerate(coin_diameters):
            diameter_mm = diameter_px * pixel_size_in_mm

            closest_coin_value = min(coin_values_dict.keys(), key=lambda x: abs(x - diameter_mm))
            coin_values.append(coin_values_dict[closest_coin_value])

            print("Coin {} with diameter of {}px ({}mm) is probably {}".format(i + 1,
                                                                               diameter_px,
                                                                               round(diameter_mm, 2),
                                                                               coin_values_dict[closest_coin_value]))
        return coin_values

    # ---------------------- Image Selection Functions ------------------------ #
    # Opens up a webcam feed and captures a single frame
    def capture_webcam_snapshot(self):
        # Opens the default webcam
        cap = cv2.VideoCapture(0)

        # Checks if the webcam is opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None

        # Capture a single frame
        ret, frame = cap.read()

        # Releases the webcam
        cap.release()

        # Checks if the capture was successful
        if not ret:
            print("Error: Could not read frame from webcam.")
            return None

        # Converts the frame from BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Converts the numpy array to a PIL Image to be used in tkinter
        image = Image.fromarray(frame_rgb)

        return image

    # Opens a file explorer to select an image
    def select_image_via_file_explorer(self):
        image_path = filedialog.askopenfilename(title="Select an image to process")
        return Image.open(image_path)

    # Uses argparse to parse the command line arguments for the image path
    def parse_commandline_arguments(self):
        parser = argparse.ArgumentParser(description=UI_WINDOW_TITLE)
        parser.add_argument(ARGPARSE_IMAGE_PATH_ARG_NAME, metavar=ARGPARSE_IMAGE_PATH_ARG_NAME, type=str, nargs="?",
                            default=ARGPARSE_DEFAULT_IMAGE_PATH,
                            help=ARGPARSE_IMAGE_PATH_ARG_HELP)
        return parser.parse_args()

    # ---------------------- Image Modification Functions ------------------------ #
    # Resizes an image to a given width and height while maintaining the aspect ratio
    def resize_pil_image(self, image, base_width):
        w_percent = base_width / float(image.width)
        h_size = int(float(image.height) * float(w_percent))
        return image.resize((base_width, h_size), Image.LANCZOS)

    # Updates the processed image label with the given image
    def update_processed_image_ui(self, image):

        # Converts the PIL Image to a numpy array
        np_array_image = np.asarray(image)

        # Converts the image to BGR
        np_array_image = self.np_rgb_to_bgr(np_array_image)

        # Converts the numpy array to a PIL Image to be used in tkinter
        image_rgb = cv2.cvtColor(np_array_image, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(image_rgb)
        updated_photo_image = ImageTk.PhotoImage(pil_img)

        # Updates the image of the processed label grid item
        self.processed_image_label.config(image=updated_photo_image)
        self.processed_image_label.image = updated_photo_image

    # ---------------------- Image Conversion Functions ------------------------ #
    # Converts from RGB to BGR
    def np_rgb_to_bgr(self, image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Converts from BGR to RGB
    def np_bgr_to_rgb(self, image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    # Converts from PIL image to numpy array image
    def pil_to_np_cv2(self, image):
        return np.array(image)

    # Converts from numpy array image to PIL image
    def np_cv2_to_pil(self, image):
        return Image.fromarray(image)

    # Converts from numpy array image to PIL image format and changes the color space from BGR to RGB
    def full_np_cv2_to_pil(self, image):
        return self.np_cv2_to_pil(self.np_bgr_to_rgb(image))

    # Converts from PIL image to numpy array image format and changes the color space from RGB to BGR
    def full_pil_to_np_cv2(self, image):
        return self.np_rgb_to_bgr(self.pil_to_np_cv2(image))

    # Converts from numpy array BGR image to a greyscale numpy array image
    def np_bgr_image_to_np_greyscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Returns the processed image
    def return_processed_image(self):
        return self.processed_image

    # Returns the original image
    def return_original_image(self):
        return self.image

    # Loads an image from a given path
    def load_image_from_path(self, image_path):
        return Image.open(image_path)

    # ---------------------- UI Functions ------------------------ #
    # Sets up the window
    def window_setup(self):
        self.root = tk.Tk()
        self.root.title(UI_WINDOW_TITLE) # Sets the window title
        self.root.attributes("-topmost", UI_TOPMOST) # Sets the window to be on top of all other windows
        self.root.resizable(UI_RESIZABLE, UI_RESIZABLE)

        # Detect button
        self.detect_btn = tk.Button(self.root, text="Detect", command=self.detect,
                                    font=(UI_DETECT_BUTTON_FONT, UI_DETECT_BUTTON_FONT_SIZE),
                                    fg=UI_DETECT_BUTTON_TEXT_COLOR)
        self.detect_btn.grid(row=0, column=0, columnspan=1, pady=10)

        # Coin counter label
        self.coin_counter_label = tk.Label(self.root, text=UI_COIN_LABEL_TEXT + "0",
                                           font=(UI_COIN_LABEL_FONT, UI_COIN_LABEL_FONT_SIZE),
                                           fg=UI_COIN_LABEL_TEXT_COLOR)
        self.coin_counter_label.grid(row=0, column=1, columnspan=1, pady=10)

        # Resizes the image to fit the window
        self.resized_image = self.resize_pil_image(self.image, UI_MAX_IMAGE_WIDTH)
        self.resized_photoimage = ImageTk.PhotoImage(self.resized_image)
        self.processed_resized_photoimage = self.resized_photoimage

        # Original image grid item
        self.original_image_label = tk.Label(self.root, image=self.resized_photoimage)
        self.original_image_label.grid(row=1, column=0, padx=5)

        # Processed image grid item
        self.processed_image_label = tk.Label(self.root, image=self.processed_resized_photoimage)
        self.processed_image_label.grid(row=1, column=1, padx=5)

        self.root.mainloop()

    # Helper function to display an image in a window, resizes the image to fit the window. Takes both numpy arrays and
    # PIL Images
    def custom_window_display(self, window_name, image, image_width):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Checks if image is a numpy array or PIL Image
        if isinstance(image, np.ndarray):
            image = self.np_cv2_to_pil(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise TypeError("Image must be a numpy array or PIL Image")

        resized_image = self.resize_pil_image(image, image_width)
        resized_image_np = np.asarray(resized_image)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(window_name, resized_image_np.shape[1], resized_image_np.shape[0])
        cv2.imshow(window_name, resized_image_np)


# Main function
if __name__ == "__main__":
    computer_vision = ComputerVision(image_path=MISC_IMAGE_PATH_PARAMETER_PATH,
                                     use_image_path_parameter_image_selection=MISC_USE_IMAGE_PATH_PARAMETER_IMAGE_SELECTION,
                                     use_webcam_snapshot_image_selection=MISC_USE_WEBCAM_SNAPSHOT_IMAGE_SELECTION,
                                     use_command_line_image_selection=MISC_COMMAND_LINE_ARGUMENT_IMAGE_SELECTION)
    computer_vision.window_setup()
