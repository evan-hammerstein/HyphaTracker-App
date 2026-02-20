import os
import csv
import cv2
from datetime import datetime

# Initialize global variables
start_point = None
end_point = None
drawing = False  # True if the mouse is pressed
rect_coords = []  # To store the rectangle coordinates

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, rect_coords

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button pressed
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse movement
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:  # Left mouse button released
        drawing = False
        end_point = (x, y)
        rect_coords = [
            start_point,
            (start_point[0], end_point[1]),
            end_point,
            (end_point[0], start_point[1])
        ]

# Load a PNG image with alpha channel support
image_path = "image.png"  # Replace with your PNG image path
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load PNG with transparency if present
if image is None:
    raise FileNotFoundError("Image not found. Check the path.")

clone = image.copy()  # Clone for resetting

# Set up an OpenCV window with fixed size matching the image dimensions
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Allow manual resizing of the window
height, width = image.shape[:2]
cv2.resizeWindow("Image", width, height)  # Resize window to match the image dimensions exactly

# Set DPI awareness to avoid Windows scaling issues (Windows-specific)
try:
    from ctypes import windll
    windll.user32.SetProcessDPIAware()  # Makes the application DPI-aware on Windows
except ImportError:
    pass

# Set up mouse callback for drawing rectangles
cv2.setMouseCallback("Image", draw_rectangle)

while True:
    temp_image = clone.copy()
    if start_point and end_point:
        cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)
    
    cv2.imshow("Image", temp_image)
    
    # Check for key presses or window closure
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("r"):  # Reset the drawing
        clone = image.copy()
        start_point, end_point = None, None
        rect_coords = []
    
    if key == ord("q") or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()

# Determine correct desktop path dynamically
desktop_paths = [
    os.path.join(os.path.expanduser("~"), "Desktop"),  # Standard desktop path
    os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"),  # OneDrive Desktop path
    os.path.join(os.path.expanduser("~"), "OneDrive - Imperial College London", "Desktop")  # Your specific OneDrive setup
]

desktop_path = None
for path in desktop_paths:
    if os.path.exists(path):
        desktop_path = path
        break

if not desktop_path:
    raise FileNotFoundError("Could not locate a valid Desktop directory.")

# Save the rectangle coordinates to a new CSV file on the desktop
if rect_coords:
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file_path = os.path.join(desktop_path, f'ROC_coordinates_{timestamp}.csv')

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y"])  # Column headers
        writer.writerows(rect_coords)  # Write each coordinate pair as a row

    print(f"Coordinates saved to {csv_file_path}")
else:
    print("No rectangle was drawn.")
