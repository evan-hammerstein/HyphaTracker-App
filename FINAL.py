import logging
import sys
import tifffile as tiff
import time
import os
import cv2
import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import math
import os



# ========== IMAGE PROCESSING FUNCTIONS ==========

# Preprocess Image
def preprocess_image(image, crop_points = [
    (1625, 1032), (1827, 3045), (1897, 5848), 
    (2614, 6323), (9328, 5879), (9875, 5354),
    (9652, 2133), (9592, 376), (1988, 780)]):
    """
    Preprocess the image by cropping (using a predefined polygon region), 
    applying Otsu's thresholding, and binarizing the image.
    
    :param image: Grayscale image as a NumPy array.
    :return: Binary image as a NumPy array (1 for foreground, 0 for background).
    """

    # Step 1: Create a mask for non-rectangular cropping
    # Create an empty mask
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Define the polygon and fill it on the mask
    polygon = np.array(crop_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)  # Fill the polygon with white (255)
    
    # Apply the mask to the image
    image = cv2.bitwise_and(image, image, mask=mask)

    # Step 2: Apply Otsu's thresholding
    threshold = threshold_otsu(image)                                           # Compute optimal threshold using Otsu's method
    binary_image = image > threshold                                            # Binarize image using the threshold
    
    # Step 3: Return the binary image
    return binary_image.astype(np.uint8)                                        # Convert to uint8 for further processing

# Skeletonize Image
def skeletonize_image(binary_image):
    """
    Skeletonize a binary image to reduce structures to 1-pixel-wide lines.
    :param binary_image: Binary image as input.
    :return: Skeletonized binary image.
    """
    return skeletonize(binary_image > 0)  # Convert to boolean and skeletonize

# Remove small objects (e.g., spores or noise)
def filter_hyphae(binary_image, min_size=100):
    """
    Remove small connected components (e.g., spores or noise) to retain only large hyphae.
    :param binary_image: Binary image of the skeleton.
    :param min_size: Minimum size (in pixels) for connected components to retain.
    :return: Filtered binary image with small components removed.
    """
    labeled_image = label(binary_image)                                         # Label connected components in the image
    filtered_image = remove_small_objects(labeled_image, min_size=min_size)     # Remove small components
    return filtered_image > 0                                                   # Return as binary image (True for retained components)


# ========== VISUALIZATION FUNCTIONS ==========
# Display Image
def show_image(image, title='Image', visuals_folder="visuals"):
    """
    Display the given image and optionally save it to the visuals folder.
    :param image: Image to display.
    :param title: Title of the image window.
    :param visuals_folder: Path to the shared visuals folder.
    """
    plt.imshow(image, cmap='gray')  # Display image in grayscale
    plt.title(title)  # Set the title of the plot
    plt.axis('off')  # Hide the axes for better visualization

    # Save the image to the visuals folder if provided
    if visuals_folder:
        save_path = os.path.join(visuals_folder, f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")

    plt.show()

# Display Skeleton with Tips and Labels
def display_tips(binary_image, tips, frame_idx, visuals_folder="visuals"):
    """
    Display the skeleton image with tips and save the visualization to the visuals folder.
    :param binary_image: Skeletonized binary image.
    :param tips: List of (row, col) coordinates of tip positions.
    :param frame_idx: Frame index to display in the title.
    :param visuals_folder: Path to the shared visuals folder.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_image, cmap='gray')  # Display the skeleton

    # Mark tips with red dots and labels
    for idx, (y, x) in enumerate(tips):
        plt.scatter(x, y, c='red', s=0.5)

    title = f"Skeleton with Tips - Frame {frame_idx}"
    plt.title(title)
    plt.axis('off')

    # Save the visualization to the visuals folder if provided
    if visuals_folder:
        save_path = os.path.join(visuals_folder, f"tips_frame_{frame_idx}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")




# Visualize tracked tips
def visualize_tracked_tips(tracked_tips, image_file, frame_idx, visuals_folder="visuals"):
    """
    Visualize tracked tips and save the visualization to the visuals folder.
    :param tracked_tips: Dictionary of tracked tips.
    :param image_file: Path to the grayscale image file.
    :param frame_idx: Frame index to visualize.
    :param visuals_folder: Path to the shared visuals folder.
    """
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')

    # Overlay tracked tips
    for tip_id, positions in tracked_tips.items():
        for pos in positions:
            if pos[0] == frame_idx:
                y, x = pos[1:]
                plt.scatter(x, y, c='red', s=5)
                plt.text(x + 2, y - 2, str(tip_id), color='yellow', fontsize=6)

    title = f"Tracked Tips - Frame {frame_idx}"
    plt.title(title)
    plt.axis('off')

    # Save the visualization to the visuals folder if provided
    if visuals_folder:
        save_path = os.path.join(visuals_folder, f"tracked_tips_frame_{frame_idx}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")

    plt.show()


# =========== HYPHAL METRICS ===========================
#=======================================================

# ========== HYPHAL TIP DETECTION ==========

# Detect endpoints


def save_to_csv(data, filepath):
    """
    Save data to a CSV file.
    
    :param data: List of rows to write to the CSV file.
    :param filepath: Full path to the output CSV file.
    """
    # Check if folder exists
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    
    # Save the CSV
    with open(filepath, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)
    print(f"Data saved to {filepath}")


import os
import numpy as np
from scipy.ndimage import convolve
from skimage.measure import label
import csv

def find_hyphal_endpoints(filtered_skeleton, frame_idx, output_folder="csv_files"):
    """
    Detect endpoints of hyphae by identifying pixels with exactly one connected neighbor.
    Save the results for each frame in a separate CSV file.
    
    :param filtered_skeleton: Skeletonized binary image.
    :param frame_idx: Index of the current frame.
    :param output_folder: Folder to store the CSV files.
    :return: List of (y, x) coordinates of detected endpoints.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")

    # Define a 3x3 kernel to identify pixels with exactly one neighbor
    kernel = np.array([[1, 1, 1], 
                       [1, 10, 1], 
                       [1, 1, 1]])
    
    # Convolve the kernel with the skeleton to count neighbors for each pixel
    convolved = convolve(filtered_skeleton.astype(int), kernel, mode='constant', cval=0)
    
    # Identify pixels with exactly one neighbor (endpoints)
    endpoints = np.argwhere((convolved == 11))
    
    # Filter endpoints to ensure they belong to large hyphae components
    labeled_skeleton = label(filtered_skeleton)  # Label connected components in the skeleton
    valid_endpoints = []  # Initialize list to store valid endpoints
    for y, x in endpoints:
        if labeled_skeleton[y, x] > 0:  # Check if endpoint belongs to a labeled component
            valid_endpoints.append((y, x))  # Add valid endpoint to the list
    
    # Save to a frame-specific CSV file
    csv_filename = os.path.join(output_folder, f"hyphal_endpoints_frame_{frame_idx}.csv")
    with open(csv_filename, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["y", "x"])  # Header row
        csv_writer.writerows(valid_endpoints)  # Write the endpoints
    
    print(f"Hyphal endpoints for frame {frame_idx} saved to {csv_filename}")
    
    return valid_endpoints


#DISTANCE TO REGIONS OF INTEREST
# Example: Regions of interest (e.g., spore centroids)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ADD VALUE FROM GUI HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 
# roi = [(int(sys.argv[]), int(sys.argv[])), (int(sys.argv[]), int(sys.argv[]))]
import os
import cv2
import numpy as np
import csv

def calculate_distances_to_roi_and_visualize(tracked_tips, tip_id, roi_vertices, images, visuals_folder, csv_folder):
    """
    Calculate the distances of a specific hyphal tip to a rectangular region of interest (ROI) and create visualizations.

    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param tip_id: The ID of the tip for which distances should be calculated.
    :param roi_vertices: List of exactly 4 vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] defining the rectangle ROI.
    :param images: List of images corresponding to each frame.
    :param visuals_folder: Folder path to store visualizations.
    :param csv_folder: Folder path to store CSV data.
    :return: List of distances to the ROI for the specified tip over all frames.
    """
    if tip_id not in tracked_tips:
        raise ValueError(f"Tip ID {tip_id} not found in tracked tips.")

    if len(roi_vertices) != 4:
        raise ValueError("ROI must be defined by exactly 4 vertices.")

    # Ensure the folders exist
    os.makedirs(visuals_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    distances = []
    visualization_data = [["Frame", "Shortest Distance to ROI (µm)"]]

    # Convert ROI vertices into a proper NumPy array
    roi_polygon = np.array(roi_vertices, dtype=np.int32)

    for frame_idx, (frame, y_tip, x_tip) in enumerate(tracked_tips[tip_id]):
        if frame_idx >= len(images):
            break  # Ensure we do not exceed the number of frames

        # Get the corresponding grayscale image for the frame
        image = images[frame_idx]
        
        # Convert the image to RGB for visualization
        visualized_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Highlight the ROI in yellow
        cv2.polylines(visualized_image, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        
        # Highlight the tip in red
        cv2.circle(visualized_image, (int(x_tip), int(y_tip)), radius=5, color=(0, 0, 255), thickness=-1)

        # Convert tip position to float for `pointPolygonTest`
        tip_point = (float(x_tip), float(y_tip))

        # Check if the tip is inside the ROI
        point_in_roi = cv2.pointPolygonTest(roi_polygon, tip_point, False)
        if point_in_roi >= 0:
            distances.append(0)
            visualization_data.append([frame_idx, "0"])
        else:
            # Calculate the shortest distance from the tip to the ROI
            shortest_distance = float('inf')
            closest_point = None
            for i in range(len(roi_vertices)):
                # Get consecutive points in the polygon
                x1, y1 = roi_vertices[i]
                x2, y2 = roi_vertices[(i + 1) % len(roi_vertices)]  # Wrap around to the first point
                
                # Compute the closest point on the line segment to the tip
                px, py = closest_point_on_line_segment(x1, y1, x2, y2, x_tip, y_tip)
                distance = np.sqrt((px - x_tip) ** 2 + (py - y_tip) ** 2)
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_point = (px, py)
            
            distances.append(shortest_distance)
            visualization_data.append([frame_idx, f"{shortest_distance:.3f}"])

            # Draw the dotted line between the tip and the closest point on the ROI
            if closest_point:
                px, py = closest_point
                draw_dotted_line(visualized_image, (int(x_tip), int(y_tip)), (int(px), int(py)), color=(255, 255, 0))
        
        # Save the visualization
        output_path = os.path.join(visuals_folder, f"tip_{tip_id}_frame_{frame_idx}.png")
        cv2.imwrite(output_path, visualized_image)

    # Save distances to a CSV file
    csv_file_path = os.path.join(csv_folder, f"distances_to_roi_tip_{tip_id}.csv")
    with open(csv_file_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(visualization_data)
    
    print(f"Distances to ROI for Tip {tip_id} saved to {csv_file_path}.")
    print(f"Visualizations saved in {visuals_folder}.")

    return distances


# Define helper functions
def closest_point_on_line_segment(x1, y1, x2, y2, x, y):
    """
    Calculate the closest point on a line segment to a given point.
    """
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return x1, y1  # The segment is a single point
    
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp t to the range [0, 1]
    return x1 + t * dx, y1 + t * dy

def draw_dotted_line(image, start, end, color, thickness=1, gap=5):
    """
    Draw a dotted line on an image between two points.
    """
    x1, y1 = start
    x2, y2 = end
    length = int(np.hypot(x2 - x1, y2 - y1))
    for i in range(0, length, gap * 2):
        start_x = int(x1 + i / length * (x2 - x1))
        start_y = int(y1 + i / length * (x2 - y1))
        end_x = int(x1 + min(i + gap, length) / length * (x2 - x1))
        end_y = int(y1 + min(i + gap, length) / length * (y2 - y1))
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)

# ========== HYPHAL METRICS ==========


#TIP GROWTH RATE
def calculate_average_growth_rate(tracked_tips, frame_interval, time_per_frame, output_folder = "csv_files", graph_folder = "graphs"):
    """
    Calculate the average growth rate of hyphal tips over a specified number of frames.
    
    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param frame_interval: Number of frames over which to calculate the growth rate.
    :param time_per_frame: Time difference between consecutive frames in hours.
    :param output_folder: Folder to store the output CSV file.
    :return: Dictionary with tip IDs as keys and average growth rates as values, 
             and the general average growth rate for all tips.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
        print(f"Output folder created: {graph_folder}")

    average_growth_rates = {}
    total_growth_rates = []  # To store growth rates for all tips
    total_time = frame_interval * time_per_frame  # Total time for the specified frame interval

    for tip_id, positions in tracked_tips.items():
        growth_distances = []
        for i in range(len(positions) - frame_interval):
            # Get the positions separated by frame_interval
            _, y1, x1 = positions[i]
            _, y2, x2 = positions[i + frame_interval]
            
            # Calculate Euclidean distance
            distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            growth_rate = distance / total_time
            growth_distances.append(growth_rate)
            total_growth_rates.append(growth_rate)  # Add to the overall growth rates

        # Calculate the average growth rate for the tip
        if growth_distances:
            average_growth_rate = sum(growth_distances) / len(growth_distances)
        else:
            average_growth_rate = 0  # If no valid growth distances are found

        average_growth_rates[tip_id] = average_growth_rate

    # Calculate the general average growth rate
    if total_growth_rates:
        general_average_growth_rate = sum(total_growth_rates) / len(total_growth_rates)
    else:
        general_average_growth_rate = 0

    # Save average growth rates to CSV
    growth_rate_data = [["Tip ID", "Average Growth Rate (µm/s)"]]
    growth_rate_data += [[tip_id, f"{rate:.3f}"] for tip_id, rate in average_growth_rates.items()]
    growth_rate_data.append([])
    growth_rate_data.append(["General Average Growth Rate", f"{general_average_growth_rate:.3f}"])

    save_to_csv(growth_rate_data, os.path.join(output_folder, "average_growth_rates.csv"))
    print("Average growth rates saved to CSV.")

    # Plot and save growth rate graph
    plt.figure(figsize=(10, 6))
    plt.bar(average_growth_rates.keys(), average_growth_rates.values(), color='blue')
    plt.xlabel("Tip ID")
    plt.ylabel("Average Growth Rate (µm/hr)")
    plt.title("Average Growth Rate of Hyphal Tips")
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_path = os.path.join(graph_folder, "average_growth_rates.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"Growth rate graph saved to {graph_path}")

    return average_growth_rates, general_average_growth_rate


#TIP GROWTH ANGLE

def calculate_growth_angles(tracked_tips, tip_id, output_folder = "csv_files", graph_folder = "graphs"):
    """
    Calculate the growth angles of a specific hyphal tip over time with respect to the horizontal.

    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param tip_id: The ID of the tip for which growth angles should be calculated.
    :param output_folder: Folder to store the output CSV file.
    :return: List of growth angles (in degrees) for the specified tip over time.
    """
    if tip_id not in tracked_tips:
        raise ValueError(f"Tip ID {tip_id} not found in tracked tips.")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
        print(f"Output folder created: {graph_folder}")

    positions = tracked_tips[tip_id]  # Get the positions of the specified tip
    growth_angles = []  # List to store growth angles

    for i in range(1, len(positions)):
        _, y1, x1 = positions[i - 1]
        _, y2, x2 = positions[i]
        
        # Compute differences
        delta_x = x2 - x1
        delta_y = y2 - y1
        
        # Calculate angle in radians and convert to degrees
        angle_radians = math.atan2(delta_y, delta_x)
        angle_degrees = math.degrees(angle_radians)
        
        growth_angles.append(angle_degrees)
    
    # Save growth angles to CSV
    growth_angle_data = [["Frame", "Growth Angle (°)"]]
    growth_angle_data += [[i + 1, f"{angle:.3f}"] for i, angle in enumerate(growth_angles)]

    save_to_csv(growth_angle_data, os.path.join(output_folder, f"growth_angles_tip_{tip_id}.csv"))
    print(f"Growth angles for Tip {tip_id} saved to CSV.")

    # Plot and save growth angle graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(growth_angles) + 1), growth_angles, marker='o', color='blue', label=f"Tip {tip_id}")
    plt.xlabel("Frame")
    plt.ylabel("Growth Angle (°)")
    plt.title(f"Growth Angles of Tip {tip_id} Over Frames")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    graph_path = os.path.join(graph_folder, f"growth_angles_tip_{tip_id}.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"Growth angle graph for Tip {tip_id} saved to {graph_path}")

    return growth_angles


def calculate_tip_size(binary_image, tip_position, pixel_area, radius_microns=10):
    """
    Calculate the size of a single tip by counting the filled pixels within a specified radius.

    :param binary_image: Binary image as a NumPy array (1 for foreground, 0 for background).
    :param tip_position: Tuple (y, x) representing the position of the tip.
    :param radius_microns: Radius around the tip in microns.
    :param pixel_area: Area per pixel in micrometers squared
    :return: Tip size in µm².
    """

    y, x = tip_position
    radius_pixels = int(np.sqrt(radius_microns**2 / pixel_area))  # Convert radius from microns to pixels

    mask = np.zeros_like(binary_image, dtype=bool)  # Create a circular mask for the ROI
    y_grid, x_grid = np.ogrid[:binary_image.shape[0], :binary_image.shape[1]]
    distance_from_tip = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
    mask[distance_from_tip <= radius_pixels] = True

    # Count filled pixels within the circular mask
    tip_pixels = np.sum(binary_image[mask])

    # Convert the count of pixels to area in microns squared
    tip_size = tip_pixels * pixel_area
    return tip_size


def track_tip_size_over_time(tracked_tips, binary_images, tip_id, pixel_area, radius_microns=10, output_folder="csv_files", graph_folder="graphs"):
    """
    Track the size of a specific tip over time across multiple frames and save the results to a CSV file.

    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param binary_images: List of binary images (one per frame).
    :param tip_id: The ID of the tip to track.
    :param radius_microns: Radius around the tip in microns.
    :param output_folder: Folder path to store the CSV file.
    :return: List of tip sizes (in µm²) over time.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")
    
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
        print(f"Output folder created: {graph_folder}")

    if tip_id not in tracked_tips:
        raise ValueError(f"Tip ID {tip_id} not found in tracked tips.")

    tip_sizes = []  # To store the size of the tip in each frame
    tip_positions = tracked_tips[tip_id]  # Get the positions of the specified tip

    for frame_idx, (frame, y, x) in enumerate(tip_positions):
        # Get the binary image for the current frame
        binary_image = binary_images[frame]

        # Calculate the size of the tip in the current frame
        tip_size = calculate_tip_size(binary_image, (y, x), radius_microns, pixel_area)
        tip_sizes.append((frame, tip_size))

    # Save the results to a CSV file
    csv_file = os.path.join(output_folder, f"tip_{tip_id}_sizes.csv")
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Tip Size (µm²)"])  # Header row
        csv_writer.writerows(tip_sizes)  # Write data rows

    print(f"Tip sizes saved to {csv_file}")

    # Plot and save the tip size graph
    frames = [entry[0] for entry in tip_sizes]
    sizes = [entry[1] for entry in tip_sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(frames, sizes, marker='o', color='green', label=f"Tip {tip_id}")
    plt.xlabel("Frame")
    plt.ylabel("Tip Size (µm²)")
    plt.title(f"Tip Size of Tip {tip_id} Over Frames")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    graph_path = os.path.join(graph_folder, f"tip_size_tip_{tip_id}.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"Tip size graph for Tip {tip_id} saved to {graph_path}")

    return tip_sizes


def calculate_overall_average_tip_size(tracked_tips, binary_images, pixel_area, radius_microns=10, output_folder="csv_files"):
    """
    Calculate the overall average size of all tips across all frames and save the result to a CSV file.

    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param binary_images: List of binary images (one per frame).
    :param radius_microns: Radius around the tip in microns for size calculation.
    :param output_folder: Folder path to store the CSV file.
    :return: The overall average tip size (µm²).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    total_size = 0
    total_count = 0

    for tip_id, positions in tracked_tips.items():
        for frame, y, x in positions:
            # Get the binary image for the current frame
            binary_image = binary_images[frame]

            # Calculate the size of the tip in the current frame
            tip_size = calculate_tip_size(binary_image, (y, x), radius_microns, pixel_area)
            total_size += tip_size
            total_count += 1

    # Calculate overall average size
    overall_average_size = total_size / total_count if total_count > 0 else 0

    # Save the result to a CSV file
    csv_file = os.path.join(output_folder, "overall_average_tip_size.csv")
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Metric", "Value"])  # Header row
        csv_writer.writerow(["Overall Average Tip Size (µm²)", overall_average_size])

    print(f"Overall average tip size saved to {csv_file}")
    return overall_average_size


#============ BRANCHING FREQUENCY ===============
import os
import csv
from scipy.spatial.distance import cdist

def calculate_branching_rate(tip_positions, distance_threshold=15, output_folder="csv_files", graph_folder="graphs"):
    """
    Calculate the branching rate/frequency of fungal hyphae over time and save to a CSV file.

    :param tip_positions: List of lists of (y, x) tip positions for each frame.
    :param distance_threshold: Maximum distance to consider tips as originating from the same source.
    :param output_folder: Folder path to store the CSV file.
    :return: List of branching events per frame and total branching events.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
        print(f"Output folder created: {graph_folder}")

    branching_events_per_frame = []  # List to store branching events for each frame
    total_branching_events = 0  # Total number of branching events

    # Iterate over consecutive frames
    for frame_idx in range(1, len(tip_positions)):
        current_tips = tip_positions[frame_idx]  # Tips in the current frame
        previous_tips = tip_positions[frame_idx - 1]  # Tips in the previous frame

        if not previous_tips or not current_tips:
            branching_events_per_frame.append(0)
            continue

        # Calculate distances between previous and current tips
        distances = cdist(previous_tips, current_tips)

        # For each tip in the previous frame, count the number of associated tips in the current frame
        branching_events = 0
        for i, _ in enumerate(previous_tips):
            # Find indices of current tips within the distance threshold
            matching_tips = np.where(distances[i] < distance_threshold)[0]
            
            # If there are more than one matching tip, it indicates branching
            if len(matching_tips) > 1:
                branching_events += len(matching_tips) - 1  # Count new branches

        # Update the branching events
        branching_events_per_frame.append(branching_events)
        total_branching_events += branching_events

    # Save the results to a CSV file
    csv_file = os.path.join(output_folder, "branching_rate.csv")
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Branching Events"])  # Header row
        csv_writer.writerows(enumerate(branching_events_per_frame))  # Frame-wise data
        csv_writer.writerow(["Total Branching Events", total_branching_events])

    print(f"Branching rate saved to {csv_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(branching_events_per_frame) + 1), branching_events_per_frame, marker='o', label="Branching Events")
    plt.title("Branching Events Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Branching Events")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graph_folder, "branching_rate_graph.png"))
    plt.close()

    print(f"Branching rate saved to {graph_folder}")


    return branching_events_per_frame, total_branching_events


# ========== MYCELIAL METRICS ==========

def find_biomass(binary_image, pixel_area):
    """
    Calculate the biomass (physical area) of the fungal structure in the binary image.

    :param binary_image: Binary image as a NumPy array (1 for foreground, 0 for background).
    :param pixel_area: Area per pixel in micrometers squared
    :return: Biomass in micrometers squared.
    """

    # Calculate biomass (number of foreground pixels * pixel area)
    biomass_pixels = np.sum(binary_image)  # Count the number of white pixels
    biomass_area = biomass_pixels * pixel_area  # Total biomass in µm²

    return biomass_area


def calculate_biomass_over_time(image_files, pixel_area, output_folder="csv_files", graph_folder="graphs"):
    """
    Calculate biomass over time for a sequence of images and save to a CSV file.

    :param image_files: List of file paths to the PNG images.
    :param pixel_area: Area per pixel in micrometers squared
    :param output_folder: Folder path to store the CSV file.
    :return: List of biomass values (one for each frame).
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
        print(f"Output folder created: {graph_folder}")

    biomass_values = []

    for frame_idx, file in enumerate(image_files):
        # Load and preprocess the image
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        binary_image = preprocess_image(image)
        
        # Calculate biomass
        biomass = find_biomass(binary_image, pixel_area)
        biomass_values.append((frame_idx, biomass))

    # Save the results to a CSV file
    csv_file = os.path.join(output_folder, "biomass_over_time.csv")
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Biomass (µm²)"])  # Header row
        csv_writer.writerows(biomass_values)  # Write data rows

    print(f"Biomass over time saved to {csv_file}")

    plt.figure(figsize=(10, 6))
    frames, biomass = zip(*biomass_values)
    plt.plot(frames, biomass, marker='o', label="Biomass")
    plt.title("Biomass Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Biomass (µm²)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graph_folder, "biomass_graph.png"))
    plt.close()

    print(f"Biomass over time saved to {graph_folder}")

    return [value[1] for value in biomass_values]


# ==========SPORES===========

def identify_spores(image, min_size, max_size, circularity_threshold):
    """
    Identify spores in the image based on size, circularity, and strong proximity to biomass.
    """
    # Preprocess the image
    threshold = threshold_otsu(image)
    binary_image = (image > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spores = []

    # Analyze each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if min_size <= area <= max_size and perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity >= circularity_threshold:
                # Check the area around the spore for biomass
                mask = np.zeros_like(cleaned_image, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

                # Extract pixels around the spore
                dilated_mask = cv2.dilate(mask, kernel, iterations=5)
                biomass_overlap = cv2.bitwise_and(dilated_mask, binary_image)
                biomass_fraction = np.sum(biomass_overlap) / np.sum(dilated_mask)

                # Require a minimum fraction of biomass overlap for connection
                if biomass_fraction > 0.5:  # At least 50% of surrounding pixels must be biomass
                    (x, y), _ = cv2.minEnclosingCircle(contour)  # Center of spore
                    spores.append({"center": (int(x), int(y)), "size": area})

    return spores



#NUMBER/SIZE/DISTRIBUTION OF SPORES (SPHERICAL STRUCTURES)
from scipy.spatial.distance import cdist

def track_spores_over_time(image_files, min_size=10, max_size=200, circularity_threshold=0.7, distance_threshold=15, output_folder="csv_files", graph_folder="graphs"):
    """
    Track spores over time across a sequence of images and output their sizes over time.
    
    :param image_files: List of file paths to the PNG images.
    :param min_size: Minimum size of objects to consider as spores.
    :param max_size: Maximum size of objects to consider as spores.
    :param circularity_threshold: Minimum circularity to consider an object as a spore.
    :param distance_threshold: Maximum distance to match spores between frames.
    :param output_folder: Folder to save the resulting CSV file.
    :return: Dictionary of tracked spores with their sizes over time.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")
    
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
        print(f"Output folder created: {graph_folder}")

    # Dictionary to store tracked spores: {spore_id: {"history": [(frame_idx, size)], "last_position": (x, y)}}
    tracked_spores = {}
    next_spore_id = 0

    spore_count_per_frame = []  # Store the number of spores detected per frame
    average_spore_size_per_frame = []  # Store the average spore size per frame

    # Process each frame
    for frame_idx, file in enumerate(image_files):
        # Load the image
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"File not found: {file}")

        # Identify spores in the current frame
        current_spores = identify_spores(image, min_size, max_size, circularity_threshold)

        # Update spore count and average size
        spore_count = len(current_spores)
        spore_count_per_frame.append((frame_idx, spore_count))

        if spore_count > 0:
            average_size = sum(spore["size"] for spore in current_spores) / spore_count
        else:
            average_size = 0
        average_spore_size_per_frame.append((frame_idx, average_size))

        if frame_idx == 0:
            # Initialize tracking for the first frame
            for spore in current_spores:
                tracked_spores[next_spore_id] = {
                    "history": [(frame_idx, spore["size"])],
                    "last_position": spore["center"],
                }
                next_spore_id += 1
            continue

        # Match spores to those in the previous frame
        previous_positions = [data["last_position"] for data in tracked_spores.values()]
        current_positions = [spore["center"] for spore in current_spores]

        if previous_positions and current_positions:
            distances = cdist(previous_positions, current_positions)

            matched_current = set()
            for spore_id, prev_position in enumerate(previous_positions):
                # Find the nearest current spore
                nearest_idx = np.argmin(distances[spore_id])
                if distances[spore_id, nearest_idx] < distance_threshold:
                    # Update the spore's history and last position
                    tracked_spores[spore_id]["history"].append((frame_idx, current_spores[nearest_idx]["size"]))
                    tracked_spores[spore_id]["last_position"] = current_spores[nearest_idx]["center"]
                    matched_current.add(nearest_idx)

            # Add new spores that were not matched
            for j, spore in enumerate(current_spores):
                if j not in matched_current:
                    tracked_spores[next_spore_id] = {
                        "history": [(frame_idx, spore["size"])],
                        "last_position": spore["center"],
                    }
                    next_spore_id += 1

    # Save spore count and average size data to CSV
    spore_count_csv = os.path.join(output_folder, "spore_count_per_frame.csv")
    with open(spore_count_csv, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Spore Count"])  # Header row
        csv_writer.writerows(spore_count_per_frame)  # Write data rows

    average_spore_size_csv = os.path.join(output_folder, "average_spore_size_per_frame.csv")
    with open(average_spore_size_csv, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Average Spore Size (µm²)"])  # Header row
        csv_writer.writerows(average_spore_size_per_frame)  # Write data rows

    # Create graphs
    frames, spore_counts = zip(*spore_count_per_frame)
    _, average_sizes = zip(*average_spore_size_per_frame)

    # Spore Count Graph
    plt.figure(figsize=(10, 6))
    plt.bar(frames, spore_counts, color='blue', alpha=0.7, label="Spore Count")
    plt.title("Spore Count Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Spore Count")
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(os.path.join(graph_folder, "spore_count_graph.png"))
    plt.close()

    # Average Spore Size Graph
    plt.figure(figsize=(10, 6))
    plt.plot(frames, average_sizes, marker='o', color='green', label="Average Spore Size")
    plt.title("Average Spore Size Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Average Spore Size (µm²)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(graph_folder, "average_spore_size_graph.png"))
    plt.close()

    print(f"Spore count and size saved to CSV, and graphs saved to {graph_folder}.")
    return tracked_spores


# ========== SEQUENCE PROCESSING ==========

# Process a sequence of images and track tips
def process_sequence(image_files, min_size=50, distance_threshold=15): # adjust threshold as needed after testing
    """
    Process a sequence of images and track hyphal tips over time.
    
    :param image_files: List of file paths to the PNG images.
    :param min_size: Minimum size for connected components (filtering small noise).
    :param distance_threshold: Maximum distance to consider two tips as the same.
    :return: Dictionary of tracked tips.
    """
    tip_positions = []  # List to store tip positions for each frame
    
    for file in image_files:
        # Load and preprocess the image
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        binary_image = preprocess_image(image)
        skeleton = skeletonize_image(binary_image)
        filtered_skeleton = filter_hyphae(skeleton, min_size=min_size)
        
        # Find tips in the current frame
        tips = find_hyphal_endpoints(filtered_skeleton)
        tip_positions.append(tips)
    
    # Track tips across frames
    tracked_tips = track_tips_across_frames(tip_positions, distance_threshold)
    
    return tracked_tips

# Match tips between frames and handle branching
def track_tips_across_frames(tip_positions, distance_threshold=15, output_folder="csv_files"):
    """
    Track hyphal tips across frames, creating separate lists for new branches.
    
    :param tip_positions: List of tip positions for each frame (list of lists of (y, x) tuples).
    :param distance_threshold: Maximum distance to consider two tips as the same.
    :param output_folder: Folder to save the resulting CSV file.
    :return: Dictionary with keys as tip IDs and values as lists of positions [(frame, y, x)].
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")
    
    # Step 1: Initialize variables for tracking
    tracked_tips = {}  # Dictionary to store tip tracking {tip_id: [(frame, y, x)]}
    next_tip_id = 0  # Unique ID for each tip (keys of dictionary)
    
    # Step 2: Process frames
    for frame_idx, current_tips in enumerate(tip_positions):
        if frame_idx == 0:
            # Initialize tracking for the first frame
            for tip in current_tips:
                tracked_tips[next_tip_id] = [(frame_idx, *tip)]
                next_tip_id += 1
            continue
        

        # Match tips to the previous frame
        previous_tips = [positions[-1][1:] for positions in tracked_tips.values()]
        """if len(previous_tips) == 0 or len(current_tips) == 0:
            distances = np.array([])  # No distances to compute
        else:
            distances = cdist(previous_tips, current_tips)"""

        distances = cdist(np.atleast_2d(previous_tips), np.atleast_2d(current_tips))
  # Compute distances between tips
        
        # Step 3: Match tips
        matched_current = set()
        for i, prev_tip in enumerate(previous_tips):
            # Find the nearest current tip within the distance threshold
            nearest_idx = np.argmin(distances[i])
            if distances[i, nearest_idx] < distance_threshold:
                tracked_tips[i].append((frame_idx, *current_tips[nearest_idx]))
                matched_current.add(nearest_idx)

        # Add new tips that were not matched
        for j, current_tip in enumerate(current_tips):
            if j not in matched_current:
                tracked_tips[next_tip_id] = [(frame_idx, *current_tip)]
                next_tip_id += 1

    # Step 4: Save the results to a CSV file
    csv_file = os.path.join(output_folder, "tracked_tips.csv")
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Tip ID", "Frame", "Y", "X"])  # Header row
        for tip_id, positions in tracked_tips.items():
            for position in positions:
                csv_writer.writerow([tip_id, position[0], position[1], position[2]])

    print(f"Tracked tips saved to {csv_file}")
    return tracked_tips


# ========== MAIN EXECUTION ==========

# ========== Input Parameters ==========

import os
import shutil

def delete_folder(folder_path):
    """
    Delete the specified folder and all its contents.
    
    :param folder_path: Path to the folder to delete.
    """
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)  # Remove the folder and all its contents
            print(f"Deleted folder: {folder_path}")
        except Exception as e:
            print(f"Error deleting folder {folder_path}: {e}")
    else:
        print(f"Folder does not exist: {folder_path}")


# Main image processing function
def process_frame(frame, thresholder, threshold_value):
    # Normalize pixel values
    if frame.max() > 255:
        frame = (255 * (frame / frame.max())).astype(np.uint8)

    # Convert to grayscale
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter
    filtered = cv2.bilateralFilter(frame, d=15, sigmaColor=30, sigmaSpace=25)

    # Apply adaptive Gaussian threshold
    Thresholded = cv2.adaptiveThreshold(filtered, 255, thresholder, cv2.THRESH_BINARY, threshold_value, 2)

    # Divide and invert
    divide = cv2.divide(Thresholded, frame, scale=255)
    divide = 255 - divide

    # Stretch intensity
    maxval = np.amax(divide) / 4
    stretch = rescale_intensity(divide, in_range=(0, maxval), out_range=(0, 255)).astype(np.uint8)

    # Morphological operations for cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(stretch, cv2.MORPH_OPEN, kernel)
    filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return filled


#SORTING FRAMES BASED ON FRAME NUMBER

# Sort the files based on the frame number extracted from the filenames
def extract_frame_number(file_path):
    file_name = os.path.basename(file_path)  # Get the file name
    # Extract the frame number using string splitting
    # Assuming filenames are in the format "processed_frame_<number>_timestamp.tif"
    parts = file_name.split('_')
    return int(parts[2])  # Frame number is the 3rd part (index 2)


def select_area(event, x, y, flags, param):
    global selected_area, resizing, resized_img, scale_factor, selection_done

    if selection_done:  # Do not allow selection after the first frame
        return

    # Scale coordinates back to the original resolution
    x, y = int(x / scale_factor), int(y / scale_factor)

    if event == cv2.EVENT_LBUTTONDOWN:  # Start selection
        selected_area = [x, y, x, y]

    elif event == cv2.EVENT_MOUSEMOVE and selected_area:  # Update rectangle dynamically
        selected_area[2], selected_area[3] = x, y

        # Draw the rectangle dynamically on the resized image
        temp_img = resized_img.copy()
        x1, y1, x2, y2 = selected_area
        cv2.rectangle(temp_img, (int(x1 * scale_factor), int(y1 * scale_factor)),
                      (int(x2 * scale_factor), int(y2 * scale_factor)), (255, 255, 255), 2)  
        cv2.imshow("Image", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:  # Finalize selection
        if selected_area:
            selected_area = [min(selected_area[0], selected_area[2]), min(selected_area[1], selected_area[3]),
                             max(selected_area[0], selected_area[2]), max(selected_area[1], selected_area[3])]
            print(f"Selected area: {selected_area}")
        selection_done = True


def select_ROI(event, x, y, flags, param):
    global selected_ROI, resizing, resized_img, scale_factor, ROIselection_done
    print("1")
    if ROIselection_done:  # Do not allow selection after the first frame
        return
    print("2")
    # Scale coordinates back to the original resolution
    x, y = int(x / scale_factor), int(y / scale_factor)
    print("3")
    if event == cv2.EVENT_LBUTTONDOWN:  # Start selection
        selected_ROI = [x, y, x, y]
        print("4")
    elif event == cv2.EVENT_MOUSEMOVE and selected_ROI:  # Update rectangle dynamically
        selected_ROI[2], selected_ROI[3] = x, y
        print("4a")
        # Draw the rectangle dynamically on the resized image
        temp_img = resized_img.copy()
        x1, y1, x2, y2 = selected_ROI
        print("5")
        cv2.rectangle(temp_img, (int(x1 * scale_factor), int(y1 * scale_factor)),
                      (int(x2 * scale_factor), int(y2 * scale_factor)), (255, 255, 255), 2)  
        cv2.imshow("Image", temp_img)
        print("6")

    elif event == cv2.EVENT_LBUTTONUP:  # Finalize selection
        print("4b")
        if selected_ROI:
            selected_ROI = [min(selected_ROI[0], selected_ROI[2]), min(selected_ROI[1], selected_ROI[3]),
                             max(selected_ROI[0], selected_ROI[2]), max(selected_ROI[1], selected_ROI[3])]
        ROIselection_done = True


# Define input and output parameters
#outputs_dir =  sys.argv[5]   Base output directory from command-line arguments
selected_area = None
ROI_done = False
selected_ROI = None
resizing = False
original_img = None
resized_img = None
scale_factor = 1.0
selection_done = False
ROIselection_done = False  # Flag to indicate selection is complete
# MAKE ALL OF THE FOLLOWING BE INPUTS FROM GUI
magnification = sys.argv[2]  # Magnification level
time_per_frame = 2  # Time difference between consecutive frames in hours
frame_interval = 2  # Frame interval for growth rate calculations
distance_threshold = 15  # Distance threshold for tip tracking
min_size_spores = 10  # Minimum size for spores
max_size_spores = 200  # Maximum size for spores
circularity_threshold = 0.7  # Circularity threshold for spores
tip_id = 260
roi_polygon = []  # ROI polygon coordinates
if magnification == "10x":
    pixel_area = 0.65**2
elif magnification == "20x":
    pixel_area = 0.33**2
elif magnification == "40x":
    pixel_area = 0.17**2
elif magnification == "100x":
    pixel_area = 0.07**2
thres_type_sys = sys.argv[3]
if thres_type_sys == "Gaussian":
    thres_type =  cv2.ADAPTIVE_THRESH_GAUSSIAN_C
else:
    thres_type = cv2.ADAPTIVE_THRESH_MEAN_C
 #Gaussian or mean on GUI
Sensitivity = (int(sys.argv[4])*2)+3

log_file = "hypha_tracker.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)



def main():
  #  try:
        logging.info("Starting HyphaTracker")
        # Create output subfolders
        outputs_dir = sys.argv[5]
        os.makedirs(outputs_dir, exist_ok=True)  # Create the directory if it doesn't exist
        folder_path = sys.argv[5]
        os.makedirs(folder_path, exist_ok=True)  # Create the directory if it doesn't exist
        csv_folder = os.path.join(outputs_dir, "csv_files")
        visuals_folder = os.path.join(outputs_dir, "visuals")
        graph_folder = os.path.join(outputs_dir, "graphs")

        os.makedirs(csv_folder, exist_ok=True)
        os.makedirs(visuals_folder, exist_ok=True)
        os.makedirs(graph_folder, exist_ok=True)
        # Process image sequence
        tip_positions_sequence = []
        biomass_values = []
        images = []
        global selected_area, selected_ROI, resizing, resized_img, scale_factor, selection_done, ROIselection_done

        tiff_file = sys.argv[1]
        frames = tiff.imread(tiff_file)
        logging.info(f"Loaded TIFF file with {len(frames)} frames.")

        for frame_idx, frame in enumerate(frames):
            logging.info(f"Processing frame {frame_idx +1}.")
            # Load and preprocess image

            
            # Crop and process the selected area
            original_img = frame.copy()
            if selected_area:
                x1, y1, x2, y2 = selected_area
                cropped_frame = original_img[y1:y2, x1:x2]  # Crop the frame
                processed_frame = process_frame(cropped_frame, thres_type, Sensitivity)
            else:
                processed_frame = process_frame(original_img, thres_type, Sensitivity)

            # Prepare resized frame for display
            height, width = processed_frame.shape[:2]
            scale_factor = 800 / width  # Resize width to 800 pixels
            resized_img = cv2.resize(processed_frame, (800, int(height * scale_factor)))
            # Handle Area Selection (Frame 0)
            if frame_idx == 0 and not selection_done:
                cv2.imshow("Image", resized_img)
                cv2.setMouseCallback("Image", select_area)
                logging.info("Waiting for area selection.")
                
                while not selection_done:  # Wait until selection is done
                    key = cv2.waitKey(1)  # Wait for 1 ms for a key press (non-blocking)
                    if key == 27:  # Esc key to exit
                        logging.warning("User exited during area selection.")
                        cv2.destroyWindow("Image")
                        break
                if selection_done:
                    cv2.destroyWindow("Image")  # Close the window after selection is complete
                    logging.info(f"Selected area: {selected_area}")

            # Handle ROI Selection (Frame 1)
            if frame_idx == 1 and not ROIselection_done and ROI_done:
                cv2.imshow("Image", resized_img)
                cv2.setMouseCallback("Image", select_ROI)
                logging.info("Waiting for ROI selection.")
                
                while not ROIselection_done:  # Wait until selection is done
                    key = cv2.waitKey(1)  # Wait for 1 ms for a key press (non-blocking)
                    if key == 27:  # Esc key to exit
                        logging.warning("User exited during ROI selection.")
                        cv2.destroyWindow("Image")
                        break
                if ROIselection_done:
                    cv2.destroyWindow("Image")  # Close the window after selection is complete
                    roi_polygon = selected_ROI
                    logging.info(f"Selected ROI: {selected_ROI}")

            skeleton = skeletonize_image(processed_frame)
            filtered_skeleton = filter_hyphae(skeleton, min_size=500)

            # Find and save hyphal endpoints
            endpoints = find_hyphal_endpoints(filtered_skeleton, frame_idx, csv_folder)
            tip_positions_sequence.append(endpoints)

            # Display and save the skeleton with tips visualized
            display_tips(processed_frame, endpoints, frame_idx, visuals_folder=visuals_folder)

            # Calculate biomass
            biomass = find_biomass(processed_frame, pixel_area)
            biomass_values.append(biomass)

                    # Save each processed frame with a unique name
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_frame_path = os.path.join(folder_path, f"processed_frame_{frame_idx + 1}_{timestamp}.tif")
            tiff.imwrite(output_frame_path, processed_frame)
            logging.info(f"Saved processed frame {frame_idx + 1} to {output_frame_path}")

        cv2.destroyAllWindows()

        # Collect all `.tif` file paths from the folder
        image_files = [
            os.path.join(folder_path, file) 
            for file in os.listdir(folder_path) 
            if file.endswith('.tif')
        ]

        # Track hyphal tips
        tracked_tips = track_tips_across_frames(tip_positions_sequence, distance_threshold, csv_folder)

        # Visualize distances to ROI
        if ROI_done:
            calculate_distances_to_roi_and_visualize(
                tracked_tips, tip_id, roi_polygon, images, visuals_folder, csv_folder
            )

        # Calculate and save metrics
        calculate_average_growth_rate(tracked_tips, frame_interval, time_per_frame, csv_folder, graph_folder)
        calculate_growth_angles(tracked_tips, tip_id, csv_folder, graph_folder)
        calculate_branching_rate(tip_positions_sequence, distance_threshold, csv_folder, graph_folder)

        # Biomass analysis
        calculate_biomass_over_time(image_files, pixel_area, csv_folder, graph_folder)

        track_spores_over_time(image_files, 10, 200, 0.7, 15, csv_folder, graph_folder)

        logging.info("Processing complete. Results saved in:", outputs_dir)
 #   except Exception as e:
       # logging.critical(f"Critical error in running: {e}")

if __name__ == "__main__":
    main()
      
