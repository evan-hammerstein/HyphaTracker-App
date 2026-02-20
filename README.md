# HyphaTracker - completed in GitHub Release! 
# https://github.com/evan-hammerstein/HyphaTracker-App/releases/tag/Release

This project is designed to analyze fungal growth using image processing techniques. It provides tools to process a sequence of tiff files representing a time-lapse of fungal growth, detect and track hyphal tips, calculate growth metrics, and visualize results. The analysis includes measurements of tip growth rates, branching frequency, spore identification, biomass quantification, and distances to regions of interest (ROIs).

Sample Images can be found here: 
# Features

**1. Interactive GUI:** File upload, metric specification, and output access and visualization

**2. Image Preprocessing:** Cropping, binarization, and skeletonization of grayscale images

**3. Hyphal Analysis:**

* Detect endpoints and calculate growth rates and angles

* Track hyphal tips and calculate distances to regions of interest (ROI)

* Analyze tip size and branching frequencies

**4. Biomass Analysis:** Measure fungal biomass over time

**5. Spore Tracking:**

* Identify spores based on size, shape, and proximity to biomass

* Track spores across multiple frames

**6. Visualization:** Generate visual outputs, including skeletonized images, tracked tips, and ROI distance visualizations

**7. Metrics Output:** Save results in CSV files and generate graphs for key metrics

# Requirements

## Software Dependencies

The code requires the following Python3 libraries:

* `sys`
* `os`
* `cv2` (OpenCV)
* `numpy`
* `skimage`
* `scipy`
* `matplotlib`
* `csv`

Please ensure all dependencies are installed before running the script.

## Input Data

* A folder containing grayscale image files (`.tif` format)

* Images should be named with frame numbers for sequential processing

## Compatible Systems

* Mac OS

# Usage Instructions

## Step 1: Prepare Input Data

Place all `.tif` image files in a single folder

Ensure file names follow a pattern to allow frame numbers to be extracted (e.g., processed_frame_1.tif)

## Step 2: Run the Script

Specify parameters like magnification, filter type, and sensitivity.

Upload the file folder for analysis and click 'Go'.

## Step 3: Review Results

Results will be saved in the specified output folders:

**1. CSV Files:** Contain metrics for tips, spores, biomass, etc

**2. Visualizations:** Skeletonized images, distance from ROI and tracked tip visualizations

**3. Graphs:** Growth rates, branching frequencies, and other metrics as `.png` files

# Key Functions

## Image Preprocessing

**`preprocess_image`:** Crops and binarizes the image

**`skeletonize_image`:** Reduces the image structures to 1-pixel-wide lines

## Hyphal Analysis

**`find_hyphal_endpoints`:** Detects hyphal tips/endpoints

**`track_tips_across_frames`:** Matches tips between frames

**`calculate_tip_size`:** Determines the size of a hyphal tip based on pixel area

**`track_tip_size_over_time`:** Tracks changes in hyphal tip size across multiple frames

**`calculate_overall_average_tip_size`:** Calculates the average size of all tips across frames

**`calculate_average_growth_rate`:** Computes the average growth rate of tips

**`calculate_growth_angles`:** Determines growth angles of tips relative to the horizontal

**`calculate_branching_rate`:** Identifies and counts overall branching events over time

## Biomass Analysis

**`find_biomass`:** Calculates the area covered by fungal biomass

**`calculate_biomass_over_time`:** Tracks biomass change across frames

## Spore Analysis

**`identify_spores`:** Detects spores based on size, shape, and proximity to hyphae

**`track_spores_over_time`:** Tracks spores across frames and calculates their size changes

## Visualization

**`show_image`:** Displays an image with the option of saving it

**`display_tips`:** Visualizes the skeletonized image with identified tips marked

**`visualize_tracked_tips`:** Shows the tracked tips across frames

**`calculate_distances_to_roi_and_visualize`:** Computes and visualizes the distances of tips from a specified ROI

## Additional Image Processing Functions

**`filter_hyphae`:** Removes small connected components (like noise or spores) to keep only large hyphae structures

**`process_frame`:** Applies normalization, thresholding, and morphological operations to enhance image quality for analysis

**`select_area`:** Allows user to interactively select an area for cropping

**`select_ROI`:** Enables selection of a rectangular region of interest (ROI)

# Outputs

Outputs are organized into three folders; csv_files, graphs, and visuals.
They are downloaded to the file explorer for access, and .tif files can be viewed on the platform.

## CSV Files:

* Hyphal tip metrics (growth rate, angles, sizes)

* Branching frequency

* Biomass over time

* Spore number and size

## Graphs:

Growth rates, branching frequencies, biomass trends, spore metrics

## Visualizations:

* Skeletonized images with tips

* Tip distance from ROI visualizations

# Customization

* Modify `distance_threshold`, `min_size`, or `circularity_threshold` to adjust sensitivity

* Update `roi_polygon` for custom regions of interest

* Change magnification to adjust pixel-to-area conversion factors

# Notes

* Ensure input images have strong enough contrast

* Review logs for any warnings or errors

# References

* JavaScript was heavily aided by ChatGPT, especially in the first made files home_renderer.js and global.js.
* CSS and HTML was aided but not pasted from ChatGPT, although troubleshooting was heavily aided.
* Python references are cited in-code

# Contact

For questions or support, please contact evan_hammerstein22@imperial.ac.uk
