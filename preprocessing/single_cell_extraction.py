import sys
import os
import cv2
import numpy as np
import csv
import shutil
from openslide import OpenSlide
from skimage import morphology
from skimage.morphology import h_maxima
from PIL import Image
from omegaconf import DictConfig
import hydra

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
script_dir = os.path.dirname(os.path.abspath(__file__))
cfgs_pth = os.path.abspath(os.path.join(script_dir, "..", "configs"))

def apply_gamma_correction(image, gamma=1.5):
    """
    Applies gamma correction to the image.
    
    Parameters:
        image (ndarray): Input image.
        gamma (float): Gamma value for correction.
    
    Returns:
        ndarray: Gamma-corrected image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def is_qualitative(patch, threshold=0.25, background_threshold=200):
    """
    Checks if a patch has a sufficient background proportion.
    
    Parameters:
        patch (ndarray): Image patch.
        threshold (float): Minimum ratio of background pixels.
        background_threshold (int): Pixel value above which a pixel is considered background.
    
    Returns:
        bool: True if patch is qualitative.
    """
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    value_channel = hsv_patch[:, :, 2]
    background_mask = value_channel > background_threshold
    background_ratio = np.sum(background_mask) / background_mask.size
    return background_ratio > threshold

def is_blurry(patch, threshold=100.0):
    """
    Checks if a patch is too blurry using the Laplacian variance method.
    
    Parameters:
        patch (ndarray): Image patch.
        threshold (float): Variance threshold.
    
    Returns:
        bool: True if patch is blurry.
    """
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray_patch, cv2.CV_64F).var()
    return laplacian_var < threshold

def middle_white_area_size(patch, shapeid, output_dir,
                           gaussian_kernel=(5,5), threshold_val=170,
                           min_size=2000, morph_kernel_size=5, open_iterations=2, erode_iterations=2,
                           distance_mask_size=5, h_max_value=1, dilate_iterations=3, h_max_thresh_multiplier=0.3):
    """
    Computes the size of the connected white area in the middle of a patch.
    
    Saves a mask image with the name '<shapeid>_mask.png' to output_dir.
    
    Parameters:
        patch (ndarray): Image patch.
        shapeid (int): Identifier for the patch (used for naming).
        output_dir (str): Directory to save the mask.
        gaussian_kernel (tuple): Kernel size for Gaussian blur.
        threshold_val (int): Threshold value for binarization.
        min_size (int): Minimum size for object removal.
        morph_kernel_size (int): Size for the morphological kernel.
        open_iterations (int): Iterations for morphological opening.
        erode_iterations (int): Iterations for erosion.
        distance_mask_size (int): Mask size for the distance transform.
        h_max_value (int): Parameter for h_maxima.
        dilate_iterations (int): Iterations for dilation.
        h_max_thresh_multiplier (float): Multiplier to set threshold on h_maxima result.
    
    Returns:
        int: The area (in pixels) of the connected white region.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    gray = 255 - hsv[:, :, 1]
    gray = cv2.GaussianBlur(gray, gaussian_kernel, 0)
    ret, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
    thresh_clean = 255 * morphology.remove_small_objects(thresh.astype(bool),
                                                         min_size=min_size, connectivity=4).astype(np.uint8)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    sure_bg = cv2.erode(opening, kernel, iterations=erode_iterations)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, distance_mask_size)
    h_max = h_maxima(dist_transform, h_max_value)
    h_max = cv2.dilate(h_max, kernel, iterations=dilate_iterations)
    ret, sure_fg = cv2.threshold(h_max, h_max_thresh_multiplier * h_max.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB), markers.astype(np.int32))
    bw = (markers > 1).astype(np.uint8) * 255

    height, width = patch.shape[:2]
    center = (width // 2, height // 2)
    stack = [center]
    count = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()

    while stack:
        x, y = stack.pop()
        if (x, y) not in visited and bw[y, x] == 255:
            visited.add((x, y))
            count += 1
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < width and 0 <= new_y < height:
                    stack.append((new_x, new_y))
    return count

def crop_and_save(complete_image, centroid, shapeid, output_dir, csv_writer, slide_id, 
                   size=96, blur_threshold=100.0, gamma=1.5, background_color=np.array([255, 255, 255], dtype=np.uint8)):
    """
    
    Work inspired by:
    References
    ----------
    .. [1] Manescu, P., Narayanan, P., Bendkowski, C. et al. Detection of 
    acute promyelocytic leukemia in peripheral blood and bone marrow with 
    annotation-free deep learning. Sci Rep 13, 2562 (2023).
    https://doi.org/10.1038/s41598-023-29160-4
    
    
    Crops a region from complete_image centered at centroid, applies quality checks,
    performs gamma correction and blending, then saves the image if conditions are met.
    
    Parameters:
        complete_image (ndarray): Full image array.
        centroid (tuple): Center coordinates for cropping.
        shapeid (int): Identifier for naming the saved patch.
        output_dir (str): Directory to save the cropped image.
        csv_writer (csv.writer): CSV writer to log the saved patch.
        slide_id (str): Identifier of the slide.
        size (int): Width and height of the patch.
        blur_threshold (float): Threshold to check for blurriness.
        gamma (float): Gamma value for correction.
        background_color (ndarray): Background color for blending.
    
    Returns:
        bool: True if patch was saved; otherwise False.
        
    """
    nx_0 = max(int(centroid[0] - size / 2), 0)
    ny_0 = max(int(centroid[1] - size / 2), 0)
    nx_1 = min(nx_0 + size, complete_image.shape[1])
    ny_1 = min(ny_0 + size, complete_image.shape[0])
    cropped_image = complete_image[ny_0:ny_1, nx_0:nx_1, :].astype(np.uint8)
    
    if (is_qualitative(cropped_image) and 
        (middle_white_area_size(cropped_image, shapeid, output_dir) > 850) and 
        not is_blurry(cropped_image, blur_threshold)):
        
        gamma_corrected_image = apply_gamma_correction(cropped_image, gamma=gamma)
        b, g, r, a = cv2.split(gamma_corrected_image)
        alpha = a.astype(float) / 255
        blended_image = np.zeros((gamma_corrected_image.shape[0], gamma_corrected_image.shape[1], 3), dtype=np.uint8)
        for c in range(3):
            blended_image[:, :, c] = (alpha * gamma_corrected_image[:, :, c] +
                                      (1 - alpha) * background_color[c]).astype(np.uint8)
        roi_file = os.path.join(output_dir, f'{shapeid}_40.png')
        cropped_image_pil = Image.fromarray(blended_image)
        if cropped_image_pil.size == (size, size):
            cropped_image_pil.save(roi_file)
            csv_writer.writerow([slide_id, f'{shapeid}_40.png', 40])
            return True
    return False

def cell_segmentation(img, gaussian_kernel=(5,5), threshold_val=127, min_size=2000, 
                         morph_kernel_size=5, morph_open_iterations=2, erode_iterations=2,
                         distance_mask_size=5, h_max_value=1, dilate_iterations=3, h_max_thresh_multiplier=0.3):
    """
    
    Work inspired by:
    References
    ----------
    .. [1] Manescu, P., Narayanan, P., Bendkowski, C. et al. Detection of 
    acute promyelocytic leukemia in peripheral blood and bone marrow with 
    annotation-free deep learning. Sci Rep 13, 2562 (2023).
    https://doi.org/10.1038/s41598-023-29160-4
    
    
    Performs white blood cell segmentation on the input image using HSV transformation
    and various morphological operations.
    
    Parameters:
        img (ndarray): Input image.
        gaussian_kernel (tuple): Kernel size for Gaussian blur.
        threshold_val (int): Threshold for binarization.
        min_size (int): Minimum object size for noise removal.
        morph_kernel_size (int): Kernel size for morphological operations.
        morph_open_iterations (int): Iterations for morphological opening.
        erode_iterations (int): Iterations for erosion.
        distance_mask_size (int): Mask size for the distance transform.
        h_max_value (int): Parameter for h_maxima.
        dilate_iterations (int): Iterations for dilation.
        h_max_thresh_multiplier (float): Multiplier for thresholding the h_maxima result.
    
    Returns:
        ndarray: Binary mask image resulting from segmentation.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = 255 - hsv[:, :, 1]
    gray = cv2.GaussianBlur(gray, gaussian_kernel, 0)
    ret, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
    thresh_clean = 255 * morphology.remove_small_objects(thresh.astype(bool),
                                                         min_size=min_size, connectivity=4).astype(np.uint8)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kernel, iterations=morph_open_iterations)
    sure_bg = cv2.erode(opening, kernel, iterations=erode_iterations)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, distance_mask_size)
    h_max = h_maxima(dist_transform, h_max_value)
    h_max = cv2.dilate(h_max, kernel, iterations=dilate_iterations)
    ret, sure_fg = cv2.threshold(h_max, h_max_thresh_multiplier * h_max.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), markers.astype(np.int32))
    bw = (markers > 1).astype(int)
    return 255 * bw.astype(np.uint8)

def process_roi(image, output_dir, csv_writer, slide_id, current_shapeid=0, 
                    max_cells_per_thumbnail=200, max_total_cells=2500, roi_name="roi_mask.png",
                    segmentation_params=None):
    """
    
    Work inspired by:
    References
    ----------
    .. [1] Manescu, P., Narayanan, P., Bendkowski, C. et al. Detection of 
    acute promyelocytic leukemia in peripheral blood and bone marrow with 
    annotation-free deep learning. Sci Rep 13, 2562 (2023).
    https://doi.org/10.1038/s41598-023-29160-4
    
    
    Generates thumbnails from the provided image by segmenting white blood cells,
    saving a mask, and cropping individual cell patches.
    
    Parameters:
        image (PIL.Image): Input slide region image.
        output_dir (str): Directory to save cropped cells.
        csv_writer (csv.writer): CSV writer to log saved patches.
        slide_id (str): Identifier of the slide.
        current_shapeid (int): Starting identifier for cells.
        max_cells_per_thumbnail (int): Maximum cells to extract from one thumbnail.
        max_total_cells (int): Maximum total cells to extract from the slide.
        roi_name (str): Name for the ROI mask image.
        segmentation_params (dict): Optional parameters for segmentation.
    
    Returns:
        int: Updated shapeid after processing.
    """
    shapeid = current_shapeid
    image_np = np.array(image)
    segmentation_params = segmentation_params or {}
    mp_masks = cell_segmentation(image_np, **segmentation_params)
    mask_path = os.path.join(output_dir, roi_name)
    Image.fromarray(mp_masks).save(mask_path)
    
    output = cv2.connectedComponentsWithStats(mp_masks, connectivity=8)
    centroids = output[3]
    saved_cells = 0
    for c in centroids:
        if saved_cells >= max_cells_per_thumbnail or shapeid >= max_total_cells:
            break
        if crop_and_save(image_np, c, shapeid, output_dir, csv_writer, slide_id):
            saved_cells += 1
            shapeid += 1
    return shapeid

def process_images(input_dir, output_dir, 
                        file_extension=".ndpi",
                        roi_size=4096, 
                        width_fraction=0.5, height_fraction=2/3,
                        max_cells_per_thumbnail=200, max_total_cells=2500,
                        segmentation_params=None):
    """
    Processes  slide images from input_dir, extracts regions of interest (ROIs),
    and generates single-cell thumbnails and a CSV log.
    
    Parameters:
        input_dir (str): Directory containing files.
        output_dir (str): Directory to save processed outputs.
        file_extension (str): File extension filter (default: ".ndpi").
        roi_size (int): Size of each ROI to extract.
        width_fraction (float): Fraction of slide width to process.
        height_fraction (float): Fraction of slide height to process.
        max_cells_per_thumbnail (int): Maximum cells to extract per ROI.
        max_total_cells (int): Maximum total cells to extract per slide.
        segmentation_params (dict): Optional segmentation parameters.
    """
    ndpi_files = [f for f in os.listdir(input_dir) if f.lower().endswith(file_extension)]
    print(f"Found files: {ndpi_files}")
    
    for ndpi_file in ndpi_files:
        print(f"Processing file: {ndpi_file}")
        base_name = os.path.splitext(ndpi_file)[0]
        csv_file_path = os.path.join(output_dir, f'{base_name}.csv')
        try:
            input_file = os.path.join(input_dir, ndpi_file)
            slide = OpenSlide(input_file)
            dimensions = slide.dimensions
            center_x, center_y = dimensions[0] // 2, dimensions[1] // 2
            grid_width = int((dimensions[0] * width_fraction) // roi_size)
            grid_height = int((dimensions[1] * height_fraction) // roi_size)
            shapeid = 0
            
            slide_dir = os.path.join(output_dir, base_name)
            os.makedirs(slide_dir, exist_ok=True)
            
            with open(csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['slide_id', 'idx', 'magnification'])
                
                for i in range(grid_height):
                    for j in range(grid_width):
                        if shapeid >= max_total_cells:
                            break
                        top_left_x = int(center_x - (grid_width // 2) * roi_size + j * roi_size)
                        top_left_y = int(center_y - (grid_height // 2) * roi_size + i * roi_size)
                        
                        roi_image = slide.read_region((top_left_x, top_left_y), 0, (roi_size, roi_size))
                        roi_filename = f"roi_{top_left_y}_{top_left_x}.png"                        
                        roi_image_rgba = roi_image.convert("RGBA")
                        shapeid = process_roi(roi_image_rgba, slide_dir, csv_writer, base_name, shapeid,
                                                   max_cells_per_thumbnail, max_total_cells, roi_filename,
                                                   segmentation_params=segmentation_params)
            zip_file_path = os.path.join(output_dir, f'{base_name}.zip')
            shutil.make_archive(base_name, 'zip', slide_dir)
            shutil.move(f'{base_name}.zip', zip_file_path)
            shutil.rmtree(slide_dir)
        except Exception as e:
            print(f"Error processing {ndpi_file}: {e}")


@hydra.main(version_base=None, config_path=cfgs_pth, config_name=None)
def main(cfg: DictConfig):
    print("Starting image processing with the following configuration:")
    print(cfg)
    process_cfg = cfg.process
    process_images(
        input_dir=process_cfg.input_dir,
        output_dir=process_cfg.output_dir,
        file_extension=process_cfg.file_extension,
        roi_size=process_cfg.roi_size,
        width_fraction=process_cfg.width_fraction,
        height_fraction=process_cfg.height_fraction,
        max_cells_per_thumbnail=process_cfg.max_cells_per_thumbnail,
        max_total_cells=process_cfg.max_total_cells,
        segmentation_params=process_cfg.segmentation_params
    )

    
if __name__ == "__main__":
    main()