import os
import time
import gc
from astropy.io import fits
import numpy as np
import cv2

# Base directory (Samples)
base_directory = r"C:\Users\lsann\Desktop\CroppingFolder"

# Output directory (cropped_cubes)
output_directory = os.path.join(base_directory, 'cropped_cubes')

# Ensure the output subdirectory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to process FITS files in a directory
def process_fits_in_directory(directory, output_directory):
    fits_file_real = None
    fits_file_simulated = None

    # Extract the name of the current subdirectory
    subdir_name = os.path.basename(directory)
    
    # Search for real and simulated FITS files
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            if 'dirty-cube' in filename:  # Identify simulated FITS files
                fits_file_simulated = os.path.join(directory, filename)
            else:  # Any other FITS file is considered a real FITS file
                fits_file_real = os.path.join(directory, filename)
    
    # If no real FITS file found, skip the directory
    if not fits_file_real:
        print(f"No real FITS file found in {directory}. Skipping directory.")
        return
    
    # Output file paths
    cropped_file_real = os.path.join(output_directory, f'cropped_{subdir_name}.fits')
    adjusted_fits_file_simulated = os.path.join(output_directory, f'cropped_{subdir_name}_dirty_cube.fits')
    temp_adjusted_fits_file_simulated = os.path.join(output_directory, 'temp_adjusted_simulated.fits')  # Temp file

    # Check if the simulated cube file exists
    simulated_exists = fits_file_simulated is not None

    # Load the real cube FITS file
    with fits.open(fits_file_real) as hdul_real:
        data_real = hdul_real[0].data
        header_real = hdul_real[0].header

    print(f"Processing real FITS file: {fits_file_real}")
    print("Real data shape:", data_real.shape)

    # Load the simulated cube FITS file if it exists
    if simulated_exists:
        with fits.open(fits_file_simulated) as hdul_simulated:
            data_simulated = hdul_simulated[0].data
        print(f"Processing simulated FITS file: {fits_file_simulated}")
        print("Simulated data shape:", data_simulated.shape)

    # Function to compute the number of channels and adjust the simulated data
    def adjust_simulated_channels(data_real, data_simulated):
        num_channels_real = data_real.shape[1] if len(data_real.shape) > 2 else 1
        num_channels_simulated = data_simulated.shape[0] if len(data_simulated.shape) > 2 else 1
        
        if num_channels_simulated > num_channels_real:
            difference = num_channels_simulated - num_channels_real

            if difference > 0:
                if difference % 2 == 0:
                    num_remove_start = difference // 2
                    num_remove_end = difference // 2
                else:
                    num_remove_start = difference // 2 + 1
                    num_remove_end = difference // 2
            
            data_simulated = data_simulated[num_remove_start:num_channels_simulated - num_remove_end, :, :]
        
        return data_simulated

    # If the simulated file exists, adjust the simulated channels to match the real cube
    if simulated_exists:
        data_simulated_adjusted = adjust_simulated_channels(data_real, data_simulated)
        print("Adjusted simulated data shape:", data_simulated_adjusted.shape)

    # Function to calculate the center coordinates (x, y)
    def calculate_center(data_slice):
        center_y = data_slice.shape[0] // 2
        center_x = data_slice.shape[1] // 2
        return center_x, center_y

    # Updated function to find boundary points using Canny edge detection with error handling
    def find_boundary_points(data_slice, threshold1=50, threshold2=150):
        normalized_slice = cv2.normalize(data_slice, None, 0, 255, cv2.NORM_MINMAX)
        img = np.uint8(normalized_slice)
        edges = cv2.Canny(img, threshold1, threshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:  # If no contours were found
            return None  # Return None to indicate that no boundary points were found
        
        contour = max(contours, key=cv2.contourArea)
        boundary_points = [(point[0][0], point[0][1]) for point in contour]
        return boundary_points

    # Function to crop a square from the center of the circle
    def crop_center_square(data_slice, center_x, center_y, side_length):
        half_side = side_length // 2
        start_x = max(center_x - half_side, 0)
        end_x = min(center_x + half_side, data_slice.shape[1])
        start_y = max(center_y - half_side, 0)
        end_y = min(center_y + half_side, data_slice.shape[0])
        cropped_data = data_slice[start_y:end_y, start_x:end_x]
        return cropped_data

    # Scan all slices to collect valid boundary points and remove empty slices
    valid_slices = []
    valid_boundary_points = []
    valid_indices = []
    num_slices = data_real.shape[1] if data_real.ndim == 4 else data_real.shape[0]
    
    for i in range(num_slices):
        if data_real.ndim == 3:
            slice_data = data_real[i, :, :]
        elif data_real.ndim == 4:
            slice_data = data_real[0, i, :, :]
        
        boundary_points = find_boundary_points(slice_data)
        if boundary_points is not None:
            valid_slices.append(slice_data)  # Keep this slice
            valid_boundary_points.extend(boundary_points)
            valid_indices.append(i)  # Keep track of valid slice indices
    
    if not valid_slices:
        print(f"No valid contours found in any slices of the FITS file: {fits_file_real}")
        return

    # Convert the list of valid slices back to a numpy array
    valid_slices = np.array(valid_slices)

    # Also remove empty slices from the simulated data, if it exists
    if simulated_exists:
        data_simulated_adjusted = data_simulated_adjusted[valid_indices, :, :]

    # Calculate the center based on the first valid slice with contours
    center_x, center_y = calculate_center(valid_slices[0])

    # Calculate the average radius and side length based on all valid boundary points
    average_radius = np.mean([np.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2) for point in valid_boundary_points])
    side_length = int((2 * average_radius) / np.sqrt(2))

    print("Center (x, y):", (center_x, center_y))
    print("Average radius:", average_radius)
    print("Side length of the square:", side_length)

    # Crop all valid slices of the real data
    cropped_slices_real = np.array([crop_center_square(slice_data, center_x, center_y, side_length) for slice_data in valid_slices])

    cropped_data_real = cropped_slices_real
    print("Cropped real data shape:", cropped_data_real.shape)

    # Update header with new dimensions for the real data
    header_real['NAXIS1'] = cropped_data_real.shape[2]
    header_real['NAXIS2'] = cropped_data_real.shape[1]
    header_real['NAXIS3'] = cropped_data_real.shape[0]

    # Save the cropped real data to a new FITS file with the updated header
    cropped_hdu_real = fits.PrimaryHDU(cropped_data_real, header=header_real)
    cropped_hdul_real = fits.HDUList([cropped_hdu_real])

    # Ensure the output file can be written without permission issues
    if os.path.exists(cropped_file_real):
        os.remove(cropped_file_real)

    cropped_hdul_real.writeto(cropped_file_real, overwrite=True)

    print(f"Cropped real FITS file saved at: {cropped_file_real}")

    # Now process the adjusted simulated data, ensuring all file handles are properly closed
    if simulated_exists:
        # Crop all slices of the adjusted simulated data using the same dimensions
        cropped_slices_simulated = np.array([crop_center_square(data_simulated_adjusted[i, :, :], center_x, center_y, side_length) for i in range(data_simulated_adjusted.shape[0])])

        cropped_data_simulated = cropped_slices_simulated
        print("Cropped adjusted simulated data shape:", cropped_data_simulated.shape)

        # Update header with new dimensions for the simulated data
        header_simulated_adjusted = fits.Header()
        header_simulated_adjusted['NAXIS'] = 3
        header_simulated_adjusted['NAXIS1'] = cropped_data_simulated.shape[2]
        header_simulated_adjusted['NAXIS2'] = cropped_data_simulated.shape[1]
        header_simulated_adjusted['NAXIS3'] = cropped_data_simulated.shape[0]

        # Write the adjusted simulated data to a temporary FITS file
        with fits.HDUList([fits.PrimaryHDU(cropped_data_simulated, header=header_simulated_adjusted)]) as hdul_temp:
            hdul_temp.writeto(temp_adjusted_fits_file_simulated, overwrite=True)

        # Explicitly delete variables holding data and force garbage collection
        del data_real, data_simulated, data_simulated_adjusted, cropped_slices_real, cropped_slices_simulated
        gc.collect()

        # Rename the temporary file to the final output file
        if os.path.exists(adjusted_fits_file_simulated):
            retries = 5
            for i in range(retries):
                try:
                    os.remove(adjusted_fits_file_simulated)
                    break  # If successful, exit the loop
                except PermissionError:
                    if i < retries - 1:  # Avoid sleeping on the last attempt
                        print(f"PermissionError: Could not remove the file. Retrying... ({i + 1}/{retries})")
                        time.sleep(5)  # Increase sleep time to 5 seconds before retrying
                    else:
                        raise  # Re-raise the exception if all retries fail

        # Use os.replace instead of os.rename
        os.replace(temp_adjusted_fits_file_simulated, adjusted_fits_file_simulated)

        print(f"Cropped adjusted simulated FITS file saved at: {adjusted_fits_file_simulated}")


# Traverse the base directory and process FITS files in the first level of subdirectories only
for root, dirs, files in os.walk(base_directory):
    # Only process directories that are direct subdirectories of the base directory
    if root == base_directory:
        for subdir in dirs:
            subdir_path = os.path.join(base_directory, subdir)
            if subdir != 'cropped_cubes':
                process_fits_in_directory(subdir_path, output_directory)
