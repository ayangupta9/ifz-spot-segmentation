import glob
import os
import random
import cv2
import numpy as np
from skimage.io import imread
import math
from scipy import ndimage
import glob
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import gaussian_filter
from skimage.transform import rotate
import torch
from tqdm.notebook import trange


def get_sample_dict(file_pattern, suffix):
    """Create a dictionary mapping sample names to file paths."""
    files = sorted(glob.glob(file_pattern))
    return {os.path.basename(f).replace(suffix, ''): f for f in files}



def read_mask(mask_path, ohe=False):
    rgb_to_class = {
        (0, 0, 0): 0,     # Black -> soil (background)
        (0, 255, 0): 1,   # Green -> vegetation
        (255, 255, 255): 0,   # white -> extra
        (255, 0, 0): 2,   # Red -> sick
    }

    mask = imread(mask_path)[..., :3]  # Only take the first 3 channels (RGB)
    reshaped_mask = mask.reshape(-1, mask.shape[-1])  # Flatten mask
    
    multiclass_mask = np.array([rgb_to_class.get(tuple(rgb), 0) for rgb in reshaped_mask])
    
    multiclass_mask = multiclass_mask.reshape(mask.shape[0], mask.shape[1])
    sick = np.uint8(multiclass_mask == 2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sick)

    channels = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= 7:
            component_mask = np.uint8(labels == i)
            channels.append(component_mask)

    if channels:
        channels = np.dstack(channels)[:,:,None]
    else:
        channels = np.zeros_like(sick)
    
    if np.sum(channels) == 0:
        new_mask = multiclass_mask
    
    else:
        new_mask = multiclass_mask + np.sum(channels, axis=-1).squeeze()

    if ohe:
        ohe_mask = np.eye(4)[new_mask]
        return np.rot90(ohe_mask, k=-1)
    
    return np.rot90(new_mask, k=-1)


def normalize(img):
    im_min = np.nanpercentile(img.flatten(), 0.5)  # modify these percentiles to adjust contrast
    im_max = np.nanpercentile(img.flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values

    def operation(channel, min_val, max_val):
        eps = 1e-7
        return (channel - min_val) / ((max_val - min_val) + eps)

    if math.isnan(im_max):
        pass

    for i in np.arange(3):
        img[i] =  operation(img[i], im_min, im_max)

    return img


def min_max_norm(img):
    return (img - np.min(img, axis=(0,1)))/(np.max(img, axis=(0,1)) - np.min(img, axis=(0,1)))


def check_nir(b4,b5,mask=None):
    if mask != None:    
        i,j = np.where(mask == 1)
        b4_mean = b4[i,j].mean()
        b5_mean = b5[i,j].mean()
    else:
        b4_mean = b4.mean()
        b5_mean = b5.mean()
        
    return (b4, b5) if b4_mean > b5_mean else (b5, b4)


def read_band(ds,idx):
    # return min_max_norm(ds.read(idx))
    return min_max_norm(ds.read(idx))


def read_data(path,mask_p=None):
    with rasterio.open(path) as dataset:
        B,G,R, thermal = read_band(dataset, 1), read_band(dataset, 2), read_band(dataset, 3), read_band(dataset, 6)
        
        band4, band5 = read_band(dataset, 4), read_band(dataset, 5)
        # tag4, tag5 = dataset.tags(4), dataset.tags(5)

        if mask_p !=None:
            mask = read_mask(mask_p)
        else:
            mask = None
        # mask = np.rot90(mask, k=-1)
        nir, rededge = check_nir(band4, band5, mask)

        return np.dstack([R,G,B,rededge,nir])
    
    
def extract_patches_with_center_rotation(img, patch_size=100, stride=50, min_angle=15, max_angle=345, mask=None):
    """
    Extracts patches from an image by rotating the image around the patch center and then extracting the patch.
    If a mask is provided, the same transformations are applied to the mask.

    Parameters:
    - img: Input image as a NumPy array.
    - patch_size: Size of the square patch.
    - stride: Stride for sliding the window.
    - min_angle: Minimum rotation angle in degrees.
    - max_angle: Maximum rotation angle in degrees.
    - mask: Optional mask image as a NumPy array (must be the same size as img).

    Returns:
    - image_patches: List of extracted image patches.
    - mask_patches: List of extracted mask patches (if mask is provided), otherwise None.
    """
    height, width = img.shape[:2]

    if mask is not None:
        if mask.shape[:2] != (height, width):
            raise ValueError("Mask must have the same dimensions as the image.")
        # Ensure mask is single-channel
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    image_patches = []
    mask_patches = [] if mask is not None else None

    # Calculate the number of patches along x and y axes
    num_patches_y = (height - patch_size) // stride + 1
    num_patches_x = (width - patch_size) // stride + 1

    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the center of the patch
            y = i * stride
            x = j * stride
            center_x = x + patch_size // 2
            center_y = y + patch_size // 2

            # Randomly select a rotation angle
            angle = np.random.uniform(min_angle, max_angle)

            # Compute rotation matrix to rotate the image around the patch center
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

            # Apply the rotation to the image
            rotated_img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Apply the rotation to the mask if provided
            if mask is not None:
                rotated_mask = cv2.warpAffine(mask, M, (width, height), flags=cv2.INTER_NEAREST,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Extract the patch from the rotated image
            x_start = x
            x_end = x + patch_size
            y_start = y
            y_end = y + patch_size

            # Check if the patch is within image bounds
            if x_start >= 0 and x_end <= width and y_start >= 0 and y_end <= height:
                # Ensure that the rotated image has valid data in the patch area
                patch = rotated_img[y_start:y_end, x_start:x_end]
                # Check if the patch contains any border pixels (i.e., black pixels introduced due to rotation)
                if np.any(patch == 0):
                    # If there are border pixels, skip rotation
                    patch = img[y_start:y_end, x_start:x_end].copy()
                    if mask is not None:
                        mask_patch = mask[y_start:y_end, x_start:x_end].copy()
                else:
                    if mask is not None:
                        mask_patch = rotated_mask[y_start:y_end, x_start:x_end]
                image_patches.append(patch)
                if mask is not None:
                    mask_patches.append(mask_patch)
            else:
                # If out of bounds, extract the patch from the original image
                patch = img[y_start:y_end, x_start:x_end].copy()
                image_patches.append(patch)
                if mask is not None:
                    mask_patch = mask[y_start:y_end, x_start:x_end].copy()
                    mask_patches.append(mask_patch)

    return image_patches, mask_patches


def rotate_crop(img):
    corrected_image = rotate(img, angle=-13.25, resize=True)

    # Find the rows and columns where the values are non-zero
    non_zero_rows = np.where(np.any(corrected_image != 0, axis=1))[0]
    non_zero_cols = np.where(np.any(corrected_image != 0, axis=0))[0]

    # Get the first and last non-zero rows and columns
    row_start, row_end = non_zero_rows[0], non_zero_rows[-1]
    col_start, col_end = non_zero_cols[0], non_zero_cols[-1]

    # Crop the image using the identified bounds
    cropped_image = corrected_image[row_start:row_end + 1, col_start:col_end + 1]
    return cropped_image


def get_final_data_paths():
    patterns_suffixes = [
        ('./corrected_masks/*.png', '_cor.png'),
        ('./multiclass_masks/*.png', '_multiclass.png'),
        ('../facundo_dataset/tif_files/*', '_ML_DEM_total.tif'),
        ('./irs_corrected/*.png', '_cor.png'),
        ('./irs_input_imgs/*.tif', '_ML_DEM_total.tif'),
    ]

    # Generate sample dictionaries
    ifz_corr_masks = get_sample_dict(*patterns_suffixes[0])
    ifz_multiclass_masks = get_sample_dict(*patterns_suffixes[1])
    ifz_input_images = get_sample_dict(*patterns_suffixes[2])
    irs_corr_masks = get_sample_dict(*patterns_suffixes[3])
    irs_input_images = get_sample_dict(*patterns_suffixes[4])

    # Merge IFZ masks, preferring corrected masks over multiclass masks
    ifz_masks = {**ifz_multiclass_masks, **ifz_corr_masks}

    # Find common samples with both images and masks
    ifz_common_samples = set(ifz_masks) & set(ifz_input_images)
    irs_common_samples = set(irs_corr_masks) & set(irs_input_images)

    # Create final lists of images and masks
    final_train_images = [ifz_input_images[sample] for sample in sorted(ifz_common_samples)] + [irs_input_images[sample] for sample in sorted(irs_common_samples)]
    final_masks = [ifz_masks[sample] for sample in sorted(ifz_common_samples)] + [irs_corr_masks[sample] for sample in sorted(irs_common_samples)]
    
    return final_train_images, final_masks    


    
def perform_stitching(data, test_preds):
    full_height, full_width, num_classes = data.shape[0], data.shape[1], 4
    full_image = torch.zeros((full_height, full_width, num_classes), dtype=torch.float32)
    weight_sum = torch.zeros((full_height, full_width, 1), dtype=torch.float32)

    base_weight_mask_np = np.zeros((128, 128))
    base_weight_mask_np[64, 64] = 1
    base_weight_mask_np = gaussian_filter(base_weight_mask_np, sigma=32)
    base_weight_mask = torch.tensor(base_weight_mask_np, dtype=torch.float32)

    patch_size, stride = 128, 64
    patch_index = 0

    for y in range(0, full_height, stride):
        for x in range(0, full_width, stride):
            patch = test_preds[patch_index]

            weight_mask = base_weight_mask.clone()  # Ensure a fresh copy for each patch
            if y + patch_size > full_height:
                weight_mask[(full_height - y):, :] = 0
            if x + patch_size > full_width:
                weight_mask[:, (full_width - x):] = 0

            weighted_patch = patch * weight_mask[..., None]

            y_end = min(y + patch_size, full_height)
            x_end = min(x + patch_size, full_width)

            # Accumulate predictions and weights
            full_image[y:y_end, x:x_end] = full_image[y:y_end, x:x_end] + weighted_patch[:y_end - y, :x_end - x]
            weight_sum[y:y_end, x:x_end] = weight_sum[y:y_end, x:x_end] + weight_mask[:y_end - y, :x_end - x, None]

            patch_index += 1
    
    full_image /= torch.maximum(weight_sum, torch.tensor(1.0))
    final_prediction = torch.argmax(full_image, dim=-1)
    return final_prediction.numpy()



if __name__ == "__main__":
    final_train_images, final_masks = get_final_data_paths()

    image_patches, mask_patches = [],[]

    for i in trange(len(final_masks)):
        mask = read_mask(final_masks[i])
        data = read_data(final_train_images[i], final_masks[i])

        img_p, mask_p = extract_patches_with_center_rotation(img=data, mask=mask, patch_size=128, stride=50, min_angle=15, max_angle=345)

        image_patches = [*image_patches, *img_p]
        mask_patches = [*mask_patches, *mask_p]


    image_patches_np = np.array(image_patches) 
    mask_patches_np = np.array(mask_patches)

    np.save('./mask_patches_np.npy', mask_patches_np)
    np.save('./image_patches_np.npy', image_patches_np)
    
    
