# MULTICLASS SEMANTIC SEGMENTATION

## Project Overview

This project implements multiclass semantic segmentation for vegetation analysis using U-Net with a ResNet backbone. The input consists of multispectral images, and the task involves classifying each pixel into one of four categories: background, vegetation, sick parts, and spots.

## Data

- **Input Images:** 
  - 5-channel TIF files containing [R, G, B, RedEdge, NIR] bands.
  
- **Classes:**
  - 0: Background
  - 1: Vegetation
  - 2: Sick parts
  - 3: Spots

- **Normalization:**
  - Min-Max scaling is applied to normalize the input data.

## Preprocessing

- **Patch Extraction:**
  - Patches of size 128x128 are extracted from the original images with stride 50, using center rotation to augment the dataset.
  - Masks are preprocessed to map RGB values to corresponding class labels.
  - Extracted patches are saved as `.npy` files for model training.
  
- **Note:** Validation on the test data without extracting patches is still under consideration.

## Model

- **Architecture:** 
  - U-Net with a ResNet18 backbone.
  
- **Customizable Parameters:**
  - Flexible encoder depth and decoder channels.
  
- **Pretrained Weights:** 
  - ImageNet weights for the encoder.

- **Class Imbalance Handling:**
  - Uses class weights based on class frequency for loss computation.

## Training

- **Loss Function:** 
  - Weighted cross-entropy loss to handle class imbalance.

- **Optimizer:** 
  - Adam optimizer with a learning rate scheduler (`ReduceLROnPlateau`).
  
- **Early Stopping:**
  - Stops training when validation loss does not improve for 10 epochs.

- **Checkpointing:**
  - Saves the best model based on validation loss, with a history of the top 5 models.

## TODOs

- [ ] Prepare requirements for this repo.
- [ ] Add config file for more flexibility instead of hard-coded values.
- [x] Complete `inference.py` and save quantitative and qualitative results.
- [ ] Implement validation on the test dataset without patch extraction.
- [ ] Fine-tune hyperparameters (learning rate, batch size, smoothing rate, etc.) for better performance. Try Optuna.
- [ ] Perform K-Fold Cross Validation.
- [ ] Add support for other architectures (e.g., DeepLabV3+).
- [ ] Try Focal Loss.