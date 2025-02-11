# realm_task_3_3_implementation_xai

## Task Description

This repository implements an automated pipeline for 3D lung segmentation with integrated XAI (Explainable AI) capabilities, optimized for CT imaging analysis, utilizing DuneAI, and combining fuzzy logic-driven methodology selection with Grad-CAM visual explanations.

Key Components:

1. **Adaptive XAI Framework:**
   1. Implements Grad-CAM visualization with automatic layer selection through fuzzy logic. The `sensitivity [0-1]` variable determines the actual gradcam layer and threshold for the generation of masked slices.
   2. Sensitivity-based threshold adjustment using a fuzzy control system
   3. Batch processing of 3D volues with slice-wise Grad-CAM overlays for each slice (axial, sagittal and coronal)
2. **NCLSC-Radiomics Dataset Integration**
   1. Preprocesses the DICOM dataset and convert it to .nrrd files
   2. Uses the converted files to produce a segmentation mask and xai Grad-CAM results
   3. Outputs Original Slices, Segmented Slices and multi planar Grad-CAM slices for each patient, for each 3D dimension
   4. Saves output slices as .png images for easier visualization

## Prerequisites

1. In order for the packages inside `requirements.txt` to be installed successfully, you need to have
   https://rustup.rs/ installed and added to your Path variable. To do this, after installation,
   add `C:\Users\<YourUsername>\.cargo\bin` to your Path system variable.
2. Unzip `weights_v7.zip` in order to create the `weights_v7.hdf5` file.

## Create the `converted_nrrds` directory

We will use the `data_conversion.py` for this task.

1. _Optional: Uncomment lines 4-6 in the `data_conversion.py` file in order to download the dataset._
2. Define the `parameters` variable, as needed.
3. _Optional: Look into the description of the dataset using `get_dataset_description()` and perform a quality check
   using `get_quality_checks()`._
4. Run `toolbox.contert_to_nrrd(export_path)`. The output .nrrd files will be generated in the `./converted_nrrds`
   directory.

## Predicted segmentation mask, images and XAI report.

We will use `batch_segmentation.py` for this.

1. Modify `model_path`, `path_to_test_data` and `save_path` according to your needs.
2. Run the script in order to begin the segmentation and xai generation process.

**Example:** `python batch_segmentation.py --model_path ./model_files --path_to_test_data ./converted_nrrds --save_path ./output_segmentations --sensitivity 0.7`

## Fairness and Bias metrics

**Note:** Fairness and bias cannot be implemented for the current task, which focuses on generating a segmentation mask and a Grad-CAM heatmap.
Quantitative analysis of fairness and bias requires:

1. Subpopulations/subgroups to be detected on a model level.
2. Some patients to not have tumors detected, so the fairness/bias model can distinguish between combinations of positive or negative classifications on each subgroup
