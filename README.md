# realm_task_3_3_implementation_xai

## General Task Description

Components developed in Task 3.3 aim to implement agnostic XAI techniques on top of AI models that are used for various tasks such as classification or segmentation.
We aim to implement **two** XAI techniques per Use Case - that would be selected dynamically from the Fuzzy system based on User's Input (sensitivity value coming from the RIANA dashboard), implement bias and fairness metrics (as agreed [here](https://maastrichtuniversity.sharepoint.com/:w:/r/sites/FSE-REALM/_layouts/15/Doc.aspx?sourcedoc=%7B9EDAE561-2787-42D1-BBB8-C9320C0B1F25%7D&file=Report%20on%20Bias%20and%20Fairness%20Metrics%20%5BTask%203.3%5D.docx&action=default&mobileredirect=true)) based on model outputs and extract outputs in a digestible manner (images, metrics, etc.)

This component, no matter the Use Case, expects as input:
- Sensitivity value (RIANA dashboard)
- Trained model (AI Orchestrator)
- Compatible dataset (AI Orchestrator)

This component, no matter the Use Case, returns as output:
- XAI methodology output (depending on the Use Case - image or json file)
- Fairness and Bias results (depending on the Use Case - nothing if we are talking for images or json file)

## Use Case 1 Specific Description

This repository implements an automated pipeline for 3D lung segmentation with integrated XAI capabilities, optimized for CT imaging analysis, utilizing DuneAI, and combining fuzzy logic-driven methodology selection with Grad-CAM visual explanations. As we have a segmentation task, no Fairness or Bias values are calculated in the model's predictions.

Key Components:

1. **Adaptive XAI Framework:**
   1. Implements Grad-CAM visualization with automatic layer selection through fuzzy logic. The `sensitivity [0-1]` variable coming from the RIANA dashboard determines the actual gradcam layer and threshold for the generation of masked slices.
   2. Sensitivity-based threshold adjustment using a fuzzy control system
   3. Batch processing of 3D volumes with slice-wise Grad-CAM overlays for each slice (axial, sagittal and coronal)
2. **NCLSC-Radiomics Dataset Integration**
   1. Preprocesses the [DICOM dataset](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) and convert it to .nrrd files as this is the accepted input of the [DuneAI model](https://github.com/primakov/DuneAI-Automated-detection-and-segmentation-of-non-small-cell-lung-cancer-computed-tomography-images) we used
   2. Uses the converted files to produce a segmentation mask and xai Grad-CAM results
   3. Outputs Original Slices, Segmented Slices and multi planar Grad-CAM slices for each patient, for each 3D dimension
   4. Saves output slices as .png images for easier visualization

## Prerequisites

1. **Python version must be 3.8.x**
2. In order for the packages inside `requirements.txt` to be installed successfully, you need to have
   https://rustup.rs/ installed and added to your Path variable. To do this, after installation,
   add `C:\Users\<YourUsername>\.cargo\bin` to your Path system variable.
3. Unzip `weights_v7.zip` in order to create the `weights_v7.hdf5` file.

## Create the `converted_nrrds` directory

**Note: This step is optional. Skip this if the dataset is already in nrrd format.**

We will use the `data_conversion.py` for this task. The actual functions used in this `.py` file are in the `pmtool` directory and can be found [here](https://github.com/primakov/precision-medicine-toolbox/tree/master/pmtool) 

1. _Optional: Uncomment lines 4-6 in the `data_conversion.py` file in order to download the dataset, if it is not pre-downloaded_
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

## Docker Instructions

### Create the image

To create the image we navigate to the root folder of the repo and execute: `docker build -t my_image .`

### Run the image

To run the created image run: `docker run lung_segmentation`. If you want to modify the run parameters, such as `model_path` or `sensitivity`, you can use `--model_path <path>` or `--sensitivity <value>`.
Default values can be found in the `Dockerfile`

## Kubeflow Components for Lung Segmentation

The `kubeflow_components/lung_segmentation_component.py` file contains Kubeflow pipeline creation logic for the components of the lung segmentation and XAI visualization, enabling seamless deployment in cloud environments.

### Pipeline Explanation

The pipeline consists of two main components:
1. **Data Preparation Component:** Downloads model files, weights, and patient data from a GitHub repository
2. **Lung Segmentation & XAI Component:** Performs lung segmentation and generates XAI visualizations using Grad-CAM

The pipeline automatically handles dependencies, performs data preprocessing, and generates segmentation results with explainable AI overlays based on user-defined sensitivity parameters.

### Pipeline Usage

Compile the pipeline to generate a deployable YAML file in the root directory: `python kubeflow_components/lung_segmentation_component.py` This will create `lung_segmentation_pipeline.yaml` which can be uploaded to a Kubeflow Pipelines instance.

### Pipeline Parameters

When running the pipeline, you need to specify:
* **github_repo_url:** URL of the repository containing model files and data (e.g. https://github.com/Metamind-Innovations/realm_task_3_3_implementation_xai_uc1)
* **sensitivity:** Value between 0.0-1.0 controlling segmentation sensitivity (default: `0.7`)
* **verbosity:** Enable/disable verbose logging (default: `True`)
* **branch:** Git branch to use (default: `main`)

### Resource Requirements

The pipeline is configured with the following resource requests and limits:

* **CPU:** 2-4 cores
* **Memory:** 4-8 GB

Adjust these values in the pipeline definition for your specific environment requirements.

### Accessing the Generated Artifacts

The pipeline stores generated artifacts in MinIO object storage within the Kubeflow namespace. To access these artifacts:

1. Set up port forwarding to the MinIO service by running `kubectl port-forward -n kubeflow svc/minio-service 9000:9000` in a terminal window
2. Access the MinIO web interface at `http://localhost:9000`
3. Log in with the default credentials 
   1. Username: `minio` 
   2. Password: `minio123`
4. Navigate to the `mlpipeline` bucket, where you'll find the folders `download-github-files` and `lung-segmentation`, containing the artifacts generated from the respective steps.

## 📜 License & Usage

All rights reserved by MetaMinds Innovations. 
