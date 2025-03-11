import logging
import os
from typing import Optional

import kfp
from kfp.dsl import Dataset, Input, Model, Output


@kfp.dsl.component(
    base_image="python:3.8-slim",
    target_image="lung-segmentation:v1",
    packages_to_install=[
        "keras==2.10.0",
        "matplotlib==3.7.5",
        "numpy==1.24.3",
        "opencv-python==4.11.0.86",
        "pandas==2.0.3",
        "plotly==6.0.0",
        "pydicom",
        "pyradiomics==3.1.0",
        "scikit-fuzzy==0.5.0",
        "scikit-learn==1.3.2",
        "scipy",
        "seaborn==0.13.2",
        "SimpleITK==2.4.1",
        "statsmodels==0.14.1",
        "tensorflow==2.10.0",
        "tqdm==4.67.1",
        "protobuf<3.20",
    ]
)
def lung_segmentation(
        model_files: Input[Model],
        input_data: Input[Dataset],
        sensitivity: float,
        output_results: Output[Dataset],
        verbosity: bool = True,
        batch_size: Optional[int] = 1,
        window_width: Optional[int] = 1500,
        window_center: Optional[int] = -600,
):
    """
    Perform lung segmentation and generate XAI visualizations using Grad-CAM

    Args:
        model_files (Input[Model]): Directory containing model_v7.json and weights_v7.hdf5
        input_data (Input[Dataset]): Directory containing input NRRD files
        sensitivity (float): Sensitivity parameter (0.0-1.0) for detection
        output_results (Output[Dataset]): Directory for output segmentations and visualizations
        verbosity (bool): Enable/disable verbose output
        batch_size (int, optional): Batch size for processing
        window_width (int, optional): CT window width parameter
        window_center (int, optional): CT window center parameter

    Returns:
        Output containing segmentation results and XAI visualizations
    """
    # Configure logging
    logging_level = logging.INFO if verbosity else logging.WARNING
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('lung_segmentation')

    # Validate input parameters
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError(f"Sensitivity must be between 0.0 and 1.0, got {sensitivity}")

    if not os.path.exists(model_files.path):
        raise FileNotFoundError(f"Model path does not exist: {model_files.path}")

    model_json_path = os.path.join(model_files.path, 'model_v7.json')
    model_weights_path = os.path.join(model_files.path, 'weights_v7.hdf5')

    if not os.path.exists(model_json_path) or not os.path.exists(model_weights_path):
        raise FileNotFoundError(
            f"Required model files not found in {model_files.path}. Need model_v7.json and weights_v7.hdf5")

    if not os.path.exists(input_data.path):
        raise FileNotFoundError(f"Input data path does not exist: {input_data.path}")

    # Install system dependencies
    logger.info("Installing system dependencies...")
    try:
        import subprocess
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "ffmpeg", "libsm6", "libxext6", "-y"], check=True)
    except subprocess.SubprocessError as e:
        logger.warning(f"Error installing system dependencies: {e}")
        logger.warning("Continuing anyway, but this might cause issues later")

    # Create output directory if it doesn't exist
    os.makedirs(output_results.path, exist_ok=True)

    # Import after dependencies are installed
    try:
        logger.info("Importing required modules...")
        from segmentation_xai.TheDuneAI import ContourPilot
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        raise

    # Log runtime parameters
    logger.info("Starting lung segmentation with parameters:")
    logger.info(f"  - Model path: {model_files.path}")
    logger.info(f"  - Input data path: {input_data.path}")
    logger.info(f"  - Output path: {output_results.path}")
    logger.info(f"  - Sensitivity: {sensitivity}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Window parameters: width={window_width}, center={window_center}")

    try:
        # Initialize model
        logger.info("Initializing ContourPilot model...")
        model = ContourPilot(
            model_path=model_files.path,
            data_path=input_data.path,
            output_path=output_results.path,
            sensitivity=sensitivity,
            verbosity=verbosity
        )

        # Run segmentation
        logger.info("Starting segmentation process...")
        model.segment()
        logger.info("Segmentation completed successfully")

        # Save metadata about the run
        with open(os.path.join(output_results.path, 'metadata.txt'), 'w') as f:
            f.write("Lung Segmentation XAI Pipeline\n")
            f.write(f"Sensitivity: {sensitivity}\n")
            f.write(f"Window parameters: width={window_width}, center={window_center}\n")
            f.write(f"Batch size: {batch_size}\n")

        # Return success
        return True

    except Exception as e:
        logger.error(f"Error during segmentation: {e}", exc_info=True)
        with open(os.path.join(output_results.path, 'error_log.txt'), 'w') as f:
            f.write(f"Error during segmentation: {str(e)}\n")
        raise


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        lung_segmentation,
        "lung_segmentation_component.yaml"
    )
