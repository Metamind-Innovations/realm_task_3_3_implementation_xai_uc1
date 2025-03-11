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

    os.makedirs(output_results.path, exist_ok=True)

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


@kfp.dsl.component(
    base_image="python:3.8-slim"
)
def download_github_files(
        github_repo_url: str,
        model_files_path: str,
        input_data_path: str,
        model_files: Output[Model],
        input_data: Output[Dataset]
):
    import os
    import time
    import urllib.request

    os.makedirs(model_files.path, exist_ok=True)
    os.makedirs(input_data.path, exist_ok=True)
    github_repo_url = github_repo_url.strip()
    print(f"Using cleaned URL: '{github_repo_url}'")

    # Example: https://github.com/Metamind-Innovations/realm_task_3_3_implementation_xai_uc1
    repo_parts = github_repo_url.split('/')
    if len(repo_parts) >= 5:
        user_name = repo_parts[-2]
        repo_name = repo_parts[-1]

        files_to_download = [
            # Model files
            {"url": f"https://raw.githubusercontent.com/{user_name}/{repo_name}/main/{model_files_path}/model_v7.json",
             "dest": os.path.join(model_files.path, "model_v7.json")},

            # Direct LFS download for weights
            {"url": f"https://github.com/{user_name}/{repo_name}/raw/main/{model_files_path}/weights_v7.hdf5",
             "dest": os.path.join(model_files.path, "weights_v7.hdf5")}
        ]

        # Try to download each file with retries
        for file_info in files_to_download:
            url = file_info["url"]
            dest = file_info["dest"]
            print(f"Downloading {url} to {dest}")

            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(url, dest)
                    print(f"Successfully downloaded {dest}")
                    break
                except Exception as e:
                    print(f"Download attempt {attempt + 1} failed: {str(e)}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"Failed to download {url} after 3 attempts")

        with open(os.path.join(input_data.path, "sample.txt"), "w") as f:
            f.write("This is a placeholder for NRRD data files.\n")
            f.write("For a real application, download actual data files from the repository.")
    else:
        raise ValueError(f"Invalid GitHub URL format: {github_repo_url}")

    print("Model files:")
    print(os.listdir(model_files.path))

    print("Input data:")
    print(os.listdir(input_data.path))


@kfp.dsl.pipeline(
    name="Lung Segmentation Pipeline",
    description="Pipeline for lung segmentation and XAI visualization"
)
def lung_segmentation_pipeline(
        github_repo_url: str,
        model_files_path: str = "model_files",
        input_data_path: str = "converted_nrrds",
        sensitivity: float = 0.7,
        verbosity: bool = True,
        batch_size: int = 1,
        window_width: int = 1500,
        window_center: int = -600
):
    # Download files from GitHub
    download_task = download_github_files(
        github_repo_url=github_repo_url,
        model_files_path=model_files_path,
        input_data_path=input_data_path
    )

    lung_segmentation_task = lung_segmentation(
        model_files=download_task.outputs["model_files"],
        input_data=download_task.outputs["input_data"],
        sensitivity=sensitivity,
        verbosity=verbosity,
        batch_size=batch_size,
        window_width=window_width,
        window_center=window_center
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=lung_segmentation_pipeline,
        package_path="lung_segmentation_pipeline.yaml"
    )
