import kfp
from kfp.dsl import Dataset, Input, Model, Output, Artifact
import os

@kfp.dsl.component(
    base_image="python:3.8-slim",
    target_image="lung-segmentation:v1",
    packages_to_install=[
        "fairlearn",
        "imbalanced_learn==0.12.4",
        "keras",
        "matplotlib==3.7.5",
        "numpy",
        "opencv_python==4.11.0.86",
        "pandas==2.0.3",
        "Pillow",
        "plotly==6.0.0",
        "pydicom",
        "pyradiomics==3.1.0",
        "reportlab==4.3.0",
        "scikit-fuzzy==0.5.0",
        "scikit-learn",
        "scipy",
        "seaborn==0.13.2",
        "SimpleITK==2.4.1",
        "scikit-image",
        "statsmodels==0.14.1",
        "tensorflow==2.10.0",
        "protobuf<3.20",
        "tqdm==4.67.1"
    ]
)
def lung_segmentation(
    model_files: Input[Model],
    input_data: Input[Dataset],
    sensitivity: float,
    output_results: Output[Dataset]
):
    """
    Perform lung segmentation and generate XAI visualizations using Grad-CAM
    
    Args:
        model_files (Input[Model]): Directory containing model_v7.json and weights files
        input_data (Input[Dataset]): Directory containing input NRRD files
        sensitivity (float): Sensitivity parameter (0.0-1.0) for detection
        output_results (Output[Dataset]): Directory for output segmentations and visualizations
    """
    import subprocess
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "ffmpeg", "libsm6", "libxext6", "-y"])
    from segmentation_xai.TheDuneAI import ContourPilot
    model = ContourPilot(
        model_path=model_files.path,
        data_path=input_data.path,
        output_path=output_results.path,
        sensitivity=sensitivity,
        verbosity=True
    )
    model.segment()

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        lung_segmentation,
        "lung_segmentation_component.yaml"
    )