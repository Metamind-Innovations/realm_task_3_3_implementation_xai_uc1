import kfp
from kfp.dsl import pipeline
from lung_segmentation_component import lung_segmentation

@kfp.dsl.pipeline(
    name="Lung Segmentation Pipeline",
    description="Pipeline for lung segmentation and XAI visualization"
)
def lung_segmentation_pipeline(
    model_files: str,
    input_data: str,
    sensitivity: float = 0.9
):
    """
    Example pipeline showing how to use the lung segmentation component
    
    Args:
        model_files (str): Path to model files
        input_data (str): Path to input NRRD files
        sensitivity (float): Sensitivity parameter
    """
    lung_seg_task = lung_segmentation(
        model_files=model_files,
        input_data=input_data,
        sensitivity=sensitivity
    )

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        lung_segmentation_pipeline,
        "lung_segmentation_pipeline.yaml"
    )