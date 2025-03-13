import kfp
from kfp.dsl import Dataset, Input, Model, Output


@kfp.dsl.component(
    base_image="python:3.8-slim",
    packages_to_install=["requests"]
)
def download_github_files(
        github_repo_url: str,
        model_files: Output[Model],
        input_data: Output[Dataset],
        branch: str = "main",
):
    import os
    import subprocess
    import shutil
    import requests
    import time

    # Create output directories
    os.makedirs(model_files.path, exist_ok=True)
    os.makedirs(input_data.path, exist_ok=True)

    # Install git and git-lfs
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git", "git-lfs", "curl"], check=True)

    # Configure git for network issues
    subprocess.run(["git", "config", "--global", "http.postBuffer", "524288000"], check=True)
    subprocess.run(["git", "config", "--global", "http.lowSpeedLimit", "1000"], check=True)
    subprocess.run(["git", "config", "--global", "http.lowSpeedTime", "60"], check=True)

    # Initialize git-lfs
    subprocess.run(["git", "lfs", "install"], check=True)

    # Extract owner and repo from URL
    parts = github_repo_url.split('/')
    owner = parts[-2]
    repo = parts[-1]

    # Create segmentation_xai directory
    os.makedirs(os.path.join(model_files.path, "segmentation_xai"), exist_ok=True)

    # Download Python module files first
    files_to_download = [
        {"url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/segmentation_xai/TheDuneAI.py",
         "dest": os.path.join(model_files.path, "segmentation_xai", "TheDuneAI.py")},
        {"url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/segmentation_xai/xai.py",
         "dest": os.path.join(model_files.path, "segmentation_xai", "xai.py")},
        {"url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/segmentation_xai/generator.py",
         "dest": os.path.join(model_files.path, "segmentation_xai", "generator.py")},
        {"url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/segmentation_xai/lung_extraction_funcs.py",
         "dest": os.path.join(model_files.path, "segmentation_xai", "lung_extraction_funcs.py")},
        {"url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/model_files/model_v7.json",
         "dest": os.path.join(model_files.path, "model_v7.json")}
    ]

    # Create empty __init__.py file so segmentation_xai is discoverable
    with open(os.path.join(model_files.path, "segmentation_xai", "__init__.py"), "w") as f:
        pass

    # Download regular files
    for file_info in files_to_download:
        url = file_info["url"]
        dest = file_info["dest"]

        # Try up to 3 times with increasing timeouts
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30 * (attempt + 1))
                if response.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded {url}")
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

    # Download weights file with curl (LFS file)
    weights_file = os.path.join(model_files.path, "weights_v7.hdf5")
    weights_url = f"https://media.githubusercontent.com/media/{owner}/{repo}/main/model_files/weights_v7.hdf5"

    try:
        print(f"Downloading weights file from: {weights_url}")
        subprocess.run(
            ["curl", "-L", "--max-time", "600", "-o", weights_file, weights_url],
            check=True
        )
        print(f"Downloaded weights file: {os.path.getsize(weights_file)} bytes")
    except subprocess.SubprocessError as e:
        print(f"Failed to download weights with curl: {str(e)}")

        # Try alternate GitHub LFS URL format
        try:
            alt_url = f"https://github.com/{owner}/{repo}/raw/{branch}/model_files/weights_v7.hdf5"
            print(f"Trying alternate URL: {alt_url}")
            subprocess.run(
                ["curl", "-L", "--max-time", "600", "-o", weights_file, alt_url],
                check=True
            )
            print(f"Downloaded weights file: {os.path.getsize(weights_file)} bytes")
        except subprocess.SubprocessError as e2:
            print(f"Failed with alternate URL: {str(e2)}")

    # Download patient data from converted_nrrds directory
    try:
        print("Attempting to clone repository to copy patient data...")
        temp_dir = "/tmp/patient_repo"

        # Use sparse checkout to get just the converted_nrrds directory
        subprocess.run(["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
                        github_repo_url, temp_dir], check=True, timeout=300)

        os.chdir(temp_dir)
        subprocess.run(["git", "sparse-checkout", "set", "converted_nrrds"], check=True)

        # Copy the patient directories
        patient_src_dir = os.path.join(temp_dir, "converted_nrrds")
        if os.path.exists(patient_src_dir):
            for patient in os.listdir(patient_src_dir):
                src_patient_dir = os.path.join(patient_src_dir, patient)
                if os.path.isdir(src_patient_dir):
                    dst_patient_dir = os.path.join(input_data.path, patient)
                    shutil.copytree(src_patient_dir, dst_patient_dir)
                    print(f"Copied patient directory: {patient}")

            print(f"Copied patient data from {patient_src_dir} to {input_data.path}")
        else:
            print(f"Patient directory not found at: {patient_src_dir}")

    except Exception as e:
        print(f"Error copying patient data: {str(e)}")
        # Create placeholder if download fails
        with open(os.path.join(input_data.path, "sample.nrrd"), "wb") as f:
            f.write(b"NRRD0001\n# This is a placeholder NRRD file\ntype: float\ndimension: 3\nsizes: 10 10 10\n")

    # List patient directories (for visibility only)
    print("Patient directories in input_data:")
    patient_count = 0
    for item in os.listdir(input_data.path):
        item_path = os.path.join(input_data.path, item)
        if os.path.isdir(item_path):
            patient_count += 1
            print(f"  Patient: {item}")
            # List files in patient directory
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                if os.path.isfile(file_path):
                    print(f"    - {file}: {os.path.getsize(file_path)} bytes")

    print(f"Total patient count: {patient_count}")


@kfp.dsl.component(
    base_image="python:3.8-slim",
    packages_to_install=[
        "networkx>=2.5",
        "scikit-image",
        "h5py",
        "tensorflow==2.10.0",
        "keras==2.10.0",
        "numpy",
        "matplotlib",
        "scikit-fuzzy",
        "SimpleITK",
        "opencv-python",
        "scipy",
        "scikit-learn",
        "tqdm",
        "pandas",
        "protobuf<3.20"
    ]
)
def lung_segmentation(
        model_files: Input[Model],  # For demo purposes use: https://github.com/Metamind-Innovations/realm_task_3_3_implementation_xai_uc1/tree/main/model_files
        input_data: Input[Dataset],  # For demo purposes use: https://github.com/Metamind-Innovations/realm_task_3_3_implementation_xai_uc1/tree/main/converted_nrrds
        output_results: Output[Dataset],
        sensitivity: float = 0.7,
        verbosity: bool = True,
) -> str:
    import os
    import sys
    import subprocess
    import shutil

    # Create output directory
    os.makedirs(output_results.path, exist_ok=True)

    # Install system dependencies
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "ffmpeg", "libsm6", "libxext6", "-y"], check=True)

    # Set up module directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    module_dir = os.path.join(current_dir, "segmentation_xai")
    os.makedirs(module_dir, exist_ok=True)

    with open(os.path.join(module_dir, "__init__.py"), "w") as f:
        pass

    # Copy module files
    segmentation_src = os.path.join(model_files.path, "segmentation_xai")
    if os.path.exists(segmentation_src):
        for item in os.listdir(segmentation_src):
            src_file = os.path.join(segmentation_src, item)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(module_dir, item))

    try:
        from segmentation_xai.TheDuneAI import ContourPilot
        model = ContourPilot(
            model_path=model_files.path,
            data_path=input_data.path,
            output_path=output_results.path,
            sensitivity=sensitivity,
            verbosity=verbosity
        )

        model.segment()

        return "Lung segmentation completed successfully"

    except Exception as e:
        import traceback
        error_message = f"Segmentation failed: {str(e)}\n\n{traceback.format_exc()}"
        with open(os.path.join(output_results.path, 'error.txt'), 'w') as f:
            f.write(error_message)
        return error_message


@kfp.dsl.pipeline(
    name="Lung Segmentation Pipeline",
    description="Pipeline for lung segmentation and XAI visualization"
)
def lung_segmentation_pipeline(
        github_repo_url: str,
        sensitivity: float = 0.7,
        verbosity: bool = True,
        branch: str = "main"
):
    # Download files from GitHub with LFS support
    download_task = download_github_files(
        github_repo_url=github_repo_url,
        branch=branch
    )

    # Perform lung segmentation
    lung_segmentation_task = lung_segmentation(
        model_files=download_task.outputs["model_files"],
        input_data=download_task.outputs["input_data"],
        sensitivity=sensitivity,
        verbosity=verbosity
    )
    lung_segmentation_task.set_cpu_request("2")
    lung_segmentation_task.set_cpu_limit("4")
    lung_segmentation_task.set_memory_request("4G")
    lung_segmentation_task.set_memory_limit("8G")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=lung_segmentation_pipeline,
        package_path="lung_segmentation_pipeline.yaml"
    )
