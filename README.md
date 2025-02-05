# realm_task_3_3_implementation_xai

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

## Fairness and Bias metrics

For this task we use the `bias_fairness.py` script. This script generates metrics such as **Equalized Odds Difference**
between gender groups and **Demographic Parity**.
It also generates the **Top 5 Predictive Features** DataFrame which shows how each feature contributes to the model's
predictions.
(E.g. If a feature `age` has an `odds_ratio` of `1.03` then that means that each year increases mortality risk by `3%`).
A `report.pdf` file is also created containing all the metrics and 2 confusion matrices for better comprehension.

In order to change the dataset, modify the `csv_path` variable in the `bias_fairness.py` script (line 178).