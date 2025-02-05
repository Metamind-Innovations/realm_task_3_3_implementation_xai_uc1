from TheDuneAI import ContourPilot

# If a CUDA GPU is available, it is used automatically by tensorflow

# Model inputs
model_path = r'./model_files/'  # path to the model files
path_to_test_data = r'./converted_nrrds'  # path to the input data that will be segmented (nrrds)
save_path = r'./output_segmentations_radiomics'  # path for the output files (nrrds)

# initialize the model
model = ContourPilot(model_path, path_to_test_data, save_path,
                     verbosity=True)  # set verbosity=True to see what is going on

# Start the segmentation process
model.segment()
