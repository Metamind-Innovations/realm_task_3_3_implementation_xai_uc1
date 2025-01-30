import os
import keras.backend as K
from TheDuneAI import ContourPilot

GPU_compute = False  # Try setting GPU_compute to True if there is an available CUDA gpu
if GPU_compute:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Choose GPU device ID
    # Check availableGPUs
    print(K.tensorflow_backend._get_available_gpus())

# Model inputs
model_path = r'./model_files/'  # path to the model files
path_to_test_data = r'./test_data_duneai'  # path to the input data that will be segmented (nrrds)
save_path = r'./output_segmentations'  # path for the output files (nrrds)

# initialize the model
model = ContourPilot(model_path, path_to_test_data, save_path, verbosity=True)  # set verbosity=True to see what is going on

# Start the segmentation process
model.segment()
