import SimpleITK as sitk
import numpy as np
import keras
import cv2
import time
import os
import shap
import matplotlib.pyplot as plt

# Import pipeline functions
import lung_extraction_funcs as le
import generator

from scipy import ndimage
from tqdm import tqdm as tqdm


class ContourPilot:
    def __init__(self, model_path, data_path, output_path='./', verbosity=False, pat_dict=None):
        self.verbosity = verbosity
        self.model1 = None
        self.__load_model__(model_path)
        if pat_dict:
            self.Patient_dict = pat_dict
        else:
            self.Patient_dict = le.parse_dataset(data_path, img_only=True)
        self.Patients_gen = generator.Patient_data_generator(self.Patient_dict, predict=True, batch_size=1,
                                                             image_size=512, shuffle=True,
                                                             use_window=True, window_params=[1500, -600],
                                                             resample_int_val=True, resampling_step=25,
                                                             extract_lungs=True, size_eval=False,
                                                             verbosity=verbosity, reshape=True, img_only=True)
        self.Output_path = output_path

    def __load_model__(self, model_path):
        json_file = open(os.path.join(model_path, 'model_v7.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model1 = keras.models.model_from_json(loaded_model_json)
        self.model1.load_weights(os.path.join(model_path, 'weights_v7.hdf5'))

    def __generate_segmentation__(self, img, params, thr=0.99):
        temp_pred_arr = np.zeros_like(img)
        if self.verbosity:
            print('Segmentation started')
            st = time.time()
        for j in range(len(img)):
            predicted_slice = self.model1.predict(img[j, ...].reshape(-1, 512, 512, 1)).reshape(512, 512)
            temp_pred_arr[j, ...] = 1 * (predicted_slice > thr)
        if self.verbosity:
            print('Segmentation is finished')
            print('time spent: %s sec.' % (time.time() - st))
        predicted_arr_temp = le.max_connected_volume_extraction(temp_pred_arr)
        temporary_mask = np.zeros(params['normalized_shape'], np.uint8)

        if params['crop_type']:
            temporary_mask[params['z_st']:params['z_end'], ...] = predicted_arr_temp[:,
                                                                  params['xy_st']:params['xy_end'],
                                                                  params['xy_st']:params['xy_end']]
        else:
            temporary_mask[params['z_st']:params['z_end'], params['xy_st']:params['xy_end'],
            params['xy_st']:params['xy_end']] = predicted_arr_temp

        if temporary_mask.shape != params['original_shape']:
            predicted_array = np.array(
                1 * (le.resize_3d_img(temporary_mask, params['original_shape'], cv2.INTER_NEAREST) > 0.5), np.int8)
        else:
            predicted_array = np.array(temporary_mask, np.int8)

        return predicted_array

    def segment(self):
        if self.model1 and self.Patient_dict and self.Output_path:
            count = 0
            for img, _, filename, params in tqdm(self.Patients_gen, desc='Progress'):

                filename = filename[0]
                params = params[0]
                img = np.squeeze(img)

                if len(img.shape) != 3 or img.shape[1:] != (512, 512):
                    raise ValueError(f"Unexpected image shape: {img.shape}. Expected (n_slices, 512, 512).")

                for slice_idx in range(img.shape[0]):
                    slice_img = img[slice_idx]
                    if len(slice_img.shape) == 2:
                        slice_img = np.expand_dims(slice_img, axis=-1)  # Add channel dimension

                    if slice_img.shape != (512, 512, 1):
                        raise ValueError(f"Unexpected reshaped slice shape: {slice_img.shape}. Expected (512, 512, 1).")

                    slice_img_batched = np.expand_dims(slice_img, axis=0)  # Add batch dimension

                    # Validate batch shape
                    if slice_img_batched.shape != (1, 512, 512, 1):
                        raise ValueError(f"Batch shape mismatch: {slice_img_batched.shape}")

                    # Configure SHAP masker and explainer
                    masker = shap.maskers.Image(np.zeros_like(slice_img), shape=(512, 512, 1))
                    explainer = shap.Explainer(self.model1, masker)

                    # Generate SHAP values
                    shap_values = explainer(slice_img_batched).reshape((1, 512, 512, 1))

                    # Plot and save SHAP values for this slice
                    plt.figure()
                    shap.plots.image(shap_values[0], show=False)
                    output_dir = os.path.join(self.Output_path, filename.split('\\')[-2] + '_(DL)')
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(os.path.join(output_dir, f"slice_{slice_idx}_shap_plot.png"))
                    plt.close()

                # Save segmentation results
                predicted_array = self.__generate_segmentation__(img, params)

                if not os.path.exists(os.path.join(self.Output_path, filename.split('\\')[-2] + '_(DL)')):
                    os.makedirs(os.path.join(self.Output_path, filename.split('\\')[-2] + '_(DL)'))

                generated_img = sitk.GetImageFromArray(predicted_array)
                generated_img.SetSpacing(params['original_spacing'])
                generated_img.SetOrigin(params['img_origin'])
                sitk.WriteImage(generated_img,
                                os.path.join(self.Output_path, filename.split('\\')[-2] + '_(DL)', 'DL_mask.nrrd'))
                temp_data = sitk.ReadImage(filename)
                sitk.WriteImage(temp_data,
                                os.path.join(self.Output_path, filename.split('\\')[-2] + '_(DL)', 'image.nrrd'))

                if count == len(self.Patients_gen):
                    return 0

                count += 1
