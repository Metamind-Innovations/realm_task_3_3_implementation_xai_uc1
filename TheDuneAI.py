import SimpleITK as sitk
import numpy as np
import keras
import cv2
import time
import os
import shap
import matplotlib.pyplot as plt
import skimage.segmentation
import tensorflow as tf
import shutil

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
        if not (self.model1 and self.Patient_dict and self.Output_path):
            return 0

        # Configure GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        # Precompile model inference function
        @tf.function(reduce_retracing=True)
        def tf_predict(batch):
            return self.model1(batch, training=False)

        for img, _, filename, params in tqdm(self.Patients_gen, desc='Patients'):
            filename = filename[0]
            params = params[0]
            img = np.squeeze(img)

            if len(img.shape) != 3 or img.shape[1:] != (512, 512):
                raise ValueError(f"Unexpected image shape: {img.shape}. Expected (n_slices, 512, 512).")

            # Create output directory once per patient
            base_name = os.path.basename(filename).split('.')[0]
            output_dir = os.path.join(self.Output_path, f"{base_name}_(DL)")
            os.makedirs(output_dir, exist_ok=True)

            # Process slices
            for slice_idx in range(2):
                try:
                    slice_2d = np.squeeze(img[slice_idx])
                    if np.max(slice_2d) < -700:
                        continue

                    # Superpixel generation
                    segments = skimage.segmentation.slic(
                        slice_2d,
                        n_segments=7,
                        compactness=25,
                        sigma=1,
                        start_label=0,
                        channel_axis=None
                    )
                    n_segments = segments.max() + 1

                    # Vectorized prediction function
                    def predict_fn(masks):
                        batch = np.zeros((len(masks), 512, 512, 1), dtype=np.float32)
                        for i in range(n_segments):
                            batch[:, segments == i, 0] = masks[:, i:i + 1] * slice_2d[segments == i]
                        return tf_predict(batch).numpy().mean(axis=(1, 2, 3)).reshape(-1, 1)

                    # SHAP explanation
                    explainer = shap.KernelExplainer(
                        predict_fn,
                        data=np.zeros((1, n_segments)),
                        link="identity",
                        l1_reg=f"num_features({min(n_segments, 4)})"
                    )

                    shap_values = explainer.shap_values(
                        np.ones((1, n_segments)),
                        nsamples=80,
                        silent=True
                    )

                    # Process SHAP values
                    if isinstance(shap_values, list):
                        if len(shap_values) == 2:
                            shap_values = shap_values[1]
                        else:
                            shap_values = shap_values[0]

                    shap_values = np.array(shap_values).squeeze()

                    if shap_values.size != n_segments:
                        raise ValueError(f"SHAP values mismatch: {shap_values.size} vs {n_segments}")

                    # Create heatmap
                    shap_heatmap = np.zeros_like(slice_2d, dtype=np.float32)
                    for i in range(n_segments):
                        if i < shap_values.size:
                            shap_heatmap[segments == i] = shap_values[i]

                    # Save visualization
                    plt.figure(figsize=(10, 10))
                    plt.imshow(slice_2d, cmap='gray')
                    plt.imshow(shap_heatmap, alpha=0.4, cmap='jet')
                    plt.axis('off')
                    plt.savefig(os.path.join(output_dir, f"slice_{slice_idx}_shap.png"),
                                bbox_inches='tight', dpi=100)
                    plt.close()

                except Exception as e:
                    print(f"Error processing slice {slice_idx}: {str(e)}")
                    continue

            # Segmentation pipeline
            try:
                predicted_array = self.__generate_segmentation__(img, params)

                generated_img = sitk.GetImageFromArray(predicted_array)
                generated_img.SetSpacing(params['original_spacing'])
                generated_img.SetOrigin(params['img_origin'])

                sitk.WriteImage(generated_img, os.path.join(output_dir, 'DL_mask.nrrd'))
                shutil.copy2(filename, os.path.join(output_dir, os.path.basename(filename)))

            except Exception as e:
                print(f"Error saving outputs: {str(e)}")
                raise

        return 0
