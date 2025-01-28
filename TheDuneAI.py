import SimpleITK as sitk
import numpy as np
import keras
import cv2
import time
import os
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# Import pipeline functions
import lung_extraction_funcs as le
import generator

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
                                                             resample_int_val=True, resampling_step=25,  # 25
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
                img = np.squeeze(img)  # Shape: (slices, 512, 512)

                predicted_array = self.__generate_segmentation__(img, params)

                # Create output directory
                output_dir = os.path.join(self.Output_path, filename.split('\\')[-2] + '_(DL)')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Save segmentation and image
                generated_img = sitk.GetImageFromArray(predicted_array)
                generated_img.SetSpacing(params['original_spacing'])
                generated_img.SetOrigin(params['img_origin'])
                sitk.WriteImage(generated_img, os.path.join(output_dir, 'DL_mask.nrrd'))
                temp_data = sitk.ReadImage(filename)
                sitk.WriteImage(temp_data, os.path.join(output_dir, 'image.nrrd'))

                # Generate LIME explanations
                mask = predicted_array
                selected_slices = np.where(np.any(mask, axis=(1, 2)))[0]

                # Adjust slices based on z_st to match preprocessed img indices
                if 'z_st' in params:
                    adjusted_slices = selected_slices - params['z_st']
                    valid_indices = (adjusted_slices >= 0) & (adjusted_slices < img.shape[0])
                    valid_slices = adjusted_slices[valid_indices]
                else:
                    valid_slices = selected_slices

                num_slices = min(3, len(valid_slices))

                if num_slices == 0:
                    print(f"No valid segmented slices found for {filename}, skipping LIME.")
                    continue

                explainer = lime_image.LimeImageExplainer()

                fig_paths = []
                for i, sl in enumerate(valid_slices[:num_slices]):
                    original_slice = img[sl, ...]

                    # Threshold to identify lung regions
                    lung_mask = (original_slice > -500).astype(np.uint8)

                    # Constrain LIME to Lung Areas
                    def batch_predict(images):
                        # Convert RGB to grayscale
                        grayscale = images[..., 0].reshape(-1, 512, 512, 1)
                        masked_images = grayscale * lung_mask[..., np.newaxis]
                        predictions = self.model1.predict(masked_images)

                        # Aggregate predictions
                        return np.mean(predictions, axis=(1, 2))

                    # Generate explanation
                    img_normalized = (original_slice - np.min(original_slice)) / (
                                np.max(original_slice) - np.min(original_slice) + 1e-8)
                    img_rgb = np.stack([img_normalized] * 3, axis=-1)

                    explanation = explainer.explain_instance(
                        img_rgb,
                        batch_predict,
                        top_labels=1,
                        hide_color=0,
                        num_samples=200,
                        num_features=5
                    )

                    # Get explanation and apply lung mask
                    temp, mask_lime = explanation.get_image_and_mask(
                        explanation.top_labels[0],
                        positive_only=True,
                        num_features=3,
                        hide_rest=False
                    )

                    # Enforce the lung mask
                    temp = temp * lung_mask[..., np.newaxis]  # Keep only lung regions
                    mask_lime = mask_lime * lung_mask  # Mask non-lung areas

                    # Plot
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(original_slice, cmap='bone')
                    plt.title(f'Slice {sl}')
                    plt.subplot(1, 2, 2)
                    plt.imshow(mark_boundaries(temp, mask_lime))
                    plt.title('LIME Explanation')
                    plt.tight_layout()

                    fig_path = os.path.join(output_dir, f'lime_explanation_{i}.png')
                    plt.savefig(fig_path, bbox_inches='tight')
                    plt.close()
                    fig_paths.append(fig_path)

                # Create PDF report
                if fig_paths:
                    pdf_path = os.path.join(output_dir, 'LIME_Report.pdf')
                    c = canvas.Canvas(pdf_path, pagesize=letter)
                    for fig in fig_paths:
                        img_reader = ImageReader(fig)
                        img_width, img_height = img_reader.getSize()
                        aspect = img_height / img_width
                        c.drawImage(fig, 50, 50, width=500, height=500 * aspect)
                        c.showPage()
                    c.save()

                count += 1
                if count == len(self.Patients_gen):
                    return 0
