import SimpleITK as sitk
import numpy as np
import keras
import cv2
import time
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
# Import pipeline functions
import lung_extraction_funcs as le
import generator

from tqdm import tqdm as tqdm
from xai import generate_lime_explanations, GradCAM, generate_xai_report


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
        self.gradcam = GradCAM(self.model1)

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

    def _generate_combined_report(self, lime_paths, gradcam_paths, metrics, output_dir):
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader

        pdf_path = os.path.join(output_dir, 'XAI_Combined_Report.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)

        # Title Page
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, 750, "Combined XAI Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 720, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        c.showPage()

        # Content Pages
        for (lime_path, gradcam_path, metric) in zip(lime_paths, gradcam_paths, metrics):
            # LIME Image
            img = ImageReader(lime_path)
            c.drawImage(img, 50, 500, width=250, height=250)

            # GRAD-CAM Image
            img = ImageReader(gradcam_path)
            c.drawImage(img, 300, 500, width=250, height=250)

            # Metrics
            c.setFont("Helvetica", 10)
            y_pos = 480
            for key, value in metric.items():
                c.drawString(50, y_pos, f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
                y_pos -= 20

            c.showPage()

        c.save()

    def segment(self):
        if self.model1 and self.Patient_dict and self.Output_path:
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

                # Create a mask from the predicted array
                mask = predicted_array
                selected_slices = np.where(np.any(mask, axis=(1, 2)))[0]

                # Adjust slices based on z_st to match preprocessed img indices
                if 'z_st' in params:
                    adjusted_slices = selected_slices - params['z_st']
                    valid_indices = (adjusted_slices >= 0) & (adjusted_slices < img.shape[0])
                    valid_slices = adjusted_slices[valid_indices]
                else:
                    valid_slices = selected_slices

                # Report generated for at most 3 slices. Change this number if needed
                num_slices = min(3, len(valid_slices))

                if num_slices == 0:
                    print(f"No valid segmented slices found for {filename}, skipping LIME.")
                    continue

                # XAI Processing
                lime_metrics = []
                gradcam_paths = []
                lime_paths = []

                for i, sl in enumerate(valid_slices[:num_slices]):
                    original_slice = img[sl, ...]
                    lung_mask = (original_slice > -500).astype(np.uint8)

                    # LIME Explanation
                    lime_metric, lime_path = generate_lime_explanations(
                        self.model1, original_slice, lung_mask, f"slice_{sl}", output_dir
                    )
                    lime_metrics.append(lime_metric)
                    lime_paths.append(lime_path)

                    # GRAD-CAM Explanation
                    gradcam_path = self.gradcam.explain(
                        original_slice,
                        original_slice,
                        output_dir,
                        f"slice_{sl}"
                    )
                    gradcam_paths.append(gradcam_path)

                    # Generate individual reports using the shared function
                generate_xai_report("LIME", lime_paths, lime_metrics, output_dir)
                generate_xai_report("GRAD-CAM", gradcam_paths, [], output_dir)

                # Generate combined report
                self._generate_combined_report(lime_paths, gradcam_paths, lime_metrics, output_dir)
