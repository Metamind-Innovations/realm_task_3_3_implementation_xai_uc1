import os
import time
from pathlib import Path

import SimpleITK as sitk
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tqdm import tqdm

import segmentation_xai.generator as generator
import segmentation_xai.lung_extraction_funcs as le
from segmentation_xai.xai import generate_gradcam_explanation, CMAP_BONE


class FuzzyXAISelector:
    def __init__(self):
        """
        Initialize the fuzzy XAI selector.

        The fuzzy system takes in a sensitivity input and outputs both a dynamic threshold
        and a selected layer for Grad-CAM explanations.

        The fuzzy system is defined as follows:

        * The input variable is `sensitivity`, which ranges from 0 to 1.0 with a step of 0.1.
        * The consequent variable is `seg_threshold`, which ranges from 0.5 to 1.0 with a step of 0.05.
        * The consequent variable is `gradcam_layer`, which ranges from 16 to 22 with a step of 1.

        The membership functions are defined in `_define_membership_functions` and the rules are defined in `_create_rules`.
        """
        self.sensitivity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'sensitivity')

        # Consequent for threshold
        self.seg_threshold = ctrl.Consequent(np.arange(0.5, 1.05, 0.05), 'seg_threshold')

        # Consequent for gradcam layer
        self.gradcam_layer = ctrl.Consequent(np.arange(16, 23, 1), 'gradcam_layer', defuzzify_method='centroid')

        # Membership functions
        self._define_membership_functions()
        self._create_rules()
        self.system = ctrl.ControlSystem(self.rules)
        self.decision_maker = ctrl.ControlSystemSimulation(self.system)

    def _define_membership_functions(self):
        # Sensitivity levels
        self.sensitivity['low'] = fuzz.trimf(self.sensitivity.universe, [0, 0, 0.5])
        self.sensitivity['medium'] = fuzz.trimf(self.sensitivity.universe, [0.5, 0.7, 0.9])
        self.sensitivity['high'] = fuzz.trimf(self.sensitivity.universe, [0.85, 1.0, 1.0])

        # Threshold membership functions
        self.seg_threshold['high'] = fuzz.trimf(self.seg_threshold.universe, [0.9, 0.95, 1.0])
        self.seg_threshold['medium'] = fuzz.trimf(self.seg_threshold.universe, [0.65, 0.75, 0.85])
        self.seg_threshold['low'] = fuzz.trimf(self.seg_threshold.universe, [0.5, 0.5, 0.6])

        # Layer selection
        for layer in range(16, 23):
            self.gradcam_layer[f'conv{layer}'] = fuzz.trimf(
                self.gradcam_layer.universe,
                [layer - 0.5, layer, layer + 0.5]
            )

    def _create_rules(self):
        self.rules = [
            ctrl.Rule(
                self.sensitivity['high'],
                (self.gradcam_layer['conv16'], self.seg_threshold['high'])
            ),
            ctrl.Rule(
                self.sensitivity['medium'],
                (self.gradcam_layer['conv19'], self.seg_threshold['medium'])
            ),
            ctrl.Rule(
                self.sensitivity['low'],
                (self.gradcam_layer['conv22'], self.seg_threshold['low'])
            )
        ]

    def decide(self, sensitivity):
        """
        Decide on the dynamic threshold and Grad-CAM layer based on the input sensitivity.

        Parameters
        ----------
        sensitivity : float
            The sensitivity of the segmentation, a value between 0 and 1.

        Returns
        -------
        out : dict
            A dictionary containing the selected Grad-CAM layer and the dynamic threshold
            for segmentation.
        """
        self.decision_maker.input['sensitivity'] = sensitivity
        self.decision_maker.compute()
        selected_layer = int(round(self.decision_maker.output['gradcam_layer']))
        dynamic_threshold = round(self.decision_maker.output['seg_threshold'], 2)
        return {"gradcam_layer": selected_layer, "seg_threshold": dynamic_threshold}


class ContourPilot:
    """Main class for medical image segmentation and XAI report generation."""

    def __init__(self, model_path, data_path, sensitivity, output_path='./', verbosity=False, pat_dict=None):
        self.verbosity = verbosity
        self.model = self._load_model(model_path)
        self.patient_dict = pat_dict if pat_dict else le.parse_dataset(data_path, img_only=True)
        self.output_path = output_path
        self.sensitivity = sensitivity

        # Initialize data generator with explicit parameter names
        self.patient_generator = generator.Patient_data_generator(
            patient_dict=self.patient_dict,
            predict=True,
            batch_size=1,
            image_size=512,
            shuffle=True,
            use_window=True,
            window_params=[1500, -600],
            resample_int_val=True,
            resampling_step=25,
            extract_lungs=True,
            size_eval=False,
            verbosity=verbosity,
            reshape=True,
            img_only=True,
        )

    def _load_model(self, model_path):
        """Load U-Net model v7 with pretrained weights for lung segmentation."""
        model_config_path = os.path.join(model_path, 'model_v7.json')
        model_weights_path = os.path.join(model_path, 'weights_v7.hdf5')

        with open(model_config_path, 'r') as json_file:
            model = keras.models.model_from_json(json_file.read())
        model.load_weights(model_weights_path)
        return model

    def _generate_segmentation(self, input_volume, processing_params, threshold=None):
        """Generate 3D segmentation mask using the loaded model."""
        if threshold is None:
            threshold = 0.99

        if self.verbosity:
            print(f'Starting segmentation for volume of shape {input_volume.shape}...')
            timer_start = time.time()

        # Initialize prediction array
        raw_predictions = np.zeros_like(input_volume)

        # Process each slice
        for slice_idx in range(len(input_volume)):
            slice_prediction = self.model.predict(
                input_volume[slice_idx, ...].reshape(-1, 512, 512, 1)
            ).reshape(512, 512)
            raw_predictions[slice_idx, ...] = 1 * (slice_prediction > threshold)

        # Post-process predictions
        processed_mask = le.max_connected_volume_extraction(raw_predictions)
        final_mask = self._reconstruct_volume(processed_mask, processing_params)

        if self.verbosity:
            print(f'Segmentation completed in {time.time() - timer_start:.2f} seconds')

        return final_mask.astype(np.int8), processed_mask

    def _reconstruct_volume(self, processed_mask, processing_params):
        """Reconstruct mask to match original DICOM dimensions and spacing."""
        reconstructed_volume = np.zeros(processing_params['normalized_shape'], dtype=np.uint8)

        # Calculate crop boundaries
        z_start, z_end = processing_params['z_st'], processing_params['z_end']
        xy_start, xy_end = processing_params['xy_st'], processing_params['xy_end']

        if processing_params['crop_type']:
            reconstructed_volume[z_start:z_end] = processed_mask[:, xy_start:xy_end, xy_start:xy_end]
        else:
            reconstructed_volume[z_start:z_end, xy_start:xy_end, xy_start:xy_end] = processed_mask

        # Resize if necessary
        if reconstructed_volume.shape != processing_params['original_shape']:
            return le.resize_3d_img(
                reconstructed_volume,
                processing_params['original_shape'],
                interp=cv2.INTER_NEAREST
            )
        return reconstructed_volume

    def _save_results(self, segmentation_mask, processing_params, source_file_path):
        """Save segmentation results and original image to NRRD format."""
        patient_id = Path(source_file_path).parent.name
        output_dir = Path(self.output_path) / f"{patient_id}_(DL)"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save segmentation mask
        sitk_mask = sitk.GetImageFromArray(segmentation_mask)
        sitk_mask.SetSpacing(processing_params['original_spacing'])
        sitk_mask.SetOrigin(processing_params['img_origin'])
        sitk.WriteImage(sitk_mask, os.path.join(output_dir, 'DL_mask.nrrd'))

        # Save original image
        original_image = sitk.ReadImage(source_file_path)
        sitk.WriteImage(original_image, os.path.join(output_dir, 'image.nrrd'))

        return output_dir

    def _get_explanation_slices(self, volume, segmentation_mask, processing_params):
        """Identify valid slices for XAI explanations."""
        valid_axial = np.where(np.any(segmentation_mask, axis=(1, 2)))[0]
        valid_coronal = np.where(np.any(segmentation_mask, axis=(0, 2)))[0]
        valid_sagittal = np.where(np.any(segmentation_mask, axis=(0, 1)))[0]

        # Adjust for preprocessing crops
        if 'z_st' in processing_params:
            valid_axial -= processing_params['z_st']
            valid_axial = valid_axial[(valid_axial >= 0) & (valid_axial < volume.shape[0])]

        return valid_axial, valid_coronal, valid_sagittal

    def segment(self):
        """Main pipeline execution method."""
        fuzzy_selector = FuzzyXAISelector()

        for volume, _, file_path, params in tqdm(self.patient_generator, desc='Processing patients'):
            volume_array = np.squeeze(volume[0])
            processing_params = params[0]
            file_path = file_path[0]

            sensitivity = float(self.sensitivity)
            print(f'Sensitivity: {sensitivity}')

            # Get both parameters from fuzzy system
            fuzzy_params = fuzzy_selector.decide(sensitivity)
            selected_layer = fuzzy_params['gradcam_layer']
            dynamic_threshold = fuzzy_params['seg_threshold']
            print(f'Selected layer: {selected_layer}, dynamic threshold: {dynamic_threshold}')

            # Generate segmentation
            segmentation_mask, processed_mask = self._generate_segmentation(volume_array, processing_params,
                                                                            threshold=dynamic_threshold)
            output_dir = self._save_results(segmentation_mask, processing_params, file_path)
            axial_expl_slices, coronal_expl_slices, sagittal_expl_slices = self._get_explanation_slices(volume_array,
                                                                                                        segmentation_mask,
                                                                                                        processing_params)

            # Generate explanations
            self._generate_xai_reports(
                volume_array,
                [axial_expl_slices, coronal_expl_slices, sagittal_expl_slices],
                output_dir,
                gradcam_layer=f'conv2d_{selected_layer}',
                processed_mask=processed_mask
            )

    def _generate_xai_reports(self, volume, slice_indices: list, output_dir, gradcam_layer, processed_mask):
        """Generate and save XAI explanations for selected slices."""
        planes = [
            ('axial', slice_indices[0]),
            ('sagittal', slice_indices[1]),
            ('coronal', slice_indices[2])
        ]

        for plane, indices in tqdm(planes, desc='Processing planes'):
            for idx in tqdm(indices, desc=f'Generating {plane.capitalize()} XAI Images', position=0, leave=True):
                if plane == 'axial':
                    ct_slice = volume[idx, :, :]
                    mask_slice = processed_mask[idx, :, :]
                elif plane == 'sagittal':
                    ct_slice = np.flipud(cv2.resize(volume[:, :, idx], (512, 512)))
                    mask_slice = cv2.resize(np.flipud(processed_mask[:, :, idx]).astype(np.float32),
                                            (512, 512)) > 0.3
                else:  # coronal
                    ct_slice = np.flipud(cv2.resize(volume[:, idx, :], (512, 512)))
                    mask_slice = cv2.resize(np.flipud(processed_mask[:, idx, :]).astype(np.float32),
                                            (512, 512)) > 0.3

                # No segmentation detected for this slice, skip
                if np.sum(mask_slice) == 0:
                    continue

                # Save original slice
                plt.figure(figsize=(5, 5))
                plt.imshow(ct_slice, cmap=CMAP_BONE)
                plt.title(f'Original {plane.capitalize()} Slice')
                plt.axis('off')
                orig_path = os.path.join(output_dir, f"original_slice_{plane}_{idx}.png")
                plt.savefig(orig_path, bbox_inches='tight', dpi=100, pad_inches=0)
                plt.close()

                # Save segmented slice
                masked_mask = np.ma.masked_where(mask_slice == 0, mask_slice)
                plt.figure(figsize=(5, 5))
                plt.imshow(ct_slice, cmap=CMAP_BONE)
                plt.imshow(masked_mask, cmap='Reds', alpha=1.0, norm=plt.Normalize(vmin=0, vmax=1))
                plt.title(f'Segmented {plane.capitalize()} Slice')
                plt.axis('off')
                seg_path = os.path.join(output_dir, f"segmented_slice_{plane}_{idx}.png")
                plt.savefig(seg_path, bbox_inches='tight', dpi=100, pad_inches=0)
                plt.close()

                # Generate Grad-CAM explanation for the current slice
                generate_gradcam_explanation(
                    self.model,
                    ct_slice,
                    output_dir,
                    f"{plane}_slice_{idx}",
                    layer_name=gradcam_layer
                )
