import os
import time

import SimpleITK as sitk
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tqdm import tqdm

import generator
import lung_extraction_funcs as le
from xai import generate_gradcam_explanation, CMAP_BONE


class FuzzyXAISelector:
    def __init__(self):
        # Input variables
        self.sensitivity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'sensitivity')

        # Output as gradcam layer
        self.gradcam_layer = ctrl.Consequent(np.arange(16, 23, 1), 'gradcam_layer', defuzzify_method='centroid')

        # Membership functions
        self._define_membership_functions()
        self._create_rules()
        self.system = ctrl.ControlSystem(self.rules)
        self.decision_maker = ctrl.ControlSystemSimulation(self.system)

    def _define_membership_functions(self):
        # Sensitivity levels
        self.sensitivity['low'] = fuzz.trimf(self.sensitivity.universe, [0, 0, 0.5])
        self.sensitivity['medium'] = fuzz.trimf(self.sensitivity.universe, [0.3, 0.5, 0.7])
        self.sensitivity['high'] = fuzz.trimf(self.sensitivity.universe, [0.5, 1, 1])

        # Layer selection
        layers = [16, 17, 18, 19, 20, 21, 22]
        for layer in layers:
            self.gradcam_layer[f'conv{layer}'] = fuzz.trimf(
                self.gradcam_layer.universe,
                [layer - 0.5, layer, layer + 0.5]
            )

    def _create_rules(self):
        self.rules = [
            ctrl.Rule(self.sensitivity['low'], self.gradcam_layer['conv16']),
            ctrl.Rule(self.sensitivity['medium'], self.gradcam_layer['conv19']),
            ctrl.Rule(self.sensitivity['high'], self.gradcam_layer['conv22'])
        ]

    def decide(self, sensitivity):
        self.decision_maker.input['sensitivity'] = sensitivity
        self.decision_maker.compute()
        return int(round(self.decision_maker.output['gradcam_layer']))


class ContourPilot:
    """Main class for medical image segmentation and XAI report generation."""

    def __init__(self, model_path, data_path, output_path='./', verbosity=False, pat_dict=None):
        self.verbosity = verbosity
        self.model = self._load_model(model_path)
        self.patient_dict = pat_dict if pat_dict else le.parse_dataset(data_path, img_only=True)
        self.output_path = output_path

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
            img_only=True
        )

    def _load_model(self, model_path):
        """Load Keras model from JSON configuration and weights."""
        model_config_path = os.path.join(model_path, 'model_v7.json')
        model_weights_path = os.path.join(model_path, 'weights_v7.hdf5')

        with open(model_config_path, 'r') as json_file:
            model = keras.models.model_from_json(json_file.read())
        model.load_weights(model_weights_path)
        return model

    def _generate_segmentation(self, input_volume, processing_params, threshold=0.99):
        """Generate 3D segmentation mask using the loaded model."""
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
        patient_id = source_file_path.split('\\')[-2]
        output_dir = os.path.join(self.output_path, f'{patient_id}_(DL)')
        os.makedirs(output_dir, exist_ok=True)

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
            volume_array = np.squeeze(volume[0])  # Remove batch dimension
            processing_params = params[0]
            file_path = file_path[0]

            # Generate segmentation
            segmentation_mask, processed_mask = self._generate_segmentation(volume_array, processing_params)
            output_dir = self._save_results(segmentation_mask, processing_params, file_path)
            axial_expl_slices, coronal_expl_slices, sagittal_expl_slices = self._get_explanation_slices(volume_array,
                                                                                                        segmentation_mask,
                                                                                                        processing_params)

            # Calculate sensitivity
            sensitivity = np.percentile(volume_array, 95) / 4095  # Normalize to 0-1

            # Get selected layer from fuzzy logic
            selected_layer = fuzzy_selector.decide(sensitivity)

            # Generate explanations
            self._generate_xai_reports(
                volume_array,
                [axial_expl_slices, coronal_expl_slices, sagittal_expl_slices],
                output_dir,
                gradcam_layer=f'conv2d_{selected_layer}',
                processed_mask=processed_mask
            )

    # TODO Optimize below function
    def _generate_xai_reports(self, volume, slice_indices: list, output_dir, gradcam_layer, processed_mask):
        """Generate and save XAI explanations for selected slices."""
        original_paths, gradcam_paths, segmented_paths = [], [], []
        valid_axial, valid_coronal, valid_sagittal = slice_indices[0], slice_indices[1], slice_indices[2]

        for idx in tqdm(valid_axial, desc='Generating Axial XAI Images'):
            ct_trans = volume[idx, :, :]
            mask_trans = processed_mask[idx, :, :]

            # Save original axial slice
            orig_trans = os.path.join(output_dir, f"original_slice_axial_{idx}.png")
            plt.figure(figsize=(5, 5))
            plt.imshow(ct_trans, cmap=CMAP_BONE)
            plt.title('Original Transverse Slice')
            plt.axis('off')
            plt.savefig(orig_trans, bbox_inches='tight', dpi=100)
            plt.close()
            original_paths.append(orig_trans)

            # Save segmented transverse slice
            seg_axial = os.path.join(output_dir, f"segmented_slice_axial_{idx}.png")
            plt.figure(figsize=(5, 5))
            plt.imshow(ct_trans, cmap=CMAP_BONE)
            plt.imshow(mask_trans, cmap='Reds', alpha=0.5)
            plt.title('Segmented Transverse Slice')
            plt.axis('off')
            plt.savefig(seg_axial, bbox_inches='tight', dpi=100)
            plt.close()
            segmented_paths.append(seg_axial)

            # Generate Grad-CAM for Transverse plane
            grad_axial = generate_gradcam_explanation(
                self.model,
                ct_trans,
                output_dir,
                f"axial_slice_{idx}",
                layer_name=gradcam_layer
            )
            gradcam_paths.append(grad_axial)

        for idx in tqdm(valid_sagittal, desc='Generating Sagittal XAI Images'):
            ct_sagittal = volume[:, :, idx]
            ct_sagittal = cv2.resize(ct_sagittal, (512, 512))  # Add resizing
            ct_sagittal = np.flipud(ct_sagittal)
            mask_sagittal = processed_mask[:, :, idx]
            mask_sagittal = np.flipud(mask_sagittal)
            mask_sagittal = cv2.resize(mask_sagittal.astype(np.float32), (512, 512)) > 0.5

            # Save visualization: use aspect='auto' so matplotlib won’t force a square plot.
            orig_sag_path = os.path.join(output_dir, f"original_slice_sagittal_{idx}.png")
            plt.figure(figsize=(5,5))
            plt.imshow(ct_sagittal, cmap=CMAP_BONE)
            plt.title('Original Sagittal Slice')
            plt.axis('off')
            plt.savefig(orig_sag_path, bbox_inches='tight', dpi=100)
            plt.close()
            original_paths.append(orig_sag_path)

            seg_sag_path = os.path.join(output_dir, f"segmented_slice_sagittal_{idx}.png")
            plt.figure(figsize=(5,5))
            plt.imshow(ct_sagittal, cmap=CMAP_BONE)
            plt.imshow(mask_sagittal, cmap='Reds', alpha=0.5)
            plt.title('Segmented Sagittal Slice')
            plt.axis('off')
            plt.savefig(seg_sag_path, bbox_inches='tight', dpi=100)
            plt.close()
            segmented_paths.append(seg_sag_path)

            grad_sag = generate_gradcam_explanation(
                self.model,
                ct_sagittal,
                output_dir,
                f"sagittal_slice_{idx}",
                layer_name=gradcam_layer
            )
            gradcam_paths.append(grad_sag)

        for idx in tqdm(valid_coronal, desc='Generating Coronal XAI Images'):
            ct_coronal = volume[:, idx, :]
            ct_coronal = cv2.resize(ct_coronal, (512, 512))  # Ensure resizing
            ct_coronal = np.flipud(ct_coronal)
            mask_coronal = processed_mask[:, idx, :]
            mask_coronal = np.flipud(mask_coronal)
            mask_coronal = cv2.resize(mask_coronal.astype(np.float32), (512, 512)) > 0.5

            ct_coronal = np.squeeze(ct_coronal)
            if len(ct_coronal.shape) == 3:
                ct_coronal = cv2.resize(ct_coronal, (512, 512))

            orig_cor_path = os.path.join(output_dir, f"original_slice_coronal_{idx}.png")
            plt.figure(figsize=(5,5))
            plt.imshow(ct_coronal, cmap=CMAP_BONE)
            plt.title('Original Coronal Slice')
            plt.axis('off')
            plt.savefig(orig_cor_path, bbox_inches='tight', dpi=100)
            plt.close()
            original_paths.append(orig_cor_path)

            seg_cor_path = os.path.join(output_dir, f"segmented_slice_coronal_{idx}.png")
            plt.figure(figsize=(5,5))
            plt.imshow(ct_coronal, cmap=CMAP_BONE)
            plt.imshow(mask_coronal, cmap='Reds', alpha=0.5)
            plt.title('Segmented Coronal Slice')
            plt.axis('off')
            plt.savefig(seg_cor_path, bbox_inches='tight', dpi=100)
            plt.close()
            segmented_paths.append(seg_cor_path)

            grad_cor = generate_gradcam_explanation(
                self.model,
                ct_coronal,
                output_dir,
                f"coronal_slice_{idx}",
                layer_name=gradcam_layer
            )
            gradcam_paths.append(grad_cor)
