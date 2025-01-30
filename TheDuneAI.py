import SimpleITK as sitk
import numpy as np
import keras
import cv2
import time
import os
import lung_extraction_funcs as le
import matplotlib.pyplot as plt
import generator
from tqdm import tqdm
from xai import generate_lime_explanations, generate_gradcam_explanation, generate_combined_report, CMAP_BONE
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyXAISelector:
    def __init__(self):
        # Input variables
        self.confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
        self.complexity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'complexity')

        # Output variable
        self.xai_method = ctrl.Consequent(np.arange(0, 2.1, 0.1), 'xai_method', defuzzify_method='centroid')

        #Membership functions
        self._define_membership_functions()
        self._create_rules()
        self.system = ctrl.ControlSystem(self.rules)
        self.decision_maker = ctrl.ControlSystemSimulation(self.system)

    def _define_membership_functions(self):
        # Confidence levels
        self.confidence['low'] = fuzz.trimf(self.confidence.universe, [0, 0, 0.5])
        self.confidence['medium'] = fuzz.trimf(self.confidence.universe, [0.3, 0.5, 0.7])
        self.confidence['high'] = fuzz.trimf(self.confidence.universe, [0.5, 1, 1])

        # Image complexity (example metric)
        self.complexity['simple'] = fuzz.trimf(self.complexity.universe, [0, 0, 0.6])
        self.complexity['complex'] = fuzz.trimf(self.complexity.universe, [0.4, 1, 1])

        # XAI method choices: 0=Grad-CAM, 1=Both, 2=LIME
        self.xai_method['gradcam'] = fuzz.trimf(self.xai_method.universe, [0, 0, 1])
        self.xai_method['both'] = fuzz.trimf(self.xai_method.universe, [0.5, 1, 1.5])
        self.xai_method['lime'] = fuzz.trimf(self.xai_method.universe, [1, 2, 2])

    def _create_rules(self):
        self.rules = [
            ctrl.Rule(self.confidence['high'] & self.complexity['simple'], self.xai_method['gradcam']),  # Best and fastest method
            ctrl.Rule(self.confidence['low'] | self.complexity['complex'], self.xai_method['both']),  # Balanced between methods
            ctrl.Rule(self.confidence['medium'], self.xai_method['both'])
        ]

    def decide(self, confidence, complexity):
        """Return selected XAI methods based on fuzzy evaluation."""
        self.decision_maker.input['confidence'] = confidence
        self.decision_maker.input['complexity'] = complexity
        self.decision_maker.compute()
        return self.decision_maker.output['xai_method']

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

        return final_mask.astype(np.int8)

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
        valid_slices = np.where(np.any(segmentation_mask, axis=(1, 2)))[0]

        # Adjust for preprocessing crops
        if 'z_st' in processing_params:
            valid_slices -= processing_params['z_st']
            valid_slices = valid_slices[(valid_slices >= 0) & (valid_slices < volume.shape[0])]

        return valid_slices[:3]  # Return max 3 slices

    def segment(self):
        """Main pipeline execution method."""
        fuzzy_selector = FuzzyXAISelector()

        for volume, _, file_path, params in tqdm(self.patient_generator, desc='Processing patients'):
            volume_array = np.squeeze(volume[0])  # Remove batch dimension
            processing_params = params[0]
            file_path = file_path[0]

            # Generate segmentation
            segmentation_mask = self._generate_segmentation(volume_array, processing_params)
            output_dir = self._save_results(segmentation_mask, processing_params, file_path)
            explanation_slices = self._get_explanation_slices(volume_array, segmentation_mask, processing_params)

            if not explanation_slices.size:
                print(f"No valid slices found for {file_path}")
                continue

            avg_confidence = np.mean(volume_array)
            complexity = np.var(volume_array)

            # Get fuzzy decision
            xai_decision = fuzzy_selector.decide(avg_confidence, complexity)
            use_lime = xai_decision >= 1.0
            use_gradcam = xai_decision <= 1.5

            # Generate explanations conditionally
            self._generate_xai_reports(
                volume_array,
                explanation_slices,
                output_dir,
                use_lime=use_lime,
                use_gradcam=use_gradcam
            )

            # Select explanation slices
            explanation_slices = self._get_explanation_slices(volume_array, segmentation_mask, processing_params)
            if not explanation_slices.size:
                print(f"No valid slices found for {file_path}")
                continue

            # Generate explanations
            self._generate_xai_reports(volume_array, explanation_slices, output_dir)

    def _generate_xai_reports(self, volume, slice_indices, output_dir, use_lime=True, use_gradcam=True):
        """Generate and save XAI explanations for selected slices."""
        original_paths, lime_paths, gradcam_paths, metrics = [], [], [], []

        for slice_idx in slice_indices:
            ct_slice = volume[slice_idx]
            lung_mask = (ct_slice > -500).astype(np.uint8)  # Simple lung tissue threshold

            # Save original slice separately
            orig_path = os.path.join(output_dir, f"original_slice_{slice_idx}.png")
            plt.figure(figsize=(5, 5))
            plt.imshow(ct_slice, cmap=CMAP_BONE)
            plt.title('Original Slice')  # Add matching title
            plt.axis('off')
            plt.savefig(orig_path, bbox_inches='tight', dpi=100)  # Match dpi with other images
            plt.close()
            original_paths.append(orig_path)

            slice_metrics = {}
            lime_path = gradcam_path = None

            if use_lime:
                slice_metrics, lime_path = generate_lime_explanations(
                    self.model, ct_slice, lung_mask, f"slice_{slice_idx}", output_dir
                )

            if use_gradcam:
                gradcam_path = generate_gradcam_explanation(
                    self.model, ct_slice, output_dir, f"slice_{slice_idx}", layer_name="conv2d_16"
                )

            lime_paths.append(lime_path)
            gradcam_paths.append(gradcam_path)
            metrics.append(slice_metrics)

        generate_combined_report(original_paths, lime_paths, gradcam_paths, metrics, output_dir)
