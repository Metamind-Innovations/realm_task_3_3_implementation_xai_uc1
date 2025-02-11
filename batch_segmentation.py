from TheDuneAI import ContourPilot
import argparse

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Run lung segmentation and xai image heatmap generation with Grad-CAM')
    parser.add_argument('--model_path', required=True,
                        help='Path to model directory containing model_v7.json and weights_v7.hdf5')
    parser.add_argument('--path_to_test_data', required=True, help='Path to input NRRD files directory')
    parser.add_argument('--save_path', required=True, help='Output directory for segmentation/xai results')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                        help='Sensitivity parameter (0.0-1.0). Higher values increase detection sensitivity')

    args = parser.parse_args()

    model = ContourPilot(
        model_path=args.model_path,
        data_path=args.path_to_test_data,
        output_path=args.save_path,
        verbosity=True
    )

    # Run segmentation/xai process
    # If a CUDA GPU is available, it is used automatically by tensorflow
    model.segment()

if __name__=='__main__':
    main()
