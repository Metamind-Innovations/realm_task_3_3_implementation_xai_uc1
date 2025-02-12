import numpy as np
from pmtool.ToolBox import ToolBox

# # Download data if not already downloaded
# ds_metadata = nbia.getSeries(collection = "NSCLC-Radiomics")
# df = nbia.downloadSeries(ds_metadata, path ='data/dcms/', format = "csv", number=3)  # 'number=3' downloads 3 samples

parameters = {
    'data_path': 'test_data_radiomics/',  # Root directory containing DICOM data
    'data_type': 'dcm',  # Specify that data is in DICOM format
    'multi_rts_per_pat': False  # Use one RTStruct/SEG per patient
}

toolbox = ToolBox(**parameters)

dataset_description = toolbox.get_dataset_description()
print(dataset_description.head(10))

print('Unique modalities found: ', np.unique(dataset_description.Modality.values)[0])

qc_params = {'specific_modality': 'ct',  # target modality: CT
             'thickness_range': [2, 5],  # slice thickness should be in range of 2..5 mm
             'spacing_range': [0.5, 1.25],  # pixel spacing should be in range of 0.5..1.25 mm
             'scan_length_range': [5, 170],  # scan should contain from 5 to 170 slices
             'axial_res': [512, 512],  # the axial resolution should be 512x512
             'kernels_list': ['standard', 'lung', 'b19f']}  # the following kernels are acceptable

qc_dataframe = toolbox.get_quality_checks(qc_params)
print(qc_dataframe)

# Convert DICOM dataset to NRRD
export_path = './'
toolbox.convert_to_nrrd(export_path)
