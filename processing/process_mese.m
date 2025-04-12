Model = mono_t2;

EchoTime  = [12.8000; 25.6000; 38.4000; 51.2000; 64.0000; 76.8000; 89.6000; 102.4000; 115.2000; 128.0000; 140.8000; 153.6000; 166.4000; 179.2000; 192.0000; 204.8000; 217.6000; 230.4000; 243.2000; 256.0000; 268.8000; 281.6000; 294.4000; 307.2000; 320.0000; 332.8000; 345.6000; 358.4000; 371.2000; 384.0000];
% EchoTime (ms) is a vector of [30X1]
Model.Prot.SEdata.Mat = [ EchoTime ];

%          |- mono_t2 object needs 2 data input(s) to be assigned:
%          |-   SEdata
%          |-   Mask

data = struct();
% SEdata.nii.gz contains [260  320    1   30] data.
data.SEdata=double(load_nii_data('mono_t2_data/SEdata.nii.gz'));
% Mask.nii.gz contains [260  320] data.
data.Mask=double(load_nii_data('mono_t2_data/Mask.nii.gz'));

FitResults = FitData(data,Model,0);

% Generic function call to save nifti outputs
FitResultsSave_nii(FitResults, 'reference/nifti/file.nii.(gz)');

Model.saveObj('my_mono_t2_config.qmrlab.mat');
