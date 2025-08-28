%% χ-separation Tool

% This tool is MATLAB-based software forseparating para- and dia-magnetic susceptibility sources (χ-separation). 
% Separating paramagnetic (e.g., iron) and diamagnetic (e.g., myelin) susceptibility sources 
% co-existing in a voxel provides the distributions of two sources that QSM does not provides. 

% χ-separation tool v1.0

% Contact E-mail: snu.list.software@gmail.com 

% Reference
% H.-G. Shin, J. Lee, Y. H. Yun, S. H. Yoo, J. Jang, S.-H. Oh, Y. Nam, S. Jung, S. Kim, F. Masaki, W. 
% Kim, H. J. Choi, J. Lee. χ-separation: Magnetic susceptibility source separation toward iron and 
% myelin mapping in the brain. Neuroimage, 2021 Oct; 240:118371.

% χ-separation tool is powered by MEDI toolbox (for BET and Complex data fitting), STI Suite (for V-SHARP), SEGUE toolbox (for SEGUE), and mritools (for ROMEO).


%% Example
% This example reconstructs χ-separation maps from multi-echo gradient 
% echo magnitude and phase data

%% Necessary preparation
function Chisep_scriptshell(input,output)
% Set x-separation tool directory path
 home_directory ='/project/ftdc_misc/spandey/sepia/chi-separation-main/Chisep_Toolbox_v1.0.1';
addpath(genpath(home_directory))
% 
% % Set MATLAB tool directory path 
% % xiangruili/dicm2nii (https://kr.mathworks.com/matlabcentral/fileexchange/42997-xiangruili-dicm2nii)
% addpath(genpath('D:\projects\QSM\sepia\xiangruili-dicm2nii-3fe1a27\dicm2nii'))
% % Tools for NIfTI and ANALYZE image (https://kr.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
addpath(genpath('/project/ftdc_misc/spandey/sepia/mritools_ubuntu-20.04_3.6.4'))
addpath(genpath('/project/ftdc_misc/spandey/sepia/xiangruili-dicm2nii-3fe1a27'))
addpath(genpath('/project/ftdc_misc/spandey/sepia/NIfTI_20140122'))
addpath(genpath('/project/ftdc_misc/spandey/sepia/STISuite_V3.0'))
addpath(genpath('/project/ftdc_misc/spandey/sepia/SEGUE_28012021'))
% 
% % Download onnxconverter Add-on, and then install it.
% % Deep Learning Toolbox Converter for ONNX Model Format 
% % (https://kr.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format)
% 
% % Set QSM tool directory path 
% % STI Suite (Version 3.0) (https://people.eecs.berkeley.edu/~chunlei.liu/software.html)
% addpath(genpath('D:\projects\QSM\sepia\STISuite_V3.0\STISuite_V3.0'))
% 
% % MEDI toolbox (http://pre.weill.cornell.edu/mri/pages/qsm.html)
% addpath(genpath('D:\projects\QSM\sepia\MEDI_toolbox'))
% 
% % SEGUE toolbox (https://xip.uclb.com/product/SEGUE)
% addpath(genpath('D:\projects\QSM\sepia\SEGUE_28012021'))
% 
% % mritools toolbox (https://github.com/korbinian90/CompileMRI.jl/releases)
% addpath(genpath('D:\projects\QSM\sepia\mritools_windows-2019_4.0.6'));
% addpath(genpath('D:\projects\QSM\analysis\chi-separation-main\chi-separation-main\Chisep_Toolbox_v1.0.1'));
%% Input data
% mag_multi_echo - multi-echo magnitude data (x, y, z, te)
% phs_multi_echo - multi-echo phase data (x, y, z, te) 
% temp = pwd;
% input1 = 'D:\projects\QSM\delete\flywheel\anat';
% input2 = 'D:\projects\QSM\delete\flywheel\output';
% output = 'D:\projects\QSM\delete\flywheel\output';
% cd(input);
% b = dir;
info = niftiinfo(input);
% cd(temp);
mag_multi_echo = niftiread([output,'/Sepia_part-mag.nii.gz']);
phs_multi_echo = niftiread([output,'/Sepia_part-phase.nii.gz']);
% mag_multi_echo = niftiread('D:\projects\QSM\7T\sub-127315\flywheel\output\Sepia_part-mag.nii\Sepia_part-mag.nii');
% phs_multi_echo = niftiread('D:\projects\QSM\7T\sub-127315\flywheel\output\Sepia_part-phase.nii\Sepia_part-phase.nii');
% r2prime - R2' map in Hz unit (x, y, z). If you don't have R2' map, use
% chi-sepnet-R2* which doesn't require R2' map.

%% Data paramters
% B0_strength, B0_direction, CF (central frequency), TE (echo time), delta_TE, voxel_size
% load('D:\projects\QSM\7T\sub-127315\flywheel\output\Sepia_header.mat');
load([output,'/Sepia_header.mat']);
% B0_direction = [0, 0, 1];
% CF = 123200000; in Hz unit
% TE = [0.00xx, 0.00xx, 0.00xx, 0.00xx, ...]; % in [s] unit
% delta_TE = 0.00xx; % in [s] unit
% voxel_size = [dx, dy, dz]; % in [mm] unit
voxel_size = voxelSize;
B0_direction = B0_dir;
%% Preprocessing
% Tukey windowing
tukey_fac = 0.4; % Recommendation: Siemens, GE: 0.4, Philips: 0
img_tukey = tukey_windowing(mag_multi_echo.* exp(1i*phs_multi_echo),tukey_fac);
mag_multi_echo = abs(img_tukey);
phs_multi_echo = angle(img_tukey);
% Compute single magnitude data
mag = sqrt(sum(abs(mag_multi_echo).^2,4));
[~, N_std] = Preprocessing4Phase(mag_multi_echo, phs_multi_echo);

%% Generate Mask
% BET from MEDI toolbox

matrix_size = size(mag);
mask_brain = BET(mag,matrix_size,voxel_size);

%% R2* mapping
% Compute R2* (need multi-echo GRE magnitude)

if(use_arlo(TE))
    % Use ARLO (More than three equi-spaced TE needed)
    r2star = r2star_arlo(mag_multi_echo,TE*1000,mask_brain); % Convert TE to [ms]
else
% Use NNLS fitting (When ARLO is not an option)
    r2star = r2star_nnls(mag_multi_echo,TE*1000,mask_brain); % Convert TE to [ms]
end

%% Compute CSF mask (Needed for Chi-separation-MEDI and Chi-separation iLSQR)
% mask_CSF = extract_CSF(r2star, mask_brain_new, voxel_size);

%% Phase unwrapping and Echo combination
% 1. ROMEO + weighted echo averaging 
parameters.TE = TE * 1000; % Convert to ms
parameters.mag = mag_multi_echo;
parameters.mask = double(mask_brain);
parameters.calculate_B0 = false;
parameters.phase_offset_correction = 'on';
parameters.voxel_size = voxel_size;
parameters.additional_flags = '-q';%'--verbose -q -i'; % settings are pasted directly to ROMEO cmd (see https://github.com/korbinian90/ROMEO for options)
parameters.output_dir = 'romeo_tmp'; % if not set pwd() is used
mkdir(parameters.output_dir);
disp(parameters.output_dir)
[unwrapped_phase, B0] = ROMEO(double(phs_multi_echo), parameters);
disp(parameters.output_dir)
unwrapped_phase(isnan(unwrapped_phase))= 0;
disp(parameters.output_dir)
% Weighted echo averaging
t2s_roi = 0.04; % in [s] unit
W = (TE).*exp(-(TE)/t2s_roi);
weightedSum = 0;
TE_eff = 0;
for echo = 1:size(unwrapped_phase,4)
    weightedSum = weightedSum + W(echo)*unwrapped_phase(:,:,:,echo)./sum(W);
    TE_eff = TE_eff + W(echo)*(TE(echo))./sum(W);
end

field_map = weightedSum/TE_eff*delta_TE.*mask_brain; % Tissue phase in rad

% 2. Complex data fitting + SEGUE
% Complex fitting from MEDI
% [field, error, residual_, phase0]=Fit_ppm_complex_TE(mag_multi_echo.*exp(-1i*phs_multi_echo), TE/1000);
% 
% 
% Inputs.Mask = double(mask_brain); % 3D binary tissue mask, same size as one phase image
% Inputs.Phase = double(field); % For opposite phase
% field_map = SEGUE(Inputs).*mask_brain; % Tissue phase in rad

%% Background field removal
% V-SHARP from STI Suite
smv_size=12;
[local_field, mask_brain_new]=V_SHARP(field_map, mask_brain,'voxelsize',voxel_size,'smvsize',smv_size);
local_field_hz = local_field / (2*pi*delta_TE); % rad to hz

%% QSM
% 1. iLSQR from STI Suite

% pad_size = [12, 12, 12];
% QSM = QSM_iLSQR(local_field,mask_brain_new,'TE',delta_TE*1e3,'B0',B0_strength,'H',B0_direction,'padsize',pad_size,'voxelsize',voxel_size);


%% χ-separation

% 1. Chi-separation-MEDI
% regularization parameters for chi_separation_MEDI

% params.b0_dir = B0_direction;
% params.CF = CF;
% params.voxel_size = voxel_size;
% params.lambda = 1;
% params.lambda_CSF = 1;
% option_data.qsm = QSM; 
% option_data.mask_CSF = mask_CSF;
% option_data.N_std = N_std;
% [x_para, x_dia, x_tot] = chi_sep_MEDI(mag, local_field_hz, r2prime, N_std, mask_brain_new, params, option_data);


% 2. Chi-separation-iLSQR
% parameters for chi_separation_iLSQR

% params.b0_dir = B0_direction;
% params.CF = CF;
% params.voxel_size = voxel_size;
% option_data.qsm = QSM;
% option_data.N_std = N_std;
% [x_para, x_dia, x_tot] = chi_sep_iLSQR(mag, local_field_hz, r2prime, mask_brain_new, params, option_data);


% 3. chi_sepnet

have_r2prime = exist('r2prime','var');

if have_r2prime
    % Use Chi-sepnet-R2'
    map = r2prime;
else
    % Use Chi-sepnet-R2*
    map = r2star;
end
Dr = 114; % This parameter is different from the original paper (Dr = 137) because the network is trained on COSMOS-reconstructed maps
[x_para, x_dia, x_tot] = chi_sepnet_general(home_directory, local_field_hz, map, mask_brain_new, Dr, B0_direction, CF, voxel_size, have_r2prime);
info.Datatype = class(x_para);
% cd (output)
filepath1 = output + "/Paramagnetic.nii";
niftiwrite(x_para,filepath1,info);
filepath2 = output + "/Diamagnetic.nii";
niftiwrite(x_dia,filepath2,info);
% save([basefolder,'\chi_result.mat'], 'x_para','x_dia');
filepath3 = output + "/r2star.nii";
niftiwrite(map,filepath2,info);
clearvars x_dia x_para;
end

