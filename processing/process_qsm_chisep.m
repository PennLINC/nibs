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
function run_chisep_script(magnitude_file, phase_file, sepia_header_file, out_dir, r2_file)
    % Set x-separation tool directory path
    home_directory ='/home/tsalo/nibs/sepia/chi-separation-main/Chisep_Toolbox_v1.0.1';
    addpath(genpath(home_directory))

    % % Set MATLAB tool directory path
    % % xiangruili/dicm2nii (https://kr.mathworks.com/matlabcentral/fileexchange/42997-xiangruili-dicm2nii)
    % addpath(genpath('D:\projects\QSM\sepia\xiangruili-dicm2nii-3fe1a27\dicm2nii'))
    % % Tools for NIfTI and ANALYZE image (https://kr.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
    addpath(genpath('/home/tsalo/nibs/sepia/mritools_ubuntu-20.04_3.6.4'))
    addpath(genpath('/home/tsalo/nibs/sepia/xiangruili-dicm2nii-3fe1a27'))
    addpath(genpath('/home/tsalo/nibs/sepia/NIfTI_20140122'))
    addpath(genpath('/home/tsalo/nibs/sepia/STISuite_V3.0'))
    addpath(genpath('/home/tsalo/nibs/sepia/SEGUE_28012021'))

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
    mag_multi_echo = niftiread(magnitude_file);
    phs_multi_echo = niftiread(phase_file);
    info = niftiinfo(magnitude_file);

    if exist(r2_file, 'file') == 2
        r2 = niftiread(r2_file);
    end

    %% Data parameters
    % B0_strength, B0_direction, CF (central frequency), TE (echo time), delta_TE, voxel_size
    sepia_header = load(sepia_header_file);
    voxel_size = sepia_header.voxelSize;
    B0_direction = sepia_header.B0_dir;
    TE = sepia_header.TE;
    delta_TE = sepia_header.delta_TE;
    CF = sepia_header.CF;

    %% Preprocessing
    % Tukey windowing
    tukey_fac = 0.4;  % Recommendation: Siemens, GE: 0.4, Philips: 0
    img_tukey = tukey_windowing(mag_multi_echo .* exp(1i * phs_multi_echo), tukey_fac);
    mag_multi_echo = abs(img_tukey);
    phs_multi_echo = angle(img_tukey);
    % Compute single magnitude data
    mag = sqrt(sum(abs(mag_multi_echo).^2, 4));
    [~, N_std] = Preprocessing4Phase(mag_multi_echo, phs_multi_echo);

    %% Generate Mask
    % BET from MEDI toolbox
    matrix_size = size(mag);
    mask_brain = BET(mag, matrix_size, voxel_size);

    %% R2* mapping
    % Compute R2* (need multi-echo GRE magnitude)
    if(use_arlo(TE))
        % Use ARLO (More than three equi-spaced TE needed)
        r2star = r2star_arlo(mag_multi_echo, TE*1000, mask_brain); % Convert TE to [ms]
    else
        % Use NNLS fitting (When ARLO is not an option)
        r2star = r2star_nnls(mag_multi_echo, TE*1000, mask_brain); % Convert TE to [ms]
    end

    have_r2 = exist('r2', 'var');
    if have_r2
        r2prime = r2 - r2star;
        map = r2prime;
        r2prime_file = fullfile(out_dir, "r2prime.nii");
        niftiwrite(r2prime, r2prime_file, info);
    else
        map = r2star;
        r2star_file = fullfile(out_dir, "r2star.nii");
        niftiwrite(r2star, r2star_file, info);
    end

    %% Phase unwrapping and Echo combination
    % 1. ROMEO + weighted echo averaging
    parameters.TE = TE * 1000; % Convert to ms
    parameters.mag = mag_multi_echo;
    parameters.mask = double(mask_brain);
    parameters.calculate_B0 = false;
    parameters.phase_offset_correction = 'on';
    parameters.voxel_size = voxel_size;
    parameters.additional_flags = '-q';
    parameters.output_dir = fullfile(out_dir, 'romeo_tmp');
    mkdir(parameters.output_dir);
    [unwrapped_phase, B0] = ROMEO(double(phs_multi_echo), parameters);
    unwrapped_phase(isnan(unwrapped_phase))= 0;

    % Weighted echo averaging
    t2s_roi = 0.04; % in [s] unit
    W = (TE) .* exp(-(TE) / t2s_roi);
    weightedSum = 0;
    TE_eff = 0;
    for echo = 1:size(unwrapped_phase, 4)
        weightedSum = weightedSum + W(echo) * unwrapped_phase(:, :, :, echo) ./ sum(W);
        TE_eff = TE_eff + W(echo) * (TE(echo)) ./ sum(W);
    end

    field_map = weightedSum / TE_eff * delta_TE .* mask_brain; % Tissue phase in rad

    %% Background field removal
    % V-SHARP from STI Suite
    smv_size = 12;
    [local_field, mask_brain_new] = V_SHARP(field_map, mask_brain, 'voxelsize', voxel_size, 'smvsize', smv_size);
    local_field_hz = local_field / (2 * pi * delta_TE); % rad to hz

    %% χ-separation
    Dr = 114; % This parameter is different from the original paper (Dr = 137) because the network is trained on COSMOS-reconstructed maps
    [x_para, x_dia, x_tot] = chi_sepnet_general(home_directory, local_field_hz, map, mask_brain_new, Dr, B0_direction, CF, voxel_size, have_r2);
    info.Datatype = class(x_para);

    paramagnetic_file = fullfile(out_dir, "Paramagnetic.nii");
    niftiwrite(x_para, paramagnetic_file, info);
    diamagnetic_file = fullfile(out_dir, "Diamagnetic.nii");
    niftiwrite(x_dia, diamagnetic_file, info);
    total_susceptibility_file = fullfile(out_dir, "Total_susceptibility.nii");
    niftiwrite(x_tot, total_susceptibility_file, info);
    clearvars x_dia x_para;

    end
