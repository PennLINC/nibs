%% χ-separation Tool

% This tool is MATLAB-based software for separating para- and dia-magnetic susceptibility sources (χ-separation).
% Separating paramagnetic (e.g., iron) and diamagnetic (e.g., myelin) susceptibility sources
% co-existing in a voxel provides the distributions of two sources that QSM does not provides.

% χ-separation tool v1.2

% Contact E-mail: snu.list.software@gmail.com

% Reference
% H.-G. Shin, J. Lee, Y. H. Yun, S. H. Yoo, J. Jang, S.-H. Oh, Y. Nam, S. Jung, S. Kim, F. Masaki, W.
% Kim, H. J. Choi, J. Lee. χ-separation: Magnetic susceptibility source separation toward iron and
% myelin mapping in the brain. Neuroimage, 2021 Oct; 240:118371.

% χ-separation tool is powered by MEDI toolbox (for BET and Complex data fitting), STI Suite (for V-SHARP), SEGUE toolbox (for SEGUE), and mritools (for ROMEO).
% Code by Byeongpil Moon (mbpil7044@snu.ac.kr)

%% Necessary preparation

% Set x-separation tool directory path
home_directory = '/cbica/projects/nibs/software/Chisep_Toolbox_v1.2';
addpath(genpath(home_directory))

% Set MATLAB tool directory path
% xiangruili/dicm2nii (https://kr.mathworks.com/matlabcentral/fileexchange/42997-xiangruili-dicm2nii)
% TS: Hopefully we don't need this.
% addpath(genpath('your_path/dicm2nii'))
% Tools for NIfTI and ANALYZE image (https://kr.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
addpath(genpath('your_path/Tools_for_NIfTI_and_ANALYZE_image'))

% Download onnxconverter Add-on, and then install it.
% Deep Learning Toolbox Converter for ONNX Model Format
% (https://kr.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format)

% Set QSM tool directory path
% STI Suite (Version 3.0) (https://people.eecs.berkeley.edu/~chunlei.liu/software.html)
addpath(genpath('/cbica/projects/nibs/software/STISuite'))

% MEDI toolbox (http://pre.weill.cornell.edu/mri/pages/qsm.html)
addpath(genpath('/cbica/projects/nibs/software/MEDItoolbox'))

% SEGUE toolbox (https://xip.uclb.com/product/SEGUE)
addpath(genpath('/cbica/projects/nibs/software/SEGUE'))

% mritools toolbox (https://github.com/korbinian90/CompileMRI.jl/releases)
addpath(genpath('/cbica/projects/nibs/software/mritools'))


%% Run options - User define
RunOptions = struct();
% 'dicom': input DICOM | 'nifti': input NIfTI | Else: custom input (.mat)
RunOptions.InputType = 'nifti';

% 'multi': multiple subjects | 'single: single-subject
RunOptions.multi = 'single';

% true: input brain mask | false: calculate brain mask
RunOptions.Mask = true;

% 'MEDI': MEDI brain extraction | 'custom': customize using FSL BET
RunOptions.Mask_method = 'MEDI';

% 'ARLO' | 'NNLS fitting' | 'Use preprocessed R2* or R2'' map'
RunOptions.R2sfit = 'Use preprocessed R2* or R2'' map';

% 'ROMEO + weighted echo averaging' | 'nonlinear complex fitting + SEGUE' | 'Laplacian'
RunOptions.Unwrap = 'ROMEO + weighted echo averaging';

% 'V-SHARP'
RunOptions.BFR = 'V-SHARP';

% 'Chi-sepnet' | 'Chi-separation (MEDI)' | 'Chi-separation (iLSQR)'
RunOptions.Chisep = 'Chi-sepnet';

% 'Deep-learning' | 'Region-growing' | 'No'
RunOptions.VesselSeg = 'Deep-learning';

% GRE smoothing: 0 ~ 0.4(Default)
RunOptions.Tukey = double(0.4);

% 0: No inverse(Default) | 1: Inverse
RunOptions.PhaseInverse = 0;

% 1: have R2' | 0: don't have R2'
RunOptions.HaveR2Prime = 1;
% r2prime - R2' map in Hz unit (x, y, z). If you don't have R2' map, use chi-sepnet-R2* which doesn't require R2' map.

% 0: generate R2' from R2* using R2pnet | 1: generate R2' from R2* using scaling
RunOptions.is_scaling = 0;
RunOptions.scaling_factor = 0.19;

% false: No denoising for R2s | true: denosing for R2s
RunOptions.denoising = false;

% true: use resolution generalization | false: don't use
RunOptions.resgen = false;
% Determine whether to use resolution generalization pipeline or to interpolate to 1 mm isotropic resolution
% 7T processing is available only with resolution generalization

RunOptions.OutputPath = '/cbica/projects/nibs/derivatives/qsm';
% Output path must not contatin ' '(spaces)

% Interpolation options (for B0 direction, Resampling)
% 'sinc' | 'spline'
RunOptions.interp_method = 'sinc';
RunOptions.sinc_window_size = 15;
% 'hann' | 'hamming' | 'blackman'
RunOptions.sinc_window_type = 'hann';

% Last stage Tukey
RunOptions.tukey_strength = 0.5;
RunOptions.tukey_pad = 0.1; %Recommend not to fix this

Data = struct();
Data.RunOptions = RunOptions;


%% Data input

if strcmp(RunOptions.multi, 'multi')
    % DICOM folder structure: 'multi_subj' > 'subj1', 'subj2', ... > 'iMag' 'iPhase'
    multi_subj_path = 'Multi_subj_path';
    subj_dir = dir([multi_subj_path,'\subj*']);
elseif strcmp(RunOptions.multi, 'single')
    single_subj_path = 'Single_subj_path';
    subj_dir(1).folder = single_subj_path; subj_dir(1).name = [];
end

for subj = 1:length(subj_dir)
if strcmp(RunOptions.InputType, 'dicom')
    pathDICOM = fullfile(subj_dir(subj).folder,subj_dir(subj).name);
    [meas,voxel_size,~,CF,~,TE, B0, B0_dir, dinfo, vendor]=load_from_DICOM(pathDICOM);
    Data.CF = double(CF);
    Data.TE = double(TE*1000);
    Data.B0dir = double(B0_dir);
    Data.VoxelSize = double(voxel_size);
    Data.Vendor = vendor;
    Data.Dinfo = dinfo;
    Data.Necho = size(meas,4);
    Data.MatrixSize = size(meas);
    Data.B0_strength = B0;
    if strcmp(vendor, 'P')
        RunOptions.Tukey = double(0);
    end
    Data.MGRE_Mag = double(abs(meas));
    Data.MGRE_Phs = double(angle(meas));

elseif strcmp(RunOptions.InputType, 'nifti')
    pathNifti_mag = [fullfile(subj_dir(subj).folder, subj_dir(subj).name), '\Mag.nii'];
    pathNifti_phs = [fullfile(subj_dir(subj).folder, subj_dir(subj).name), '\Phase.nii'];
    % magnitude
    nii_file = load_untouch_nii(pathNifti_mag);
    Data.MGRE_Mag = rot90(double(nii_file.img));
    % phase
    nii_file_phs = load_untouch_nii(pathNifti_phs);
    maxval = max(double(nii_file_phs.img(:)));
    minval = min(double(nii_file_phs.img(:)));
    Data.MGRE_Phs = (rot90(double(nii_file_phs.img))-(minval+maxval)/2)/(maxval-minval)*2*pi;
    % info
    VoxelSize_org = double(nii_file.hdr.dime.pixdim(2:4));
    Data.VoxelSize = VoxelSize_org([2,1,3]);
    Data.Necho = size(Data.MGRE_Mag,4);
    Data.CF = 0;
    Data.B0_strength = 0;
    Data.TE = [];
    Data.MatrixSize = size(Data.MGRE_Mag);
    Data.nifti_template = nii_file;
end

Data.output_root = [RunOptions.OutputPath,filesep,'chisep_output_',char(datetime('now','Format',"MM-dd-yy_HH.mm.ss"))];
mkdir(Data.output_root);

clearvars -except Params Data type_dir subj subj_dir path type type_path RunOptions home_directory

%% Fill in necessary parameters if empty
% Data.TE = [];                     % [ms]  [row vector]
% Data.B0dir = [];                  % []    [row vector]
% Data.CF = [];                     % [Hz]
% Data.B0_strength = [];            % [T]   B0_strength = CF / 42.58e6;


%% Params_check

% Force even dimension
input_field = {'MGRE_Mag','MGRE_Phs'};
for i = 1:length(input_field)
    [Data.(cell2mat(input_field(i))),x_odd,y_odd,z_odd] = even_pad(Data.(cell2mat(input_field(i))));
end
RunOptions.EvenSizePadding = [x_odd,y_odd,z_odd];
Data.MatrixSize = size(Data.MGRE_Mag);

% TE shape correction
if size(Data.TE,2) > 1
    Data.TE = Data.TE';
end

% Vendor options
if isfield(Data, 'Vendor') && strcmp(Data.Vendor,'P')
    RunOptions.Tukey = double(0);
    Data.RunOptions = RunOptions;
end
% if isfield(Data, 'Vendor') && (strcmp(Data.Vendor,'G') || strcmp(Data.Vendor,'S'))
%     RunOptions.PhaseInverse = 1;
%     Data.RunOptions = RunOptions;
% end


%% Tukey windowing
imgc = Data.MGRE_Mag .* exp(1i*Data.MGRE_Phs * (-1)^(RunOptions.PhaseInverse));
imgc = tukey_windowing(imgc,RunOptions.Tukey);
Data.MGRE_Mag_Tukey = abs(imgc);
Data.MGRE_Phs_Tukey = angle(imgc);

clearvars imgc


%% Brain mask (Range [0,1])
disp("=================< Brain masking >=================")
if RunOptions.Mask
    Data.Mask = load('mask.mat');
else
    if strcmp(RunOptions.Mask_method,'MEDI')                                % Use MEDI BET
        Data.Mask = BET(Data.MGRE_Mag_Tukey(:,:,:,1), Data.MatrixSize(1:3), Data.VoxelSize);
%         Data.Mask = double(imerode(Data.Mask, strel('sphere',2)));
        Data.Mask = double(Data.Mask);
    else                                                                    % customizing using FSL
        mat2nii_ungz(Data.MGRE_Mag_Tukey,[Data.output_root,'\mag_tmp'])
        cmd = ['/home/user/fsl/bin/bet ',[Data.output_root,'\mag_tmp '],[Data.output_root,'\BET'], ' -m -R -f 0.55 -g 0.15 -S'];%-f 0.7 -g -0.08
        [status, result] = system(fsl_PathCorr(cmd));
        mask_brain = fliplr(rot90(niftiread([Data.output_root,'\BET_mask.nii.gz'])));
        Data.Mask = imerode(imdilate(mask_brain,strel('sphere',2)),strel('sphere',4));
    end
end

clearvars mask_brain


%% R2* fitting (Range [0,100])
disp("==================< R2* fitting >==================")
if strcmp(RunOptions.R2sfit, 'Use preprocessed R2* or R2'' map')            % If you have R2s or R2p
    Data.R2s = load('R2s.mat');
    if RunOptions.HaveR2Prime == 1
        Data.R2p = load('R2p.mat');
    end
else                                                                        % R2s fitting
    if strcmp(RunOptions.R2sfit, 'ARLO')
        Data.dTE = Data.TE(2) - Data.TE(1);
        if (~isempty(find(abs(Data.TE - Data.TE(1) - Data.dTE*(0:length(Data.TE)-1)') > 0.001, 1)) && (length(Data.TE)>2))
            disp("WARNING:ARLO doesn't work!!")
            RunOptions.R2sfit = 'NNLS fitting';
        end
    end

    if strcmp(RunOptions.R2sfit,'ARLO')
        if RunOptions.denoising
            denoised = denoise_SVD(Data.MGRE_Mag_Tukey, Data.Mask);
            Data.R2s = r2star_arlo(denoised,Data.TE,Data.Mask);
        else
            Data.R2s = r2star_arlo(Data.MGRE_Mag_Tukey,Data.TE,Data.Mask);
        end
    elseif strcmp(RunOptions.R2sfit,'NNLS fitting')
        if RunOptions.denoising
            denoised = denoise_SVD(Data.MGRE_Mag_Tukey, Data.Mask);
            Data.R2s = r2star_nnls(denoised,Data.TE,Data.Mask);
        else
            Data.R2s = r2star_nnls(Data.MGRE_Mag_Tukey,Data.TE, Data.Mask);
        end
    end
end
Data.R2s(Data.R2s < 0) = 0;

if RunOptions.HaveR2Prime                                                   % Use Chi-sepnet-R2'
    Data.map = Data.R2p;
else                                                                        % Use Chi-sepnet-R2*
    Data.map = Data.R2s;
end
Data.map(Data.map < 0) = 0;


%% Calculate & correct bias field for Philips data
if isfield(Data,'Vendor')
    if(Data.Vendor == 'P')
        disp('Detecting bias field for Philips data')
        [biasField, detected] =  CustomBiasCorrection_step1(Data.MGRE_Phs_Tukey,logical(Data.Mask),Data.MGRE_Mag_Tukey);
        if detected
            Data.MGRE_Phs_BiasCor = CustomBiasCorrection_step2(Data.MGRE_Phs_Tukey,biasField);
            Data.RunOptions.PhilipsBiasCor = true;
        end
    end
end
if (isfield(Data,'MGRE_Phs_BiasCor'))
    phase = Data.MGRE_Phs_BiasCor;
elseif(isfield(Data,'MGRE_Phs_Tukey'))
    phase = Data.MGRE_Phs_Tukey;
else
    phase = Data.MGRE_Phs;
end


%% Phase Unwrapping (Range [-10,10] [rad])
% [unwrapped_phase[w*TE, angle]-> Echo combine -> UnwrappedPhase[w*dTE, angle]]
disp("================< Phase unwrapping >===============")
if(strcmp(RunOptions.Unwrap,'ROMEO + weighted echo averaging'))
    parameters.TE = Data.TE;
    parameters.mag = Data.MGRE_Mag_Tukey;
    parameters.mask = double(Data.Mask);
    parameters.calculate_B0 = false;
    parameters.phase_offset_correction = 'on';
    parameters.voxel_size = Data.VoxelSize;
    parameters.additional_flags = '-q';
    parameters.output_dir = [Data.output_root,'\romeo_tmp'];
    mkdir(parameters.output_dir);

    [unwrapped_phase, B0] = ROMEO(double(phase), parameters);
    unwrapped_phase(isnan(unwrapped_phase))= 0;

    % Weighted echo averaging
    TE_s = Data.TE/1000;
    t2s_roi = 0.04;
    W = (TE_s).*exp(-(TE_s)/t2s_roi);
    weightedSum = 0;
    TE_eff = 0;
    for echo = 1:size(unwrapped_phase,4)
        weightedSum = weightedSum + W(echo)*unwrapped_phase(:,:,:,echo)./sum(W);
        TE_eff = TE_eff + W(echo)*TE_s(echo)./sum(W);
    end

    Data.UnwrappedPhase = weightedSum / TE_eff * (TE_s(2)-TE_s(1)) .* Data.Mask;

elseif(strcmp(RunOptions.Unwrap,'nonlinear complex fitting + SEGUE'))
    % Complex fitting from MEDI
    [field, error, residual_, phase0]=Fit_ppm_complex_TE(Data.MGRE_Mag_Tukey.*exp(-1i*phase), Data.TE);

    Inputs.Mask = double(Data.Mask); % 3D binary tissue mask, same size as one phase image
    Inputs.Phase = double(field); % For opposite phase
    Data.UnwrappedPhase = SEGUE(Inputs) .* Data.Mask; % Tissue phase in rad

elseif(strcmp(RunOptions.Unwrap,'Laplacian'))
    % Weighted echo combine + Laplacian
    [phase, N_std] = Preprocessing4Phase(Data.MGRE_Mag_Tukey,Data.MGRE_Phs_Tukey);
    pad_size=[12 12 12];
    [Data.UnwrappedPhase_, ~] = MRPhaseUnwrap(phase,'voxelsize',Data.VoxelSize,'padsize',pad_size);
    Data.UnwrappedPhase = Data.UnwrappedPhase / Data.dTE;

    % Laplacian + Echo sum
    pad_size=[12 12 12];
    [Data.UnwrappedPhase_, ~] = MRPhaseUnwrap(Data.MGRE_Phs_Tukey,'voxelsize',Data.VoxelSize,'padsize',pad_size);
    Data.UnwrappedPhase = sum(Data.UnwrappedPhase_,4) / sum(Data.TE);

    clearvars field_map pad_size
end


%% Background field removal (Range [-5,5])
% [local_field_hz [hz]]
disp("============< Background field removal >============")
if(strcmp(RunOptions.BFR,'V-SHARP'))
    [Data.local_field, Data.mask_brain_new]=V_SHARP(Data.UnwrappedPhase, Data.Mask,'voxelsize', Data.VoxelSize,'smvsize', 12);
    Data.delta_TE = (Data.TE(2)-Data.TE(1))/1000;
    Data.local_field_hz = double(Data.local_field) / (2*pi*Data.delta_TE); % rad to hz
end

%% Compute CSF mask
% Needed for Chi-separation-MEDI, Chi-separation iLSQR,
% and Region-growing algorithm-based vessel segmentation
Data.mask_CSF = extract_CSF(Data.R2s, Data.mask_brain_new, Data.VoxelSize);

%% QSM
% % 1. iLSQR from STI Suite
% pad_size = [12, 12, 12];
% Data.QSM = QSM_iLSQR(Data.local_field, Data.mask_brain_new,'TE',Data.delta_TE*1e3,'B0',Data.B0_strength,'H',Data.B0dir,'padsize',pad_size,'voxelsize',Data.VoxelSize');


%% Chi separation
disp("============< χ-separation processing >============")
switch RunOptions.Chisep
    case 'Chi-sepnet'
        Dr = 114; % This parameter is different from the original paper (Dr = 137) because the network is trained on COSMOS-reconstructed maps
        if RunOptions.resgen
            % Use the resolution generalization pipeline. Resolution of input data is retained in the resulting chi-separation maps
            [Data.x_para, Data.x_dia, Data.x_tot, Data.qsm_map, Data.r2p_map] = chi_sepnet_general_new_wResolGen(home_directory, Data.local_field_hz, Data.map, Data.mask_brain_new, Dr, ...
                Data.B0dir, Data.CF, Data.VoxelSize, RunOptions.HaveR2Prime, Data.B0_strength, RunOptions.is_scaling, RunOptions.scaling_factor, RunOptions.interp_method, RunOptions.sinc_window_size, RunOptions.sinc_window_type);
        else
            % Interpolate the input maps to 1 mm isotropic resolution.
            [Data.x_para, Data.x_dia, Data.x_tot, Data.qsm_map, Data.r2p_map] = chi_sepnet_general_sinc(home_directory, Data.local_field_hz, Data.map, Data.mask_brain_new, Dr, ...
                Data.B0dir, Data.CF, Data.VoxelSize, RunOptions.HaveR2Prime, Data.B0_strength, RunOptions.is_scaling, RunOptions.scaling_factor, RunOptions.interp_method, RunOptions.sinc_window_size, RunOptions.sinc_window_type);
        end

    case 'Chi-separation (MEDI)'
        Data.mag = sqrt(sum(Data.MGRE_Mag_Tukey.^2,4)) .* Data.mask_brain_new;
        Data.local_field_hz = Data.local_field_hz .* Data.mask_brain_new;
        Data.r2prime = Data.map .* Data.mask_brain_new;
        [~, N_std] = Preprocessing4Phase(Data.MGRE_Mag_Tukey, Data.MGRE_Phs_Tukey);
        params.b0_dir = Data.B0dir;
        params.CF = Data.CF;
        params.voxel_size = Data.VoxelSize;
        params.TE = Data.TE;
        params.lambda = 1;
        params.lambda_CSF = 1;
        params.Dr = 137;
        option_data.qsm = Data.QSM;
        option_data.mask_CSF = Data.mask_CSF;
        option_data.N_std = N_std;
        option_data.wG = [];
        option_data.wG_r2p = [];
        option_data.mask_FastRelax = zeros(size(Data.r2prime));
        option_data.mask_SlowRelax = zeros(size(Data.r2prime));
        [Data.x_para, Data.x_dia, Data.x_tot] = chi_sep_MEDI(Data.mag, Data.local_field_hz, Data.r2prime, N_std, Data.mask_brain_new, params, option_data);

    case 'Chi-separation (iLSQR)'
        Data.mag = sqrt(sum(Data.MGRE_Mag_Tukey.^2,4)) .* Data.mask_brain_new;
        Data.local_field_hz = Data.local_field_hz .* Data.mask_brain_new;
        Data.r2prime = Data.map .* Data.mask_brain_new;
        [~, N_std] = Preprocessing4Phase(Data.MGRE_Mag_Tukey, Data.MGRE_Phs_Tukey);
        params.b0_dir = Data.B0dir;
        params.CF = Data.CF;
        params.voxel_size = Data.VoxelSize;
        params.Dr = 137;
        option_data.qsm = Data.QSM;
        option_data.N_std = N_std;
        [Data.x_para, Data.x_dia, Data.x_tot] = chi_sep_iLSQR(Data.mag, Data.local_field_hz, Data.r2prime, Data.mask_brain_new, params, option_data);

end


if strcmp(run_options.interp_method, 'sinc')
    tukey_strength = run_options.tukey_strength;
    tukey_pad = run_options.tukey_pad;
    x_para = real(tukey_windowing(x_para,tukey_strength,round(size(x_para).*tukey_pad))) .* mask;
    x_dia = real(tukey_windowing(x_dia,tukey_strength,round(size(x_dia).*tukey_pad))) .* mask;
    x_tot = real(tukey_windowing(x_tot,tukey_strength,round(size(x_tot).*tukey_pad))) .* mask;
    qsm_map = real(tukey_windowing(qsm_map,tukey_strength,round(size(qsm_map).*tukey_pad))) .* mask;
    r2p_map = real(tukey_windowing(r2p_map,tukey_strength,round(size(r2p_map).*tukey_pad))) .* mask;

    x_para(x_para < 0) = 0;
    x_dia(x_dia < 0) = 0;
    r2p_map(r2p_map < 0) = 0;
end


%% Vessel Segmentation
disp("==============< Vessel segmentation >==============")
switch RunOptions.VesselSeg
    case 'Deep-learning'
        [Data.vesselMask_para, Data.vesselMask_dia] = vesselSegmentation_Chiseparation_DL(home_directory, Data.x_para, Data.x_dia, Data.mask_brain_new, Data.VoxelSize);

    case 'Region-growing'
        % Params for vessel enhancement filter (MFAT, Default)
        params.tau = 0.02; params.tau2 = 0.35; params.D = 0.3;
        params.spacing = Data.VoxelSize;
        params.scales = 4; params.sigmas = [0.25,0.5,0.75,1];
        params.whiteondark = true;

        % params for Seed Generation
        params.alpha = 2; % Threshold for large vessel seeds
        params.beta = 1; % Threshold for small vessel seeds
        params.mipSlice = round(16 / params.spacing(3) / 2) * 2;
        params.overlap = params.mipSlice / 2;

        % params for Region Growing
        params.limit = [0.5, -0.5]; %% gamma1 and gamma2
        params.Aniso_Thresh = 0.0012;
        params.similarity = 0.5; % see (Eq. 3)

        seedInput.img1 = Data.R2s; seedInput.img2 = Data.x_para .* Data.x_dia;
        baseInput.img1 = Data.x_para; baseInput.img2 = Data.x_dia;

        [paraMask_init, diaMask_init, homogeneityMeasure_p, homogeneityMeasure_d] = ...
                    vesselSegmentation_Chiseparation(seedInput, baseInput, Data.mask_brain_new, min(Data.mask_brain_new, 1 - Data.mask_CSF), params);
        Data.vesselMask_para = filterVesselsByAnisotropy(paraMask_init, homogeneityMeasure_p, params.Aniso_Thresh);
        Data.vesselMask_dia  = filterVesselsByAnisotropy(diaMask_init, homogeneityMeasure_d, params.Aniso_Thresh);

        clear paraMask_init diaMask_init homogeneityMeasure_p homogeneityMeasure_d seedInput baseInput

    case 'No'
        % No vessel segmentation
end


%% Save
disp("==================< Saving data >==================")
if ~(sum(RunOptions.EvenSizePadding) == 0)
    input_field = {'x_para', 'x_dia', 'x_tot','qsm_map','Mask','R2s','R2p','UnwrappedPhase','local_field_hz','mask_brain_new','vesselMask_para','vesselMask_dia'};
    for i = 1:length(input_field)
        if isfield(Data,cell2mat(input_field(i)))
            [Data.(cell2mat(input_field(i)))] = even_unpad(Data.(cell2mat(input_field(i))),RunOptions.EvenSizePadding);
        end
    end
end


SaveData_Chisep(Data, RunOptions)

end


function SaveData_Chisep(Data, RunOptions)

outdir = Data.output_root;
mkdir(outdir);

Options = RunOptions;

tmp = pwd;
eval(['cd(''' outdir ''');']);
if strcmp(RunOptions.InputType, 'nifti')
    save('results.mat','Data' ,'Options','-V7.3')
    [save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.x_dia, 'ChiDia');
    save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    [save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.x_para, 'ChiPara');
    save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
    [save_func, Data.nii_file, Data.save_name]=load_nii_template_and_make_nii(Data, Data.x_tot, 'ChiTot');
    save_func(Data.nii_file,[Data.output_root,'\',Data.save_name]);
elseif strcmp(Options.InputType, 'dicom')
    save('results.mat','Data' ,'Options','-V7.3')
    info.SeriesDescription = 'X-para [ppb]';
    info.StudyDescription = 'X-separation';
    info.SeriesInstance = 1;
    info.WindowCenter = 50;
    info.WindowWidth = 100;
    info.RescaleSlope = 0.1;
    info.RescaleIntercept = 0;
    save_as_DICOM(Data.x_para*10000,Data.Dinfo,info,'ChiPara');

    info.SeriesDescription = 'X-dia [ppb]';
    info.StudyDescription = 'X-separation';
    info.SeriesInstance = 2;
    info.WindowCenter = 50;
    info.WindowWidth = 100;
    info.RescaleSlope = 0.1;
    info.RescaleIntercept = 0;
    save_as_DICOM(Data.x_dia*10000,Data.Dinfo,info,'ChiDia');

    info.SeriesDescription = 'X-total [ppb]';
    info.StudyDescription = 'X-separation';
    info.SeriesInstance = 3;
    info.WindowCenter = 0;
    info.WindowWidth = 200;
    info.RescaleSlope = 0.1;
    info.RescaleIntercept = 0;
    save_as_DICOM(Data.x_tot*10000,Data.Dinfo,info,'ChiTot');
else
    disp('Nifti or DICOM input were not found. Saving result.mat ...')
    msgbox('Nifti or DICOM input were not found. Saving result.mat ...');
    save('results.mat','Data' ,'Options','-V7.3')
end


eval(['cd(''' tmp ''');']);

end


function [save_func, nii_file, save_name]=load_nii_template_and_make_nii(Data, data, save_name)
    voxel_size = Data.VoxelSize;

    if isfield(Data,'nifti_template')
        nii_file = Data.nifti_template;
    else
        nii_file = [];
    end

    if isempty(nii_file)
        save_func = @save_nii;
        save_name = [save_name, '.nii'];
        origin = [1 1 1];
        nii_file = make_nii(rot90(data,-1), voxel_size, origin);
        [q, nii_file.hdr.hist.pixdim(1)] = CalculateQuatFromB0Dir(Data.B0dir);

        nii_file.hdr.hist.quatern_b = q(2);
        nii_file.hdr.hist.quatern_c = q(3);
        nii_file.hdr.hist.quatern_d = q(4);
        nii_file.hdr.hist.originator = origin;
    else
        save_func = @save_untouch_nii;
        nii_file.img = rot90(data,-1);
    end

    nii_file.hdr.dime.datatype = 16;
    nii_file.hdr.dime.dim(5) = size(data,4);
    nii_file.hdr.dime.dim(1) = ndims(data);

    nii_file.hdr.dime.scl_inter = 0;
    nii_file.hdr.dime.scl_slope = 1;

    nii_file.hdr.hist.magic = 'n+1';
end