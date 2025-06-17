% Add general Path
addpath(genpath("/cbica/projects/nibs/software/sepia-1.2.2.6/"));
sepia_addpath;

% Define paths
base_path = '/cbica/projects/nibs/';
output_dir = '{{ output_dir }}';
% Concatenated phase image
input(1).name = '{{ phase_file }}';

% Concatenated, skull-stripped magnitude image
input(2).name = '{{ mag_file }}';
input(3).name = '';  % Leave empty if not needed
input(4).name = fullfile(base_path, 'code', 'processing', 'sepia_header.mat');

% Mask filename
mask_filename = '{{ mask_file }}';

% Algorithm parameters (same as before)
algorParam = struct();
algorParam.general.isBET = 0;
algorParam.general.isInvert = 0;
algorParam.general.isRefineBrainMask = 0;

% Total field recovery algorithm parameters
algorParam.unwrap.echoCombMethod = 'ROMEO total field calculation';
algorParam.unwrap.offsetCorrect = 'On';
algorParam.unwrap.mask = 'SEPIA mask';
algorParam.unwrap.qualitymaskThreshold = 0.5;
algorParam.unwrap.useRomeoMask = 0;
algorParam.unwrap.isEddyCorrect = 0;
algorParam.unwrap.isSaveUnwrappedEcho = 1;
algorParam.unwrap.excludeMaskThreshold = 0.5;
% algorParam.unwrap.excludeMethod = 'Weighting map';

% Background field removal algorithm parameters
algorParam.bfr.refine_method = '3D Polynomial';
algorParam.bfr.refine_order = 4;
algorParam.bfr.erode_radius = 0;
algorParam.bfr.erode_before_radius = 0;
algorParam.bfr.method = 'PDF';
algorParam.bfr.tol = 0.1;
algorParam.bfr.iteration = 50;
algorParam.bfr.padSize = 40;

% QSM algorithm parameters
algorParam.qsm.reference_tissue = 'CSF';
algorParam.qsm.method = 'MEDI';
algorParam.qsm.lambda = 1000;
algorParam.qsm.wData = 1;
algorParam.qsm.percentage = 90;
algorParam.qsm.zeropad = [0 0 0];
algorParam.qsm.isSMV = 1;
algorParam.qsm.radius = 3;
algorParam.qsm.merit = 1;
algorParam.qsm.isLambdaCSF = 1;
algorParam.qsm.lambdaCSF = 100;

if ~exist(output_dir, 'dir')
    mkdir(output_dir); % This creates all required parent directories
end

% Run SEPIA process
sepiaIO(input, output_dir, mask_filename, algorParam);
