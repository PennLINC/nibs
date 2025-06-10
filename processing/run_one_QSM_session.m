function run_one_session(subid, sesid)
% Always treat inputs as strings
subid = char(subid);
sesid = char(sesid);

% Add general Path
addpath(genpath("/cbica/projects/pennlinc_qsm/software/sepia-1.2.2.6/")); 
sepia_addpath;

% Define paths
base_path = '/cbica/projects/pennlinc_qsm/';
input(1).name = fullfile(base_path, 'data', 'bids_directory', ['sub-',subid], ['ses-',sesid],'qsm', ...
    ['sub-', subid, '_ses-', sesid, '_phase.nii.gz']);
input(2).name = fullfile(base_path, 'data', 'bids_directory', ['sub-',subid], ['ses-',sesid],'qsm', ...
    ['sub-', subid, '_ses-', sesid, '_mag.nii.gz']);
input(3).name = '';  % Leave empty if not needed
input(4).name = fullfile(base_path, 'scripts','tools','sepia_header.mat');

% Output base name
output_basename = fullfile(base_path, 'output','SEPIA', ['sub-',subid], ['ses-',sesid], ['sub-', subid, '_ses-', sesid]);

% Mask filename
mask_filename = fullfile(base_path, 'output', 'skullStripAndRegistration', ['sub-',subid], ['ses-',sesid],'anat', ...
    ['sub-', subid, '_ses-', sesid, '_T1BrainMask_in_mag_space.nii.gz']);

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

output_dir = fileparts(output_basename); % Get parent directory path
if ~exist(output_dir, 'dir')
    mkdir(output_dir); % This creates all required parent directories
end

% Run SEPIA process
fprintf('Processing subject %s, session %s...\n', subid, sesid);
sepiaIO(input, output_basename, mask_filename, algorParam);
end
