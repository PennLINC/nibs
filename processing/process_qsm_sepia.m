% Add general Path
% SEPIA install location. Override with the SEPIA_HOME environment variable;
% otherwise fall back to the local qsm-software checkout.
sepia_home = getenv('SEPIA_HOME');
if isempty(sepia_home)
    sepia_home = '/mnt/c/Users/tsalo/Documents/linc/qsm-software/sepia-1.2.2.6';
end
addpath(genpath(sepia_home));
sepia_addpath;

% Add every toolbox in the qsm-software folder to the path. SEPIA's selected
% algorithms pull in dependencies it does not bundle (e.g. MEDI for the BET
% brain extraction, MRI_susceptibility_calculation_12072021 for the dirTik
% solver), so add the whole tree rather than enumerating them. Resolve the
% folder from NIBS_SOFTWARE_ROOT (override via the env var), matching the
% chi-separation script; otherwise fall back to the local qsm-software checkout.
software_root = getenv('NIBS_SOFTWARE_ROOT');
if isempty(software_root)
    software_root = '/mnt/c/Users/tsalo/Documents/linc/qsm-software';
end
addpath(genpath(software_root));

% Define paths
output_dir = '{{ output_dir }}';
% Concatenated phase image
input(1).name = '{{ phase_file }}';

% Concatenated magnitude image
input(2).name = '{{ mag_file }}';
input(3).name = [];  % Leave empty if not needed
input(4).name = '{{ header_file }}';

mask_file = '{{ mask_file }}';

% Algorithm parameters (same as before)
algorParam = struct();
% Use the supplied (sMRIPrep-derived) brain mask passed to sepiaIO below rather
% than letting SEPIA run its own BET, so the SEPIA Chimap is bounded by the same
% brain mask as the R2*/R2' and chi-separation outputs. isBET is therefore 0.
% isRefineBrainMask is also 0: the refinement re-derives a tighter mask from
% phase quality, which over-eroded the supplied mask; keep the supplied mask
% as-is.
algorParam.general = struct( ...
    'isBET', 0, ...
    'fractional_threshold', 0.15, ...
    'gradient_threshold', 0, ...
    'isInvert', 0, ...
    'isRefineBrainMask', 0);
% Define the 'unwrap' sub-structure
algorParam.unwrap = struct( ...
    'echoCombMethod', 'Optimum weights', ...
    'unwrapMethod', 'ROMEO', ...
    'isEddyCorrect', 0, ...
    'isSaveUnwrappedEcho', 1);
% Define the 'bfr' sub-structure
algorParam.bfr = struct( ...
    'refine_method', 'None', ...
    'refine_order', 4, ...
    'erode_radius', 0, ...
    'erode_before_radius', 0, ...
    'method', 'VSHARP', ...
    'radius', [10, 9, 8, 7, 6, 5, 4, 3]);
% Define the 'qsm' sub-structure
algorParam.qsm = struct( ...
    'reference_tissue', 'CSF', ...
    'method', 'MRI Suscep. Calc.', ...
    'solver', 'Direct Tikhonov', ...
    'lambda', 0.05);

if ~exist(output_dir, 'dir')
    mkdir(output_dir); % This creates all required parent directories
end

% Run SEPIA process. The third argument is the brain mask: pass the supplied
% sMRIPrep-derived mask so SEPIA does not recompute one with BET.
sepiaIO(input, output_dir, mask_file, algorParam);
