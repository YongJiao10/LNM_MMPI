function single_subj_script(subj)
% Single-subject lesion quantification script using LQT.

if nargin < 1
    error('Subject ID is required.');
end

%% ======================= Clean MATLAB path ============================
restoredefaultpath;            % Reset to MATLAB factory path
rehash toolboxcache;           % Clear cached paths
warning('off','MATLAB:dispatcher:nameConflict');

clc

%% ---------------------------- Basic paths ----------------------------
% Working directory
working_dir = '/Users/yojiao/SFC_project/LNM_voxel/';

% LQT root path
lqt_path = '/Shared/jianglab/3_Data_Working/LNM/tools/Lesion_Quantification_Toolkit/';

% Add required LQT support tools
addpath(genpath([lqt_path 'Support_Tools']));
% Add LQT core functions
addpath(genpath([lqt_path 'Functions']));

%% ---------------------------- Config structure ----------------------------
cfg = struct();

% Path to DSI_Studio program
cfg.dsi_path = '/Shared/pinc/sharedopt/apps/DSI_Studio/Linux/x86_64/20200122/dsi_studio';

% Path to HCP1065 .fib template and tractography atlas
cfg.source_path = [lqt_path 'Support_Tools/Tractography_Atlas/HCP_1065'];

% Path to output directory (patient and atlas result directories will be created within this directory)
cfg.out_path = [working_dir 'Iowa_sLNM/'];

% Ensure output directory exists
if ~exist(cfg.out_path, 'dir')
    mkdir(cfg.out_path);
end

% Path to lesion (pre-registered to MNI template)
cfg.lesion_path = ['/Shared/boeslab/Data/Lesion/Iowa/Registry/lesionMasks/Chronic/MNI152/2mm/' subj '.nii.gz'];

% Path to parcellation (should have identical dimensions to lesion and be in MNI template space)
cfg.parcel_path = ['/Users/yojiao/SFC_project/' ...
                   'Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_2mm.nii.gz'];

% Patient ID (used as prefix for output files)
cfg.pat_id = subj;

% File suffix -- used as suffix for output files. Atlas name is recommended (e.g. AAL, Power, Gordon, etc.).
cfg.file_suffix = 'Schaefer100_Tian16';

%% ---------------------------- Connectivity options ----------------------------
% Connectivity type ('end' or 'pass'): if 'end', connections are defined based on
% streamline endpoints. If 'pass', connections are defined based on streamline
% pass-throughs. 'end' is recommended (most conservative).
cfg.con_type = 'end';

% Percent spared threshold for computing SSPLs (e.g. 100 means that only fully
% spared regions will be included in SSPL calculation; 1 means that all regions
% with at least 1% spared will be included. Default is 50)
cfg.sspl_spared_thresh = 50;

%% ---------------------------- Load parcellation info ----------------------------
% (Visualization only; disabled for now)

% load([lqt_path ...
%     'Support_Tools/Parcellations/Schaefer_Yeo/Plus_Subcort/' ...
%     'Schaefer2018_200Parcels_7Networks_order_plus_subcort.mat']);

% cfg.node_label = t.RegionName; % n_regions-by-1 cell array of strings (parcel names)
% cfg.node_color = t.NetworkID;  % n_regions-by-1 array of ints (network assignments)

%% ---------------------------- Summary figure options ----------------------------
% Show summary figures for damage and disconnection measures (single-subject runs only).
cfg.show_summary_figs  = 0;

% Subsequent options are only relevant if cfg.show_summary_figs == 1.
% Parcel coordinates used for plotting ball-and-stick brain graph. If not supplied,
% they will be estimated from the parcel file.
% cfg.parcel_coords      = [t.X, t.Y, t.Z];
cfg.parcel_coords      = [];

% Percentile threshold for displaying SSPL increases. Will only display SSPL
% increases above percentile threshold (Default = 90).
% cfg.delta_sspl_thresh  = 99;

% Parcel damage threshold for displaying SSPL increases. Will not display SSPL
% increases for parcels with % damage greater than or equal to the threshold
% (default = 100)
% cfg.parcel_dmg_thresh  = 100;

% Tract disconnection threshold for displaying tracts in summary figure. Will
% not display data for tracts with % disconnection below threshold (Default = 5)
% cfg.tract_sdc_thresh   = 5;

% Smoothing kernel size
cfg.smooth             = 2;

%% ---------------------------- Run LQT computations ----------------------------
% Get parcel damage for patient
util_get_parcel_damage(cfg);

% Get tract SDC for patient
util_get_tract_discon(cfg);

% Get parcel SDC and SSPL measures for patient
util_get_parcel_cons(cfg);

% Create summary figures (optional)
% if cfg.show_summary_figs == 1
%     util_plot_summaries(cfg);
% end

%% ---------------------------- Done ----------------------------
disp(['+++ ' subj ' lesion mapping Finished!']);
disp(['+++ Output measures are stored at ' cfg.out_path]);
disp(['+++ Thank you for patient waitting, you can check the results now']);

end