%% Prepare paths for the archived traditional MATLAB implementation.
legacyDir = fileparts(mfilename('fullpath'));
codesDir = fileparts(legacyDir);
addpath(genpath(legacyDir));

% The original scripts use paths such as ../cal and ../Calibration_Polar.
% Keeping the working directory at codes/ preserves that original behavior.
cd(codesDir);
disp(['Traditional MATLAB code loaded from: ', legacyDir]);
disp(['MATLAB working directory set to: ', codesDir]);
