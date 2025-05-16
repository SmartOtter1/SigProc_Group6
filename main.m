clc; close all; clear; 
addpath(genpath('functions'));
%% Phase 1
% This script only contains preprocessing and feature extraction steps.
% Feature Engineering Classification and evaluation of different
% classifiers can be found in the runML.m script.
%% Variable explenation
% params: Structure to save all necessary settings for preprocessing and
% evaluating the signals. 

% numFiles: number of files which should be loaded

% allFeatures: Cell structure including all the saved features from the
% different files. Each cell represents one data file, each column in a
% cell represents one feature, and each row represents an epoch. 82
% features were extracted and saved -> each cell has 82 columns
% number of rows varies depending on the length of the signals from the
% file. Stored from workspace.

% allLabels: Cell structure including all saved labels representing the
% stage of each epoch. Each cell represnets one data file, including one
% column with the labels and m rows representing the number of epochs of
% the signal. Stored from workspace.

% hdr: Structure in which all details about the different signals are
% stored.
% record: Variable including the stored data for the different signals.
% stages: Variable defining the stage every second for the data set.
 
% signals: Structure which contains all signals from data. Is used to
% filter different signals. 

% epochs: Structure including all signals, but this time seperated into
% epochs of same length. Depending on the sampling frequency of the signal
% the number of epochs change. 

% epochLabels: Variable of length of number of epochs including one stage
% per epoch. 

% *Cut: Structure/Variable storing the cut variables to reduce noise of the
% signals at the end. Was cut depending on the variance of the EEG signal.
% Stored from workspace.
% * = epochs, epochLabels, signals, stages

% features: mxn matrix (m=number of epochs, n=number of features) storing
% all extracted features for data file. Includes EEG, EEGsec, EOGL, EOGR,
% EMG signal.

% featureNames: Array storing all the feature labels for feature matrix. 

% featureTable: Table storing feature matrix including featureNames at top
% row. Is stored in workspace.

%% Parameters
params.epochDuration = 30; % Seconds
params.signalThreshold = 1.0e+04; % Variance threshold to cut noise at end of actual signal
params.stageValues = [5, 4, 3, 2, 0]; % different stages given in stages variable
params.stageNames = {'Wake', 'N1', 'N2', 'N3', 'REM'}; % stage names correlated to stage values
params.bands = struct('delta', [0.5,4], 'theta', [4,8], ...
                    'alpha', [8,13], 'beta', [13,30], 'gamma', [30,40], 'SEM', [0.5 1; 0.25 0.5; 0.125 0.25; 0.0625 0.125]);
params.windows = struct('EMG_RMS', 1, 'EMG_overlap', 0.5, 'Cutting_Signal', 5); % EMG_RMS, and EMG_overlap applied in RMS feature calclation of EMG signal, Cutting_Signal defines number of epochs necessary where signalThreshold is fulfilled to cut noise at end of signal

% Define signals to process (case-sensitive to match EDF headers)
params.signalTypes = {
    struct('name','EEG','filter',[0.1 35],'features',true),...
    struct('name','EEGsec','filter',[0.1 35],'features',true),...
    struct('name','EOGR','filter',[0.5 15],'features',true),...
    struct('name','EOGL','filter',[0.5 15],'features',true),...
    struct('name','EMG','filter', [0.5 10],'features',true)
};% filter defines the cut off frequencies for the butterworth filter, features defines if there will be features extracted from this signal

%% Initialize storage
numFiles = 10; % number of Files
allFeatures = cell(numFiles,1); % storage for feature extraction
allLabels = cell(numFiles,1); % storage for stages of epochs

%% Process all files
% This for loop iterates through the number of files and loads and
% preprocesses the various signals within the data file. Further it
% extracts the various features from the signals and aligns stages to each
% epoch. 
% Relevant variables are saved from the workspace and can be called in
% following phases to reduce computational costs. 

for fileNum = 1:numFiles
    fprintf('\nProcessing file R%d...\n', fileNum);
    
    % Load data with validation, might have to change path in loadData_Ru
    try
        [hdr, record, stages] = loadData_Ru(fileNum);
    catch ME
        warning('Failed to load file R%d: %s', fileNum, ME.message);
        continue;
    end
    
    
    % Preprocess available signals, filtering using butterworth
    signals = struct();
    for s = 1:length(params.signalTypes)
        sigType = params.signalTypes{s};
        if any(strcmp(hdr.label, sigType.name)) % Exact match
            signals.(sigType.name) = preprocessSignal_Ru(...
                record, hdr, sigType.name, sigType.filter, length(stages));
        end
    end
    
    % Epoch processing and stage aligning to each epoch
    [epochs, epochLabels] = epochProcessing_Ru(signals, stages, hdr, params);
    
    % Cut epochs, signals and stages depending on the variance of the
    % EEG signal. All signals are cut to the same length.
    [epochsCut, epochLabelsCut, signalsCut, stagesCut] = cuttingVariables(epochs, epochLabels, signals, stages, params);

    % Feature extraction with validation
    if ~isempty(fieldnames(epochsCut))
        [features, featureNames] = featureExtraction_Ru(epochsCut, hdr, params);

        % Store and display
        allFeatures{fileNum} = features;
        allLabels{fileNum} = epochLabelsCut;
        
        % Display features for this file
        featureTable = array2table(features, 'VariableNames', featureNames);

        disp(['First 5 epochs for R' num2str(fileNum) ':']);
        disp(featureTable(1:min(5,size(features,1)),:));
    else
        warning('No epochs processed for file R%d', fileNum);
    end
    
    %% Plotting signals     CAVE: This can be very computational demanding!
% % plotting signals before and after cutting, 
% % It is advisible to add a brake point here.
%   creatingVisualisation(signals, signalsCut, stages, stagesCut, hdr)

    %% Saving variables from workspace
%   overwrites existing files
    filename = ['signals_filtered_' num2str(fileNum) '.mat'];
    savingVariables(filename, signalsCut);
    filename = ['epochs_' num2str(fileNum) '.mat'];
    savingVariables(filename, epochsCut);
    filename = ['stages_' num2str(fileNum) '.mat'];
    savingVariables(filename, stagesCut);
    filename = ['epochLabels_' num2str(fileNum) '.mat'];
    savingVariables(filename, epochLabelsCut);

    filename = ['featureTable_' num2str(fileNum) '.mat'];
    savingVariables(filename, featureTable);

end

%% save allFeatures and allLabels as cell structure

filename = ['allFeatures.mat'];
savingVariables(filename, allFeatures);
filename = ['allLabels.mat'];
savingVariables(filename, allLabels);

%% Plot Signals
% only plots the last file signals, since those are saved in workspace
% within these variables

figure
subplot(6,1,1);
plot((1:length(signals.EEG))/125,signals.EEG);
hold on 
subplot(6,1,2);
plot((1:length(signalsCut.EEG))/125,signalsCut.EEG);
hold on 
subplot(6,1,3);
plot((1:length(signals.EOGL))/50,signals.EOGL);
hold on
subplot(6,1,4);
plot((1:length(signalsCut.EOGL))/50,signalsCut.EOGL);
hold on
subplot(6,1,5);
plot(1:length(stages), stages);
hold on 
subplot(6,1,6);
plot(1:length(stagesCut), stagesCut);


