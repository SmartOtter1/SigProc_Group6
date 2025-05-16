clc; close all; clear; 
addpath(genpath('functions'));
%% Phase 2
% This script only contains Feature Engineering Classification and
% evaluation of different classifiers.
% Preprocessing and feature extraction steps can be found in the runML.m
% script.

%% Variable explenation
% params: Structure to save all necessary settings for preprocessing and
% evaluating the signals. 
%
% numFiles: number of files which should be loaded
%
% allFeatures: Cell structure including all the saved features from the
% different files. Each cell represents one data file, each column in a
% cell represents one feature, and each row represents an epoch. 82
% features were extracted and saved -> each cell has 82 columns
% number of rows varies depending on the length of the signals from the
% file.
%
% allLabels: Cell structure including all saved labels representing the
% stage of each epoch. Each cell represnets one data file, including one
% column with the labels and m rows representing the number of epochs of
% the signal.
%
% fsIndex: Logical variable including the highest rated features for
% training the classifier according to a wrapped feature selection. 
%
% *Model: Cell structure storing the RF or KNN model settings for the
% different itterations. * = rf, knn
% 
% *Results: Structure including true labels and predicted labels for each
% iteration. Used to evaluate model performance. * = rf, knn

%% Parameters
params.epochDuration = 30; % Seconds
params.signalThreshold = 1.0e+04; % Variance threshold to cut noise at end of actual signal
params.stageValues = [5, 4, 3, 2, 0]; % different stages given in stages variable
params.stageNames = {'Wake', 'N1', 'N2', 'N3', 'REM'}; % stage names correlated to stage values
params.bands = struct('delta', [0.5,4], 'theta', [4,8], ...
                    'alpha', [8,13], 'beta', [13,30], 'gamma', [30,40], 'SEM', [0.5 1; 0.25 0.5; 0.125 0.25; 0.0625 0.125]); % EEG wave frequencies and SEM frequencies to extract features later on
params.windows = struct('EMG_RMS', 1, 'EMG_overlap', 0.5, 'Cutting_Signal', 5); % EMG_RMS, and EMG_overlap applied in RMS feature calclation of EMG signal, Cutting_Signal defines number of epochs necessary where signalThreshold is fulfilled to cut noise at end of signal

% Define signals to process (case-sensitive to match EDF headers)
params.signalTypes = {
    struct('name','EEG','filter',[0.1 35],'features',true),...
    struct('name','EEGsec','filter',[0.1 35],'features',true),...
    struct('name','EOGR','filter',[0.5 15],'features',true),...
    struct('name','EOGL','filter',[0.5 15],'features',true),...
    struct('name','EMG','filter', [0.5 10],'features',true)
}; % filter defines the cut off frequencies for the butterworth filter, features defines if there will be features extracted from this signal

%% Initialize storage
numFiles = 10; % number of Files 

%% loading data
% Signal processing and feature extraction steps were already completed. By
% saving the features and stages of the files the classifier can now be
% trained without having to load and preprocess the signals each time. In
% the following script the different classifiers are trained and evaluated
% using the previously extracted features and stages from the data sets.
% The preprocessing and feature extraction of the signals can be found in
% the main.m file.
allFeatures = cell(numFiles,1);
allLabels = cell(numFiles,1);
for i = 1:numFiles
    filename = sprintf('featureTable_%d.mat', i);  % Assuming 'i' is a number and the file has a .mat extension
    features = load(filename);
    allFeatures{i} = table2array(features.variable);
end
for i = 1:numFiles
    filename = sprintf('epochLabels_%d.mat', i);  % Assuming 'i' is a number and the file has a .mat extension
    stages = load(filename);
    allLabels{i} = stages.variable;
end

filename = 'wrappedfeaturesIndex.mat';
fsIndex = load(filename);
fsIndex = fsIndex.variable;

%% Model training and evaluation
% trainModel_Ru builds the model once using 80% training and 20%
% testing data
% Only RF model is trained and evaluated.
if ~isempty(allFeatures)
    [rfModel_Ru, rfResults_Ru] = trainModel_Ru(allFeatures, allLabels, params);
    evaluateModel_Ru(rfModel, results.XTest, results.YTest, params.stageNames);
else
    error('No features extracted from any file');
end
%%
% trainModel_Repeat builds the model 10 times using 80% training and
% 20% testing data. The data used for training and testing is chagned each
% time, but randomly chosen.
% RF and KNN model is trained and evaluated.
if ~isempty(allFeatures)
    [rfModel, rfResults, knnModel, knnResults] = trainModel_Repeat(allFeatures, allLabels, params);
    evaluateModel(rfResults, knnResults);
else
    error('No features extracted from any file');
end

%%
% trainModel_LOO builds the model 10 times using 9 files as training data
% and one as testing data, where it iterates through the different files to
% always use a different file as testing data.
% RF and KNN model is trained and evaluated.

if ~isempty(allFeatures)
    [rfModelLOO, rfResultsLOO, knnModelLOO, knnResultsLOO] = trainModel_LOO(allFeatures, allLabels, params, fsIndex);
    evaluateModel(rfResultsLOO, knnResultsLOO);
else
    error('No features extracted from any file');
end

%%
% trainModel_LOO_wrapped builds the model 10 times using 9 files as training data
% and one as testing data, where it iterates through the different files to
% always use a different file as testing data.
% Only using wrapped selected features.
% RF and KNN model is trained and evaluated.

if ~isempty(allFeatures)
    [rfModelLOOwrapped, rfResultsLOOwrapped, knnModelLOOwrapped, knnResultsLOOwrapped] = trainModel_LOO_wrapped(allFeatures, allLabels, params, fsIndex);
    evaluateModel(rfResultsLOOwrapped, knnResultsLOOwrapped);
else
    error('No features extracted from any file');
end

%% KNN Models evaluation
% Trains KNN models with different settings and compares model performance.
if ~isempty(allFeatures)
    [knnModels, knnResults] = trainKNNModels(allFeatures, allLabels, params);
    evaluateKNNModels(knnResults)
else
    error('No features extracted from any file');
end

%% wrapped feature selection
% % The script below extracts the best performing features applying the
% % wrapped feature selection. The fsIndex can be saved from the workspace by
% % running 'save vector of wrapped features' part. fsIndex can later be
% % restored for further tests.

% X = vertcat(allFeatures{:});  % Combine features
% Y = vertcat(allLabels{:});    % Combine labels
% 
% fun = @(Xtrain, Ytrain, Xtest, Ytest) ...
%     loss(fitcensemble(Xtrain, Ytrain, 'Method', 'Bag'), Xtest, Ytest);
% 
% opts = statset('display','iter');  % Show progress
% [fsIndex, history] = sequentialfs(fun, X, Y, ...
%     'cv', 5, ...                   % 5-fold cross-validation
%     'options', opts, ...
%     'direction', 'forward');      % Forward selection
% 
% X_selected = X(:, fsIndex);
%% save vector of wrapped features
% filename = ['wrappedfeaturesIndex.mat'];
% savingVariables(filename, fsIndex);

