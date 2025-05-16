% Assume: allFeatures and allLabels are both 10x1 cell arrays
% Each cell contains a table/array of features and corresponding labels
function [rfModel, rfResults, knnModel, knnResults] = trainModel_Repeat(allFeatures, allLabels, params)
    [rfModel, rfResults] = trainRFModel(allFeatures, allLabels, params);
    [knnModel, knnResults] = trainKNNModel(allFeatures, allLabels, params);
end

function [rfModels, rfResults] = trainRFModel(allFeatures, allLabels, params)
    % Combine results to matrix
    allFeatures = vertcat(allFeatures{:});
    allLabels = vertcat(allLabels{:});   

    % Convert labels
    allLabelsCat = categorical(allLabels, params.stageValues, params.stageNames);
    numRepeats = 10; % Number of repetitions
    rng(42); % For reproducibility
    
    rfModels = cell(numRepeats,1); % To store models
    rfResults = struct([]);
    
    for i = 1:numRepeats
        % Stratified 80/20 train-test split
        cv = cvpartition(allLabelsCat, 'HoldOut', 0.2, 'Stratify', true);
        
        % Store current split
        XTrain = allFeatures(training(cv), :);
        YTrain = allLabelsCat(training(cv));
        XTest = allFeatures(test(cv), :);
        YTest = allLabelsCat(test(cv));
    
        % Cost matrix setup
        classNames = categories(YTrain);
        costMatrix = ones(numel(classNames)) - eye(numel(classNames));
        n1Idx = find(strcmp(classNames, 'N1'));
        if ~isempty(n1Idx)
            costMatrix(n1Idx, :) = 10;
            costMatrix(:, n1Idx) = 10;
            costMatrix(n1Idx, n1Idx) = 0;
        end
    
        % Train model
        template = templateTree('Reproducible', true);
        rfModel = fitcensemble(XTrain, YTrain, ...
            'Method', 'Bag', ...
            'Learners', template, ...
            'Cost', costMatrix, ...
            'NumLearningCycles', 100);
    
        % Save model and results
        rfModels{i} = rfModel;
        % Predict and store results
        predY = predict(rfModel, XTest);
        rfResults(i).trueLabels = YTest;
        rfResults(i).predLabels = predY;
    end
    
    % Optionally save all models/results
    %save('rfModels_allSplits.mat', 'rfModels', 'allResults');
end

function [knnModels, knnResults] = trainKNNModel(allFeatures, allLabels, params)
    % Combine results to matrix
    allFeatures = vertcat(allFeatures{:});
    allLabels = vertcat(allLabels{:});   

    % Convert labels
    allLabelsCat = categorical(allLabels, params.stageValues, params.stageNames);
    numRepeats = 10; % Number of repetitions
    rng(42); % For reproducibility
    
    knnModels = cell(numRepeats,1); % To store models
    knnResults = struct([]);
    
    for i = 1:numRepeats
        % Stratified 80/20 train-test split
        cv = cvpartition(allLabelsCat, 'HoldOut', 0.2, 'Stratify', true);
        
        % Store current split
        XTrain = allFeatures(training(cv), :);
        YTrain = allLabelsCat(training(cv));
        XTest = allFeatures(test(cv), :);
        YTest = allLabelsCat(test(cv));
    
        % find best kValue
        % kValue = findKValue(XTrain, YTrain, XTest, YTest);
        kValue = 13;
    

       % Train KNN model
       % knnModel = fitcknn(XTrain, YTrain, 'NumNeighbors', kValue);
        knnModel = fitcknn(XTrain, YTrain, ...
                    'NumNeighbors', kValue, ...
                    'Distance',  'seuclidean', ...
                    'DistanceWeight', 'squaredinverse', ...
                    'Standardize', false); 


        % Save model and results
        knnModels{i} = knnModel;
        % Predict and store results
        predY = predict(knnModel, XTest);
        knnResults(i).trueLabels = YTest;
        knnResults(i).predLabels = predY;
    end
    
    % Optionally save all models/results
    %save('rfModels_allSplits.mat', 'rfModels', 'allResults');
end

 %% find optimal k value
function bestK = findKValue(X_train,Y_train, X_test, Y_test)
    % Find Best k for k-NN
    kValues = 1:2:15;
    bestK = kValues(1);
    bestAccuracy = 0;
    
    for k = kValues
        knnModel = fitcknn(X_train, Y_train, 'NumNeighbors', k);
        predictedLabels = predict(knnModel, X_test);
        accuracy = sum(predictedLabels == Y_test) / length(Y_test) * 100;
        fprintf('Test Accuracy (k=%d): %.2f%%\n', k, accuracy);
        
        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            bestK = k;
        end
    end
end
    
