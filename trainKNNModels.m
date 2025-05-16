function [knnModels, knnResults] = trainKNNModels(allFeatures, allLabels, params)
    % Combine results to matrix
    allFeatures = vertcat(allFeatures{:});
    allLabels = vertcat(allLabels{:});   

    % Convert labels
    allLabelsCat = categorical(allLabels, params.stageValues, params.stageNames);
    numRepeats = 10; % Number of repetitions
    rng(42); % For reproducibility
    
    % Define parameter combinations to test
    distanceMetrics = {'euclidean', 'seuclidean', 'cityblock', 'chebychev', ...
                       'minkowski', 'cosine', 'correlation', ...
                       'spearman', 'hamming', 'jaccard'};
    %distanceMetrics = {'euclidean'};
    distanceWeights = {'equal', 'inverse', 'squaredinverse'};
    
    % Initialize structures to store all models and results
    knnModels = struct();
    knnResults = struct();
    
    % Create all combinations of parameters (simpler approach)
    [distGrid, weightGrid] = meshgrid(distanceMetrics, distanceWeights);
    paramCombinations = [distGrid(:), weightGrid(:)];
    
    for combIdx = 1:size(paramCombinations, 1)
        currentDist = paramCombinations{combIdx, 1};
        currentWeight = paramCombinations{combIdx, 2};
        
        % Create field name for this combination
        fieldName = sprintf('%s_%s', currentDist, currentWeight);
        
        % Initialize storage for this parameter combination
        knnModels.(fieldName) = cell(numRepeats, 1);
        knnResults.(fieldName) = struct('trueLabels', {}, 'predLabels', {});
        
        for i = 1:numRepeats
            % Stratified 80/20 train-test split
            cv = cvpartition(allLabelsCat, 'HoldOut', 0.2, 'Stratify', true);
            
            % Store current split
            XTrain = allFeatures(training(cv), :);
            YTrain = allLabelsCat(training(cv));
            XTest = allFeatures(test(cv), :);
            YTest = allLabelsCat(test(cv));
            
            % Find best k for this parameter combination
            kValue = findKValue(XTrain, YTrain, XTest, YTest, currentDist, currentWeight);
            
            % Train KNN model with current parameters
            knnModel = fitcknn(XTrain, YTrain, ...
                'NumNeighbors', kValue, ...
                'Distance', currentDist, ...
                'DistanceWeight', currentWeight, ...
                'Standardize', strcmp(currentDist, 'seuclidean')); % Standardize only for seuclidean
            
            % Save model and results
            knnModels.(fieldName){i} = knnModel;
            
            % Predict and store results
            predY = predict(knnModel, XTest);
            knnResults.(fieldName)(i).trueLabels = YTest;
            knnResults.(fieldName)(i).predLabels = predY;
        end
    end
end
%% find optimal k value for given distance and weight parameters
function bestK = findKValue(X_train, Y_train, X_test, Y_test, distance, weight)
    % Find Best k for k-NN
    kValues = 1:2:30;
    bestK = kValues(1);
    bestAccuracy = 0;
    
    for k = kValues
        knnModel = fitcknn(X_train, Y_train, ...
            'NumNeighbors', k, ...
            'Distance', distance, ...
            'DistanceWeight', weight, ...
            'Standardize', strcmp(distance, 'seuclidean'));
            
        predictedLabels = predict(knnModel, X_test);
        accuracy = sum(predictedLabels == Y_test) / length(Y_test) * 100;
        fprintf('Test Accuracy (k=%d, dist=%s, weight=%s): %.2f%%\n', ...
                k, distance, weight, accuracy);
        
        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            bestK = k;
        end
    end
end

% Helper function for all combinations
function C = allcomb(varargin)
    args = varargin;
    n = numel(args);
    [F{1:n}] = ndgrid(args{:});
    for i = n:-1:1
        G(:,i) = F{i}(:);
    end
    C = num2cell(G, 2);
end