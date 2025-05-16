% Assume: allFeatures and allLabels are both 10x1 cell arrays
% Each cell contains a table/array of features and corresponding labels
function [rfModel, rfResults, knnModel, knnResults] = trainModel_LOO(allFeatures, allLabels, params)
    
    % initialize model creation, RF and KNN model created
    [rfModel, rfResults] = trainRFModel_LOO(allFeatures, allLabels, params);
    [knnModel, knnResults] = trainKNNModel_LOO(allFeatures, allLabels, params);

end

function [models, results] = trainRFModel_LOO(allFeatures, allLabels, params)
    
    % initialising variables where models and results are stored
    numFolds = numel(allFeatures);
    models = cell(numFolds, 1); 
    results = struct();
    
    for i = 1:numFolds

        % Split data selecting one file as testing and the other 9 files as
        % training data
        testX = allFeatures{i};
        testY = allLabels{i};
        
        XTrain = vertcat(allFeatures{[1:i-1, i+1:end]});
        YTrain = vertcat(allLabels{[1:i-1, i+1:end]});
        
        % Convert labels to categorical
        trainYCat = categorical(YTrain, params.stageValues, params.stageNames);
        testYCat = categorical(testY, params.stageValues, params.stageNames);
        
        % Cost matrix
        classNames = categories(trainYCat);
        costMatrix = ones(numel(classNames)) - eye(numel(classNames));
        n1Idx = find(strcmp(classNames, 'N1'));
        if ~isempty(n1Idx)
            costMatrix(n1Idx, :) = 10;
            costMatrix(:, n1Idx) = 10;
            costMatrix(n1Idx, n1Idx) = 0;
        end
    
        % Train Random Forest model
        template = templateTree('Reproducible', true);
        rfModel = fitcensemble(XTrain, trainYCat, ...
            'Method', 'Bag', ...
            'Learners', template, ...
            'Cost', costMatrix, ...
            'NumLearningCycles', 100);
        
        models{i} = rfModel;
        
        % Predict and store results
        predY = predict(rfModel, testX);
        results(i).trueLabels = testYCat;
        results(i).predLabels = predY;
    end
end

function [models, results] = trainKNNModel_LOO(allFeatures, allLabels, params)
    
    % initialising variables where models and results are stored
    numSubjects = numel(allFeatures);
    models = cell(numSubjects,1);
    results = struct([]);
    
    for i = 1:numSubjects

        % Split data selecting one file as testing and the other 9 files as
        % training data
        testX = allFeatures{i};
        testY = allLabels{i};
        
        trainIdx = setdiff(1:numSubjects, i);
        XTrain = vertcat(allFeatures{trainIdx});
        YTrain = vertcat(allLabels{trainIdx});
        
        % Convert labels to categorical
        YTrainCat = categorical(YTrain, params.stageValues, params.stageNames);
        YTestCat  = categorical(testY,  params.stageValues, params.stageNames);

        % best kValue was found as 13, after running findKValue()
        % kValue = findKValue(XTrain, YTrainCat, testX, YTestCat);
        kValue = 13;
    
       % Train KNN model
        knnModel = fitcknn(XTrain, YTrainCat, ...
                    'NumNeighbors', kValue, ...
                    'Distance',  'seuclidean', ...
                    'DistanceWeight', 'squaredinverse', ...
                    'Standardize', false); 
   
        % Store model
        models{i} = knnModel;
        
        % Predict
        YPred = predict(knnModel, testX);
        
        % Store results
        results(i).trueLabels = YTestCat;
        results(i).predLabels = YPred;
    end
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