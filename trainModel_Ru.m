function [rfModel, results] = trainModel(allFeatures, allLabels, params)
    
    % Combine results to matrix
    allFeatures = vertcat(allFeatures{:});
    allLabels = vertcat(allLabels{:});
    
    % select 
    % Convert labels
    allLabelsCat = categorical(allLabels, params.stageValues, params.stageNames);
    
    % Train-test split
    rng(42);
    cv = cvpartition(allLabelsCat, 'HoldOut', 0.2, 'Stratify', true);
    results.XTrain = allFeatures(training(cv), :);
    results.YTrain = allLabelsCat(training(cv));
    results.XTest = allFeatures(test(cv), :);
    results.YTest = allLabelsCat(test(cv));
    
    % Cost matrix
    classNames = categories(results.YTrain);
    costMatrix = ones(5,5) - eye(5);
    n3Idx = find(strcmp(classNames, 'N3'));
    costMatrix(n3Idx, :) = 10;
    costMatrix(:, n3Idx) = 10;
    costMatrix(n3Idx, n3Idx) = 0;
    
    % Train model
    template = templateTree('Reproducible', true);
    rfModel = fitcensemble(results.XTrain, results.YTrain, ...
        'Method', 'Bag', ...
        'Learners', template, ...
        'Cost', costMatrix, ...
        'NumLearningCycles', 100);
    save('trained_rf_model.mat', 'rfModel');
end