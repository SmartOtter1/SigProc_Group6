function evaluateModel(model, XTest, YTest, stageNames)
    predLabels = predict(model, XTest);
    
    % Metrics
    confMat = confusionmat(YTest, predLabels);
    precision = diag(confMat) ./ sum(confMat, 2);
    recall = diag(confMat) ./ sum(confMat, 1)';
    f1Scores = 2 * (precision .* recall) ./ (precision + recall);
    
    % Display
    fprintf('Test Accuracy: %.2f%%\n', sum(diag(confMat))/sum(confMat(:))*100);
    T = table(precision, recall, f1Scores, 'RowNames', stageNames);
    disp(T);
    
    % Confusion matrix
    figure('Position', [100 100 800 600]);
    confusionchart(YTest, predLabels, ...
        'Title', 'Confusion Matrix (Random Forest)', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
end