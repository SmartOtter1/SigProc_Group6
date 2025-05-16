function evaluateModel(rfResults, knnResults)
    % Assuming:
    % rfResults and knnResults are 1x10 structs with .predLabels and .trueLabels
    % rfModels and knnModels are 1x10 cell arrays (can be unused here if you just evaluate predictions)
    
    % Initialize metrics
    numFolds = 10;

    rfMetrics = struct('Accuracy', [], 'Precision', [], 'Recall', [], 'F1', [], 'Kappa', []);
    knnMetrics = rfMetrics;
    
    % Helper function to calculate metrics
    calcMetrics = @(trueLabels, predLabels) struct( ...
        'Accuracy', sum(predLabels == trueLabels) / numel(trueLabels), ...
        'ConfMat', confusionmat(trueLabels, predLabels), ...
        'ClassReport', classMetrics(trueLabels, predLabels), ...
        'Kappa', cohensKappa(trueLabels, predLabels) ...
    );
    
    % Loop over each fold
    for i = 1:numFolds
        % RF
        rf = rfResults(i);
        rfEval = calcMetrics(rf.trueLabels, rf.predLabels);
        rfMetrics.Accuracy(end+1) = rfEval.Accuracy;
        rfMetrics.Precision(:,i) = rfEval.ClassReport.Precision;
        rfMetrics.Recall(:,i) = rfEval.ClassReport.Recall;
        rfMetrics.F1(:,i) = rfEval.ClassReport.F1Score;
        rfMetrics.Kappa(end+1) = rfEval.Kappa;
    
        % KNN
        knn = knnResults(i);
        knnEval = calcMetrics(knn.trueLabels, knn.predLabels);
        knnMetrics.Accuracy(end+1) = knnEval.Accuracy;
        knnMetrics.Precision(:,i) = knnEval.ClassReport.Precision;
        knnMetrics.Recall(:,i) = knnEval.ClassReport.Recall;
        knnMetrics.F1(:,i) = knnEval.ClassReport.F1Score;
        knnMetrics.Kappa(end+1) = knnEval.Kappa;


    end
    
    % Print summary statistics
    fprintf('\n=== Random Forest ===\n');
    fprintf('Mean Accuracy: %.2f%%\n', mean(rfMetrics.Accuracy)*100);
    fprintf('Mean Kappa: %.2f\n', mean(rfMetrics.Kappa));
    disp('Per-Class Precision, Recall, F1:');
    disp(mean(rfMetrics.Precision, 2));
    disp(mean(rfMetrics.Recall, 2));
    disp(mean(rfMetrics.F1, 2));
    
    fprintf('\n=== KNN ===\n');
    fprintf('Mean Accuracy: %.2f%%\n', mean(knnMetrics.Accuracy)*100);
    fprintf('Mean Kappa: %.2f\n', mean(knnMetrics.Kappa));
    disp('Per-Class Precision, Recall, F1:');
    disp(mean(knnMetrics.Precision, 2));
    disp(mean(knnMetrics.Recall, 2));
    disp(mean(knnMetrics.F1, 2));
    

    % Plot overall accuracy comparison
    figure;
    bar([mean(rfMetrics.Accuracy)*100, mean(knnMetrics.Accuracy)*100]);
    set(gca, 'XTickLabel', {'RF', 'KNN'});
    ylabel('Accuracy (%)');
    title('Mean Accuracy Comparison');
    
    % Plot Cohen’s Kappa comparison
    figure;
    bar([mean(rfMetrics.Kappa), mean(knnMetrics.Kappa)]);
    set(gca, 'XTickLabel', {'RF', 'KNN'});
    ylabel('Cohen''s Kappa');
    title('Cohen''s Kappa Comparison');
    
    % Plot per-class F1-score
    classNames = categories(rfResults(1).trueLabels);
    figure;
    bar([mean(rfMetrics.F1,2), mean(knnMetrics.F1,2)]);
    legend('RF', 'KNN');
    xticklabels(classNames);
    xtickangle(45);
    ylabel('F1 Score');
    title('Per-Class F1 Score Comparison');
    
    % Plot confusion matrix (RF example)
    avgConfMat_RF = zeros(length(classNames));
    for i = 1:numFolds
        avgConfMat_RF = avgConfMat_RF + confusionmat(rfResults(i).trueLabels, rfResults(i).predLabels, 'Order', classNames);
    end
    avgConfMat_RF = avgConfMat_RF / numFolds;
    
    figure;
    heatmap(classNames, classNames, round(avgConfMat_RF), ...
        'Title', 'Average Confusion Matrix - RF', ...
        'XLabel', 'Predicted', ...
        'YLabel', 'True');
    
    % Plot confusion matrix (KNN example)
    avgConfMat_KNN = zeros(length(classNames));
    for i = 1:numFolds
        avgConfMat_KNN = avgConfMat_KNN + confusionmat(knnResults(i).trueLabels, knnResults(i).predLabels, 'Order', classNames);
    end
    avgConfMat_KNN = avgConfMat_KNN / numFolds;
    
    figure;
    heatmap(classNames, classNames, round(avgConfMat_KNN), ...
        'Title', 'Average Confusion Matrix - KNN', ...
        'XLabel', 'Predicted', ...
        'YLabel', 'True');
    
    % Create bar plot Accuracy
    numModels = 10;
    figure;
    bar([rfMetrics.Accuracy; knnMetrics.Accuracy]', 'grouped');
    legend('RF', 'KNN');
    xticks(1:numModels);
    xticklabels(1:numModels);
    xtickangle(45);
    ylabel('Accuracy');
    title('Per-Model Accuracy Comparison');
        
    % Create bar plot Kappa value
    figure;
    bar([rfMetrics.Kappa; knnMetrics.Kappa]', 'grouped');
    legend('RF', 'KNN');
    xticks(1:numModels);
    xticklabels(1:numModels);
    xtickangle(45);
    ylabel('Cohens Kappa Score');
    title('Per-Model Cohens Kappa Comparison');

end

% Helper function 

function metrics = classMetrics(trueLabels, predLabels)
    classes = categories(trueLabels);
    numClasses = numel(classes);
    confMat = confusionmat(trueLabels, predLabels, 'Order', classes);

    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1 = zeros(numClasses, 1);

    for c = 1:numClasses
        TP = confMat(c,c);
        FP = sum(confMat(:,c)) - TP;
        FN = sum(confMat(c,:)) - TP;
        precision(c) = TP / (TP + FP + eps);
        recall(c) = TP / (TP + FN + eps);
        f1(c) = 2 * (precision(c) * recall(c)) / (precision(c) + recall(c) + eps);
    end

    metrics = struct('Precision', precision, 'Recall', recall, 'F1Score', f1);
end

function kappa = cohensKappa(trueLabels, predLabels)
    confMat = confusionmat(trueLabels, predLabels);
    total = sum(confMat(:));
    p0 = trace(confMat) / total;
    rows = sum(confMat, 2);
    cols = transpose(sum(confMat, 1));
    pe = (rows' * cols) / (total^2);
    kappa = (p0 - pe) / (1 - pe + eps);
end

