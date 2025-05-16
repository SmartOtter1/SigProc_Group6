% Cuts variables depending on the variance of the EEG signal. If five
% epochs are above the set threshold the epochs are cut at the first
% detected epoch. Signals and stages are cut at the corresponding sample.

function [epochs, epochLabels, signals, stages] = cuttingVariables(epochs, epochLabels, signals, stages, params)
    window =  params.windows.Cutting_Signal; % 5
    threshold = params.signalThreshold; % 1.0e+04
    numEpochs = length(epochs.EEG); % number of total epochs in the EEG signal
  
    % calculating the variance of each EEG epoch 
    varEpochs = zeros(numEpochs); 
    for i = 1:numEpochs
            epochData = epochs.EEG{i};
            varEpochs(i) = var(epochData);
    end
    
    % find the first epoch with higher variance then threshold
    cutIdx = findStableHighVariance(varEpochs, threshold, window);
    
    % cut signals at cutIdx
    signals = cuttingSignals(signals,epochs, cutIdx);
    [epochs, epochLabels] = cuttingEpochs(epochs, epochLabels, cutIdx);
    stages = stages(1:cutIdx*params.epochDuration);
end

function [epochs,epochLabels] = cuttingEpochs(epochs, epochLabels, cutIdx)
    numEpochs = length(epochs.EEG);
    % check if cutIdx is smaller then signal length
    if cutIdx<numEpochs
        sigNames = fieldnames(epochs);
        % trim epochs of each signal to cutIdx
        for s = 1:length(sigNames)
            epochs.(sigNames{s}) = epochs.(sigNames{s})(1:cutIdx);
        end
        epochLabels = epochLabels(1:cutIdx);
        fprintf('Epochs are cut at epoch number %s', cutIdx);
    else
        disp('Epochs are not cut.');
    end
end

function signals = cuttingSignals(signals, epochs, cutIdx)
    numEpochs = length(epochs.EEG);
    % check if cutIdx is smaller then signal length
    if cutIdx<numEpochs
        sigNames = fieldnames(signals);
        % cut signals at equivalent sample 
        for s = 1:length(sigNames)
            numSamples = length(epochs.(sigNames{s}){1});
            signals.(sigNames{s}) = signals.(sigNames{s})(1:cutIdx*numSamples);
        end
        fprintf('Signals are cut at sample number %s', cutIdx*numSamples);
    else
        disp('Signals are not cut.');
    end

end

function idx = findStableHighVariance(varVec, threshold, windowSize)
    % Create a logical vector where the condition is met
    isHigh = varVec > threshold;
    
    % Use moving sum to check if the next `windowSize` entries are all high
    highRun = movsum(isHigh, [0 windowSize-1]) == windowSize;
    
    % Find first index where the run starts
    idx = find(highRun, 1, 'first');
    
    % Return empty if not found
    if isempty(idx)
        disp('No stable high-variance segment found. Returning last window of signal.');
        idx = length(varVec);  % no cutting, therefore idx set to last idx number
    end
end