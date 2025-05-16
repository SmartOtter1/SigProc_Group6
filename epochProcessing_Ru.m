    % Get sampling frequency from all the different signals to continue cutting
    % the signals into 30s epochs. 
    % Returns variables: 
    % epochs: Structure variable including all signals cut into 30s epochs
    % taking differen sampling frequencies into account
    % epochLabels: Array of stages aligned to epochs
function [epochs, epochLabels] = epochProcessing_Ru(signals, stages, hdr, params)
    sigNames = fieldnames(signals);
    Fs = cell(4,1);
    samplesPerEpoch = cell(4,1);
    for s = 1:length(sigNames)
        Fs{s} = hdr.samples(find(strcmpi(hdr.label, sigNames{s}), 1) / hdr.duration);
        samplesPerEpoch{s} = params.epochDuration * Fs{s};
    end

    % Initialize epoch structure
    epochs = struct();
    for s = 1:length(sigNames)
        epochs.(sigNames{s}) = {};
    end
    
    % Segment all signals
    numEpochs = floor(length(signals.(sigNames{1})) / samplesPerEpoch{1});
    for s = 1:length(sigNames)
        trimmed = signals.(sigNames{s})(1:numEpochs*samplesPerEpoch{s});
        epochs.(sigNames{s}) = mat2cell(trimmed, 1, ...
            repmat(samplesPerEpoch{s}, 1, numEpochs))';
    end
    
    % Label epochs (same as before)
    stagesPerEpoch = params.epochDuration;
    epochLabels = zeros(numEpochs, 1);
    for i = 1:numEpochs
        epochStages = stages((i-1)*stagesPerEpoch+1 : min(i*stagesPerEpoch, end));
        [uniqueStages, ~, idx] = unique(epochStages);
        counts = accumarray(idx, 1);
        [~, maxIdx] = max(counts);
        if ismember(uniqueStages(maxIdx), params.stageValues)
            epochLabels(i) = uniqueStages(maxIdx);
        else
            epochLabels(i) = NaN;
        end
    end
    
    % Remove invalid epochs
    validEpochs = ~isnan(epochLabels);
    for s = 1:length(sigNames)
        epochs.(sigNames{s}) = epochs.(sigNames{s})(validEpochs);
    end
    epochLabels = epochLabels(validEpochs);
end