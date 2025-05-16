function [features, featureNames] = featureExtraction_EMG(epochs, hdr, params)
    % Initialize feature matrix
    numEpochs = length(epochs.EEG); % Assuming EEG always exists
    featureNames = getFeatureNames();

    numEMGFeatures = 16;
    totalFeatures = numEMGFeatures;
    features = zeros(numEpochs, totalFeatures);
    
    % Process each signal type in fixed order
    colOffset = 0;
    
    % EMG Features (16)
    features(:, colOffset+(1:numEMGFeatures)) = extractEMGFeatures(epochs.EMG, hdr, params);
end

%% EMG feature extraction
function emgFeatures = extractEMGFeatures(epochs, hdr, params)
%   emgFeatures - [numEpochs x 11] matrix of features:
%     1 RMS amplitude of envelope
%     2 Mean absolute value (MAV)
%     3 Variance (VAR)
%     4 Waveform length (WL)
%     5 Integrated EMG (IEMG)
%     6 Slope sign changes (SSC)
%     7 Mean value (MV)
%     8 Simple square integral (SSI)
%     9 Maximum peak value (MPV)
%    10 Mean frequency
%    11 Median frequency

 % Sampling rate for EMG
    Fs = hdr.samples(find(strcmpi(hdr.label, 'EMG'), 1)) / hdr.duration;
    numEpochs = length(epochs);

    emgFeatures = zeros(numEpochs,16);

    % Design envelope smoothing filter
    [b_env, a_env] = butter(2, 10/(Fs/2), 'low');
    epsilon = 0.02;

    for i = 1:numEpochs
        data = epochs{i};  % already bandpass filtered raw EMG
        % Full-wave rectification and smoothing
        env = abs(data);
        env = filtfilt(b_env, a_env, env);
        % Artifact removal
        z = zscore(env);
        env(abs(z) > 5) = median(env);

        % 1 RMS amplitude of envelope
        f1 = rms(env);
        % 2 Mean absolute value (MAV)
        f2 = mean(abs(data));
        % 3 Variance (VAR)
        f3 = var(data);
        % 4 Waveform length (WL)
        f4 = sum(abs(diff(data)));
        % 5 Integrated EMG (IEMG)
        f5 = sum(abs(data));
        % 6 Slope sign changes (SSC)
        N = length(data);
        sscCount = 0;
        for k = 2:N-1
            if ((data(k) > data(k-1)) ~= (data(k) > data(k+1))) && ...
               (abs(data(k)-data(k-1)) >= epsilon || abs(data(k)-data(k+1)) >= epsilon)
                sscCount = sscCount + 1;
            end
        end
        f6 = sscCount;
        % 7 Mean value (MV)
        f7 = mean(data);
        % 8 Simple square integral (SSI)
        f8 = sum(data.^2);
        % 9 Maximum peak value (MPV)
        f9 = max(abs(data));

        % Frequency-domain via Welch PSD
        [psd, freq] = pwelch(data, hamming(2*Fs), Fs, 2^nextpow2(2*Fs), Fs);
        totalP = sum(psd);
        % 10 Mean frequency
        f10 = sum(freq .* psd) / totalP;
        % 11 Median frequency (50th percentile)
        cumP = cumsum(psd) / totalP;
        idx = find(cumP >= 0.5, 1);
        f11 = freq(idx);

        % Burst Detection
        [burstMask, burstOnsets, burstOffsets] = detectBursts(data, Fs);
        f12 = length(burstOnsets);
        f13 = sum(burstMask) / Fs;  % in seconds

        % Muscle Tone Analysis
        muscleTone = analyzeMuscleTone(data, Fs);

        f14= muscleTone.meanAmplitude;
        f15=muscleTone.rmsAmplitude;
        f16=muscleTone.duration;

        % Combine features for this epoch
        emgFeatures(i,:) = [f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16];
    end
end


%% Feature Extraction - Burst Detection
% When the signal is above its baseline acitvity (bursts)
% To set threshold - use sliding window since we have a quite large dataset
function [burstMask, burstOnsets, burstOffsets] = detectBursts(envelope, Fs)

    % Parameters
    winLengthSec = 3;              % Sliding window length in seconds
    winLength = round(winLengthSec * Fs);
    thresholdFactor = 1.5;

    % Initialize output
    burstMask = false(size(envelope));

    % Slide window over signal
    stepSize = round(0.5 * winLength);  % 50% overlap
    for i = 1:stepSize:length(envelope) - winLength
        idx = i:i + winLength - 1;
        segment = envelope(idx);

        % Compute local threshold
        localThresh = mean(segment) + thresholdFactor * std(segment);

        % Mark burst if above threshold
        burstMask(idx) = burstMask(idx) | (segment > localThresh);
    end

    % Detect rising/falling edges
    burstDiff = diff([0, burstMask, 0]);
    burstOnsets = find(burstDiff == 1);
    burstOffsets = find(burstDiff == -1) - 1;

    % Filter short bursts
    minDuration = 0.05 * Fs;
    valid = (burstOffsets - burstOnsets + 1) >= minDuration;
    burstOnsets = burstOnsets(valid);
    burstOffsets = burstOffsets(valid);

    % Clean burst mask
    burstMask = false(size(envelope));
    for i = 1:length(burstOnsets)
        burstMask(burstOnsets(i):burstOffsets(i)) = true;
    end
end


%% Muscle Tone Analysis
function muscleTone = analyzeMuscleTone(envelope, Fs)

    % Mean amplitude (muscle tone proxy)
    meanAmp = mean(envelope);
    
    % RMS amplitude
    rmsAmp = rms(envelope);

    % Duration in seconds
    duration = length(envelope) / Fs;

    % Package output into a structure
    muscleTone.meanAmplitude = meanAmp;
    muscleTone.rmsAmplitude = rmsAmp;
    muscleTone.duration = duration;
end

function featureNames = getFeatureNames()
    % EMG Features (17)
    emgBase = {
    'EMG_rms';    
    'EMG_mean';
        'EMG_variance';
        'EMG_waveformlength';
        'EMG_IEMG';
        'EMG_SSC';
        'EMG_MV';
        'EMG_SSI';
        'EMG_MPV';
        'EMG_meanFreq';
        'EMG_medFreq';
        'EMG_burstDetec';
        'EMG_burstDetecSum';
        'EMG_toneMean';
        'EMG_toneRMS';
        'EMG_toneDuration'
    };
    % Combine all names
    featureNames = [emgBase];

end