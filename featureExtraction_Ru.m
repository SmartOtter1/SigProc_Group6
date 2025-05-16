function [features, featureNames] = featureExtraction_Ru(epochs, hdr, params)
    % Initialize feature matrix
    numEpochs = length(epochs.EEG); % Assuming EEG always exists
    featureNames = getFeatureNames();
    % define number of features for each signal
    numEEGFeatures = 16;
    numEOGFeatures = 14; 
    numEOGcombinedFeatures = 5;
    numEMGFeatures = 17;
    totalFeatures = numEEGFeatures*2 + numEOGFeatures*2 + numEOGcombinedFeatures + numEMGFeatures; % EEG + EEGsec + EOGR + EOGL + EOG combined + EMG
    features = zeros(numEpochs, totalFeatures);
    
    % Process each signal type in fixed order
    colOffset = 0;
    
    % EEG Features (16)
    features(:, colOffset+(1:numEEGFeatures)) = extractEEGFeatures(epochs.EEG, hdr, params);
    colOffset = colOffset + numEEGFeatures;
    
    % EEGsec Features (16)
    features(:, colOffset+(1:numEEGFeatures)) = extractEEGFeatures(epochs.EEGsec, hdr, params);
    colOffset = colOffset + numEEGFeatures;
    
    % EOGR Features (14)
    features(:, colOffset+(1:numEOGFeatures)) = extractEOGFeatures(epochs.EOGR, hdr, params);
    colOffset = colOffset + numEOGFeatures;
    
    % EOGL Features (14)
    features(:, colOffset+(1:numEOGFeatures)) = extractEOGFeatures(epochs.EOGL, hdr, params);
    colOffset = colOffset + numEOGFeatures;

    % EOGL and EOGR combined Features (5)
    features(:, colOffset+(1:numEOGcombinedFeatures)) = extractEOGcombinedFeatures(epochs.EOGL, epochs.EOGR, hdr, params);
    colOffset = colOffset + numEOGcombinedFeatures;

    % EMG Features (17)
    features(:, colOffset+(1:numEMGFeatures)) = extractEMGFeatures(epochs.EMG, hdr, params);
end

%% EEG feature extraction
function eegFeatures = extractEEGFeatures(epochs, hdr, params)
    Fs = hdr.samples(find(strcmpi(hdr.label, 'EEG'), 1)) / hdr.duration;
    numEpochs = length(epochs);
    eegFeatures = zeros(numEpochs, 16);
    
    for i = 1:numEpochs
        epochData = epochs{i};
        
        % Time-domain features (6)
        eegFeatures(i,1) = mean(epochData);
        eegFeatures(i,2) = var(epochData);
        eegFeatures(i,3) = skewness(epochData);
        eegFeatures(i,4) = kurtosis(epochData);
        eegFeatures(i,5) = zerocrossrate(epochData);
        [~, ~, eegFeatures(i,6)] = hjorth_parameters(epochData);
        
        % Frequency-domain features (10)
        [psd, f] = pwelch(epochData, hamming(2*Fs), Fs, 2^nextpow2(2*Fs), Fs);
        totalPower = sum(psd);
        
        % Relative band powers (5)
        bandNames = fieldnames(params.bands);
        for b = 1:numel(bandNames)
            freqRange = params.bands.(bandNames{b});
            idx = f >= freqRange(1) & f <= freqRange(2);
            eegFeatures(i,6+b) = sum(psd(idx)) / totalPower * 100;
        end
        
        % Spectral features (2)
        psdNorm = psd / (sum(psd) + eps);
        eegFeatures(i,12) = -sum(psdNorm .* log2(psdNorm + eps)); % Spectral entropy
        cumPower = cumsum(psdNorm);
        temp = find(cumPower >= 0.9, 1, 'first');
        if ~isempty(temp)
            eegFeatures(i,13) = f(temp);  % SEF90
        else
            eegFeatures(i,13) = 0;       % Fallback if no value meets the condition
        end

        % Wavelet features (3)
        [C, L] = wavedec(epochData, 5, 'db4');
        eegFeatures(i,14) = sum(appcoef(C, L, 'db4', 5).^2); % Delta
        eegFeatures(i,15) = sum(detcoef(C, L, 4).^2); % Theta
        eegFeatures(i,16) = sum(detcoef(C, L, 3).^2); % Alpha
    end
end

%% EOG features extraction (updated for pipeline integration)
function eogFeatures = extractEOGFeatures(epochs, hdr, params)
    % Get sampling rate from header
    if contains(inputname(1), 'EOGR')
        Fs = hdr.samples(find(strcmpi(hdr.label, 'EOGR'),1))/hdr.duration;
    else % EOGL
        Fs = hdr.samples(find(strcmpi(hdr.label, 'EOGL'),1))/hdr.duration;
    end
    
    numEpochs = length(epochs);
    eogFeatures = zeros(numEpochs, 14); % 14 features per EOG channel
    
    for i = 1:numEpochs
        signal = epochs{i};

        % Time-domain features (8)
        eogFeatures(i,1) = mean(signal);
        eogFeatures(i,2) = sum(signal.^2);
        eogFeatures(i,3) = rms(signal) / mean(abs(signal));
        eogFeatures(i,4) = std(signal);
        eogFeatures(i,5) = skewness(signal);
        eogFeatures(i,6) = kurtosis(signal);
        eogFeatures(i,7) = var(signal);
        eogFeatures(i,8) = zerocrossrate(signal);

        % Frequency-domain features (3)
        N = length(signal);
        Y = fft(signal);
        P = abs(Y(1:N/2)).^2; % Power spectrum
        freq = linspace(0, Fs/2, N/2); % Frequency vector
        
        % Compute relative energy in 0-2 Hz and 2-4 Hz
        eogFeatures(i,9) = sum(P);
        eogFeatures(i,10) = sum(P(freq >= 0 & freq < 2)) / eogFeatures(i,9);
        eogFeatures(i,11) = sum(P(freq >= 2 & freq < 4)) / eogFeatures(i,9);
        
        % 1. Movement Density
        deriv = diff(signal);
        eogFeatures(i,12) = sum(abs(deriv) > 10)/length(deriv); % Threshold 10µV
        
        % 2. Blink Rate
        [rate, characteristics] = detectBlinks(signal, Fs);
        eogFeatures(i,13) = rate; % blinks/min 

        % 3. SEM Proportion
        semMask = abs(deriv) < 50 & abs(signal(1:end-1)) > 5; % Slow movement criteria
        eogFeatures(i,14) = sum(semMask)/length(semMask);
        
        
    end
end

function eogcombinedFeatures = extractEOGcombinedFeatures(epochL, epochR, hdr, params)% SEM feature extraction
            
    Fs = hdr.samples(find(strcmpi(hdr.label, 'EOGR'),1))/hdr.duration;
    numEpochs = length(epochR);

    eogcombinedFeatures = zeros(numEpochs, 5); % 5 features per EOG channel
    freqBands = params.bands.SEM;
    for i = 1:numEpochs
        signalL = epochL{i};
        signalR = epochR{i};
            
        % SEM detection
        % calculating heo signal by taking difference of both
        % records, used for SEM feature detection
        signalSEM = (signalL-signalR);
        
        % Step 1: Convert Signal to Frequency Domain
        N = length(signalSEM); 
        spectrum = abs(fft(signalSEM)); % Compute magnitude of FFT
        spectrum = spectrum(1:N/2+1); % Keep only positive frequencies
        freqs = (0:N/2) * (Fs/N); % Frequency axis
        
        % Step 2: Decompose Signal into Frequency Bands
        
        filterOrder = 4; % Butterworth filter order
        decomposed_signals = zeros(length(signalSEM), size(freqBands, 1));
        
        for s = 1:size(freqBands, 1)
            [b, a] = butter(filterOrder, freqBands(s, :) / (Fs/2), 'bandpass');
            decomposed_signals(:, s) = filtfilt(b, a, signalSEM);
        end
    
        for s = 1: size(decomposed_signals,2)
            eogcombinedFeatures(i,s) = sum(decomposed_signals(:,s).^2);
        end

        % REM feature extraction         
        % Compute first derivatives of EOG signals
        diff1 = diff(signalL);
        diff2 = diff(-signalR);
    
        % Find indices where both signals are increasing beyond
        % threshold and return the number of points which fulfill
        % criteria
        risingIndices = find(diff1 > 1.5 & diff2 > 1.5);
        eogcombinedFeatures(i,5) = length(risingIndices);

    end
end

function [activity, mobility, complexity] = hjorth_parameters(signal)
    diff1 = diff(signal);
    diff2 = diff(diff1);
    activity = var(signal);
    mobility = std(diff1)/std(signal);
    complexity = std(diff2)/std(diff1)/mobility;
end



function [rate, characteristics] = detectBlinks(signal, Fs)
    % Find peaks in the negative direction (EOG blinks are typically negative)
    [blinkAmps, blinkLocs] = findpeaks(-signal, ...
        'MinPeakHeight', prctile(-signal, 97.5), ... % Top 5% amplitudes
        'MinPeakDistance', round(0.1*Fs)); % 300ms refractory period
    
    % Calculate characteristics
    rate = length(blinkLocs) / (length(signal)/Fs) * 60; % blinks/min
    
    % Blink properties
    characteristics = struct();
    if ~isempty(blinkLocs)
        durations = zeros(size(blinkLocs));
        for i = 1:length(blinkLocs)
            % Find blink start and end
            winStart = max(1, blinkLocs(i)-round(0.1*Fs));
            winEnd = min(length(signal), blinkLocs(i)+round(0.1*Fs));
            window = signal(winStart:winEnd);
            
            % Duration as FWHM
            halfMax = blinkAmps(i)/2;
            crossPoints = find(window <= -halfMax);
            if length(crossPoints) >= 2
                durations(i) = (crossPoints(end)-crossPoints(1))/Fs;
            end
        end
        characteristics.mean_duration = mean(durations(durations>0));
        characteristics.mean_amplitude = mean(-blinkAmps);
    else
        characteristics.mean_duration = 0;
        characteristics.mean_amplitude = 0;
    end
end

function [slowRatio, rapidRatio] = classifyEyeMovements(signal, Fs)
    % Velocity calculation
    velocity = abs(diff(signal)) * Fs; % µV/s
    
    % Thresholds (adjust based on your data)
    slowThreshold = 50; % µV/s
    rapidThreshold = 100; % µV/s
    
    % Classify movements
    isSlow = velocity > slowThreshold & velocity <= rapidThreshold;
    isRapid = velocity > rapidThreshold;
    
    % Calculate ratios
    slowRatio = sum(isSlow)/length(velocity);
    rapidRatio = sum(isRapid)/length(velocity);
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

    emgFeatures = zeros(numEpochs,17);


    rmsSignal = cell(numEpochs,1);
     % Design envelope smoothing filter
    [b_env, a_env] = butter(2, 10/(Fs/2), 'low');
    epsilon = 0.02;
    for i = 1:numEpochs
        epochData = epochs{i};

        windowLengthSec = params.windows.EMG_RMS;
        windowSamples = Fs * windowLengthSec;  % = 250 samples
        
        overlapSamples = round(params.windows.EMG_overlap * windowSamples);  % 50% overlap = 125 samples
        step = windowSamples - overlapSamples;
        numWindows = floor((length(epochData) - windowSamples) / step) + 1;
        
        rmsValues = zeros(1, numWindows);
        
        for j = 1:numWindows
            startIdx = (j-1)*step + 1;
            endIdx = startIdx + windowSamples - 1;
            
            windowData = epochData(startIdx:endIdx);
            rmsValues(j) = sqrt(mean(windowData.^2));
        end
        rmsSignal{i} = transpose(rmsValues);
    end

    meanSignal = mean(cell2mat(rmsSignal));
    % Initialize cell array for peaks
    pksSignal = cell(numEpochs,1);
    pksLoc = cell(numEpochs,1);
    
    % Loop through each cell to find peaks above meanValue
    for i = 1:numEpochs
        dataRMS = rmsSignal{i};  % Get current signal (59x1)
        [pks, loc] = findpeaks(dataRMS, 'MinPeakHeight', meanSignal);  % Detect peaks above mean
        pksSignal{i} = pks;  % Store in corresponding cell
        pksLoc{i} = loc;
        f0 = length(pks);

 
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
        emgFeatures(i,:) = [f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16];
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


%% Feature Names table 
function featureNames = getFeatureNames()
    % EEG Features (16)
    eegBase = {
        'EEG_mean';
        'EEG_variance';
        'EEG_skewness';
        'EEG_kurtosis';
        'EEG_zero_crossing_rate';
        'EEG_hjorth_complexity';
        'EEG_delta_power';
        'EEG_theta_power';
        'EEG_alpha_power';
        'EEG_beta_power';
        'EEG_gamma_power';
        'EEG_spectral_entropy';
        'EEG_SEF90';
        'EEG_wavelet_delta';
        'EEG_wavelet_theta';
        'EEG_wavelet_alpha'
    };
    
    % EEGsec Features (same as EEG)
    eegsecBase = strrep(eegBase, 'EEG_', 'EEGsec_');
    
    % EOG Features (14)
    eogBase = {
        'EOGR_mean';
        'EOGR_power';
        'EOGR_ffactor';
        'EOGR_standard_deviation';
        'EOGR_skewness';
        'EOGR_kurtosis';
        'EOGR_variance'
        'EOG_zero_crossing_rate';
        'EOGR_frequencyPower';
        'EOGR_frequencyPower0_2Hz';
        'EOGR_frequencyPower2_4Hz';
        'EOGR_movement_density';
        'EOGR_blink_rate';
        'EOGR_SEM_proportion';
        'EOGL_mean';
        'EOGL_power';
        'EOGL_ffactor';
        'EOGL_standard_deviation';
        'EOGL_skewness';
        'EOGL_kurtosis';
        'EOGL_variance'
        'EOGL_zero_crossing_rate';
        'EOGL_frequencyPower';
        'EOGL_frequencyPower0_2Hz';
        'EOGL_frequencyPower2_4Hz';
        'EOGL_movement_density';
        'EOGL_blink_rate';
        'EOGL_SEM_proportion'
    };
    % EOG combined Features (5)
    eogcombinedBase = {
        'EOGR_L_wEnergy1';
        'EOGR_L_wEnergy2';
        'EOGR_L_wEnergy3';
        'EOGR_L_wEnergy4';
        'EOGR_L_risingIndices';
        };


    % EMG Features (17)
    emgBase = {
        'EMG_RMS_Amplitudes';
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
    featureNames = [eegBase; eegsecBase; eogBase;eogcombinedBase;emgBase];
end










