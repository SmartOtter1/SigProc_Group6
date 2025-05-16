function filteredSignal = preprocessSignal(record, hdr, signalName, filterRange, stageNumber)
    sigIdx = find(strcmpi(hdr.label, signalName), 1);
    Fs = hdr.samples(sigIdx) / hdr.duration;
    rawSignal = record(sigIdx, :);
    
    % Bandpass filter
    [b_band, a_band] = butter(6, filterRange/(Fs/2), 'bandpass');
    filteredSignal = filtfilt(b_band, a_band, rawSignal);
    
    % Notch filter (for EEG only)
    if contains(signalName, 'EEG', 'IgnoreCase', true)
        [b_notch, a_notch] = iirnotch(50/(Fs/2), 1/(Fs/2));
        filteredSignal = filtfilt(b_notch, a_notch, filteredSignal);
    end
end