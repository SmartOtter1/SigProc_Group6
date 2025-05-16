function creatingVisualisation(variable, variableCut, stages, stagesCut, hdr)
   
    % plotting variable before and after cutting
    cutPlot(variable, variableCut, hdr);
    
    % plotting stages
    stagesPlot(stages, stagesCut)
end

function cutPlot(variable, variableCut, hdr)
    sigNames = fieldnames(variable);
     Fs = cell(4,1);
    for s = 1:length(sigNames)
        Fs{s} = hdr.samples(find(strcmpi(hdr.label, sigNames{s}), 1) / hdr.duration);
    end
    
        for s = 1:length(sigNames)
            figure 
            subplot(2,1,1);
            plot((1:length(variable.(sigNames{s})))/Fs{s},variable.(sigNames{s}));
            title(sprintf('Original %s', sigNames{s}));
            xlabel('Time (s)');
            ylabel('Amplitude');
            grid on;
            hold on 
            subplot(2,1,2);
            plot((1:length(variableCut.(sigNames{s})))/Fs{s},variableCut.(sigNames{s}));
            title(sprintf('Cut %s', sigNames{s}));
            xlabel('Time (s)');
            ylabel('Amplitude');
            grid on;
  
        end
end

function stagesPlot(stages, stagesCut)
    figure 
    subplot(2,1,1);
    plot(1:length(stages), stages);
    title('Original stages');
    xlabel('Time (s)');
    ylabel('Stage');
    grid on;
    hold on 
    subplot(2,1,2);
    plot(1:length(stagesCut), stagesCut);
    title('Cut stages');
    xlabel('Time (s)');
    ylabel('Stage');
    grid on;
end