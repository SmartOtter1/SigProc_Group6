function [hdr, record, stages] = loadData(fileNum)
    edfFilename = sprintf('Scripts/Data/R%d.edf', fileNum); %adapted path
    xmlFilename = sprintf('Scripts/Data/R%d.xml', fileNum); %adapted path
    
    % Read files
    [hdr, record] = edfread(edfFilename);
    [~, stages] = readXML(xmlFilename);
end