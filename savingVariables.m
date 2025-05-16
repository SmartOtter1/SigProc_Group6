function savingVariables(filename, variable )
    % Check if file exists
    
    if exist(filename, 'file')
        fprintf('File "%s" already exists. Overwriting...\n', filename);
    end
    if isa(variable, 'struct')
        % Save the variable
        save(filename, '-struct','variable');
        fprintf('Variable saved successfully to "%s".\n', filename);
    else
        save(filename, 'variable');
        fprintf('Variable saved successfully to "%s".\n', filename);
    end
end