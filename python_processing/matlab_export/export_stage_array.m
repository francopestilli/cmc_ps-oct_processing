function export_stage_array(output_dir, stage_name, array)
%EXPORT_STAGE_ARRAY Save a MATLAB intermediate array for Python validation.
%   export_stage_array(output_dir, stage_name, array) writes both a .mat file
%   and a compact metadata text file. Use this inside the MATLAB processing
%   pipeline after important stages such as raw reshaping, background
%   subtraction, interpolation, FFT, contrast computation, en-face projection,
%   and stitching.

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

mat_path = fullfile(output_dir, [stage_name '.mat']);
save(mat_path, 'array', '-v7.3');

meta_path = fullfile(output_dir, [stage_name '_metadata.txt']);
fid = fopen(meta_path, 'w');
fprintf(fid, 'stage=%s\n', stage_name);
fprintf(fid, 'class=%s\n', class(array));
fprintf(fid, 'size=%s\n', mat2str(size(array)));
fclose(fid);
end
