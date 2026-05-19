# MATLAB intermediate export helpers

Use `export_stage_array.m` inside the MATLAB pipeline to save intermediate arrays for validation.

Example:

```matlab
export_stage_array('./matlab_reference', 'raw_reshaped', raw_reshaped);
export_stage_array('./matlab_reference', 'background_corrected', corrected);
export_stage_array('./matlab_reference', 'fft_magnitude', fft_magnitude);
```

Then compare against Python outputs:

```bash
psoct-process validate --matlab-output ./matlab_reference --python-output ./python_output --report validation_report.json
```
