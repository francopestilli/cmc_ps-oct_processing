from pathlib import Path
import numpy as np

from psoct_processing.validation import compare_arrays, compare_stage_directories


def test_compare_arrays_identical():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = compare_arrays(arr, arr.copy(), "identity")
    assert result.rmse == 0.0
    assert result.max_abs_error == 0.0
    assert result.shape_matlab == (3, 4)


def test_compare_stage_directories(tmp_path: Path):
    matlab = tmp_path / "matlab"
    python = tmp_path / "python"
    matlab.mkdir()
    python.mkdir()
    arr = np.ones((2, 3), dtype=np.float32)
    np.save(matlab / "stage_a.npy", arr)
    np.save(python / "stage_a.npy", arr + 1)
    report = tmp_path / "report.json"
    results = compare_stage_directories(matlab, python, output_json=report)
    assert len(results) == 1
    assert results[0].mean_abs_error == 1.0
    assert report.exists()
