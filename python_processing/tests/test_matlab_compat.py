from pathlib import Path

import numpy as np

from psoct_processing.contrasts import compute_psoct_reflectivity
from psoct_processing.enface import combo_mask_cross_d
from psoct_processing.io import parse_labview_header, read2024_scan_stack
from psoct_processing.reconstruct import interp_and_cdp, interpolate_wavelengths_octoplus_2024
from psoct_processing.stitch import stitch_tile_matrix_matlab


def test_parse_labview_header():
    h = parse_labview_header("alineLength=1024; blineLength=500; buffersPerFile=10; name='x';")
    assert h.get_int("alineLength") == 1024
    assert h.get_int("buffersPerFile") == 10
    assert h.values["name"] == "x"


def test_read2024_scan_stack(tmp_path: Path):
    aline, bline, nb = 4, 3, 2
    values = np.arange(aline * 2 * bline * nb, dtype=">i2")
    path = tmp_path / "Slice_1_Tile_1_840_1.dat"
    with path.open("wb") as f:
        f.write(f"alineLength={aline}; blineLength={bline}; buffersPerFile={nb};\n".encode())
        # MATLAB fwrite/fread + reshape column-major equivalent.
        f.write(values.tobytes())
    raw, bg, bg_lines, meta = read2024_scan_stack(str(tmp_path / "Slice_1_Tile_1_840_"), [1], byte_order="big")
    assert raw.shape == (aline * 2, bline, nb)
    assert bg.shape == (aline * 2,)
    assert bg_lines.shape == (aline * 2, nb)
    assert meta["shape"] == raw.shape


def test_interp_and_cdp_shapes():
    rng = np.random.default_rng(1)
    ch1 = rng.normal(size=(1024, 5))
    ch2 = rng.normal(size=(1024, 5))
    cal = interpolate_wavelengths_octoplus_2024()
    cdp1, cdp2 = interp_and_cdp(ch1, ch2, cal, auto_peak_corr_cut=10)
    assert cdp1.shape == (502, 5)
    assert cdp2.shape == (502, 5)
    ref = compute_psoct_reflectivity(cdp1, cdp2)
    assert ref.shape == cdp1.shape
    assert np.isfinite(ref).all()


def test_combo_mask_cross_d():
    v = np.array([[[1, 10]], [[5, 20]], [[9, 30]]], dtype=float)
    out = combo_mask_cross_d(v, db_limit=6, endp=3)
    assert out.shape == (1, 2)
    assert out[0, 0] == 9
    assert out[0, 1] == 20


def test_stitch_tile_matrix_matlab_shape():
    tm = np.array([[1, 2], [3, 4]])
    tiles = {i: np.ones((10, 8), dtype=np.float32) * i for i in range(1, 5)}
    out = stitch_tile_matrix_matlab(tm, tiles, overlap_percent=25)
    assert out.shape == (18, 14)
    assert np.isfinite(out).all()
