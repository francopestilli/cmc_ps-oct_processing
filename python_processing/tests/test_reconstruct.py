import numpy as np

from psoct_processing.config import ReconstructionConfig
from psoct_processing.reconstruct import reconstruct_volume


def test_reconstruct_volume_positive_depth_half():
    x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    y = reconstruct_volume(x, ReconstructionConfig(fft_size=8, window="none", subtract_dc=False))
    assert y.shape == (4, 4)
    assert np.iscomplexobj(y)
