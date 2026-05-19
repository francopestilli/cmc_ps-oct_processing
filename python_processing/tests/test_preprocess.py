import numpy as np

from psoct_processing.preprocess import subtract_background


def test_subtract_background_preserves_shape():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    y = subtract_background(x)
    assert y.shape == x.shape
