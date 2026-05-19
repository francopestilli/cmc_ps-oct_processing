import numpy as np

from psoct_processing.stitch import stitch_tiles


def test_stitch_two_by_two():
    tiles = [np.ones((10, 10), dtype=np.float32) * i for i in range(4)]
    out = stitch_tiles(tiles, tile_grid=(2, 2), overlap_fraction=0.1)
    assert out.ndim == 2
    assert out.shape == (19, 19)
