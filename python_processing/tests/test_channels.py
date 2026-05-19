from pathlib import Path

import numpy as np

from psoct_processing.config import IOConfig, ChannelLayoutConfig, ChannelConfig
from psoct_processing.io import canonicalize_channels, read_raw_dat


def test_axis_channel_layout_moves_channel_to_front():
    cfg = IOConfig(
        shape=(4, 2, 8),
        spectral_axis=-1,
        channels=ChannelLayoutConfig(
            n_channels=2,
            layout="axis",
            channel_axis=1,
            channels=[ChannelConfig(name="co_pol", index=0), ChannelConfig(name="cross_pol", index=1)],
        ),
    )
    data = np.zeros((4, 2, 8), dtype=np.float32)
    out, channels, spectral_axis = canonicalize_channels(data, cfg)
    assert out.shape == (2, 8, 4)
    assert channels.names == ("co_pol", "cross_pol")
    assert spectral_axis == 1


def test_interleaved_alines_layout():
    cfg = IOConfig(
        shape=(3, 6, 8),
        spectral_axis=-1,
        channels=ChannelLayoutConfig(
            n_channels=2,
            layout="interleaved_alines",
            channels=[ChannelConfig(name="co_pol", index=0), ChannelConfig(name="cross_pol", index=1)],
        ),
    )
    data = np.zeros((3, 6, 8), dtype=np.float32)
    out, channels, spectral_axis = canonicalize_channels(data, cfg)
    assert out.shape == (2, 8, 3, 3)
    assert spectral_axis == 1
    assert channels.index("cross_pol") == 1


def test_read_raw_dat_with_channel_schema(tmp_path: Path):
    arr = np.arange(3 * 6 * 8, dtype=np.int16)
    path = tmp_path / "test.dat"
    arr.tofile(path)
    cfg = IOConfig(
        shape=(3, 6, 8),
        spectral_axis=-1,
        channels=ChannelLayoutConfig(
            n_channels=2,
            layout="interleaved_alines",
            channels=[ChannelConfig(name="co_pol", index=0), ChannelConfig(name="cross_pol", index=1)],
        ),
    )
    raw = read_raw_dat(path, cfg)
    assert raw.data.shape == (2, 8, 3, 3)
    assert raw.get_channel("co_pol").shape == (8, 3, 3)
