from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ChannelMap:
    """Names and axis conventions for channel-aware arrays."""

    names: tuple[str, ...]
    axis: int = 0

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown channel {name!r}. Available channels: {list(self.names)}") from exc


@dataclass(frozen=True)
class RawAcquisition:
    """Raw channel-aware acquisition data.

    The package convention is that channel-aware arrays use channel axis 0.
    The spectral axis stored here is the axis of the spectral dimension after
    channel canonicalization.
    """

    data: np.ndarray
    path: Path
    metadata: Mapping[str, object]
    channels: ChannelMap
    spectral_axis: int

    def get_channel(self, name: str) -> np.ndarray:
        return np.take(self.data, self.channels.index(name), axis=self.channels.axis)


@dataclass(frozen=True)
class ReconstructedVolume:
    data: np.ndarray
    channels: ChannelMap
    depth_axis: int
    metadata: Mapping[str, object]

    def get_channel(self, name: str) -> np.ndarray:
        return np.take(self.data, self.channels.index(name), axis=self.channels.axis)
