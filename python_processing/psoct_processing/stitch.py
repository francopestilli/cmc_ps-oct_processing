from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tifffile


def linear_weight(shape: tuple[int, int], overlap_px: int) -> np.ndarray:
    h, w = shape
    weight = np.ones((h, w), dtype=np.float32)
    if overlap_px <= 0:
        return weight
    ramp = np.linspace(0.0, 1.0, overlap_px, dtype=np.float32)
    weight[:, :overlap_px] *= ramp[None, :]
    weight[:, -overlap_px:] *= ramp[::-1][None, :]
    weight[:overlap_px, :] *= ramp[:, None]
    weight[-overlap_px:, :] *= ramp[::-1][:, None]
    return weight


def infer_grid(n_tiles: int) -> tuple[int, int]:
    rows = int(math.floor(math.sqrt(n_tiles)))
    while rows > 1 and n_tiles % rows != 0:
        rows -= 1
    cols = int(math.ceil(n_tiles / rows))
    return rows, cols


def stitch_tiles(tiles: list[np.ndarray], tile_grid: tuple[int, int] | None = None, overlap_fraction: float = 0.15, blending: str = "linear") -> np.ndarray:
    if not tiles:
        raise ValueError("No tiles supplied for stitching.")
    if any(t.ndim != 2 for t in tiles):
        raise ValueError("This initial stitcher expects 2D en-face tiles.")

    rows, cols = tile_grid or infer_grid(len(tiles))
    if rows * cols < len(tiles):
        raise ValueError("Tile grid is too small for number of tiles.")

    th, tw = tiles[0].shape
    if any(t.shape != (th, tw) for t in tiles):
        raise ValueError("All tiles must have the same shape in the initial stitcher.")

    overlap_y = int(round(th * overlap_fraction))
    overlap_x = int(round(tw * overlap_fraction))
    step_y = th - overlap_y
    step_x = tw - overlap_x
    out_h = step_y * (rows - 1) + th
    out_w = step_x * (cols - 1) + tw

    acc = np.zeros((out_h, out_w), dtype=np.float32)
    weights = np.zeros((out_h, out_w), dtype=np.float32)
    base_weight = linear_weight((th, tw), max(overlap_y, overlap_x)) if blending == "linear" else np.ones((th, tw), dtype=np.float32)

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y = r * step_y
        x = c * step_x
        if blending == "overwrite":
            acc[y:y+th, x:x+tw] = tile
            weights[y:y+th, x:x+tw] = 1.0
        else:
            acc[y:y+th, x:x+tw] += tile.astype(np.float32) * base_weight
            weights[y:y+th, x:x+tw] += base_weight

    return acc / np.maximum(weights, 1e-12)


def load_tiff_tiles(paths: list[str | Path]) -> list[np.ndarray]:
    return [tifffile.imread(str(p)).astype(np.float32) for p in paths]


def blending_matrix_matlab(tile_matrix: np.ndarray, overlap_px: int, tile_height: int, tile_width: int) -> list[list[np.ndarray]]:
    """Replicate MATLAB BlendingMatrix.m tile-specific linear weights."""
    tm = np.asarray(tile_matrix)
    rows, cols = tm.shape
    ov = int(overlap_px)
    if ov <= 0:
        return [[np.ones((tile_height, tile_width), dtype=np.float32) for _ in range(cols)] for _ in range(rows)]

    xcry = np.arange(1, ov + 1, dtype=np.float32)
    ycry = (1.0 / ov) * xcry - (1.0 / (2.0 * ov))
    yc = ycry[::-1]
    xc = yc.reshape(-1, 1)
    xcf = xc[::-1]

    def ones() -> np.ndarray:
        return np.ones((tile_height, tile_width), dtype=np.float32)

    S = ones(); S[:ov, :] *= xcf; S[:, :ov] *= ycry; S[:, -ov:] *= yc; S[-ov:, :] *= xc
    C1 = ones(); C1[:, -ov:] *= yc; C1[-ov:, :] *= xc
    E1 = ones(); E1[:, :ov] *= ycry; E1[:, -ov:] *= yc; E1[-ov:, :] *= xc
    C2 = ones(); C2[:, :ov] *= ycry; C2[-ov:, :] *= xc
    E2 = ones(); E2[:ov, :] *= xcf; E2[:, :ov] *= ycry; E2[-ov:, :] *= xc
    C3 = ones(); C3[:ov, :] *= xcf; C3[:, :ov] *= ycry
    E3 = ones(); E3[:ov, :] *= xcf; E3[:, :ov] *= ycry; E3[:, -ov:] *= yc
    C4 = ones(); C4[:ov, :] *= xcf; C4[:, -ov:] *= yc
    E4 = ones(); E4[:ov, :] *= xcf; E4[:, -ov:] *= yc; E4[-ov:, :] *= xc
    R1 = ones(); R1[:, -ov:] *= yc
    R2 = ones(); R2[-ov:, :] *= xc
    R3 = ones(); R3[:, :ov] *= ycry
    R4 = ones(); R4[:ov, :] *= xcf
    R5 = ones(); R5[:, :ov] *= ycry; R5[:, -ov:] *= yc
    R6 = ones(); R6[:ov, :] *= xcf; R6[-ov:, :] *= xc

    out: list[list[np.ndarray]] = [[ones() for _ in range(cols)] for _ in range(rows)]
    if rows == 1:
        for j in range(cols):
            out[0][j] = R1 if j == 0 else R3 if j == cols - 1 else R5
    elif cols == 1:
        for i in range(rows):
            out[i][0] = R2 if i == 0 else R4 if i == rows - 1 else R6
    else:
        for i in range(rows):
            for j in range(cols):
                if i == 0 and j == 0:
                    out[i][j] = C1
                elif i == 0 and j == cols - 1:
                    out[i][j] = C2
                elif i == rows - 1 and j == cols - 1:
                    out[i][j] = C3
                elif i == rows - 1 and j == 0:
                    out[i][j] = C4
                elif i == 0:
                    out[i][j] = E1
                elif j == 0:
                    out[i][j] = E4
                elif i == rows - 1:
                    out[i][j] = E3
                elif j == cols - 1:
                    out[i][j] = E2
                else:
                    out[i][j] = S
    return out


def stitch_tile_matrix_matlab(tile_matrix: np.ndarray, tiles_by_id: dict[int, np.ndarray], overlap_percent: float, flip: bool = False) -> np.ndarray:
    """Replicate MStitchFCN_Vlad.m in memory for 2D en-face tiles."""
    tm = np.asarray(tile_matrix, dtype=int)
    rows, cols = tm.shape
    first = np.asarray(tiles_by_id[int(tm[0, 0])], dtype=np.float32)
    y, x = first.shape
    ov = int(round((overlap_percent / 100.0) * x))
    a1 = y - ov
    a2 = x - ov
    d1 = (a2 * cols) + ov
    d2 = (a1 * rows) + ov
    weights = blending_matrix_matlab(tm, ov, y, x)
    acc = np.zeros((y + a1 * rows, x + a2 * cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            tile = np.asarray(tiles_by_id[int(tm[i, j])], dtype=np.float32)
            if tile.shape != (y, x):
                raise ValueError("All tiles must have the same 2D shape.")
            if flip:
                tile = np.flip(tile, axis=0)
            yy = a1 * i
            xx = a2 * j
            acc[yy:yy + y, xx:xx + x] += weights[i][j] * tile
    return acc[:d2, :d1]
