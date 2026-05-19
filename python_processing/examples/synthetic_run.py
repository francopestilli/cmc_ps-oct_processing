from pathlib import Path

import numpy as np

raw = Path("synthetic_raw")
raw.mkdir(exist_ok=True)
x = np.random.default_rng(1).normal(size=(16, 32, 256)).astype("<i2")
x.tofile(raw / "synthetic.dat")
print("Wrote synthetic_raw/synthetic.dat")
print("Run:")
print("psoct-process run --input-dir synthetic_raw --output-dir synthetic_processed --config config/default.yaml")
print("Set io.shape: [16, 32, 256] in config/default.yaml before running.")
