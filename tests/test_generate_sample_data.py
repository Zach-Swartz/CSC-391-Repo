import os
import numpy as np
from project_2.phase_2.tools.generate_sample_data import generate_sample_data
from project_2.phase_2 import common_utils as cu


def test_generate_and_load(tmp_path):
    out = tmp_path / "sample_data"
    out = str(out)
    generated = generate_sample_data(out)
    assert os.path.isdir(generated)
    npz = os.path.join(generated, 'calibration_results_example.npz')
    assert os.path.exists(npz)
    mtx, dist = cu.load_calibration(npz)
    assert mtx.shape == (3, 3)
    assert dist is not None
