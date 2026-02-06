import interpolator as intp
import numpy as np

import xtrack as xt

fieldmap_file = "cm70.fluk.gz"

field_interpolator_2d = intp.load_field_map(fieldmap_file)

def field_3d(x, y, s):
    """Wrapper to call the 3D field interpolator."""
    bx, by = field_interpolator_2d(x, y)
    bz = np.zeros_like(bx)  # Assuming Bz is zero
    return bx, by, bs

length_scraper = 3.0  # m
num_slices = 20

length_slice = length_scraper / num_slices



