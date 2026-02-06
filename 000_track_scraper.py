import interpolator as intp
import numpy as np

import xtrack as xt

fieldmap_file = "cm70.fluk.gz"

field_interpolator = intp.load_field_map(fieldmap_file)

