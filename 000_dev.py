import xtrack as xt
import numpy as np

x, y, Bx, By = np.loadtxt("cm70.fluk.gz", unpack=True, delimiter=',')

# units are cm, fields in Tesla