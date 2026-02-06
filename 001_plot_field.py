import numpy as np
import matplotlib.pyplot as plt
from interpolator import load_field_map, plot_field_map, plot_cut_y, plot_field_lines

plt.close('all')

field = load_field_map("cm70.fluk.gz", scale_xy=0.01)  # convert cm to m

plot_field_map(field, component="Bmag")
plot_cut_y(field, y_value_cm=3.0e-2)  # 3 cm
plot_field_lines(field)

plt.show()
