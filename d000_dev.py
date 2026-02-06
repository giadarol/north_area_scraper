import numpy as np
import matplotlib.pyplot as plt
from interpolator import load_field_map, plot_field_map, plot_cut_y, plot_field_lines

plt.close('all')

if __name__ == "__main__":
    # Example: plot the field magnitude and show on screen.
    field = load_field_map("cm70.fluk.gz", scale_xy=0.01)  # convert cm to m
    fig, ax = plot_field_map(field, component="Bmag")
    fig_cut, ax_cut = plot_cut_y(field, y_value_cm=3.0e-2)  # 3 cm
    fig_stream, ax_stream = plot_field_lines(field)
    plt.show()  # shows both if interactive backend is available
