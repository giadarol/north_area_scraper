import interpolator as intp
import numpy as np

import xtrack as xt

# Beamline in which we will insert the scraper
env = xt.Environment()
line = env.new_line(components=[
    env.new('m0', xt.Marker, at=0.),
    env.new('m1', xt.Marker, at=5.),
    env.new('m2', xt.Marker, at=10.),
])
# environment can be retrieved as line.env

# Load the field map
fieldmap_file = "cm70.fluk.gz"
field_interpolator_2d = intp.load_field_map(fieldmap_file)

# Build a callable compatible with Xsuite Boris integrator (needs 3 components)
def field_3d(x, y, s):
    """Wrapper to call the 3D field interpolator."""
    bx, by = field_interpolator_2d(x, y)
    bs = np.zeros_like(bx)  # Assuming Bz is zero
    return bx, by, bs

# Build scraper slices
length_scraper = 3.0  # m
num_slices = 20
num_steps_per_slice = 50
length_slice = length_scraper / num_slices
scraper_slices = []
for ii in range(num_slices):
    nn = f'scraper_slice_{ii}'
    env.elements[nn] = xt.BorisSpatialIntegrator(fieldmap_callable=field_3d,
                                        s_start=0,
                                        s_end=length_slice,
                                        n_steps=num_steps_per_slice)
    scraper_slices.append(nn)

# Make a line corresponding to the scraper alone
scraper = env.new_line(components=scraper_slices)

# Insert the scraper line between m1 and m2
line.insert(scraper, anchor='start', at=1.5)

line.get_table().show()
# prints:
# name                         s element_type           isthick ...
# m0                           0 Marker                   False
# ||drift_2                    0 Drift                     True
# scraper_slice_0            1.5 BorisSpatialIntegrator    True
# scraper_slice_1           1.65 BorisSpatialIntegrator    True
# scraper_slice_2            1.8 BorisSpatialIntegrator    True
# scraper_slice_3           1.95 BorisSpatialIntegrator    True
# ...

line.set_particle_ref('proton', energy0=10e9)

