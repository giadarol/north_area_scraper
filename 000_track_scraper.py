import interpolator as intp
import numpy as np

import xtrack as xt

# TODO:
# - remove log and check speed


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
field_interpolator_2d = intp.load_field_map(fieldmap_file, scale_xy=0.01)  # cm to m

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

# Check particles in the horizontal mid plane (y=0)
p_x = line.build_particles(x=np.linspace(-0.3, 0.3, 21))
line.track(p_x, multi_element_monitor_at='_all_')
s_x = line.record_multi_element_last_track.get('s', turn=0)
x_x = line.record_multi_element_last_track.get('x', turn=0)

# Check particles in the vertical mid plane (x=0)
p_y = line.build_particles(y=np.linspace(-0.3, 0.3, 21))
line.track(p_y, multi_element_monitor_at='_all_')
s_y = line.record_multi_element_last_track.get('s', turn=0)
y_y = line.record_multi_element_last_track.get('y', turn=0)

# Time tracking for many particles
n_time = 10000
p_time = line.build_particles(x=np.linspace(-0.3, 0.3, 101))
line.track(p_time, time=True)
print(f"Tracking time for {n_time} particles: {line.time_last_track:.3f} s")

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(s_x.T, x_x.T, '-')
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.suptitle('Horizontal mid plane (y=0)')

plt.figure(2)
plt.plot(s_y.T, y_y.T, '-')
plt.xlabel('s [m]')
plt.ylabel('y [m]')
plt.suptitle('Vertical mid plane (x=0)')

# Field profile on the horizontal mid plane (y=0)
x_grid = np.linspace(-0.3, 0.3, 1001)
bx, by, bz = field_3d(x_grid, 0*x_grid, 0*x_grid)
plt.figure(100)
plt.plot(x_grid, by, label='Bx')
plt.xlabel('x [m]')
plt.ylabel('Field [T]')
plt.title('Field profile on the horizontal mid plane (y=0)')

y_grid = np.linspace(-0.3, 0.3, 1001)
bx, by, bz = field_3d(0*y_grid, y_grid, 0*y_grid)
plt.figure(101)
plt.plot(y_grid, bx, label='Bx')
plt.xlabel('y [m]')
plt.ylabel('Field [T]')
plt.title('Field profile on the vertical mid plane (x=0)')

plt.show()

