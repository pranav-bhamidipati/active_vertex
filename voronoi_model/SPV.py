from voronoi_model_periodic import *
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,voronoi_plot_2d,Delaunay

vor = Tissue()
vor.noise = 1e-2
vor.generate_cells(100)
vor.make_init(9)
shift = vor.L/2

vor.x0 = np.remainder(vor.x0+shift,vor.L)
vor.A0 = 0.9
print(vor.P0/np.sqrt(vor.A0))
vor.eta = 0#0.01
vor.kappa_A = 0.1
vor.kappa_P = 0.1

import time

vor.triangulate_periodic(vor.x0)


t0 = time.time()
for i in range(int(1e4)):
    vor.triangulate_periodic(vor.x0)
t1 = time.time()
print("1e4 iterations in",t1-t0,"s")

# vor.set_t_span(0.05,80)
# # vor.run_simulation_profile()
# vor.simulate_periodic()
# vor.check_forces(vor.x,vor.F)
# vor.x_save = np.mod(vor.x_save + vor.L/2,vor.L)
# vor.check_forces(np.mod(vor.x+vor.L/2,vor.L),vor.F)
# print(vor.M.shape)
# vor.animate(n_frames=50)
#
# %timeit