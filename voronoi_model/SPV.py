from voronoi_model.voronoi_model_periodic import *
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

# import time
#
vor.triangulate_periodic(vor.x0)

# vor.x0 += np.random.normal(0,0.2,vor.x0.shape)
# tri,v_neighbours = vor.equiangulate(vor.x0,vor.tris,vor.v_neighbours)
# TRI = tri.copy()
# vor.triangulate_periodic(vor.x0)
#
self = vor
W = np.array([[2.0,1.0],[1.0,2.0]])
nE = 45
N_dict = {"E":nE,"T":self.n_c-nE}

c_types = np.zeros(self.n_c,dtype=np.int32)
j = 0
for k,c_type in enumerate(N_dict):
    j1 = N_dict[c_type]
    c_types[j:j+j1] = k
    j += j1

cell_i,cell_j = np.meshgrid(c_types,c_types,indexing="ij")
J = W[cell_i,cell_j]

tri = self.tris
J_CCW = J[tri,np.roll(tri,1,axis=-1)]
J_CW = J[tri,np.roll(tri,-1,axis=-1)]


#
# k = 0
# for Tri in TRI:
#     j = 0
#     for TRi in vor.tris:
#         if np.all(np.unique(Tri) - np.unique(TRi)==0):
#             j +=1
#     if j == 0:
#         print("NO",k)
#     k+=1
# #
# #
# # t0 = time.time()
# # for i in range(int(1e4)):
# #     vor.triangulate_periodic(vor.x0)
# # t1 = time.time()
# # print("1e4 iterations in",t1-t0,"s")

vor.set_t_span(0.05,20)
# vor.run_simulation_profile()
vor.simulate_periodic()
vor.check_forces(vor.x,vor.F)
vor.x_save = np.mod(vor.x_save + vor.L/2,vor.L)
vor.check_forces(np.mod(vor.x+vor.L/2,vor.L),vor.F)
print(vor.M.shape)
vor.animate(n_frames=50)

# %timeit