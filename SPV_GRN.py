from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt


vor = Tissue()
vor.generate_cells(600)
vor.make_init(7)
vor.set_interaction(W = 0.08 * np.array([[0, 1], [1, 0]]),pE=0)

# vor.P0 = 3.00
p0 = 3.80 #3.81
vor.A0 = 0.86
vor.P0 = p0*np.sqrt(vor.A0)
print(vor.P0)

vor.v0 = 1e-2
vor.Dr = 0.01
vor.kappa_A = 0.2
vor.kappa_P = 0.1
vor.a = 0.3
vor.k = 2

vor.set_t_span(0.01,50)


##GRN params
vor.tau = 10 #this rescales the timescale of the grn wrt the dynamics of the cell movement (NOT your tau)
vor.alpha = 1
vor.p = 2
vor.K = 0.1
vor.delT = 0
vor.leak = 0
vor.Sender = np.zeros(vor.n_c,dtype=np.bool) #Sender is a boolean array saying whether cells are or aren't the always +ve cell
vor.Sender[int(vor.n_c/2)] = True
vor.sender_val = 1
vor.dT = vor.dt*40 #this is your tau, the time seperation delay.  #This needs to be > 0 given the way the simulation is set up. See self.simulate_GRN

vor.simulate_GRN()
print(vor.E_save[-1])
vor.animate_GRN(n_frames=50)
