from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt


vor = Tissue()
vor.generate_cells(100)
vor.make_init(7)
alpha = 0.08
vor.set_interaction(W = alpha*np.array([[0, 1], [1, 0]]),pE=0.5)

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




vor.set_t_span(0.025,200)
vor.simulate()


vor.animate(n_frames=50)


vor.get_self_self()
fig, ax = plt.subplots()
ax.plot(vor.self_self)
ax.set(xlabel="Time",ylabel="Fraction of self-self interactions")
fig.savefig("self_self.pdf")


ratio = 0.5
P0_eff = alpha*ratio/vor.kappa_P + vor.P0
p0_eff = P0_eff/np.sqrt(vor.A0)
print(p0_eff)

"""
Stat to measure q for each cell. 
And compare with neighbourhood and thus p0_eff
And MSD (short time-scale)
"""