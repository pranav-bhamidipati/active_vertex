from voronoi_model.voronoi_model_periodic import *
import numpy as np
import matplotlib.pyplot as plt

vor = Tissue()
vor.noise = 1e-2
vor.generate_cells(100)
vor.make_init(9)
vor.set_interaction(W = 0.16*np.array([[2, 0.5], [0.5, 2]]))

vor.A0 = 0.86
vor.P0 = 3.12
vor.eta = 1e-2
vor.kappa_A = 0.2
vor.kappa_P = 0.1

vor.set_t_span(0.01,80)
vor.simulate_periodic()

vor.get_self_self()
fig, ax = plt.subplots()
ax.plot(vor.self_self)
ax.set(xlabel="Time",ylabel="Fraction of self-self interactions")
fig.savefig("self_self.pdf")

vor.animate(n_frames=30)


