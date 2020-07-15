import triangle as tr
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
from matplotlib import animation
from line_profiler import LineProfiler
import math
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm


class Cell:
    def __init__(self):
        self.v_id = []
        self.vs = []
        self.Del_n = []
        self.x = []

    def sort_vertices(self):
        angles = np.arctan2(self.vs[:, 1] - self.x[1], self.vs[:, 0] - self.x[0])
        self.vs = np.array([self.vs])[np.argsort(angles)]
        self.v_id = np.array([self.v_id])[np.argsort(angles)]
        self.Del_n = np.array([self.Del_n])[np.argsort(angles)]

    def get_P(self):
        self.P = np.sum(np.sqrt(((self.vs - np.roll(self.vs,1,axis=0))**2).sum()))

    def get_A(self):
        v_roll = np.roll(self.vs,1,axis=0)
        self.A = 0.5*np.sum(np.absolute(self.vs[:,0]*v_roll[:,1] - self.vs[:,1]*v_roll[:,0]))

class Tissue:
        def __init__(self):
            self.n_c = []
            self.n_v = []
            self.cells = []
            self.x0 = []
            self.vs = []
            self.tris = []

            self.A0 = 0.86
            self.P0 = 3.4641
            self.kappa_A = 1
            self.kappa_P = 1

            self.J = []
            self.c_types = []

        def generate_cells(self,n_c):
            """
            Generate the cells.

            Saves **Cell** objects, which will be utilised later in the simulation.
            :param n_c: number of cells (np.int32)
            :return: self.cells, a list of **Cell** objects
            """
            self.n_c = n_c
            self.cells = [Cell() for i in range(n_c)]
            return self.cells

        def hexagonal_lattice(self,rows=3, cols=3, noise=0.0005):
            """
            Assemble a hexagonal lattice

            :param rows: Number of rows in lattice
            :param cols: Number of columns in lattice
            :param noise: Noise added to cell locs (Gaussian SD)
            :return: points (nc x 2) cell coordinates.
            """
            points = []
            for row in range(rows * 2):
               for col in range(cols):
                   x = (col + (0.5 * (row % 2))) * np.sqrt(3)
                   y = row * 0.5
                   x += np.random.normal(0,noise)
                   y += np.random.normal(0,noise)
                   points.append((x, y))
            points = np.asarray(points)
            return points


        def make_init(self,L):
            """
            Make initial condition. Currently, this is a hexagonal lattice + noise

            Uses the stored self.noise as the noise input.

            Makes reference to the self.hexagonal_lattice function, then crops down to the reference frame

            Stores:
                self.n_c = number of cells
                self.x0 = (nc x 2) matrix denoting cell coordinates
                self.x = clone of self.x0

            :param L: Domain size/length (np.float32)
            """
            self.L = L
            self.x0 = self.hexagonal_lattice(int(np.sqrt(3*self.n_c)),int(np.sqrt(3*self.n_c)))
            self.x0 = self.x0[self.x0.max(axis=1) < L*0.95]
            self.x = self.x0
            self.n_c = self.x0.shape[0]

        def set_interaction(self,W = 0.16*np.array([[2, 0.5], [0.5, 2]]),pE = 0.5):
            nE = int(self.n_c*pE)
            N_dict = {"E": nE, "T": self.n_c - nE}

            c_types = np.zeros(self.n_c, dtype=np.int32)
            j = 0
            for k, c_type in enumerate(N_dict):
                j1 = N_dict[c_type]
                c_types[j:j + j1] = k
                j += j1
            np.random.shuffle(c_types)

            cell_i, cell_j = np.meshgrid(c_types, c_types, indexing="ij")
            J = W[cell_i, cell_j]
            self.J = J
            self.c_types = c_types


        def get_vertex_periodic(self,centroids, tri):
            """
            Get vertex locations, given cell centroid positions and triangulation. I.e. calculate the circumcentres of
            each triangle

            :param centroids: Cell coordinates (nc x 2)
            :param tri: Triangulation (nv x 3)
            :return V: Vertex coordinates (nv x 2)
            """
            C = centroids[tri]
            V = circumcenter_periodic(C,self.L)
            return V

        def assign_vertices(self):
            """
            Generate the CV_matrix, an (nc x nv x 3) array, considering the relationship between cells and vertices/triangulation.
            Essentially an array expression of the triangulation

            Uses the stored self.tris, the (nv x 3) array denoting the triangulation.

            :return self.CV_matrix: array representation of the triangulation (nc x nv x 3)
            """
            CV_matrix = np.zeros((self.n_c, self.n_v, 3))
            for i in range(3):
               CV_matrix[self.tris[:, i], np.arange(self.n_v), i] = 1
            self.CV_matrix = CV_matrix
            return self.CV_matrix

        def set_t_span(self,dt,tfin):
            """
            Set the temporal running parameters

            :param dt: Time-step (np.float32)
            :param tfin: Final time-step (np.float32)
            :return self.t_span: Vector of times considered in the simulation (nt x 1)
            """
            self.dt, self.tfin = dt,tfin
            self.t_span = np.arange(0,tfin,dt)
            return self.t_span

        def check_forces(self,x,F):
            """
            Plot the forces (quiver) on each cell (voronoi)

            To be used as a quick check.

            :param x: Cell coordinates (nc x 2)
            :param F: Forces on each cell (nc x 2)
            """
            Vor = Voronoi(x)
            fig, ax = plt.subplots()
            ax.set(aspect=1)
            voronoi_plot_2d(Vor, ax=ax)
            ax.scatter(x[:, 0], x[:, 1])
            ax.quiver(x[:, 0], x[:, 1], F[:, 0], F[:, 1])
            fig.show()

        def voronoi_finite_polygons_2d(self, vor, radius=None):
            """
            Reconstruct infinite voronoi regions in a 2D diagram to finite
            regions.

            Parameters
            ----------
            vor : Voronoi
                Input diagram
            radius : float, optional
                Distance to 'points at infinity'.

            Returns
            -------
            regions : list of tuples
                Indices of vertices in each revised Voronoi regions.
            vertices : list of tuples
                Coordinates for revised Voronoi vertices. Same as coordinates
                of input vertices, with 'points at infinity' appended to the
                end.

            """

            if vor.points.shape[1] != 2:
                raise ValueError("Requires 2D input")

            new_regions = []
            new_vertices = vor.vertices.tolist()

            center = vor.points.mean(axis=0)
            if radius is None:
                radius = vor.points.ptp().max()

            # Construct a map containing all ridges for a given point
            all_ridges = {}
            for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
                all_ridges.setdefault(p1, []).append((p2, v1, v2))
                all_ridges.setdefault(p2, []).append((p1, v1, v2))

            # Reconstruct infinite regions
            for p1, region in enumerate(vor.point_region):
                vertices = vor.regions[region]

                if all(v >= 0 for v in vertices):
                    # finite region
                    new_regions.append(vertices)
                    continue

                # reconstruct a non-finite region
                ridges = all_ridges[p1]
                new_region = [v for v in vertices if v >= 0]

                for p2, v1, v2 in ridges:
                    if v2 < 0:
                        v1, v2 = v2, v1
                    if v1 >= 0:
                        # finite ridge: already in the region
                        continue

                    # Compute the missing endpoint of an infinite ridge

                    t = vor.points[p2] - vor.points[p1]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = vor.vertices[v2] + direction * radius

                    new_region.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())

                # sort region counterclockwise
                vs = np.asarray([new_vertices[v] for v in new_region])
                c = vs.mean(axis=0)
                angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
                new_region = np.array(new_region)[np.argsort(angles)]

                # finish
                new_regions.append(new_region.tolist())

            return new_regions, np.asarray(new_vertices)


        def plot_vor(self,x,ax):
            """
            Plot the Voronoi.

            Takes in a set of cell locs (x), tiles these 9-fold, plots the full voronoi, then crops to the field-of-view

            :param x: Cell locations (nc x 2)
            :param ax: matplotlib axis
            """

            L = self.L
            grid_x, grid_y = np.mgrid[-1:2, -1:2]
            grid_x[0, 0], grid_x[1, 1] = grid_x[1, 1], grid_x[0, 0]
            grid_y[0, 0], grid_y[1, 1] = grid_y[1, 1], grid_y[0, 0]
            y = np.vstack([x + np.array([i * L, j * L]) for i, j in np.array([grid_x.ravel(), grid_y.ravel()]).T])

            c_types_print = np.tile(self.c_types,9)
            bleed = 0.1
            c_types_print = c_types_print[(y<L*(1+bleed)).all(axis=1)+(y>-L*bleed).all(axis=1)]
            y = y[(y<L*(1+bleed)).all(axis=1)+(y>-L*bleed).all(axis=1)]
            # Vor = Voronoi(y)
            # voronoi_plot_2d(Vor, ax=ax)
            regions, vertices = self.voronoi_finite_polygons_2d(Voronoi(y))


            ax.set(aspect=1,xlim=(0,self.L),ylim=(0,self.L))
            if type(self.c_types) is list:
                # ax.scatter(x[:, 0], x[:, 1],color="grey",zorder=1000)
                for region in regions:
                    polygon = vertices[region]
                    plt.fill(*zip(*polygon), alpha=0.4, color="grey")

            else:
                cols = "red","blue"
                for j,i in enumerate(np.unique(self.c_types)):
                    ax.scatter(x[self.c_types==i, 0], x[self.c_types==i, 1],color=cols[i],zorder=1000)
                patches = []
                for i, region in enumerate(regions):
                    patches.append(Polygon(vertices[region], True,facecolor=cols[c_types_print[i]],edgecolor="white",alpha=0.5))

                p = PatchCollection(patches, match_original=True)
                # p.set_array(c_types_print)
                ax.add_collection(p)


        def remove_repeats(self,tri):
            """
            For a given triangulation (nv x 3), remove repeated entries (i.e. rows)

            The triangulation is first re-ordered, such that the first cell id referenced is the smallest. Achieved via
            the function order_tris. (This preserves the internal order -- i.e. CCW)

            Then remove repeated rows via lexsort.

            NB: order of vertices changes via the conventions of lexsort

            Inspired by...
            https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array

            :param tri: (nv x 3) matrix, the triangulation
            :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
            """
            tri = order_tris(tri)
            sorted_tri = tri[np.lexsort(tri.T), :]
            row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))
            return sorted_tri[row_mask]




        def triangulate_periodic(self,x):
            """
            Calculates the periodic triangulation on the set of points x.

            Stores:
                self.n_v = number of vertices (int32)
                self.tris = triangulation of the vertices (nv x 3) matrix.
                    Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
                    (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
                self.vs = coordinates of each vertex; (nv x 2) matrix
                self.v_neighbours = vertex ids (i.e. rows of self.vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
                    In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of self.tris
                self.neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix

            :param x: (nc x 2) matrix with the coordinates of each cell
            """

            #1. Tile cell positions 9-fold to perform the periodic triangulation
            #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
            #   and the rest are translations
            L = self.L #Domain size (square)
            grid_x,grid_y = np.mgrid[-1:2,-1:2]
            grid_x[0,0],grid_x[1,1] = grid_x[1,1],grid_x[0,0]
            grid_y[0,0],grid_y[1,1] = grid_y[1,1],grid_y[0,0]
            y = np.vstack([x + np.array([i*L,j*L]) for i,j in np.array([grid_x.ravel(),grid_y.ravel()]).T])

            # #1b. Reduce excess grid for efficiency.
            # self.bleed = 0.5
            # y = y[(y<L*(1+self.bleed)).all(axis=1)+(y>-L*self.bleed).all(axis=1)]


            #2. Perform the triangulation on y
            #   The **triangle** package (tr) returns a dictionary, containing the triangulation.
            #   This triangulation is extracted and saved as tri
            t = tr.triangulate({"vertices": y})
            tri = t["triangles"]
            n_c = x.shape[0]

            #3. Generate a "normal cell" mask for y, i.e. cells that are considered in x
            normal_cells = np.zeros(n_c*9)
            normal_cells[:n_c] = 1

            #4. Find triangles with **at least one** cell within the "true" frame (i.e. with **at least one** "normal cell")
            #   (Ignore entries with -1, a quirk of the **triangle** package, which denotes boundary triangles
            #   Generate a mask -- one_in -- that considers such triangles
            #   Save the new triangulation by applying the mask -- new_tri
            tri = tri[(tri != -1).all(axis=1)]
            one_in = (tri<n_c).any(axis=1)
            new_tri = tri[one_in]

            #5. Remove repeats in new_tri
            #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
            #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
            #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details
            n_tri = self.remove_repeats(np.mod(new_tri,n_c))

            #6. Calculate vertex positions, the circumcentre of the three cells. See function doc-string for details
            V = self.get_vertex_periodic(x,n_tri)

            #7. Store outputs
            self.n_v = n_tri.shape[0]
            self.vs = V
            self.tris = n_tri

            #8. Manually calculate the neighbours. See doc_string for conventions.
            n_neigh = get_neighbours(n_tri)
            self.v_neighbours = n_neigh
            self.neighbours = V[n_neigh]


        def tri_angles(self,x,tri):
            return tri_angles(x, tri,self.L)
            #
            # L = self.L
            # three = np.arange(3).astype(int)
            # i_b = np.mod(three+1,3)
            # i_c = np.mod(three+2,3)
            #
            # C = x[tri]
            # a2 = (np.mod(C[:,i_b,0]-C[:,i_c,0]+L/2,L)-L/2)**2 + (np.mod(C[:,i_b,1]-C[:,i_c,1]+L/2,L)-L/2)**2
            # b2 = (np.mod(C[:,:,0]-C[:,i_c,0]+L/2,L)-L/2)**2 + (np.mod(C[:,:,1]-C[:,i_c,1]+L/2,L)-L/2)**2
            # c2 = (np.mod(C[:,i_b,0]-C[:,:,0]+L/2,L)-L/2)**2 + (np.mod(C[:,i_b,1]-C[:,:,1]+L/2,L)-L/2)**2
            #
            # cos_Angles = (b2+c2-a2)/(2*np.sqrt(b2)*np.sqrt(c2))
            # Angles = np.arccos(cos_Angles)
            # return Angles

        def equiangulate(self,x,tri,v_neighbours):
            return equiangulate(x, tri, v_neighbours, self.L)
            # three = np.arange(3).astype(np.int32)
            # nv = tri.shape[0]
            # no_flips = False
            # Angles = self.tri_angles(x, tri)
            # while no_flips is False:
            #     flipped = 0
            #     for i in range(nv):
            #         for k in range(3):
            #             neighbour = v_neighbours[i,k]
            #             if neighbour > i:
            #                 k2 = np.inner(v_neighbours[neighbour]==i,three)
            #                 theta = Angles[i,k] + Angles[neighbour,k2]
            #                 if theta > np.pi:
            #                     flipped+=1
            #                     tri[i,np.mod(k+2,3)] = tri[neighbour,k2]
            #                     tri[neighbour,np.mod(k2+2,3)] = tri[i,k]
            #                     v_neighbours = get_neighbours_j(tri, v_neighbours, [i,neighbour])
            #                     Angles[[i,neighbour]] = self.tri_angles(x,tri[[i,neighbour]])
            #     no_flips = flipped==0

        def get_P_periodic(self,neighbours, vs):
            """
            Calculates perimeter of each cell (considering periodic boundary conditions)

            :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
            :param vs: (nv x 2) matrix considering coordinates of each vertex
            :return: self.P saves the areas of each cell
            """

            P_m = np.empty((neighbours.shape[0], neighbours.shape[1]))
            for i in range(3):
                Neighbours = np.remainder(neighbours[:,i] - vs + self.L/2,self.L) - self.L/2
                P_m[:, i] = np.sqrt((Neighbours[:,0]) ** 2 + (Neighbours[:,1]) ** 2) #* self.boundary_mask

            PP_mat = np.zeros(P_m.shape)
            for i in range(3):
               PP_mat[:, i] = (P_m[:, np.mod(i + 1, 3)] + P_m[:, np.mod(i + 2, 3)]) / 2

            self.P = np.zeros((self.n_c))
            for i in range(3):
               self.P += np.dot(self.CV_matrix[:, :, i], PP_mat[:, i])

            return self.P

        def get_A_periodic(self,neighbours, vs):
            """
            Calculates area of each cell (considering periodic boundary conditions)

            :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
            :param vs: (nv x 2) matrix considering coordinates of each vertex
            :return: self.A saves the areas of each cell
            """
            AA_mat = np.empty((neighbours.shape[0], neighbours.shape[1]))
            Cents = np.array([(self.CV_matrix.T * self.x[:, 0]).sum(axis=-1),(self.CV_matrix.T * self.x[:, 1]).sum(axis=-1)]).T
            for i in range(3):
                Neighbours = np.remainder(neighbours[:, np.mod(i+2,3)] - Cents[:,i] + self.L / 2, self.L)
                Vs = np.remainder(vs - Cents[:,i] + self.L / 2, self.L)
                AA_mat[:, i] = 0.5 * (Neighbours[:, 0] * Vs[:, 1] - Neighbours[:, 1] * Vs[:, 0])

            self.A = np.zeros((self.n_c))
            for i in range(3):
                self.A += np.dot(self.CV_matrix[:,:,i],AA_mat[:,i])
            return self.A

        def get_F_periodic(self,neighbours,vs):
            """
            Calculate the forces acting on each cell via the SPV formalism.

            Detailed explanations of each stage are described in line, but overall the strategy leverages the chain-rule
            (vertex-wise) decomposition of the expression for forces acting on each cell. Using this, contributions from
            each vertex is calculated **on the triangulation**, without the need for explicitly calculating the voronoi
            polygons. Hugely improves efficiency

            :param neighbours:
            :param vs:
            :return:
            """
            ### Given orientation is preserved in the triangulation, can perform entire calculation in parallel.
            ### Modifications are implemented to account for periodic boundary conditions (see np.remainder / np.mod)


            ## 1. For each vertex h, define orientation of neighbouring vertices.
            # For a given cell j, the neighbouring vertex oriented CCW with respect to the vertex h is denoted h_CCW
            # Likewise for h_CW
            h = np.empty((self.n_v,3,2))
            for i in range(3):
                h[:,i] = vs
            h_CCW = np.roll(neighbours, 1, axis=1)
            h_CW = np.roll(neighbours, -1, axis=1)
            # h_CCW = np.roll(neighbours, 0, axis=1)
            # h_CW = np.roll(neighbours, 2, axis=1)


            ## 2. Calculate the area and perimeter vector terms within eq. A8,
            # zet1 = (h3y - h7y,h3x-h7x) {line 2}
            # zet2 = (h2x-h7x/ ...) {line 3}
            zet1 = (np.flip(np.remainder(h_CCW - h_CW + self.L/2,self.L) - self.L/2,axis=2))#*np.array([1,-1])) #unsure about this line -- seems to be non-buggy with and without the additional multiplication. Multiplication is invoking the cross product w e_z
            zet2a = np.remainder(h - h_CW + self.L/2,self.L) - self.L/2
            zet2b = np.remainder(h - h_CCW + self.L/2,self.L) - self.L/2
            zet2a_norm, zet2b_norm = np.sqrt((zet2a**2).sum(axis=2)),np.sqrt((zet2b**2).sum(axis=2))
            zet2 = np.empty(zet1.shape)
            for i in range(2):
                zet2[:,:,i] = zet2a[:,:,i]/zet2a_norm + zet2b[:,:,i]/zet2b_norm
            zet2 = zet2

            J_CCW = self.J[self.tris, np.roll(self.tris, 1, axis=-1)]
            J_CW = self.J[self.tris, np.roll(self.tris, -1, axis=-1)]

            zet3 = np.empty(zet1.shape)
            for i in range(2):
                zet3[:,:,i] = J_CW*zet2a[:,:,i]/zet2a_norm + J_CCW*zet2b[:,:,i]/zet2b_norm

            ## 3. Find areas and perimeters of the cells and restructure data wrt. the triangulation
            vA = self.A[self.tris]
            vP = self.P[self.tris]

            #4. Calculate ∂h/∂r. This is for cell i (which may or may not be cell j, and triangulates with cell j)
            # This last two dims are a Jacobinan (2x2) matrix, defining {x,y} for h and r. See function description for details
            DHDR = dhdr_periodic(self.x[self.tris],vs,self.L)#order is wrt cell i


            #5. Now calculate the force component for each vertex, with respect to the 3 neighbouring cells
            #   This is essentially decomposing the chain rule of the expression of F for each cell by vertex
            #   M_sum is a (nv,2,3) matrix. This is the force contribution for each cell of a given triangle/vertex (3rd dim). Considering {x,y} components (2nd dim)
            #       Within the function, this calculates (direct and indirect) contributions of each cell wrt each other cell (i.e. 3x3), then adds them together
            M_sum = make_M(DHDR,self.n_v,self.kappa_A,self.kappa_P,vA,vP,self.A0,self.P0,zet1,zet2,zet3)
            self.M = M_sum

            #6. Compile force components wrt. cells by using the cell-to-vertex connection matrix.
            #       Force on cell_i = SUM_{vertices of cell i} {forces at each vertex wrt. cell i}
            F = compile_cell_forces(self.CV_matrix,M_sum,self.n_c)
            return F

        def simulate_periodic(self):
            n_t = self.t_span.size
            self.n_t = n_t
            x = self.x0.copy()
            self.x = x.copy()
            self.x_save = np.zeros((n_t,self.n_c,2))
            self.tri_save = np.zeros((n_t,self.tris.shape[0],3),dtype=np.int32)
            for i in range(n_t):
                print(i)
                self.triangulate_periodic(x)
                self.tri_save[i] = self.tris
                self.assign_vertices()
                self.get_A_periodic(self.neighbours,self.vs)
                self.get_P_periodic(self.neighbours,self.vs)
                F = self.get_F_periodic(self.neighbours,self.vs)
                self.F = F
                x += self.dt*(F + np.random.normal(0,np.sqrt(self.eta/self.dt),(self.n_c,2)))
                x = np.mod(x,self.L)
                self.x = x
                self.x_save[i] = x
            return self.x_save

        def get_self_self(self):
            self.self_self = np.zeros(self.n_t)
            for i, tri in enumerate(self.tri_save):
                C_types = self.c_types[tri]
                C_self_self = (C_types-np.roll(C_types,1,axis=1))==0
                self.self_self[i] = np.sum(C_self_self)/C_self_self.size



        def animate(self,n_frames = 100,file_name=None, dir_name="plots"):
            """
            Animate the simulation, saving to an mp4 file.

            Parameters
            ----------
            n_frames : int
                Number of frames to animate. Spaced evenly throughout **x_save**

            file_name : str
                Name of the file. If **None** given, generates file-name based on the time of simulation

            dir_name: str
                Directory name to save the plot.


            """
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)

            skip = int((self.x_save.shape[0])/n_frames)
            def animate(i):
                ax1.cla()
                self.plot_vor(self.x_save[skip*i],ax1)
                ax1.set(aspect=1, xlim=(0, self.L), ylim=(0, self.L))

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, bitrate=1800)
            if file_name is None:
                file_name = "animation %d" % time.time()
            an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
            an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)

@jit(nopython=True,cache=True)
def dhdr_periodic(rijk_,vs,L):
    """Calculates ∂hi/dr where a,b,c are the coords of ri,rj,rk (cell centroids in triangulation)

    Last two dims: ((dhx/drx,dhx/dry),(dhy/drx,dhy/dry))

    These are lifted from Mathematica"""
    rijk = np.empty_like(rijk_)
    for i in range(3):
        rijk[:,i,:] = np.remainder(rijk_[:,i] - vs + L/2,L) - L/2

    DHDR = np.empty(rijk.shape + (2,))
    for i in range(3):
        ax,ay = rijk[:,np.mod(i,3),0],rijk[:,np.mod(i,3),1]
        bx, by = rijk[:, np.mod(i+1,3), 0], rijk[:, np.mod(i+1,3), 1]
        cx, cy = rijk[:, np.mod(i+2,3), 0], rijk[:, np.mod(i+2,3), 1]
        #dhx/drx
        DHDR[:, i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/drx
        DHDR[:, i, 1,0] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                    2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhx/dry
        DHDR[:, i, 0, 1] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/dry
        DHDR[:, i, 1,1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)


    return DHDR
#
# @jit(nopython=True,cache=True)
# def get_neighbours(tri):
#     neigh = np.zeros_like(tri,dtype=np.int32)
#     for i in range(3):
#         JK = tri[:, np.mod(np.arange(1, 3) + i, 3)]
#         for j, jk in enumerate(JK):
#             mask1,mask2 = np.zeros(tri.shape[0],dtype=np.bool_),np.zeros(tri.shape[0],dtype=np.bool_)
#             for k in range(3):
#                 mask1+=(tri[:,k] == jk[0])
#                 mask2+=(tri[:,k] == jk[1])
#             Neigh = np.nonzero(mask1*mask2)[0]
#             neigh[j, i] = Neigh[Neigh != j][0]
#     return neigh

@jit(nopython=True)
def get_neighbours(tri,neigh=None):
    """Update: more efficient algorithm to prevent double-counting

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)
    """
    if neigh is None:
        neigh = np.ones_like(tri,dtype=np.int32)*-1
    n_v = tri.shape[0]
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip,tri_sample_flip)).reshape(3,2)
        for k in range(3):
            if neigh[j,k]==-1:
                neighb,l = np.nonzero((tri_compare[:,:,0]==tri_i[k,0])*(tri_compare[:,:,1]==tri_i[k,1]))
                neighb,l = neighb[0],l[0]
                neigh[j,k] = neighb
                neigh[neighb,np.mod(2-l,3)] = j
    return neigh

@jit(nopython=True,cache=True)
def get_neighbours_j(tri,neigh,js):
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in js:
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            neighb, l = np.nonzero((tri_compare[:, :, 0] == tri_i[k, 0]) * (tri_compare[:, :, 1] == tri_i[k, 1]))
            neighb, l = neighb[0], l[0]
            neigh[j, k] = neighb
            neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh

@jit(nopython=True,cache=True)
def order_tris(tri):
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i,Min],tri[i,np.mod(Min+1,3)],tri[i,np.mod(Min+2,3)]
    return tri


@jit(nopython=True,cache=True)
def circumcenter(C):
    ri, rj, rk = C.transpose(1,2,0)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size,2),dtype=np.float64)
    vs[:,0],vs[:,1] = ux,uy
    return vs



@jit(nopython=True,cache=True)
def circumcenter_periodic(C,L):
    ri, rj, rk = C.transpose(1,2,0)
    r_mean = (ri+rj+rk)/3
    disp = r_mean - L / 2
    ri,rj,rk = np.mod(ri-disp,L),np.mod(rj-disp,L),np.mod(rk-disp,L)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size,2),dtype=np.float64)
    vs[:,0],vs[:,1] = ux,uy
    vs = np.mod(vs+disp.T,L)
    return vs


@jit(nopython=True,cache=True)
def make_M(DHDR,n_v,kappa_A,kappa_P,vA,vP,A0,P0,zet1,zet2,zet3):
    M = np.zeros((2, n_v, 2, 3))
    for i in range(3):
        for j in range(3):
            for Fdim in range(2):
                M[:, :, Fdim, i] += DHDR[:, i, Fdim].T * (kappa_A * (vA[:, j] - A0) * zet1[:, j].T +
                                                            2 * kappa_P * (vP[:, j]-P0) * zet2[:, j].T + zet3[:,j].T)
    M_out = M[0] + M[1]
    return M_out


@jit(nopython=True,cache=True)
def compile_cell_forces(CV_matrix,M_sum,n_c):
    dEdr = np.zeros((n_c, 2))
    for i in range(3):
        dEdr += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(M_sum[:, :, i])
    F = -dEdr
    return F

@jit(nopython=True,cache=True)
def np_apply_along_axis(func1d, axis, arr):
    """https://github.com/numba/numba/issues/1269"""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1],dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0],dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@jit(nopython=True,cache=True)
def tri_angles(x, tri,L):
    three = np.array([0,1,2])
    i_b = np.mod(three + 1, 3)
    i_c = np.mod(three + 2, 3)

    C = np.empty((x.shape[0],3,2))
    for i, TRI in enumerate(tri):
        C[i] = x[TRI]
    a2 = (np.mod(C[:, i_b, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
                np.mod(C[:, i_b, 1] - C[:, i_c, 1] + L / 2, L) - L / 2) ** 2
    b2 = (np.mod(C[:, :, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
                np.mod(C[:, :, 1] - C[:, i_c, 1] + L / 2, L) - L / 2) ** 2
    c2 = (np.mod(C[:, i_b, 0] - C[:, :, 0] + L / 2, L) - L / 2) ** 2 + (
                np.mod(C[:, i_b, 1] - C[:, :, 1] + L / 2, L) - L / 2) ** 2

    cos_Angles = (b2 + c2 - a2) / (2 * np.sqrt(b2) * np.sqrt(c2))
    Angles = np.arccos(cos_Angles)
    return Angles


@jit(nopython=True,cache=True)
def equiangulate(x, tri, v_neighbours,L):
    three = np.array([0,1,2])
    nv = tri.shape[0]
    no_flips = False
    Angles = tri_angles(x, tri,L)
    while no_flips is False:
        flipped = 0
        for i in range(nv):
            for k in range(3):
                neighbour = v_neighbours[i, k]
                if neighbour > i:
                    k2 = ((v_neighbours[neighbour] == i)* three).sum()
                    theta = Angles[i, k] + Angles[neighbour, k2]
                    if theta > np.pi:
                        flipped += 1
                        tri[i, np.mod(k + 2, 3)] = tri[neighbour, k2]
                        tri[neighbour, np.mod(k2 + 2, 3)] = tri[i, k]
                        v_neighbours = get_neighbours_j(tri, v_neighbours, [i, neighbour])
                        new_tris = np.empty((2,3),dtype=np.int32)
                        new_tris[0],new_tris[1] = tri[i],tri[neighbour]
                        new_angles = tri_angles(x, new_tris,L)
                        Angles[i] = new_angles[0]
                        Angles[neighbour] = new_angles[1]
        no_flips = flipped == 0
    return tri,v_neighbours


