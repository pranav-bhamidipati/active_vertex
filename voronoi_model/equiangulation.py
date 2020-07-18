
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


@jit(nopython=True,cache=True)
def equiangulate2(x, tri, v_neighbours,L):
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
                        v_neighbours = get_neighbours(tri)
        no_flips = flipped == 0
    return tri,v_neighbours



@jit(nopython=True,cache=True)
def get_neighbours_j(tri,neigh,js):
    """
    Finds updates the neighbours file only for a subset of vertices, provided by the list "js"
    ~~beta~~
    """
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