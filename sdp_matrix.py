import random as r
import numpy as np

# Creation of a SDP matrix
def matrice_SDP(n, nb_extra_diag):
    A = np.zeros([n, n])
    coord = []
    for i in range(0, n, 1):
        for j in range(0, i, 1):
            coord.append([i, j])
    for i in range(0, nb_extra_diag, 1):
        c = r.randint(0, len(coord)-1)
        val = r.randint(0,50)
        A[coord[c][0]][coord[c][1]] = val
        A[coord[c][1]][coord[c][0]] = val
        del coord[c]
    for i in range(0, n, 1):
        s = 0
        for j in range(0, n, 1):
            s = s + A[i][j]
        A[i][i] = r.randint(s + 1, 2 * s +1)
    return A

def random_vector(n):
    B = np.zeros((n,1))
    for i in range(0,n,1):
        B[i][0] = r.randint(0,50)
    return B
