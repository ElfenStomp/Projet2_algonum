import numpy as np
import scipy.linalg as spl  # for the function solve.
import matplotlib.pyplot as plt # in order to display the datas.
import sdp_matrix as sdp #generate sdp matrix.
import conjgrad as conj #function of the conjugate gradient method


def tst_conjgrad():
        # initialized first test matrix
    M = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])
    A = np.dot(M,np.transpose(M))
    print("A= \n", A, "\n Dimension de A: ", np.shape(A))
    B = np.array([[1.],[2.],[3.]])
    print("B = \n", B, "\n Dimension de B: ", np.shape(B))
    X = np.array([[0.],[0.],[0.]]) # initialized null 3-dimensioned vector
    imax = 10**3
    p = 10**(-100)
    print("\n Précision à ",p," près \n")
        # not accurate comparison between conjgrad and np.linalg.solve
    X1 = conj.conjgrad_precond(A, B, X, imax, p)
    X2 = spl.solve(A, B)
    print("\n Avec la méthode du gradient conjugué on a X = \n", X1)
    print("Avec la fonction solve de la biliothèque scipy.linalg on a X = \n", X2)

    input("Appuyer sur entrée pour continuer ...")

    print("\n Test avec une matrice aléatoire générée par la fonction matrice_SDP : \n")
    matrix_size = int(input("Entrer la taille de la matrice A: "))
    nbr_extra_diag = int(input("Ainsi que le nombre d'elements extra-diagonaux de A: "))
    curve_nbr = int(input("Veuiller preciser le nombre de courbes à tracer : "))

        # absolute convergence speed
    t = np.linspace(-5, 0, 50) # The x-axis gives the precision.
    B = sdp.random_vector(matrix_size)

    for c_index in range(curve_nbr):
        A = sdp.matrice_SDP(matrix_size, nbr_extra_diag)
        Xzero = np.zeros((matrix_size,1))
        X2 = spl.solve(A, B)

        print("\n A = \n", A)
        print("\n B = \n", B)
        print("\n Xzero = \n", Xzero)
        print("\n X_solve = \n", X2)
        tab = []
        for i in t:
            if (c_index < curve_nbr/2 ):
                X_diff = conj.conjgrad(A, B, Xzero, imax, 10**(-i)) - X2
            else:
                X_diff = conj.conjgrad_precond(A, B, Xzero, imax, 10**(-i)) - X2
            scalar_product = np.transpose(X_diff).dot(X_diff)[0][0]
            tab.append(scalar_product)
        plt.plot(t, tab, label="produit scalaire " + str(c_index + 1))
        if(c_index >= curve_nbr/2 ):
            print("scalar_product = ", scalar_product)
    input("Appuyer sur entrée pour continuer ...")

    plt.legend()
    plt.xlabel("précision 10^(-x)")
    plt.show()
    
tst_conjgrad()
