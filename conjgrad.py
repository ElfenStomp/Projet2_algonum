# Méthode du gradient conjugué

    # Bibliothèques nécessaires

import numpy as np
import math as m
import scipy.linalg as spl # pour la fonction solve, qui résoud une équation de type AX = B
import matplotlib.pyplot as plt # pour tracer la courbe de comparaisons entre la méthode du gradient conjugué
                                # et la fonction solve de la bibliothèque scipy.linalg
import sdp_matrix as sdp # pour générer des matrices symétriques définies positives à diagonale dominante

# 1
"""
 Syntaxe plus "propre" :
_ while ( i < imax && sqrt(rsold) > 10**(-10)) et on ajoute la ligne d'incrémentation i += 1 dans
la boucle. Avec imax, un nombre d'itérations maximal de l'orde de 10^3. De telle sorte qu'on peut enlever
le test if dans la boucle for et remplacer cette boucle for par une boucle while.
_ Le vecteur solution X que l'on recherche est un argument de la fonction, il peut donc être très
éloigné de la valeur à laquelle on veut aboutir. Cependant, cela ralentit-il vraiment la convergence de r vers 0
i.e de AX vers B ?
_ Le produit entre A et X est matriciel, il serait judicieux de le préciser par une fonction
intermédiaire, de même que r'*r est un produit scalaire entre 2 vecteurs.
_ r' est le vecteur transposé de r, cela n'est pourtant pas clair dans le programme.
_ C'est également le cas des opérations +/- sur des vecteurs/matrices, qui doivent être implémentées correctement
en Python.
_ La fonction ne retourne rien.
"""

# 2
"""
L'algorithme de décomposition de Choleski donne une formule pour calculer les coefficients de la matrice T, telle que
A = T * transposée(T) (on note M' la transposée de M). Le problème AX = B devient T*T'X = B.
Or inverse(T*T') = inverse(T') * inverse(T), toutes les 2 inversibles car sinon A ne le serait pas, donc
X = inverse(T') * inverse(T) * B.

La méthode du gradient conjugué fait tendre X vers le vecteur solution en faisant tendre
R = B - AX vers le vecteur nul. Pour ce qui est de la complexité, celle-ci est en imax*0(n²). En effet, on ne fait jamais
de produit matriciel entre 2 matrices carrées de taille n x n mais entre une matrice carrée de dimension n
et un vecteur de taille n. Avec un nombre imax d'itération constant dans la boucle.

Or Choleski est en O(n^3), on gagne donc un facteur n. Si on veut calculer le gain de complexité par rapport
à N = n², on a une complexité linéaire par rapport à N pour la méthode du gradient conjugué et en O(N^(3/2)) pour Choleski.

Pour n faible, on a n^3 < imax * n². Donc pour n < imax, la complexité de Choleski est meilleure que celle-ci
du gradient conjugué.

(à  voir avec ceux qui traitent la 1ère partie pour répondre à cette question)
"""

# 3
def is_symdefpos(M):
    # vérification de la symétrie de M
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            if (M[i][j] != M[j][i]):
                return False

    # vérification de la positivité des valeurs propres
    for l in spl.eig(M)[0]:
        if (type(l) == complex or l < 0) and abs(l) > 10**(-10):
            return False

    # vérification de l'inversibilité de A
    if spl.det(M) == 0:
        return False

    return True

def conjgrad(A,B,X,imax,p):
    if (is_symdefpos(A) == False):
        # On vérifie que A soit bien symétrique définie positive.
        print("\n A n'est pas symétrique définie positive")
        return np.zeros((np.shape(A)[0],1))
    R = B - A.dot(X)
    P = R
    rs_old = np.transpose(R).dot(R)
    for i in range(1, imax + 1):
        Ap = A.dot(P)
        alpha = rs_old / np.transpose(P).dot(Ap)
        X = X + (alpha * P)
        R = R - (alpha * Ap)
        rs_new = np.transpose(R).dot(R)
        if (m.sqrt(rs_new) < p):
            break
        P = R + (rs_new/rs_old) * P
        rs_old = rs_new
    print("\n R = \n", R)
    print("\n X = \n", X)
    return X

def conjgrad_precond(A,B,X,imax,p):
    if (is_symdefpos(A) == False):
        # On vérifie que A soit bien symétrique définie positive.
        print("\n A n'est pas symétrique définie positive")
        return np.zeros((np.shape(A)[0],1))
    R = B - A.dot(X)
    #M = matrix_inv_approx(A)
    M = spl.inv(A)
    Z = M.dot(R)
    P = Z
    rs_old = np.transpose(R).dot(Z)
    rs_old_beta = np.transpose(Z).dot(R)
    for i in range(1, imax + 1):
        Ap = A.dot(P)
        alpha = rs_old / np.transpose(P).dot(Ap)
        X = X + (alpha * P)
        R = R - (alpha * Ap)
        if (m.sqrt(np.transpose(R).dot(R)) < p):
            break
        Z = M.dot(R)
        rs_new = np.transpose(R).dot(Z)
        rs_new_beta = np.transpose(Z).dot(R)
        P = Z + (rs_new_beta/rs_old_beta) * P
        rs_old = rs_new
        rs_old_beta = rs_new_beta
    print("\n R = \n", R)
    print("\n X = \n", X)
    return X


def tst_conjgrad():
        # Initialisation des arguments de conjgrad
    M = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])
    A = np.dot(M,np.transpose(M))
    print("A= \n", A, "\n Dimension de A: ", np.shape(A))
    B = np.array([[1.],[2.],[3.]])
    print("B = \n", B, "\n Dimension de B: ", np.shape(B))
    X = np.array([[0.],[0.],[0.]]) # On initialise X au vecteur nul de R^3
    imax = 10**3
    p = 10**(-100)
    print("\n Précision à ",p," près \n")
        # Comparaison sommaire entre conjgrad et np.linalg.solve
    X1 = conjgrad_precond(A, B, X, imax, p)
    X2 = spl.solve(A, B)
    print("\n Avec la méthode du gradient conjugué on a X = \n", X1)
    print("Avec la fonction solve de la biliothèque scipy.linalg on a X = \n", X2)

    input("Appuyer sur entrée pour continuer ...")

    print("\n Test avec une matrice aléatoire générée par la fonction matrice_SDP : \n")
    matrix_size = int(input("Entrer la taille de la matrice A: "))
    nbr_extra_diag = int(input("Ainsi que le nombre d'elements extra-diagonaux de A: "))
    curve_nbr = int(input("Veuiller preciser le nombre de courbes à tracer : "))

        # Vitesse de convergence absolue
    t = np.linspace(-5, 0, 50) # L'axe des abscisses qui représente la précision
    B = sdp.random_vector(matrix_size)

    for c_index in range(curve_nbr):
        A = sdp.matrice_SDP(matrix_size, nbr_extra_diag)
        #B = sdp.random_vector(matrix_size)
        Xzero = np.zeros((matrix_size,1))
        X2 = spl.solve(A, B)

        print("\n A = \n", A)
        print("\n B = \n", B)
        print("\n Xzero = \n", Xzero)
        print("\n X_solve = \n", X2)
        tab = []
        for i in t:
            if (c_index < curve_nbr/2 ):
                X_diff = conjgrad(A, B, Xzero, imax, 10**(-i)) - X2
            else:
                X_diff = conjgrad_precond(A, B, Xzero, imax, 10**(-i)) - X2
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
