# conjugate gradient method

    # library needed

import numpy as np
import math as m
import scipy.linalg as spl # in order to solve Ax=b

import sdp_matrix as sdp # generate sdp matrix
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
    # check if M is symmetric
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            if (M[i][j] != M[j][i]):
                return False

    # check if M is defined positive
    for l in spl.eig(M)[0]:
        if (type(l) == complex or l < 0) and abs(l) > 10**(-10):
            return False

    # check if M is invertible
    if spl.det(M) == 0:
        return False

    return True

def conjgrad(A,B,X,imax,p):
    if (is_symdefpos(A) == False):
        # check if A is sdp
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
        # check if A is sdp
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
