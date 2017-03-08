# Méthode du gradient conjugué

    # Bibliothèques nécessaires

import numpy as np
import math as m
import scipy.linalg as spl # pour la fonction solve, qui résoud une équation de type AX = B
import matplotlib.pyplot as plt # pour tracer la courbe de comparaisons entre la méthode du gradient conjugué
                                # et la fonction solve de la bibliothèque scipy.linalg

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
R = B - AX vers le vecteur nul.

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

def conjgrad_2(A,B,X,imax,p):
    if (is_symdefpos(A) == False):
        # On vérifie que A soit bien symétrique définie positive.
        print("\n A n'est pas symétrique définie positive")
        return np.zeros((np.shape(A)[0],0))
    R = B - A.dot(X)
    P = R
    rs_old = np.transpose(R).dot(R)
    for i in range(1, imax + 1):
        Ap = A.dot(P)
        alpha = rs_old / np.transpose(P).dot(Ap)
        #print("\n X = \n", X)
        X = X + (alpha * P)
        #print("\n R = \n", R)
        R = R - (alpha * Ap)
        rs_new = np.transpose(R).dot(R)
        if (rs_new < p):
            break
        P = R + (rs_new/rs_old) * P
        rs_old = rs_new
    return X

def conjgrad(A,B,X,imax,p):
    if (is_symdefpos(A) == False):
        # On vérifie que A soit bien symétrique définie positive.
        print("\n A n'est pas symétrique définie positive")
        return np.zeros((np.shape(A)[0],0))
    R = B - A.dot(X)
    P = R
    rs_old = np.transpose(R).dot(R)
    i = 0
    while (i < imax and rs_old >= p):
        Ap = A.dot(P)
        alpha = rs_old / np.transpose(P).dot(Ap)
        X = X + (alpha * P)
        print("\n X = \n", X)
        R = R - (alpha * Ap)
        print("\n R = \n", R)
        rs_new = np.transpose(R).dot(R)
        P = R + (rs_new/rs_old) * P
        rs_old = rs_new
        i += 1
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
    p = 10**(-10)
    print("\n Précision à ",p," près \n")
        # Comparaison sommaire entre conjgrad et np.linalg.solve
    X1 = conjgrad_2(A, B, X, imax, p)
    X2 = spl.solve(A, B)
    print("\n Avec la méthode du gradient conjugué on a X = \n", X1)
    print("Avec la fonction solve de la biliothèque scipy.linalg on a X = \n", X2)

    while True :
        answer = int(input("Ecrivez 1 si vous souhaitez continuer le programme pour afficher les courbes de convergence de X en fonction de la précision ou 0 si vous souhaitez arrêter le programme: "))
        if answer == 1 :
            print("Suite du programme")
            break
        elif answer == 0 :
            print("Arrêt du programme de test")
            return
        else:
            print("Réponse invalide")
            continue
        # Vitesse de convergence selon la précision (graphes avec matplotlib)
    t = np.linspace(0, 10, 100) # L'axe des abscisses qui représente la précision
    Xzero = np.array([[0.],[0.],[0.]])

    # Les valeurs du vecteur X solution calculées avec la méthode du gradient conjugué
    x0_conjgrad = [conjgrad_2(A, B, Xzero, imax, 10**(-i))[0] for i in t]
    x1_conjgrad = [conjgrad_2(A, B, Xzero, imax, 10**(-i))[1] for i in t]
    x2_conjgrad = [conjgrad_2(A, B, Xzero, imax, 10**(-i))[2] for i in t]

    # Les valeurs du vecteur X solution calculées avec la fonction solve de la bibliothèque scipy.linalg
    x0_solve = [X2[0] for i in t]
    x1_solve = [X2[1] for i in t]
    x2_solve = [X2[2] for i in t]

    plt.plot(t, x0_conjgrad, color='red', label="X[0]_conjgrad")
    plt.plot(t, x1_conjgrad, color='blue', label="X[1]_conjgrad")
    plt.plot(t, x2_conjgrad, color='green', label="X[2]_conjgrad")
    plt.plot(t, x0_solve, color='red', ls="dotted", label="X[0]_solve")
    plt.plot(t, x1_solve, color='blue', ls="dotted", label="X[1]_solve")
    plt.plot(t, x2_solve, color='green', ls="dotted", label="X[2]_solve")

    plt.legend()
    plt.xlabel("précision")
    plt.show()

#def conjgrad_precond():
