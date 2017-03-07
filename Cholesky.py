import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import math
import random
import time

##Affichage_de_la_matrice_M
def display(M):
    m = len(M[0])
    for i in range(0, m, 1):
        print(M[i])
        print("")
    print("_____________________________")

##Cholesky
def cholesky(A):
    m = len(A[0])
    L = np.zeros([m, m])
    for i in range(0, m, 1):                            
        for k in range(0, i, 1):
            L[i][i] += L[i][k]**2
        L[i][i] = math.sqrt(A[i][i] - L[i][i])
        for j in range(i + 1, m, 1):                      
            for k in range(0, i, 1):
                L[j][i] += L[i][k] * L[j][k]
            L[j][i] = (A[i][j] - L[j][i]) / L[i][i]
    return L

##Creation_de_matrice_SDP
def matrice_SDP(n, nb_extra_diag):
    A = np.zeros([n, n])
    coord = []
    for i in range(0, n, 1):
        for j in range(0, i, 1):
            coord.append([i, j])
    for i in range(0, nb_extra_diag, 1):
        c = random.randint(0, len(coord)-1)
        val = random.randint(0,50)
        A[coord[c][0]][coord[c][1]] = val
        A[coord[c][1]][coord[c][0]] = val
        del coord[c]
    for i in range(0, n, 1):
        s = 0
        for j in range(0, n, 1):
            s = s + A[i][j]
        A[i][i] = random.randint(s + 1, 2 * s +1)
    return A
        
##Cholesky_imcomplet
def cholesky_incomplet(A):
    m = len(A[0])
    L = np.zeros([m, m])
    for i in range(0, m, 1):      
        if A[i][i] != 0:
            for k in range(0, i, 1):
                L[i][i] += L[i][k]**2
            L[i][i] = math.sqrt(A[i][i] - L[i][i])
        for j in range(i + 1, m, 1): 
            if A[j][i] != 0:
                for k in range(0, i, 1):
                    L[j][i] += L[i][k] * L[j][k]
                L[j][i] = (A[i][j] - L[j][i]) / L[i][i]
    return L 

##Test_Cholesky_incomplet

def test_cholesky_incomplet():
    n = 100
    nb_extra_diag = ((n**2) // 2) - n - 1 
    t1 = np.arange(0, nb_extra_diag, 50)
    t2=[]
    t3=[]
    l=len(t1)
    for i in range(0, nb_extra_diag, 50):
        A=matrice_SDP(n, i)
        tmps=time.time()
        cholesky(A)
        t2.append(time.time()-tmps)
        tmps=time.time()
        cholesky_incomplet(A)
        t3.append(time.time()-tmps)
        
    plt.figure(1)
    plt.subplot(211)
    line1, =plt.plot(t1, t2, color="blue", label="Choleski")
    line2, =plt.plot(t1, t3, color="red", label="Cholesky incomplet")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.xlabel("Nombre de termes extra-diagonaux")
    plt.ylabel("Temps d'execution (en seconde)")

##
def preconditionneur(A):
    print("cond(A): ", np.linalg.cond(A))
    L = cholesky(A)
    print("cond(M**(-1)A): ", np.linalg.cond(np.dot(np.linalg.inv(np.dot(L, np.transpose(L))), A)))
    
##PARTIE_1
#1.Cholesky
A=np.array([[1, 1, 1, 1], [1, 5, 5, 5], [1, 5, 14, 14], [1, 5, 14, 15]])
print("Matrice A :")
display(A)
print("Cholesky(A) :")
display(cholesky(A))

"""Complexite: (n**3)/6 additions et multiplications, (n*(n-1))/2 divisions, n evaluations de racines carrees. Donc la complexite est en teta(1/3 *n**3)."""

#2.

"""
1. L'algorithme de Cholesky à une compléxité en O(n**3). 
2. A * x = b
   -> L * L_t * x = b      n**3  (Cholesky)     
   -> L * y = b            n**2  (Pivot descendant)
   -> L_t * x = y          n**2  (Pivot montant)
   
   Au final la complexite est de l'ordre de O(n**3)
"""

#3.Creation de matrice SDP
"""Prend en entree la taille de la matrice souhaitee et le nombre de termes extra-diagonaux (<n**2/2)"""
print("Matrice SDP de taille 5 avec 3 termes extra-diagonaux non nuls :")
display(matrice_SDP(5, 3))

#4.Cholesky incomplet
test_cholesky_incomplet()
#TODO: calculer la complexité

#5.
preconditionneur(A)

