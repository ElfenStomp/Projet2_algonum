import numpy as np
import matplotlib.pyplot as plt
import math
import random

def display(M):
    m = len(M[0])
    for i in range(0, m, 1):
        print(M[i])
        print("")
    print("_____________________________")
    
##Transposé
def transposer_triangulaire_inf(L):
    m = len(L[0])
    L_t = np.zeros([m, m])
    for i in range(0, m, 1):
        for j in range(0, i+1, 1):
            L_t[j][i] = L[i][j]
    return L_t
#Pas besoin de calculer L_t (prend de la place mémoire inutile)

##Cholesky
def cholesky(A):
    m = len(A[0])
    L = np.zeros([m, m])
    for i in range(0, m, 1):                            
        for k in range(0, i, 1):
            L[i][i] += L[i][k]**2
        L[i][i] = math.sqrt(A[i][i] - L[i][i])
        for j in range(i+1, m, 1):                      
            for k in range(0, i, 1):
                L[j][i] += L[i][k]*L[j][k]
            L[j][i] = (A[i][j] - L[j][i]) / L[i][i]
    return L
#Complexité: (n**3)/6 additions et multiplications, (n*(n-1))/2 divisions, n évaluations de racines carrées

##Cholesky imcomplet
def cholesky_imcomplet(A):
    m = len(A[0])
    L = np.zeros([m, m])
    for i in range(0, m, 1):      
        if A[i][i] != 0:
            for k in range(0, i, 1):
                L[i][i] += L[i][k]**2
            L[i][i] = math.sqrt(A[i][i] - L[i][i])
        for j in range(i+1, m, 1): 
            if A[j][i] != 0:
                for k in range(0, i, 1):
                    L[j][i] += L[i][k]*L[j][k]
                L[j][i] = (A[i][j] - L[j][i]) / L[i][i]
    return L 
np.ch
##
def matrice_creation(n, nb_extra_diag):
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
        A[i][i] = random.randint(s, 2*s)
     
    display(A)
ch
##
def preconditionneur(A):
    L = cholesky(A)
    display(L)
    L_t = transposer_triangulaire_inf(L)
    display(L_t) 
    print(np.linalg.cond(A))
    print(np.linalg.cond(np.dot(np.linalg.inv(np.dot(L, L_t)), A)))
    
##BEGIN
A=np.array([[1, 1, 1, 1], [1, 5, 5, 5], [1, 5, 14, 14], [1, 5, 14, 15]])
preconditionneur(A)
n = 5
nb_extra_diag = 2
matrice_creation(n, nb_extra_diag) 

##Partie 1 
"""
1. L'algorithme de Cholesky à une compléxité en O(n**3). 
2. A * x = b
   L * L_t * x = b      n**3
   -> L * y = b         n**2
   -> L_t * x = y       n**2
   
   Au final la compléxité est de l'ordre de O(n**3)
"""
## 0