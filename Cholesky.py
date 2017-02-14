import numpy as np
import matplotlib.pyplot as plt
import math

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

##
def matrice_creation():
    None
    
##
def matrice_chaleur(n):
    m = n**2
    A = np.zeros([m, m])
    for i in range(0, m, 1):
        A[i][i] = -4        #diagonale
        if i%n != 0:   
            A[i][i-1] = 1   #extra diagonale inferieur
            A[i-1][i] = 1   #extra diagonale supérieur
        if (i >= n):
            A[i-n][i] = 1   #diagonale supérieur
            A[i][i-n] = 1   #diagonale inférieur
    display(A)

##
def preconditionneur(A):
    L = cholesky(A)
    display(L)
    L_t = transposer_triangulaire_inf(L)
    display(L_t) 
    print(np.linalg.cond(A))
    print(np.linalg.cond(np.dot(np.linalg.inv(np.dot(L, L_t)), A)))

matrice_chaleur(2)
A=np.array([[1, 1, 1, 1], [1, 5, 5, 5], [1, 5, 14, 14], [1, 5, 14, 15]])
preconditionneur(A)

##Partie 1 
"""
1. L'algorithme de Cholesky à une compléxité en O(n**3). 
2. A * x = b
   L * L_t * x = b      n**3
   -> L * y = b         n**2
   -> L_t * x = y       n**2
   
   Au final la compléxité est de l'ordre de O(n**3)
"""
## 