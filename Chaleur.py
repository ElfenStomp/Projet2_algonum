import numpy as np
import matplotlib.pyplot as plt
import math

def display(M):
    m = len(M[0])
    for i in range(0, m, 1):
        print(M[i])
        print("")
    print("_____________________________")

##
def matrice_chaleur(N):
    m = N**2
    A = np.zeros([m, m])
    for i in range(0, m, 1):
        A[i][i] = -4        #diagonale
        if i%N != 0:   
            A[i][i-1] = 1   #extra diagonale inferieur
            A[i-1][i] = 1   #extra diagonale supérieur
        if (i >= N):
            A[i-N][i] = 1   #diagonale supérieur
            A[i][i-N] = 1   #diagonale inférieur
    return A

##
def creation_b(N):
    b = np.zeros([N**2, 1])
    return b 

##BEGIN
N = 2
A = matrice_chaleur(N)
display(A)
b = creation_b(N)
h = 1 / (N + 1)