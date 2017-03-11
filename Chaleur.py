import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg as spl# to use the function solve

import Cholesky as cho
import conjgrad as conj

##gauss_tool
def gauss_up(T, b):
    n = len(T[0])
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        y = b[i]
        for j in range(i + 1, n, 1):
            y = y -(T[i, j] * x[j])
        x[i] = y/(T[i, i])
    return x

def gauss_down(T, b):
    n = len(T[0])
    x = np.zeros(n)
    for i in range(0, n, 1):
        y = b[i]
        for j in range(0, i, 1):
            y = y -(T[i, j] * x[j])
        x[i] = y/(T[i, i])
    return x

##Creation_de_la_matrice_de_l_equation_de_la_chaleur
def heat_matrix(N):
    m = N**2
    A = np.zeros([m, m])
    for i in range(0, m, 1):
        A[i][i] = -4        #diagonale
        if i % N != 0:   
            A[i][i - 1] = 1   #extra diagonale inferieur
            A[i - 1][i] = 1   #extra diagonale supérieur
        if (i >= N):
            A[i - N][i] = 1   #diagonale supérieur
            A[i][i - N] = 1   #diagonale inférieur
    return A

##Creation_du_vecteur_b
def creation_b_north(N):
    b = np.zeros([N**2, 1])
    for i in range(0, N, 1):
        b[i] = 1
    return b 
    
def creation_b_center(N):
    b = np.zeros([N**2, 1])
    center = ((N**2)//2) - 1 + (N//2)
    if N == 1:
        b[center] = 1
        return b
    b[center - N] = 1
    b[center - N + 1] = 1 
    b[center] = 1 
    b[center + 1] = 1     
    return b 

##Resolution_Cholesky
def resolution_cholesky(N, A, b):
    h = 1 / (N + 1)
    L = cho.cholesky(-A)
    y = gauss_down(L, b)
    x = gauss_up(np.transpose(L), y)
    x = (x * (h**2))   # *(-1) ?
    return x


##Affichage_repartition_chaleur
def display_heat(x):
    x = np.reshape(x, (math.sqrt(len(x)), math.sqrt(len(x))))
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap=plt.get_cmap('hot'), interpolation='nearest')
    fig.colorbar(im)
    plt.show()

##BEGIN    
##heat_matrix example
N = 2
A = heat_matrix(N)
cho.display(A)

N = 30
A = heat_matrix(N)
#b = creation_b_nord(N)
b = creation_b_center(N)


##CHOLESKY
#x = resolution_cholesky(N, A, b)
#display_heat(x)

##CONJUGATE
imax = 10**3 #iteration number
p = 10**(-10) #precision
##conjuge gradiant
x = np.zeros((N**2,1))
x = conj.conjgrad(-A, b, x, imax, p)
display_heat(x)

##conjuge gradiant precond
x = np.zeros((N**2,1))
x = conj.conjgrad_precond(-A, b, x, imax, p)
display_heat(x)
