import numpy as np
import matplotlib.pyplot as plt
import math

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

##Pivot de Gauss
def montee_gauss(T, b):
    n = len(T[0])
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        y = b[i]
        for j in range(i + 1, n, 1):
            y = y -(T[i, j] * x[j])
        x[i] = y/(T[i, i])
    return x

def descente_gauss(T, b):
    n = len(T[0])
    x = np.zeros(n)
    for i in range(0, n, 1):
        y = b[i]
        for j in range(0, i, 1):
            y = y -(T[i, j] * x[j])
        x[i] = y/(T[i, i])
    return x

##Creation_de_la_matrice_de_l_equation_de_la_chaleur
def matrice_chaleur(N):
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
def creation_b_nord(N):
    b = np.zeros([N**2, 1])
    for i in range(0, N, 1):
        b[i] = 1
    return b 
    
def creation_b_centre(N):
    b = np.zeros([N**2, 1])
    b[((N**2)//2) - 1 + (N//2)] = 1    
    return b 

##Resolution_Cholesky
def resolution_cholesky(N, b):
    h = 1 / (N + 1)
    A = matrice_chaleur(N)
    L = cholesky(-A)
    y = descente_gauss(L, b)
    x = montee_gauss(np.transpose(L), y)
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
N = 2
A = matrice_chaleur(N)
display(A)
N = 20
#b = creation_b_nord(N)
b = creation_b_centre(N)
x = resolution_cholesky(N, b)
display_heat(x)


