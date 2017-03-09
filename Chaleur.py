import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg as spl# pour la fonction solve, qui resoud une équation de type AX = B

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
def resolution_cholesky(N, b):
    h = 1 / (N + 1)
    A = matrice_chaleur(N)
    L = cholesky(-A)
    y = descente_gauss(L, b)
    x = montee_gauss(np.transpose(L), y)
    x = (x * (h**2))   # *(-1) ?
    return x


# 3
def is_symdefpos(M):
    # vérification de la symétrie de M
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            if (M[i][j] != M[j][i]):
                return False

            
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
        if (math.sqrt(rs_new) < p):
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
        if (math.sqrt(np.transpose(R).dot(R)) < p):
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

##Affichage_repartition_chaleur
def display_heat(x):
    x = np.reshape(x, (math.sqrt(len(x)), math.sqrt(len(x))))
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap=plt.get_cmap('hot'), interpolation='nearest')
    fig.colorbar(im)
    plt.show()
##BEGIN
##CHOLESKY
N = 2
A = matrice_chaleur(N)
display(A)
N = 50
#b = creation_b_nord(N)
b = creation_b_centre(N)
#x = resolution_cholesky(N, b)
#display_heat(x)

##CONJUGE
A = matrice_chaleur(N)
##conjuge gradiant
imax = 10**3 #iteration number
p = 10**(-10) #precision
x = np.zeros((2500,1))
x = conjgrad(-A, b, x, imax, p)
display_heat(x)

##conjuge gradiant precond
x = conjgrad_precond(-A, b, x, imax, p)
display_heat(x)
