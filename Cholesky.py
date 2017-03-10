import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import math
import random
import time

import sdp_matrix as sdp

##Affichage_de_la_matrice_M
def display(M):
    l = len(M)
    for i in range(0, l, 1):
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
        A=sdp.matrice_SDP(n, i)
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
    
