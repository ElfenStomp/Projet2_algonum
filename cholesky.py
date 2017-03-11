import numpy as np
import math
import random

import sdp_matrix as sdp

##Display of the matrix M.
def display(M):
    l = len(M)
    for i in range(0, l, 1):
        print(M[i])
        print("")
    print("_____________________________")

##Cholesky
##A: SDP matrix 
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

##Incomplete Cholesky
##A: SDP matrix
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

##preconditioned
## A: SDP matrix
## display cond(A) and cond(M**-1 A) with M = L*transpose(L)
def preconditionneur(A):
    print("cond(A): ", np.linalg.cond(A))
    L = cholesky(A)
    print("cond(M**(-1)A): ", np.linalg.cond(np.dot(np.linalg.inv(np.dot(L, np.transpose(L))), A)))
    
