import numpy as np
import matplotlib.pyplot as plt
import math

def display(M):
    m = len(M)
    for i in range(0, m, 1):
        print(M[i])
        print("")
    print("__________________")

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
    
            
    


    
A = np.zeros([3, 3])
b = np.zeros(3)
for i in range (0, 3, 1):
    b[i] = 2
    for j in range (0, 3, 1):
        A[i][j] = 1
display(montee_gauss(A, b))
display(descente_gauss(A, b))
