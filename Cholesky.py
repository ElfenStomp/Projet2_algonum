import numpy as np
import matplotlib.pyplot as plt
import math

def display(t):
    for i in range(0, len(t[0]), 1):
        print(t[i])
    
def matrice_creation(n):
    m = n**2
    t=np.zeros([m, m])
    for i in range(0, m, 1):
        t[i][i] = -4        #diagonale
        if i%n != 0:   
            t[i][i-1] = 1   #extra diagonale inferieur
            t[i-1][i] = 1   #extra diagonale supérieur
        if (i >= n):
            t[i-n][i] = 1   #diagonale supérieur
            t[i][i-n] = 1   #diagonale inférieur
        
            
    display(t)
    
matrice_creation(4)
            