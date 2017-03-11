import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import sdp_matrix as sdp
import Cholesky as cho

##Test incomplete cholesky and cholesky
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
        cho.cholesky(A)
        t2.append(time.time()-tmps)
        tmps=time.time()
        cho.cholesky_incomplet(A)
        t3.append(time.time()-tmps)
        
    plt.figure(1)
    plt.subplot(211)
    line1, =plt.plot(t1, t2, color="blue", label="Choleski")
    line2, =plt.plot(t1, t3, color="red", label="Cholesky incomplet")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.xlabel("Nombre de termes extra-diagonaux")
    plt.ylabel("Temps d'execution (en seconde)")
    plt.show()
    
    

##PARTIE_1
#1.Cholesky
A=np.array([[1, 1, 1, 1], [1, 5, 5, 5], [1, 5, 14, 14], [1, 5, 14, 15]])
print("Matrice A :")
cho.display(A)
print("Cholesky(A) :")
cho.display(cho.cholesky(A))

"""Complexite: (n**3)/6 additions et multiplications, (n*(n-1))/2 divisions, n evaluations de racines carrees. Donc la complexite est en teta(1/3 *n**3)."""

#2.

"""
1. L'algorithme de Cholesky à une compléxité en O(n**3). 
2. A * x = b
   -> L * L_t * x = b      n**3  (Cholesky)     
   -> L * y = b            n**2  (Pivot descendant)
   -> L_t * x = y          n**2  (Pivot montant)
   
   Au final la complexite est de l'ordre de O(n**3)
"""

#3.Creation de matrice SDP
"""Prend en entree la taille de la matrice souhaitee et le nombre de termes extra-diagonaux (<n**2/2)"""
print("Matrice SDP de taille 5 avec 3 termes extra-diagonaux non nuls :")
cho.display(sdp.matrice_SDP(5, 3))

#4.Cholesky incomplet
test_cholesky_incomplet()
#TODO: calculer la complexité

#5.
cho.preconditionneur(A)
