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

##BEGIN
matrice_chaleur(2)