import numpy as np
import time

def eliminacion_DD(A,b):
    matriz_a = np.insert(A, A.shape[0], b, 1)
    x_solucion = np.zeros_like(b)

    n = len(b)

    for j in range(n):
        for i in range(j+1,n):
            factor = matriz_a[i,j]/matriz_a[j,j]
            matriz_a[i, 0:n+1] = matriz_a[i, 0:n+1] - factor*matriz_a[j, 0:n+1]

    for k in range(n-1,- 1,-1):
        x_solucion[k] = (matriz_a[k,n]-np.dot(matriz_a[k,k+1:n],x_solucion[k+1:n]))/matriz_a[k,k]
    return x_solucion

def eliminacion_con_pivoteo(A, b):
    matrix_a = np.insert(A, A.shape[0], b, 1)
    x_solucion = np.zeros_like(b)

    n = len(b)

    if matrix_a[0, 0] == 0:
        for j in range(n + 1):
            for i in range(j + 1, n):
                matrix_a[[j, i]] = matrix_a[[i, j]]
                break

    for j in range(n + 1):
        for i in range(j + 1, n):
            factor = matrix_a[i, j] / matrix_a[j, j]
            matrix_a[i, 0:n+1] = matrix_a[i, 0:n+1] - factor * matrix_a[j, 0:n+1]

    for k in range(n-1, -1, -1):
        x_solucion[k] = (matrix_a[k, n] - np.dot(matrix_a[k, k+1:n], x_solucion[k+1:n])) / matrix_a[k, k]

    return x_solucion

def gauss_seidel_mat(A,b,x0, tol=1e-6):
    D = np.diag(np.diag(A))
    U = D - np.triu(A)
    L = D - np.tril(A)
    cont = 0
    T_g = np.dot(np.linalg.inv(D-L),U)       #para calcular la inversa

    C_g = np.dot(np.linalg.inv(D-L),b)
    print(T_g,C_g)
    eigvalues, eigvectores = np.linalg.eig(T_g)
    radio_espectral = max(abs(eigvalues))
    print(radio_espectral)

    if radio_espectral >= 1:
        return 0
    else:
        error = 1
        while error > tol:
            cont += 1
            x1 = np.dot(T_g,x0) + C_g
            error = max(abs(x1-x0))
            x0 = x1
    return x1

def gauss_seidel_sumas(A,b,x0,tol):
    error = 1
    cont = 0
    n = len(b)
    while error > tol:
        x_new = np.zeros_like(b)
        cont += 1
        for i in range(n):
            suma = 0
            for j in range(i):
                suma += A[i,j]*x_new[j]
            suma2 = 0
            for j in range(i+1,n):
                suma2 += A[i,j]*x0[j]

            x_new[i] = (b[i] - suma - suma2)/A[i,i]
        error = max(abs(x_new-x0))
        x0 = x_new
        print(f"vamoas en la itereacion k = {cont}:\n{x_new}\n")
    return x0

def esDiagonal(A,b):
    n = len(b)
    for i in range(n):
        diagonal = sum(abs(A[i,0:n]))-abs(A[i,i])
        if abs(A[i,i]) < diagonal:
            return False
    return True

def Jacobi_suma(A,b,x0,tol=10**-6):
    Nmax = 50
    conteo = 0
    error = 1
    n = len(b)
    x_k = np.zeros_like(b)
    if not esDiagonal:
        return None
    
    while error>tol and conteo< Nmax: #and iter < Nmax:
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += A[i,j]*x0[j]
            x_k[i] = (b[i]-suma)/A[i,i]
        conteo += 1
        error = max(abs(x_k - x0))
        x0 = x_k.copy()
    return x0

def Jacobi_matrices(A,b,x0,tol):
    D = np.diag(np.diag(A))
    U = D - np.triu(A)
    L = D - np.tril(A)
    cont = 0

    T_jab = np.dot(np.linalg.inv(D), L+U)

    C_jab = np.dot(np.linalg.inv(D), b)

    eigvalues, eigvectores = np.linalg.eigh(T_jab)
    radio_espectral = max(abs(eigvalues))

    if radio_espectral > 1:
        return -1
    error = 1
    inicial = time.time()
    while error > tol:
        cont += 1
        x1 = np.dot(T_jab,x0) + C_jab
        if cont == 3:
            continue
        else:
            print(f"vamos en la iteracion x_{cont}: {x1}")
        error = max(abs(x1-x0))
        x0 = x1
    tiempo_ejecucion = time.time() - inicial
    print(f"el tiempo de demora fue de {tiempo_ejecucion}")
    return x1