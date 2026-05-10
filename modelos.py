import numpy as np
import sel as sl
import sympy as sp
import matplotlib.pyplot as plt
def crear_matriz_polinomica(x_data):
    n = len(x_data)
    A = np.zeros([n,n])
    A[0:n,0] = 1.0

    for j in range(1,n):
        for i in range(0,n):
            A[i,j] = A[i,j-1] * x_data[i]
    return A

def polinomial_simple(x_data, y_data):
    x = sp.symbols('x')
    A = crear_matriz_polinomica(x_data)
    coeficientes = sl.eliminacion_DD(A,y_data)
    Polinomio = sum(coeficientes[i] * (x ** i) for i in range(len(x_data)))
    return Polinomio

def Lagrange(x_data,y_data):
    x = sp.symbols('x')
    sumaPolinomio = 0
    for i in range(len(x_data)):
        Li=1
        for j in range(len(x_data)):
            if j != i:
                Li *= (x-x_data[j])/(x_data[i] - x_data[j])
        sumaPolinomio += Li*y_data[i]
    return sp.expand(sumaPolinomio)

# Ajustes Lineales
def minimos_cuadrados(x_data, y_data):
    n = len(x_data)
    Sx = sum(x_data)
    Sy = sum(y_data)
    Syx = sum(x_data*y_data)
    Sx2 = sum(x_data**2)
    m = (Sy*Sx-n*Syx)/(Sx**2 - n*Sx2)
    b = (Sx*Syx-Sy*Sx2)/(Sx**2-n*Sx2)
    return m,b



def grafiacion(x_data,y_data):
    plt.figure(figsize=(12,12), dpi=100)
    plt.suptitle("Escalas de transformacion para conjunto de datos", fontsize=12, fontweight="bold")
    plt.subplot(331)
    plt.plot(x_data,y_data, 'or', label="datos v")
    plt.title("Datos originales", fontsize=12, fontweight="bold")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()

    plt.subplot(332)
    plt.plot(x_data, y_data**2, 'og', label="$y^2$")
    plt.title("Transformacion para $y^2$", fontsize=12, fontweight="bold")
    plt.xlabel('x')
    plt.ylabel('$y^2$')
    plt.grid()
    plt.legend()

    plt.subplot(333)
    plt.plot(x_data,1/y_data, 'ob', label=r"$\frac{1}{y}$")
    plt.title(r"Transformacion para $\frac{1}{y}$", fontsize=12, fontweight="bold")
    plt.xlabel('x')
    plt.ylabel(r'$\frac{1}{y}$')
    plt.grid()
    plt.legend()

    plt.subplot(334)
    plt.plot(x_data,y_data**(1/2), 'om', label="$\sqrt{y}$")
    plt.title("Transformacion para $\sqrt{y}$", fontsize=12, fontweight="bold")
    plt.xlabel('x')
    plt.ylabel('$\sqrt{y}$')
    plt.grid()
    plt.legend()

    plt.subplot(335)
    plt.plot(x_data,np.log(y_data), 'ok', label="$\log{y}$")
    plt.title("Transformacion para $\log{y}$", fontsize=12, fontweight="bold")
    plt.xlabel('x')
    plt.ylabel('$\log{y}$')
    plt.grid()
    plt.legend()

    plt.subplot(336)
    plt.plot(x_data**2,y_data, 'oy', label="$x^2$")
    plt.title("Transformacion para $x^2$", fontsize=12, fontweight="bold")
    plt.xlabel('$x^2$')
    plt.ylabel('$y$')
    plt.grid()
    plt.legend()

    plt.subplot(337)
    plt.plot(x_data**3,y_data, 'oc', label="$x^3$")
    plt.title("Transformacion para $x^3$", fontsize=12, fontweight="bold")
    plt.xlabel('$x^3$')
    plt.ylabel('$y$')
    plt.grid()
    plt.legend()

    plt.subplot(338)
    plt.plot(np.log(x_data),y_data, 'or', label="$\log{x}$")
    plt.title("Transformacion para $\log{x}$", fontsize=12, fontweight="bold")
    plt.xlabel('$\log{x}$')
    plt.ylabel('$y$')
    plt.grid()
    plt.legend()

    plt.subplot(339)
    plt.plot(np.log(x_data),np.log(y_data), 'or', label="$\log{x}$")
    plt.title("Transformacion para $\log{x}$ and $\log{y}$", fontsize=12, fontweight="bold")
    plt.xlabel('$\log{x}$')
    plt.ylabel('$\log{y}$')
    plt.grid()
    plt.legend()

    plt.tight_layout()

def coeficiente_determinacion(x,y):
    y_bar = sum(y)/len(y)
    m,b = minimos_cuadrados(x,y)
    y_mod = lambda x: m*x + b
    numerador = sum((y-y_mod(x))**2)
    denominador = sum((y-y_bar)**2)
    R2 = 1 - numerador/denominador
    return R2