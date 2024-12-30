import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


def E(x):
    return 3 if x <= 1 else 5


def e(n, i, x):
    h = 2 / n

    xi = i * h
    xi_minus_1 = (i - 1) * h
    xi_plus_1 = (i + 1) * h

    if xi_minus_1 <= x < xi:
        return (x - xi_minus_1) / (xi - xi_minus_1)
    elif xi <= x <= xi_plus_1:
        return (xi_plus_1 - x) / (xi_plus_1 - xi)
    else: 
        return 0.0


def e_derivative(n, i, x):
    h = 2 / n

    xi = i * h
    xi_minus_1 = (i - 1) * h
    xi_plus_1 = (i + 1) * h

    if xi_minus_1 <= x < xi:
        return 1 / (xi - xi_minus_1)
    elif xi <= x <= xi_plus_1:
        return -1 / (xi_plus_1 - xi)
    else: 
        return 0.0
    

def create_matrix(n):
    B = [[0 for _ in range(n)] for _ in range(n)]
    L = [0 for _ in range(n)]

    for j in range(n):
        L[j] = -30 * e(n, j, 0)

    for i in range(n):
        for j in range(n):
            B[i][j] = integrate(n, i, j) - 3 * e(n, i, 0) * e(n, j, 0)

    return B, L


def integrate(n, i, j):
    if abs(i - j) > 1:
        return 0
    
    h = 2 / n
    a = max(max(i, j) - 1, 0) * h
    b = min(min(i, j) + 1, n) * h

    integrand = lambda x : E(x) * e_derivative(n, i, x) * e_derivative(n, j, x)
    result, _ = spi.quad(integrand, a, b)
    return result


def solve_matrix(B, L):
    return np.linalg.solve(B, L)


def create_plot(n, u_vector):
    x_values = np.linspace(0, 2, n + 1)
    u_values = [0] * (n + 1)

    for idx, x in enumerate(x_values):
        u_x = 0
        for i in range(n):
            u_x += u_vector[i] * e(n, i, x)
        u_values[idx] = u_x

    plt.title('Wykres odkształcenia sprężystego')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.plot(x_values, u_values, label='u(x)')
    plt.show()
    

def main():
    n = int(input("Podaj n: "))
    B, L = create_matrix(n)
    u_vector = solve_matrix(B, L)
    create_plot(n, u_vector)


main()