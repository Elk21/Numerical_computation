# Laba 4      Eiler method
# Laba 5      Runge-Kutta method
# Laba 4-5    Compair Runge-Kutt and Eiler
# Laba 6      LU decomposition
# Laba 7      System of linear algebraic equations solver

from sys import setrecursionlimit
import time
from numpy import abs, exp
from pandas import DataFrame, set_option
from scipy import linalg, pi, array, matmul

setrecursionlimit(5000)

set_option('display.height', 1000)
set_option('display.width', 1000)

X0 = 0
X1 = 1
Y0 = exp(1) / (exp(1) + 1)
# Number of points
N = 300
step = (X1 - X0) / N
rungeAr = []
eilerAr = []


# Initial function
def func(x, y):
    return x * x * y


# Actual value
def f(x):
    return Y0 * pow(exp(1), pow(x, 3) / 3)


def eiler(n):
    if n == 0:
        eilerAr.append([(X0 + n * step), f(X0 + n * step), Y0, abs(f(X0 + n * step) - Y0)])
        return Y0

    yi = eiler(n - 1)
    xi = X0 + n * step
    c1 = yi + step * func(xi, yi)
    # c1 = c0 + ((X0 + n * step) - (X0 + (n - 1) * step)) * func((X0 + n * step), c0)
    eilerAr.append([(X0 + n * step), f(X0 + n * step), c1, abs(f(X0 + n * step) - c1)])
    return c1


def runge(n):
    if n == 0:
        out = [(X0 + n * step), Y0, f(X0 + n * step), abs(f(X0 + n * step) - Y0)]
        rungeAr.append(out)
        return Y0

    xi = X0 + step * (n - 1)
    yi = runge(n - 1)
    d = yi + step * func(xi + step / 2, yi + step * func(xi, yi) / 2)
    rungeAr.append([xi, f(xi), d, abs(f(xi + step) - d)])
    return d


def printEiler():
    e = DataFrame(eilerAr, columns=['Xi', 'y(Xi)', 'Eiler', 'Error E'])
    print(e)


def printRunge():
    r = DataFrame(rungeAr, columns=['Xi', 'y(Xi)', 'Runge', 'Error R'])
    print(r)


def printRungeEiler():
    r = DataFrame(rungeAr, columns=['Xi', 'y(Xi)', 'Runge', 'Error R'])
    e = DataFrame(eilerAr, columns=['Xi', 'y(Xi)', 'Eiler', 'Error E'])
    r['Eiler'] = e['Eiler']
    r['Error E'] = e['Error E']
    print(r)


# ************
# LABA 6 - 7
# ************

# # Source matrix laba6
# A = [[8 * pi, 2, 0, 0],
#      [1, 6 * pi, 3, 3],
#      [0, 3, 5 * pi, 1],
#      [2, 2, 1, 5 * pi]]
# # Source vector laba7
# b = [3, 5, 7, 9]

# Ruslan
# Source matrix laba6
A = [[5 * pi, 1, 3, 0],
     [2, 4 * pi, 2, 2],
     [0, 3, 8 * pi, 0],
     [3, 3, 1, 6 * pi]]
# Source vector laba7
b = [3, 6, 1, 6]


P, L, U = linalg.lu(A)


def combine(L, U):
    LU = L.copy()
    for i in range(0, len(L)):
        for j in range(i, len(L[0])):
            LU[i][j] = U[i][j]
    return LU


def lux(L, b):
    x = [0, 0, 0, 0]
    x[0] = b[0] / L[0][0]
    x[1] = (b[1] - L[1][0] * x[0]) / L[1][1]
    x[2] = (b[2] - L[2][0] * x[0] - L[2][1] * x[1]) / L[2][2]
    x[3] = (b[3] - L[3][0] * x[0] - L[3][1] * x[1] - L[3][2] * x[2]) / L[3][3]
    return x


def uxy(U, y):
    x = [0, 0, 0, 0]
    x[3] = y[3] / U[3][3]
    x[2] = (y[2] - U[2][3] * x[3]) / U[2][2]
    x[1] = (y[1] - U[1][3] * x[3] - U[1][2] * x[2]) / U[1][1]
    x[0] = (y[0] - U[0][3] * x[3] - U[0][2] * x[2] - U[0][1] * x[1]) / U[0][0]
    return x


# Laba 6
def printLU():
    print('\n\nMatrix A:\n')
    print(array(A))
    print('\n\nMatrix L:\n')
    print(array(L))
    print('\n\nMatrix U:\n')
    print(array(U))
    print('\n\nMatrix L * U - A\n')
    print(matmul(L, U) - A)
    print('\n\nMatrix LU:\n')
    print(combine(L, U))


# Laba 7
def printSOLAE():
    print('\n\nMatrix A:\n')
    print(array(A))
    print('\n\nMatrix L:\n')
    print(array(L))
    print('\n\nMatrix U:\n')
    print(array(U))
    print('\n\ny = b * L^-1\n')
    print(lux(L, b))

    print('\n\nx = y * U^-1\n')
    x = uxy(U, lux(L, b))
    print(x)
    print('\n\nA * x - b\n')
    print(matmul(A, x) - b)


# Engine of this program
# DO NOT TOUCH THIS
def selectProgram():
    print('\n\n************************************************************')
    print('Program names:\n\n'
          'Laba        Algorithm                                  Name\n\n'
          'Laba 4      Eiler                                  -   eiler\n'
          'Laba 5      Runge-Kutt                             -   runge\n'
          'Laba 4-5    Compair Runge-Kutt and Eiler           -   re\n'
          'Laba 6      LU decomposition                       -   lu\n'
          'Laba 7      System of linear algebraic equations   -   solae')
    print('************************************************************')
    print('Type 0 to exit\nType program name: ')
    cin = input()

    if cin == '0':
        return 0
    elif cin == 'eiler':
        printEiler()
        time.sleep(2)
        selectProgram()
    elif cin == 'runge':
        printRunge()
        time.sleep(2)
        selectProgram()
    elif cin == 'runge+eiler' or cin == 'er' or cin == 're':
        printRungeEiler()
        time.sleep(2)
        selectProgram()
    elif cin == 'lu':
        printLU()
        time.sleep(2)
        selectProgram()
    elif cin == 'solae':
        printSOLAE()
        time.sleep(2)
        selectProgram()
    else:
        for i in range(10):
            print('TI TUPOI UEBOK')
        print('\n\nIncorrect program name\nTry again')
        time.sleep(2)
        selectProgram()


def engine():
    runge(N)
    eiler(N)
    selectProgram()


engine()
