import math


def sphere(x):# Funcao Sphere
    return sum(i ** 2 for i in x)


def rastrigin(x):# Funcao Rastrigin
    return 10 * len(x) + sum((i ** 2) - 10 * math.cos(2 * math.pi * i) for i in x)


def rosenbrock(x):# Funcao Rosenbrock
    return sum(100 * ((x[i+1] - x[i] ** 2) ** 2) + (x[i] - 1) ** 2 for i in range(len(x)-1))
