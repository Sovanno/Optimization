def f(x):
    return x[0]**2 + x[1]**2 + x[0] * 2 + x[1] * 3 - 6
    
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd

def hook_jeeves(x_start, delta, epsilon):
    df = pd.DataFrame(columns=['x', 'y', 'f(x, y)'])
    x = x_start.copy()
    x_new = x_start.copy()

    while True:
        # Исследовательский поиск
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += delta
            x_minus = x.copy()
            x_minus[i] -= delta
            if f(x_plus) < f(x):
                x_new[i] = x_plus[i]
            elif f(x_minus) < f(x):
                x_new[i] = x_minus[i]
        # Узловой поиск
        x_pattern = 2*x_new - x
        if f(x_pattern) < f(x):
            x = x_pattern.copy()
            df = pd.concat([df, pd.DataFrame({'x': [x[0]], 'y': [x[1]], 'f(x, y)': [f(x)]})], ignore_index=True)
        else:
            if np.linalg.norm(delta) < epsilon:
                df = pd.concat([df, pd.DataFrame({'x': [x[0]], 'y': [x[1]], 'f(x, y)': [f(x)]})], ignore_index=True)
                break
            delta /= 2
            x_new = x.copy()

    return x, f(x), df

def gauss_zeidel(x_start, n, epsilon):
    df = pd.DataFrame(columns=['x', 'y', 'f(x, y)'])
    y = np.copy(x_start)
    k = 1
    j = 1

    while True:
        # Одномерный поиск
        res = minimize_scalar(lambda lmbd: f(y + lmbd * np.eye(n)[j-1]))
        lmbd = res.x

        # Обновление y
        y = y + lmbd * np.eye(n)[j-1]

        if j < n:
            j += 1
            df = pd.concat([df, pd.DataFrame({'x': [y[0]], 'y': [y[1]], 'f(x, y)': [f(y)]})], ignore_index=True)
        else:
            if np.linalg.norm(y - x_start) < epsilon:
                df = pd.concat([df, pd.DataFrame({'x': [y[0]], 'y': [y[1]], 'f(x, y)': [f(y)]})], ignore_index=True)
                break
            else:
                x_start = y
                j = 1
                k += 1
                df = pd.concat([df, pd.DataFrame({'x': [y[0]], 'y': [y[1]], 'f(x, y)': [f(y)]})], ignore_index=True)
    
    return y, f(y), df


def rosenbrock(x_start, alpha, epsilon):
    df = pd.DataFrame(columns=['x', 'y', 'f(x, y)'])
    x = x_start.copy()

    while True:
        x_prev = x.copy()

        # Поиск вдоль оси x
        f_prev = f(x_start)
        x_start[0] -= alpha * grad_f(x_start)[0]

        if abs(f(x_start) - f_prev) < epsilon:
            break

        # Поиск вдоль оси y
        f_prev = f(x_start)
        x_start[1] -= alpha * grad_f(x_start)[1]

        if abs(f(x_start) - f_prev) < epsilon:
            break

        # Проверка критерия остановки
        if np.linalg.norm(x_start - x_prev) < epsilon:
            break

        df = pd.concat([df, pd.DataFrame({'x': [x_start[0]], 'y': [x_start[1]], 'f(x, y)': [f(x_start)]})],
                       ignore_index=True)

    return x_start, f(x_start), df

def grad_f(x):
    dfdx = 2*x[0] + 2
    dfdy = 2*x[1] + 3
    return np.array([dfdx, dfdy])


def rapid_descent(x_start, epsilon):
    df = pd.DataFrame(columns=['x', 'y', 'f(x, y)'])

    while True:
        g = grad_f(x_start)

        if np.linalg.norm(g) < epsilon:
            df = pd.concat([df, pd.DataFrame({'x': [x_start[0]], 'y': [x_start[1]], 'f(x, y)': [f(x_start)]})],ignore_index=True)
            break

        # направление спуска
        S = -g / np.linalg.norm(g)

        res = minimize_scalar(lambda alpha: f(x_start + alpha * S))
        alpha_opt = res.x

        x_start = x_start + alpha_opt * S
        df = pd.concat([df, pd.DataFrame({'x': [x_start[0]], 'y': [x_start[1]], 'f(x, y)': [f(x_start)]})],ignore_index=True)

    return x_start, f(x_start), df
