import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import math

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def f(x):
    return x**5-x**2
def dichotomy(a, b, e, pand):
    x = (a + b) / 2
    x1 = x - e / 2
    x2 = x + e / 2
    f1 = f(x1)
    f2 = f(x2)
    if e == 0:
        x = (a + b) / 2
        cel = f(x)
        return cel, x, pand
    elif abs(b - a) < 2 * e:
        x = (a + b) / 2
        cel = f(x)
        return cel, x, pand
    elif f1 > f2:
        a = x1
        pand = pd.concat([pand, pd.DataFrame({'a': [a], 'b': [b], 'x': [x]})], ignore_index=True)
        return dichotomy(a, b, e, pand)
    else:
        b = x2
        pand = pd.concat([pand, pd.DataFrame({'a': [a], 'b': [b], 'x': [x]})], ignore_index=True)
        return dichotomy(a, b, e, pand)

def golden_sec(a, b, e, pand):
    t = 0.618
    L = abs(b - a)
    x1 = a + L * t
    x2 = b - L * t
    f1 = f(x1)
    f2 = f(x2)
    if e == 0:
        cel = min(f1, f2)
        return cel, pand
    elif L < e:
        cel = min(f1, f2)
        return cel, pand
    elif f1 > f2:
        b = x1
        pand = pd.concat([pand, pd.DataFrame({'a': [a], 'b': [b], 'L': [L]})], ignore_index=True)
        return golden_sec(a, b, e, pand)
    else:
        a = x2
        pand = pd.concat([pand, pd.DataFrame({'a': [a], 'b': [b], 'L': [L]})], ignore_index=True)
        return golden_sec(a, b, e, pand)

def half(a, b, e):
    l = 2 * e
    n = -int(np.log(l / (b - a)) / (np.log(2) - np.log(1)))
    k = 0
    while k != n - 1:
        xk = 1 / 2 * (a + b)

        x = symbols('x')
        ux = diff(f(x), x)
        fi = ux.subs({x: xk})

        if fi == 0:
            break
        elif fi > 0:
            b = xk
        else:
            a = xk

        k += 1

    asd = (a + b) / 2
    return f(asd), n

def fibonacci(n):
    if n <= 0:
        return [0]
    elif n == 1:
        return [0, 1]
    else:
        fib_lst = [0, 1]
        while len(fib_lst) < n:
            next_fib = fib_lst[-1] + fib_lst[-2]
            fib_lst.append(next_fib)
        return fib_lst

def fibonacci_search(a, b, epsilon):
    if epsilon == 0:
        return 0
    fib_lst = fibonacci(math.ceil((b - a) / epsilon))
    n = len(fib_lst) - 1
    x1 = a + (fib_lst[n-1] / fib_lst[n]) * (b - a)
    x2 = b - (fib_lst[n-1] / fib_lst[n]) * (b - a)
    y1 = f(x1)
    y2 = f(x2)

    while n > 2:
        if y1 > y2:
            b = x1
            x1 = x2
            y1 = y2
            L = b - a
            x2 = b - (fib_lst[n - 2] / fib_lst[n - 1]) * L
            y2 = f(x2)
        else:
            a = x2
            x2 = x1
            y2 = y1
            L = b - a
            x1 = a + (fib_lst[n - 2] / fib_lst[n - 1]) * L
            y1 = f(x1)
        n = n - 1
    return min(y1, y2)
def the_purpose_of_the_work():
    st.title("Цель работы")
    st.write("Изучение и анализ поисковых алгоритмов минимизациифункции одной переменной: дихотомического, Фибоначчи и «золотого сечения».")

def research_assignment():
    st.title("Исходное задание на исследование")

    st.header("Вариант - 9")
    st.write("Функция")
    st.latex("f(x) = x^5 - x^2")
    st.latex("a = 0")
    st.latex("b = 1")

def block_diagrams():
    st.title("Блок-схемы алгоритмов")

    st.header("Метод дихтомии")
    file_path = 'Project/The_dichotomy_method.txt'
    text_content = read_txt_file(file_path)
    st.text(text_content)

    st.header("Метод Фибоначчи")
    file_path = 'Project/Fibonacci_method.txt'
    text_content = read_txt_file(file_path)
    st.text(text_content)

    st.header("Метод золотого сечения")
    file_path = 'Project/Golden_section_method.txt'
    text_content = read_txt_file(file_path)
    st.text(text_content)

def tables():
    st.title("Таблицы с результатами исследований по каждому методу")

    methods = {
        "Метод дихтомии": 1,
        "Метод Фибоначчи": 2,
        "Метод золотого сечения": 3,
        "Метод деления пополам": 4
    }

    method = st.selectbox("Выберите метод", tuple(methods.keys()))

    if method == "Метод дихтомии":
        e = st.number_input('Введите e', step=0.00001)
        st.write(e)
        df = pd.DataFrame(columns=['a', 'b', 'x'])
        df = pd.concat([df, pd.DataFrame({'a': [0], 'b': [1], 'x': [0.5]})], ignore_index=True)
        cel, x, df = dichotomy(0, 1, e, df)
        st.write("Значение целевой функции")
        st.latex(cel)
        st.write("Значение x")
        st.latex(x)
        st.write("Таблица с итерациями")
        st.write(df)

    elif method == "Метод Фибоначчи":
        e = st.number_input('Введите e', step=0.00001)
        st.write(e)
        cel = fibonacci_search(0, 1, e)
        st.write("Значение целевой функции")
        st.latex(cel)
        return 0

    elif method == "Метод золотого сечения":
        e = st.number_input('Введите e', step=0.00001)
        st.write(e)
        df = pd.DataFrame(columns=['a', 'b', 'L'])
        df = pd.concat([df, pd.DataFrame({'a': [0], 'b': [1], 'L': [1]})], ignore_index=True)
        cel, df = golden_sec(0, 1, e, df)
        st.write("Значение целевой функции")
        st.latex(cel)
        st.write("Таблица с итерациями")
        st.write(df)
        return 0

    elif method == "Метод деления пополам":
        e = st.number_input('Введите e', step=0.00001)
        st.write(e)
        cel = half(0, 1, e)
        st.write("Значение целевой функции")
        st.latex(cel)
        return 0


def chart():
    st.title("График зависимости количества вычислений целевой функции от логарифма задаваемой точности E")

    methods = {
        "Метод дихтомии": 1,
        "Метод золотого сечения": 3
    }

    method = st.selectbox("Выберите метод", tuple(methods.keys()))

    if method == "Метод дихтомии":
        list_ind = []
        e = np.logspace(-10, 0.1, num=10)
        df = pd.DataFrame(columns=['a', 'b', 'x'])
        df = pd.concat([df, pd.DataFrame({'a': [0], 'b': [1], 'x': [0.5]})], ignore_index=True)

        for ee in e:
            cel, x, df = dichotomy(0, 1, ee, df)
            last_index = df.index[-1] - 1
            list_ind.append(last_index)


        fig, ax = plt.subplots()
        ax.plot(np.log10(e), list_ind)
        ax.set_xlabel('Логарифм точности (log e)')
        ax.set_ylabel('Количество вычислений целевой функции')
        ax.set_title('Зависимость количества вычислений от точности')
        ax.grid(True)
        st.pyplot(fig)

    elif method == "Метод золотого сечения":
        list_ind = []
        e = np.logspace(-10, 0.1, num=10)
        df = pd.DataFrame(columns=['a', 'b', 'L'])
        df = pd.concat([df, pd.DataFrame({'a': [0], 'b': [1], 'L': [1]})], ignore_index=True)

        for ee in e:
            cel, df = golden_sec(0, 1, ee, df)
            last_index = df.index[-1] - 1
            list_ind.append(last_index)

        fig, ax = plt.subplots()
        ax.plot(np.log10(e), list_ind)
        ax.set_xlabel('Логарифм точности (log e)')
        ax.set_ylabel('Количество вычислений целевой функции')
        ax.set_title('Зависимость количества вычислений от точности')
        ax.grid(True)
        st.pyplot(fig)
        return 0

def analysis():
    st.title("Анализ полученных результатов и выводы")
    st.write("Оптимизация - это процесс нахождения оптимального решения для задачи с заданными ограничениями и целями. Он основан на поиске наилучшего значения целевой функции или набора значений переменных в рамках определенных условий.")
    st.write("Задача условной оптимизации - это задача оптимизации, в которой существуют ограничения на значения переменных. Цель состоит в нахождении оптимального решения, которое удовлетворяет заданным условиям ограничений.")
    st.write("Задача безусловной оптимизации - это задача оптимизации, в которой нет ограничений на значения переменных. Цель состоит в нахождении оптимального значения целевой функции при заданных переменных без дополнительных условий.")
    st.write("Задача одномерной оптимизации - это задача оптимизации, в которой присутствует только одна переменная. Цель состоит в нахождении оптимального значения этой переменной, которое минимизирует или максимизирует целевую функцию.")
    st.write("Задача многомерной оптимизации - это задача оптимизации, в которой присутствуют несколько переменных. Цель состоит в нахождении оптимального набора значений этих переменных, которые минимизируют или максимизируют целевую функцию.")
    st.write("Унимодальные функции - это функции, которые имеют только один экстремум (максимум или минимум) на определенном интервале. Например, параболическая функция y = ax^2 + bx + c является унимодальной.")
    st.write("Методы одномерной оптимизации могут быть классифицированы на методы на основе деления интервала (например, метод дихотомического деления, метод золотого сечения), методы на основе сравнения значений функции (например, метод скользящего окна), и методы на основе аппроксимации (например, метод Фибоначчи, метод квадратичной интерполяции).")
    st.write("Задачи одномерного поиска играют важную роль в общей задаче оптимизации, так как они помогают находить оптимальные значения для каждого измерения (переменной) в многомерной оптимизационной задаче.")
    st.write("Метод дихотомического деления основан на разделении интервала посередине и выборе подинтервала, содержащего оптимальное значение целевой функции. Процесс повторяется, пока не будет достигнута заданная точность.")
    st.write("Метод Фибоначчи основан на последовательности чисел Фибоначчи и нахождении оптимальной точки путем сужения интервала по золотому сечению. Он требует меньше итераций по сравнению с методом дихотомического деления.")
    st.write("Метод золотого сечения также использует деление интервала по золотому сечению, где отношение двух полей интервала равно золотому сечению. Это помогает ускорить процесс приближения к оптимальной точке.")
    st.write("Отличия, у этих методов есть, например мне болше всего понравился метод Дихотомии, от проще, быстрее решает, но точнее получается у метода золотого сечения, метод Фибоначчи оказался очень массивным в плане необходимости в железе, по этому не понравился")
    st.write("У метода Дихотомии сложность логарифмическая, а у других методов линейная сложность")

pages = {
    "Цель работы": the_purpose_of_the_work,
    "Исходное задание на исследование": research_assignment,
    "Блок-схемы алгоритмов": block_diagrams,
    "Таблицы с результатами исследований по каждому методу": tables,
    "График зависимости количества вычислений целевой функции от логарифма задаваемой точности E": chart,
    "Анализ полученных результатов и выводы": analysis
}

page = st.sidebar.selectbox("Выберите страницу", tuple(pages.keys()))

pages[page]()
