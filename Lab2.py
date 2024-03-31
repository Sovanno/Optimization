import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import math
from opt_methods import hook_jeeves, gauss_zeidel, rosenbrock, rapid_descent
import pandas as pd

with st.sidebar:
    selected = option_menu(
        menu_title="Hub",
        options = [
                   "Цель работы", 
                   "Исходное задание на исследование", 
                   "Таблицы с результатами исследований по каждому методу", 
                   "График зависимости количества вычислений целевой функции от логарифма задаваемой точности E", 
                   "Анализ полученных результатов и выводы"
                   ],
        default_index=0,
        icons=["info-circle-fill", "info-square", "table", "graph-up", "database-check"],
        styles={
            "nav-link-selected": {'background-color': 'purple'},
        },
        )
    
if selected == "Цель работы":
    st.title(f"{selected}") 
    st.header("Изучение, анализ и программная реализация алгоритмов многомерной оптимизации: метода Гаусса-Зейделя, метода наискорейшего спуска, метода Хука и Дживса, метода Розенброка")

if selected == "Исходное задание на исследование":
    st.title(f"{selected}")
    st.header("Вариант - 9")
    st.subheader("Функция:")
    st.latex("f(x, y) = x^2 + y^2 + 2x + 3y - 6")

if selected == "Таблицы с результатами исследований по каждому методу":

    st.title("Таблицы с результатами исследований по каждому методу")

    methods = {
        "Метод Хука и Дживса": 1,
        "Метод Гаусса-Зейделя": 2,
        "Метод Розенброка": 3,
        "Метод быстрого спуска": 4
    }

    method = st.selectbox("Выберите метод", tuple(methods.keys()))

    if method == "Метод Хука и Дживса":
        x_0 = st.number_input('Введите x0', step = 0.01)
        y_0 = st.number_input('Введите y0', step = 0.01)
        x_start = np.array([x_0, y_0])

        st.write(x_start)

        delta = st.number_input('Введите параметр delta', step = 0.0001)
        st.write(delta)
        epsilon = st.number_input('Введите параметр epsilon', step = 0.0001)
        st.write(epsilon)


        if st.button("Calculate"):
            x, cel, df = hook_jeeves(x_start, delta, epsilon)

            st.write("Значение целевой функции")
            st.latex(cel)
            st.write("Значение x, y")
            st.latex(x)
            st.write("Таблица с итерациями")
            st.write(df)

    if method == "Метод Гаусса-Зейделя":

        x_0 = st.number_input('Введите x0', step = 0.01)
        y_0 = st.number_input('Введите y0', step = 0.01)
        x_start = np.array([x_0, y_0])

        st.write(x_start)

        n = st.number_input('Введите параметр n', step = 1)
        st.write(n)
        epsilon = st.number_input('Введите параметр epsilon', step = 0.0001)
        st.write(epsilon)

        if st.button("Calculate"):
            x, cel, df = gauss_zeidel(x_start, n, epsilon)

            st.write("Значение целевой функции")
            st.latex(cel)
            st.write("Значение x")
            st.latex(x)
            st.write("Таблица с итерациями")
            st.write(df)

    if method == "Метод Розенброка":

        x_0 = st.number_input('Введите x0', step = 0.01)
        y_0 = st.number_input('Введите y0', step = 0.01)
        x_start = np.array([x_0, y_0])

        st.write(x_start)

        alpha = st.number_input('Введите параметр alpha', step = 0.1)
        st.write(alpha)
        epsilon = st.number_input('Введите параметр epsilon', step = 0.0001)
        st.write(epsilon)

        if st.button("Calculate"):
            x, cel, df = rosenbrock(x_start, alpha, epsilon)

            st.write("Значение целевой функции")
            st.latex(cel)
            st.write("Значение x")
            st.latex(x)
            st.write("Таблица с итерациями")
            st.write(df)

    if method == "Метод быстрого спуска":

        x_0 = st.number_input('Введите x0', step = 0.01)
        y_0 = st.number_input('Введите y0', step = 0.01)
        x_start = np.array([x_0, y_0])

        st.write(x_start)

        epsilon = st.number_input('Введите параметр epsilon', step = 0.0001)
        st.write(epsilon)

        if st.button("Calculate"):
            x, cel, df = rapid_descent(x_start, epsilon)

            st.write("Значение целевой функции")
            st.latex(cel)
            st.write("Значение x")
            st.latex(x)
            st.write("Таблица с итерациями")
            st.write(df)

if selected == "График зависимости количества вычислений целевой функции от логарифма задаваемой точности E":
    st.title("Таблицы с результатами исследований по каждому методу")

    methods = {
        "Метод Хука и Дживса": 1,
        "Метод Гаусса-Зейделя": 2,
        "Метод Розенброка": 3,
        "Метод быстрого спуска": 4
    }

    method = st.selectbox("Выберите метод", tuple(methods.keys()))

    if method == "Метод Хука и Дживса":

        list_ind = []
        e = np.logspace(-10, 0.1, num=10)

        x_0 = st.number_input('Введите x0', step=0.01)
        y_0 = st.number_input('Введите y0', step=0.01)
        x_start = np.array([x_0, y_0])

        st.write(x_start)

        delta = st.number_input('Введите параметр delta', step=0.0001)
        st.write(delta)

        if st.button("Calculate"):

            for ee in e:
                x, cel, df = hook_jeeves(x_start, delta, ee)
                last_index = df.index[-1] - 1
                list_ind.append(last_index)

            fig, ax = plt.subplots()
            ax.plot(np.log10(e), list_ind)
            ax.set_xlabel('Логарифм точности (log e)')
            ax.set_ylabel('Количество вычислений целевой функции')
            ax.set_title('Зависимость количества вычислений от точности')
            ax.grid(True)
            st.pyplot(fig)

    if method == "Метод Гаусса-Зейделя":

        list_ind = []
        e = np.logspace(-10, 0.1, num=10)

        x_0 = st.number_input('Введите x0', step=0.01)
        y_0 = st.number_input('Введите y0', step=0.01)
        x_start = np.array([x_0, y_0])

        st.write(x_start)

        n = st.number_input('Введите параметр n', step=1)
        st.write(n)

        if st.button("Calculate"):

            for ee in e:
                x, cel, df = gauss_zeidel(x_start, n, ee)
                last_index = df.index[-1] - 1
                list_ind.append(last_index)

            fig, ax = plt.subplots()
            ax.plot(np.log10(e), list_ind)
            ax.set_xlabel('Логарифм точности (log e)')
            ax.set_ylabel('Количество вычислений целевой функции')
            ax.set_title('Зависимость количества вычислений от точности')
            ax.grid(True)
            st.pyplot(fig)

    if method == "Метод Розенброка":

        list_ind = []
        e = np.logspace(-10, 0.1, num=10)

        x_0 = st.number_input('Введите x0', step=0.01)
        y_0 = st.number_input('Введите y0', step=0.01)
        x_start = np.array([x_0, y_0])

        st.write(x_start)

        alpha = st.number_input('Введите параметр alpha', step=0.1)
        st.write(alpha)

        if st.button("Calculate"):

            for ee in e:
                x, cel, df = rosenbrock(x_start, alpha, ee)
                last_index = df.index[-1] - 1
                list_ind.append(last_index)

            fig, ax = plt.subplots()
            ax.plot(np.log10(e), list_ind)
            ax.set_xlabel('Логарифм точности (log e)')
            ax.set_ylabel('Количество вычислений целевой функции')
            ax.set_title('Зависимость количества вычислений от точности')
            ax.grid(True)
            st.pyplot(fig)

    if method == "Метод быстрого спуска":

        list_ind = []
        e = np.logspace(-10, 0.1, num=10)

        x_0 = st.number_input('Введите x0', step=0.01)
        y_0 = st.number_input('Введите y0', step=0.01)
        x_start = np.array([x_0, y_0])

        st.write(x_start)

        if st.button("Calculate"):

            for ee in e:
                x, cel, df = rapid_descent(x_start, ee)
                last_index = df.index[-1] - 1
                list_ind.append(last_index)

            fig, ax = plt.subplots()
            ax.plot(np.log10(e), list_ind)
            ax.set_xlabel('Логарифм точности (log e)')
            ax.set_ylabel('Количество вычислений целевой функции')
            ax.set_title('Зависимость количества вычислений от точности')
            ax.grid(True)
            st.pyplot(fig)

if selected == "Анализ полученных результатов и выводы":
    st.title(f"{selected}")
    st.subheader("В ходе данной лабораторной работы были реализованы и исследованы методы многомерной оптимизации: метод Хука-Дживса, метод Гаусса-Зейделя, метод Розенброка и метод наискорейшего спуска.")
    st.subheader("Метод Хука-Дживса, основанный на поиске по образцу, показал хорошие результаты. Однако требует большого числа итераций, что может замедлить процесс оптимизации.")
    st.subheader("Метод Гаусса-Зейделя также оказался эффективным. Он способен находить локальные минимумы функции и быстро сходится к решению. Однако возможны ситуации, когда метод расходится.")
    st.subheader("Метод Розенброка, применяемый для решения нелинейных задач оптимизации, показал себя с достоинством. Он является гибким и универсальным методом, но для некоторых функций может сходиться медленно.")
    st.subheader("Однако метод наискорейшего спуска проявил себя как наиболее эффективный в данной лабораторной работе. Он позволяет быстро сходиться к локальному минимуму функции, за счет выбора оптимального направления спуска. Этим методом удалось достичь оптимума с наименьшим числом итераций и затратить наименьшее время в сравнении с остальными методами.")
    st.subheader("Таким образом, в данной работе было реализовано несколько методов многомерной оптимизации, каждый из которых имеет свои достоинства и недостатки. Однако метод наискорейшего спуска оказался наиболее эффективным, что подтверждает его применимость при решении задач оптимизации.")
