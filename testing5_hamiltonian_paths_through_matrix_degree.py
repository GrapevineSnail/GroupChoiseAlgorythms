import math
import matplotlib.pyplot as plot
import matplotlib.patches
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import networkx as nx
from networkx import *
import numpy as np
import sympy
import inspect


def Hamiltonian_paths_through_matrix_degree(Weights_matrix):

    def is_cycle_in_sym_path(sym_path):
        for sym in sym_path.args:
            if sym.args != ():  # (s**2).args -> (s,2); (s).args -> ()
                return True
        if len(np.unique(list(map(str, sym_path.args)))) != len(sym_path.args):
            return True
        return False

    def F_zeroing(M, symbols):
        n = len(M)  # квадратная матрица
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i][j] = 0
                elif M[i][j] != 0:
                    M[i][j] = M[i][j].expand()
                    if "+" in str(M[i][j]):
                        paths = list(M[i][j].args)
                    else:
                        paths = [M[i][j]]
                    for k in range(len(paths)):
                        if is_cycle_in_sym_path(symbols[i]*paths[k]*symbols[j]):
                            paths[k] = ""
                    paths = [p for p in paths if p != ""]
                    M[i][j] = sum(paths)
        return M

    n = len(Weights_matrix)
    # Q = [[0 if (abs(Weights_matrix[i][j]) == math.inf or
    #             Weights_matrix[i][j] == 0)
    Q = [[0 if (abs(Weights_matrix[i][j]) == math.inf)
          else 1
          for j in range(n)]
         for i in range(n)]
    Q = np.array(Q)
    for i in range(n):
        Q[i][i] = 0  # зануление диагонали

    Q_paths_cnt = np.linalg.matrix_power(
        Q, n-1)  # Paths count between vertices
    print('Paths count between vertices\n', Q_paths_cnt)

    paths_cnt = 0  # Total paths count
    for i in range(n):
        for j in range(n):
            if i != j:
                paths_cnt += Q_paths_cnt[i][j]
    print('Total non-circular paths count:', paths_cnt,'\n')

    sym2ind = {
        sympy.Symbol('a{0}'.format(i), commutative=False):i 
        for i in range(n)}
    ind2sym = dict(zip(sym2ind.values(), sym2ind.keys()))

    H = np.array([[ ind2sym[j] if Q[i][j] == 1 else int(0)
        for j in range(n)] 
        for i in range(n)])
    
    print("H\n", H, "\n")
    print("Q\n", Q, "\n")
    for i in range(2, n):
        Q_quote = H.dot(Q)
        print("Q_quote{0}\n".format(i), Q_quote, "\n")
        Q = F_zeroing(Q_quote, list(sym2ind.keys()))
        print("Q{0}\n".format(i), Q, "\n")

    Paths_matrix = [[[] for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if Q[i][j] != 0:
                if n > 2:
                    if "+" in str(Q[i][j]):
                        some_sym_paths = list(Q[i][j].args)
                    else:
                        some_sym_paths = [ Q[i][j] ]

                    for sym_path in some_sym_paths:
                        Paths_matrix[i][j].append(
                            list(map(lambda x: sym2ind[x], 
                            (ind2sym[i]*sym_path*ind2sym[j]).args))
                            )
                elif n == 2:
                    Paths_matrix[i][j].append([i] + [j])
    return Paths_matrix


def index2letter(index):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[index]


def vertices_list2symbolic_path(vertices_of_path):
    text = ""
    for v in vertices_of_path:
        text += index2letter(v) + ">"
    text = text[:-1]
    return text


def many_vertices_lists2symbolic_paths(paths):
    text = ""
    for p in paths:
        text += vertices_list2symbolic_path(p) + " + "
    text = text[:-3]
    return text


def main():
    inf = math.inf
    Weights_matrix = [[inf, inf, 1, inf],
                      [1, inf, inf, 1],
                      [inf, 1, inf, inf],
                      [1, inf, 1, inf]]
    # ans:
    # acbd,
    # bdac,
    # cbda,
    # dcba, dacb
    
    # Weights_matrix = [[inf]]
    
    # Weights_matrix = [[inf, 1],
    #                   [inf, 0]]
    #ans: ab

    # Weights_matrix = [[inf, 1],
    #                   [1, 0]]
    #ans: ab, ba

    # Weights_matrix = [[inf, 1, 1],
    #                   [inf, inf, 1],
    #                   [inf, inf, 0]]
    #ans: abc

    # Weights_matrix = [[inf, inf, 26, 30, inf],
    #                   [25, inf, inf, 33, inf],
    #                   [inf, 29, inf, inf, 24],
    #                   [inf, inf, 28, inf, inf],
    #                   [23, 27, inf, 31, inf]]
    # ans:
    # adceb,acebd,
    # bdcea,baced,badce,
    # cebad,
    # dceba,
    # edcba,eadcb,ebadc,eacbd,

    # Weights_matrix = [[inf, 1, 1],
    #                   [1, inf, 1],
    #                   [1, 1, inf]]
    # ans:
    # ['', 'A>C>B', 'A>B>C']
    # ['B>C>A', '', 'B>A>C']
    # ['C>B>A', 'C>A>B', '']

    # Weights_matrix = [[1, 1, 1],
    #                   [1, 1, 1],
    #                   [1, 1, 1]]
    # ans:
    # ['', 'A>C>B', 'A>B>C']
    # ['B>C>A', '', 'B>A>C']
    # ['C>B>A', 'C>A>B', '']

    # Weights_matrix = [[1, 1, inf, inf],
    #                   [1, 1, 1, inf],
    #                   [inf, 1, 1, 1],
    #                   [inf, inf, 1, 1]]
    # ans: abcd, dcba

    # Weights_matrix = [[1, inf, inf, inf],
    #                   [inf, 1, inf, inf],
    #                   [inf, inf, 1, inf],
    #                   [inf, inf, inf, 1]]
    # ans: ---

    # Weights_matrix = [[inf, inf, inf, 1, 1],
    #                 [1, inf, 1, inf, 1],
    #                 [1, inf, inf, 1, inf],
    #                 [inf, 1, inf, inf, inf],
    #                 [inf, inf, inf, inf, inf]]
    # ans: cadbe, cdbae, dbcae

    paths_matrix = Hamiltonian_paths_through_matrix_degree(Weights_matrix)
    n = len(Weights_matrix)
    for i in range(n):
        for j in range(n):
            paths_matrix[i][j] = many_vertices_lists2symbolic_paths(
                paths_matrix[i][j])
        print(paths_matrix[i])


# print(type(symbol_mult('a',"b")))
# print(type(symbol_mult(1,2)))
# print(type(symbol_mult('1',9)))
# print(type(symbol_mult('2','2')))
# print(symbol_mult('22',1))
main()
