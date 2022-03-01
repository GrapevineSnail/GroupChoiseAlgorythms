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

def F_zeroing(M,S):
    n = len(M)#квадратная матрица
    for i in range(n):
        for j in range(n):
            if i == j: M[i][j] = 0
            elif M[i][j] != 0:
                M[i][j] = M[i][j].expand()
                paths = str(M[i][j]).split('+')
                paths = [p.replace(" ","") for p in paths]
                for k in range(len(paths)):
                    if (str(S[i]) or str(S[j])) in paths[k]:
                        paths[k] = ""
                    else:
                        for s in S:
                            if (paths[k].count(str(s)) > 1) or \
                               str(s**2) in paths[k]:
                                paths[k] = ""
                paths = [p for p in paths if p != ""]
                M[i][j] = 0
                for p in paths:
                    sp = p.split('*')
                    sp = [sympy.Symbol(s,commutative=False) for s in sp]
                    expr = 1
                    for s in sp:
                        expr *= s
                    M[i][j] += expr
    return M

def hamiltonian_paths_through_matrix_degree(Weights_matrix):
    n = len(Weights_matrix)
    Q = [[ 0 if (abs(Weights_matrix[i][j]) == math.inf or
                 Weights_matrix[i][j] == 0)
           else 1
           for j in range(n)]
         for i in range(n)]
    Q = np.array(Q)
    for i in range(n):
        Q[i][i] = 0#зануление диагонали
    Q_paths_cnt = np.linalg.matrix_power(Q, n-1)
    print('Paths count between vertices\n',Q_paths_cnt)
    paths_cnt = 0
    for i in range(n):
        for j in range(n):
            if i!=j:
                paths_cnt += Q_paths_cnt[i][j]
    print('Total paths count:',paths_cnt)

##    sym2ind = {'a{0}'.format(i):i for i in range(n)}
    sym2ind = {index2letter(i).lower():i for i in range(n)}
    ind2sym = dict(zip(sym2ind.values(), sym2ind.keys()))
    
    H = np.array([[sympy.Symbol(ind2sym[j], commutative=False)
                    if Q[i][j] == 1 else int(0)
                    for j in range(n)]
                  for i in range(n)])
    #print("H\n",H)
    print("Q\n",Q,"\n")
    S = [sympy.Symbol(ind2sym[j],commutative=False) for j in range(n)]
    #print("Symbols",S)
    for i in range(2,n):
        Q_quote = H.dot(Q)
        print("Q_quote{0}\n".format(i),Q_quote,"\n")
        Q = F_zeroing(Q_quote,S)
        print("Q{0}\n".format(i),Q,"\n")
    
    def symbolic_path2index_path(string):
        string.replace(" ","")
        path = string.split('*')
        for i in range(len(path)):
            path[i] = sym2ind[path[i]]
        return path
    
    Paths_matrix = [[ [] for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if Q[i][j] != 0:
                some_paths = str(Q[i][j]).split('+')
                for path in some_paths:
                    Paths_matrix[i][j].append(
                        [i] + symbolic_path2index_path(path) + [j])
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
    Weights_matrix = [[0, 0, 1, 0],
                        [1, 0, 0, 1],
                        [0, 1, 0, 0],
                        [1, 0, 1, 0]]
##    Weights_matrix = [[0, 1, 1],
##                        [0, 0, 1],
##                        [0, 0, 0]]
##    Weights_matrix = [[0, 0, 26, 30, 0],
##                      [25, 0, 0, 33, 0],
##                      [0, 29, 0, 0, 24],
##                      [0, 0, 28, 0, 0],
##                      [23, 27, 0, 31, 0]]
##    Weights_matrix = [[0, 1, 1],
##                        [1, 0, 1],
##                        [1, 1, 0]]
##    Weights_matrix = [[1, 1, 1],
##                        [1, 1, 1],
##                        [1, 1, 1]]
##    Weights_matrix = [[1, 1, inf, inf],
##                        [1, 1, 1, inf],
##                        [inf, 1, 1, 1],
##                        [inf, inf, 1, 1]]
##    Weights_matrix = [[1, 0, 0, 0],
##                        [0, 1, 0, 0],
##                        [0, 0, 1, 0],
##                        [0, 0, 0, 1]]
    
##    Weights_matrix = [[0, 0, 0, 1, 1],
##                      [1, 0, 1, 0, 1],
##                      [1, 0, 0, 1, 0],
##                      [0, 1, 0, 0, 0],
##                      [0, 0, 0, 0, 0]]
    paths_matrix = hamiltonian_paths_through_matrix_degree(Weights_matrix)
    n = len(Weights_matrix)
    for i in range(n):
        for j in range(n):
            paths_matrix[i][j] = many_vertices_lists2symbolic_paths(
                paths_matrix[i][j])
##            print("{0}->{1} paths: {2}".format(
##                index2letter(i),index2letter(j),paths))
        print(paths_matrix[i])
##    print(type(symbol_mult('a',"b")))
##    print(type(symbol_mult(1,2)))
##    print(type(symbol_mult('1',9)))
##    print(type(symbol_mult('2','2')))
##    print(symbol_mult('22',1))
main()
