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

def graph_matrix2adjacency_list(Weights_matrix):
    n = len(Weights_matrix)#квадратная матрица
    Adjacency_list = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if abs(Weights_matrix[i][j]) not in [0, math.inf, float("inf")]:
                Adjacency_list[i].append(j)
    return Adjacency_list

def all_simple_paths_betweenAB(Adjacency_list, idA, idB):
        #simple - без циклов
        n = len(Adjacency_list)
        Paths = []
        cur_path = []
        visited = [False for i in range(n)]
        def enter_in_vertex(v):
            cur_path.append(v)
            visited[v] = True#зашли в вершину
        def leave_vertex(v):
            cur_path.pop()
            visited[v] = False#вышли - поднялись выше по цепочке
        def dfs(v):#на каких случаях dfs лучше чем уоршалл??
            enter_in_vertex(v)
            if v == idB:#нашли путь
                Paths.append(cur_path.copy())
            else:
                for next_v in Adjacency_list[v]:
                    if visited[next_v] == False:
                        dfs(next_v)#идём туда, куда ещё не входили
            leave_vertex(v)
            return 0
        dfs(idA)
        return Paths

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
    Weights_matrix = [[0, 0, 26, 30, 0],
                      [25, 0, 0, 33, 0],
                      [0, 29, 0, 0, 24],
                      [0, 0, 28, 0, 0],
                      [23, 27, 0, 31, 0]]
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
    n = len(Weights_matrix)
    for k in range(1,n+1):
        paths_matrix = [[None for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                paths = all_simple_paths_betweenAB(
                    graph_matrix2adjacency_list(Weights_matrix),i,j)
                paths = [p for p in paths if len(p) == k]
                paths_matrix[i][j] = many_vertices_lists2symbolic_paths(paths)
    ##            print("{0}->{1} paths: {2}".format(
    ##                index2letter(i),index2letter(j),paths))
            print(paths_matrix[i])
        print()
main()
