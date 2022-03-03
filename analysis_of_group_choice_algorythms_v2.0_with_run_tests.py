import math
import matplotlib.pyplot as plot
import matplotlib.patches
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import networkx as nx
from networkx import *
import numpy as np
import sympy
import inspect
import itertools
import os
import datetime

def visualize_graph(Weights_matrix, path):
    plot.clf()
    graph = nx.DiGraph()
    for i in range(n):
        graph.add_node(index2symbol(i, n-1))
        for j in range(n):
            if (i != j and abs(Weights_matrix[i][j]) != math.inf):
                graph.add_edge(index2symbol(i, n-1),
                               index2symbol(j, n-1), 
                               weight = Weights_matrix[i][j])
    indicating_color = '#FF8A00'
    node_edge_color_map = []
##    if path not in [None,[]]:
##        start_node = index2symbol(path[0], n-1)
##        finish_node = index2symbol(path[-1], n-1)
    for node in graph:
        node_edge_color_map.append('black')
##        if path not in [None,[]]:
##            if node in [start_node, finish_node]:
##                node_edge_color_map.append(indicating_color)
##            else:
##                node_edge_color_map.append('black')
    edges = [(u,v) for (u,v) in graph.edges(data = False)]
    edges_cnt = len(edges)
    edges_color_map = ['black' for i in range(edges_cnt)]
    edges_width_map = [1 for i in range(edges_cnt)]
    if path not in [None,[]]:
        path_edges = [(index2symbol(path[i], n-1),
                       index2symbol(path[i+1], n-1))
                       for i in range(len(path)-1)]
        for i in range(edges_cnt):
            if edges[i] in path_edges:
                edges_color_map[i] = indicating_color
                edges_width_map[i] = 2
                path_edges.pop(path_edges.index(edges[i]))
                      
    position = nx.circular_layout(graph) #position = nx.planar_layout(graph)
    nx.draw_networkx_nodes(
        graph,
        pos=position,
        node_color=button_background,
        node_shape='h',
        node_size=700,
        edgecolors=node_edge_color_map)
    nx.draw_networkx_labels(
        graph,
        pos=position,
        font_size=11,
        font_color='black',
        font_family='serif',
        font_weight="bold")
    nx.draw_networkx_edges(
        graph,
        pos=position,
        connectionstyle='arc3, rad = 0.05',
        edgelist = graph.edges(),
        width=edges_width_map,
        edge_color=edges_color_map,
        alpha=1.0,
        arrowstyle=matplotlib.patches.ArrowStyle(
            "-|>", head_length=0.8, head_width=0.3),
        arrowsize=12,
        node_size=700,
        node_shape='h',
        min_source_margin=0,
        min_target_margin=5)
    nx.draw_networkx_edge_labels(
        graph,
        pos=position,
        label_pos=0.2,
        font_size=8,
        edge_labels=nx.get_edge_attributes(graph, "weight"),
        bbox=dict(boxstyle='round', ec='#00000080', fc='#FFFFFF80'),
        alpha=1.0,
        rotate=True)
    plot.show()

###
def inds_of_max(M):
    max_ = -math.inf
    for i in range(len(M)):
        for j in range(len(M[i])):
            for elem in M[i][j]:
                if elem > max_: max_ = elem
    inds = []
    for i in range(len(M)):
        for j in range(len(M[i])):
            for k in range(len(M[i][j])):
                if M[i][j][k] == max_: inds.append((i,j,k))
    return inds, max_

def Hamming_distance(R1,R2):#вход: матрицы одинаковой размерности только из 1 и 0
    R1 = np.array(R1)
    R2 = np.array(R2)
    iscorrect1 = np.all((R1 == 0) + (R1 == 1))
    iscorrect2 = np.all((R2 == 0) + (R2 == 1))
    if R1.shape != R2.shape or not iscorrect1 or not iscorrect2:
        raise ValueError()
        return None
    disparity = list(map(abs,R1-R2))
    return sum(sum(disparity))
def sum_of_distances(R, ListOfOtherR):
    #вторым параметром - список матриц смежности
    return sum([Hamming_distance(R,Ri) for Ri in ListOfOtherR])

def weights_of_path(vertices_list, Weights_matrix):
    l = len(vertices_list)
    if l == 0 :
        weights_list = []
    if l == 1:#путь (a) "ничего не делать" - нет пути, так как нет петли
        weights_list = []
    if l > 1:#включает и путь-петлю (a,a)
        weights_list = [Weights_matrix[vertices_list[i]][vertices_list[i+1]]
                        for i in range(l-1)]
    return weights_list

def path_weight(vertices_list, Weights_matrix):
    return sum(weights_of_path(vertices_list, Weights_matrix))
def path_strength(vertices_list, Weights_matrix):
    W = weights_of_path(vertices_list, Weights_matrix)
    if W == []:
        return -math.inf
    return min(W)
def path_sum_distance(vertices_list, R_list):
    R = make_single_R_profile_matrix(vertices_list)
    return sum_of_distances(R, R_list)

def Paths_weights_matrix(Paths_matrix, Weights_matrix):
    return [ [
        [path_weight(path, Weights_matrix)
         for path in Paths_matrix[i][j]]
        for j in range(n)] for i in range(n)]
def Paths_strengths_matrix(Paths_matrix, Weights_matrix):
    return [ [
        [path_strength(path, Weights_matrix)
         for path in Paths_matrix[i][j]]
        for j in range(n)] for i in range(n)]
def Paths_sum_distances_matrix(Paths_matrix, R_list):
    return [ [
        [path_sum_distance(path, R_list)
         for path in Paths_matrix[i][j]]
        for j in range(n)] for i in range(n)]

def Paths_weights_list(Paths_list, Weights_matrix):
    return [ path_weight(path, Weights_matrix)
             for path in Paths_list]
def Paths_strengths_list(Paths_list, Weights_matrix):
    return [ path_strength(path, Weights_matrix)
             for path in Paths_list]
def Paths_sum_distances_list(Paths_list, R_list):
    return [ path_sum_distance(path, R_list)
             for path in Paths_list]

def Paths_matrix(Rankings_list):
    PM = [[ [] for j in range(n)] for i in range(n)]
    for ranking in Ranking_list:
        i = ranking[0]
        j = ranking[-1]
        PM[i][j].append(ranking)
    return PM
def Rankings_list(Paths_matrix):
    RL = []
    if n == 1:
        RL.append([0])
    else:
        for i in range(n):
            for j in range(n):
                for paths in Paths_matrix[i][j]:
                    if type(paths) == list: RL.append(paths)
    return RL

def matrix2adjacency_list(Weights_matrix):
    n = len(Weights_matrix)#квадратная матрица
    Adjacency_list = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and \
            abs(Weights_matrix[i][j]) not in [math.inf, float("inf")]:
                Adjacency_list[i].append(j)
    return Adjacency_list

def make_aggregated_R_profile_matrix(weight_C_matrix):#adjacency_matrix
    R = [[ 1 if
           abs(weight_C_matrix[i][j]) != math.inf
           else 0
           for j in range(n)] for i in range(n)]
    return R
def make_weight_C_matrix(summarized_P_matrix):
    C = [[ -math.inf if
           i == j or summarized_P_matrix[i][j] < summarized_P_matrix[j][i]
           else summarized_P_matrix[i][j] - summarized_P_matrix[j][i]
           for j in range(n)] for i in range(n)]
    return C
def make_summarized_P_matrix(list_of_R_matrices):
    P = np.array([[0 for j in range(n)] for i in range(n)])
    for Rj in list_of_R_matrices:
        P += np.array(Rj)
    return P.tolist()
def make_single_R_profile_matrix(single_profile):
    profile = np.array(single_profile)
    Rj = [[0 for j in range(n)] for i in range(n)]
    for i in range(len(profile)):
        p = profile[i]
        for p2 in profile[i+1:]:
            Rj[p][p2] = 1
    return Rj#adjacency_matrix
def Make_used_matrices(list_of_profiles):#аргумент: матрицы n*n, их m штук
    R_list = [make_single_R_profile_matrix(p) for p in list_of_profiles]
    P = make_summarized_P_matrix(R_list)
    C = make_weight_C_matrix(P)
    R = make_aggregated_R_profile_matrix(C)
    return (R_list, P, C, R)
###

def All_various_rankings():
    def permutations_of_elements(elements):
        return [ list(p) for p in itertools.permutations(elements) ]
    Rankings = []
    if n == 1:
        Rankings.append([0])
    else:
        for i in range(n):
            for j in range(n):
                if i != j:
                    if n > 2:
                        middle_vetrices = [v for v in range(n) if v!=i and v!=j]
                        for p in permutations_of_elements(middle_vetrices):
                            Rankings.append([i] + p + [j])
                    elif n == 2:
                        Rankings.append([i] + [j])
    return Rankings

def Linear_medians(R_list):
    All_rankings = All_various_rankings()
    Distances = Paths_sum_distances_list(All_rankings, R_list)
    Dmin = min(Distances)
    Rankings_list = [ All_rankings[i] for i in range(len(All_rankings))
                      if Distances[i] == Dmin ]
    return Rankings_list, Dmin

def hamiltonian_paths_through_matrix_degree(Weights_matrix):
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
    n = len(Weights_matrix)
    Q = [[ 0 if (abs(Weights_matrix[i][j]) == math.inf or
                 Weights_matrix[i][j] == 0)
           else 1
           for j in range(n)]
         for i in range(n)]
    Q = np.array(Q)
    for i in range(n):
        Q[i][i] = 0#зануление диагонали
    Q_paths_cnt = np.linalg.matrix_power(Q, n-1)#Paths count between vertices
    paths_cnt = 0#Total paths count
    for i in range(n):
        for j in range(n):
            if i!=j:
                paths_cnt += Q_paths_cnt[i][j]
                
    sym2ind = {'a{0}'.format(i):i for i in range(n)}
    ind2sym = dict(zip(sym2ind.values(), sym2ind.keys()))
    
    H = np.array([[sympy.Symbol(ind2sym[j], commutative=False)
                    if Q[i][j] == 1 else int(0)
                    for j in range(n)]
                  for i in range(n)])
    S = [sympy.Symbol(ind2sym[j],commutative=False) for j in range(n)]
    for i in range(2,n):
        Q_quote = H.dot(Q)
        Q = F_zeroing(Q_quote,S)

    def symbolic_path2index_path(string):
        path = string.split('*')
        for i in range(len(path)):
            path[i] = sym2ind[path[i]]
        return path
    
    Paths_matrix = [[ [] for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if Q[i][j] != 0:
                if n > 2:
                    some_paths = str(Q[i][j]).replace(" ","").split('+')
                    for path in some_paths:
                        Paths_matrix[i][j].append(
                            [i] + symbolic_path2index_path(path) + [j])
                elif n == 2:
                    Paths_matrix[i][j].append([i] + [j])
    return Paths_matrix

def HP_max_length(Weights_matrix):
    Paths_matrix = hamiltonian_paths_through_matrix_degree(Weights_matrix)
    PL = Paths_weights_matrix(Paths_matrix, Weights_matrix)
    inds, max_length = inds_of_max(PL)
    #матрица длиннейших путей - longest paths (LP)
    LP_matrix = [[[] for j in range(n)] for i in range(n)]
    for (i,j,k) in inds:
        LP_matrix[i][j].append(Paths_matrix[i][j][k])
    return Rankings_list(LP_matrix)

def HP_max_strength(Weights_matrix):
    Paths_matrix = hamiltonian_paths_through_matrix_degree(Weights_matrix)
    PS = Paths_strengths_matrix(Paths_matrix, Weights_matrix)
    inds, max_strength= inds_of_max(PS)
    #матрица сильнейших путей - strongest paths (SP)
    SP_matrix = [[[] for j in range(n)] for i in range(n)]
    for (i,j,k) in inds:
        SP_matrix[i][j].append(Paths_matrix[i][j][k])
    return Rankings_list(SP_matrix)

def Schulze_method(Weights_matrix):
##    def is_chain_correct_Schulze(vertices_list, Weights_matrix):
##        #все звенья пути должны удовлетворять: W[C(i),C(i+1)]>W[C(i+1),C(i)]
##        l = len(vertices_list)
##        for i in range(l-1):
##            if Weights_matrix[vertices_list[i]][vertices_list[i+1]] <= \
##               Weights_matrix[vertices_list[i+1]][vertices_list[i]]:
##                return False
##        return True
    def all_simple_paths_betweenAB(Adjacency_list, idA, idB):
        #simple - без циклов
        Paths = []
        if idA != idB:
            n = len(Adjacency_list)
            cur_path = []
            visited = [False for i in range(n)]
            def enter_in_vertex(v):
                cur_path.append(v)
                visited[v] = True#зашли в вершину
            def leave_vertex(v):
                cur_path.pop()
                visited[v] = False#вышли - поднялись выше по цепочке
            def dfs(v):
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
    def strongest_paths_betweenAB(Weights_matrix, 
                                  All_paths_betweenAB, idA, idB):
        Paths = [path for path in All_paths_betweenAB]
##                 if is_chain_correct_Schulze(path, Weights_matrix)]
        Weights = [weights_of_path(path, Weights_matrix) for path in Paths]
        Strongest_paths = []
        max_strength = -math.inf
        l = len(Paths)
        if l > 0:
            strengths = []
            for i in range(l):
                strengths.append(min(Weights[i]))
            max_strength = max(strengths)
            for i in range(l):
                if max_strength == strengths[i]:
                    Strongest_paths.append(Paths[i])
        return (max_strength, Strongest_paths)
    
    Adjacency_list = matrix2adjacency_list(Weights_matrix)
    Paths_matrix = [[all_simple_paths_betweenAB(Adjacency_list, i, j)
                     for j in range(n)] for i in range(n)]
    #матрица сильнейших путей - strongest paths (SP)
    SP_matrix = [[0 for j in range(n)] for i in range(n)]
    #матрица сил сильнейших путей - Power
    Power = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            strength, S_paths = strongest_paths_betweenAB(
                Weights_matrix, Paths_matrix[i][j], i, j)
            SP_matrix[i][j] = S_paths
            Power[i][j] = strength
    def compare_MORETHAN(A,B):
        if Power[A][B] > Power[B][A]:
            return True #побеждает A. (A > B) т.е. morethan
        return False
    def compare_EQUIV(A,B):
        if not compare_MORETHAN(A,B) and not compare_MORETHAN(B,A):
            return True
        return False
    def is_winner(A):
        for B in range(n):
            if B != A and Power[A][B] < Power[B][A]:
                return False
        return True
    #set of winners
    S = [ A for A in range(n) if is_winner(A)]
    #матрица строгих сравнений A>B <=> comparsion[A][B]=1
    #должна быть транзитивна A>B and B>C => A>C
    comparsion = [[ 1 if compare_MORETHAN(i,j) else 0
                    for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and Power[i][j] == 0:
                for k in range(n):
                    if compare_MORETHAN(i,k) and compare_MORETHAN(k,j):
                        comparsion[i][j] = 1
    ranking = []
    if sum([sum(c) for c in comparsion]) == (n**2 - n)/2:
        ranking = [i for i in range(n)]#результирующее ранжирование по Шульце
        ranking.reverse()
        for i in range(n):
            j = i
            while j > 0 and comparsion[ranking[j]][ranking[j-1]]:
                ranking[j], ranking[j-1] = ranking[j-1], ranking[j]
                j -= 1           
##    ranking = [i for i in range(n)]#результирующее ранжирование по Шульце
##    ranking.reverse()
##    for i in range(n):
##        j = i
##        while j > 0 and \
##              (compare_MORETHAN(ranking[j],ranking[j-1]) or \
##               compare_EQUIV(ranking[j],ranking[j-1])):
##            ranking[j], ranking[j-1] = ranking[j-1], ranking[j]
##            j -= 1
    return S, ranking

def Execute_algorythms(list_of_profiles):
    Methods_names = ['HP_max_length',
                     'HP_max_strength',
                     'Schulze_method',
                     'Linear_medians',
                     'All_rankings']
    Methods_frames = [frame_table_HP_max_length,
                      frame_table_HP_max_strength,
                      frame_table_Schulze_method,
                      frame_table_Linear_medians,
                      frame_table_All_rankings]
    Methods_checkbuttons = [ _HP_max_length.get(),
                             _HP_max_strength.get(),
                             _Schulze_method.get(),
                             _Linear_medians.get(),
                             _All_rankings.get() ]
    Frames = {Methods_names[i]:Methods_frames[i]
                      for i in range(len(Methods_names))}
    Checkbuttons = {Methods_names[i]:Methods_checkbuttons[i]
                      for i in range(len(Methods_names))}
    
    R_list, P, C, R = Make_used_matrices(list_of_profiles)
    Params = {'R_list':R_list, 'P':P, 'C':C, 'R':R, 'median_dist':None,
              'Schulze_ranking':None, 'Schulze_winners':None}
    MRankings = {}
    if sum(Methods_checkbuttons):
        _, Params['median_dist'] = Linear_medians(R_list)
        if Checkbuttons['HP_max_length']:
            name = 'HP_max_length'
            MRankings[name] = HP_max_length(C)
        if Checkbuttons['HP_max_strength']:
            name = 'HP_max_strength'
            MRankings[name] = HP_max_strength(C)
        if Checkbuttons['Schulze_method']:
            name = 'Schulze_method'
            Params['Schulze_winners'], Params['Schulze_ranking'] = \
                                       Schulze_method(C)
            MRankings[name] = [Params['Schulze_ranking']]
        if Checkbuttons['Linear_medians']:
            name = 'Linear_medians'
            MRankings[name], Params['median_dist'] = \
                                          Linear_medians(R_list)
        if Checkbuttons['All_rankings']:
            name = 'All_rankings'
            MRankings[name] = All_various_rankings()
        Intersect = []
        if len(MRankings) > 1:
            div = "*"
            def list2symbolic_string(l):
                s = ""
                if l == []:
                    return s
                for e in l:
                    s += str(e) + div
                return s[:-len(div)]
            def symbolic_string2list(s):
                l = s.split(div)
                l = [int(e) for e in l]
                return l
            Sets = [ set( list2symbolic_string(l) for l in rankings )
                     for rankings in MRankings.values()
                     if rankings not in ([],[[]]) ]
            Intersect = set.intersection(*Sets)
            Intersect = [ symbolic_string2list(s) for s in Intersect]
##        print_result_rankings(Frames, Params, MRankings, Intersect)
##        visualize_graph(C, None)
        if Params['Schulze_ranking'] == None:
            if Intersect != []:
                raise IntersectIsFound
        else:
            if Params['Schulze_ranking'] != []:
                if Intersect != []:
                    raise IntersectIsFound
                raise SchulzeRankingExists
                
    else:
        messagebox.showwarning("", "Выберите метод")



###
def index2symbol(index, max_index):
    if max_index > 25:
        return str(index + 1)
    else:
        symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return symbols[index]

def symbol2index(symbol):
    if symbol.isalpha() and len(symbol) == 1:
        symbol = symbol.upper()
        symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        correspondence = {symbols[i]:i for i in range(len(symbols))}
        return int(correspondence[symbol])
    elif symbol.isdigit() and int(symbol) > 0:
        return int(symbol) - 1
    else:
        raise ValueError("Неверный символ")
    
def index_path2symbolic_path(indices):
    text = ""
    div = ">"
    max_index = max(indices)
    for i in indices:
        text += index2symbol(i, max_index) + div
    text = text[:-len(div)]
    return text

def list2string(a):
    text = ""
    div = ", "
    for elem in a:
        text += str(elem) + div
    return text[:-len(div)]  

def matrix2string(Matrix):#для удобства печати матриц
    n = len(Matrix)
    m = len(Matrix[0])
    max_widths = [3 for j in range(m)]
    for i in range(n):
        for j in range(m):
            if(len(str(Matrix[i][j])) > max_widths[j]):
                max_widths[j] = len(str(Matrix[i][j]))
    string = ""
    for i in range(n):
        for j in range(m):
            string += "[{:{fill}{align}{width}}]".format(
                Matrix[i][j], 
                fill='_', 
                align='^', 
                width= max_widths[j]+2 if m>5 else max(max_widths)+2)
        string += '\n'
    string = string[:-1]
    return string
###

def print_result_rankings(Frames, Params,
                          Methods_rankings, Mutual_rankings):
    R_list = Params['R_list']
    Weights_matrix = Params['C']
    text = "Минимальное суммарное расстояние Хэмминга\n" + \
           "для мажоритарного графа: {}".format(
               sum_of_distances(Params['R'], R_list))
    label_output['text'] = text
    for name in Methods_rankings:
        table_frame = Frames[name]
        Result_rankings = Methods_rankings[name]
        if Result_rankings not in [None, [], [[]]]:
            Lengths = Paths_weights_list(Result_rankings, Weights_matrix)
            Strengths = Paths_strengths_list(Result_rankings, Weights_matrix)
            Distances = Paths_sum_distances_list(Result_rankings, R_list)
            new_table_output(table_frame, Result_rankings,
                             Lengths, Strengths, Distances,
                             Params['median_dist'], Mutual_rankings)
        if name == 'Schulze_method':
            print_winners(table_frame,
                          Params['Schulze_winners'],
                          Params['Schulze_ranking'])

def print_winners(table_frame, winners, ranking):
    T = ""
    if ranking in [None, []]: T += "Ранжирование невозможно. "
    T += "Победители:"      
    lbl = Label(table_frame, **label_opts, **border_opts, anchor=E,
                text=T)
    lbl.grid(**grid_optsE, row=n+4, column=0)
    
    winnrs = Label(table_frame, **label_opts, **relief_opts, padx=3,
                   text=list2string([index2symbol(w, n-1)
                                     for w in winners]) )
    winnrs.grid(**grid_optsW, row=n+4, column=1)
    table_frame.grid()
    frame_table_output.grid()
    change_fieldsize_for_scrolling()

def read_table():
    global table_input, labels_columns_inp
    grid_forget_output()
    window0.focus()
    try:
        if n > max_count_of_alternatives:
            raise ValueError(
                "Количество альтернатив n слишком велико.\n"+\
                "Программа может зависнуть.")
        table_values = np.array([ [int(symbol2index(table_input[i][j].get()))
                            for j in range(m)] for i in range(n)])
        def elem_j_accepted(j):
            labels_columns_inp[j].config(background=window_background)
        def elem_j_incorrect(j):
            labels_columns_inp[j].config(background=error_color)
        number_of_accepted_elements = 0
        values = [i for i in range(n)]
        for j in range(m):
            profile_of_expert = table_values[:,j]
            if sorted(profile_of_expert) == values:
                elem_j_accepted(j)
                number_of_accepted_elements += n
            else:
                elem_j_incorrect(j)
        if number_of_accepted_elements != n*m:
            raise Exception()
    except Exception as e:
        messagebox.showwarning("", "Введите кооректные данные.\n" + str(e))
    else:
        Execute_algorythms(table_values.transpose())

def new_table_output(table_frame, Result_rankings,
                     Lengths, Strengths, Distances,
                     median_dist, Mutual_rankings):
    global labels_columns_out, labels_rankings_out, table_output, \
           label_L, label_S, label_D, table_info_output
    ### задание управляющих элементов
    r = len(Result_rankings)
    labels_columns_out = [Label(table_frame, **label_smallfont_opts,
                                **border_opts,
                                text="Ранжи-\nрование {0}".format(j+1))
                    for j in range(r)]
    labels_rankings_out = [Label(table_frame, **label_opts, **border_opts,
                                  text="Место {0}".format(i+1))
                            for i in range(n)]
    cell_opts = {'width':5, **relief_opts}
    table_output = [[ Label(table_frame, **cell_opts, **input_field_opts,
                            anchor=W, padx=2)
                     for j in range(r)]
                    for i in range(n)]
    label_L = Label(table_frame, **label_opts, **border_opts,
                    text="Длина:")
    label_S = Label(table_frame, **label_opts, **border_opts,
                    text="Сила:")
    label_D = Label(table_frame, **label_smallfont_opts, **border_opts,
                    justify=LEFT, text="Суммарное расстояние\nХэмминга:")
    table_info_output = [[Label(table_frame, **label_opts, **cell_opts)
                          for j in range(r)]
                         for i in range(3)]
    def min_and_max(List):
        inf = float("inf")
        L = [elem for elem in List if abs(elem) != inf]
        if L == []:
            return (-inf,-inf)
        return (min(L), max(L))
    
    MinsMaxes = [min_and_max(Lengths),
                 min_and_max(Strengths),
                 min_and_max(Distances)]
    color_min = '#BBEEFF'
    color_max = '#EEBBFF'
    color_mutual = '#CCFFCC'
    def highlight_characteristic(Characteristic, Ch_index, j):
        Ch_min = MinsMaxes[Ch_index][0]
        Ch_max = MinsMaxes[Ch_index][1]
        if Ch_min != Ch_max:
            if Characteristic[j] == Ch_min:
                table_info_output[Ch_index][j]['background'] = color_min
            if Characteristic[j] == Ch_max:
                table_info_output[Ch_index][j]['background'] = color_max
    
    for j in range(r):
        for i in range(len(Result_rankings[j])):
            table_output[i][j]['text'] = index2symbol(
                Result_rankings[j][i], n-1)
        if Result_rankings[j] in Mutual_rankings:
            for i in range(len(Result_rankings[j])):
                table_output[i][j]['background'] = color_mutual
        table_info_output[0][j]['text'] = Lengths[j]
        table_info_output[1][j]['text'] = Strengths[j]
        table_info_output[2][j]['text'] = Distances[j]
        highlight_characteristic(Lengths, 0, j)
        highlight_characteristic(Strengths, 1, j)
        highlight_characteristic(Distances, 2, j)
        if Distances[j] == median_dist:
            table_info_output[2][j]['text'] += "\nМедиана"
            table_info_output[2][j]['width'] = len("Медиана")
            table_info_output[2][j].config(**smallfont_opts)
    
    grid_output_table(table_frame)

def new_table_input():
    global table_input, labels_columns_inp, labels_rankings_inp, table_fromfile
    try:
        spinbox_n.set(str(n))
        spinbox_m.set(str(m))
    except Exception:
        pass
    table_frame = frame_table_input
    ### задание управляющих элементов      
    labels_columns_inp = [Label(table_frame, **label_smallfont_opts,
                                **border_opts,
                                text="Эксперт {0}".format(j+1))
                      for j in range(m)]
    labels_rankings_inp= [Label(table_frame, **label_opts, **border_opts,
                          text="Место {0}".format(i+1))
                    for i in range(n)]
    table_input = [[ttk.Combobox(table_frame, **input_field_opts, width=4,
                            values=[index2symbol(i, n-1) for i in range(n)],
                            state="readonly")
                    for j in range(m)]
                   for i in range(n)]
    for j in range(m):
        for i in range(n):
            table_input[i][j].current(i)
    if 'table_fromfile' in globals() and len(table_fromfile) > 0:
        for i in range(n):
            for j in range(m):
                table_input[i][j].current(table_fromfile[i][j])
        table_fromfile = []
    button_read_table.config(**button_opts_enabled)
    grid_input_table()  

def read_file():
    global n, m, table_fromfile
    table_fromfile = []
    s = entry_forfile.get()
    try:    
        table_fromfile = []
        f = open(s,'r')
        strings = f.readlines()
        f.close()
        for s in strings:
            s.replace("\n","")
            list_ = list( map(symbol2index, s.split()) )
            if list_ != []:
                table_fromfile.append(list_)
        if len(table_fromfile) == 0:
            raise ValueError("Пустой файл")
        nn = len(table_fromfile[0])
        for list_ in table_fromfile:
            if len(list_) != nn:
                raise ValueError("Неверная длина строк")
        if max(list(map(max,table_fromfile))) >= nn:
            raise ValueError("Неверное обозначение альтернативы")
    except FileNotFoundError:
        messagebox.showerror("", "Файл не найден")
    except Exception as e:
        messagebox.showerror("", "Файл некорректен.\n" + str(e))
    else:
        m = len(table_fromfile)
        n = nn
        table_fromfile = np.array(table_fromfile).transpose().tolist()
        grid_forget_output()
        grid_forget_input()
        new_table_input()

def read_n_and_m():
    global n, m
    n_prev = n
    m_prev = m
    try:
        n = spinbox_n.get()
        m = spinbox_m.get()
        n = int(n)
        m = int(m)
        if n > 0 and m > 0:
            if n > max_number_for_spinbox or m > max_number_for_spinbox \
               or n*m > max_number_for_cells:
                raise ValueError(
                    "Число альтернатив n и/или число экспертов m слишком большое.")
            elif n == n_prev and m == m_prev:
                pass
            else:          
                grid_forget_output()
                grid_forget_input()
                new_table_input()
        else:
            raise ValueError()
    except Exception as e:
        messagebox.showwarning("", "Введите кооректные данные.\n" + str(e))

def grid_forget_input():### уборка управляющих элементов
##    for item in frame_table_input.grid_slaves():
##        item.grid_forget()
##    frame_table_input.grid_remove()
##    change_fieldsize_for_scrolling()
    pass
def grid_forget_output():### уборка управляющих элементов
##    for frame in frame_table_output.grid_slaves():
##        for item in frame.grid_slaves():
##            item.grid_forget()
##        frame.grid_remove()
##    label_output.grid_remove()
##    frame_table_output.grid_remove()
##    change_fieldsize_for_scrolling()
    pass
def grid_input_table():### размещение управляющих элементов
##    global labels_columns_inp, labels_rankings_inp, table_input
##    if 'labels_columns_inp' in globals():
##        for j in range(len(labels_columns_inp)):
##            labels_columns_inp[j].grid(**pad0, row = 0, column = j+1)
##    if 'labels_rankings_inp' in globals():
##        for i in range(len(labels_rankings_inp)):
##            labels_rankings_inp[i].grid(**pad0, row = i+1, column = 0)
##    if 'table_input' in globals():
##        for i in range(len(table_input)):
##            for j in range(len(table_input[i])):
##                table_input[i][j].grid(**pad0, row = i+1, column = j+1)
##    frame_table_input.grid()
##    change_fieldsize_for_scrolling()
    pass
def grid_output_table(frame_of_particular_method):### размещение управляющих элементов
##    global labels_columns_out, labels_rankings_out, table_output, \
##           label_L, label_S, label_D, table_info_output
##    if 'labels_columns_out' in globals():
##        for j in range(len(labels_columns_out)):
##            labels_columns_out[j].grid(**pad0, row = 0, column = j+1)
##    if 'labels_rankings_out' in globals():
##        n = len(labels_rankings_out)
##        for i in range(n):
##            labels_rankings_out[i].grid(**pad0, sticky=E,
##                                         row = i+1, column = 0)
##        if 'label_L' in globals():
##            label_L.grid(**grid_optsE, row = n+1, column = 0)
##        if 'label_S' in globals():
##            label_S.grid(**grid_optsE, row = n+2, column = 0)
##        if 'label_D' in globals():
##            label_D.grid(**grid_optsE, row = n+3, column = 0)
##    if 'table_output' in globals():
##        for i in range(len(table_output)):
##            for j in range(len(table_output[i])):
##                table_output[i][j].grid(**pad0, row = i+1, column = j+1)
##    if 'table_info_output' in globals():
##        for i in range(len(table_info_output)):
##            for j in range(len(table_info_output[i])):
##                table_info_output[i][j].grid(**pad0, 
##                                            row = n + i+1, column = j+1)
##    frame_of_particular_method.grid()
##    frame_table_output.grid()
##    if label_output['text'] != "": label_output.grid()
##    change_fieldsize_for_scrolling()
    pass
def change_fieldsize_for_scrolling():
    frame1.update()
    frame2.update()
    widths = [frame1.winfo_width(),frame2.winfo_width()]
    heights = [frame1.winfo_height(),frame2.winfo_height()]
    Wid = 1*w//12 + sum(widths)
    Hei = 1*h//12 + max(heights)
    canvas1.config(scrollregion=(0, 0, Wid, Hei))
    frame0.config(width=Wid, height=Hei)


### Характеристики и опции для управляющих элементов, глобальные переменные
n = 0
m = 0
max_count_of_alternatives = 8
max_number_for_spinbox = 300
max_number_for_cells = 3600
error_color = '#FFBBBB'
input_bg_color='white'
window_background = 'antique white'#'#FAEBD7'
button_background = 'bisque'#'#FFE4C4'
disabled_button_background = '#D9D9D9'
button_borderwidth = 5
font = "Book Antiqua"
font_mono = 'Courier New'
font_size = 10
font_opts = {'font':(font, font_size)}
smallfont_opts = {'font':(font, font_size-2)}
border_opts = {'relief':"solid", 'borderwidth':1}
relief_opts = {'relief':"sunken", 'borderwidth':1}
frame_opts = {'background':window_background}
label_opts = {'background':window_background, 'font':(font, font_size)}
input_field_opts = {'background':input_bg_color,
                    'font':(font_mono, font_size)}
label_smallfont_opts = {'background':window_background,
                        'font':(font, font_size-2)}
button_opts_enabled = {'background':button_background,
	       'relief':RAISED,
	       'borderwidth':button_borderwidth,
	       'font':(font, font_size),
               'state':NORMAL}
button_opts_disabled = {'background':disabled_button_background,
	       'relief':RAISED,
	       'borderwidth':button_borderwidth,
	       'font':(font, font_size),
               'state':DISABLED}
pad3 = {'padx':3, 'pady':3}
pad0 = {'padx':0, 'pady':0}
grid_optsNW = {'sticky':NW, **pad3}
grid_optsW = {'sticky':W, **pad3}
grid_optsE = {'sticky':E, **pad3}
grid_optsNSEW = {'sticky':N+S+E+W, **pad3}
pack_optsTW = {'side':TOP, 'anchor':W}
pack_optsLW = {'side':LEFT, 'anchor':W, **pad3}
###

### Главное окно
window0 = Tk()
window0.config(
    background=window_background,
    relief=SUNKEN,
    borderwidth=3)
w = 2*window0.winfo_screenwidth()//3
h = 2*window0.winfo_screenheight()//3
window0.geometry('{}x{}'.format(w, h))
window0.title("Анализ алгоритмов группового выбора, \
использующих пути в орграфе")
###

### Задание и первичное размещение управляющих элементов

### Для скорллинга
# холст
canvas1 = Canvas(window0,
	     background=window_background,
	     width=window0.winfo_width(), height=window0.winfo_height(),
	     scrollregion=(0, 0, w, h))
# скроллбары
scrollbarY1 = Scrollbar(window0,
                        command=canvas1.yview,
                        orient=VERTICAL)
scrollbarX1 = Scrollbar(window0,
                        command=canvas1.xview,
                        orient=HORIZONTAL)
canvas1.config(
    yscrollcommand=scrollbarY1.set,
    xscrollcommand=scrollbarX1.set)
scrollbarY1.pack(side=RIGHT, fill=Y)
scrollbarX1.pack(side=BOTTOM, fill=X)
canvas1.pack(side=LEFT, expand=YES, fill=BOTH)
# главный фрейм
frame0 = Frame(canvas1, bd=0, background=window_background,
               width=canvas1.winfo_width(), height=canvas1.winfo_height())
canvas1.create_window((0, 0), window=frame0, anchor=NW)
###

### Управляющие элементы
info_file = "Текстовый файл должен содержать m значимых (не пробельных) \
строк. Переводов строк и пробелов может быть сколько угодно. \
В одной строке должен быть записан профиль одного эксперта: \
в виде чисел от 1 до n, либо в виде букв английского алфавита, \
нумерация должна быть строго последовательной: например, \
если n = 3, то можно использовать только цифры '1,2,3' и ,буквы 'A,B,C'. \
Символы в каждой строке разделяются пробелами. Чем левее стоит \
альтернатива в строке, тем она более предпочтительна. \
\nФайл должен лежать в той же директории, что и программа, инчае \
при нажатии кнопки 'Ввод из файла' будет выдана ошибка 'файл не найден'. \
\nФормат имени файла: 'file_name.txt' (без кавычек). \
\nВ таком же виде нужно писать имя в поле для имени файла."
info_input = "Добро пожаловать в программу!\n\n\
Ввести профили предпочтений экспертов можно через текстовый файл или \
последовательно задавая: n - число альтернатив-кандидиатов, m - число \
экспертов (экспертных профилей, бюллетеней). После нажатия на кнопку \
'Ввести n и m' появится табличка, в которой каждый столбец соответствует \
одному эксперту. В столбце альтернативы сортируются в порядке \
предпочтительности - чем выше, тем более предпочтительен кандидат. \
Кандидаты могут нумероваться буквами или цифрами от 1 до n. \
Нужно следить за тем, чтобы у эксперта не повторялись альтернативы, \
иначе будет выведена ошибка (с подсветкой того эксперта, у которого \
в профиле неверная расстановка). \
\nЕсли профили вводятся из файла, то после написания имени файла в поле \
для ввода, нужно нажать кнопку 'Ввод из файла'. Затем появится табличка, \
если файл прочитан. \
\nПосле ввода профилей можно нажать кнопку 'Пуск', перед этим выбрав \
методы (поставить галочки), которые хотели бы запустить."
info_output = "После нажатия 'Пуска' с правой стороны должны появиться \
результирующие ранжирования и информация к ним. \
\nПод таблицей для ввода пишется минимально возможное суммарное \
расстояние от любого ранжирования до всех экспертов. То есть, меньше \
просто быть не может.\
\nКаждому методу соответствует множество ранжирований. Исключение \
составляет метод Шульце - у него может быть 1 или 0 ранжирований. \
Кроме всего в этом методе считается множество победителей. \
\nКаждое ранжирование представляет собой столбец, и под каждым столбцом - \
информация о нём - длина пути-ранжирования, его сила и суммарное \
расстояние от экспертов. \
\nДля каждого метода - для его списка ранжирований используется цветная \
подсветка. Данная характеристика - это либо длина, либо сила, либо \
расстояние. \
\n Синий - минимум по данной характеристике среди ранжирований данного \
метода. \
\n Розовый - максимум по данной характеристике среди ранжирований данного \
метода. \
\n Зелёным цветом указаны те ранжирования, которые были выданы \
одновременно всеми выбранными методами, имеющими ранжирования. \
\nНапример, если у Шульце нет ранжирований, а мы выбрали три метода, \
включая Шульце, то зелёным будут подсвечены ранжирования, которые \
встречаются одновременно в двух других методах. \
\n"
menu0 = Menu(window0)
window0.config(menu=menu0)
menu0.add_command(label="Требования к файлам",
                  command = lambda: messagebox.showinfo("",info_file))
menu0.add_command(label="Пояснения ко вводу информации",
                  command = lambda: messagebox.showinfo("",info_input))
menu0.add_command(label="Пояснения к результату работы программы",
                  command = lambda: messagebox.showinfo("",info_output))

frame1 = Frame(master=frame0, **frame_opts)
frame1.grid(**grid_optsNW, row = 0, column = 0)
frame2 = Frame(master=frame0, **frame_opts)
frame2.grid(**grid_optsNW, row = 0, column = 1)

frame1up = Frame(frame1, **frame_opts)
frame1up.grid(**grid_optsNW, row = 0, column = 0)
frame1down = Frame(frame1, **frame_opts)
frame1down.grid(**grid_optsNW, row = 1, column = 0)

frame_n_m = LabelFrame(frame1up, **label_opts, text="Выбор n и m")
frame_n_m.grid(**grid_optsNW, row = 0, column = 0)

frame_checkbuttons = LabelFrame(frame1up, **label_opts, text="Выбор метода")
frame_checkbuttons.grid(**grid_optsW, row = 0, column = 1)

frame_forfile = LabelFrame(frame1up, **label_opts, text="Импорт из txt-файла")
frame_forfile.grid(**grid_optsNSEW, row = 1, column = 0, columnspan = 3)

frame_table_input = LabelFrame(frame1down, **label_opts,
                               text="Ввод таблицы ранжирований")
frame_table_input.grid(**grid_optsNW, row = 0, column = 0)

frame_table_output = LabelFrame(frame2, **label_opts, relief='flat',
                                text="Результирующие ранжирования")
frame_table_output.grid(**grid_optsNW, row = 0, column = 1)

frame_table_HP_max_length = LabelFrame(frame_table_output, **label_opts,
                     text="Гамильтоновы пути максимальной длины")
frame_table_HP_max_length.grid(**grid_optsNW, row = 0, column = 0)
frame_table_HP_max_strength = LabelFrame(frame_table_output, **label_opts,
                     text="Гамильтоновы пути наибольшей силы")
frame_table_HP_max_strength.grid(**grid_optsNW, row = 1, column = 0)
frame_table_Schulze_method = LabelFrame(frame_table_output, **label_opts,
                     text="Ранжирование по алгоритму Шульце")
frame_table_Schulze_method.grid(**grid_optsNW, row = 2, column = 0)
frame_table_Linear_medians = LabelFrame(frame_table_output, **label_opts,
                     text="Линейные медианы")
frame_table_Linear_medians.grid(**grid_optsNW, row = 3, column = 0)
frame_table_All_rankings= LabelFrame(frame_table_output, **label_opts,
                     text="Всевозможные ранжирования")
frame_table_All_rankings.grid(**grid_optsNW, row = 4, column = 0)

label1 = Label(frame_n_m, **label_opts, text="Число n альтернатив")
label1.grid(**grid_optsW,
            row = 0, column = 0)
spinbox_n = IntVar()
Spinbox(frame_n_m, **input_field_opts,
    from_=1, to=999,
    width=6,
    textvariable = spinbox_n
    ).grid(
        **grid_optsW,
        row = 0, column = 1)

label2 = Label(frame_n_m, **label_opts, text="Число m экспертов")
label2.grid(**grid_optsW,
            row = 1, column = 0)
spinbox_m = IntVar()
Spinbox(frame_n_m, **input_field_opts,
    from_=1, to=999,
    width=6,
    textvariable = spinbox_m
    ).grid(
        **grid_optsW,
        row = 1, column = 1)

button_read_n_and_m = Button(frame_n_m, **button_opts_enabled,
                             text="Ввод n и m",
                             command=read_n_and_m)
button_read_n_and_m.grid(**pad3, sticky = W+E,
                         row = 2, column = 0,
                         columnspan = 2)

_HP_max_length = IntVar()
_HP_max_strength = IntVar()
_Schulze_method = IntVar()
_Linear_medians = IntVar()
_All_rankings = IntVar()

_HP_max_length.set(0)
_HP_max_strength.set(0)
_Schulze_method.set(0)
_Linear_medians.set(0)
_All_rankings.set(0)

checkbutton_opts = {'master':frame_checkbuttons, **label_opts}
checkbutton1 = Checkbutton(**checkbutton_opts, variable=_HP_max_length,
                           text = "Гамильтоновы пути максимальной длины"
                           ).pack(**pack_optsTW)
checkbutton2 = Checkbutton(**checkbutton_opts, variable=_HP_max_strength,
                           text = "Гамильтоновы пути наибольшей силы",
                           ).pack(**pack_optsTW)
checkbutton3 = Checkbutton(**checkbutton_opts, variable=_Schulze_method,
                           text = "Ранжирование по алгоритму Шульце",
                           ).pack(**pack_optsTW)
checkbutton4 = Checkbutton(**checkbutton_opts, variable=_Linear_medians,
                           text = "Линейные медианы",
                           ).pack(**pack_optsTW)
checkbutton5 = Checkbutton(**checkbutton_opts, variable=_All_rankings,
                           text = "Всевозможные ранжирования",
                           ).pack(**pack_optsTW)

button_read_table = Button(frame1up, **button_opts_disabled,
                           text="Пуск!",
                           command=read_table)
button_read_table.grid(**grid_optsNSEW, row = 0, column = 2)

entry_forfile = Entry(frame_forfile, **input_field_opts)
entry_forfile.insert(0,"test.txt")
entry_forfile.pack(side=LEFT, expand = True, fill = X)
button_forfile = Button(frame_forfile, **button_opts_enabled,
                        text = "Ввод из файла",
                        command=read_file,
                        ).pack(**pack_optsLW)

label_output = Label(frame1, **label_opts,
                     bg=window_background, justify=LEFT,
                     text="")
label_output.grid(**grid_optsNW, row = 2, column = 0)

class IntersectIsFound(Exception): pass
class SchulzeRankingExists(Exception): pass
def run_tests():
    t0 = datetime.datetime.now()
##    directory = "D:/Учёба/ЛАБЫ и др/НИР/Программы/tests_n2_m3/"
    directory = "D:/Учёба/ЛАБЫ и др/НИР/Программы/tests_n3_m3/"
##    directory = "D:/Учёба/ЛАБЫ и др/НИР/Программы/tests_n4_m3/"
##    directory = "D:/Учёба/ЛАБЫ и др/НИР/Программы/tests_n3_m5/"
##    directory = "D:/Учёба/ЛАБЫ и др/НИР/Программы/tests_n4_m5_part1/"
##    directory = "D:/Учёба/ЛАБЫ и др/НИР/Программы/tests_n4_m5_part2/"
    fout = open('D:/Учёба/ЛАБЫ и др/НИР/Программы/stats.txt','a')
    fout.write(directory + "\n")
    fout.close()
    files = list(filter( 
        lambda s: s[-4:] == ".txt", os.listdir(directory) ))  
    def run_method_and_make_output():
        enumerator = 0
        total = 0
        schulzeR = 0
        if not _Schulze_method.get(): schulzeR = -math.inf
        info = ""
        for filename in files:
            entry_forfile.delete(0,END)
            entry_forfile.insert(0, directory + filename)
            print(filename)
            read_file()
            try:
                read_table()
            except IntersectIsFound:
                enumerator += 1
                schulzeR += 1
            except SchulzeRankingExists:
                schulzeR += 1
            total += 1
            if total % 10000 == 0:
                fout = open(
                    'D:/Учёба/ЛАБЫ и др/НИР/Программы/stats.txt','a')
                fout.write(
                    "         Всего тестов: {}\n".format(total) +\
                    "  Присутствий медианы: {}\n".format(enumerator) +\
                    "\n"
                    )
                fout.close()
        info += "         Всего тестов: {}\n".format(total)
        info += "  Ранжирований Шульце: {}\n".format(schulzeR)
        info += "  Присутствий медианы: {}\n".format(enumerator)
        info += "     Время выполнения: {}\n\n".format(
            datetime.datetime.now()-t0)
        return info
    _Linear_medians.set(1)
    case = _Linear_medians.get() and not _All_rankings.get()
    L = _HP_max_length.get()
    S = _HP_max_strength.get()
    Sch = _Schulze_method.get()
    text = ""
    if case and L and not S and not Sch:
        text += "Гамильтоновы пути максимальной длины:\n"
        text += run_method_and_make_output()
    elif case and S and not L and not Sch:
        text += "Гамильтоновы пути наибольшей силы:\n"
        text += run_method_and_make_output()
    elif case and Sch and not L and not S:
        text += "Ранжирование по алгоритму Шульце:\n"
        text += run_method_and_make_output()
    else:
        messagebox.showinfo("", "Надо выбрать метод.")
    fout = open('D:/Учёба/ЛАБЫ и др/НИР/Программы/stats.txt','a')
    fout.write(text)
    fout.close()
    print("Время выполнения:", datetime.datetime.now()-t0)
    
button_run_tests = Button(frame1, **button_opts_enabled,
                           text="Прогнать тесты", command=run_tests)
button_run_tests.grid(**grid_optsNW, row = 3, column = 0)

grid_forget_input()
grid_forget_output()
###
window0.mainloop()