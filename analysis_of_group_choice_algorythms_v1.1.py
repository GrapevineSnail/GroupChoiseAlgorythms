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

def visualize_graph(Weights_matrix, Hamiltonian_path):
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
##    start_node = index2symbol(Hamiltonian_path[0], n-1)
##    finish_node = index2symbol(Hamiltonian_path[-1], n-1)
    for node in graph:
        node_edge_color_map.append('black')
##        if node in [start_node, finish_node]:
##            node_edge_color_map.append(indicating_color)
##        else:
##            node_edge_color_map.append('black')
    edges = [(u,v) for (u,v) in graph.edges(data = False)]
    edges_cnt = len(edges)
    edges_color_map = ['black' for i in range(edges_cnt)]
    edges_width_map = [1 for i in range(edges_cnt)]
    path_edges = [(index2symbol(Hamiltonian_path[i], n-1),
                   index2symbol(Hamiltonian_path[i+1], n-1))
                   for i in range(len(Hamiltonian_path)-1)]
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

def graph_matrix2adjacency_list(Weights_matrix):
    n = len(Weights_matrix)#квадратная матрица
    Adjacency_list = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if abs(Weights_matrix[i][j]) not in [math.inf, float("inf")]:
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
           else summarized_P_matrix[i][j]
           for j in range(n)] for i in range(n)]
    return C

def make_summarized_P_matrix(list_of_R_matrices):
    P = np.array([[0 for j in range(n)] for i in range(n)])
    for Rj in list_of_R_matrices:
        P += np.array(Rj)
    return P.tolist()

def make_single_R_profile_matrix(single_profile):#adjacency_matrices
    profile = np.array(single_profile)
    Rj = [[0 for j in range(n)] for i in range(n)]
    for i in range(len(profile)):
        p = profile[i]
        for p2 in profile[i+1:]:
            Rj[p][p2] = 1
    return Rj

def Make_used_matrices(list_of_profiles):#все матрицы n*n, матриц m штук
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

def Linear_diagonals(R_list):
    All_rankings = All_various_rankings()
    Distances = Paths_sum_distances_list(All_rankings, R_list)
    Dmin = min(Distances)
    Rankings_list = [ All_rankings[i] for i in range(len(All_rankings))
                      if Distances[i] == Dmin ]
    return Rankings_list

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
##    print("Q",n-1,"\n",Q,sep='',end="\n")

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
    return (Paths_matrix, LP_matrix, Rankings_list(LP_matrix))

def HP_max_strength(Weights_matrix):
    Paths_matrix = hamiltonian_paths_through_matrix_degree(Weights_matrix)
    PS = Paths_strengths_matrix(Paths_matrix, Weights_matrix)
    inds, max_strength= inds_of_max(PS)
    #матрица сильнейших путей - strongest paths (SP)
    SP_matrix = [[[] for j in range(n)] for i in range(n)]
    for (i,j,k) in inds:
        SP_matrix[i][j].append(Paths_matrix[i][j][k])
    return (Paths_matrix, SP_matrix, Rankings_list(SP_matrix))

def Schulze_method(Weights_matrix):
    def is_path_correct_Schulze(vertices_list, Weights_matrix):
        #все звенья пути должны удовлетворять: W[C(i),C(i+1)]>W[C(i+1),C(i)]
        l = len(vertices_list)
        for i in range(l-1):
            if Weights_matrix[vertices_list[i]][vertices_list[i+1]] <= \
               Weights_matrix[vertices_list[i+1]][vertices_list[i]]:
                return False
        return True
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
    def strongest_paths_betweenAB(Weights_matrix, 
                                  All_paths_betweenAB, idA, idB):
        Paths = [path for path in All_paths_betweenAB
                 if is_path_correct_Schulze(path, Weights_matrix)]
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
    
    Adjacency_list = graph_matrix2adjacency_list(Weights_matrix)
    Paths_matrix = [[all_simple_paths_betweenAB(Adjacency_list, i, j)
                     for j in range(n)] for i in range(n)]
    #матрица сильнейших путей - strongest paths (SP)
    SP_matrix = [[0 for j in range(n)] for i in range(n)]
    #матрица сил сильнеёших путей - strengths of the strongest paths (SSP)
    SSP_matrix = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            strength, S_paths = strongest_paths_betweenAB(
                Weights_matrix, Paths_matrix[i][j], i, j)
            SP_matrix[i][j] = S_paths
            SSP_matrix[i][j] = strength
    def compare_MORETHAN(A,B):
        if SSP_matrix[A][B] > SSP_matrix[B][A]:
            return True #побеждает A. (A > B) т.е. morethan
        return False
    def compare_EQUIV(A,B):#не менять местами if-ы 
        if compare_MORETHAN(A,B) and compare_MORETHAN(B,A): return True
        if compare_MORETHAN(A,B): return False
        if compare_MORETHAN(B,A): return False
        return True #условно - несравнимость
    ranking = [i for i in range(n)]#результирующее ранжирование по Шульце
    ranking.reverse()
    for i in range(n):
        j = i
        while j > 0 and \
              (compare_MORETHAN(ranking[j],ranking[j-1]) or \
               compare_EQUIV(ranking[j],ranking[j-1])):
            ranking[j], ranking[j-1] = ranking[j-1], ranking[j]
            j -= 1
    return (Paths_matrix, SP_matrix, SSP_matrix, ranking)

def Execute_algorythms(list_of_profiles):
    R_list, P, C, R = Make_used_matrices(list_of_profiles)
##    print("P\n",np.array(P))
    if method.get() == "HP_max_length":
        Paths_matrix, LP_matrix, Rankings = HP_max_length(C)
        print_result(R_list, C, LP_matrix, Rankings)
##        for HP in Rankings:
##            visualize_graph(C, HP)
    elif method.get() == "HP_max_strength":
        Paths_matrix, SP_matrix, Rankings = HP_max_strength(C)
        print_result(R_list, C, SP_matrix, Rankings)
##        for HP in Rankings:
##            visualize_graph(C, HP)
    elif method.get() == "Schulze_method":
        Paths_matrix, SP_matrix, SSP_matrix, ranking = Schulze_method(C)
        print_result(R_list, C, Paths_matrix, [ranking])
    elif method.get() == "Linear_diagonals":
        Rankings = Linear_diagonals(R_list)
        print_result(R_list, C, None, Rankings)
##        Paths_matrix, ranking = Linear_diagonals(R)
##        print_result(R_list, C, Paths_matrix, [ranking], R)
    elif method.get() == "All_rankings":
        Rankings = All_various_rankings()
        print_result(R_list, C, None, Rankings)
    else:
        messagebox.showinfo("", "Выберите метод")

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

def print_result(R_list, Weights_matrix, Paths_matrix, Result_rankings):
    grid_forget_output()
    text = ""
    if Paths_matrix != None:
        PW = Paths_weights_matrix(Paths_matrix, Weights_matrix)
        PS = Paths_strengths_matrix(Paths_matrix, Weights_matrix)
        PD = Paths_sum_distances_matrix(Paths_matrix, R_list)
        P = [[ list2string([ index_path2symbolic_path(path)
                             for path in Paths_matrix[i][j] ])
               for j in range(n)] for i in range(n)]
        PW = [[list2string(PW[i][j]) for j in range(n)] for i in range(n)]
        PS = [[list2string(PS[i][j]) for j in range(n)] for i in range(n)]
        PD = [[list2string(PD[i][j]) for j in range(n)] for i in range(n)]
        text += "Пути из i в j:\n" + \
                matrix2string(P) + "\n\n" + \
                "Длины путей:\n" + \
                matrix2string(PW) + "\n\n" + \
                "Силы путей:\n" + \
                matrix2string(PS) + "\n\n" + \
                "Суммарные расстояния до профилей экспертов:\n" + \
                matrix2string(PD) + "\n\n"
    label_output['text'] = text
    Lengths = Paths_weights_list(Result_rankings, Weights_matrix)
    Strengths = Paths_strengths_list(Result_rankings, Weights_matrix)
    Distances = Paths_sum_distances_list(Result_rankings, R_list)
    new_table_output(Result_rankings, Lengths, Strengths, Distances)

def read_table():
    global table_input, labels_experts
    window0.focus()
    table_values = np.array([ [int(symbol2index(table_input[i][j].get()))
                        for j in range(m)] for i in range(n)])
    def elem_j_accepted(j):
        labels_experts[j].config(background=window_background)
    def elem_j_incorrect(j):
        labels_experts[j].config(background=error_color)
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
        messagebox.showinfo("", "Введите кооректные данные")
    else:
        Execute_algorythms(table_values.transpose()) 

def new_table_output(Result_rankings, Lengths, Strengths, Distances):
    global labels_ranks, labels_alters_output, table_output, \
           label_L, label_S, label_D, table_LSD_output
    grid_forget_output()
    table_frame = frame_output
    ### задание управляющих элементов
    r = len(Result_rankings)
    labels_ranks = [Label(table_frame, **label_opts, **label_border,
                            text="Ранжирование {0}".format(j+1))
                    for j in range(r)]
    labels_alters_output = [Label(table_frame, **label_opts, **label_border,
                                  text="Место {0}".format(i+1))
                            for i in range(n)]
    cell_opts = {'width':6+2, 'relief':"sunken", 'borderwidth':1}
    table_output = [[ Label(table_frame, **cell_opts,
                            background=input_bg_color,
                            anchor=W, padx=2)
                     for j in range(r)]
                    for i in range(n)]
    label_L = Label(table_frame, **label_opts, **label_border,
                      text="Длина:")
    label_S = Label(table_frame, **label_opts, **label_border,
                        text="Сила:")
    label_D = Label(table_frame, **label_opts, **label_border,
                    text="Суммарное\nрасстояние\nХэмминга")
    label_D.config(font = (font,8))
    table_LSD_output = [[ Label(table_frame, **label_opts, **cell_opts)
                          for j in range(r)]
                        for i in range(3)]
    
    def set_characteristic(Characteristic, Ch_index, j):
        table_LSD_output[Ch_index][j]['text'] = Characteristic[j]
        values_set = set(Characteristic)
        values_set.discard(-float("inf"))
        values_set.discard(float("inf"))
        try:
            Ch_min = min(values_set)
            Ch_max = max(values_set)
            highlight_min = {'background':'#BBEEFF'}
            highlight_max = {'background':'#FFBBFF'}
            if Ch_min != Ch_max:
                if Characteristic[j] == Ch_min:
                    table_LSD_output[Ch_index][j].config(**highlight_min)
                if Characteristic[j] == Ch_max:
                    table_LSD_output[Ch_index][j].config(**highlight_max)
        except Exception:
            pass
    for j in range(r):
        for i in range(len(Result_rankings[j])):
            table_output[i][j]['text'] = index2symbol(
                Result_rankings[j][i], n-1)
        labels_ranks[j].config(font = (font,8))
        set_characteristic(Lengths, 0, j)
        set_characteristic(Strengths, 1, j)
        set_characteristic(Distances, 2, j)
    grid_output()

def new_table_input():
    global table_input, labels_experts, labels_alters, table_fromfile
    grid_forget_output()
    grid_forget_input()
    update_spinboxes()
    table_frame = frame_table_input
    ### задание управляющих элементов
    labels_experts = [Label(table_frame, **label_opts, **label_border,
                            text="Эксперт {0}".format(j+1))
                      for j in range(m)]
    labels_alters= [Label(table_frame, **label_opts, **label_border,
                          text="Место {0}".format(i+1))
                    for i in range(n)]
    table_input = [[ttk.Combobox(table_frame, width=6,
                            values=[index2symbol(i, n-1) for i in range(n)],
                            state="readonly")
                    for j in range(m)]
                   for i in range(n)]
    for i in range(n):
        for j in range(m):
            table_input[i][j].current(i)
    if 'table_fromfile' in globals() and len(table_fromfile) > 0:
        for i in range(n):
            for j in range(m):
                table_input[i][j].current(table_fromfile[i][j])
        table_fromfile = []
    button_read_table.config(**button_opts_enabled)
    grid_input()

def update_spinboxes():
    try:
        spinbox_n.set(str(n))
        spinbox_m.set(str(m))
    except Exception:
        pass

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
        messagebox.showwarning("", "Файл не найден")
    except Exception as e:
        messagebox.showwarning("", "Файл некорректен.\n" + str(e))
    else:
        m = len(table_fromfile)
        n = nn
        table_fromfile = np.array(table_fromfile).transpose().tolist()
        new_table_input()

def read_n_and_m():
    global n, m
    n_prev = n
    m_prev = m
    n = spinbox_n.get()
    m = spinbox_m.get()
    try:
        n = int(n)
        m = int(m)
        if n > 0 and m > 0:
            if n == n_prev and m == m_prev:
                pass
            else:
                new_table_input()
        else:
            raise ValueError()
    except ValueError:
        messagebox.showinfo("", "Введите кооректные данные")

def grid_forget_input():
    global labels_experts, labels_alters, table_input
    if 'labels_experts' in globals():
        for j in range(len(labels_experts)):
            labels_experts[j].grid_forget()
    if 'labels_alters' in globals():
        for i in range(len(labels_alters)):
            labels_alters[i].grid_forget()
    if 'table_input' in globals():
        for i in range(len(table_input)):
            for j in range(len(table_input[i])):
                table_input[i][j].grid_forget()
    change_fieldsize_for_scrolling()
    
def grid_input():### размещение управляющих элементов
    global labels_experts, labels_alters, table_input
    if 'labels_experts' in globals():
        for j in range(len(labels_experts)):
            labels_experts[j].grid(**pad0, row = 0, column = j+1)
    if 'labels_alters' in globals():
        for i in range(len(labels_alters)):
            labels_alters[i].grid(**pad0, row = i+1, column = 0)
    if 'table_input' in globals():
        for i in range(len(table_input)):
            for j in range(len(table_input[i])):
                table_input[i][j].grid(**pad0, row = i+1, column = j+1)
    change_fieldsize_for_scrolling()
    
def grid_forget_output():### уборка управляющих элементов
    global labels_ranks, labels_alters_output, table_output, \
           label_L, label_S, label_D, table_LSD_output
    if 'labels_ranks' in globals():
        for j in range(len(labels_ranks)):
            labels_ranks[j].grid_forget()
    if 'labels_alters_output' in globals():
        for i in range(len(labels_alters_output)):
            labels_alters_output[i].grid_forget()
    if 'table_output' in globals():
        for i in range(len(table_output)):
            for j in range(len(table_output[i])):
                table_output[i][j].grid_forget()
    if 'label_L' in globals():
        label_L.grid_forget()
    if 'label_S' in globals():
        label_S.grid_forget()
    if 'label_D' in globals():
        label_D.grid_forget()
    if 'table_LSD_output' in globals():
        for i in range(3):
            for j in range(len(table_LSD_output[i])):
                table_LSD_output[i][j].grid_forget()
    frame_output.grid_forget()
    label_output.grid_forget()
    change_fieldsize_for_scrolling()

def grid_output():### размещение управляющих элементов
    global labels_ranks, labels_alters_output, table_output, \
           label_L, label_S, label_D, table_LSD_output
    if 'labels_ranks' in globals():
        for j in range(len(labels_ranks)):
            labels_ranks[j].grid(**pad0, row = 0, column = j+1)
    if 'labels_alters_output' in globals():
        
        n = len(labels_alters_output)
        for i in range(n):
            labels_alters_output[i].grid(**pad0, row = i+1, column = 0)
        if 'label_L' in globals():
            label_L.grid(**grid_optsE, row = n+1, column = 0)
        if 'label_S' in globals():
            label_S.grid(**grid_optsE, row = n+2, column = 0)
        if 'label_D' in globals():
            label_D.grid(**grid_optsE, row = n+3, column = 0)
    if 'table_output' in globals():
        for i in range(len(table_output)):
            for j in range(len(table_output[i])):
                table_output[i][j].grid(**pad0, row = i+1, column = j+1)
    if 'table_LSD_output' in globals():
        for i in range(3):
            for j in range(len(table_LSD_output[i])):
                table_LSD_output[i][j].grid(**pad0,
                                            row = n + i+1, column = j+1)
    frame_output.grid(**grid_optsNW, row = 0, column = 1)
    if label_output['text'] != "":
        label_output.grid(**grid_optsNW, row = 0, column = 2)
    change_fieldsize_for_scrolling()
       
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
error_color = '#FFBBBB'
input_bg_color='white'
window_background = 'antique white'#'#FAEBD7'
button_background = 'bisque'#'#FFE4C4'
disabled_button_background = '#D9D9D9'
button_borderwidth = 5
font = "Book Antiqua"
font_mono = 'Courier New'
font_size = 10
label_border = {'borderwidth':1, 'relief':"solid"}

frame_opts = {'background':window_background}
label_opts = {'background':window_background, 'font':(font, font_size)}
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

frame_radiobuttons = LabelFrame(frame1up, **label_opts, text="Выбор метода")
frame_radiobuttons.grid(**grid_optsW, row = 0, column = 1)

frame_forfile = LabelFrame(frame1up, **label_opts, text="Импорт из txt-файла")
frame_forfile.grid(**grid_optsNSEW, row = 1, column = 0, columnspan = 3)

frame_table_input = LabelFrame(frame1down, **label_opts,
                               text="Ввод таблицы ранжирований")
frame_table_input.grid(**grid_optsNW, row = 0, column = 0)

frame_output = LabelFrame(frame2, **label_opts,
                     text="Результирующие ранжирования")

label1 = Label(frame_n_m, **label_opts, text="Число n альтернатив")
label1.grid(**grid_optsW,
            row = 0, column = 0)
spinbox_n = IntVar()
Spinbox(#n
    frame_n_m,
    from_=1,
    to=99,
    width=6,
    font=(font, font_size),
    textvariable = spinbox_n
    ).grid(
        **grid_optsW,
        row = 0, column = 1)

label2 = Label(frame_n_m, **label_opts, text="Число m экспертов")
label2.grid(**grid_optsW,
            row = 1, column = 0)
spinbox_m = IntVar()
Spinbox(#m
    frame_n_m,
    from_=1,
    to=99,
    width=6,
    font=(font, font_size),
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


method = StringVar()
method.set('_')
radiobutton_opts = {'master':frame_radiobuttons,
                    **label_opts, 'variable':method}
radiobutton1 = Radiobutton(**radiobutton_opts,
                           text = "Гамильтоновы пути наибольшей длины",
                           value = "HP_max_length").pack(**pack_optsTW)
radiobutton2 = Radiobutton(**radiobutton_opts,
                           text = "Гамильтоновы пути наибольшей силы",
                           value = "HP_max_strength").pack(**pack_optsTW)
radiobutton3 = Radiobutton(**radiobutton_opts,
                           text = "Ранжирование по алгоритму Шульце",
                           value = "Schulze_method").pack(**pack_optsTW)
radiobutton4 = Radiobutton(**radiobutton_opts,
                           text = "Линейные диагонали",
                           value = "Linear_diagonals").pack(**pack_optsTW)
radiobutton5 = Radiobutton(**radiobutton_opts,
                           text = "Всевозможные ранжирования",
                           value = "All_rankings").pack(**pack_optsTW)

button_read_table = Button(frame1up, **button_opts_disabled,
                           text="Пуск!",
                           command=read_table)
button_read_table.grid(**grid_optsNSEW, row = 0, column = 2)

entry_forfile = Entry(frame_forfile)
entry_forfile.insert(0,"test.txt")
entry_forfile.pack(side=LEFT, expand = True, fill = X)
button_forfile = Button(frame_forfile, **button_opts_enabled,
                        text = "Ввод из файла",
                        command=read_file,
                        ).pack(**pack_optsLW)

label_output = Label(frame2,
                     borderwidth = 1, relief="solid",
                     bg=window_background,
                     font=(font_mono, font_size),
                     text="")

change_fieldsize_for_scrolling()
###
window0.mainloop()
