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


def visualize_graph(Weights_matrix, path):
    plot.clf()
    graph = nx.DiGraph()
    for i in range(n):
        graph.add_node(index2symbol(i, n-1))
        for j in range(n):
            if (i != j and abs(Weights_matrix[i][j]) != math.inf):
                graph.add_edge(index2symbol(i, n-1),
                               index2symbol(j, n-1),
                               weight=Weights_matrix[i][j])
    indicating_color = '#FF8A00'
    node_edge_color_map = []
# if path not in [None,[]]:
##        start_node = index2symbol(path[0], n-1)
##        finish_node = index2symbol(path[-1], n-1)
    for node in graph:
        node_edge_color_map.append('black')
# if path not in [None,[]]:
# if node in [start_node, finish_node]:
# node_edge_color_map.append(indicating_color)
# else:
# node_edge_color_map.append('black')
    edges = [(u, v) for (u, v) in graph.edges(data=False)]
    edges_cnt = len(edges)
    edges_color_map = ['black' for i in range(edges_cnt)]
    edges_width_map = [1 for i in range(edges_cnt)]
    if path not in [None, []]:
        path_edges = [(index2symbol(path[i], n-1),
                       index2symbol(path[i+1], n-1))
                      for i in range(len(path)-1)]
        for i in range(edges_cnt):
            if edges[i] in path_edges:
                edges_color_map[i] = indicating_color
                edges_width_map[i] = 2
                path_edges.pop(path_edges.index(edges[i]))

    position = nx.circular_layout(graph)  # position = nx.planar_layout(graph)
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
        edgelist=graph.edges(),
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


def Hamming_distance(R1, R2):
    # вход: матрицы одинаковой размерности только из 1 и 0
    R1 = np.array(R1)
    R2 = np.array(R2)
    iscorrect1 = np.all((R1 == 0) + (R1 == 1))
    iscorrect2 = np.all((R2 == 0) + (R2 == 1))
    if R1.shape != R2.shape or not iscorrect1 or not iscorrect2:
        raise ValueError()
    disparity = list(map(abs, R1-R2))
    return sum(sum(disparity))


def sum_Hamming_distance(R, List_of_other_R):
    # вторым параметром - список матриц смежности
    return sum([Hamming_distance(R, R_other) for R_other in List_of_other_R])


def weights_of_path(vertices_list, Weights_matrix):
    weights_list = []
    l = len(vertices_list)
    # при l = 0 нет пути
    # при l = 1 путь (a) "ничего не делать" - нет пути, так как нет петли
    if l > 1:  # включает и путь-петлю (a,a)
        weights_list = [
            Weights_matrix[vertices_list[i]][vertices_list[i+1]]
            for i in range(l-1)]
    return weights_list


def path_length(vertices_list, Weights_matrix):
    return sum(weights_of_path(vertices_list, Weights_matrix))


def path_strength(vertices_list, Weights_matrix):
    wp = weights_of_path(vertices_list, Weights_matrix)
    if wp == []:
        return -math.inf
    return min(wp)


def path_sum_Hamming_distance(vertices_list, R_list):
    return sum_Hamming_distance(
        make_single_R_profile_matrix(vertices_list), R_list)


def Paths_lenghts_matrix(Paths_matrix, Weights_matrix):
    return [[
        [path_length(path, Weights_matrix)
         for path in Paths_matrix[i][j]]
        for j in range(n)] for i in range(n)]


def Paths_strengths_matrix(Paths_matrix, Weights_matrix):
    return [[
        [path_strength(path, Weights_matrix)
         for path in Paths_matrix[i][j]]
        for j in range(n)] for i in range(n)]


def Paths_sum_Hamming_distances_matrix(Paths_matrix, R_list):
    return [[
        [path_sum_Hamming_distance(path, R_list)
         for path in Paths_matrix[i][j]]
        for j in range(n)] for i in range(n)]


def Paths_weights_list(Paths_list, Weights_matrix):
    return [path_length(path, Weights_matrix)
            for path in Paths_list]


def Paths_strengths_list(Paths_list, Weights_matrix):
    return [path_strength(path, Weights_matrix)
            for path in Paths_list]


def Paths_sum_Hamming_distances_list(Paths_list, R_list):
    return [path_sum_Hamming_distance(path, R_list)
            for path in Paths_list]


def Paths_list2matrix(Paths_list):
    PM = [[[] for j in range(n)] for i in range(n)]
    for path in Paths_list:
        i = path[0]
        j = path[-1]
        PM[i][j].append(path)
    return PM


def Paths_matrix2list(Paths_matrix):
    PL = []
    if n == 1:
        PL.append([0])
    else:
        for i in range(n):
            for j in range(n):
                for paths in Paths_matrix[i][j]:
                    if type(paths) == list:
                        PL.append(paths)
    return PL


def matrix2adjacency_list(Weights_matrix):
    n = len(Weights_matrix)  # квадратная матрица
    Adjacency_list = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and \
                    abs(Weights_matrix[i][j]) not in [math.inf, float("inf")]:
                Adjacency_list[i].append(j)
    return Adjacency_list


def make_sum_R_profile_matrix(weight_C_matrix):  # adjacency_matrix
    R = [[1 if abs(weight_C_matrix[i][j]) != math.inf
          else 0
          for j in range(n)] for i in range(n)]
    return R


def make_weight_C_matrix(summarized_P_matrix):
    C = [[-math.inf if
          i == j or summarized_P_matrix[i][j] < summarized_P_matrix[j][i]
          else summarized_P_matrix[i][j] - summarized_P_matrix[j][i]
          for j in range(n)] for i in range(n)]
    return C


def make_summarized_P_matrix(list_of_R_matrices):
    P = np.zeros((n, n))
    for Rj in list_of_R_matrices:
        P += np.array(Rj)
    return P.tolist()


def make_single_R_profile_matrix(single_profile):
    l = len(single_profile)
    if l != n or l != len(np.unique(single_profile)):
        return None
    profile = np.array(single_profile)
    Rj = [[0 for j in range(l)] for i in range(l)]
    for i in range(l):
        candidate = profile[i]
        for candidate_2 in profile[i+1:]:
            Rj[candidate][candidate_2] = 1
    return Rj  # adjacency_matrix


def Make_used_matrices(list_of_profiles):  # аргумент: матрицы n*n, их m штук
    R_list = [make_single_R_profile_matrix(p) for p in list_of_profiles]
    P = make_summarized_P_matrix(R_list)
    C = make_weight_C_matrix(P)
    R = make_sum_R_profile_matrix(C)
    return (R_list, P, C, R)
###


def All_various_rankings():
    def permutations_of_elements(elements):
        return [list(p) for p in itertools.permutations(elements)]
    Rankings = []
    if n == 1:
        Rankings.append([0])
    else:
        for i in range(n):
            for j in range(n):
                if i != j:
                    middle_vetrices = [
                        v for v in range(n) if v != i and v != j]
                    for p in permutations_of_elements(middle_vetrices):
                        Rankings.append([i] + p + [j])
    return Rankings


def Linear_medians(R_list):
    All_rankings = All_various_rankings()
    Distances = Paths_sum_Hamming_distances_list(All_rankings, R_list)
    median_dist = min(Distances)
    Rankings_list = [All_rankings[i] for i in range(len(All_rankings))
                     if Distances[i] == median_dist]
    return Rankings_list, median_dist


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
    Q = [[0 if (abs(Weights_matrix[i][j]) == math.inf)
          else 1
          for j in range(n)]
         for i in range(n)]
    Q = np.array(Q)
    for i in range(n):
        Q[i][i] = 0  # зануление диагонали

    Q_paths_cnt = np.linalg.matrix_power(
        Q, n-1)  # Paths count between vertices

    paths_cnt = 0  # Total paths count
    for i in range(n):
        for j in range(n):
            if i != j:
                paths_cnt += Q_paths_cnt[i][j]

    sym2ind = {
        sympy.Symbol('a{0}'.format(i), commutative=False): i
        for i in range(n)}
    ind2sym = dict(zip(sym2ind.values(), sym2ind.keys()))

    H = np.array([[ind2sym[j] if Q[i][j] == 1 else int(0)
                   for j in range(n)]
                  for i in range(n)])

    for i in range(2, n):
        Q_quote = H.dot(Q)
        Q = F_zeroing(Q_quote, list(sym2ind.keys()))

    Paths_matrix = [[[] for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if Q[i][j] != 0:
                if n > 2:
                    if "+" in str(Q[i][j]):
                        some_sym_paths = list(Q[i][j].args)
                    else:
                        some_sym_paths = [Q[i][j]]

                    for sym_path in some_sym_paths:
                        Paths_matrix[i][j].append(
                            list(map(lambda x: sym2ind[x],
                                     (ind2sym[i]*sym_path*ind2sym[j]).args))
                        )
                elif n == 2:
                    Paths_matrix[i][j].append([i] + [j])
    return Paths_matrix


def inds_of_max(M):
    max_elem = -math.inf
    for i in range(len(M)):
        for j in range(len(M[i])):
            for elem in M[i][j]:
                if elem > max_elem:
                    max_elem = elem
    indices_of_max_elems = []
    for i in range(len(M)):
        for j in range(len(M[i])):
            for k in range(len(M[i][j])):
                if M[i][j][k] == max_elem:
                    indices_of_max_elems.append((i, j, k))
    return (indices_of_max_elems, max_elem)


def HP_max_length(Weights_matrix):
    Paths_matrix = Hamiltonian_paths_through_matrix_degree(Weights_matrix)
    indices, max_length = inds_of_max(
        Paths_lenghts_matrix(Paths_matrix, Weights_matrix)
    )
    # матрица длиннейших путей - longest paths (LP)
    LP_matrix = [[[] for j in range(n)] for i in range(n)]
    for (i, j, k) in indices:
        LP_matrix[i][j].append(Paths_matrix[i][j][k])
    return Paths_matrix2list(LP_matrix)


def HP_max_strength(Weights_matrix):
    Paths_matrix = Hamiltonian_paths_through_matrix_degree(Weights_matrix)
    indices, max_strength = inds_of_max(
        Paths_strengths_matrix(Paths_matrix, Weights_matrix)
    )
    # матрица сильнейших путей - strongest paths (SP)
    SP_matrix = [[[] for j in range(n)] for i in range(n)]
    for (i, j, k) in indices:
        SP_matrix[i][j].append(Paths_matrix[i][j][k])
    return Paths_matrix2list(SP_matrix)


def Schulze_method(Weights_matrix):
    def all_simple_paths_betweenAB(Adjacency_list, idA, idB):
        Paths = []  # simple - без циклов
        if idA != idB:
            n = len(Adjacency_list)
            cur_path = []
            visited = [False for i in range(n)]

            def enter_in_vertex(v):
                cur_path.append(v)
                visited[v] = True  # зашли в вершину

            def leave_vertex(v):
                cur_path.pop()
                visited[v] = False  # вышли - поднялись выше по цепочке

            def dfs(v):
                enter_in_vertex(v)
                if v == idB:  # нашли путь
                    Paths.append(cur_path.copy())
                else:
                    for next_v in Adjacency_list[v]:
                        if visited[next_v] == False:
                            dfs(next_v)  # идём туда, куда ещё не входили
                leave_vertex(v)
                return 0
            dfs(idA)
        return Paths

    def strongest_paths_betweenAB(Weights_matrix,
                                  All_paths_betweenAB, idA, idB):
        Paths = [path for path in All_paths_betweenAB]
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
    # матрица сильнейших путей - strongest paths (SP)
    SP_matrix = [[0 for j in range(n)] for i in range(n)]
    # матрица сил сильнейших путей - Power
    Power = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            strength, S_paths = strongest_paths_betweenAB(
                Weights_matrix, Paths_matrix[i][j], i, j)
            SP_matrix[i][j] = S_paths
            Power[i][j] = strength

    def compare_MORETHAN(A, B):
        if Power[A][B] > Power[B][A]:
            return True  # побеждает A. (A > B) т.е. morethan
        return False

    def compare_EQUIV(A, B):
        if not compare_MORETHAN(A, B) and not compare_MORETHAN(B, A):
            return True
        return False

    def is_winner(A):
        for B in range(n):
            if B != A and Power[A][B] < Power[B][A]:
                return False
        return True
    S = [A for A in range(n) if is_winner(A)]  # set of winners
    # матрица строгих сравнений A>B <=> comparsion[A][B]=1
    # должна быть транзитивна A>B and B>C => A>C
    comparsion = [[1 if compare_MORETHAN(i, j) else 0
                   for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and Power[i][j] == 0:
                for k in range(n):
                    if compare_MORETHAN(i, k) and compare_MORETHAN(k, j):
                        comparsion[i][j] = 1
    ranking = None
    if sum([sum(c) for c in comparsion]) == (n**2 - n)/2:  # количество единиц при транзитивности
        # результирующее ранжирование по Шульце
        ranking = [i for i in range(n)]
        ranking.reverse()
        for i in range(n):
            j = i
            while j > 0 and comparsion[ranking[j]][ranking[j-1]]:
                ranking[j], ranking[j-1] = ranking[j-1], ranking[j]
                j -= 1
    return S, ranking


def Execute_algorythms(list_of_profiles):
    names = [
        'HP_max_length',
        'HP_max_strength',
        'Schulze_method',
        'Linear_medians',
        'All_rankings']
    frames = [
        frame_table_HP_max_length,
        frame_table_HP_max_strength,
        frame_table_Schulze_method,
        frame_table_Linear_medians,
        frame_table_All_rankings]
    checkbuttons = [
        cb_HP_max_length.get(),
        cb_HP_max_strength.get(),
        cb_Schulze_method.get(),
        cb_Linear_medians.get(),
        cb_All_rankings.get()]
    methods_num = len(names)
    Methods_frames = {
        names[i]: frames[i] for i in range(methods_num)
    }
    Methods_checkbuttons = {
        names[i]: checkbuttons[i] for i in range(methods_num)
    }
    Methods_rankings = {
        names[i]: None for i in range(methods_num)
    }

    R_list, P, C, R = Make_used_matrices(list_of_profiles)
    Params = {
        'R_list': R_list, 'P': P, 'C': C, 'R': R,
        'median_dist': None,
        'Schulze_ranking': None,
        'Schulze_winners': None}

    if sum(checkbuttons):
        _useless, Params['median_dist'] = Linear_medians(R_list)
        if Methods_checkbuttons['HP_max_length']:
            Methods_rankings['HP_max_length'] = HP_max_length(C)
        if Methods_checkbuttons['HP_max_strength']:
            Methods_rankings['HP_max_strength'] = HP_max_strength(C)
        if Methods_checkbuttons['Schulze_method']:
            Params['Schulze_winners'], Params['Schulze_ranking'] = \
                Schulze_method(C)
            if Params['Schulze_ranking'] != None:
                Methods_rankings['Schulze_method'] = [ 
                    Params['Schulze_ranking']]
        if Methods_checkbuttons['Linear_medians']:
            Methods_rankings['Linear_medians'], Params['median_dist'] =  \
                Linear_medians(R_list)
        if Methods_checkbuttons['All_rankings']:
            Methods_rankings['All_rankings'] = All_various_rankings()

        is_rankings_of_method_exist = [
            0 if rankings == None else 1
            for rankings in Methods_rankings.values()]

        Intersect = None
        if sum(is_rankings_of_method_exist) > 1:
            separator = "*"

            def list2symbolic_string(l):
                s = ""
                if l == []:
                    return s
                for e in l:
                    s += str(e) + separator
                return s[:-len(separator)]

            def symbolic_string2list(s):
                l = [int(e) for e in s.split(separator)]
                return l
            Sets = [
                set(
                    list2symbolic_string(ranking)
                    for ranking in single_method_rankings if ranking != None)
                for single_method_rankings in Methods_rankings.values()
                if single_method_rankings not in ([], [[]], None)]
            if Sets not in ([], None):
                Intersect = set.intersection(*Sets)
                Intersect = [symbolic_string2list(s) for s in Intersect]
        draw_result_rankings(Params, Methods_frames,
                             Methods_rankings, Intersect)
##        visualize_graph(C, None)
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
        correspondence = {symbols[i]: i for i in range(len(symbols))}
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


def matrix2string(Matrix):  # для удобства печати матриц
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
                width=max_widths[j]+2 if m > 5 else max(max_widths)+2)
        string += '\n'
    string = string[:-1]
    return string
###


def draw_result_rankings(
        Params, Methods_frames, Methods_rankings, Mutual_rankings):
    R_list = Params['R_list']
    Weights_matrix = Params['C']
    label_output['text'] = "Минимальное суммарное расстояние Хэмминга\n\
для мажоритарного графа: {}".format(
        sum_Hamming_distance(Params['R'], R_list))
    for name in Methods_rankings:
        Result_rankings = Methods_rankings[name]
        if Result_rankings != None:
            Lengths = Paths_weights_list(Result_rankings, Weights_matrix)
            Strengths = Paths_strengths_list(Result_rankings, Weights_matrix)
            Distances = Paths_sum_Hamming_distances_list(
                Result_rankings, R_list)
            new_table_output(Methods_frames[name], Result_rankings,
                             Lengths, Strengths, Distances,
                             Params['median_dist'], Mutual_rankings)
        if name == 'Schulze_method' and Params['Schulze_winners'] != None:
            draw_winners(Methods_frames[name],
                         Params['Schulze_winners'], Params['Schulze_ranking'])


def draw_winners(frame_of_method, winners, ranking):
    text = ""
    if ranking == None:
        text += "Ранжирование невозможно. "
    text += "Победители:"
    label_text = Label(
        frame_of_method, **label_opts, **border_opts, anchor=E, text=text)
    label_text.grid(**grid_optsE, row=n+4, column=0)

    label_winners = Label(frame_of_method, **label_opts, **relief_opts, padx=3,
                          text=list2string([index2symbol(w, n-1)
                                            for w in winners]))
    label_winners.grid(**grid_optsW, row=n+4, column=1)
    frame_of_method.grid()
    frame_output_all_tables.grid()
    change_fieldsize_for_scrolling()


def read_table():
    global table_input, labels_captions_top_inp
    grid_forget_output()
    window0.focus()
    try:
        if n > max_count_of_alternatives:
            raise ValueError(
                "Количество альтернатив n слишком велико.\n" +
                "Программа может зависнуть.")
        table_values = np.array([[int(symbol2index(table_input[i][j].get()))
                                  for j in range(m)] for i in range(n)])

        def elem_j_accepted(j):
            labels_captions_top_inp[j].config(background=window_background)

        def elem_j_incorrect(j):
            labels_captions_top_inp[j].config(background=error_color)
        number_of_accepted_elements = 0
        values = [i for i in range(n)]
        for j in range(m):
            profile_of_expert = table_values[:, j]
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


def new_table_output(frame_of_method, Result_rankings,
                     Lengths, Strengths, Distances,
                     median_dist, Mutual_rankings):
    global labels_captions_top_out, labels_captions_left_out, table_output, \
        table_info_output, labels_info_out
    # задание управляющих элементов
    r = len(Result_rankings)
    labels_captions_top_out = [Label(frame_of_method, **label_smallfont_opts,
                                     **border_opts,
                                     text="Ранжи-\nрование {0}".format(j+1))
                               for j in range(r)]
    labels_captions_left_out = [Label(frame_of_method, **label_opts, **border_opts,
                                      text="Место {0}".format(i+1))
                                for i in range(n)]
    cell_opts = {'width': 5, **relief_opts}
    table_output = [[Label(frame_of_method, **cell_opts, **input_field_opts,
                           anchor=W, padx=2)
                     for j in range(r)]
                    for i in range(n)]
    labels_info_out = [
        Label(frame_of_method, **label_opts, **border_opts, text="Длина:"),
        Label(frame_of_method, **label_opts, **border_opts, text="Сила:"),
        Label(frame_of_method, **label_smallfont_opts, **border_opts,
              justify=LEFT, text="Суммарное расстояние\nХэмминга:")
    ]
    table_info_output = [[Label(frame_of_method, **label_opts, **cell_opts)
                          for j in range(r)]
                         for i in range(3)]

    def min_and_max(List):
        inf = float("inf")
        L = [elem for elem in List if abs(elem) != inf]
        if L == []:
            return (-inf, -inf)
        return (min(L), max(L))

    MinsMaxes = [min_and_max(Lengths),
                 min_and_max(Strengths),
                 min_and_max(Distances)]
    color_min = '#BBEEFF'
    color_max = '#EEBBFF'
    color_mutual = '#CCFFCC'

    def highlight_characteristic(Characteristic, ranking_index, Ch_index):
        Ch_min = MinsMaxes[Ch_index][0]
        Ch_max = MinsMaxes[Ch_index][1]
        if Ch_min != Ch_max:
            if Characteristic[ranking_index] == Ch_min:
                table_info_output[Ch_index][ranking_index]['background'] = color_min
            if Characteristic[ranking_index] == Ch_max:
                table_info_output[Ch_index][ranking_index]['background'] = color_max

    def highlight_mutual_ranking(Result_rankings, ranking_index, Mutual_rankings):
        if Mutual_rankings != None and \
                Result_rankings[ranking_index] in Mutual_rankings:
            for i in range(len(Result_rankings[ranking_index])):
                table_output[i][ranking_index]['background'] = color_mutual

    for j in range(r):
        for i in range(len(Result_rankings[j])):
            table_output[i][j]['text'] = index2symbol(
                Result_rankings[j][i], n-1)
        table_info_output[0][j]['text'] = Lengths[j]
        table_info_output[1][j]['text'] = Strengths[j]
        table_info_output[2][j]['text'] = Distances[j]
        highlight_mutual_ranking(Result_rankings, j, Mutual_rankings)
        highlight_characteristic(Lengths, j, 0)
        highlight_characteristic(Strengths, j, 1)
        highlight_characteristic(Distances, j, 2)
        if Distances[j] == median_dist:
            table_info_output[2][j]['text'] += "\nМедиана"
            table_info_output[2][j]['width'] = len("Медиана")
            table_info_output[2][j].config(**smallfont_opts)

    grid_output_table(frame_of_method)


def new_table_input():
    global table_input, labels_captions_top_inp, labels_captions_left_inp, table_fromfile
    try:
        spinbox_n.set(str(n))
        spinbox_m.set(str(m))
    except Exception:
        pass
    table_frame = frame_input_profiles
    # задание управляющих элементов
    labels_captions_top_inp = [Label(table_frame, **label_smallfont_opts,
                                     **border_opts,
                                     text="Эксперт {0}".format(j+1))
                               for j in range(m)]
    labels_captions_left_inp = [Label(table_frame, **label_opts, **border_opts,
                                      text="Место {0}".format(i+1))
                                for i in range(n)]
    table_input = [[ttk.Combobox(table_frame, **input_field_opts, width=4,
                                 values=[index2symbol(i, n-1)
                                         for i in range(n)],
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
        f = open(s, 'r')
        strings = f.readlines()
        f.close()
        for s in strings:
            s.replace("\n", "")
            list_ = list(map(symbol2index, s.split()))
            if list_ != []:
                table_fromfile.append(list_)
        if len(table_fromfile) == 0:
            raise ValueError("Пустой файл")
        nn = len(table_fromfile[0])
        for list_ in table_fromfile:
            if len(list_) != nn:
                raise ValueError("Неверная длина строк")
        if max(list(map(max, table_fromfile))) >= nn:
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


def grid_forget_input():  # уборка управляющих элементов
    for item in frame_input_profiles.grid_slaves():
        item.grid_remove()
    frame_input_profiles.grid_remove()
    change_fieldsize_for_scrolling()


def grid_forget_output():  # уборка управляющих элементов
    for frame in frame_output_all_tables.grid_slaves():
        for item in frame.grid_slaves():
            item.grid_remove()
        frame.update()
        frame.grid_remove()
    label_output.grid_remove()
    frame_output_all_tables.update()
    frame_output_all_tables.grid_remove()
    change_fieldsize_for_scrolling()


def grid_input_table():  # размещение управляющих элементов
    global labels_captions_top_inp, labels_captions_left_inp, \
        table_input
    if 'labels_captions_top_inp' in globals():
        for j in range(len(labels_captions_top_inp)):
            labels_captions_top_inp[j].grid(**pad0, row=0, column=j+1)
    if 'labels_captions_left_inp' in globals():
        for i in range(len(labels_captions_left_inp)):
            labels_captions_left_inp[i].grid(**pad0, row=i+1, column=0)
    if 'table_input' in globals():
        for i in range(len(table_input)):
            for j in range(len(table_input[i])):
                table_input[i][j].grid(**pad0, row=i+1, column=j+1)
    frame_input_profiles.grid()
    change_fieldsize_for_scrolling()


# размещение управляющих элементов
def grid_output_table(frame_of_particular_method: Widget):
    global labels_captions_top_out, labels_captions_left_out, \
        table_output, labels_info_out, table_info_output
    if 'labels_captions_top_out' in globals():
        for j in range(len(labels_captions_top_out)):
            labels_captions_top_out[j].grid(**pad0, row=0, column=j+1)
    if 'labels_captions_left_out' in globals():
        n = len(labels_captions_left_out)
        for i in range(n):
            labels_captions_left_out[i].grid(
                **pad0, sticky=E, row=i+1, column=0)
        if 'labels_info_out' in globals():
            for i in range(3):
                labels_info_out[i].grid(**grid_optsE, row=n+1+i, column=0)
    if 'table_output' in globals():
        for i in range(len(table_output)):
            for j in range(len(table_output[i])):
                table_output[i][j].grid(**pad0, row=i+1, column=j+1)
    if 'table_info_output' in globals():
        for i in range(len(table_info_output)):
            for j in range(len(table_info_output[i])):
                table_info_output[i][j].grid(**pad0, row=n+1+i, column=j+1)
    frame_of_particular_method.grid()
    frame_output_all_tables.grid()
    if label_output['text'] != "":
        label_output.grid()
    change_fieldsize_for_scrolling()


def change_fieldsize_for_scrolling():            
    frame1.update()
    frame2.update()
    frame0.update()
    canvas1.config(scrollregion=canvas1.bbox(ALL))


# Характеристики и опции для управляющих элементов, глобальные переменные
n = 0
m = 0
max_count_of_alternatives = 8
max_number_for_spinbox = 300
max_number_for_cells = 3600
error_color = '#FFBBBB'
input_bg_color = 'white'
window_background = 'antique white'  # '#FAEBD7'
button_background = 'bisque'  # '#FFE4C4'
disabled_button_background = '#D9D9D9'
button_borderwidth = 5
font = "Book Antiqua"
font_mono = 'Courier New'
font_size = 10
font_opts = {'font': (font, font_size)}
smallfont_opts = {'font': (font, font_size-2)}
border_opts = {'relief': "solid", 'borderwidth': 1}
relief_opts = {'relief': "sunken", 'borderwidth': 1}
frame_opts = {'background': window_background}
label_opts = {'background': window_background, 'font': (font, font_size)}
input_field_opts = {'background': input_bg_color,
                    'font': (font_mono, font_size)}
label_smallfont_opts = {'background': window_background,
                        'font': (font, font_size-2)}
button_opts_enabled = {'background': button_background,
                       'relief': RAISED,
                       'borderwidth': button_borderwidth,
                       'font': (font, font_size),
                       'state': NORMAL}
button_opts_disabled = {'background': disabled_button_background,
                        'relief': RAISED,
                        'borderwidth': button_borderwidth,
                        'font': (font, font_size),
                        'state': DISABLED}
pad3 = {'padx': 3, 'pady': 3}
pad0 = {'padx': 0, 'pady': 0}
grid_optsNW = {'sticky': NW, **pad3}
grid_optsW = {'sticky': W, **pad3}
grid_optsE = {'sticky': E, **pad3}
grid_optsNSEW = {'sticky': N+S+E+W, **pad3}
pack_optsTW = {'side': TOP, 'anchor': W}
pack_optsLW = {'side': LEFT, 'anchor': W, **pad3}
###

# Главное окно
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
window0.update()
###

# Задание и первичное размещение управляющих элементов

# Для скорллинга
# холст
canvas1 = Canvas(window0,
                 background=window_background,
                 confine=True,
                 width=w, height=w,
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


def on_mousewheel(event):
    if canvas1.winfo_height() < canvas1.bbox("all")[3]:
        canvas1.yview_scroll(int(-1*(event.delta/80)), "units")


canvas1.bind_all("<MouseWheel>", on_mousewheel, add='+')

# главный фрейм
frame0 = Frame(canvas1, background=window_background)
canvas1.create_window((0, 0), window=frame0, anchor=NW)
###

# Управляющие элементы
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
                  command=lambda: messagebox.showinfo("", info_file))
menu0.add_command(label="Пояснения ко вводу информации",
                  command=lambda: messagebox.showinfo("", info_input))
menu0.add_command(label="Пояснения к результату работы программы",
                  command=lambda: messagebox.showinfo("", info_output))

frame1 = Frame(master=frame0, **frame_opts)
frame1.grid(**grid_optsNW, row=0, column=0)
frame2 = Frame(master=frame0, **frame_opts)
frame2.grid(**grid_optsNW, row=0, column=1)

frame1top = Frame(frame1, **frame_opts)
frame1top.grid(**grid_optsNW, row=0, column=0)
frame1bottom = Frame(frame1, **frame_opts)
frame1bottom.grid(**grid_optsNW, row=1, column=0)

frame_n_m = LabelFrame(frame1top, **label_opts, text="Выбор n и m")
frame_n_m.grid(**grid_optsNW, row=0, column=0)

frame_checkbuttons = LabelFrame(frame1top, **label_opts, text="Выбор метода")
frame_checkbuttons.grid(**grid_optsW, row=0, column=1)

frame_forfile = LabelFrame(frame1top, **label_opts, text="Импорт из txt-файла")
frame_forfile.grid(**grid_optsNSEW, row=1, column=0, columnspan=3)

frame_input_profiles = LabelFrame(frame1bottom, **label_opts,
                                  text="Ввод таблицы ранжирований")
frame_input_profiles.grid(**grid_optsNW, row=0, column=0)

frame_output_all_tables = LabelFrame(frame2, **label_opts,
                                     text="Результирующие ранжирования")
frame_output_all_tables.grid(**grid_optsNW, row=0, column=1)

frame_table_HP_max_length = LabelFrame(frame_output_all_tables, **label_opts,
                                       text="Гамильтоновы пути максимальной длины")
frame_table_HP_max_length.grid(**grid_optsNW, row=0, column=0)
frame_table_HP_max_strength = LabelFrame(frame_output_all_tables, **label_opts,
                                         text="Гамильтоновы пути наибольшей силы")
frame_table_HP_max_strength.grid(**grid_optsNW, row=1, column=0)
frame_table_Schulze_method = LabelFrame(frame_output_all_tables, **label_opts,
                                        text="Ранжирование по алгоритму Шульце")
frame_table_Schulze_method.grid(**grid_optsNW, row=2, column=0)
frame_table_Linear_medians = LabelFrame(frame_output_all_tables, **label_opts,
                                        text="Линейные медианы")
frame_table_Linear_medians.grid(**grid_optsNW, row=3, column=0)
frame_table_All_rankings = LabelFrame(frame_output_all_tables, **label_opts,
                                      text="Всевозможные ранжирования")
frame_table_All_rankings.grid(**grid_optsNW, row=4, column=0)

label1 = Label(frame_n_m, **label_opts, text="Число n альтернатив")
label1.grid(**grid_optsW,
            row=0, column=0)
spinbox_n = IntVar()
Spinbox(frame_n_m, **input_field_opts,
        from_=1, to=999,
        width=6,
        textvariable=spinbox_n
        ).grid(
    **grid_optsW,
    row=0, column=1)

label2 = Label(frame_n_m, **label_opts, text="Число m экспертов")
label2.grid(**grid_optsW,
            row=1, column=0)
spinbox_m = IntVar()
Spinbox(frame_n_m, **input_field_opts,
        from_=1, to=999,
        width=6,
        textvariable=spinbox_m
        ).grid(
    **grid_optsW,
    row=1, column=1)

button_read_n_and_m = Button(frame_n_m, **button_opts_enabled,
                             text="Ввод n и m",
                             command=read_n_and_m)
button_read_n_and_m.grid(**pad3, sticky=W+E,
                         row=2, column=0,
                         columnspan=2)

cb_HP_max_length = IntVar()
cb_HP_max_strength = IntVar()
cb_Schulze_method = IntVar()
cb_Linear_medians = IntVar()
cb_All_rankings = IntVar()

cb_HP_max_length.set(0)
cb_HP_max_strength.set(0)
cb_Schulze_method.set(0)
cb_Linear_medians.set(0)
cb_All_rankings.set(0)

checkbutton_opts = {'master': frame_checkbuttons, **label_opts}
checkbutton1 = Checkbutton(**checkbutton_opts, variable=cb_HP_max_length,
                           text="Гамильтоновы пути максимальной длины"
                           ).pack(**pack_optsTW)
checkbutton2 = Checkbutton(**checkbutton_opts, variable=cb_HP_max_strength,
                           text="Гамильтоновы пути наибольшей силы",
                           ).pack(**pack_optsTW)
checkbutton3 = Checkbutton(**checkbutton_opts, variable=cb_Schulze_method,
                           text="Ранжирование по алгоритму Шульце",
                           ).pack(**pack_optsTW)
checkbutton4 = Checkbutton(**checkbutton_opts, variable=cb_Linear_medians,
                           text="Линейные медианы",
                           ).pack(**pack_optsTW)
checkbutton5 = Checkbutton(**checkbutton_opts, variable=cb_All_rankings,
                           text="Всевозможные ранжирования",
                           ).pack(**pack_optsTW)

button_read_table = Button(frame1top, **button_opts_disabled,
                           text="Пуск!\n→",
                           command=read_table)
button_read_table.grid(**grid_optsNSEW, row=0, column=2)

entry_forfile = Entry(frame_forfile, **input_field_opts)
entry_forfile.insert(0, "test.txt")
entry_forfile.pack(side=LEFT, expand=True, fill=X)
button_forfile = Button(frame_forfile, **button_opts_enabled,
                        text="Ввод из файла",
                        command=read_file,
                        ).pack(**pack_optsLW)

label_output = Label(frame1bottom, **label_opts,
                     bg=window_background, justify=LEFT,
                     text="")
label_output.grid(**grid_optsNW, row=1, column=0)

grid_forget_input()
grid_forget_output()
###
window0.mainloop()
