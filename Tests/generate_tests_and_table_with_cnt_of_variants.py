from ntpath import join
import os
import math
import numpy as np
import itertools
import sympy as sp
from prettytable import PrettyTable

print("Enter count of alternatives n: ",end='')
n = int(input())
print("Enter count of experts m: ",end='')
m = int(input())
print("Generate files? y/n: ",end='')
is_generate_tests = input()

# n = 3
# m = 3

# количество перестановок с повторениями из n! элементов по m позициям
def count_of_expert_profiles_variants(n, m):
    if n == 1:
        return 1
    elif m == 1:
        # количество вариантов размещения n альтернатив по n местам (перестановки)
        return sp.factorial(n)
    else:  # каждый вариант размещения альтернатив - это один элемент, который могут выбрать один или несколько экспертов
        return sp.factorial(sp.factorial(n) - 1 + m) / \
            ( sp.factorial(sp.factorial(n)-1) * sp.factorial(m) )


max_experts_num = 7
max_alternatives_num = 7
table_cnt_of_EPV = [[count_of_expert_profiles_variants(N+1, M+1) 
for M in range(max_experts_num)]
for N in range(max_alternatives_num)]

table = PrettyTable()
table.field_names = [""] + \
    ['m = {}'.format(M+1) for M in range(max_experts_num)]
for N in range(max_alternatives_num):
    table.add_row(['n = {}'.format(N+1)] + \
        [table_cnt_of_EPV[N][M] for M in range(max_experts_num)])
table.align = 'l'
print(table)

# =================================================

directory_with_tests = os.getcwd()


def make_file(text, file_name):
    file_txt = open(directory_with_tests + "\{}.txt".format(file_name), "w")
    file_txt.write(text)
    file_txt.close()


def list_to_string(List):
    return " ".join(map(str, List))


def lists_to_strings(Lists):
    return "\n".join(map(list_to_string, Lists))


def make_files_with_tests():
    global n, m
    single_profile_variants = list(
        itertools.permutations([i+1 for i in range(n)]))
    all_profiles_variants = list(itertools.combinations_with_replacement(
        single_profile_variants, m))
    number_len = len(str(table_cnt_of_EPV[n-1][m-1]))
    for i in range(len(all_profiles_variants)):
        text = lists_to_strings(all_profiles_variants[i])
        make_file(text, 'n{0}m{1}_t{2:0>{3}d}'.format(n,m,i+1,number_len))


if is_generate_tests == "y":
    make_files_with_tests()
    print("Расположение файлов:", directory_with_tests)
