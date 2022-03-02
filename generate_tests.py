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
#n = 3
#m = 5
print("Generate files? y/n: ",end='')
is_generate_tests = input()

def count_of_combinations_with_replacement(n,m):
    if n == 1:
        return 1
    elif m == 1:
        return sp.factorial(n)
    else:
        return sp.factorial(sp.factorial(n) - 1 + m)/ \
               (sp.factorial(sp.factorial(n)-1)*sp.factorial(m))

table = PrettyTable()
max_ = 7
table.field_names = [""] + ['m = {}'.format(M) for M in range(1,max_)]
for N in range(1,max_):
    table.add_row( ['n = {}'.format(N)] + \
                   [count_of_combinations_with_replacement(N,M)
                    for M in range(1,max_)])
table.align = 'l'
print(table)

#=================================================


directory_with_tests = os.getcwd()

def make_file(text, file_name):
    file_txt = open(directory_with_tests + "\{}.txt".format(file_name), "w")
    file_txt.write(text)
    file_txt.close()

def list_to_line(l):
    return str(l).replace(",","")[1:-1]

def lists_to_lines(Lists):
    lines = ""
    for l in Lists:
        lines += list_to_line(l) + '\n'
    return lines

def make_files_with_tests():
    global n, m
    single_profile_variants = list(
        itertools.permutations([i+1 for i in range(n)]))
    all_profiles = list(itertools.combinations_with_replacement(
        single_profile_variants, m))
    for i in range(len(all_profiles)):
        text = lists_to_lines(all_profiles[i])
        make_file(text, 't{}'.format(i+1))

if is_generate_tests == "y":
    make_files_with_tests()
    print("Расположение файлов:", directory_with_tests)


