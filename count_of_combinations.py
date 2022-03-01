import numpy as np
import sympy as sp
from prettytable import PrettyTable

def fun(n,m):
	if n == 1:
		return 1
	elif m == 1:
		return sp.factorial(n)
	else:
		return sp.factorial(sp.factorial(n) - 1 + m)/ \
                       (sp.factorial(sp.factorial(n)-1)*sp.factorial(m))
table = PrettyTable()
table.field_names = [""] + ['m = {}'.format(m) for m in range(1,7)]
for n in range(1,7):
    table.add_row(['n = {}'.format(n)] + [fun(n,m) for m in range(1,7)])
table.align = 'l'
print(table)
