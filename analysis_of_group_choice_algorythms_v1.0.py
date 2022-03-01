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

def read_table():
    global table1, labels_experts, matrix_fromfile
    window0.focus()
    n = len(table1)
    m = len(table1[0])
    table_values = np.array([ [int(table1[i][j].get())
                        for j in range(m)] for i in range(n)])
    def elem_ij_accepted(i,j):
        labels_experts[j].config(background=window_background)
    def elem_ij_incorrect(i,j):
        labels_experts[j].config(background=error_color)
    number_of_elements = 0
    values = [i+1 for i in range(n)]
    for j in range(m):
        profile_of_expert = table_values[:,j]
        print(profile_of_expert,values)
        if sorted(profile_of_expert) == values:
            for i in range(n):
                elem_ij_accepted(i,j)
                number_of_elements += 1
        else:
            for v in values:
                indices = [i for i in range(n) if
                           profile_of_expert[i] == v]
                print(indices)
                if len(indices) > 1:
                    for i in indices:
                        elem_ij_incorrect(i,j)
        print(number_of_elements)
    if number_of_elements != n*m:
        messagebox.showinfo("", "Введите кооректные данные")
    else:
        print("good")
        
    
##def далее():        
##    print("Ввод: ")
##    print(matrix2string(adjacency_matrix), end="\n\n")###
##    if(number_of_elements == n*n):
##        spinbox1.delete(0, END)
##        label1.grid_forget()
##        (text, SSP_matrix, ranking) = Schulze_method(adjacency_matrix)
##        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
##        rnk = [ranking[i] for i in range(n)]
##        if n <= 26:
##            for i in range(n): rnk[i] = letters[ranking[i]]
##        else:
##            for i in range(n): rnk[i] = ranking[i] + 1
##        rnk = str(rnk)        
##        print("Матрица сил сильнейших путей:\n",
##              matrix2string(SSP_matrix), sep='', end='\n\n')
##        print("Ранжирование: ", rnk, sep='', end='\n\n')
##        print("Длина пути-ранжирования (цена): ",
##              path_length(ranking, adjacency_matrix), sep='', end='\n\n')
##        
##        label3['text'] = "Матрица сил сильнейших путей:\n"+\
##                         matrix2string(SSP_matrix)+\
##                         "\n\nСилы и пути:\n"+text
##        label3.grid(
##            **place_optsNW,
##            row = 1, column = 1,
##            rowspan = 1000)
##        label3.update()
##        
##        label2['text'] = "Ранжирование:\n"+rnk+"\n"+\
##                         "Длина пути-ранжирования (цена): {0}".format(
##                             path_length(ranking, adjacency_matrix))
##        label2.grid(
##            **place_optsNW,
##            row = 0, column = 1) 
##        label2.update()
##        
##        change_fieldsize_for_scrolling(n)
##        visualize_graph(adjacency_matrix, ranking)
##    else:
##        messagebox.showinfo("", "Введите кооректные данные")

##def activate_table():
##    label1.grid(
##        **place_optsNW,
##        row = 2, column = 0,
##        columnspan = 1)
##    label2.grid_forget()
##    label3.grid_forget()
##    spinbox1.delete(0, END)
##    n = len(entry2)
##    for i in range(n):
##        for j in range(n):
##            table[i][j].config(state = NORMAL)
##            if(entry2[i][j].get() == "inf"):
##                entry2[i][j].delete(0,last=END)

def new_table(n,m):
    global table1, labels_experts, matrix_fromfile
    table_frame = frame21
##    label1.grid(
##        **place_optsNW,
##        row = 2, column = 0)
##    label2['text'] = ""
##    label2.update()
##    label2.grid_forget()
##    label3['text'] = ""
##    label3.update()
##    label3.grid_forget()
##    entry_forfile.config(state = DISABLED)
##    button_forfile['state'] = DISABLED
##    button_forfile['background'] = disabled_button_background
##    button2['state'] = NORMAL
##    button2['background'] = button_background
##    button3['state'] = NORMAL
##    button3['background'] = button_background
    if 'labels_experts' in globals():
        for j in range(len(labels_experts)):
            labels_experts[j].grid_forget()
    if 'labels_alters' in globals():
        for i in range(len(labels_alters)):
            labels_experts[i].grid_forget()
    if 'table1' in globals():
        for i in range(len(table1)):
            for j in range(len(table1[0])):
                table1[i][j].grid_forget()
    ### задание управляющих элементов
    labels_experts = [Label(table_frame, **label_opts,
                            borderwidth=1, relief="solid",
                            text="Эксперт {0}".format(j+1))
                      for j in range(m)]
    labels_alters= [Label(table_frame, **label_opts,
                          borderwidth=1, relief="solid",
                          text="Альтернатива {0}".format(i+1))
                    for i in range(n)]
    
    table1 = [[ttk.Combobox(table_frame, width=6,
                            values=[i+1 for i in range(n)],
                            state="readonly")
               for j in range(m)]
              for i in range(n)]
    for i in range(n):
        for j in range(m):
            table1[i][j].current(i)

##    if 'matrix_fromfile' in globals() and len(matrix_fromfile) > 0:
##        for i in range(n):
##            for j in range(n):
##                table1[i][j].delete(0,last=END)
##                table1[i][j].insert(0, matrix_fromfile[i][j])
##        matrix_fromfile = []
    
    ### размещение управляющих элементов
    pads = {'padx':0, 'pady':0}
    for j in range(m):
        labels_experts[j].grid(
            **pads,
            row = 0, column = j+1)
    for i in range(n):
        labels_alters[i].grid(
            {**place_optsW, **pads},
            row = i+1, column = 0)
    for i in range(n):
        for j in range(m):
            table1[i][j].grid(
                **pads,
                row = i+1, column = j+1)
    change_fieldsize_for_scrolling(n,m)

def read_n_and_m():
    n = spinbox_n.get()
    m = spinbox_m.get()
    try:
        n = int(n)
        m = int(m)
        if n > 0 and m > 0:
            new_table(n,m)
        else:
            raise ValueError()
    except ValueError:
        messagebox.showinfo("", "Введите кооректные данные")
        
def change_fieldsize_for_scrolling(n,m):
    frame1.update()
    frame2.update()
    widths = [frame1.winfo_width(),frame2.winfo_width()]
    heights = [frame1.winfo_height(),frame2.winfo_height()]
    Wid = 1*w//12 + max(widths)
    Hei = 1*h//12 + sum(heights)
    canvas1.config(scrollregion=(0, 0, Wid, Hei))
    frame0.config(width=Wid, height=Hei)

##def enter_number_of_nodes():
##    entry_forfile.insert(0,'test.txt')###
##    entry_forfile.focus()
##    button2['state'] = DISABLED
##    button3['state'] = DISABLED
##    button2['background'] = disabled_button_background
##    button3['background'] = disabled_button_background
##
    
### характеристики управляющих элементов
error_color = '#FFCCCC'
input_bg_color='white'
window_background = 'antique white'#'#FAEBD7'
button_background = 'bisque'#'#FFE4C4'
disabled_button_background = '#D9D9D9'
button_borderwidth = 5
font = "Book Antiqua"
font_mono = 'Courier New'
font_size = 10
###
### главное окно
window0 = Tk()
w = 2*window0.winfo_screenwidth()//3
h = window0.winfo_screenheight()//2
window0.config(
    background=window_background,
    relief=SUNKEN,
    borderwidth=3)
window0.geometry('{}x{}'.format(w, h))
window0.title("Анализ алгоритмов группового выбора, \
использующих пути в орграфе")

##if 'window0' in globals():
##    pass

### Задание и первичное размещение управляющих элементов
### Для скорллинга
# холст
canvas1 = Canvas(window0,
	     background=window_background,
	     width=300, height=200,
	     scrollregion=(0, 0, 2*w/3, 2*h/3))
# скроллбары
scrollbarY1 = Scrollbar(window0, command=canvas1.yview,
		    orient=VERTICAL)
scrollbarX1 = Scrollbar(window0, command=canvas1.xview,
		    orient=HORIZONTAL)
canvas1.config(yscrollcommand=scrollbarY1.set,
	   xscrollcommand=scrollbarX1.set)
scrollbarY1.pack(side=RIGHT, fill=Y)
scrollbarX1.pack(side=BOTTOM, fill=X)
canvas1.pack(side=LEFT, expand=YES, fill=BOTH)
# главный фрейм
frame0 = Frame(canvas1,
	   bd=0, background=window_background,
	   width=2*w/3, height=2*h/3)
canvas1.create_window((0, 0), window=frame0, anchor=NW)
###

### Опции для элементов
frame_opts = {'background':window_background}
label_opts = {'background':window_background,
	       'font':(font, font_size)}
button_opts = {'background':button_background,
	       'relief':RAISED,
	       'borderwidth':button_borderwidth,
	       'font':(font, font_size)}
place_optsNW = {'sticky':NW, 'padx':2, 'pady':2}
place_optsW = {'sticky':W, 'padx':2, 'pady':2}
pack_optsTW = {'side':TOP, 'anchor':W}
pack_optsLW = {'side':LEFT, 'anchor':W}
###

### Управляющие элементы
frame1 = Frame(
    master=frame0,
    **frame_opts
    )
frame1.grid(
    **place_optsNW,
    row = 0, column = 0
    )

frame11 = LabelFrame(
	frame1,
	**label_opts,
	text="Выбор n и m"
        )
frame11.grid(
    **place_optsW,
    row = 0, column = 0
    )

label1 = Label(
	frame11,
	**label_opts,
	text="Число n альтернатив"
        ).grid(
            **place_optsW,
            row = 0, column = 0
            )
spinbox_n = IntVar()
Spinbox(#n
    frame11,
    from_=1,
    to=99,
    width=6,
    font=(font, font_size),
    textvariable = spinbox_n
    ).grid(
        **place_optsW,
        row = 0, column = 1
        )

label2 = Label(
	frame11,
        **label_opts,
	text="Число m экспертов"
        ).grid(
            **place_optsW,
            row = 1, column = 0
            )
spinbox_m = IntVar()
Spinbox(#m
    frame11,
    from_=1,
    to=99,
    width=6,
    font=(font, font_size),
    textvariable = spinbox_m
    ).grid(
        **place_optsW,
        row = 1, column = 1
        )

button1 = Button(
    frame11,
    **button_opts,
    text="Ввод n и m",
    command=read_n_and_m
    ).grid(
        row = 2, column = 0,
        columnspan = 2
        )

frame12 = LabelFrame(
	frame1,
	**label_opts,
	text="Импорт из txt-файла"
        )
frame12.grid(
    **place_optsW,
    row = 1, column = 0,
    columnspan = 2
    )

entry_forfile = Entry(
    frame12,
    width=45)
entry_forfile.grid(
    **place_optsW,
    row = 0, column = 1
    )
button_forfile = Button(
    frame12,
    text = "Ввод матрицы из файла:",
    **button_opts#,
    #command=read_file
    )
button_forfile.grid(
    **place_optsW,
    row = 0, column = 0
    )

frame13 = LabelFrame(
	frame1,
	**label_opts,
	text="Выбор метода"
        )
frame13.grid(
    **place_optsW,
    row = 0, column = 1
    )
method = StringVar()
method.set('_')
radiobutton1 = Radiobutton(
	frame13,
	**label_opts,
	text = "Гамильтоновы пути наибольшей длины",
	variable = method,
	value = "HP_max_length").pack(**pack_optsTW)
radiobutton2 = Radiobutton(
	frame13,
	**label_opts,
	text = "Гамильтоновы пути наибольшей силы",
	variable = method,
	value = "HP_max_strength").pack(**pack_optsTW)
radiobutton3 = Radiobutton(
	frame13,
	**label_opts,
	text = "Ранжирование по алгоритму Шульце",
	variable = method,
	value = "Schulze_method").pack(**pack_optsTW)
radiobutton4 = Radiobutton(
	frame13,
	**label_opts,
	text = "Линейная диагональ",
	variable = method,
	value = "Linear_diagonal_(median)").pack(**pack_optsTW)

button2 = Button(
	frame1,
        **button_opts,
        width=5,
        height=5,
	text="Пуск!",
        command=read_table
	).grid(
            row = 0, column = 3
            )

frame2 = Frame(frame0, **frame_opts)
frame2.grid(
    **place_optsW,
    row = 1, column = 0,
    rowspan = 1, columnspan = 1
    )

LabelFrame(
    frame2,
    bg = "black"
    ).grid(
        sticky="ns",
        row = 0, column = 1,
        rowspan = 2
        )

frame21 = Frame(frame2, **frame_opts)
frame21.grid(
    **place_optsW,
    row = 1, column = 0
    )

label3 = Label(
	frame2,
        **label_opts,
	text="Ввод таблицы ранжирований"
        ).grid(
            **place_optsW,
            row = 0, column = 0
            )

frame22 = LabelFrame(frame2, **frame_opts)
frame22.grid(
    **place_optsW,
    row = 1, column = 2
    )

label4 = Label(
	frame2,
        **label_opts,
	text="Результирующее ранжирование"
        ).grid(
            **place_optsW,
            row = 0, column = 2
            )



##combobox1 = ttk.Combobox(
##    frame2,
##    values=["Альтернатива default"],
##    state="readonly"
##    ).pack(**pack_optsLW)




###
window0.mainloop()
